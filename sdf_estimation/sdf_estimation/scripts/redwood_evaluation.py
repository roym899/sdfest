"""Script to run randomized rendering evaluation on synthetic data.

Usage:
  python rendering_evaluation.py --config configs/config.yaml --ann_path ./anns/
  --data_path ./redwood/ --out_folder ./results

ann_path expected structure:
    ann_path/annotations.json
    ann_path/00031.obj
    ann_path/00131.obj
    ...
    ann_path/31231.obj

data_path expected structure
    data_path/category_1/rgbd/00031/...
    data_path/category_2/rgbd/00131/...
    ...
    data_path/category_N/rgbd/31231/...

See configs/redwood_evaluation.yaml for all supported arguments.
See simple_setup for pipeline parameters.

Specific parameters:
    category_configs:
        dictionary mapping category string to specific configuration file
    log_folder:
        if passed each optimization step is logged and can be played back with
        play_log.py
    visualize_optimization: whether to visualize optimization while at it
    visualize_points: whether to show result pointclouds after optimization
    visualize_meshes: whether to show result mesh after optimization
    visualize_results: whether to show result mesh after optimization
    samples: number of evaluation samples
    mode:
        evaluation mode, one of: full | seg
    metrics:
        dictionary of metrics to evaluate
        each key, is interpreted as the name of the metric, each value has to be a dict
        containing f and kwargs, where f is fully qualified name of the function to
        evaluate and kwargs is a dictionary of keyword arguments if applicable
        f gets ground truth points as first, and estimated points as second
        parameter
"""

import argparse
import copy
from collections import defaultdict
import glob
from datetime import datetime
from pydoc import locate
import json
import math
import os
import random
import time
from tqdm import tqdm
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tabulate import tabulate
import torch
import yoco

from sdf_single_shot import quaternion_utils
from sdf_estimation import synthetic
from sdf_differentiable_renderer import Camera
from sdf_estimation.simple_setup import SDFPipeline

stop_vis = False
reconstruction_vis = 1
update_vis = True


def toggle_reconstruction(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Schedule ox."""
    global reconstruction_vis, update_vis
    update_vis = True
    reconstruction_vis = (reconstruction_vis + 1) % 3


def quit_visualizer(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Schedule ox."""
    global stop_vis
    stop_vis = True


def load_rgbd(rgb_path: str, depth_path: str) -> o3d.geometry.RGBDImage:
    """Load RGBD image."""
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False
    )


def load_pc(rgb_path: str, depth_path: str) -> o3d.geometry.PointCloud:
    """Load pointcloud.

    Assumes PrimeSenseDefault camera.
    """
    rgbd = load_rgbd(rgb_path, depth_path)
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        ),
    )


def glob_exts(path: str, exts: List[str]) -> List[str]:
    """Return all files in a nested directory with extensions matching.

    Directory is scanned recursively.

    Args:
        path: root path to search
        exts: extensions that will be checked, must include separator (e.g., ".obj")

    Returns:
        List of paths in the directory with matching extension.
    """
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True))

    return files


class Evaluator:
    """Class to evaluate SDF pipeline on synthetic data."""

    def __init__(self, config: dict) -> None:
        """Construct evaluator and initialize pipeline."""
        self.base_config = config
        self.cam = Camera(**self.base_config["camera"])

    def run(self) -> None:
        """Run the evaluation."""
        # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
        metrics_list_dict = defaultdict(list)  # maps from categories to list of metrics
        with open(
            os.path.join(self.base_config["ann_path"], "annotations.json"), "r"
        ) as f:
            ann_dict = json.load(f)

        for seq_id, seq_dict in tqdm(ann_dict.items()):
            config = copy.deepcopy(self.base_config)
            if "category" in seq_dict:
                yoco.load_config(
                    self.base_config["category_configs"][seq_dict["category"]], config
                )

            self.pipeline = SDFPipeline(config)
            seq_folder = glob.glob(os.path.join(config["data_path"], "*/*/", seq_id))[0]

            gt_mesh = synthetic.Mesh(
                path=os.path.join(config["ann_path"], seq_dict["mesh"]),
                scale=1.0,  # do not resize mesh, as it is already at right size
                rel_scale=True,
                center=False,
            )
            for pose_ann in seq_dict["pose_anns"]:
                gt_mesh.position = pose_ann["position"]
                gt_mesh.orientation = pose_ann["orientation"]
                rgb_path = os.path.join(seq_folder, "rgb", pose_ann["rgb_file"])
                depth_path = os.path.join(seq_folder, "depth", pose_ann["depth_file"])
                rgbd_image = load_rgbd(rgb_path, depth_path)
                pc = load_pc(rgb_path, depth_path)
                color_array = np.asarray(rgbd_image.color)
                depth_array = np.asarray(rgbd_image.depth)
                if self.base_config["visualize_input"]:
                    o3d.visualization.draw_geometries(
                        [gt_mesh.get_transformed_o3d_geometry(), pc]
                    )
                    plt.subplot(2, 1, 1)
                    plt.imshow(color_array)
                    plt.subplot(2, 1, 2)
                    plt.imshow(depth_array)
                    plt.show()
                mask = self._get_mask(color_array, depth_array, gt_mesh)
                metrics = self._evaluate(color_array, depth_array, mask, gt_mesh, pc)
                metrics_list_dict[seq_dict["category"]].append(metrics)

        results_dict = Evaluator._compute_metric_statistics(metrics_list_dict)
        self._save_and_print_results(results_dict)

    def _get_mask(
        self, color: np.array, depth: np.array, gt_mesh: synthetic
    ) -> np.array:
        if self.base_config["mode"] == "full":
            raise NotImplementedError("full mode not implemented. Use seg.")
        elif self.base_config["mode"] == "seg":
            gt_depth = synthetic.draw_depth_geometry(gt_mesh, self.cam)
            mask = np.abs(gt_depth - depth) < 0.1 * (gt_depth != 0) * (depth != 0)
            if self.base_config["visualize_input"]:
                _, axes = plt.subplots(1, 3, sharex=True, sharey=True)
                axes[0].imshow(depth)
                axes[1].imshow(gt_depth)
                axes[2].imshow(mask)
                plt.show()
            return mask
        else:
            print("Unsupported mode. Must be one of full|seg.")
            exit()

    def _save_and_print_results(self, results_dict: Dict) -> None:
        """Save results and config to yaml file and print results as table.

        Args:
            results_dict:
                dictionary mapping category to metric statistics as returned by
                Evaluator._compute_metric_statistics
        """
        os.makedirs(self.base_config["out_folder"], exist_ok=True)
        run_name = self.base_config["run_name"]
        filename = (
            f"real_eval_{run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml"
        )
        out_path = os.path.join(self.base_config["out_folder"], filename)
        combined_dict = {**self.base_config, "results": results_dict}
        yoco.save_config_to_file(out_path, combined_dict)
        print(f"Results saved to: {out_path}")
        for category, metric_stats in results_dict.items():
            print(f"\nResults for {category}")
            print(tabulate([{"metric": k, **v} for k, v in metric_stats.items()]))

    @staticmethod
    def _compute_metric_statistics(metrics_list_dict: Dict) -> Dict:
        """Compute mean and standard deviation for each metric.

        Args:
            metrics_list_dict:
                dictionary mapping from category to list of metric dictionaries as
                returned by _evaluate_file

        Returns:
            Statistic for each metric, category and all categories in the provided
            metrics dictionaries.
            The returned dictionary keys will be categories + all category.
            The values will be dictionaries the name of the metrics as keys and each
            value will be another dictionary containing the keys mean, var, and std.
        """
        metric_stats_dict = defaultdict(
            lambda: defaultdict(lambda: {"mean": 0, "var": 0, "count": 0})
        )

        for cat, metrics_list in metrics_list_dict.items():
            for metrics in metrics_list:
                for name, val in metrics.items():
                    metric_stats_dict[cat][name]["mean"] += val
                    metric_stats_dict["all"][name]["mean"] += val
                    metric_stats_dict[cat][name]["count"] += 1
                    metric_stats_dict["all"][name]["count"] += 1

        print(metric_stats_dict)

        for cat, metric_stats in metric_stats_dict.items():
            for _, stats in metric_stats.items():
                stats["mean"] /= stats["count"]

        print(metric_stats_dict)

        for cat, metrics_list in metrics_list_dict.items():
            for metrics in metrics_list:
                for name, val in metrics.items():
                    metric_stats_dict[cat][name]["var"] += (
                        val - metric_stats_dict[cat][name]["mean"]
                    ) ** 2
                    metric_stats_dict["all"][name]["var"] += (
                        val - metric_stats_dict["all"][name]["mean"]
                    ) ** 2

        print(metric_stats_dict)

        for cat, metric_stats in metric_stats_dict.items():
            for _, stats in metric_stats.items():
                stats["var"] /= stats["count"]
                stats["std"] = math.sqrt(stats["var"])

        print(metric_stats_dict)

        # convert defaultdicts to dictionaries
        for cat, metric_stats in metric_stats_dict.items():
            metric_stats_dict[cat] = dict(metric_stats)
        metric_stats_dict = dict(metric_stats_dict)
        return metric_stats_dict

    def _evaluate(
        self,
        color: np.array,
        depth: np.array,
        mask: np.array,
        gt_mesh: synthetic.Mesh,
        pc: o3d.geometry.PointCloud,
    ) -> dict:
        """Evaluate a input.

        This will generate depth images from a few views with the mesh centered, at
        fixed distance,  to the camera.

        Args:
            color: the rgb array, shape (H,W,3)
            depth: the depth array, shape (H,W)
            mask: the binary segmentation mask, shape (H,W)
            gt_mesh: the ground truth mesh to evaluate against
            pc: input pointcloud, used for visualization

        Returns:
            Evaluation metrics as specified in config.
        """
        log_path = self._get_log_path()

        # for d in inputs["depth_images"]:
        #     plt.imshow(d.cpu().detach().numpy())
        #     plt.show()

        position, orientation, scale, shape = self.pipeline(
            depth_images=torch.from_numpy(depth).to(self.base_config["device"]),
            masks=torch.from_numpy(mask).to(self.base_config["device"]),
            color_images=torch.from_numpy(color).to(self.base_config["device"]),
            visualize=self.base_config["visualize_optimization"],
            log_path=log_path,
        )
        out_mesh = self.pipeline.generate_mesh(shape, scale, True)

        # Convert output from OpenGL to Open3D convention (same as gt meshes)
        out_mesh.position = (
            quaternion_utils.quaternion_apply(torch.tensor([1.0, 0, 0, 0]), position[0])
            .detach()
            .cpu()
            .numpy()
        )
        out_mesh.orientation = (
            quaternion_utils.quaternion_multiply(
                torch.tensor([1.0, 0, 0, 0]), orientation[0]
            )
            .detach()
            .cpu()
            .numpy()
        )
        gt_mesh = gt_mesh.get_transformed_o3d_geometry()
        out_mesh = out_mesh.get_transformed_o3d_geometry()
        gt_mesh.paint_uniform_color([0.7, 0.4, 0.2])
        out_mesh.paint_uniform_color([0.2, 0.4, 0.7])
        gt_pts = gt_mesh.sample_points_uniformly(
            number_of_points=self.base_config["samples"]
        )
        out_pts = out_mesh.sample_points_uniformly(
            number_of_points=self.base_config["samples"]
        )
        gt_pts_np = np.asarray(gt_pts.points)
        out_pts_np = np.asarray(out_pts.points)

        metric_dict = {}
        for metric_name, m in self.base_config["metrics"].items():
            metric_dict[metric_name] = float(
                locate(m["f"])(gt_pts_np, out_pts_np, **m["kwargs"])
            )
        self.visualize_result(gt_mesh, out_mesh, gt_pts, out_pts, pc)

        return metric_dict

    def visualize_result(
        self,
        mesh_gt: o3d.geometry.TriangleMesh,
        mesh_est: o3d.geometry.TriangleMesh,
        pts_gt: o3d.geometry.PointCloud,
        pts_est: o3d.geometry.PointCloud,
        pc: o3d.geometry.PointCloud,
    ) -> None:
        """Visualize result of a single evaluation.

        Args:
            mesh_gt: the ground truth mesh to visualize
            mesh_est: the estimated mesh to visualize
            pts_1: the first pointcloud to visualize
            pts_2: the second pointcloud to visualize
        """
        global update_vis, reconstruction_vis, stop_vis
        # generate coordinate frames of cameras
        cam_meshes = []
        frame_mesh = synthetic.Mesh(
            mesh=o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            ),
            rel_scale=True,
        )
        cam_meshes.append(frame_mesh.get_transformed_o3d_geometry())

        KEY_ESCAPE = 256
        if self.base_config["visualize_results"]:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.register_key_callback(key=ord("R"), callback_func=toggle_reconstruction)
            vis.register_key_callback(key=KEY_ESCAPE, callback_func=quit_visualizer)
            vis.create_window(width=640, height=480)
            update_vis = True
            stop_vis = False
            time.sleep(0.1)
            first = True
            while True:
                if update_vis:
                    print(reconstruction_vis)

                    vis.clear_geometries()

                    # Visualize result
                    if reconstruction_vis == 1:
                        vis.add_geometry(pts_est, reset_bounding_box=False)
                    if reconstruction_vis == 2:
                        vis.add_geometry(mesh_est, reset_bounding_box=False)
                    vis.add_geometry(pc, reset_bounding_box=first)

                    update_vis = False
                    first = False

                vis.poll_events()
                vis.update_renderer()

                if stop_vis:
                    break

        if self.base_config["visualize_points"]:
            o3d.visualization.draw_geometries([pts_gt, pts_est] + cam_meshes)
            time.sleep(0.1)

        if self.base_config["visualize_meshes"]:
            # Visualize result
            o3d.visualization.draw_geometries(
                [
                    mesh_gt,
                    mesh_est,
                ]
                + cam_meshes
            )
            time.sleep(0.1)

    def _get_log_path(self) -> Optional[str]:
        """Return unique filename in log folder.

        If log path is None, None will be returned.
        """
        log_path = None
        if self.base_config["log_folder"] is not None:
            os.makedirs(self.base_config["log_folder"], exist_ok=True)
            filename = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
            log_path = os.path.join(self.base_config["log_folder"], filename)
        return log_path


def main() -> None:
    """Entry point of the evaluation program."""
    parser = argparse.ArgumentParser(
        description="SDF shape and pose estimation evaluation on synthetic data"
    )
    parser.add_argument(
        "--config", default="configs/rendering_evaluation.yaml", nargs="+"
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_folder", required=True)
    config = yoco.load_config_from_args(parser)

    evaluator = Evaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
