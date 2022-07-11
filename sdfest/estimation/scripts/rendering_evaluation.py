"""Script to run randomized rendering evaluation on synthetic data.

Usage:
  python rendering_evaluation.py --config configs/config.yaml --data_path ./data/
  --out_folder ./results

See configs/rendering_evaluation.yaml for all supported arguments.
See simple_setup for pipeline parameters.

Specific parameters:
    log_folder:
        if passed each optimization step is logged and can be played back with
        play_log.py
    visualize_optimization: whether to visualize optimization while at it
    visualize_points: whether to show result pointclouds after optimization
    visualize_meshes: whether to show result mesh after optimization
    camera_distance: mesh distance from the camera
    num_views: list of number of views to evaluate for each mesh
    mesh_scale: the applied scale, see rel_scale
    rel_scale:
        if True, the original mesh will be scaled by mesh_scale, if False the original
        mesh will be scaled such that its largest extent is mesh_scale * 2
    samples: number of evaluation samples
    ablation_configs:
        used to specify specific settings for ablation study
        dictionary, in which each key maps to a config dictionary which will be applied
        on existing settings
    metrics:
        dictionary of metrics to evaluate
        each key, is interpreted as the name of the metric, each value has to be a dict
        containing f and kwargs, where f is fully qualified name of the function to
        evaluate and kwargs is a dictionary of keyword arguments if applicable
        f gets ground truth points as first, and estimated points as second
        parameter
    seed:
        seed used for view generation and sampling of points
"""

import argparse
from collections import defaultdict
import copy
from datetime import datetime
import glob
import math
import os
from pydoc import locate
import random
import time
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import yoco

import sdfest
from sdfest.initialization import quaternion_utils
from sdfest.estimation import synthetic
from sdfest.differentiable_renderer import Camera
from sdfest.estimation.simple_setup import SDFPipeline


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
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
        if self.base_config["ablation_configs"]:
            ablation_results_dict = {}
            for name, ablation_config in self.base_config["ablation_configs"].items():
                config = yoco.load_config(
                    ablation_config, copy.deepcopy(self.base_config)
                )
                self._set_seed(config["seed"])
                ablation_results_dict[name] = self._evaluate_config(config)
            self._save_and_print_results(ablation_results_dict)
        else:
            self._set_seed(self.base_config["seed"])
            results_dict = self._evaluate_config(self.base_config)
            self._save_and_print_results(results_dict)

    @staticmethod
    def _set_seed(seed: int = 0) -> None:
        random.seed(seed)

    def _evaluate_config(self, config: dict) -> dict:
        results_dict = {}
        self.pipeline = SDFPipeline(config)
        for views in config["num_views"]:
            metrics_list = []
            files = glob_exts(config["data_path"], [".obj", ".off"])
            files.sort()
            for file in tqdm(files):
                metrics = self._evaluate_file(file, views, config)
                metrics_list.append(metrics)

            results_dict[views] = Evaluator._compute_metric_statistics(metrics_list)
        return results_dict

    def _save_and_print_results(self, results_dict: Dict) -> None:
        """Save results and config to yaml file and print results as table.

        Args:
            results_dict:
                dictionary containing the results that should be saved
        """
        os.makedirs(self.base_config["out_folder"], exist_ok=True)
        run_name = self.base_config["run_name"]
        filename = (
            f"rend_eval_{run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml"
        )
        out_path = os.path.join(self.base_config["out_folder"], filename)
        combined_dict = {**self.base_config, "results": results_dict}
        yoco.save_config_to_file(out_path, combined_dict)
        print(f"Results saved to: {out_path}")

    @staticmethod
    def _compute_metric_statistics(metrics_list: List) -> Dict:
        """Compute mean and standard deviation for each metric.

        Args:
            metrics_list: metric dictionaries as returned by _evaluate_file.

        Returns:
            Statistic for each metric in the provided metrics dictionaries.
            The returned dictionary keys will be the name of the metrics. Each value
            will be another dictionary containing the keys mean, var, and std.
        """
        metric_stats = defaultdict(lambda: {"mean": 0, "var": 0})
        for metrics in metrics_list:
            for name, val in metrics.items():
                metric_stats[name]["mean"] += val

        for _, stats in metric_stats.items():
            stats["mean"] /= len(metrics_list)

        for metrics in metrics_list:
            for name, val in metrics.items():
                metric_stats[name]["var"] += (val - metric_stats[name]["mean"]) ** 2

        for _, stats in metric_stats.items():
            stats["var"] /= len(metrics_list)
            stats["std"] = math.sqrt(stats["var"])

        metric_stats = dict(metric_stats)
        return metric_stats

    def _generate_views(self, mesh: synthetic.Mesh, num_views: int) -> Dict:
        """Generate random views around mesh.

        Args:
            mesh:
                mesh to generate views of,
                position and orientation will be assumed to be in world coordinates
            num_views: number of views to generate

        Returns:
            Dictionary containing the following keys.

            depth_images:
                the depth map containing the distance along the camera's z-axis,
                shape (num_views, H, W)
            masks:
                binary mask of the object to estimate, same shape as depth_images
            color_images:
                color images of objects to estimate, shape (num_views, H, W, 3)
                note that this is currently just zero
            camera_positions:
                position of camera in world coordinates for each image,
                shape (num_views, 3)
            camera_orientations:
                orientation of camera in world-frame as normalized quaternion,
                quaternion is in scalar-last convention,
                this is the quaternion that transforms a point from camera to world,
                shape (num_views, 4)
        """
        views_dict = defaultdict(lambda: list())

        mesh.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        mesh_position = torch.tensor(mesh.position)
        mesh_orientation = torch.tensor(mesh.orientation)

        for _ in range(num_views):
            while True:
                # OpenGL convention camera
                camera_orientation = generate_uniform_quaternion()  # ogl to world
                camera_position = mesh_position - quaternion_utils.quaternion_apply(
                    camera_orientation,
                    torch.tensor([0, 0, -self.base_config["camera_distance"]]),
                )  # transform camera position s.t. object lies on principal axis

                # Transform mesh into camera frame, now with Open3D convention camera
                camera_orientation_o3d = quaternion_utils.quaternion_multiply(
                    camera_orientation,  # ogl to world
                    torch.tensor([1.0, 0, 0, 0]),  # o3d to ogl
                )  # quaternion: o3d to world
                mesh_orientation_cam = quaternion_utils.quaternion_multiply(
                    quaternion_utils.quaternion_invert(
                        camera_orientation_o3d
                    ),  # world to o3d
                    mesh_orientation,  # obj to world
                )  # quaternion: obj to o3d
                mesh.position = np.array([0, 0, self.base_config["camera_distance"]])
                mesh.orientation = mesh_orientation_cam.numpy()
                depth_np = synthetic.draw_depth_geometry(mesh, self.cam)
                depth = torch.tensor(depth_np)

                if (depth != 0).any():
                    views_dict["depth_images"].append(depth)
                    views_dict["masks"].append(depth != 0)
                    views_dict["color_images"].append(torch.zeros(depth.shape + (3,)))
                    views_dict["camera_positions"].append(camera_position)
                    views_dict["camera_orientations"].append(camera_orientation)
                    break

                print("Warning: invalid depth generated, skipping this sample")

        mesh.position = mesh_position.numpy()
        mesh.orientation = mesh_orientation.numpy()

        return {
            k: torch.stack(v).to(self.base_config["device"])
            for k, v in views_dict.items()
        }

    def _evaluate_file(self, path: str, num_views: int, config: dict) -> dict:
        """Evaluate a single mesh.

        This will generate depth images from a few views with the mesh centered, at
        fixed distance,  to the camera.

        Args:
            path: The path of the obj file.
            num_views: The number of views to generate and use in the optimization.
            config: Configuration dictionary.

        Returns:
            Evaluation metrics as specified in config.
        """
        gt_mesh = synthetic.Mesh(
            path=path,
            scale=self.base_config["mesh_scale"],
            rel_scale=self.base_config["rel_scale"],
            center=True,
        )
        inputs = self._generate_views(gt_mesh, num_views)

        log_path = self._get_log_path()

        position, orientation, scale, shape = self.pipeline(
            **inputs,
            visualize=self.base_config["visualize_optimization"],
            log_path=log_path,
            shape_optimization=config["shape_optimization"],
        )
        out_mesh = self.pipeline.generate_mesh(shape, scale, True)

        # Output and ground truth are in world frame
        out_mesh.position = position[0].detach().cpu().numpy()
        out_mesh.orientation = orientation[0].detach().cpu().numpy()
        gt_mesh = gt_mesh.get_transformed_o3d_geometry()
        out_mesh = out_mesh.get_transformed_o3d_geometry()
        gt_mesh.paint_uniform_color([0.7, 0.4, 0.2])
        out_mesh.paint_uniform_color([0.2, 0.4, 0.7])
        gt_pts = gt_mesh.sample_points_uniformly(
            number_of_points=self.base_config["samples"], seed=self.base_config["seed"]
        )
        out_pts = out_mesh.sample_points_uniformly(
            number_of_points=self.base_config["samples"], seed=self.base_config["seed"]
        )
        gt_pts_np = np.asarray(gt_pts.points)
        out_pts_np = np.asarray(out_pts.points)

        metric_dict = {}
        for metric_name, m in self.base_config["metrics"].items():
            metric_dict[metric_name] = float(
                locate(m["f"])(gt_pts_np, out_pts_np, **m["kwargs"])
            )

        self.visualize_result(gt_mesh, out_mesh, gt_pts, out_pts, inputs)

        return metric_dict

    def visualize_result(
        self,
        mesh_1: o3d.geometry.TriangleMesh,
        mesh_2: o3d.geometry.TriangleMesh,
        pts_1: o3d.geometry.PointCloud,
        pts_2: o3d.geometry.PointCloud,
        inputs: Optional[dict] = None,
    ) -> None:
        """Visualize result of a single evaluation.

        Args:
            mesh_1: the first mesh to visualize
            mesh_2: the second mesh to visualize
            pts_1: the first pointcloud to visualize
            pts_2: the second pointcloud to visualize
            inputs: the input dictionary as producted by Evaluator._generate_view
        """
        # generate coordinate frames of cameras
        cam_meshes = []
        if inputs is not None:
            # visualize OpenGL convention camera
            for t_c2w, quat_c2w in zip(
                inputs["camera_positions"], inputs["camera_orientations"]
            ):
                frame_mesh = synthetic.Mesh(
                    mesh=o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1, origin=[0, 0, 0]
                    ),
                    rel_scale=True,
                )
                frame_mesh.position = t_c2w.cpu().numpy()
                frame_mesh.orientation = quat_c2w.cpu().numpy()
                cam_meshes.append(frame_mesh.get_transformed_o3d_geometry())

        if self.base_config["visualize_meshes"]:
            # Visualize result
            o3d.visualization.draw_geometries(
                [
                    mesh_1,
                    mesh_2,
                ]
                + cam_meshes
            )
            time.sleep(0.1)

        if self.base_config["visualize_points"]:
            o3d.visualization.draw_geometries([pts_1, pts_2] + cam_meshes)
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


def generate_uniform_quaternion() -> torch.Tensor:
    """Generate a normalized uniform quaternion.

    Following the method from K. Shoemake, Uniform Random Rotations, 1992.

    See: http://planning.cs.uiuc.edu/node198.html

    Returns:
        Uniformly distributed unit quaternion on the estimator's device.
    """
    u1, u2, u3 = random.random(), random.random(), random.random()
    return torch.tensor(
        [
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
        ]
    )


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
    config = yoco.load_config_from_args(
        parser, search_paths=[".", "~/.sdfest/", sdfest.__path__[0]]
    )

    evaluator = Evaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
