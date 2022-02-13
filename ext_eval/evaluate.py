"""Script to run pose and shape evaluation for different datasets and methods."""
import argparse
import os
from datetime import datetime
from typing import Tuple, Optional
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
import open3d as o3d
from sdf_estimation import metrics
from sdf_single_shot import pointset_utils, quaternion_utils
from sdf_differentiable_renderer import Camera
from sdf_single_shot.utils import str_to_object

import yoco

from method_wrappers import MethodWrapper, PredictionDict


def visualize_estimation(
    color_image: torch.Tensor,
    depth_image: torch.Tensor,
    local_cv_position: torch.Tensor,
    local_cv_orientation_q: torch.Tensor,
    camera: Camera,
    instance_mask: Optional[torch.Tensor] = None,
    extents: Optional[torch.Tensor] = None,
    reconstructed_points: Optional[torch.Tensor] = None,
    reconstructed_mesh: Optional[o3d.geometry.TriangleMesh] = None,
    vis_camera_json: Optional[str] = None,
    render_options_json: Optional[str] = None,
    vis_path: Optional[str] = None,
) -> None:
    """Visualize prediction and ask for confirmation.

    Args:
        color_image: The unmasked color image. Shape (H,W,3), RGB, 0-1, float.
        depth_image: The unmasked depth image. Shape (H,W), float (meters along z).
        local_cv_position: The position in the OpenCV camera frame. Shape (3,).
        local_cv_orientation_q:
            The orientation in the OpenCV camera frame.  Scalar last, shape (4,).
        extents: Extents of the bounding box. Not visualized if None. Shape (3,).
        instance_mask: The instance mask. No masking if None. Shape (H,W).
        reconstructed_points:
            Reconstructed points in object coordinate frame. Not visualized if None.
            The points must already metrically scaled.
            Shape (M,3).
        reconstructed_mesh:
            Reconstructed mesh in object coordinate frame. Not visualized if None.
            The mesh must already metrically scaled.
        vis_camera_json:
            Path to open3d camera options json file that will be applied.
            Generated by pressing p in desired view.
            No render options will be applied if None.
        vis_path:
            If not None, the image will be rendered off screen and saved at the
            specified path.
    Returns:
        True if confirmation was positive. False if negative.
    """
    o3d_geometries = []

    local_cv_position = local_cv_position.cpu().double().numpy()  # shape (3,)
    local_cv_orientation_q = local_cv_orientation_q.cpu().double().numpy()  # shape (4,)

    if instance_mask is not None:
        valid_depth_mask = (depth_image != 0) * instance_mask
    else:
        valid_depth_mask = depth_image != 0
    pointset_colors = color_image[valid_depth_mask]
    masked_pointset = pointset_utils.depth_to_pointcloud(
        depth_image,
        camera,
        normalize=False,
        mask=instance_mask,
        convention="opencv",
    )
    o3d_points = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(masked_pointset.cpu().numpy())
    )
    o3d_points.colors = o3d.utility.Vector3dVector(pointset_colors.cpu().numpy())
    o3d_geometries.append(o3d_points)

    # coordinate frame
    local_cv_orientation_m = Rotation.from_quat(local_cv_orientation_q).as_matrix()
    o3d_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d_frame.rotate(
        local_cv_orientation_m,
        center=np.array([0.0, 0.0, 0.0])[:, None],
    )
    o3d_frame.translate(local_cv_position[:, None])
    o3d_geometries.append(o3d_frame)

    o3d_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    o3d_geometries.append(o3d_cam_frame)

    if extents is not None:
        extents = extents.cpu().double().numpy()
        o3d_obb = o3d.geometry.OrientedBoundingBox(
            center=local_cv_position[:, None],
            R=local_cv_orientation_m,
            extent=extents[:, None],
        )
        o3d_geometries.append(o3d_obb)

    if reconstructed_points is not None:
        o3d_rec_points = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(reconstructed_points.cpu().numpy())
        )
        o3d_rec_points.rotate(
            local_cv_orientation_m,
            center=np.array([0.0, 0.0, 0.0])[:, None],
        )
        o3d_rec_points.translate(local_cv_position[:, None])
        o3d_geometries.append(o3d_rec_points)

    if reconstructed_mesh is not None:
        # copy the mesh to keep original unmoved
        posed_mesh = o3d.geometry.TriangleMesh(reconstructed_mesh)
        posed_mesh.rotate(
            local_cv_orientation_m,
            center=np.array([0.0, 0.0, 0.0])[:, None],
        )
        posed_mesh.translate(local_cv_position[:, None])
        posed_mesh.compute_vertex_normals()
        o3d_geometries.append(posed_mesh)

    vis = o3d.visualization.Visualizer()
    if vis_camera_json is not None:
        vis_camera = o3d.io.read_pinhole_camera_parameters(vis_camera_json)
        width = vis_camera.intrinsic.width
        height = vis_camera.intrinsic.height
    else:
        width = 800
        height = 600
        vis_camera = None
    vis.create_window(width=width, height=height, visible=(vis_path is None))

    for g in o3d_geometries:
        vis.add_geometry(g)

    if vis_camera is not None:
        view_control = vis.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(vis_camera)

    if render_options_json is not None:
        render_option = vis.get_render_option()
        render_option.load_from_json(render_options_json)

    if vis_path is not None:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(vis_path, do_render=True)
    else:
        vis.run()

    # vis.destroy_window()
    # o3d.visualization.draw_geometries(o3d_geometries)


class Evaluator:
    """Class to evaluate various pose and shape estimation algorithms."""

    # ShapeNetV2 convention for all objects and datasets assumed
    # for simplicity assume all cans, bowls and bottles to be rotation symmetric
    SYMMETRY_AXIS_DICT = {
        "mug": None,
        "laptop": None,
        "camera": None,
        "can": 1,
        "bowl": 1,
        "bottle": 1,
    }

    def __init__(self, config: dict) -> None:
        """Initialize model wrappers and evaluator."""
        self._parse_config(config)

    def _parse_config(self, config: dict) -> None:
        """Read config and initialize method wrappers."""
        self._init_dataset(config["dataset_config"])

        self._visualize_input = config["visualize_input"]
        self._visualize_prediction = config["visualize_prediction"]
        self._visualize_gt = config["visualize_gt"]
        self._fast_eval = config["fast_eval"]
        self._store_visualization = config["store_visualization"]
        self._run_name = (
            f"{self._dataset_name}_eval_{config['run_name']}_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        self._out_folder = config["out_folder"]
        self._metrics = config["metrics"]
        self._num_gt_points = config["num_gt_points"]
        self._vis_camera_json = config["vis_camera_json"]
        self._render_options_json = config["render_options_json"]

        self._cam = Camera(**config["camera"])
        self._init_wrappers(config["methods"])

        self._config = config

    def _init_dataset(self, dataset_config: dict) -> None:
        """Initialize reading of dataset.

        This includes sanity checks whether the provided path is correct.
        """
        self._dataset_name = dataset_config["name"]
        print(f"Initializing {self._dataset_name} dataset...")
        dataset_type = str_to_object(dataset_config["type"])
        self._dataset = dataset_type(config=dataset_config["config_dict"])
        # Faster but probably only worth it if whole evaluation supports batches
        # self._dataloader = DataLoader(self._dataset, 1, num_workers=8)
        if len(self._dataset) == 0:
            print(f"No images found for dataset {self._dataset_name}")
            exit()
        print(f"{len(self._dataset)} samples found for dataset {self._dataset_name}.")

    def _init_wrappers(self, method_configs: dict) -> None:
        """Initialize method wrappers."""
        self._wrappers = {}
        for method_dict in method_configs.values():
            method_name = method_dict["name"]
            print(f"Initializing {method_name}...")
            wrapper_type = str_to_object(method_dict["wrapper_type"])
            self._wrappers[method_name] = wrapper_type(
                config=method_dict["config_dict"], camera=self._cam
            )

    def _eval_method(self, method_name: str, method_wrapper: MethodWrapper) -> None:
        """Run and evaluate method on all samples."""
        print(f"Run {method_name}...")
        self._init_metrics()
        indices = list(range(len(self._dataset)))
        random.shuffle(indices)
        for i in tqdm(indices):
            if self._fast_eval and i % 10 != 0:
                continue
            sample = self._dataset[i]
            if self._visualize_input:
                _, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2)
                ax1.imshow(sample["color"].numpy())
                ax2.imshow(sample["depth"].numpy())
                ax3.imshow(sample["mask"].numpy())
                plt.show()

            prediction = method_wrapper.inference(
                color_image=sample["color"],
                depth_image=sample["depth"],
                instance_mask=sample["mask"],
                category_str=sample["category_str"],
            )

            if self._visualize_gt:
                visualize_estimation(
                    color_image=sample["color"],
                    depth_image=sample["depth"],
                    local_cv_position=sample["position"],
                    local_cv_orientation_q=sample["quaternion"],
                    reconstructed_mesh=self._dataset.load_mesh(sample["obj_path"]),
                    extents=sample["scale"],
                    camera=self._cam,
                    vis_camera_json=self._vis_camera_json,
                    render_options_json=self._render_options_json,
                )
            if self._visualize_prediction:
                visualize_estimation(
                    color_image=sample["color"],
                    depth_image=sample["depth"],
                    local_cv_position=prediction["position"],
                    local_cv_orientation_q=prediction["orientation"],
                    extents=prediction["extents"],
                    reconstructed_points=prediction["reconstructed_pointcloud"],
                    reconstructed_mesh=prediction["reconstructed_mesh"],
                    camera=self._cam,
                    vis_camera_json=self._vis_camera_json,
                    render_options_json=self._render_options_json,
                )
            if self._store_visualization:
                out_folder = os.path.join(
                    self._out_folder, self._run_name, "visualization"
                )
                os.makedirs(out_folder, exist_ok=True)
                vis_path = os.path.join(out_folder, f"{i:06}_{method_name}.jpg")
                visualize_estimation(
                    color_image=sample["color"],
                    depth_image=sample["depth"],
                    local_cv_position=prediction["position"],
                    local_cv_orientation_q=prediction["orientation"],
                    extents=prediction["extents"],
                    reconstructed_points=prediction["reconstructed_pointcloud"],
                    reconstructed_mesh=prediction["reconstructed_mesh"],
                    camera=self._cam,
                    vis_camera_json=self._vis_camera_json,
                    render_options_json=self._render_options_json,
                    vis_path=vis_path,
                )

            self._eval_prediction(prediction, sample)
        self._finalize_metrics(method_name)

    def _eval_prediction(self, prediction: PredictionDict, sample: dict) -> None:
        """Evaluate all metrics for a prediction."""
        # correctness metric
        for metric_name in self._metrics.keys():
            self._eval_metric(metric_name, prediction, sample)

    def _init_metrics(self) -> None:
        """Initialize metrics."""
        self._metric_data = {}
        for metric_name, metric_config_dict in self._metrics.items():
            self._metric_data[metric_name] = self._init_metric_data(metric_config_dict)

    def _init_metric_data(self, metric_config_dict: dict) -> dict:
        """Create data structure necessary to compute a metric."""
        metric_data = {}
        if "position_thresholds" in metric_config_dict:
            pts = metric_config_dict["position_thresholds"]
            dts = metric_config_dict["deg_thresholds"]
            its = metric_config_dict["iou_thresholds"]
            fts = metric_config_dict["f_thresholds"]
            metric_data["correct_counters"] = np.zeros(
                (
                    len(pts),
                    len(dts),
                    len(its),
                    len(fts),
                    self._dataset.num_categories + 1,
                )
            )
            metric_data["total_counters"] = np.zeros(self._dataset.num_categories + 1)
        elif "pointwise_f" in metric_config_dict:
            metric_data["means"] = np.zeros(self._dataset.num_categories + 1)
            metric_data["m2s"] = np.zeros(self._dataset.num_categories + 1)
            metric_data["counts"] = np.zeros(self._dataset.num_categories + 1)
        else:
            raise NotImplementedError("Unsupported metric configuration.")
        return metric_data

    def _eval_metric(
        self, metric_name: str, prediction: PredictionDict, sample: dict
    ) -> None:
        """Evaluate and update single metric for a single prediction.

        Args:
            metric_name: Name of metric to evaluate.
            prediction: Dictionary containing prediction data.
            sample: Sample containing ground truth information.
        """
        metric_config_dict = self._metrics[metric_name]
        if "position_thresholds" in metric_config_dict:  # correctness metrics
            self._eval_correctness_metric(metric_name, prediction, sample)
        elif "pointwise_f" in metric_config_dict:  # pointwise reconstruction metrics
            self._eval_pointwise_metric(metric_name, prediction, sample)
        else:
            raise NotImplementedError(
                f"Unsupported metric configuration with name {metric_name}."
            )

    def _eval_correctness_metric(
        self, metric_name: str, prediction: PredictionDict, sample: dict
    ) -> None:
        """Evaluate and update single correctness metric for a single prediction.

        Args:
            metric_name: Name of metric to evaluate.
            prediction: Dictionary containing prediction data.
            sample: Sample containing ground truth information.
        """
        metric_dict = self._metrics[metric_name]
        correct_counters = self._metric_data[metric_name]["correct_counters"]
        total_counters = self._metric_data[metric_name]["total_counters"]
        category_id = sample["category_id"]
        total_counters[category_id] += 1
        total_counters[-1] += 1
        gt_points, pred_points = self._get_points(sample, prediction, True)
        for pi, p in enumerate(metric_dict["position_thresholds"]):
            for di, d in enumerate(metric_dict["deg_thresholds"]):
                for ii, i in enumerate(metric_dict["iou_thresholds"]):
                    for fi, f in enumerate(metric_dict["f_thresholds"]):
                        correct = metrics.correct_thresh(
                            position_gt=sample["position"].cpu().numpy(),
                            position_prediction=prediction["position"].cpu().numpy(),
                            orientation_gt=Rotation.from_quat(sample["quaternion"]),
                            orientation_prediction=Rotation.from_quat(
                                prediction["orientation"]
                            ),
                            extent_gt=sample["scale"].cpu().numpy(),
                            extent_prediction=prediction["extents"].cpu().numpy(),
                            points_gt=gt_points,
                            points_prediction=pred_points,
                            position_threshold=p,
                            degree_threshold=d,
                            iou_3d_threshold=i,
                            fscore_threshold=f,
                            rotational_symmetry_axis=self.SYMMETRY_AXIS_DICT[
                                sample["category_str"]
                            ],
                        )
                        correct_counters[pi, di, ii, fi, category_id] += correct
                        correct_counters[pi, di, ii, fi, -1] += correct  # all

    def _eval_pointwise_metric(
        self, metric_name: str, prediction: PredictionDict, sample: dict
    ) -> None:
        """Evaluate and update single pointwise metric for a single prediction.

        Args:
            metric_name: Name of metric to evaluate.
            prediction: Dictionary containing prediction data.
            sample: Sample containing ground truth information.
        """
        metric_config_dict = self._metrics[metric_name]
        means = self._metric_data[metric_name]["means"]
        m2s = self._metric_data[metric_name]["m2s"]
        counts = self._metric_data[metric_name]["counts"]
        category_id = sample["category_id"]
        point_metric = str_to_object(metric_config_dict["pointwise_f"])

        gt_points, pred_points = self._get_points(
            sample, prediction, metric_config_dict["posed"]
        )

        result = point_metric(
            gt_points.numpy(), pred_points.numpy(), **metric_config_dict["kwargs"]
        )

        # Use Welfords algorithm to update mean and variance
        # for category
        counts[category_id] += 1
        delta = result - means[category_id]
        means[category_id] += delta / counts[category_id]
        delta2 = result - means[category_id]
        m2s[category_id] += delta * delta2

        # for all
        counts[-1] += 1
        delta = result - means[-1]
        means[-1] += delta / counts[-1]
        delta2 = result - means[-1]
        m2s[-1] += delta * delta2

    def _get_points(
        self, sample: dict, prediction: PredictionDict, posed: bool
    ) -> Tuple[np.ndarray]:
        # load ground truth mesh
        gt_mesh = self._dataset.load_mesh(sample["obj_path"])
        gt_points = torch.from_numpy(
            np.asarray(gt_mesh.sample_points_uniformly(self._num_gt_points).points)
        )
        pred_points = prediction["reconstructed_pointcloud"]

        # transform points if posed
        if posed:
            gt_points = quaternion_utils.quaternion_apply(
                sample["quaternion"], gt_points
            )
            gt_points += sample["position"]
            pred_points = quaternion_utils.quaternion_apply(
                prediction["orientation"], pred_points
            )
            pred_points += prediction["position"]
        return gt_points, pred_points

    def _finalize_metrics(self, method_name: str) -> None:
        """Finalize metrics after all samples have been evaluated.

        Also writes results to disk and create plot if applicable.
        """
        out_folder = os.path.join(self._out_folder, self._run_name)
        os.makedirs(out_folder, exist_ok=True)
        yaml_path = os.path.join(out_folder, "results.yaml")

        self._results_dict[method_name] = {}
        for metric_name, metric_dict in self._metrics.items():
            if "position_thresholds" in metric_dict:  # correctness metrics
                correct_counter = self._metric_data[metric_name]["correct_counters"]
                total_counter = self._metric_data[metric_name]["total_counters"]
                correct_percentage = correct_counter / total_counter
                self._results_dict[method_name][
                    metric_name
                ] = correct_percentage.tolist()
                self._create_metric_plot(
                    method_name,
                    metric_name,
                    metric_dict,
                    correct_percentage,
                    out_folder,
                )
            elif "pointwise_f" in metric_dict:  # pointwise reconstruction metrics
                counts = self._metric_data[metric_name]["counts"]
                m2s = self._metric_data[metric_name]["m2s"]
                means = self._metric_data[metric_name]["means"]
                variances = m2s / counts
                stds = np.sqrt(variances)
                self._results_dict[method_name][metric_name] = {
                    "means": means.tolist(),
                    "variances": variances.tolist(),
                    "std": stds.tolist(),
                }
            else:
                raise NotImplementedError(
                    f"Unsupported metric configuration with name {metric_name}."
                )

        results_dict = {**self._config, "results": self._results_dict}
        yoco.save_config_to_file(yaml_path, results_dict)
        print(f"Results saved to: {yaml_path}")

    def _create_metric_plot(
        self,
        method_name: str,
        metric_name: str,
        metric_dict: dict,
        correct_percentage: np.ndarray,
        out_folder: str,
    ) -> None:
        """Create metric plot if applicable.

        Applicable means only one of the thresholds has multiple values.

        Args:
            correct_percentage:
                Array holding the percentage of correct predictions.
                Shape (NUM_POS_THRESH,NUM_DEG_THRESH,NUM_IOU_THRESH,NUM_CATEGORIES + 1).
        """
        axis = None
        for i, s in enumerate(correct_percentage.shape[:4]):
            if s != 1 and axis is None:
                axis = i
            elif s != 1:  # multiple axis with != 1 size
                return
        if axis is None:
            return
        axis_to_threshold_key = {
            0: "position_thresholds",
            1: "deg_thresholds",
            2: "iou_thresholds",
            3: "f_thresholds",
        }
        threshold_key = axis_to_threshold_key[axis]
        x_values = metric_dict[threshold_key]

        for category_id in range(self._dataset.num_categories + 1):
            y_values = correct_percentage[..., category_id].flatten()
            if category_id in self._dataset.category_id_to_str:
                label = self._dataset.category_id_to_str[category_id]
            else:
                label = "all"
            plt.plot(x_values, y_values, label=label)

        figure_path = os.path.join(out_folder, f"{method_name}_{metric_name}.png")
        plt.xlabel(threshold_key)
        plt.ylabel("Correct")
        plt.legend()
        plt.grid()
        plt.savefig(figure_path)
        plt.close()

    def run(self) -> None:
        """Run the evaluation."""
        self._results_dict = {}
        for method_name, method_wrapper in self._wrappers.items():
            self._eval_method(method_name, method_wrapper)


def main() -> None:
    """Entry point of the evaluation program."""
    parser = argparse.ArgumentParser(
        description="Pose and shape estimation evaluation on REAL275 data"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_folder", required=True)
    config = yoco.load_config_from_args(parser)

    evaluator = Evaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
