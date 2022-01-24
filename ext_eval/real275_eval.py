"""Script to run evaluation on REAL275 dataset."""
import argparse
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
import open3d as o3d
from sdf_estimation import metrics
from sdf_single_shot import pointset_utils
from sdf_differentiable_renderer import Camera
from sdf_single_shot.datasets.nocs_dataset import NOCSDataset
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
            Reconstructed points. In object coordinate frame. Not visualized if None.
            Shape (M,3).
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
        reconstructed_mesh.rotate(
            local_cv_orientation_m,
            center=np.array([0.0, 0.0, 0.0])[:, None],
        )
        reconstructed_mesh.translate(local_cv_position[:, None])
        o3d_geometries.append(reconstructed_mesh)

    o3d.visualization.draw_geometries(o3d_geometries)


class REAL275Evaluator:
    """Class to evaluate various pose and shape estimation algorithms on REAL275."""

    NUM_CATEGORIES = 6  # (excluding all)
    SYMMETRY_AXIS_DICT = {
        "mug": None,
        "laptop": None,
        "camera": None,
        "can": 1,
        "bowl": 1,
        "bottle": 1,
    }
    CATEGORY_ID_TO_STR = {
        0: "bottle",
        1: "bowl",
        2: "camera",
        3: "can",
        4: "laptop",
        5: "mug",
        6: "all",
    }

    def __init__(self, config: dict) -> None:
        """Initialize model wrappers and evaluator."""
        self._parse_config(config)

    def _parse_config(self, config: dict) -> None:
        """Read config and initialize method wrappers."""
        self._visualize_input = config["visualize_input"]
        self._visualize_prediction = config["visualize_prediction"]
        self._visualize_gt = config["visualize_gt"]
        self._store_visualization = config["store_visualization"]
        self._detection = config["detection"]
        self._run_name = (
            f"real275_eval_{config['run_name']}_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        self._out_folder = config["out_folder"]
        self._metrics = config["metrics"]

        self._cam = Camera(**config["camera"])
        self._init_wrappers(config["methods"])
        self._init_dataset(config)

        self._config = config

    def _init_dataset(self, config: dict) -> None:
        """Initialize reading of dataset.

        This includes sanity checks whether the provided path is correct.
        """
        print("Initializing NOCS dataset...")
        self._dataset = NOCSDataset(
            config={
                "root_dir": config["data_path"],
                "split": "real_test",
                "camera_convention": "opencv",
                "scale_convention": "full",
                "remap_y_axis": "y",  # ShapeNet convention
                "remap_x_axis": "-z",  # ShapeNet convention
            }
        )
        # Faster but probably only worth it if whole evaluation supports batches
        # self._dataloader = DataLoader(self._dataset, 1, num_workers=8)
        if len(self._dataset) == 0:
            print(f"No images found for data path {config['data_path']}")
            exit()
        print(f"{len(self._dataset)} detections found.")

    def _init_wrappers(self, method_configs: dict) -> None:
        """Initialize method wrappers."""
        self._wrappers = {}
        for method_name, method_dict in method_configs.items():
            if method_name.startswith("__"):
                continue
            print(f"Initializing {method_name}...")
            wrapper_type = str_to_object(method_dict["type"])
            self._wrappers[method_name] = wrapper_type(
                config=method_dict["config_dict"], camera=self._cam
            )

    def _eval_method(self, method_name: str, method_wrapper: MethodWrapper) -> None:
        """Run and evaluate method on all samples."""
        print(f"Run {method_name}...")
        self._init_metrics()
        count = 0
        for sample in tqdm(self._dataset):
            count += 1
            if count >= 100:
                break
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
                category_id=sample["category_id"],
                category_str=sample["category_str"],
            )

            if self._visualize_gt:
                visualize_estimation(
                    color_image=sample["color"],
                    depth_image=sample["depth"],
                    local_cv_position=sample["position"],
                    local_cv_orientation_q=sample["quaternion"],
                    camera=self._cam,
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
                )
            if self._store_visualization:
                # TODO store visualization of result on disk
                pass

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
            metric_data["correct_counters"] = np.zeros(
                (len(pts), len(dts), len(its), self.NUM_CATEGORIES + 1)
            )
            metric_data["total_counters"] = np.zeros(self.NUM_CATEGORIES + 1)
        elif "pointwise_f" in metric_config_dict:
            metric_data["sum"] = np.zeros(self.NUM_CATEGORIES + 1)
            metric_data["total_counters"] = np.zeros(self.NUM_CATEGORIES + 1)
        else:
            raise NotImplementedError(f"Unsupported metric configuration.")
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
        total_counters[category_id - 1] += 1
        total_counters[6] += 1
        for pi, p in enumerate(metric_dict["position_thresholds"]):
            for di, d in enumerate(metric_dict["deg_thresholds"]):
                for ii, i in enumerate(metric_dict["iou_thresholds"]):
                    correct = metrics.correct_thresh(
                        position_gt=sample["position"].cpu().numpy(),
                        position_prediction=prediction["position"].cpu().numpy(),
                        orientation_gt=Rotation.from_quat(sample["quaternion"]),
                        orientation_prediction=Rotation.from_quat(
                            prediction["orientation"]
                        ),
                        extent_gt=sample["scale"].cpu().numpy(),
                        extent_prediction=prediction["extents"].cpu().numpy(),
                        position_threshold=p,
                        degree_threshold=d,
                        iou_3d_threshold=i,
                        rotational_symmetry_axis=self.SYMMETRY_AXIS_DICT[
                            sample["category_str"]
                        ],
                    )
                    correct_counters[pi, di, ii, category_id - 1] += correct
                    correct_counters[pi, di, ii, 6] += correct  # all

    def _eval_pointwise_metric(
        self, metric_name: str, prediction: PredictionDict, sample: dict
    ) -> None:
        """Evaluate and update single pointwise metric for a single prediction.

        Args:
            metric_name: Name of metric to evaluate.
            prediction: Dictionary containing prediction data.
            sample: Sample containing ground truth information.
        """
        pass
        # TODO posed / or canonical reconstruction metric (chamfer ?)

    def _finalize_metrics(self, method_name: str) -> None:
        """Finalize metrics after all samples have been evaluated.

        Also writes results to disk and create plot if applicable.
        """
        out_folder = os.path.join(self._out_folder, self._run_name)
        os.makedirs(out_folder, exist_ok=True)
        yaml_path = os.path.join(out_folder, "results.yaml")

        self._results_dict[method_name] = {}
        for metric_name, metric_dict in self._metrics.items():
            correct_counter = self._metric_data[metric_name]["correct_counters"]
            total_counter = self._metric_data[metric_name]["total_counters"]
            correct_percentage = correct_counter / total_counter
            self._results_dict[method_name][metric_name] = correct_percentage.tolist()
            self._create_metric_plot(
                method_name, metric_name, metric_dict, correct_percentage, out_folder
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
        for i, s in enumerate(correct_percentage.shape[:3]):
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
        }
        threshold_key = axis_to_threshold_key[axis]
        x_values = metric_dict[threshold_key]

        for category_id in range(7):
            y_values = correct_percentage[..., category_id].flatten()
            plt.plot(x_values, y_values, label=self.CATEGORY_ID_TO_STR[category_id])

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
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_folder", required=True)
    config = yoco.load_config_from_args(parser)

    evaluator = REAL275Evaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
