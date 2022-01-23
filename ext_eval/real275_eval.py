"""Script to run evaluation on REAL275 dataset.

Based on the following repos to follow the same evaluation
https://github.com/hughw19/NOCS_CVPR2019
https://github.com/xuchen-ethz/neural_object_fitting
"""
import argparse
import copy
import glob
import os
from datetime import datetime
import pickle
import random
from typing import Optional

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
import yoco
import open3d as o3d
from sdf_single_shot import pointset_utils
from sdf_differentiable_renderer import Camera
from sdf_single_shot.datasets.nocs_dataset import NOCSDataset
from sdf_single_shot.utils import str_to_object
from method_wrappers import MethodWrapper, PredictionDict

from sdf_estimation.simple_setup import SDFPipeline
from sdf_estimation.scripts.real_data import load_real275_rgbd
from sdf_estimation import metrics


# TODO visualize bounding box
def visualize_estimation(
    color_image: torch.Tensor,
    depth_image: torch.Tensor,
    local_cv_position: torch.Tensor,
    local_cv_orientation: torch.Tensor,
    camera: Camera,
    instance_mask: Optional[torch.Tensor] = None,
) -> None:
    """Visualize prediction and ask for confirmation.

    Args:
        color_image: The unmasked color image. Shape (H,W,3), RGB, 0-1, float.
        depth_image: The unmasked depth image. Shape (H,W), float (meters along z).
        instance_mask: The instance mask. No masking if None. Shape (H,W).
        category: Category string.
        local_cv_position: The position in the OpenCV camera frame. Shape (3,).
        local_cv_orientation:
            The orientation in the OpenCV camera frame.  Scalar last, shape (4,).
    Returns:
        True if confirmation was positive. False if negative.
    """
    o3d_geometries = []

    local_cv_position = local_cv_position.cpu().double().numpy()  # shape (3,)
    local_cv_orientation = local_cv_orientation.cpu().double().numpy()  # shape (4,)

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
    o3d_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d_frame.rotate(
        Rotation.from_quat(local_cv_orientation).as_matrix(),
        center=np.array([0.0, 0.0, 0.0])[:, None],
    )
    o3d_frame.translate(local_cv_position[:, None])
    o3d_geometries.append(o3d_frame)

    o3d_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    o3d_geometries.append(o3d_cam_frame)

    o3d.visualization.draw_geometries(o3d_geometries)


class REAL275Evaluator:
    """Class to evaluate various pose and shape estimation algorithms on REAL275."""

    NUM_CATEGORIES = 6  # (excluding all)

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
            if count > 1000:
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
            )

            if self._visualize_gt:
                visualize_estimation(
                    color_image=sample["color"],
                    depth_image=sample["depth"],
                    local_cv_position=sample["position"],
                    local_cv_orientation=sample["quaternion"],
                    camera=self._cam,
                )
            if self._visualize_prediction:
                visualize_estimation(
                    color_image=sample["color"],
                    depth_image=sample["depth"],
                    local_cv_position=prediction["position"],
                    local_cv_orientation=prediction["orientation"],
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
        self._correct_counters = {}
        self._total_counters = {}
        for metric_name, metric_dict in self._metrics.items():
            pts = metric_dict["position_thresholds"]
            dts = metric_dict["deg_thresholds"]
            its = metric_dict["iou_thresholds"]
            self._correct_counters[metric_name] = np.zeros(
                (len(pts), len(dts), len(its), self.NUM_CATEGORIES + 1)
            )
            self._total_counters[metric_name] = np.zeros(self.NUM_CATEGORIES + 1)

    def _eval_metric(
        self, metric_name: str, prediction: PredictionDict, sample: dict
    ) -> None:
        """Evaluate and update single metric for a single prediction.

        Args:
            metric_name: Name of metric to evaluate.
            prediction: Dictionary containing prediction data.
            sample: Sample containing ground truth information.
        """
        metric_dict = self._metrics[metric_name]
        correct_counter = self._correct_counters[metric_name]
        total_counter = self._total_counters[metric_name]
        cat = sample["category_id"]
        total_counter[cat - 1] += 1
        total_counter[6] += 1
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
                        rotational_symmetry_axis=None,  # TODO where to get this from hardcode?
                    )
                    correct_counter[pi, di, ii, cat - 1] += correct
                    correct_counter[pi, di, ii, 6] += correct  # all

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
            correct_counter = self._correct_counters[metric_name]
            total_counter = self._total_counters[metric_name]
            correct_percentage = correct_counter / total_counter
            self._results_dict[method_name][metric_name] = correct_percentage.tolist()
            # TODO create plot if applicable

        results_dict = {**self._config, "results": self._results_dict}
        yoco.save_config_to_file(yaml_path, results_dict)
        print(f"Results saved to: {yaml_path}")

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
