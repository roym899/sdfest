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
                    local_cv_position=sample["position"],
                    local_cv_orientation=sample["orientation"],
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
        """Evaluate and update single metric for a single prediction."""
        metric_dict = self._metrics[metric_name]
        correct_counter = self._correct_counters[metric_name]
        total_counter = self._total_counters[metric_name]
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
                    cat = sample["category_id"]
                    correct_counter[pi, di, ii, cat - 1] += correct
                    correct_counter[pi, di, ii, 6] += correct  # all
                    total_counter[cat - 1] += 1
                    total_counter[6] += 1

        # TODO posed / or canonical reconstruction metric (chamfer ?)

    def _prepare_metric_args(self, prediction: PredictionDict, sample: dict) -> dict:
        """Convert prediction and sample dicts to format required by metric."""
        return {}

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


def old_stuff():
    os.makedirs(config["out_folder"], exist_ok=True)

    categories = ["bottle", "bowl", "camera", "can", "laptop", "mug"]
    pipeline_dict = {}  # maps category to category-specific pipeline

    # create per-categry models
    for category in categories:
        if category in config["category_configs"]:
            category_config = yoco.load_config(
                config["category_configs"][category], copy.deepcopy(config)
            )
            pipeline_dict[category] = SDFPipeline(category_config)

    nocs_file_names = sorted(os.listdir(os.path.join(config["data_path"], "nocs_det")))
    random.shuffle(nocs_file_names)

    for nocs_file_name in tqdm(nocs_file_names):
        nocs_file_path = os.path.join(config["data_path"], "nocs_det", nocs_file_name)
        nocs_dict = pickle.load(open(nocs_file_path, "rb"), encoding="utf-8")
        results_dict = copy.deepcopy(nocs_dict)
        rgb_file_path = (
            nocs_dict["image_path"].replace("data/real/test", config["data_path"])
            + "_color.png"
        )
        rgb, depth, _, _ = load_real275_rgbd(rgb_file_path)
        masks = nocs_dict["pred_mask"]  # (rows, cols, num_masks,), binary masks
        category_ids = nocs_dict["pred_class_ids"] - 1  # (num_masks,), class ids
        results_dict["pred_RTs_sdfest"] = np.zeros_like(results_dict["pred_RTs"])

        for mask_id, category_id in enumerate(category_ids):
            category = categories[category_id]  # name of category
            mask = masks[:, :, mask_id]  # (rows, cols,)

            # skip unsupported category
            if category not in pipeline_dict:
                results_dict["pred_RTs_sdfest"][mask_id] = np.eye(4, 4)
                continue

            # apply estimation
            pipeline = pipeline_dict[category]
            depth_tensor = torch.from_numpy(depth).float().to(config["device"])
            rgb_tensor = torch.from_numpy(rgb).to(config["device"])
            mask_tensor = torch.from_numpy(mask).to(config["device"]) != 0

            position, orientation, scale, shape = pipeline(
                depth_tensor.clone(),
                mask_tensor,
                rgb_tensor,
                visualize=config["visualize_optimization"],
            )

            position_gl = position.detach().cpu()
            orientation_gl = orientation.detach().cpu()

            # visualize prediction
            # position_cv = pointset_utils.change_position_camera_convention(
            #     position_gl, "opengl", "opencv"
            # )
            # orientation_cv = pointset_utils.change_orientation_camera_convention(
            #     orientation_gl, "opengl", "opencv"
            # )
            # visualize_estimation(
            #     rgb_tensor,
            #     depth_tensor,
            #     mask_tensor,
            #     position_cv,
            #     orientation_cv,
            #     pipeline.cam,
            # )

            rot_matrix = Rotation.from_quat(orientation_gl).as_matrix()

            transform_gl = np.eye(4)
            transform_gl[0:3, 0:3] = rot_matrix * scale.detach().cpu().numpy() * 2
            transform_gl[0:3, 3] = position_gl

            # TODO compute tight bounding box with marching cubes

            # OpenGL -> OpenCV convention
            transform_cv = transform_gl.copy()
            transform_cv[1, :] *= -1
            transform_cv[2, :] *= -1

            # fix canonical convention
            fix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            transform_cv[0:3, 0:3] = transform_cv[0:3, 0:3] @ fix

            # NOCS Format
            # RT is 4x4 transformation matrix, where the rotation matrix
            # includes isotropic NOCS scale
            # RT is the transform from NOCS coordinate to camera coordinates
            # scale includes the tight bb size di separate for each axis
            # isotropic NOCS scale is sqrt(d1^2+d2^2+d3^2)

            # print(rot_matrix @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            # nocs_rot = nocs_dict["gt_RTs"][mask_id][0:3, 0:3]
            # nocs_rot_scale = np.sqrt((nocs_rot @ nocs_rot.T)[0, 0])
            # print(np.cbrt(np.linalg.det(nocs_rot)), nocs_rot_scale)
            # nocs_rot_noscale = nocs_rot / nocs_rot_scale
            # print(np.linalg.det(nocs_rot_noscale))
            # print(np.linalg.det(nocs_dict["pred_RTs"][mask_id, 0:3, 0:3]))
            # print(transform_cv)
            # print(nocs_rot_noscale)
            # print(transform_cv[0:3,0:3] @ np.array([1,0,0]))
            # print(nocs_rot_noscale[0:3,0:3] @ np.array([1,0,0]))
            # print(nocs_rot, nocs_rot_scale)
            # print(nocs_dict["gt_RTs"][3])
            # print(r_no_scale)
            # print(np.linalg.det(r_no_scale))

            # Visualize Ground Truth (show full depth, as masks <-> RTs order doesn't match)
            # gt_position = torch.from_numpy(nocs_dict["gt_RTs"][mask_id][0:3, 3])[None]
            # gt_orientation = torch.from_numpy(
            #     Rotation.from_matrix(nocs_rot_noscale).as_quat()
            # )[None]
            # visualize_estimation(
            #     rgb_tensor,
            #     depth_tensor,
            #     depth_tensor != 0,
            #     gt_position,
            #     gt_orientation,
            #     pipeline.cam,
            # )

            # store result
            results_dict["pred_RTs_sdfest"][mask_id] = transform_cv

        f = open(os.path.join(config["out_folder"], nocs_file_name), "wb")
        pickle.dump(results_dict, f, -1)  # Why -1 (from xuchen-ethz's repo)?
        f.close()


if __name__ == "__main__":
    main()
