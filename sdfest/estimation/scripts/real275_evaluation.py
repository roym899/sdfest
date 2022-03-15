"""Script to run evaluation on REAL275 dataset.

Based on the following repos to follow the same evaluation
https://github.com/hughw19/NOCS_CVPR2019
https://github.com/xuchen-ethz/neural_object_fitting
"""
import argparse
import copy
import os
import pickle
import random

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
import yoco
import open3d as o3d
from sdf_single_shot import pointset_utils
from sdf_differentiable_renderer import Camera

from sdf_estimation.simple_setup import SDFPipeline
from sdf_estimation.scripts.real_data import load_real275_rgbd


def visualize_estimation(
    color_image: torch.Tensor,
    depth_image: torch.Tensor,
    instance_mask: torch.Tensor,
    local_cv_position: torch.Tensor,
    local_cv_orientation: torch.Tensor,
    camera: Camera,
) -> None:
    """Visualize prediction and ask for confirmation.

    Args:
        color_image: The unmasked color image. Shape (H,W,3), RGB, 0-1, float.
        depth_image: The unmasked depth image. Shape (H,W), float (meters along z).
        instance_mask: The instance mask. Shape (H,W).
        category: Category string.
        local_cv_position: The position in the OpenCV camera frame. Shape (1, 3,).
        local_cv_orientation:
            The orientation in the OpenCV camera frame.  Scalar last, shape (1, 4,).
    Returns:
        True if confirmation was positive. False if negative.
    """
    o3d_geometries = []

    local_cv_position = local_cv_position[0].cpu().double().numpy()  # shape (3,)
    local_cv_orientation = local_cv_orientation[0].cpu().double().numpy()  # shape (4,)

    valid_depth_mask = (depth_image != 0) * instance_mask
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


def main() -> None:
    """Entry point of the evaluation program."""
    parser = argparse.ArgumentParser(
        description="SDF pose estimation evaluation on REAL275 data"
    )
    parser.add_argument(
        "--config", default="configs/real275_evaluation.yaml", nargs="+"
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_folder", required=True)
    config = yoco.load_config_from_args(parser)

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
