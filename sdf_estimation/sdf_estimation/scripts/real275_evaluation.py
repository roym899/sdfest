"""Script to run evaluation on REAL275 dataset.

Based on the following repos to follow the same evaluation
https://github.com/hughw19/NOCS_CVPR2019
https://github.com/xuchen-ethz/neural_object_fitting
"""
import argparse
import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
import yoco

from sdf_estimation.simple_setup import SDFPipeline
from sdf_estimation.scripts.real_data import load_real275_rgbd


def main() -> None:
    """Entry point of the evaluation program."""
    parser = argparse.ArgumentParser(
        description="SDF pose estimation evaluation on REAL275 data"
    )
    parser.add_argument(
        "--config", default="configs/rendering_evaluation.yaml", nargs="+"
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
                results_dict["pred_RTs_sdfest"][mask_id] = np.eye(4,4)
                continue

            # apply estimation
            pipeline = pipeline_dict[category]
            depth_tensor = torch.from_numpy(depth).float().to(config["device"])
            rgb_tensor = torch.from_numpy(rgb).to(config["device"])
            mask_tensor = torch.from_numpy(mask).to(config["device"]) != 0

            position, orientation, scale, shape = pipeline(
                depth_tensor,
                mask_tensor,
                rgb_tensor,
                visualize=config["visualize_optimization"],
            )

            position = position.detach().cpu()
            rot_matrix = Rotation.from_quat(orientation.detach().cpu()).as_matrix()

            transform_gl = np.eye(4)
            transform_gl[0:3, 0:3] = rot_matrix * scale.detach().cpu().numpy() * 2
            transform_gl[0:3, 3] = position

            # TODO compute tight bounding box with marching cubes

            # OpenGL -> OpenCV convention
            transform_cv = transform_gl.copy()
            transform_cv[1,:] *= -1
            transform_cv[2,:] *= -1

            # fix canonical convention
            fix = np.array([[0,0,-1],[-1,0,0],[0,1,0]])
            transform_cv[0:3,0:3] = transform_cv[0:3,0:3] @ fix

            # NOCS Format
            # RT is 4x4 transformation matrix, where the rotation matrix
            # includes isotropic NOCS scale
            # RT is the transform from NOCS coordinate to camera coordinates
            # scale includes the tight bb size di separate for each axis
            # isotropic NOCS scale is sqrt(d1^2+d2^2+d3^2)

            # print(transform_cv[0:3,0:3])
            # nocs_rot = nocs_dict["gt_RTs"][3][0:3, 0:3]
            # nocs_rot_scale = np.sqrt((nocs_rot @ nocs_rot.T)[0,0])
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

            # store result
            results_dict["pred_RTs_sdfest"][mask_id] = transform_cv

        f = open(os.path.join(config["out_folder"], nocs_file_name), "wb")
        pickle.dump(results_dict, f, -1)  # Why -1 (from xuchen-ethz's repo)?



if __name__ == "__main__":
    main()
