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

import torch
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

    for nocs_file_name in nocs_file_names:
        nocs_file_path = os.path.join(config["data_path"], "nocs_det", nocs_file_name)
        nocs_dict = pickle.load(open(nocs_file_path, "rb"), encoding="utf-8")
        rgb_file_path = (
            nocs_dict["image_path"].replace("data/real/test", config["data_path"])
            + "_color.png"
        )
        rgb, depth, _, _ = load_real275_rgbd(rgb_file_path)
        masks = nocs_dict["pred_mask"]  # (rows, cols, num_masks,), binary masks
        category_ids = nocs_dict["pred_class_ids"] - 1  # (num_masks,), class ids

        for mask_id, category_id in enumerate(category_ids):
            category = categories[category_id]  # name of category
            mask = masks[:, :, mask_id]  # (rows, cols,)

            # skip unsupported category
            if category not in pipeline_dict:
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

            # TODO: convert to transformation matrix and adjust convention

            # TODO: store result


if __name__ == "__main__":
    main()
