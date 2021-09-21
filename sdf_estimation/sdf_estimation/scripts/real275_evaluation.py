"""Script to run evaluation on REAL275 dataset.

Based on the following repos to follow the same evaluation
https://github.com/hughw19/NOCS_CVPR2019
https://github.com/xuchen-ethz/neural_object_fitting
"""
import argparse
import os
import pickle
import matplotlib.pyplot as plt

import yoco

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

    categories = ['bottle','bowl','camera','can','laptop','mug']
    model_dict = {}  # maps category to category-specific model
    model_dict["mug"] = 1

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
        category_ids = nocs_dict["pred_class_ids"] - 1 # (num_masks,), class ids

        print(nocs_dict["gt_handle_visibility"]) 

        for mask_id, category_id in enumerate(category_ids):
            category = categories[category_id]  # name of category
            mask = masks[:,:,mask_id]  # (rows, cols,)

            # skip unsupported category
            if category not in model_dict:
                continue

            # apply estimation


if __name__ == "__main__":
    main()
