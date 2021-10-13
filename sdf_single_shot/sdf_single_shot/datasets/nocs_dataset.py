"""Module providing dataset class for NOCS datasets (CAMERA / REAL)."""
from glob import glob
from typing import List, NamedTuple, Optional
import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sdf_differentiable_renderer import Camera


class SampleSpecification(NamedTuple):
    """Specifies a single sample item.

    Each returned sample is a dictionary of multiple sample items.

    name:
        Specifies the key in the sample dictionary.
    type:
        Specifies the data type. One of:
            "color", "depth", "mask", "pointcloud"
    options:
        Data type specific options.
    """

    name: str
    type: str
    options: Optional[dict] = None


class NOCSDataset(torch.utils.data.Dataset):
    """Dataset class for NOCS dataset.

    CAMERA* and REAL* are training sets.
    CAMERA25 and REAL275 are test data.
    Some papers use CAMERA25 as validation when benchmarking REAL275.

    Datasets can be found here:
    https://github.com/hughw19/NOCS_CVPR2019/tree/master

    Expected directory format:
        {root_dir}/real_train/...
        {root_dir}/real_test/...
        {root_dir}/gts/...
        {root_dir}/obj_models/...
        {root_dir}/camera_composed_depth/...
        {root_dir}/camera_val25k/...
        {root_dir}/camera_train/...
    Which is easily obtained by downloading all the provided files and extracting them
    into the same directory.

    Necessary preprocessing of this data is performed during first initialization per
    and is saved to
        {root}/sdfest_pre/...
    """

    category_id_to_str = {
        0: "unknown",
        1: "bottle",
        2: "bowl",
        3: "camera",
        4: "can",
        5: "laptop",
        6: "mug",
    }
    category_str_to_id = {v: k for k, v in category_id_to_str.items()}

    # TODO category filter
    def __init__(
        self,
        root_dir: str,
        split: str,
        sample_specifications: List[SampleSpecification],
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir:
                Root dir of dataset. See class documentation for further information.
            split:
                The dataset split. The following strings are supported:
                "camera_train": 275000 images, synthetic foreground + real background
                "camera_val": 25000 images, synthetic foreground + real background
                "real_train": 4300 images in 7 scenes, real
                "real_test": 2750 images in 6 scenes, real
            sample_specifications: Specification for a returned sample.
        """
        self._root_dir = root_dir
        self._split = split
        self._preprocess_path = os.path.join(root_dir, "sdfest_pre", split)
        if not os.path.isdir(self._preprocess_path):
            self._preprocess_dataset()
        self._sample_files = self._get_sample_files()
        self._camera = self._get_split_camera()
        self._sample_specifications = sample_specifications

    def __len__(self) -> int:
        """Return number of sample in dataset."""
        return len(self._sample_files)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample of the dataset.

        Args:
            idx: Index of the instance.
        Returns:
            Sample containing the keys as specified by provided SampleSpecifications.
        """
        sample_file = self._sample_files[idx]
        sample_data = pickle.load(open(sample_file, "rb"))
        sample = self._sample_from_sample_data(sample_data)
        return sample

    def _preprocess_dataset(self) -> None:
        """Create preprocessing files for the current split.

        One file per sample, which currently means per valid object mask will be
        created.

        Preprocessing will be stored on disk to {root_dir}/sdfest_pre/...
        This function will not store the preprocessing, so it still has to be loaded
        afterwards.
        """
        os.makedirs(self._preprocess_path)

        color_files = self._get_color_files()
        counter = 0
        for color_file in color_files:
            depth_file = color_file.replace("color", "depth")
            mask_file = color_file.replace("color", "mask")
            meta_file = color_file.replace("color.png", "meta.txt")
            meta_data = pd.read_csv(meta_file, sep=" ", header=None)
            mask_img = np.asarray(Image.open(mask_file), dtype=np.uint8)
            if mask_img.ndim == 3 and mask_img.shape[2] == 4:  # CAMERA masks are RGBA
                mask = mask_img[:, :, 0]  # use first channel only
            else:  # REAL masks are grayscale
                mask = mask_img
            mask_ids = np.unique(mask).tolist()
            for mask_id in mask_ids:
                if mask_id == 255:  # 255 is background
                    continue
                match = meta_data[meta_data.iloc[:, 0] == mask_id]
                if match.empty:
                    print(f"Warning: mask {mask_id} not found in {meta_file}")
                elif match.shape[0] != 1:
                    print(f"Warning: mask {mask_id} not unique in {meta_file}")

                category_id = match.iloc[0, 1]
                if category_id == 0:  # unknown / distractor object
                    continue
                # TODO get transform, position, quaternion, scale
                # TODO symmetry
                # TODO canonical frame alignment (potentially different conventions)
                sample_info = {
                    "color_file": color_file,
                    "depth_file": depth_file,
                    "mask_file": mask_file,
                    "mask_id": mask_id,
                    "category_id": category_id,
                    # "shapenet_synset": mask_file,
                    # "shapenet_id": mask_file,
                    # "real_object_id": mask_file,
                    # "transform": transform,
                    # "position": position,
                    # "quaternion": orientation,
                    # "scale": scale,
                }
                out_file = os.path.join(self._preprocess_path, f"{counter:08}.pkl")
                pickle.dump(sample_info, open(out_file, "wb"))
                counter += 1

    def _get_color_files(self) -> list:
        """Return list of paths of color images of the selected split."""
        if self._split == "camera_train":
            glob_pattern = os.path.join(self._root_dir, "train", "**", "*_color.png")
            return glob(glob_pattern, recursive=True)
        elif self._split == "camera_val":
            glob_pattern = os.path.join(self._root_dir, "val", "**", "*_color.png")
            return glob(glob_pattern, recursive=True)
        elif self._split == "real_train":
            glob_pattern = os.path.join(
                self._root_dir, "real_train", "**", "*_color.png"
            )
            return glob(glob_pattern, recursive=True)
        elif self._split == "real_test":
            glob_pattern = os.path.join(
                self._root_dir, "real_test", "**", "*_color.png"
            )
            return glob(glob_pattern, recursive=True)
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")

    def _get_sample_files(self) -> list:
        """Return sorted list of sample file paths.

        Sample files are generated by NOCSDataset._preprocess_dataset.
        """
        glob_pattern = os.path.join(self._preprocess_path, "*.pkl")
        sample_files = glob(glob_pattern)
        sample_files.sort()
        return sample_files

    def _get_split_camera(self) -> None:
        """Return camera information for selected split."""
        # from: https://github.com/hughw19/NOCS_CVPR2019/blob/master/detect_eval.py
        if self._split in ["real_train", "real_test"]:
            return Camera(
                width=640,
                height=480,
                fx=591.0125,
                fy=590.16775,
                cx=322.525,
                cy=244.11084,
                pixel_center=0.0,
            )
        elif self._split in ["camera_train", "camera_val"]:
            return Camera(
                width=640,
                height=480,
                fx=577.5,
                fy=577.5,
                cx=319.5,
                cy=239.5,
                pixel_center=0.0,
            )

    def _sample_from_sample_data(self, sample_data: dict) -> dict:
        """Create dictionary containing sample items as specified."""
        sample = {}
        for sample_specification in self._sample_specifications:
            sample.update(self._create_sample_item(sample_specification, sample_data))
        return sample

    def _create_sample_item(
        self, sample_specification: SampleSpecification, sample_data: dict
    ) -> dict:
        """Create a single sample item based on specification and data."""
        print(sample_data)
        if sample_specification.type == "color":
            color = np.asarray(Image.open(sample_data["color_file"]))
            return {sample_specification.name: color}
        elif sample_specification.type == "depth":
            depth = (
                np.asarray(
                    Image.open(sample_data["depth_file"]), dtype=np.float64
                )
                * 0.001
            )
            return {sample_specification.name: depth}
        elif sample_specification.type == "pointcloud":
            # TODO support centering and masking of point cloud
            # TODO support noisy mask
            depth = (
                np.asarray(
                    Image.open(sample_data["depth_file"]), dtype=np.float64
                )
                * 0.001
            )
            raise NotImplementedError()
        elif sample_specification.type == "mask":
            mask = Image.open(sample_data["mask_file"])
            return {sample_specification.name: mask}
        else:
            raise ValueError(
                f"Unsupported SampleSpecification.type: {sample_specification.type}"
            )
