"""Module providing dataset class for NOCS datasets (CAMERA / REAL)."""
from glob import glob
from typing import NamedTuple, Optional
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.io import read_image


class Sample(NamedTuple):
    """Represents a single data sample."""

    color: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    pointcloud: Optional[np.ndarray] = None
    masked_pointcloud: Optional[np.ndarray] = None


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

    def __init__(
        self,
        root_dir: str,
        split: str,
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
        """
        self._root_dir = root_dir
        self._split = split
        self._preprocess_path = os.path.join(root_dir, "sdfest_pre", split)
        if not os.path.isdir(self._preprocess_path):
            self._preprocess_dataset()

    def __len__(self) -> int:
        """Return number of sample in dataset."""
        # TODO compute length in preprocessing
        return self._length

    def __getitem__(self, idx: int) -> Sample:
        """Return a sample of the dataset.

        Args:
            idx: Index of the instance.
        Returns:
            inputs: Inputs to the network, as specified in the constructor.
            targets: Targets for the given input, as specified in the constructor.
        """
        # TODO load sample, and convert to desired input format
        return Sample()

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
        for color_file in tqdm(color_files):
            depth_file = color_file.replace("color", "depth")
            mask_file = color_file.replace("color", "mask")
            meta_file = color_file.replace("color.png", "meta.txt")
            meta_data = pd.read_csv(meta_file, sep=" ", header=None)
            mask_img = read_image(mask_file)
            if mask_img.shape[0] == 4:  # CAMERA masks are RGBA
                mask = mask_img.long()[0] + 256 * mask_img[1] + 65536 * mask_img[2]
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
                sample_info = {
                    "color": color_file,
                    "depth": depth_file,
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
