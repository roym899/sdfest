"""Module providing dataset class for NOCS datasets (CAMERA / REAL)."""
from glob import glob
import os
import pickle
from typing import Optional

from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sdf_differentiable_renderer import Camera

from sdf_single_shot import pointset_utils


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
        mask_pointcloud: bool = False,
        normalize_pointcloud: bool = False,
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
            mask_pointcloud: Whether the returned pointcloud will be masked.
            normalize_pointcloud:
                Whether the returned pointcloud and position will be normalized, such
                that pointcloud centroid is at the origin.
        """
        self._root_dir = root_dir
        self._split = split
        self._preprocess_path = os.path.join(root_dir, "sdfest_pre", split)
        if not os.path.isdir(self._preprocess_path):
            self._preprocess_dataset()
        self._sample_files = self._get_sample_files()
        self._camera = self._get_split_camera()
        self._mask_pointcloud = mask_pointcloud
        self._normalize_pointcloud = normalize_pointcloud

    def __len__(self) -> int:
        """Return number of sample in dataset."""
        return len(self._sample_files)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample of the dataset.

        Args:
            idx: Index of the instance.
        Returns:
            Sample containing the following items:
                "color"
                "depth"
                "mask"
                "pointcloud"
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
            depth_file = self._depth_file_from_color_file(color_file)
            mask_file = color_file.replace("color", "mask")
            meta_file = color_file.replace("color.png", "meta.txt")
            meta_data = pd.read_csv(meta_file, sep=" ", header=None)
            mask_img = np.asarray(Image.open(mask_file), dtype=np.uint8)
            if mask_img.ndim == 3 and mask_img.shape[2] == 4:  # CAMERA masks are RGBA
                mask = mask_img[:, :, 0]  # use first channel only
            else:  # REAL masks are grayscale
                mask = mask_img
            mask_ids = np.unique(mask).tolist()
            gt_id = 0  # GT only contains valid objects of interests and is 0-indexed
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
                (
                    position,
                    orientation,
                    max_extent,
                    nocs_scale,
                    nocs_transform,
                ) = self._get_pose_and_scale(color_file, gt_id)
                sample_info = {
                    "color_file": color_file,
                    "depth_file": depth_file,
                    "mask_file": mask_file,
                    "mask_id": mask_id,
                    "category_id": category_id,
                    # "shapenet_synset": mask_file,
                    # "shapenet_id": mask_file,
                    # "real_object_id": mask_file,
                    "nocs_transform": nocs_transform,
                    "position": position,
                    "quaternion": orientation,
                    "nocs_scale": nocs_scale,
                    "max_extent": max_extent,
                }
                out_file = os.path.join(self._preprocess_path, f"{counter:08}.pkl")
                pickle.dump(sample_info, open(out_file, "wb"))
                counter += 1
                gt_id += 1

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
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")

    def _sample_from_sample_data(self, sample_data: dict) -> dict:
        """Create dictionary containing a single sample."""
        color = torch.from_numpy(np.asarray(Image.open(sample_data["color_file"])))
        depth = torch.from_numpy(
            np.asarray(Image.open(sample_data["depth_file"]), dtype=np.float64) * 0.001
        )
        # TODO support noisy mask
        instances_mask = torch.from_numpy(
            np.asarray(Image.open(sample_data["mask_file"]))
        )
        instance_mask = instances_mask == sample_data["mask_id"]

        pointcloud_mask = instance_mask if self._mask_pointcloud else None
        pointcloud = pointset_utils.depth_to_pointcloud(
            depth, self._camera, mask=pointcloud_mask
        )
        if self._normalize_pointcloud:
            pointcloud, centroid = pointset_utils.normalize_points(pointcloud)

        sample = {
            "color": color,
            "depth": depth,
            "pointcloud": pointcloud,
            "mask": instance_mask,
        }
        return sample

    def _depth_file_from_color_file(self, color_file: str) -> str:
        """Return path to depth file from path to color file."""
        if self._split in ["real_train", "real_test"]:
            depth_file = color_file.replace("color", "depth")
        elif self._split in ["camera_train"]:
            depth_file = color_file.replace("color", "composed")
            depth_file = depth_file.replace("/train/", "/camera_full_depths/train/")
        elif self._split in ["camera_val"]:
            depth_file = color_file.replace("color", "composed")
            depth_file = depth_file.replace("/val/", "/camera_full_depths/val/")
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")
        return depth_file

    def _get_pose_and_scale(self, color_file: str, gt_id: int) -> tuple:
        """Return position, orientation, scale and NOCS transform.

        Returns:
            position (np.ndarray):
                Position of object center in camera frame. Shape (3,).
            orientation (np.ndarray):
                Orientation of object in camera frame.
                Scalar-last quaternion, shape (4,).
            max_extent (float):
                Maximum side length of bounding box.
            nocs_scale (float):
                Diagonal of NOCS bounding box.
            nocs_transformation:
                Transformation from centered [-0.5,0.5]^3 NOCS coordinates to camera.
        """
        # TODO OpenGL / CV camera convention?
        gts_path = self._get_gts_path(color_file)
        if gts_path is None:
            # TODO compute new ground truth information from nocs map
            position = (
                orientation
            ) = max_extent = nocs_scale = nocs_transformation = None
        else:
            gts_data = pickle.load(open(gts_path, "rb"))
            nocs_transformation = gts_data["gt_RTs"][gt_id]
            position = nocs_transformation[0:3, 3]
            rot_scale = nocs_transformation[0:3, 0:3]
            nocs_scales = np.sqrt(np.sum(rot_scale ** 2, axis=0))
            rotation_matrix = rot_scale / nocs_scales[:, None]
            nocs_scale = nocs_scales[0]
            orientation = Rotation.from_matrix(rotation_matrix).as_quat()
            max_extent = None

        return position, orientation, max_extent, nocs_scale, nocs_transformation

    def _get_gts_path(self, color_file: str) -> Optional[str]:
        """Return path to gts file for a color_file.

        Return None if split does not have ground truth information.
        """
        if self._split == "real_test":
            gts_folder = os.path.join(self._root_dir, "gts", "real_test")
        elif self._split == "camera_val":
            gts_folder = os.path.join(self._root_dir, "gts", "val")
        else:
            return None

        path = os.path.normpath(color_file)
        split_path = path.split(os.sep)
        number = path[-14:-10]
        gts_file_name = f"results_{split_path[-3]}_{split_path[-2]}_{number}.pkl"
        gts_file = os.path.join(gts_folder, gts_file_name)
        return gts_file
