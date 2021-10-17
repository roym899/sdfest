"""Module providing dataset class for NOCS datasets (CAMERA / REAL)."""
from glob import glob
import os
import pickle
from typing import TypedDict, Optional

from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from PIL import Image
from sdf_differentiable_renderer import Camera
import yoco

from sdf_single_shot import pointset_utils
from sdf_single_shot.datasets import nocs_utils


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

    class Config(TypedDict, total=False):
        """Configuration dictionary for NOCSDataset.

        Attributes:
            root_dir:
            split:
                The dataset split. The following strings are supported:
                    "camera_train": 275000 images, synthetic objects + real background
                    "camera_val": 25000 images, synthetic objects + real background
                    "real_train": 4300 images in 7 scenes, real
                    "real_test": 2750 images in 6 scenes, real
            mask_pointcloud: Whether the returned pointcloud will be masked.
            normalize_pointcloud:
                Whether the returned pointcloud and position will be normalized, such
                that pointcloud centroid is at the origin.
            scale_convention:
                Which scale is returned. The following strings are supported:
                    "diagonal": Length of bounding box' diagonal.
                    "max": Maximum side length of bounding box.
                    "half_max": Half maximum side length of bounding box.
            camera_convention:
                Which camera convention is used for position and orientation. One of:
                    "opengl": x right, y up, z back
                    "opencv": x right, y down, z forward
                Note that this does not influence how the dataset is processed, only the
                returned position and quaternion.
            up_axis:
                Which object axis corresponds to up axis in samples.
                One of: "x", "y", "z".
        """

        root_dir: str
        split: str
        mask_pointcloud: bool
        normalize_pointcloud: bool
        scale_convention: str
        camera_convention: str
        up_axis: str

    # TODO add support for up_axis
    # TODO add support for camera convention
    # TODO add support for scale convention

    default_config: Config = {
        "root_dir": None,
        "split": None,
        "mask_pointcloud": False,
        "normalize_pointcloud": False,
    }

    # TODO category filter
    # TODO use config dict for dataset
    # TODO uniform dataset code, adapt generated_dataset
    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the dataset.

        Args:
            config:
                Root dir of dataset. Provided dictionary will be merged with
                default_dict. See NOCSDataset.Config for supported keys.
        """
        self._config = yoco.load_config(config, default_dict=NOCSDataset.default_config)
        self._root_dir = self._config["root_dir"]
        self._split = self._config["split"]
        self._camera = self._get_split_camera()
        self._preprocess_path = os.path.join(self._root_dir, "sdfest_pre", self._split)
        if not os.path.isdir(self._preprocess_path):
            self._preprocess_dataset()
        self._sample_files = self._get_sample_files()
        self._mask_pointcloud = self._config["mask_pointcloud"]
        self._normalize_pointcloud = self._config["normalize_pointcloud"]

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

        color_paths = self._get_color_files()
        counter = 0
        for color_path in color_paths:
            depth_path = self._depth_path_from_color_path(color_path)
            mask_path = self._mask_path_from_color_path(color_path)
            meta_path = self._meta_path_from_color_path(color_path)
            meta_data = pd.read_csv(
                meta_path, sep=" ", header=None, converters={2: lambda x: str(x)}
            )
            instances_mask = self._load_mask(mask_path)
            mask_ids = np.unique(instances_mask).tolist()
            gt_id = 0  # GT only contains valid objects of interests and is 0-indexed
            for mask_id in mask_ids:
                if mask_id == 255:  # 255 is background
                    continue
                match = meta_data[meta_data.iloc[:, 0] == mask_id]
                if match.empty:
                    print(f"Warning: mask {mask_id} not found in {meta_path}")
                elif match.shape[0] != 1:
                    print(f"Warning: mask {mask_id} not unique in {meta_path}")

                meta_row = match.iloc[0]
                category_id = meta_row.iloc[1]
                if category_id == 0:  # unknown / distractor object
                    continue
                # TODO symmetry
                # TODO canonical frame alignment (potentially different conventions)
                (
                    position,
                    orientation_q,
                    max_extent,
                    nocs_scale,
                    nocs_transform,
                ) = self._get_pose_and_scale(color_path, mask_id, gt_id, meta_row)

                obj_path = self._get_obj_path(meta_row)
                sample_info = {
                    "color_path": color_path,
                    "depth_path": depth_path,
                    "mask_path": mask_path,
                    "mask_id": mask_id,
                    "category_id": category_id,
                    "obj_path": obj_path,
                    "nocs_transform": nocs_transform,
                    "position": position,
                    "orientation_q": orientation_q,
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
        color = torch.from_numpy(np.asarray(Image.open(sample_data["color_path"])))
        depth = self._load_depth(sample_data["depth_path"])
        # TODO support noisy mask
        instances_mask = self._load_mask(sample_data["mask_path"])
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
            "position": sample_data["position"],
            "orientation": sample_data["orientation_q"],  # TODO orientation repr
            "scale": sample_data["max_extent"],  # TODO scale repr
        }
        return sample

    def _depth_path_from_color_path(self, color_path: str) -> str:
        """Return path to depth file from color filepath."""
        if self._split in ["real_train", "real_test"]:
            depth_path = color_path.replace("color", "depth")
        elif self._split in ["camera_train"]:
            depth_path = color_path.replace("color", "composed")
            depth_path = depth_path.replace("/train/", "/camera_full_depths/train/")
        elif self._split in ["camera_val"]:
            depth_path = color_path.replace("color", "composed")
            depth_path = depth_path.replace("/val/", "/camera_full_depths/val/")
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")
        return depth_path

    def _mask_path_from_color_path(self, color_path: str) -> str:
        """Return path to mask file from color filepath."""
        mask_path = color_path.replace("color", "mask")
        return mask_path

    def _meta_path_from_color_path(self, color_path: str) -> str:
        """Return path to meta file from color filepath."""
        meta_path = color_path.replace("color.png", "meta.txt")
        return meta_path

    def _nocs_map_path_from_color_path(self, color_path: str) -> str:
        """Return NOCS map filepath from color filepath."""
        nocs_map_path = color_path.replace("color.png", "coord.png")
        return nocs_map_path

    def _get_pose_and_scale(
        self, color_path: str, mask_id: int, gt_id: int, meta_row: pd.Series
    ) -> tuple:
        """Return position, orientation, scale and NOCS transform.

        All of those follow OpenCV (x right, y down, z forward) convention.

        Args:
            color_path: Path to the color file.
            mask_id: Instance id in the instances mask.
            gt_id:
                Ground truth id. This is 0-indexed id of valid instances in meta file.
            meta_row:
                Matching row of meta file. Contains necessary information about mesh.

        Returns:
            position (np.ndarray):
                Position of object center in camera frame. Shape (3,).
            quaternion (np.ndarray):
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
        gts_path = self._get_gts_path(color_path)
        obj_path = self._get_obj_path(meta_row)
        if gts_path is None:  # camera_train and real_train
            # use ground truth NOCS mask to perform alignment
            (
                position,
                rotation_matrix,
                nocs_scale,
                nocs_transform,
            ) = self._estimate_object(color_path, mask_id)
            orientation = Rotation.from_matrix(rotation_matrix).as_quat()
            max_extent = self._get_max_extent_from_obj(obj_path)
        else:  # camera_val and real_test
            gts_data = pickle.load(open(gts_path, "rb"))
            nocs_transform = gts_data["gt_RTs"][gt_id]
            position = nocs_transform[0:3, 3]
            rot_scale = nocs_transform[0:3, 0:3]
            nocs_scales = np.sqrt(np.sum(rot_scale ** 2, axis=0))
            rotation_matrix = rot_scale / nocs_scales[:, None]
            nocs_scale = nocs_scales[0]
            orientation = Rotation.from_matrix(rotation_matrix).as_quat()
            if "gt_scales" in gts_data:  # camera val
                # CAMERA / ShapeNet meshes are normalized s.t. diagonal == 1
                gt_scales = gts_data["gt_scales"][gt_id]
                max_extent = np.max(gt_scales) * nocs_scale
            else:  # real test
                # REAL object meshes are not (!) normalized
                max_extent = self._get_max_extent_from_obj(obj_path)

        return position, orientation, max_extent, nocs_scale, nocs_transform

    def _get_gts_path(self, color_path: str) -> Optional[str]:
        """Return path to gts file from color filepath.

        Return None if split does not have ground truth information.
        """
        if self._split == "real_test":
            gts_folder = os.path.join(self._root_dir, "gts", "real_test")
        elif self._split == "camera_val":
            gts_folder = os.path.join(self._root_dir, "gts", "val")
        else:
            return None

        path = os.path.normpath(color_path)
        split_path = path.split(os.sep)
        number = path[-14:-10]
        gts_filename = f"results_{split_path[-3]}_{split_path[-2]}_{number}.pkl"
        gts_path = os.path.join(gts_folder, gts_filename)
        return gts_path

    def _get_obj_path(self, meta_row: pd.Series) -> str:
        """Return path to object file from meta data row."""
        if "camera" in self._split:  # ShapeNet mesh
            synset_id = meta_row.iloc[2]
            object_id = meta_row.iloc[3]
            obj_path = os.path.join(
                self._root_dir,
                "obj_models",
                self._split.replace("camera_", ""),
                synset_id,
                object_id,
                "model.obj",
            )
        elif "real" in self._split:  # REAL mesh
            object_id = meta_row.iloc[2]
            obj_path = os.path.join(
                self._root_dir, "obj_models", self._split, object_id + ".obj"
            )
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")
        return obj_path

    def _get_max_extent_from_obj(self, obj_path: str) -> float:
        """Return maximum extent of bounding box from obj filepath.

        Note that this is normalized extent (with diagonal == 1) in the case of CAMERA
        dataset, and unnormalized (i.e., metric) extent in the case of REAL dataset.
        """
        mesh = o3d.io.read_triangle_mesh(obj_path)
        vertices = np.asarray(mesh.vertices)
        extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        max_extent = np.max(extents)
        return max_extent

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load mask from mask filepath."""
        mask_img = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if mask_img.ndim == 3 and mask_img.shape[2] == 4:  # CAMERA masks are RGBA
            instances_mask = mask_img[:, :, 0]  # use first channel only
        else:  # REAL masks are grayscale
            instances_mask = mask_img
        return torch.from_numpy(instances_mask)

    def _load_depth(self, depth_path: str) -> torch.Tensor:
        """Load depth from depth filepath."""
        depth = torch.from_numpy(
            np.asarray(Image.open(depth_path), dtype=np.float64) * 0.001
        )
        return depth

    def _load_nocs_map(self, nocs_map_path: str) -> torch.Tensor:
        """Load NOCS map from NOCS map filepath.

        Returns:
            NOCS map where each channel corresponds to one dimension in NOCS.
            Coordinates are normalized to [0,1], shape (H,W,3).
        """
        nocs_map = torch.from_numpy(
            np.asarray(Image.open(nocs_map_path), dtype=np.float64) / 255
        )
        # z-coordinate has to be flipped
        # see https://github.com/hughw19/NOCS_CVPR2019/blob/14dbce775c3c7c45bb7b19269bd53d68efb8f73f/dataset.py#L327 # noqa: E501
        nocs_map[:, :, 2] = 1 - nocs_map[:, :, 2]
        return nocs_map[:, :, :3]

    def _estimate_object(self, color_path: str, mask_id: int) -> tuple:
        """Estimate pose and scale through ground truth NOCS map."""
        depth_path = self._depth_path_from_color_path(color_path)
        depth = self._load_depth(depth_path)
        mask_path = self._mask_path_from_color_path(color_path)
        instances_mask = self._load_mask(mask_path)
        instance_mask = instances_mask == mask_id
        nocs_map_path = self._nocs_map_path_from_color_path(color_path)
        nocs_map = self._load_nocs_map(nocs_map_path)
        valid_instance_mask = instance_mask * depth != 0
        nocs_map[~valid_instance_mask] = 0
        centered_nocs_points = nocs_map[valid_instance_mask] - 0.5
        measured_points = pointset_utils.depth_to_pointcloud(
            depth, self._camera, mask=valid_instance_mask, convention="opencv"
        )
        return nocs_utils.estimate_similarity_transform(
            centered_nocs_points, measured_points, verbose=False
        )
