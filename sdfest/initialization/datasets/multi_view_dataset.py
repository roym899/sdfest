"""Module which provides SDFDataset class."""
import glob
import os
from typing import TypedDict

import numpy as np
import open3d as o3d
import torch
import yoco

from sdfest.initialization import pointset_utils, quaternion_utils, so3grid

# TODO add abstraction for transforming conventions
# (this is common to all datasets, and shouldn't be repeated in each dataset)


class MultiViewDataset(torch.utils.data.Dataset):
    """Dataset class for UFOMap multi view dataset.

    Expected directory format:
        {root_dir}/{split}/{seq_id:08d}/cloud_{id:05d}.pcd
        {root_dir}/{split}/{seq_id:08d}/depth_{id:05d}.tiff
        {root_dir}/{split}/{seq_id:08d}/rgba_{id:05d}.tiff
        {root_dir}/{split}/{seq_id:08d}/segmentation_{id:05d}.tiff
        {root_dir}/{split}/{seq_id:08d}/segmentation_{id:05d}.tiff
        {root_dir}/{split}/{seq_id:08d}/sdf.npy
        {root_dir}/{split}/{seq_id:08d}/metadata.json
    """

    class Config(TypedDict, total=False):
        """Configuration dictionary for MultiViewDataset.

        Attributes:
            root_dir: See MultiViewDataset docstring.
            split: The dataset split. See MultiViewDataset.
            normalize_pointset:
                Whether the returned pointset and position will be normalized, such
                that pointset centroid is at the origin.
            scale_convention:
                Which scale is returned. The following strings are supported:
                    "diagonal":
                        Length of bounding box' diagonal. This is what NOCS uses.
                    "max": Maximum side length of bounding box.
                    "half_max": Half maximum side length of bounding box.
                    "full": Bounding box side lengths. Shape (3,).
            orientation_repr:
                Which orientation representation is used. One of:
                    "quaternion"
                    "discretized"
            orientation_grid_resolution:
                Resolution of the orientation grid.
                Only used if orientation_repr is "discretized".
            remap_y_axis:
                If not None, the original y-axis will be mapped to the provided axis.
                Resulting coordinate system will always be right-handed.
                This is typically the up-axis.
                To get ShapeNetV2 alignment: y / None
                One of: "x", "y", "z", "-x", "-y", "-z"
            remap_x_axis:
                If not None, the original x-axis will be mapped to the provided axis.
                Resulting coordinate system will always be right-handed.
                To get ShapeNetV2 alignment: x / None
                One of: "x", "y", "z", "-x", "-y", "-z"
        """

        root_dir: str
        split: str
        normalize_pointset: bool
        scale_convention: str

    default_config: Config = {
        "root_dir": None,
        "split": None,
        "normalize_pointset": False,
        "scale_convention": "half_max",
        "orientation_repr": "quaternion",
        "orientation_grid_resolution": None,
        "remap_y_axis": None,
        "remap_x_axis": None,
    }

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize dataset.

        Args:
            config:
                Configuration dictionary for dataset.
                See MultiViewDataset.Config for details.
        """
        config = yoco.load_config(config, current_dict=MultiViewDataset.default_config)
        self._parse_config(config)
        self._init_dataset()

    def _parse_config(self, config: dict) -> None:
        self._root_dir = config["root_dir"]
        self._split = config["split"]
        self._normalize_pointset = config["normalize_pointset"]
        self._scale_convention = config["scale_convention"]
        self._remap_y_axis = config["remap_y_axis"]
        self._remap_x_axis = config["remap_x_axis"]
        self._orientation_repr = config["orientation_repr"]
        if self._orientation_repr == "discretized":
            self._orientation_grid = so3grid.SO3Grid(
                config["orientation_grid_resolution"]
            )

    def _init_dataset(self) -> None:
        dataset_dir = os.path.join(self._root_dir, self._split)
        pcd_glob = os.path.join(dataset_dir, "**", "*.pcd")
        self._pcd_files = glob.glob(pcd_glob, recursive=True)

    def __len__(self) -> int:
        """Return number of sample in dataset."""
        return len(self._pcd_files)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample of the dataset.

        Args:
            idx: Index of the instance.

        Returns:
            Sample containing the following items:
                "pointset": The voxelized pointset containing all occupied cells.
                "position": The position of the SDF. torch.FloatTensor, shape (3,).
                "quaternion":
                    The orientation of the SDF in the pointset as a quaternion.
                    Scalar-last quaternion, torch.FloatTensor, shape (4,).
                "orientation":
                    The orientation of the SDF in the specified orientation
                    representation. Based on specified orientation_representation.
                    torch.FloatTensor, shape depending on orientation_representation.
                "scale":
                    The scale of the SDF. Based on specified scale_convention.
                "sdf":
                    The discretized signed distance field.
                    torch.FloatTensor, Shape (N, N, N).
        """
        pcd_path = self._pcd_files[idx]
        dir_path = os.path.dirname(pcd_path)
        sdf_path = os.path.join(dir_path, "sdf.npy")
        meta_path = os.path.join(dir_path, "metadata.json")

        # TODO position

        # TODO quaternion

        # Pointset
        o3d_pointset = o3d.io.read_point_cloud(pcd_path)
        np_pointset = np.asarray(o3d_pointset.points, dtype=np.float32)
        pointset = torch.from_numpy(np_pointset)

        if self._normalize_pointset:
            pointset, _ = pointset_utils.normalize_points(pointset)
            # position = position - centroid

        # SDF
        np_sdf = np.load(sdf_path)
        sdf = torch.from_numpy(np_sdf).float()

        # TODO Scale

        return {
            "pointset": pointset,
            "position": None,
            "quaternion": None,
            "orientation": None,
            "scale": None,
            "sdf": sdf,
        }
