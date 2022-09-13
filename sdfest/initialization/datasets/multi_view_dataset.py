"""Module which provides SDFDataset class."""
import glob
import json
import os
from typing import Optional, TypedDict

import numpy as np
import open3d as o3d
import torch
import yoco
from scipy.spatial.transform import Rotation

from sdfest.initialization import pointset_utils, quaternion_utils, so3grid
from sdfest.initialization.datasets import dataset_utils

# TODO add abstraction for transforming conventions
# (this is common to all datasets, and shouldn't be repeated in each dataset)


class MultiViewDataset(torch.utils.data.Dataset):
    """Dataset class for UFOMap multi view dataset.

    Expected directory format:
        {root_dir}/{split}/{seq_id:08d}/cloud_{id:05d}.pcd
        {root_dir}/{split}/{seq_id:08d}/depth_{id:05d}.tiff
        {root_dir}/{split}/{seq_id:08d}/rgba_{id:05d}.png
        {root_dir}/{split}/{seq_id:08d}/segmentation_{id:05d}.png
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
                See sdfest.initialization.datasets.dataset_utils.get_scale.
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
        self.set_orientation_repr(
            config["orientation_repr"], config.get("orientation_grid_resolution")
        )

    def _init_dataset(self) -> None:
        dataset_dir = os.path.join(self._root_dir, self._split)
        pcd_glob = os.path.join(dataset_dir, "**", "*.pcd")
        self._pcd_files = glob.glob(pcd_glob, recursive=True)
        self._pcd_files.sort()

    def set_orientation_repr(
        self, orientation_repr: str, orientation_grid_resolution: Optional[int] = None
    ) -> None:
        """Change orientation representation of returned samples.

        Args:
            orientation_repr:
                Which orientation representation is used. One of:
                    "quaternion"
                    "discretized"
            orientation_grid_resolution:
                Resolution of the orientation grid.
                Only used if orientation_repr is "discretized".
        """
        self._orientation_repr = orientation_repr
        if self._orientation_repr == "discretized":
            self._orientation_grid = so3grid.SO3Grid(orientation_grid_resolution)

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
                    For "quaternion": FloatTensor, scalar-last quaternion, shape (4,).
                    For "discretized": LongTensor, scalar.
                "scale":
                    The scale of the SDF. Note that this is not exactly the same as
                    a tight bounding box. Based on specified scale_convention.
                    torch.FloatTensor, scalar or shape (3,).
                "sdf":
                    The discretized signed distance field.
                    torch.FloatTensor, Shape (N, N, N).
        """
        pcd_path = self._pcd_files[idx]
        dir_path = os.path.dirname(pcd_path)
        sdf_path = os.path.join(dir_path, "sdf.npy")
        meta_path = os.path.join(dir_path, "metadata.json")

        # Load meta data
        with open(meta_path) as json_file:
            meta_data = json.load(json_file)

        # Position
        position = torch.Tensor(meta_data["obj_position"])

        # Orientation
        orientation_q = torch.Tensor(meta_data["obj_quaternion"])
        orientation_q = torch.roll(orientation_q, -1)  # scalar-first -> scalar-last
        orientation_q = self._change_axis_convention(orientation_q)
        orientation = self._quat_to_orientation_repr(orientation_q)

        # Pointset
        o3d_pointset = o3d.io.read_point_cloud(pcd_path)
        np_pointset = np.asarray(o3d_pointset.points, dtype=np.float32)
        pointset = torch.from_numpy(np_pointset)

        if self._normalize_pointset:
            pointset, centroid = pointset_utils.normalize_points(pointset)
            position = position - centroid

        # SDF
        np_sdf = np.load(sdf_path)
        sdf = torch.from_numpy(np_sdf).float()

        # Scale
        sdf_scale_factor = meta_data["sdf_extent"]  # scalar, SDF max extent / tight bb
        tight_extents = torch.Tensor(meta_data["obj_extents"])
        sdf_extents = sdf_scale_factor * tight_extents
        scale = dataset_utils.get_scale(sdf_extents, self._scale_convention)

        return {
            "pointset": pointset,
            "position": position,
            "quaternion": orientation_q,
            "orientation": orientation,
            "scale": scale,
            "sdf": sdf,
        }

    def _change_axis_convention(self, orientation_q: torch.Tensor) -> tuple:
        """Adjust axis convention for the orientation.

        Args:
            orientation_q:
                Quaternion representing an orientation. Scalar-last, shape (4,).

        Returns:
            Quaternion representing orientation_q with remapped x and y axes.
            Scalar-last, shape (4,).
        """
        if self._remap_y_axis is None and self._remap_x_axis is None:
            return orientation_q
        elif self._remap_y_axis is None or self._remap_x_axis is None:
            raise ValueError("Either both or none of remap_{y,x}_axis have to be None.")

        rotation_o2n = self._get_o2n_object_rotation_matrix()

        # quaternion so far: original -> camera
        # we want a quaternion: new -> camera
        rotation_n2o = rotation_o2n.T

        quaternion_n2o = torch.from_numpy(Rotation.from_matrix(rotation_n2o).as_quat())

        remapped_orientation_q = quaternion_utils.quaternion_multiply(
            orientation_q, quaternion_n2o
        )  # new -> original -> camera

        return remapped_orientation_q

    def _get_o2n_object_rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix which rotates original to new object coordinates."""
        rotation_o2n = np.zeros((3, 3))  # original to new object convention
        if self._remap_y_axis == "x":
            rotation_o2n[0, 1] = 1
        elif self._remap_y_axis == "-x":
            rotation_o2n[0, 1] = -1
        elif self._remap_y_axis == "y":
            rotation_o2n[1, 1] = 1
        elif self._remap_y_axis == "-y":
            rotation_o2n[1, 1] = -1
        elif self._remap_y_axis == "z":
            rotation_o2n[2, 1] = 1
        elif self._remap_y_axis == "-z":
            rotation_o2n[2, 1] = -1
        else:
            raise ValueError("Unsupported remap_y_axis {self.remap_y}")

        if self._remap_x_axis == "x":
            rotation_o2n[0, 0] = 1
        elif self._remap_x_axis == "-x":
            rotation_o2n[0, 0] = -1
        elif self._remap_x_axis == "y":
            rotation_o2n[1, 0] = 1
        elif self._remap_x_axis == "-y":
            rotation_o2n[1, 0] = -1
        elif self._remap_x_axis == "z":
            rotation_o2n[2, 0] = 1
        elif self._remap_x_axis == "-z":
            rotation_o2n[2, 0] = -1
        else:
            raise ValueError("Unsupported remap_x_axis {self.remap_y}")

        # infer last column
        rotation_o2n[:, 2] = 1 - np.abs(np.sum(rotation_o2n, 1))  # rows must sum to +-1
        rotation_o2n[:, 2] *= np.linalg.det(rotation_o2n)  # make special orthogonal
        if np.linalg.det(rotation_o2n) != 1.0:  # check if special orthogonal
            raise ValueError("Unsupported combination of remap_{y,x}_axis. det != 1")
        return rotation_o2n

    def _quat_to_orientation_repr(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to selected orientation representation.

        Args:
            quaternion:
                The quaternion to convert, scalar-last, shape (4,).

        Returns:
            The same orientation as represented by the quaternion in the chosen
            orientation representation.
        """
        if self._orientation_repr == "quaternion":
            return quaternion
        elif self._orientation_repr == "discretized":
            index = self._orientation_grid.quat_to_index(quaternion.numpy())
            return torch.tensor(
                index,
                dtype=torch.long,
            )
        else:
            raise NotImplementedError(
                f"Orientation representation {self._orientation_repr} is not supported."
            )
