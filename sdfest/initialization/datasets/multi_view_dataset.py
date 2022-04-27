"""Module which provides SDFDataset class."""
from typing import Tuple, Optional, Iterator, TypedDict

import open3d as o3d
import torch
import yoco

class MultiViewDataset(torch.utils.data.IterableDataset):
    class Config(TypedDict, total=False):
        """Configuration dictionary for MultiViewDataset.

        Attributes:
            width: The width of the generated images in px.
            height: The height of the generated images in px.
            fov_deg: The horizontal fov in deg.
            z_min:
                Minimum z value (i.e., distance from camera) for the SDF.
                Note that positive z means in front of the camera, hence z_sampler
                should in most cases return positive values.
            z_max:
                Maximum z value (i.e., distance from camera) for the SDF.
            extent_mean:
                Mean extent of the SDF.
                Extent is the total side length of an SDF.
            extent_std:
                Standard deviation of the SDF scale.
            pointcloud: Whether to generate pointcloud or depth image.
            normalize_pose:
                Whether to center the augmented pointcloud at 0,0,0.
                Ignored if pointcloud=False
            orientation_repr:
                Which orientation representation is used. One of:
                    "quaternion"
                    "discretized"
            orientation_grid_resolution:
                Resolution of the orientation grid.
                Only used if orientation_repr is "discretized".
        """
        width: int
        height: int
        fov_deg: float
        z_min: float
        z_max: float
        extent_mean: float
        extent_std: float

    default_config: Config = {
        "device": "cuda",
        "width": 640,
        "height": 480,
        "fov_deg": 90,
    }

    def __init__(
        self,
        config: dict,
    ) -> None:
        config = yoco.load_config(config, current_dict=MultiViewDataset.default_config)
