"""Wrapper for pose and shape estimation methods."""
from abc import ABC
from typing import Optional, TypedDict

import torch

from sdf_differentiable_renderer import Camera


class PredictionDict(TypedDict):
    """Pose and shape prediction.

    Attributes:
        position:
            Position of object center in camera frame. OpenCV convention. Shape (3,).
        orientation:
            Orientation of object in camera frame. OpenCV convention.
            Scalar-last quaternion, shape (4,).
        extents:
            Bounding box side lengths., shape (3,).
        reconstructed_pointcloud:
            Reconstructed pointcloud in object frame.
            None if method does not perform reconstruction.
    """

    position: torch.Tensor
    orientation: torch.Tensor
    extents: torch.Tensor
    reconstructed_pointcloud: Optional[torch.Tensor]


class MethodWrapper(ABC):
    """Interface class for pose and shape estimation methods."""

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_id: int,
    ) -> PredictionDict:
        pass


class CASSWrapper(MethodWrapper):
    """Wrapper class for CASS."""

    def __init__(self, config: dict, camera: Camera) -> None:
        """Initialize and load CASS model."""
        pass

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_id: int,
    ) -> PredictionDict:
        return {
            "position": torch.tensor([0,0,0]),
            "orientation": torch.tensor([0,0,0,1]),
            "extents": torch.tensor([1,1,1]),
            "reconstructed_pointcloud": torch.tensor([[0,0,0]])
        }


class NOCSWrapper:
    """Wrapper class for NOCS."""

    def __init__(self, config: dict, camera: Camera) -> None:
        """Initialize and load NOCS model."""
        pass

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_id: int,
    ) -> PredictionDict:
        return {
            "position": torch.tensor([0,0,0]),
            "orientation": torch.tensor([0,0,0,1]),
            "extents": torch.tensor([1,1,1]),
            "reconstructed_pointcloud": torch.tensor([[0,0,0]])
        }
