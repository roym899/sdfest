"""Wrapper for pose and shape estimation methods."""
from abc import ABC

import torch

from sdf_differentiable_renderer import Camera


class MethodWrapper(ABC):
    def inference(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        category: int,
    ) -> None:
        pass


class CASSWrapper(MethodWrapper):
    """Wrapper class for CASS."""

    def __init__(self, config: dict, camera: Camera) -> None:
        """Initialize and load CASS model."""
        pass

    def inference(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        category: int,
    ) -> dict:
        pass


class NOCSWrapper:
    """Wrapper class for NOCS."""

    def __init__(self, config: dict, camera: Camera) -> None:
        """Initialize and load NOCS model."""
        pass

    def inference(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        category: int,
    ):
        pass
