"""Wrapper for NOCS."""
import torch

from sdf_differentiable_renderer import Camera


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
