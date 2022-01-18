"""Wrapper for CASS."""
import torch

from sdf_differentiable_renderer import Camera


class CASSWrapper:
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
    ) -> None:
        pass
