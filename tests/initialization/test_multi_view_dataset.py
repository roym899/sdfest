"""Tests for multi_view_dataset module."""
import os

import torch

from sdfest.initialization.datasets import multi_view_dataset


def test_multi_view_dataset() -> None:
    """Test MultiViewDataset class."""
    mvd = multi_view_dataset.MultiViewDataset(
        {
            "root_dir": os.path.join(os.path.dirname(__file__), "multi_view_data"),
            "split": "train",
        }
    )
    assert len(mvd) == 2

    sample_0 = mvd[0]
    assert sample_0["pointset"].shape[0] > 50
    assert sample_0["pointset"].shape[1] == 3
    assert sample_0["pointset"].dtype == torch.float
    assert sample_0["sdf"].shape == (64, 64, 64)
    assert sample_0["sdf"].dtype == torch.float
    assert isinstance(sample_0["scale"], float)

    # Test with and without normalization
    mvd._normalize_pointset = True
    sample_0 = mvd[0]
    assert torch.isclose(torch.mean(sample_0["pointset"]), torch.Tensor([0]))

    mvd._normalize_pointset = False
    sample_0 = mvd[0]
    assert torch.mean(sample_0["pointset"]) != 0

    # Test different scale conventions
    mvd._scale_convention = "max"
    sample_0 = mvd[0]
    assert 0.1 < sample_0["scale"] < 0.15
    mvd._scale_convention = "half_max"
    sample_0 = mvd[0]
    assert 0.05 < sample_0["scale"] < 0.1
