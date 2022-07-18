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
    assert sample_0["position"].shape == (3,)
    assert sample_0["quaternion"].shape == (4,)
    assert sample_0["pointset"].shape[0] > 50
    assert sample_0["pointset"].shape[1] == 3
    assert sample_0["pointset"].dtype == torch.float
    assert sample_0["sdf"].shape == (64, 64, 64)
    assert sample_0["sdf"].dtype == torch.float
    assert isinstance(sample_0["scale"], float)

    # Test with and without normalization
    mvd._normalize_pointset = False
    sample_0 = mvd[0]
    centroid = torch.mean(sample_0["pointset"], dim=0)
    absolute_position = sample_0["position"]
    assert torch.all(centroid != 0)

    mvd._normalize_pointset = True
    sample_0 = mvd[0]
    normalized_position = sample_0["position"]
    assert torch.allclose(normalized_position, absolute_position - centroid)
    assert torch.isclose(torch.mean(sample_0["pointset"]), torch.Tensor([0]))

    # Test different scale conventions
    mvd._scale_convention = "max"
    sample_0 = mvd[0]
    assert 0.1 < sample_0["scale"] < 0.15
    mvd._scale_convention = "half_max"
    sample_0 = mvd[0]
    assert 0.05 < sample_0["scale"] < 0.1

    # Test different orientation conventions
    mvd.set_orientation_repr("quaternion")
    sample_q = mvd[0]
    assert sample_q["orientation"].shape == (4,)
    mvd.set_orientation_repr("discretized", 1)
    sample_d1 = mvd[0]
    assert sample_d1["orientation"].dtype == torch.long
