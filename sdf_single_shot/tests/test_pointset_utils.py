"""Tests for pointset_utils module."""

import torch
from sdf_single_shot import pointset_utils


def test_normalize_points() -> None:
    """Test normalize_points function."""
    # test no batch dimension
    points = torch.rand(10, 3)
    _, centroid = pointset_utils.normalize_points(points)
    assert centroid.shape == (3,)

    # test with batch dimension
    points = torch.rand(5, 10, 3)
    _, centroids = pointset_utils.normalize_points(points)
    assert centroids.shape == (5, 3)

    # test higher dimension
    points = torch.rand(7, 5, 10, 3)
    _, centroids = pointset_utils.normalize_points(points)
    assert centroids.shape == (7, 5, 3)

    # single point test
    points = torch.tensor([[1.0, 1.0, 1.0]])
    norm_points, centroid = pointset_utils.normalize_points(points)
    assert torch.all(points == torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.all(norm_points == torch.tensor([[0.0, 0, 0]]))
    assert torch.all(centroid == torch.tensor([[1.0, 1.0, 1.0]]))
