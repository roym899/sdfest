"""Tests for pointset_utils module."""

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from sdfest.initialization import pointset_utils


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


def test_camera_convention_changes() -> None:
    """Test functions to change camera convention."""
    point_cv = torch.tensor([1.0, 1.0, 1.0])
    point_gl = pointset_utils.change_position_camera_convention(
        point_cv, "opencv", "opengl"
    )
    expected_point_gl = torch.tensor([1.0, -1.0, -1.0])
    assert torch.all(point_gl == expected_point_gl)
    point_cv_2 = pointset_utils.change_position_camera_convention(
        point_cv, "opencv", "opencv"
    )
    expected_point_cv2 = torch.tensor([1.0, 1.0, 1.0])
    assert torch.all(point_cv_2 == expected_point_cv2)

    # changing convention of transform is the same changing position and orientation
    rotation_mat_cv = Rotation.from_euler(
        "xyz", np.array([100, 70, -30]), degrees=True
    ).as_matrix()
    rotation_q_cv = Rotation.from_matrix(rotation_mat_cv).as_quat()
    s = 1.0
    position_cv = np.array([0.3, 1.0, 10.0])
    transform_cv = np.eye(4)
    transform_cv[:3, :3] = s * rotation_mat_cv
    transform_cv[:3, 3] = position_cv
    transform_cv = torch.from_numpy(transform_cv)
    transform_gl = pointset_utils.change_transform_camera_convention(
        transform_cv, "opencv", "opengl"
    )
    # going back to original orientation
    rotation_mat_gl = transform_gl[0:3, 0:3]
    rotation_q_gl = torch.from_numpy(Rotation.from_matrix(rotation_mat_gl).as_quat())
    rotation_q_cv_2 = pointset_utils.change_orientation_camera_convention(
        rotation_q_gl, "opengl", "opencv"
    ).numpy()
    assert np.allclose(rotation_q_cv, rotation_q_cv_2)

    # going back to original position
    position_gl = transform_gl[0:3, 3]
    position_cv_2 = pointset_utils.change_position_camera_convention(
        position_gl, "opengl", "opencv"
    ).numpy()
    assert np.allclose(position_cv, position_cv_2)
