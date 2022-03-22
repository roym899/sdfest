"""Module containing loss functions."""

import torch

from sdfest.initialization import quaternion_utils


def nn_loss(points_from: torch.Tensor, points_to: torch.Tensor) -> torch.Tensor:
    """Compute the distance to the closest neighbor in the other set of points.

    Params:
        points_from:
            The first point set. Shape NxD, with N points of dimension D.
        points_to:
            The second point set. Shape MxD, with M points of dimension D.

    Returns:
        Squared distance from all points in the points_from set to the closest point in
        points to set. Output shape is (N,).
    """
    a = torch.sum(points_from ** 2, dim=1)
    b = torch.mm(points_from, points_to.t())
    c = torch.sum(points_to ** 2, dim=1)

    # compute the distance matrix
    d = -2 * b + a.unsqueeze(1) + c.unsqueeze(0)
    d[d < 0] = 0  # TODO why is it negative sometimes? numerical issues?
    d, _ = d.min(1)
    return d


def pc_loss(
    points: torch.Tensor,
    position: torch.Tensor,
    orientation: torch.Tensor,
    scale: torch.Tensor,
    sdf: torch.Tensor,
) -> torch.Tensor:
    """Compute trilinerly interpolated SDF value at point positions.

    Args:
        points:
            pointcloud in camera frame, shape (M, 4)
        position:
            position of SDF center in camera frame, shape (3,)
        orientation:
            quaternion representing orientation of SDF, shape (4,)
        scale:
            half-width of SDF volume
        sdf:
            volumetric signed distance field, shape (res, res, res),
            assuming same resolution along each axis

    Returns:
        Trilinearly interpolated distance at the passed points 0 if outside of SDF
        volume. Distance is in world coordinates (i.e., after scaling the SDF).
    """
    q = orientation / torch.norm(orientation)  # to get normalization gradients
    obj_points = points - position.unsqueeze(0)

    # Quaternion to rotation matrix
    # Note that we use conjugate here since we want to transform points from
    # world to object coordinates and the quaternion describes rotation of
    # object coordinate system in world coordinates.
    R = obj_points.new_zeros(3, 3)

    R[0, 0] = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    R[0, 1] = 2 * (q[0] * q[1] + q[2] * q[3])
    R[0, 2] = 2 * (q[0] * q[2] - q[3] * q[1])

    R[1, 0] = 2 * (q[0] * q[1] - q[2] * q[3])
    R[1, 1] = 1 - 2 * (q[0] * q[0] + q[2] * q[2])
    R[1, 2] = 2 * (q[1] * q[2] + q[3] * q[0])

    R[2, 0] = 2 * (q[0] * q[2] + q[3] * q[1])
    R[2, 1] = 2 * (q[1] * q[2] - q[3] * q[0])
    R[2, 2] = 1 - 2 * (q[0] * q[0] + q[1] * q[1])

    obj_points = (R @ obj_points.T).T

    # Transform to canonical coordintes obj_point in [-1,1]^3
    obj_point = obj_points / scale

    # Compute cell and cell position
    res = sdf.shape[0]  # assuming same resolution along each axis
    grid_size = 2.0 / (res - 1)
    c = torch.floor((obj_point + 1.0) * (res - 1) * 0.5)
    mask = torch.logical_or(
        torch.min(c, dim=1)[0] < 0, torch.max(c, dim=1)[0] > res - 2
    )
    c = torch.clip(c, 0, res - 2)  # base cell index of each point
    cell_position = c * grid_size - 1.0  # base cell position of each point
    sdf_indices = c.new_empty((obj_point.shape[0], 8), dtype=torch.long)
    sdf_indices[:, 0] = c[:, 0] * res ** 2 + c[:, 1] * res + c[:, 2]
    sdf_indices[:, 1] = c[:, 0] * res ** 2 + c[:, 1] * res + c[:, 2] + 1
    sdf_indices[:, 2] = c[:, 0] * res ** 2 + (c[:, 1] + 1) * res + c[:, 2]
    sdf_indices[:, 3] = c[:, 0] * res ** 2 + (c[:, 1] + 1) * res + c[:, 2] + 1
    sdf_indices[:, 4] = (c[:, 0] + 1) * res ** 2 + c[:, 1] * res + c[:, 2]
    sdf_indices[:, 5] = (c[:, 0] + 1) * res ** 2 + c[:, 1] * res + c[:, 2] + 1
    sdf_indices[:, 6] = (c[:, 0] + 1) * res ** 2 + (c[:, 1] + 1) * res + c[:, 2]
    sdf_indices[:, 7] = (c[:, 0] + 1) * res ** 2 + (c[:, 1] + 1) * res + c[:, 2] + 1
    sdf_view = sdf.view([-1])
    point_cell_position = (obj_point - cell_position) / grid_size  # [0,1]^3
    sdf_values = torch.take(sdf_view, sdf_indices)

    # trilinear interpolation
    sdf_value = sdf_values.new_empty(obj_points.shape[0])
    # sdf_value = obj_point[:, 0]
    sdf_value = (
        (
            sdf_values[:, 0] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 4] * point_cell_position[:, 0]
        )
        * (1 - point_cell_position[:, 1])
        + (
            sdf_values[:, 2] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 6] * point_cell_position[:, 0]
        )
        * point_cell_position[:, 1]
    ) * (1 - point_cell_position[:, 2]) + (
        (
            sdf_values[:, 1] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 5] * point_cell_position[:, 0]
        )
        * (1 - point_cell_position[:, 1])
        + (
            sdf_values[:, 3] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 7] * point_cell_position[:, 0]
        )
        * point_cell_position[:, 1]
    ) * point_cell_position[
        :, 2
    ]
    sdf_value[mask] = 0
    return sdf_value * scale


def point_constraint_loss(
    orientation_q: torch.Tensor, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Compute Euclidean distance between rotated source point and target point.

    Args:
        orientation_q:
            Orientation of object in world / camera frame as quaternion.
            Scalar-last convention. Shape (4,).
        source: Point in object frame, which will be transformed. (3,).
        target: Point in rotated oject frame. Shape (3,).
    Returns:
        Euclidean norm between R(orientation_q) @ source - target. Scalar.
    """
    rotated_source = quaternion_utils.quaternion_apply(orientation_q, source)
    return torch.linalg.norm(rotated_source - target)
