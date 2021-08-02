"""Utility functions to handle pointsets."""
import torch
from typing import Optional


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    """Normalize pointset to have zero mean.

    Normalization will be performed along second last dimension.

    Args:
        points:
            the pointsets which will be modified in place, shape (N, M, D),
            or shape (M, D), N pointsets with M points of dimension D

    Return:
        centroids:
            the means of the pointclouds used to normalize points
            shape (N, D) or (D,), for (N, M, D) and (M, D) inputs, respectively
    """
    centroids = torch.mean(points, dim=-2, keepdim=True)
    normalized_points = points - centroids
    return normalized_points, centroids.squeeze()


def depth_to_pointcloud(
    depth_image: torch.Tensor,
    f: float,
    normalize: bool,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert depth image to pointcloud.

    Args:
        depth_image: The depth image to convert to pointcloud.
        f: The focal length of the depth sensor.
        normalize: Whether to normalize the pointcloud with 0 centroid.
        mask: Only points with mask != 0 will be added to pointcloud

    Returns:
        the pointcloud in the camera frame, in OpenGL convention
    """
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    if mask is None:
        indices = torch.nonzero(depth_image, as_tuple=True)
    else:
        indices = torch.nonzero(depth_image * mask, as_tuple=True)
    depth_values = depth_image[indices]
    points = torch.cat(
        (
            indices[1][:, None].float(),
            indices[0][:, None].float(),
            depth_values[:, None],
        ),
        dim=1,
    )

    # OpenGL coordinate system
    final_points = torch.empty_like(points)
    final_points[:, 0] = (points[:, 0] - width / 2) * points[:, 2] / f
    final_points[:, 1] = (height / 2 - points[:, 1]) * points[:, 2] / f
    final_points[:, 2] = points[:, 2] * (-1.0)

    if normalize:
        final_points, _ = normalize_points(final_points)

    return final_points
