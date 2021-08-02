"""Functions to handle transformations with quaternions.

Inspired by PyTorch3D, but using scalar-last convention and not enforcing scalar > 0.
https://github.com/facebookresearch/pytorch3d
"""

import torch


def quaternion_multiply(
    quaternions_1: torch.Tensor, quaternions_2: torch.Tensor
) -> torch.Tensor:
    """Multiply two quaternions representing rotations.

    The returning quaternion will represent the composition of the
    passed quaternions.

    Normal broadcasting rules apply.

    Args:
        quaternions_1:
            normalized quaternions of shape (..., 4), scalar-last convention
        quaternions_2:
            normalized quaternions of shape (..., 4), scalar-last convention
    """
    ax, ay, az, aw = torch.unbind(quaternions_1, -1)
    bx, by, bz, bw = torch.unbind(quaternions_2, -1)
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ow = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((ox, oy, oz, ow), -1)


def quaternion_apply(quaternions: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate points by quaternions representing rotations.

    The returned points are rotated by the rotations representing quaternions.

    Normal broadcasting rules apply.

    Args:
        quaternions:
            normalized quaternions of shape (..., 4), scalar-last convention
        points:
            points of shape (..., 3)
    """
    points_as_quaternions = points.new_zeros(points.shape[:-1] + (4,))
    points_as_quaternions[..., :-1] = points
    return quaternion_multiply(
        quaternion_multiply(quaternions, points_as_quaternions),
        quaternion_invert(quaternions),
    )[..., :-1]


def quaternion_invert(quaternions: torch.Tensor) -> torch.Tensor:
    """Invert quaternions representing orientations.

    Args:
        quaternions:
            the quaternions to invert, shape (..., 4), scalar-last convention
    Returns:
        inverted quaternions, same shape as quaternions
    """
    return quaternions * quaternions.new_tensor([-1, -1, -1, 1])
