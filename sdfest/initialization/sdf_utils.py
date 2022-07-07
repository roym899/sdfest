"""Utility functions to handle voxel-based signed distance fields."""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


def sdf_to_pointcloud(
    sdf: np.array,
    position: np.array,
    orientation: np.array,
    scale: float,
    max_points: Optional[int] = None,
    threshold: float = 0,
):
    """Convert SDF to pointcloud.

    Puts a point onto each cell vertex with a value < threshold.

    Args:
        sdf: The values of the voxelized SDF. Shape (D, D, D).
        position: The position of the SDF center. Shape (3,).
        orientation:
            The orientation of the SDF as a normalized quaternion.
            This is the quaternion that will be applied to each point.
            Scalar-last convention, shape (4,).
        scale: The half-length of the SDF.
        max_points: Maximum number of points in the pointcloud.
        threshold: The threshold below which a voxel will be included in the pointcloud.

    Returns:
        The pointcloud as a Nx3 array.
    """
    grid_size = 2.0 / (sdf.shape[0] - 1.0)
    indices = np.argwhere(sdf <= threshold)
    points = (indices * grid_size - 1.0) * scale
    rot_matrix = Rotation.from_quat(orientation).as_matrix()
    points = (rot_matrix @ points.T).T
    points += position
    if max_points is not None and max_points < points.shape[0]:
        points = points[np.random.choice(points.shape[0], 2, replace=False), :]
    return points


if __name__ == "__main__":
    sdf = np.random.random((10, 10, 10))
    pos = np.array([1.0, 0.5, 0.0])
    orientation = np.array([0.0, 0.0, 0.0, 1.0])
    orientation /= np.linalg.norm(orientation)
    points = sdf_to_pointcloud(sdf, pos, orientation, 0.15, 0.5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()
