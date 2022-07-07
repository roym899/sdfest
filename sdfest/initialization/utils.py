"""General functions for experiments and pytorch."""
import inspect
from pydoc import locate
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from sdfest.initialization import quaternion_utils


def str_to_object(name: str) -> Any:
    """Try to find object with a given name.

    First scope of calling function is checked for the name, then current environment
    (in which case name has to be a fully qualified name). In the second case, the
    object is imported if found.

    Args:
        name: Name of the object to resolve.
    Returns:
        The object which the provided name refers to. None if no object was found.
    """
    # check callers local variables
    caller_locals = inspect.currentframe().f_back.f_locals
    if name in caller_locals:
        return caller_locals[name]

    # check callers global variables (i.e., imported modules etc.)
    caller_globals = inspect.currentframe().f_back.f_globals
    if name in caller_globals:
        return caller_globals[name]

    # check environment
    return locate(name)


def visualize_sample(sample: Optional[dict] = None, prediction: Optional[dict] = None):
    """Visualize sample and prediction.

    Assumes the following conventions and keys
        "scale": Half maximum side length of bounding box.
        "quaternion: Scalar-last orientation of object.
    """
    pointset = sample["pointset"].cpu().numpy()
    plt.imshow(sample["mask"].cpu().numpy())
    plt.show()
    plt.imshow(sample["depth"].cpu().numpy())
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((1, 1, 1))
    max_points = 500
    if len(pointset) > max_points:
        indices = np.random.choice(len(pointset), replace=False, size=max_points)
        ax.scatter(pointset[indices, 0], pointset[indices, 1], pointset[indices, 2])
    else:
        ax.scatter(pointset[:, 0], pointset[:, 1], pointset[:, 2])

    _plot_coordinate_frame(ax, sample)

    _plot_bounding_box(ax, sample)

    set_axes_equal(ax)

    plt.show()


def _plot_coordinate_frame(ax, sample):
    axis_pts = sample["scale"].cpu() * torch.eye(3)
    axis_pts = quaternion_utils.quaternion_apply(sample["quaternion"].cpu(), axis_pts)
    axis_pts = axis_pts + sample["position"].cpu()
    axis_pts = axis_pts.numpy()
    origin = sample["position"].cpu()
    _plot_line(ax, origin, axis_pts[0], "r")
    _plot_line(ax, origin, axis_pts[1], "g")
    _plot_line(ax, origin, axis_pts[2], "b")


def _plot_bounding_box(ax, sample):
    border_pts = (
        torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        * sample["scale"].cpu()
    )
    border_pts = quaternion_utils.quaternion_apply(
        sample["quaternion"].cpu(), border_pts
    )
    border_pts = border_pts + sample["position"].cpu()
    _plot_line(ax, border_pts[0], border_pts[1], "k")
    _plot_line(ax, border_pts[0], border_pts[2], "k")
    _plot_line(ax, border_pts[0], border_pts[4], "k")
    _plot_line(ax, border_pts[1], border_pts[3], "k")
    _plot_line(ax, border_pts[1], border_pts[5], "k")
    _plot_line(ax, border_pts[2], border_pts[3], "k")
    _plot_line(ax, border_pts[2], border_pts[6], "k")
    _plot_line(ax, border_pts[3], border_pts[7], "k")
    _plot_line(ax, border_pts[4], border_pts[5], "k")
    _plot_line(ax, border_pts[4], border_pts[6], "k")
    _plot_line(ax, border_pts[5], border_pts[7], "k")
    _plot_line(ax, border_pts[6], border_pts[7], "k")


def _plot_line(ax, pt1, pt2, *args, **kwargs):
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], *args, **kwargs)


def set_axes_equal(ax) -> None:
    """Make axes of 3D plot have equal scale.

    This ensures that spheres appear as spheres, cubes as cubes, ...
    This is needed since Matplotlib's ax.set_aspect('equal') and
    and ax.axis('equal') are not supported for 3D.

    From: https://stackoverflow.com/a/31364297

    Args:
      ax: A Matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def dict_to(data_dict: dict, device: torch.device) -> dict:
    """Move values in dictionary of type torch.Tensor to a specfied device.

    Args:
        data_dict: Dictionary to be iterated over.
        device: Device to move objects of type torch.Tensor to.
    Returns:
        Dictionary containing the same keys and values as data_dict, but with all
        objects of type torch.Tensor moved to the specified device.
    """
    new_data_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            new_data_dict[k] = v.to(device)
        else:
            new_data_dict[k] = v
    return new_data_dict
