"""General functions for experiments and pytorch."""
import inspect
from pydoc import locate
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


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
    """Visualize sample and prediction."""
    print(sample["position"])
    print(sample["quaternion"])
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

    set_axes_equal(ax)

    plt.show()


def save_checkpoint(path: str, model: torch.nn.Module, optimizer, iteration, run_name):
    """Save a checkpoint during training."""
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_name": run_name,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    """Load a checkpoint during training.

    Args:
        path: Path of the checkpoint file as produced by save_checkpoint.

    """
    print(f"Loading checkpoint at {path} ...")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    run_name = checkpoint["run_name"]

    print("Checkpoint loaded")

    model.train()  # training mode

    return model, optimizer, iteration, run_name


def load_model(path, model):
    """Load model weights from path."""
    print(f"Loading model from checkpoint at {path} ...")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded")
    return model


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
