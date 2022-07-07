"""General functions for experiments and pytorch."""
from typing import Union

import matplotlib.pyplot as plt
import torch


def visualize_sdf_reconstruction(
    sdf: torch.Tensor, sdf_reconstruction: torch.Tensor, show: bool = False
):
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)

    if show:
        plt.show()

    return fig


def visualize_batch(sdfs: torch.Tensor, show: bool = False):
    fig = plt.figure()
    num_sdfs = sdfs.shape[0]

    # find nice layout
    cols = 1
    rows = 1
    while rows * cols < num_sdfs:
        if cols / rows > 4 / 3:
            rows += 1
        else:
            cols += 1

    for c in range(num_sdfs):
        plt.subplot(rows, cols, c + 1)

    if show:
        plt.show()

    return fig


def save_checkpoint(
    path: str, model: torch.nn.Module, optimizer, iteration, epoch, run_name
):
    """Save a checkpoint during training.

    Args:

    """
    torch.save(
        {
            "iteration": iteration,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_name": run_name,
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    """Load a checkpoint during training.

    Args:
    """
    print(f"Loading checkpoint at {path} ...")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    epoch = checkpoint["epoch"]
    run_name = checkpoint["run_name"]

    print("Checkpoint loaded")

    model.train()  # training mode

    return model, optimizer, iteration, run_name, epoch


def str_to_tsdf(x: str) -> Union[bool, float]:
    """Convert string to expected values for tsdf setting.

    Args:
        x: A string containing either some representation of False or a float.
    Returns:
        False or float.
    """
    if x.lower() in ("no", "false", "f", "n", "0"):
        return False
    return float(x)


class View(torch.nn.Module):
    """Wrapper of torch's view method to use with nn.Sequential."""

    def __init__(self, shape):
        """Construct the module."""
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        """Reshape the tensor."""
        return x.view(*self.shape)
