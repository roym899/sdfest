"""General functions for experiments and pytorch."""
import inspect
from pydoc import locate
import sys

import torch
from typing import Any


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
