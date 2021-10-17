"""General functions for experiments and pytorch."""
import torch


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
