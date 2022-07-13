"""Common utility functions."""
import os
from typing import Optional

from cpas_toolbox.utils import download
import torch
import yoco


def load_model_weights(
    path: str,
    model: torch.nn.Module,
    map_location: torch.device,
    model_weights_url: Optional[str] = None,
) -> None:
    """Load model weights from path or download weights from URL if file does not exist.

    Args:
        path: Path to model weights.
        model: Path to model weights.
        map_location: See torch.load.
        model_weights_url: URL to download model weights from if path does not exist.
    """
    resolved_path = yoco.resolve_path(
        path, search_paths=[".", "~/.sdfest/model_weights/"]
    )

    if not os.path.exists(resolved_path):
        if model_weights_url is not None:
            if not os.path.isabs(resolved_path):
                resolved_path = os.path.expanduser(
                    os.path.join("~/.sdfest/model_weights", resolved_path)
                )
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            print(f"Model weights {resolved_path} not found.")
            print(f"Downloading from {model_weights_url}")
            download(model_weights_url, resolved_path)
        else:
            print(f"Model weights {resolved_path} not found. Aborting.")
            exit(0)

    state_dict = torch.load(resolved_path, map_location=map_location)
    model.load_state_dict(state_dict)
