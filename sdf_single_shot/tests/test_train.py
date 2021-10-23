"""Test training script."""
import sys
import os

from sdf_single_shot.scripts import train

def test_single_iteration_train() -> None:
    """Test training for a single iteration."""
    # use default config with 1 iteration
    os.environ["WANDB_MODE"] = "disabled"
    sys.argv = ["", "--iterations", "1"]
    train.main()
