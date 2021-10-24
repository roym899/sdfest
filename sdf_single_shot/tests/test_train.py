"""Test training script."""
import sys
import os

from pytest import FixtureRequest

from sdf_single_shot.scripts import train
import test_nocs_dataset


def test_single_iteration_train_sdfvae() -> None:
    """Test training for a single iteration."""
    # use default config with 1 iteration
    os.environ["WANDB_MODE"] = "disabled"
    sys.argv = ["", "--iterations", "1"]
    train.main()

# def test_single_iteration_train_nocs(request: FixtureRequest, tmp_path: str) -> None:
#     """Test training for NOCS datasets."""
#     # use default config with 1 iteration
#     os.environ["WANDB_MODE"] = "disabled"
#     sys.argv = ["", "--iterations", "1"]
#     train.main()
