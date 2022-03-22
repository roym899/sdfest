"""Test training script."""
import shutil
import sys
import os

from pytest import FixtureRequest

from sdfest.initialization.scripts import train


def test_single_iteration_train_sdfvae(request: FixtureRequest, tmp_path: str) -> None:
    """Test training for a single iteration."""
    # use default config (generated_dataset) with 1 iteration
    tests_dir = request.fspath.dirname
    os.environ["WANDB_MODE"] = "disabled"
    os.chdir(tmp_path)
    sys.argv = [
        "",
        "--iterations",
        "1",
        "--config",
        os.path.join(tests_dir, "../configs/", "default.yaml"),
    ]
    train.main()


def test_multi_dataset_training(request: FixtureRequest, tmp_path: str) -> None:
    """Test training for NOCS datasets."""
    tests_dir = request.fspath.dirname
    nocs_dataset_dir = os.path.join(tests_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    # use default config with 1 iteration
    os.environ["WANDB_MODE"] = "disabled"
    os.chdir(tmp_path)
    sys.argv = [
        "",
        "--iterations",
        "10",
        "--config",
        os.path.join(tests_dir, "test_configs", "discretized.yaml"),
        "--datasets.camera_train.config_dict.root_dir",
        str(tmp_path),
        "--datasets.real_train.config_dict.root_dir",
        str(tmp_path),
        "--visualization_iteration",
        "1",
        "--validation_iteration",
        "1",
        "--checkpoint_iteration",
        "1",
    ]
    train.main()
