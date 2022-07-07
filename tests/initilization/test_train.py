"""Test training script."""
import shutil
import sys
import os

from pytest import FixtureRequest

from sdfest.initialization.scripts import train


def test_generatedview_training(request: FixtureRequest, tmp_path: str) -> None:
    """Test training with views generated from VAE."""
    tests_dir = request.fspath.dirname
    nocs_dataset_dir = os.path.join(tests_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    # use default config with 1 iteration
    os.environ["WANDB_MODE"] = "disabled"
    os.chdir(tmp_path)
    sys.argv = [
        "",
        "--config",
        os.path.join(tests_dir, "test_configs", "discretized.yaml"),
        "--datasets.camera_train.config_dict.root_dir",
        str(tmp_path),
        "--datasets.real_train.config_dict.root_dir",
        str(tmp_path),
        "--validation_datasets.camera_val.config_dict.root_dir",
        str(tmp_path),
    ]
    train.main()


def test_nocs_training(request: FixtureRequest, tmp_path: str) -> None:
    """Test training for NOCS datasets."""
    tests_dir = request.fspath.dirname
    nocs_dataset_dir = os.path.join(tests_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    # use default config with 1 iteration
    os.environ["WANDB_MODE"] = "disabled"
    os.chdir(tmp_path)
    sys.argv = [
        "",
        "--config",
        os.path.join(tests_dir, "test_configs", "discretized_nocs.yaml"),
        "--datasets.camera_train.config_dict.root_dir",
        str(tmp_path),
        "--datasets.real_train.config_dict.root_dir",
        str(tmp_path),
        "--validation_datasets.camera_val.config_dict.root_dir",
        str(tmp_path),
    ]
    train.main()
