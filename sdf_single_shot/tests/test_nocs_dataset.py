"""Tests for nocs_dataset module."""
import os
import shutil

import matplotlib.pyplot as plt
from pytest import FixtureRequest
import torch

from sdf_single_shot.datasets.nocs_dataset import NOCSDataset


def test_nocsdataset_preprocessing(request: FixtureRequest, tmp_path: str) -> None:
    """Test preprocessing of different NOCS dataset splits."""
    # create copy of NOCS test directory
    root_dir = request.fspath.dirname
    nocs_dataset_dir = os.path.join(root_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    camera_train = NOCSDataset(
        root_dir=tmp_path,
        split="camera_train",
    )
    camera_val = NOCSDataset(
        root_dir=tmp_path,
        split="camera_val",
    )
    real_train = NOCSDataset(
        root_dir=tmp_path,
        split="real_train",
    )
    real_test = NOCSDataset(
        root_dir=tmp_path,
        split="real_test",
    )

    # check correct number of files
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "camera_train"))) == 19
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "camera_val"))) == 23
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "real_train"))) == 5
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "real_test"))) == 5

    assert len(camera_train) == 19
    assert len(camera_val) == 23
    assert len(real_train) == 5
    assert len(real_test) == 5


def test_nocsdataset_getitem(request: FixtureRequest, tmp_path: str) -> None:
    """Test getting different samples from NOCS dataset."""
    root_dir = request.fspath.dirname
    nocs_dataset_dir = os.path.join(root_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    real_train = NOCSDataset(
        root_dir=tmp_path,
        split="real_train",
    )
    sample = real_train[0]
    assert sample["color"].shape == (480, 640, 3)
    assert sample["depth"].shape == (480, 640)
    assert sample["mask"].shape == (480, 640)
    valid_depth_points = torch.sum(sample["depth"] != 0)
    assert sample["pointcloud"].shape == (valid_depth_points, 3)

    camera_train = NOCSDataset(
        root_dir=tmp_path,
        split="camera_train",
    )
    camera_val = NOCSDataset(
        root_dir=tmp_path,
        split="camera_val",
    )
    assert camera_train[0]["depth"].shape == (480, 640)
    assert camera_val[0]["depth"].shape == (480, 640)
