"""Tests for nocs_dataset module."""
import os
import shutil

import pandas as pd
from pytest import FixtureRequest
import torch

from sdf_single_shot.datasets.nocs_dataset import NOCSDataset


def _create_datasets(root_dir: str, tmp_path: str) -> tuple:
    nocs_dataset_dir = os.path.join(root_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    camera_train = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "camera_train",
        }
    )
    camera_val = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "camera_val",
        }
    )
    real_train = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "real_train",
        }
    )
    real_test = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "real_test",
        }
    )
    return camera_train, camera_val, real_train, real_test


def test_nocsdataset_preprocessing(request: FixtureRequest, tmp_path: str) -> None:
    """Test preprocessing of different NOCS dataset splits."""
    camera_train, camera_val, real_train, real_test = _create_datasets(
        request.fspath.dirname, tmp_path
    )

    # check correct number of files
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "camera_train"))) == 4
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "camera_val"))) == 2
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "real_train"))) == 5
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "real_test"))) == 5

    assert len(camera_train) == 4
    assert len(camera_val) == 2
    assert len(real_train) == 5
    assert len(real_test) == 5


def test_nocsdataset_getitem(request: FixtureRequest, tmp_path: str) -> None:
    """Test getting different samples from NOCS dataset."""
    camera_train, camera_val, real_train, real_test = _create_datasets(
        request.fspath.dirname, tmp_path
    )
    sample = real_train[0]
    assert sample["color"].shape == (480, 640, 3)
    assert sample["depth"].shape == (480, 640)
    assert sample["mask"].shape == (480, 640)
    valid_depth_points = torch.sum(sample["depth"] != 0)
    assert sample["pointcloud"].shape == (valid_depth_points, 3)

    assert camera_train[0]["depth"].shape == (480, 640)
    assert camera_val[0]["depth"].shape == (480, 640)
    assert real_test[0]["depth"].shape == (480, 640)


def test_nocsdataset_gts_path(request: FixtureRequest, tmp_path: str) -> None:
    """Test generation of ground truth path from color path."""
    root_dir = request.fspath.dirname
    _, camera_val, _, real_test = _create_datasets(
        root_dir, tmp_path
    )

    gts_path = real_test._get_gts_path(
        os.path.join(root_dir, "real_test", "scene_1", "0000_color.png")
    )
    assert os.path.isfile(gts_path)

    gts_path = camera_val._get_gts_path(
        os.path.join(root_dir, "val", "00000", "0000_color.png")
    )
    assert os.path.isfile(gts_path)


def test_nocsdataset_get_pose_and_scale(request: FixtureRequest, tmp_path: str) -> None:
    """Test getting pose and scale from NOCS dataset."""
    camera_train, camera_val, real_train, real_test = _create_datasets(
        request.fspath.dirname, tmp_path
    )
    # TODO check that all datasets return correct data


def test_nocsdataset_get_obj_path(request: FixtureRequest, tmp_path: str) -> None:
    """Test getting pose and scale from NOCS dataset."""
    camera_train, camera_val, real_train, real_test = _create_datasets(
        request.fspath.dirname, tmp_path
    )
    # check that all datasets return correct data
    assert os.path.isfile(
        camera_train._get_obj_path(
            pd.Series([0, 0, "02876657", "ab6792cddc7c4c83afbf338b16b43f53"])
        )
    )
    assert os.path.isfile(
        camera_val._get_obj_path(
            pd.Series([0, 0, "03642806", "fdec2b8af5dd988cef56c22fd326c67"])
        )
    )
    assert os.path.isfile(
        real_train._get_obj_path(pd.Series([0, 0, "mug2_scene3_norm"]))
    )
    assert os.path.isfile(
        real_test._get_obj_path(pd.Series([0, 0, "bowl_white_small_norm"]))
    )
