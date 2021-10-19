"""Tests for nocs_dataset module."""
import os
import shutil
from typing import Optional

import pandas as pd
from pytest import FixtureRequest
import torch

from sdf_single_shot.datasets.nocs_dataset import NOCSDataset


def _create_datasets(
    root_dir: str, tmp_path: str, category_str: Optional[str] = None
) -> tuple:
    nocs_dataset_dir = os.path.join(root_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    camera_train = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "camera_train",
            "category_str": category_str,
        }
    )
    camera_val = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "camera_val",
            "category_str": category_str,
        }
    )
    real_train = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "real_train",
            "category_str": category_str,
        }
    )
    real_test = NOCSDataset(
        {
            "root_dir": tmp_path,
            "split": "real_test",
            "category_str": category_str,
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


def test_nocsdataset_category_filtering(request: FixtureRequest, tmp_path: str) -> None:
    """Test category-based filtering of datasets."""
    camera_train, camera_val, real_train, real_test = _create_datasets(
        request.fspath.dirname, tmp_path, category_str="mug"
    )

    assert len(camera_train) == 1
    assert len(camera_val) == 0
    assert len(real_train) == 1
    assert len(real_test) == 1


def test_nocsdataset_getitem(request: FixtureRequest, tmp_path: str) -> None:
    """Test getting different samples from NOCS dataset."""
    datasets = _create_datasets(request.fspath.dirname, tmp_path)
    for dataset in datasets:
        sample = dataset[0]
        assert sample["color"].shape == (480, 640, 3)
        assert sample["depth"].shape == (480, 640)
        assert sample["mask"].shape == (480, 640)
        valid_depth_points = torch.sum(sample["depth"] != 0)
        assert sample["pointcloud"].shape == (valid_depth_points, 3)

        # test camera convention
        dataset._camera_convention = "opencv"
        sample_cv = dataset[0]
        dataset._camera_convention = "opengl"
        sample_gl = dataset[0]
        assert sample_cv["position"][2] > 0
        assert sample_gl["position"][2] < 0

        # test scale conventions
        dataset._scale_convention = "full"
        full_scale = dataset[0]["scale"]
        dataset._scale_convention = "max"
        max_scale = dataset[0]["scale"]
        dataset._scale_convention = "half_max"
        half_max_scale = dataset[0]["scale"]
        dataset._scale_convention = "diagonal"
        diagonal_scale = dataset[0]["scale"]
        assert full_scale.shape == (3,)
        assert max_scale == torch.max(full_scale)
        assert half_max_scale == 0.5 * max_scale
        assert diagonal_scale == torch.linalg.norm(full_scale)


def test_nocsdataset_gts_path(request: FixtureRequest, tmp_path: str) -> None:
    """Test generation of ground truth path from color path."""
    root_dir = request.fspath.dirname
    _, camera_val, _, real_test = _create_datasets(root_dir, tmp_path)

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
