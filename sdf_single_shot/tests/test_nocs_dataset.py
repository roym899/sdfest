"""Tests for nocs_dataset module."""
import os
import shutil
from typing import Optional

import pandas as pd
from pytest import FixtureRequest
import torch

from sdf_single_shot.datasets.nocs_dataset import NOCSDataset
from sdf_single_shot import quaternion_utils, so3grid, utils


def create_datasets(
    root_dir: str, tmp_path: str, category_str: Optional[str] = None
) -> tuple:
    """Create NOCS dataset for all different splits."""
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
    camera_train, camera_val, real_train, real_test = create_datasets(
        request.fspath.dirname, tmp_path
    )

    # check correct number of files
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "camera_train"))) == 5
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "camera_val"))) == 3
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "real_train"))) == 6
    assert len(os.listdir(os.path.join(tmp_path, "sdfest_pre", "real_test"))) == 6

    assert len(camera_train) == 4
    assert len(camera_val) == 2
    assert len(real_train) == 5
    assert len(real_test) == 5


def test_nocsdataset_category_filtering(request: FixtureRequest, tmp_path: str) -> None:
    """Test category-based filtering of datasets."""
    camera_train, camera_val, real_train, real_test = create_datasets(
        request.fspath.dirname, tmp_path, category_str="mug"
    )

    assert len(camera_train) == 1
    assert len(camera_val) == 0
    assert len(real_train) == 1
    assert len(real_test) == 1


def test_nocsdataset_getitem(request: FixtureRequest, tmp_path: str) -> None:
    """Test getting different samples from NOCS dataset."""
    datasets = create_datasets(request.fspath.dirname, tmp_path)
    for dataset in datasets:
        sample = dataset[0]
        assert sample["color"].shape == (480, 640, 3)
        assert sample["depth"].shape == (480, 640)
        assert sample["mask"].shape == (480, 640)
        valid_depth_points = torch.sum(sample["depth"] != 0)
        assert sample["pointset"].shape == (valid_depth_points, 3)

        # test camera convention
        dataset._mask_pointcloud = True
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

        # test axis convention
        dataset._scale_convention = "full"  # to check permutation of extents
        dataset._remap_y_axis = "y"
        dataset._remap_x_axis = "x"
        scales = dataset[0]["scale"]
        orientation_q = dataset[0]["orientation"]
        dataset._remap_y_axis = "x"
        dataset._remap_x_axis = "-y"
        scales_2 = dataset[0]["scale"]
        orientation_q_2 = dataset[0]["orientation"]
        assert torch.allclose(scales[[1, 0, 2]], scales_2)
        # object point in first convention
        test_point = torch.tensor([0.1, 0.5, 0.7])
        # same object point in second convention
        test_point_2 = torch.tensor([0.5, -0.1, 0.7])
        # transform both points to camera
        cam_point = quaternion_utils.quaternion_apply(orientation_q, test_point)
        cam_point_2 = quaternion_utils.quaternion_apply(orientation_q_2, test_point_2)
        assert torch.allclose(cam_point, cam_point_2)

        # test orientation representation
        dataset._orientation_repr = "quaternion"
        orientation_q = dataset[0]["orientation"]
        assert orientation_q.shape == (4,)
        dataset._orientation_repr = "discretized"
        dataset._orientation_grid = so3grid.SO3Grid(3)
        orientation_d = dataset[0]["orientation"]
        assert orientation_d.shape == ()


def test_nocsdataset_gts_path(request: FixtureRequest, tmp_path: str) -> None:
    """Test generation of ground truth path from color path."""
    root_dir = request.fspath.dirname
    _, camera_val, _, real_test = create_datasets(root_dir, tmp_path)

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
    camera_train, camera_val, real_train, real_test = create_datasets(
        request.fspath.dirname, tmp_path
    )
    # TODO check that all datasets return correct data


def test_nocsdataset_get_obj_path(request: FixtureRequest, tmp_path: str) -> None:
    """Test getting pose and scale from NOCS dataset."""
    camera_train, camera_val, real_train, real_test = create_datasets(
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
