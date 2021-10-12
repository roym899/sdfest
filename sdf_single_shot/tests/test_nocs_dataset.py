"""Tests for nocs_dataset module."""
import os
import shutil

from sdf_single_shot.datasets.nocs_dataset import NOCSDataset

def test_nocsdataset_preprocessing(request, tmp_path):
    # create copy of NOCS test directory
    root_dir = request.fspath.dirname
    nocs_dataset_dir = os.path.join(root_dir, "nocs_data")
    shutil.copytree(nocs_dataset_dir, tmp_path, dirs_exist_ok=True)
    camera_train = NOCSDataset(root_dir=tmp_path, split="camera_train")
    camera_val = NOCSDataset(root_dir=tmp_path, split="camera_val")
    real_train = NOCSDataset(root_dir=tmp_path, split="real_train")
    real_test = NOCSDataset(root_dir=tmp_path, split="real_test")
