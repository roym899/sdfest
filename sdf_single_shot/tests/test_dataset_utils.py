"""Test function from dataset_utils module."""

from test_nocs_dataset import create_datasets
from sdf_single_shot.datasets import dataset_utils

from pytest import FixtureRequest
import torch


def test_collate_samples(request: FixtureRequest, tmp_path: str) -> None:
    """Test preprocessing of different NOCS dataset splits."""
    datasets = create_datasets(
        request.fspath.dirname, tmp_path
    )
    for dataset in datasets:
        samples = [dataset[0], dataset[1]]
        batch = dataset_utils.collate_samples(samples)
        assert batch["pointset"].shape[0] == 2
        assert batch["depth"].shape[0] == 2
