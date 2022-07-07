"""Test function from dataset_utils module."""
from typing import Iterator

from test_nocs_dataset import create_datasets
from sdfest.initialization.datasets import dataset_utils

from pytest import FixtureRequest
import torch


class _ConstantValueDataset(torch.utils.data.IterableDataset):
    def __init__(self, value: float) -> None:
        self._value = value

    def __iter__(self) -> Iterator:
        """Return infinite iterator that returns a constant value."""
        while True:
            yield self._value


def test_collate_samples(request: FixtureRequest, tmp_path: str) -> None:
    """Test preprocessing of different NOCS dataset splits."""
    datasets = create_datasets(request.fspath.dirname, tmp_path)
    for dataset in datasets:
        samples = [dataset[0], dataset[1]]
        batch = dataset_utils.collate_samples(samples)
        assert batch["pointset"].shape[0] == 2
        assert batch["depth"].shape[0] == 2


def test_multi_data_loader(request: FixtureRequest, tmp_path: str) -> None:
    """Test multi data loader class."""
    datasets = create_datasets(
        request.fspath.dirname,
        tmp_path,
    )
    dls = [torch.utils.data.DataLoader(dataset) for dataset in datasets]
    mdl = dataset_utils.MultiDataLoader(
        data_loaders=dls, probabilities=[0.25, 0.25, 0.25, 0.25]
    )
    count = 0
    for _ in mdl:
        count += 1
        if count > 10:
            break

    # test with mock datasets
    mdl = dataset_utils.MultiDataLoader(
        data_loaders=[
            _ConstantValueDataset(0),
            _ConstantValueDataset(1),
            _ConstantValueDataset(2),
            _ConstantValueDataset(3),
        ],
        probabilities=[0.5, 0.25, 0.25, 0],
    )
    count = 0
    item_count = [0, 0, 0, 0]
    for sample in mdl:
        count += 1
        item_count[sample] += 1
        if count > 1000:
            break
    assert 400 <= item_count[0] <= 600
    assert 150 <= item_count[1] <= 350
    assert 150 <= item_count[2] <= 350
    assert item_count[3] == 0
