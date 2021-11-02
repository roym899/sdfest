"""Utility functions to handle various datasets."""
import random
from typing import Iterator, List

import numpy as np
import torch


def collate_samples(samples: List[dict]) -> dict:
    """Collate sample dictionaries.

    Performs standard batching and additionally batches pointsets by taking subset of
    points.
    Also supports non-tensor types, which will be returned as standard lists.

    Reduces all pointsets to a common size based on the smallest set.

    Args:
        samples:
            Dictionary containing various types of data.
            All keys except "pointset" will use standard batching.
            All samples are expected to contain the same keys.
    Returns:
        Dictionary containing same keys as each sample.
        For "pointset" key:
            Tensor of size (N, M_min, D) where N is the batch size, M_min the number of
            points in the smallest pointset and D the number of channels per point.
    """
    batch = {}

    for key in samples[0].keys():
        if key == "pointset":
            batch_size = len(samples)

            smallest_set = min(s["pointset"].shape[0] for s in samples)
            # limit number of points to limit memory usage
            smallest_set = min(smallest_set, 2500)

            sample_pointset = samples[0]["pointset"]

            channels = sample_pointset.shape[-1]
            device = sample_pointset.device
            batch["pointset"] = torch.empty(
                batch_size, smallest_set, channels, device=device
            )
            for i, sample in enumerate(samples):
                num_points = sample["pointset"].shape[0]
                point_indices = random.sample(range(0, num_points), smallest_set)
                batch["pointset"][i] = sample["pointset"][point_indices]
        elif isinstance(samples[0][key], torch.Tensor):
            # standard batching for torch tensors
            batch[key] = torch.stack([s[key] for s in samples])
        else:
            # standard list for other data types
            batch[key] = [s[key] for s in samples]

    return batch


class MultiDataLoader:
    """Wrapper for multiple dataloaders."""

    def __init__(
        self,
        data_loaders: List[torch.utils.data.DataLoader],
        probabilities: List[float],
    ) -> None:
        """Initialize the class."""
        self._data_loaders = data_loaders
        self._data_loader_iterators = [iter(dl) for dl in self._data_loaders]
        self._probabilities = probabilities
        assert len(self._data_loaders) == len(self._probabilities)

    def __iter__(self) -> Iterator:
        """Return infinite iterator which returns samples from sampled data_loader."""
        while True:
            i = np.random.choice(
                np.arange(len(self._probabilities)), p=self._probabilities
            )
            try:
                yield next(self._data_loader_iterators[i])
            except StopIteration:
                self._data_loader_iterators[i] = iter(self._data_loaders[i])
