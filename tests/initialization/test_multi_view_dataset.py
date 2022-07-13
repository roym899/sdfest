"""Tests for multi_view_dataset module."""
import os

from sdfest.initialization.datasets import multi_view_dataset


def test_multi_view_dataset() -> None:
    """Test MultiViewDataset class."""
    mvd = multi_view_dataset.MultiViewDataset(
        {
            "root_dir": os.path.join(os.path.dirname(__file__), "multi_view_data"),
            "split": "train",
        }
    )
    assert len(mvd) == 2
