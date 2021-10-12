"""Module providing dataset class for NOCS datasets (CAMERA / REAL)."""
from typing import NamedTuple, Optional
import os

import numpy as np
import torch


class Sample(NamedTuple):
    """Represents a single data sample."""

    color: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    pointcloud: Optional[np.ndarray] = None
    masked_pointcloud: Optional[np.ndarray] = None


class NOCSDataset(torch.utils.data.Dataset):
    """Dataset class for NOCS dataset.

    CAMERA* and REAL* are training sets.
    CAMERA25 and REAL275 are test data.
    Some papers use CAMERA25 as validation when benchmarking REAL275.

    Datasets can be found here:
    https://github.com/hughw19/NOCS_CVPR2019/tree/master

    Expected directory format:
        {root_dir}/real_train/...
        {root_dir}/real_test/...
        {root_dir}/gts/...
        {root_dir}/obj_models/...
        {root_dir}/camera_composed_depth/...
        {root_dir}/camera_val25k/...
        {root_dir}/camera_train/...
    Which is easily obtained by downloading all the provided files and extracting them
    into the same directory.

    Necessary preprocessing of this data is performed during first initialization per
    and is saved to
        {root}/sdfest_pre/...
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir:
                Root dir of dataset. See class documentation for further information.
            split:
                The dataset split. The following strings are supported:
                "camera_train": 275000 images, synthetic foreground + real background
                "camera_val": 25000 images, synthetic foreground + real background
                "real_train": 4300 images in 7 scenes, real
                "real_test": 2750 images in 6 scenes, real
        """
        self._root_dir = root_dir
        self._split = split
        self._preprocess_path = os.path.join(root_dir, "sdfest_pre", split)
        if not os.path.isdir(self._preprocess_path):
            self._preprocess_dataset()

    def __len__(self) -> int:
        """Return number of sample in dataset."""
        # TODO compute length in preprocessing
        return self._length

    def __getitem__(self, idx: int) -> Sample:
        """Return a sample of the dataset.

        Args:
            idx: Index of the instance.
        Returns:
            inputs: Inputs to the network, as specified in the constructor.
            targets: Targets for the given input, as specified in the constructor.
        """
        # TODO load sample, and convert to desired input format
        inputs = None
        # TODO load target, and convert to desired target format
        targets = None
        return Sample()

    def _preprocess_dataset(self):
        """Create preprocessing files for the current split.

        Preprocessing will be stored on disk to {root_dir}/sdfest_pre/...
        This function will not store the preprocessing, so it still has to be loaded
        afterwards.
        """
        # TODO create common dictionary + format + files for each split
        # TODO anns in memory or on disk?
        pass
