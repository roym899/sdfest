"""Module providing dataset class for NOCS datasets (CAMERA / REAL)."""
# import torch


class NOCSDataset(torch.utils.data.Dataset):
    """Dataset class for NOCS dataset.
    CAMERA* and REAL* are training sets.
    CAMERA25 and REAL275 are test data.
    Some papers use CAMERA25 as validation when benchmarking REAL275.

    Datasets can be found here:
    https://github.com/hughw19/NOCS_CVPR2019/tree/master

    Expected folder format:
        {folder}/real_train/...
        {folder}/real_test/...
        {folder}/gts/...
        {folder}/obj_models/...
        {folder}/camera_composed_depth/...
        {folder}/camera_val25k/...
        {folder}/camera_train/...
    Which is easily obtained by downloading all the provided files and extracting them into
    the same directory.

    Necessary preprocessing of this data is performed during first initialization and is
    saved to
        {folder}/sdfest_pre/...
    """
    def __init__(
        self,
        folder: str,
        split: str,
    ):
        """Initialize the dataset."""
        pass

    def _preprocess_dataset(self):
        pass
