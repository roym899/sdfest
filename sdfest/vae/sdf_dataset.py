"""Module which provides SDFDataset class."""
import os
import torch
import numpy as np


class SDFDataset(torch.utils.data.Dataset):
    """Dataset of SDF volumes stored in .npy format.

    Expected dataset format:
        {sdf_folder}/00000.npy
        {sdf_folder}/00001.npy
        ...
    """

    def __init__(self, sdf_folder: str):
        """Construct the dataset.

        Args:
            sdf_folder: The folder containing the npy files.
        """
        self.path = sdf_folder

        self.size = len([f for f in os.listdir(sdf_folder) if f.endswith(".npy")])

    def __len__(self):
        """Return the number of images in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return SDF volume at a specific index.

        Args:
            idx: The index of the sdf file to retrieve.
        Returns:
            The loaded SDF volume.
        """
        sdf_path = os.path.join(self.path, f"{idx:05}.npy")
        sdf_np = np.load(sdf_path)
        return torch.as_tensor(sdf_np).unsqueeze(0)
