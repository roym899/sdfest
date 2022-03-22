"""Tests for generated_dataset module."""
import os
import warnings

from pytest import FixtureRequest
import torch
import yoco

from sdfest.initialization.datasets.generated_dataset import (
    SDFVAEViewDataset,
)
from sdfest.vae.sdf_vae import SDFVAE


def _create_sdf_vae(config_path: str) -> SDFVAE:
    """Create SDF VAE from config path."""
    config = yoco.load_config_from_file(config_path)
    vae = SDFVAE(
        sdf_size=64,
        latent_size=config["latent_size"],
        encoder_dict=config["encoder"],
        decoder_dict=config["decoder"],
        device="cuda",
    ).to("cuda")
    state_dict = torch.load(config["model"], map_location="cuda")
    vae.load_state_dict(state_dict)
    return vae


def test_sdf_vae_view_dataset(request: FixtureRequest) -> None:
    """Test generated samples of SDF VAE View dataset."""
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Skipping SDFVAEViewDataset tests.")
        return

    sdfest.vae_config_path = os.path.join(request.fspath.dirname, "vae_model", "mug.yaml")
    sdfest.vae = _create_sdf_vae(sdf_vae_config_path)
    SDFVAEViewDataset(
        config={
            "extent_mean": 0.1,
            "extent_std": 0,
            "pointcloud": True,
            "normalize_pose": True,
            "z_min": 0.15,
            "z_max": 1.0,
        },
        vae=sdf_vae,
    )


def test_data_generation(request: FixtureRequest) -> None:
    """Test sample output from SDFVAEViewDataset for various settings."""
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Skipping SDFVAEViewDataset tests.")
        return

    sdfest.vae_config_path = os.path.join(request.fspath.dirname, "vae_model", "mug.yaml")
    sdfest.vae = _create_sdf_vae(sdf_vae_config_path)
    dataset = SDFVAEViewDataset(
        config={
            "extent_mean": 0.1,
            "extent_std": 0,
            "pointcloud": False,
            "normalize_pose": True,
            "z_min": 0.15,
            "z_max": 1.0,
        },
        vae=sdf_vae,
    )
    # depth
    sample = next(iter(dataset))
    assert sample["depth"].shape == (480, 640)

    # pointcloud
    dataset._pointcloud = True
    sample = next(iter(dataset))
    assert sample["pointset"].shape[1] == 3

    # with Gaussian blur and mask noise
    dataset._mask_noise = True
    dataset._gaussian_noise_probability = 1.0
    dataset._render_threshold = 0.005
    sample = next(iter(dataset))
