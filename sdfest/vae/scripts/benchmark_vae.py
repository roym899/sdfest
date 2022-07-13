"""Script to benchmark VAE architecture in isolation.

Usage:
    python -m sdfest.vae.scripts.benchmark_vae \
           --config initialization/configs/vae_models/mug.yaml \
           --device cuda
"""
import argparse
import time

import torch
import yoco

import sdfest
from sdfest.vae.sdf_vae import SDFVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for SDFVAE.")
    config = yoco.load_config_from_args(
        parser, search_paths=[".", "~/.sdfest/", sdfest.__path__[0]]
    )
    sdf_vae = SDFVAE(
        sdf_size=64,
        latent_size=config["latent_size"],
        encoder_dict=config["encoder"],
        decoder_dict=config["decoder"],
        device=config["device"],
        tsdf=config["tsdf"],
    ).to(config["device"])
    sdf_vae.eval()

    # Forward only
    torch.set_grad_enabled(False)
    for _ in range(1000):
        torch.cuda.synchronize()
        start = time.time()

        test = sdf_vae.inference(n=1)

        torch.cuda.synchronize()
        end = time.time()
        print("forward only:", end - start)

    # Forward + Backward
    torch.set_grad_enabled(True)
    latent = torch.zeros(
        config["latent_size"], device=config["device"], requires_grad=True
    ).unsqueeze(0)
    for _ in range(1000):
        torch.cuda.synchronize()
        start = time.time()

        test = sdf_vae.decode(latent)
        loss = torch.sum(test)
        loss.backward()

        torch.cuda.synchronize()
        end = time.time()
        print("forward + backward:", end - start)
