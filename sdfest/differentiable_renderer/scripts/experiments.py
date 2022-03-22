"""Experiments with differentiable SDF renderer.

Usage:
    python -m sdfest.differentiable_renderer.scripts.experiments --sdf_path data/shapenet_processed/mug_filtered/00001.npy --scale 0.15  \
    --scale_off --pos 0 0 -0.4 --pos_off 0.03 0.03 0.03 --rot 400 90 0 --rot_off 10 10 10 --gpu --visualize 10 --steps 1000
"""
import torch
import time
import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from sdfest.differentiable_renderer.sdf_renderer import render_depth, render_depth_gpu


@click.command()
@click.option("--sdf_path", type=click.Path(exists=True), required=True)
@click.option("--ref_sdf_path", type=click.Path(exists=True), default=None)
@click.option("--pos", type=float, nargs=3, default=None)
@click.option("--rot", type=float, nargs=3, default=None)
@click.option("--scale", type=float, default=None)
@click.option("--pos_off", type=float, nargs=3, default=None)
@click.option("--rot_off", type=float, nargs=3, default=None)
@click.option("--scale_off", type=float, default=None)
@click.option("--steps", type=int, default=100)
@click.option("--width", type=int, default=640)
@click.option("--height", type=int, default=480)
@click.option("--fov", type=float, default=90)
@click.option("--gpu", is_flag=True)
@click.option("--visualize", type=int, default=None)
@click.option("--threshold", type=float, default=0.01)
def offset_experiment(
    sdf_path,
    ref_sdf_path,
    pos,
    rot,
    scale,
    pos_off,
    rot_off,
    scale_off,
    steps,
    width,
    height,
    fov,
    gpu,
    visualize,
    threshold,
):
    # set device
    if gpu and not torch.cuda.is_available():
        gpu = False
    device = torch.device("cuda" if gpu else "cpu")

    if pos is None or not pos:
        pos = [0.0, 0.0, 0.0]
    if rot is None or not rot:
        rot = [0.0, 0.0, 0.0]
    if pos_off is None or not pos_off:
        pos_off = [0.0, 0.0, 0.0]
    if rot_off is None or not rot_off:
        rot_off = [0.0, 0.0, 0.0]
    if ref_sdf_path is None:
        ref_sdf_path = sdf_path

    ref_pos, ref_rot = torch.tensor(pos, device=device), torch.tensor(
        rot, device=device
    )
    ref_scale_inv = torch.tensor(1.0 / scale, device=device)
    pos_off, rot_off = torch.tensor(pos_off, device=device), torch.tensor(
        rot_off, device=device
    )
    pos = ref_pos + pos_off
    rot = ref_rot + rot_off
    scale_inv = torch.tensor(1.0 / (scale + scale_off), device=device)
    sdf = torch.from_numpy(np.load(sdf_path)).float().to(device)
    ref_sdf = torch.from_numpy(np.load(ref_sdf_path)).float().to(device)
    quat = (
        torch.from_numpy(Rotation.from_euler("XYZ", rot.cpu(), True).as_quat())
        .float()
        .to(device)
    )
    ref_quat = (
        torch.from_numpy(Rotation.from_euler("XYZ", ref_rot.cpu(), True).as_quat())
        .float()
        .to(device)
    )

    # Enable gradients for all optimizable variables
    quat.requires_grad_()
    pos.requires_grad_()
    sdf.requires_grad_()
    scale_inv.requires_grad_()

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plt.ion()
    plt.show()

    # Generate reference image
    with torch.no_grad():
        if gpu:
            t1 = time.time()
            ref = render_depth_gpu(
                ref_sdf, ref_pos, ref_quat, ref_scale_inv, width, height, fov, threshold
            )
            # ref[:,0:250] = 0
            # ref[:,200:] = 0
        else:
            t1 = time.time()
            ref = render_depth(
                ref_sdf, ref_pos, ref_quat, ref_scale_inv, width, height, fov, threshold
            )

    optimizer = torch.optim.Adam([pos, quat, scale_inv], lr=0.001)

    # run the optimization
    step = 0
    total_time = 0
    while step < steps:
        t1 = time.time()
        if gpu:
            img = render_depth_gpu(
                sdf, pos, quat, scale_inv, width, height, fov, threshold
            )
        else:
            img = render_depth(sdf, pos, quat, scale_inv, width, height, fov, threshold)
        mask = (img > 0) & (ref > 0)
        error = (img - ref) ** 2
        loss = torch.mean(torch.abs((img - ref)[mask]))
        loss.backward()
        with torch.no_grad():
            step += 1
            optimizer.step()
            optimizer.zero_grad()
            quat /= torch.sqrt(torch.sum(quat ** 2))  # normalize quaternion
            total_time += time.time() - t1
            print(f"loss: {loss} step: {step} s/step {total_time / step}")

            if visualize is not None and step % visualize == 0:
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                ax1.imshow(img.detach().cpu().numpy(), cmap="gist_heat_r")
                ax2.imshow(img.detach().cpu().numpy(), cmap="gist_heat_r", alpha=0.5)
                ax2.imshow(ref.detach().cpu().numpy(), cmap="gist_heat_r", alpha=0.5)
                vmax = torch.max(error[mask]).detach().cpu().numpy()
                ax3.imshow(
                    error.detach().cpu().numpy(), cmap="inferno", vmin=0, vmax=vmax
                )
                ax4.imshow(ref.detach().cpu().numpy(), cmap="gist_heat_r")
                plt.pause(0.00001)


if __name__ == "__main__":
    offset_experiment()
