"""Script to train model."""
import argparse
from datetime import datetime
import os
import time

import torch
import torch.utils.tensorboard
import wandb
import yoco

from torchinfo import summary

import sdfest

from sdfest.vae import utils
from sdfest.vae import sdf_utils
from sdfest.vae.sdf_vae import SDFVAE
from sdfest.vae.sdf_dataset import SDFDataset

from sdfest.differentiable_renderer import Camera, render_depth_gpu

from sdfest.initialization import pointset_utils

if "DISPLAY" not in os.environ or "localhost" in os.environ["DISPLAY"]:
    print("Using EGL instead of pyglet for OffscreenRendering")
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def pc_loss(points, position, orientation, scale, sdf):
    """Compute trilinerly interpolated SDF value at the points positions.

    Args:
        points:
            Mx4 pointcloud in camera frame.
        position:
            Position of SDF center in camera frame.
        orientation:
            Quaternion representing orientation of SDF.
        scale:
            Half-width of SDF volume.
        sdf:
            Volumetric signed distance field.
    Returns:
        Trilinearly interpolated distance at the passed points
        0 if outside of SDF volume.
    """
    q = orientation / torch.norm(orientation)  # to get normalization gradients
    obj_points = points - position.unsqueeze(0)

    # Quaternion to rotation matrix
    # Note that we use conjugate here since we want to transform points from
    # world to object coordinates and the quaternion describes rotation of
    # object coordinate system in world coordinates.
    R = obj_points.new_zeros(3, 3)

    R[0, 0] = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    R[0, 1] = 2 * (q[0] * q[1] + q[2] * q[3])
    R[0, 2] = 2 * (q[0] * q[2] - q[3] * q[1])

    R[1, 0] = 2 * (q[0] * q[1] - q[2] * q[3])
    R[1, 1] = 1 - 2 * (q[0] * q[0] + q[2] * q[2])
    R[1, 2] = 2 * (q[1] * q[2] + q[3] * q[0])

    R[2, 0] = 2 * (q[0] * q[2] + q[3] * q[1])
    R[2, 1] = 2 * (q[1] * q[2] - q[3] * q[0])
    R[2, 2] = 1 - 2 * (q[0] * q[0] + q[1] * q[1])

    obj_points = (R @ obj_points.T).T

    # Transform to canonical coordintes obj_point in [-1,1]^3
    obj_point = obj_points / scale

    # Compute cell and cell position
    res = sdf.shape[0]  # assuming same resolution along each axis
    grid_size = 2.0 / (res - 1)
    c = torch.floor((obj_point + 1.0) * (res - 1) * 0.5)
    mask = torch.logical_or(
        torch.min(c, dim=1)[0] < 0, torch.max(c, dim=1)[0] > res - 2
    )
    c = torch.clip(c, 0, res - 2)  # base cell index of each point
    cell_position = c * grid_size - 1.0  # base cell position of each point
    sdf_indices = c.new_empty((obj_point.shape[0], 8), dtype=torch.long)
    sdf_indices[:, 0] = c[:, 0] * res**2 + c[:, 1] * res + c[:, 2]
    sdf_indices[:, 1] = c[:, 0] * res**2 + c[:, 1] * res + c[:, 2] + 1
    sdf_indices[:, 2] = c[:, 0] * res**2 + (c[:, 1] + 1) * res + c[:, 2]
    sdf_indices[:, 3] = c[:, 0] * res**2 + (c[:, 1] + 1) * res + c[:, 2] + 1
    sdf_indices[:, 4] = (c[:, 0] + 1) * res**2 + c[:, 1] * res + c[:, 2]
    sdf_indices[:, 5] = (c[:, 0] + 1) * res**2 + c[:, 1] * res + c[:, 2] + 1
    sdf_indices[:, 6] = (c[:, 0] + 1) * res**2 + (c[:, 1] + 1) * res + c[:, 2]
    sdf_indices[:, 7] = (c[:, 0] + 1) * res**2 + (c[:, 1] + 1) * res + c[:, 2] + 1
    sdf_view = sdf.view([-1])
    point_cell_position = (obj_point - cell_position) / grid_size  # [0,1]^3
    sdf_values = torch.take(sdf_view, sdf_indices)

    # trilinear interpolation
    sdf_value = sdf_values.new_empty(obj_points.shape[0])
    # sdf_value = obj_point[:, 0]
    sdf_value = (
        (
            sdf_values[:, 0] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 4] * point_cell_position[:, 0]
        )
        * (1 - point_cell_position[:, 1])
        + (
            sdf_values[:, 2] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 6] * point_cell_position[:, 0]
        )
        * point_cell_position[:, 1]
    ) * (1 - point_cell_position[:, 2]) + (
        (
            sdf_values[:, 1] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 5] * point_cell_position[:, 0]
        )
        * (1 - point_cell_position[:, 1])
        + (
            sdf_values[:, 3] * (1 - point_cell_position[:, 0])
            + sdf_values[:, 7] * point_cell_position[:, 0]
        )
        * point_cell_position[:, 1]
    ) * point_cell_position[
        :, 2
    ]
    sdf_value[mask] = 0
    return sdf_value


def train(config):
    """Train the model.

    Args:
        config: The training and model config.
    """
    batch_size = wandb.config.batch_size = config["batch_size"]
    iterations = wandb.config.iterations = config["iterations"]
    learning_rate = wandb.config.learning_rate = config["learning_rate"]
    latent_size = wandb.config.latent_size = config["latent_size"]
    l2_large_weight = wandb.config.l2_large_weight = config["l2_large_weight"]
    l2_small_weight = wandb.config.l2_small_weight = config["l2_small_weight"]
    l1_large_weight = wandb.config.l1_large_weight = config["l1_large_weight"]
    l1_small_weight = wandb.config.l1_small_weight = config["l1_small_weight"]
    kld_weight = wandb.config.kld_weight = config["kld_weight"]
    pc_weight = wandb.config.pc_weight = config["pc_weight"]
    encoder_dict = wandb.config.encoder_dict = config["encoder"]
    decoder_dict = wandb.config.decoder_dict = config["decoder"]
    dataset_path = wandb.config.dataset_path = config["dataset_path"]
    tsdf = wandb.config.tsdf = config["tsdf"]

    # init dataset, dataloader, model and optimizer
    dataset = SDFDataset(dataset_path)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    camera = Camera(640, 480, 320, 320, 320, 240, pixel_center=0.5)

    if "device" not in config or config["device"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])

    sdfvae = SDFVAE(
        sdf_size=64,
        latent_size=latent_size,
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
        device=device,
        tsdf=tsdf,
    ).to(device)

    print(str(sdfvae))

    summary(sdfvae, (1, 1, 64, 64, 64), device=device)

    optimizer = torch.optim.Adam(sdfvae.parameters(), lr=learning_rate)

    # load checkpoint if provided
    if "checkpoint" in config and config["checkpoint"] is not None:
        # TODO: checkpoint should always go together with model config!
        model, optimizer, current_iteration, run_name, epoch = utils.load_checkpoint(
            config["checkpoint"], sdfvae, optimizer
        )
    else:
        current_iteration = 0
        current_epoch = 0
        run_name = f"sdfvae_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"

    # create SummaryWriter for logging and intermediate output
    writer = torch.utils.tensorboard.SummaryWriter(f"runs/" + run_name)

    model_base_path = os.path.join(os.getcwd(), "models", run_name)
    program_starts = time.time()
    warm_up_iterations = 1000
    stop = False
    while current_iteration <= iterations:
        current_epoch += 1
        for sdf_volumes in data_loader:
            sdf_volumes = sdf_volumes.to(device)

            # train first N iters with SDF instead of TSDF to stabilize early training
            if current_iteration > warm_up_iterations:
                sdfvae.prepare_input(sdf_volumes)

            recon_sdf_volumes, mean, log_var, z = sdfvae(
                sdf_volumes, enforce_tsdf=False
            )

            if tsdf is not False and current_iteration > warm_up_iterations:
                # during training, clamp the reconstructed SDF only where both target
                # and output are outside the TSDF range
                mask = torch.logical_and(
                    torch.abs(sdf_volumes) >= tsdf, torch.abs(recon_sdf_volumes) >= tsdf
                )
                recon_sdf_volumes_temp = recon_sdf_volumes
                recon_sdf_volumes = recon_sdf_volumes_temp.clone()
                recon_sdf_volumes[mask] = recon_sdf_volumes_temp[mask].clamp(
                    -tsdf, tsdf
                )

            # Compute losses, use negative log-likelihood here
            # Note: for average negative log-likelihood all of these have to be divided
            # by the batch size. Probably would be better to keep losses comparable for
            # varying batch size.
            l1_error = torch.abs(recon_sdf_volumes - sdf_volumes)
            l2_error = l1_error**2
            loss_l2_small = torch.sum(l2_error[torch.abs(sdf_volumes) < 0.1])
            loss_l2_large = torch.sum(l2_error[torch.abs(sdf_volumes) >= 0.1])
            loss_l1_small = torch.sum(l1_error[torch.abs(sdf_volumes) < 0.1])
            loss_l1_large = torch.sum(l1_error[torch.abs(sdf_volumes) >= 0.1])
            loss_pc = 0

            depth_images = torch.empty((batch_size, 480, 640), device=device)
            pointclouds = []
            # compute point clouds from sdf volumes at zero crossings
            for recon_sdf_volume, sdf_volume, depth_image in zip(
                recon_sdf_volumes, sdf_volumes, depth_images
            ):
                with torch.no_grad():
                    import random
                    import math

                    u1, u2, u3 = random.random(), random.random(), random.random()
                    q = (
                        torch.tensor(
                            [
                                math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
                                math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                                math.sqrt(u1) * math.sin(2 * math.pi * u3),
                                math.sqrt(u1) * math.cos(2 * math.pi * u3),
                            ]
                        )
                        .unsqueeze(0)
                        .to(device)
                    )
                    s = torch.Tensor([1.0]).to(device)
                    p = torch.Tensor([0.0, 0.0, -5.0]).to(device)

                    depth_image = render_depth_gpu(
                        sdf_volume[0],
                        p,
                        q,
                        s,
                        threshold=0.01,
                        camera=camera,
                    )
                    pointcloud = pointset_utils.depth_to_pointcloud(depth_image, camera)
                loss_pc = loss_pc + torch.sum(
                    pc_loss(pointcloud, p, q[0], s, recon_sdf_volume[0]) ** 2
                )

            loss_kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            loss = (
                l2_small_weight * loss_l2_small
                + l2_large_weight * loss_l2_large
                + l1_small_weight * loss_l1_small
                + l1_large_weight * loss_l1_large
                + pc_weight * loss_pc
                + loss_kld
                * (kld_weight if current_iteration > warm_up_iterations else 0)
            )

            print(f"Iteration {current_iteration}, epoch {current_epoch}, loss {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), current_iteration)
            writer.add_scalar("loss l2 small", loss_l2_small.item(), current_iteration)
            writer.add_scalar("loss l2 large", loss_l2_large.item(), current_iteration)
            writer.add_scalar("loss l1 small", loss_l1_small.item(), current_iteration)
            writer.add_scalar("loss l1 large", loss_l1_large.item(), current_iteration)
            writer.add_scalar("loss pc", loss_pc.item(), current_iteration)
            writer.add_scalar("loss kl", loss_kld, current_iteration)

            wandb.log(
                {
                    "total loss": loss.item(),
                    "loss l2 small": loss_l2_small.item(),
                    "loss l2 large": loss_l2_large.item(),
                    "loss l1 small": loss_l1_small.item(),
                    "loss l1 large": loss_l1_large.item(),
                    "loss kl": loss_kld.item(),
                    "loss pc": loss_pc.item(),
                },
                step=current_iteration,
            )

            current_iteration += 1

            with torch.no_grad():
                if current_iteration % 1000 == 0:
                    # show one reconstruction
                    mean = mean[0].cpu()
                    figure = sdf_utils.visualize_sdf_reconstruction(
                        sdf_volumes[0, 0].cpu().numpy(),
                        recon_sdf_volumes[0, 0].cpu().numpy(),
                    )
                    if figure.gca().has_data():
                        writer.add_figure(
                            tag="reconstruction",
                            figure=figure,
                            global_step=current_iteration,
                        )
                        wandb.log({"reconstruction": figure}, step=current_iteration)

                    # generate 8 samples from the prior
                    output, _ = sdfvae.inference(n=8, enforce_tsdf=True)
                    figure = sdf_utils.visualize_sdf_batch(
                        output.squeeze().cpu().numpy()
                    )
                    if figure.gca().has_data():
                        writer.add_figure(
                            tag="samples from prior",
                            figure=figure,
                            global_step=current_iteration,
                        )
                        wandb.log(
                            {"samples from prior": figure}, step=current_iteration
                        )

                    # generate 4 samples from prior
                    output, _ = sdfvae.inference(n=4, enforce_tsdf=True)
                    figure = sdf_utils.visualize_sdf_batch_columns(
                        output.squeeze().cpu().numpy()
                    )
                    if figure.gca().has_data():
                        writer.add_figure(
                            tag="samples from prior, columns",
                            figure=figure,
                            global_step=current_iteration,
                        )
                        wandb.log(
                            {"samples from prior, columns": figure},
                            step=current_iteration,
                        )

            #     if current_iteration % 10000 == 0:
            #         os.makedirs(model_base_path, exist_ok=True)
            #         checkpoint_path = os.path.join(model_base_path, F"{current_iteration}.ckp")
            #         utils.save_checkpoint(
            #             path=checkpoint_path,
            #             model=sdfvae,
            #             optimizer=optimizer,
            #             iteration=current_iteration,
            #             run_name=run_name)

            if current_iteration > iterations:
                break
        if current_iteration > iterations:
            break

    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    # save the final model
    torch.save(sdfvae.state_dict(), os.path.join(wandb.run.dir, f"{wandb.run.name}.pt"))
    config_path = os.path.join(wandb.run.dir, f"{wandb.run.name}.yaml")
    config["model"] = os.path.join(".", f"{wandb.run.name}.pt")
    yoco.save_config_to_file(config_path, config)


def main():
    """Entry point of the program."""
    # define the arguments
    parser = argparse.ArgumentParser(description="Training script for SDFVAE.")

    # parse arguments
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--batch_size", type=lambda x: int(float(x)))
    parser.add_argument("--iterations", type=lambda x: int(float(x)))
    parser.add_argument("--latent_size", type=lambda x: int(float(x)))
    parser.add_argument("--tsdf", type=utils.str_to_tsdf, default=False)
    parser.add_argument("--kld_weight", type=lambda x: float(x))
    parser.add_argument("--l2_large_weight", type=lambda x: float(x))
    parser.add_argument("--l2_small_weight", type=lambda x: float(x))
    parser.add_argument("--l1_large_weight", type=lambda x: float(x))
    parser.add_argument("--l1_small_weight", type=lambda x: float(x))
    parser.add_argument("--pc_weight", type=lambda x: float(x))
    parser.add_argument("--sign_weight", type=lambda x: float(x))
    parser.add_argument("--bce_weight", type=lambda x: float(x))
    parser.add_argument("--learning_rate", type=lambda x: float(x))
    parser.add_argument("--device")
    parser.add_argument("--config", default="configs/default.yaml", nargs="+")
    config = yoco.load_config_from_args(
        parser, search_paths=[".", "~/.sdfest/", sdfest.__path__[0]]
    )
    train(config)


if __name__ == "__main__":
    wandb.init(project="sdf-vae")
    main()
