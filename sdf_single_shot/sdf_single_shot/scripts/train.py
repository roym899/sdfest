"""Script to train model."""
import argparse
from datetime import datetime
import os
import random
import time

import numpy as np
from sdf_vae.sdf_vae import SDFVAE
import torch
import torchinfo
import wandb
import yoco

from sdf_single_shot.datasets.generated_dataset import (
    SDFVAEViewDataset,
    collate_pointsets,
)
from sdf_single_shot.sdf_pose_network import SDFPoseNet, SDFPoseHead
from sdf_single_shot.pointnet import VanillaPointNet
from sdf_single_shot import sdf_utils, utils

os.environ["PYOPENGL_PLATFORM"] = "egl"

MODULE_DICT = {c.__name__: c for c in [SDFPoseHead, VanillaPointNet]}


class Trainer:
    """Trainer for single shot pose and shape estimation network."""

    def __init__(self, config: dict) -> None:
        """Construct trainer.

        Args:
            config: The configuration for model and training.
        """
        self.config = config

    def run(self) -> None:
        """Train the model."""
        wandb.init(project="sdf_single_shot", config=self.config)

        device = self.get_device()

        # init dataset and dataloader
        vae = self.create_sdfvae()

        # init model to train
        sdf_pose_net = SDFPoseNet(
            backbone=MODULE_DICT[self.config["backbone_type"]](
                **self.config["backbone"]
            ),
            head=MODULE_DICT[self.config["head_type"]](
                shape_dimension=self.config["vae"]["latent_size"], **self.config["head"]
            ),
        ).to(device)

        # deterministic samples (needs to be done after model initialization, as it
        # can have varying number of parameters)
        torch.manual_seed(0)
        random.seed(torch.initial_seed())  # to get deterministic examples

        dataset = SDFVAEViewDataset(
            vae,
            device,
            orientation_repr=self.config["head"]["orientation_repr"],
            orientation_grid=sdf_pose_net._head._grid,
            **self.config["generated_dataset"],
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["batch_size"],
            collate_fn=collate_pointsets,
        )

        # print network summary
        torchinfo.summary(sdf_pose_net, (1, 500, 3), device=device)

        # init optimizer
        optimizer = torch.optim.Adam(
            sdf_pose_net.parameters(), lr=self.config["learning_rate"]
        )

        # load checkpoint if provided
        if "checkpoint" in self.config and self.config["checkpoint"] is not None:
            # TODO: checkpoint should always go together with model config!
            (
                sdf_pose_net,
                optimizer,
                current_iteration,
                run_name,
            ) = utils.load_checkpoint(
                self.config["checkpoint"], sdf_pose_net, optimizer, device
            )
        else:
            current_iteration = 0
            run_name = (
                f"sdf_single_shot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
            )

        wandb.config.run_name = (
            run_name  # to allow association of ckpts with wandb runs
        )

        model_base_path = os.path.join(os.getcwd(), "models", run_name)

        # backup config to model directory
        os.makedirs(model_base_path, exist_ok=True)
        config_path = os.path.join(model_base_path, "config.yaml")
        yoco.save_config_to_file(config_path, self.config)

        program_starts = time.time()
        while current_iteration <= self.config["iterations"]:
            for inp, targets in data_loader:
                out = sdf_pose_net(inp)

                loss = self.compute_loss(out, targets, current_iteration)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # compute metrics / i.e., loss and representation independent metrics
                with torch.no_grad():
                    # extract quaternion from orientation representation
                    target_quaternions = targets[4]
                    if self.config["head"]["orientation_repr"] == "quaternion":
                        predicted_quaternions = out[3]
                    elif self.config["head"]["orientation_repr"] == "discretized":
                        predicted_quaternions = torch.empty_like(target_quaternions)
                        for i, v in enumerate(out[3]):
                            index = v.argmax().item()
                            quat = sdf_pose_net._head._grid.index_to_quat(index)
                            predicted_quaternions[i, :] = torch.tensor(quat)
                    else:
                        raise NotImplementedError(
                            "Orientation representation "
                            f"{self.config['head']['orientation_repr']}"
                            " is not supported"
                        )
                    geodesic_error = torch.mean(
                        torch.acos(
                            torch.clip(
                                torch.abs(
                                    torch.sum(
                                        target_quaternions * predicted_quaternions,
                                        dim=1,
                                    )
                                ),
                                0,
                                1,
                            )
                        )
                    )
                    wandb.log(
                        {
                            "metric geodesic distance": geodesic_error.item(),
                        },
                        step=current_iteration,
                    )

                current_iteration += 1
                with torch.no_grad():
                    if current_iteration % 5000 == 0:
                        # generate unseen input and target
                        test_inp, _ = next(iter(data_loader))
                        test_out = sdf_pose_net(test_inp)
                        input_pointcloud = test_inp[0].detach().cpu().numpy()
                        input_pointcloud = np.hstack(
                            (
                                input_pointcloud,
                                np.full((input_pointcloud.shape[0], 1), 0),
                            )
                        )
                        output_sdfs = vae.decode(test_out[0])
                        output_sdf = output_sdfs[0][0].detach().cpu().numpy()
                        output_position = test_out[1][0].detach().cpu().numpy()
                        output_scale = test_out[2][0].detach().cpu().numpy()
                        if self.config["head"]["orientation_repr"] == "quaternion":
                            quat = test_out[3][0].detach().cpu().numpy()
                        elif self.config["head"]["orientation_repr"] == "discretized":
                            quat = sdf_pose_net._head._grid.index_to_quat(
                                test_out[3][0].argmax()
                            )
                        else:
                            raise NotImplementedError(
                                "Orientation representation "
                                f"{self.config['head']['orientation_repr']}"
                                " is not supported"
                            )
                        output_pointcloud = sdf_utils.sdf_to_pointcloud(
                            output_sdf, output_position, quat, output_scale
                        )
                        output_pointcloud = np.hstack(
                            (
                                output_pointcloud,
                                np.full((output_pointcloud.shape[0], 1), 1),
                            )
                        )
                        pointcloud = np.vstack((input_pointcloud, output_pointcloud))

                        wandb.log(
                            {"point_cloud": wandb.Object3D(pointcloud)},
                            step=current_iteration,
                        )

                    if current_iteration % 100000 == 0:
                        checkpoint_path = os.path.join(
                            model_base_path, f"{current_iteration}.ckp"
                        )
                        utils.save_checkpoint(
                            path=checkpoint_path,
                            model=sdf_pose_net,
                            optimizer=optimizer,
                            iteration=current_iteration,
                            run_name=run_name,
                        )

                if current_iteration > self.config["iterations"]:
                    break
            if current_iteration > self.config["iterations"]:
                break

        now = time.time()
        print(
            "It has been {0} seconds since the loop started".format(
                now - program_starts
            )
        )

        # save the final model
        torch.save(
            sdf_pose_net.state_dict(),
            os.path.join(wandb.run.dir, f"{wandb.run.name}.pt"),
        )
        config_path = os.path.join(wandb.run.dir, f"{wandb.run.name}.yaml")
        self.config["model"] = os.path.join(".", f"{wandb.run.name}.pt")
        yoco.save_config_to_file(config_path, self.config)

    def get_device(self) -> torch.device:
        """Create device based on config."""
        if "device" not in self.config or self.config["device"] is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config["device"])

    def create_sdfvae(self) -> SDFVAE:
        """Create SDFVAE based on config.

        Returns:
            The SDFVAE on the specified device, with weights from specified model.
        """
        device = self.get_device()
        vae = SDFVAE(
            sdf_size=64,
            latent_size=self.config["vae"]["latent_size"],
            encoder_dict=self.config["vae"]["encoder"],
            decoder_dict=self.config["vae"]["decoder"],
            device=device,
        ).to(device)
        state_dict = torch.load(self.config["vae"]["model"], map_location=device)
        vae.load_state_dict(state_dict)
        return vae

    def compute_loss(
        self, out: tuple, targets: tuple, current_iteration: int
    ) -> torch.Tensor:
        """Compute , loss_componentstotal weighed loss.

        Args:
            out: Output of SDFPoseNet.
            targets: Tuple of targets:
                - latent shape, shape (N,D),
                - position, shape (N,3),
                - scale, shape (N,),
                - orientation, shape (N,M) (depends on orientation representation).
            current_iteration:
                Current trainig iteration. Used to log loss terms to wandb.
        Returns:
            The combined loss. Shape (1,).
        """
        loss_latent_l2 = torch.nn.functional.mse_loss(out[0], targets[0])
        loss_position_l2 = torch.nn.functional.mse_loss(out[1], targets[1])
        loss_scale_l2 = torch.nn.functional.mse_loss(out[2], targets[2])
        if self.config["head"]["orientation_repr"] == "quaternion":
            loss_orientation = torch.mean(1 - torch.sum((out[3] * targets[3]), 1) ** 2)
        elif self.config["head"]["orientation_repr"] == "discretized":
            loss_orientation = torch.nn.functional.cross_entropy(out[3], targets[3])
        else:
            raise NotImplementedError(
                "Orientation repr "
                f"{self.config['head']['orientation_repr']}"
                " not supported."
            )

        loss = (
            loss_latent_l2
            + self.config["position_weight"] * loss_position_l2
            + self.config["scale_weight"] * loss_scale_l2
            + self.config["orientation_weight"] * loss_orientation
        )

        wandb.log(
            {
                "total loss": loss.item(),
                "loss latent": loss_latent_l2.item(),
                "loss position": loss_position_l2.item(),
                "loss scale": loss_scale_l2.item(),
                "loss orientation": loss_orientation.item(),
            },
            step=current_iteration,
        )

        return loss


def main() -> None:
    """Entry point of the program."""
    # define the arguments
    parser = argparse.ArgumentParser(description="Training script for init network.")
    parser.add_argument("--config", default="configs/default.yaml", nargs="+")
    config = yoco.load_config_from_args(parser)
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
