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

from sdf_single_shot.datasets import dataset_utils
from sdf_single_shot.datasets.nocs_dataset import NOCSDataset
from sdf_single_shot.datasets.generated_dataset import SDFVAEViewDataset
from sdf_single_shot.sdf_pose_network import SDFPoseNet, SDFPoseHead
from sdf_single_shot.pointnet import VanillaPointNet
from sdf_single_shot import quaternion_utils, sdf_utils, utils

os.environ["PYOPENGL_PLATFORM"] = "egl"

MODULE_DICT = {c.__name__: c for c in [SDFPoseHead, VanillaPointNet]}


class Trainer:
    """Trainer for single shot pose and shape estimation network."""

    def __init__(self, config: dict) -> None:
        """Construct trainer.

        Args:
            config: The configuration for model and training.
        """
        self._read_config(config)

    def _read_config(self, config: dict) -> None:
        self._config = config
        self._validation_iteration = config["validation_iteration"]
        self._visualization_iteration = config["visualization_iteration"]
        self._checkpoint_iteration = config["checkpoint_iteration"]

        # propagate orientation representation and category to datasets
        datasets = (
            list(self._config["datasets"].values())
            + list(self._config["validation_datasets"].values())
        )
        for dataset in datasets:
            dataset["config_dict"]["orientation_repr"] = config["orientation_repr"]
            if "orientation_grid_resolution" in config:
                dataset["config_dict"]["orientation_grid_resolution"] = config[
                    "orientation_grid_resolution"
                ]
            if "category_str" in config:
                dataset["config_dict"]["category_str"] = config["category_str"]

        # propagate orientation representation to init head
        self._config["head"]["orientation_repr"] = config["orientation_repr"]
        if "orientation_grid_resolution" in config:
            self._config["head"]["orientation_grid_resolution"] = config[
                "orientation_grid_resolution"
            ]

    def run(self) -> None:
        """Train the model."""
        wandb.init(project="sdf_single_shot", config=self._config)

        self._device = self.get_device()

        # init dataset and dataloader
        self.vae = self.create_sdfvae()

        # init model to train
        self._sdf_pose_net = SDFPoseNet(
            backbone=MODULE_DICT[self._config["backbone_type"]](
                **self._config["backbone"]
            ),
            head=MODULE_DICT[self._config["head_type"]](
                shape_dimension=self._config["vae"]["latent_size"],
                **self._config["head"],
            ),
        ).to(self._device)

        # deterministic samples (needs to be done after model initialization, as it
        # can have varying number of parameters)
        torch.manual_seed(0)
        random.seed(torch.initial_seed())  # to get deterministic examples

        # print network summary
        torchinfo.summary(self._sdf_pose_net, (1, 500, 3), device=self._device)

        # init optimizer
        self._optimizer = torch.optim.Adam(
            self._sdf_pose_net.parameters(), lr=self._config["learning_rate"]
        )

        # load checkpoint if provided
        if "checkpoint" in self._config and self._config["checkpoint"] is not None:
            # TODO: checkpoint should always go together with model config!
            (
                self._sdf_pose_net,
                self._optimizer,
                self._current_iteration,
                self._run_name,
            ) = utils.load_checkpoint(
                self._config["checkpoint"],
                self._sdf_pose_net,
                self._optimizer,
                self._device,
            )
        else:
            self._current_iteration = 0
            self._run_name = (
                f"sdf_single_shot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
            )

        wandb.config.run_name = (
            self._run_name  # to allow association of ckpts with wandb runs
        )

        self._model_base_path = os.path.join(os.getcwd(), "models", self._run_name)

        self._multi_data_loader = self._create_multi_data_loader()

        # backup config to model directory
        os.makedirs(self._model_base_path, exist_ok=True)
        config_path = os.path.join(self._model_base_path, "config.yaml")
        yoco.save_config_to_file(config_path, self._config)

        program_starts = time.time()
        for samples in self._multi_data_loader:
            print(self._current_iteration)
            for k, v in samples.items():
                samples[k] = v.to(self._device)
            latent_shape, position, scale, orientation = self._sdf_pose_net(
                samples["pointset"]
            )
            predictions = {
                "latent_shape": latent_shape,
                "position": position,
                "scale": scale,
                "orientation": orientation,
            }

            loss = self._compute_loss(samples, predictions)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._compute_metrics(samples, predictions)

            if self._current_iteration % self._visualization_iteration == 0:
                self._generate_visualizations()

            if self._current_iteration % self._validation_iteration == 0:
                self._compute_validation_metrics()

            if self._current_iteration % self._checkpoint_iteration == 0:
                self._save_checkpoint()

            self._current_iteration += 1
            if self._current_iteration > self._config["iterations"]:
                break

        now = time.time()
        print(f"Training finished after {now-program_starts} seconds.")

        # save the final model
        torch.save(
            self._sdf_pose_net.state_dict(),
            os.path.join(wandb.run.dir, f"{wandb.run.name}.pt"),
        )
        config_path = os.path.join(wandb.run.dir, f"{wandb.run.name}.yaml")
        self._config["model"] = os.path.join(".", f"{wandb.run.name}.pt")
        yoco.save_config_to_file(config_path, self._config)

    def get_device(self) -> torch.device:
        """Create device based on config."""
        if "device" not in self._config or self._config["device"] is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._config["device"])

    def create_sdfvae(self) -> SDFVAE:
        """Create SDFVAE based on config.

        Returns:
            The SDFVAE on the specified device, with weights from specified model.
        """
        device = self.get_device()
        vae = SDFVAE(
            sdf_size=64,
            latent_size=self._config["vae"]["latent_size"],
            encoder_dict=self._config["vae"]["encoder"],
            decoder_dict=self._config["vae"]["decoder"],
            device=device,
        ).to(device)
        state_dict = torch.load(self._config["vae"]["model"], map_location=device)
        vae.load_state_dict(state_dict)
        return vae

    def _compute_loss(
        self,
        samples: dict,
        predictions: dict,
    ) -> torch.Tensor:
        """Compute total loss.

        Args:
            samples:
                Samples dictionary containing a subset of the following keys:
                    "latent_shape": Shape (N,D).
                    "position": Shape (N,3).
                    "scale": Shape (N,).
                    "orientation":
                        Shape (N,4) for quaternion representation.
                        Shape (N,) for discretized representation.
            predictions: Dictionary containing the following keys:
                "latent_shape": Shape (N,D).
                "position": Shape (N,3).
                "scale": Shape (N,).
                "orientation":
                    Shape (N,4) for quaternion representation.
                    Shape (N,R) for discretized representation.
        Returns:
            The combined loss. Scalar.
        """
        log_dict = {}

        loss = 0

        if "latent_shape" in samples:
            loss_latent_l2 = torch.nn.functional.mse_loss(
                predictions["latent_shape"], samples["latent_shape"]
            )
            log_dict["loss latent"] = loss_latent_l2.item()
            loss = loss + loss_latent_l2

        if "position" in samples:
            loss_position_l2 = torch.nn.functional.mse_loss(
                predictions["position"], samples["position"]
            )
            log_dict["loss position"] = loss_position_l2.item()
            loss = loss + self._config["position_weight"] * loss_position_l2

        if "scale" in samples:
            loss_scale_l2 = torch.nn.functional.mse_loss(
                predictions["scale"], samples["scale"]
            )
            log_dict["loss scale"] = loss_scale_l2.item()
            loss = loss + self._config["scale_weight"] * loss_scale_l2

        if "orientation" in samples:
            if self._config["head"]["orientation_repr"] == "quaternion":
                loss_orientation = quaternion_utils.simple_quaternion_loss(
                    predictions["orientation"], samples["orientation"]
                )
            elif self._config["head"]["orientation_repr"] == "discretized":
                loss_orientation = torch.nn.functional.cross_entropy(
                    predictions["orientation"], samples["orientation"]
                )
            else:
                raise NotImplementedError(
                    "Orientation repr "
                    f"{self._config['head']['orientation_repr']}"
                    " not supported."
                )
            log_dict["loss orientation"] = loss_orientation.item()
            loss = loss + self._config["orientation_weight"] * loss_orientation

        log_dict["total loss"] = loss.item()

        wandb.log(
            log_dict,
            step=self._current_iteration,
        )

        return loss

    def _create_multi_data_loader(self) -> dataset_utils.MultiDataLoader:
        data_loaders = []
        probabilities = []
        for dataset_dict in self._config["datasets"].values():
            dataset_type = utils.str_to_object(dataset_dict["type"])
            if dataset_type == SDFVAEViewDataset:
                dataset = dataset_type(
                    config=dataset_dict["config_dict"],
                    vae=self.vae,
                )
            elif dataset_type is not None:
                dataset = dataset_type(config=dataset_dict["config_dict"])
            else:
                raise NotImplementedError(
                    f"Dataset type {dataset_dict['type']} not supported."
                )
            probabilities.append(dataset_dict["probability"])
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self._config["batch_size"],
                collate_fn=dataset_utils.collate_samples,
            )
            data_loaders.append(data_loader)
        return dataset_utils.MultiDataLoader(data_loaders, probabilities)

    def _compute_metrics(self, samples: dict, predictions: dict) -> None:
        # compute metrics / i.e., loss and representation independent metrics
        with torch.no_grad():
            # extract quaternion from orientation representation
            target_quaternions = samples["quaternion"]
            if self._config["head"]["orientation_repr"] == "quaternion":
                predicted_quaternions = predictions["orientation"]
            elif self._config["head"]["orientation_repr"] == "discretized":
                predicted_quaternions = torch.empty_like(target_quaternions)
                for i, v in enumerate(predictions["orientation"]):
                    index = v.argmax().item()
                    quat = self._sdf_pose_net._head._grid.index_to_quat(index)
                    predicted_quaternions[i, :] = torch.tensor(quat)
            else:
                raise NotImplementedError(
                    "Orientation representation "
                    f"{self._config['head']['orientation_repr']}"
                    " is not supported"
                )
            geodesic_error = quaternion_utils.geodesic_distance(
                target_quaternions, predicted_quaternions
            )
            wandb.log(
                {
                    "metric geodesic distance": geodesic_error.item(),
                },
                step=self._current_iteration,
            )

    def _generate_visualizations(self) -> None:
        # generate visualizations
        with torch.no_grad():
            if self._current_iteration % 5000 == 0:
                # generate unseen input and target
                test_samples = next(iter(self._multi_data_loader))
                for k, v in test_samples.items():
                    test_samples[k] = v.to(self._device)
                test_out = self._sdf_pose_net(test_samples["pointset"])
                input_pointcloud = test_samples["pointset"][0].detach().cpu().numpy()
                input_pointcloud = np.hstack(
                    (
                        input_pointcloud,
                        np.full((input_pointcloud.shape[0], 1), 0),
                    )
                )
                output_sdfs = self.vae.decode(test_out[0])
                output_sdf = output_sdfs[0][0].detach().cpu().numpy()
                output_position = test_out[1][0].detach().cpu().numpy()
                output_scale = test_out[2][0].detach().cpu().numpy()
                if self._config["head"]["orientation_repr"] == "quaternion":
                    quat = test_out[3][0].detach().cpu().numpy()
                elif self._config["head"]["orientation_repr"] == "discretized":
                    quat = self._sdf_pose_net._head._grid.index_to_quat(
                        test_out[3][0].argmax()
                    )
                else:
                    raise NotImplementedError(
                        "Orientation representation "
                        f"{self._config['head']['orientation_repr']}"
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
                    step=self._current_iteration,
                )

    def _compute_validation_metrics(self) -> None:
        pass

    def _save_checkpoint(self) -> None:
        checkpoint_path = os.path.join(
            self._model_base_path, f"{self._current_iteration}.ckp"
        )
        utils.save_checkpoint(
            path=checkpoint_path,
            model=self._sdf_pose_net,
            optimizer=self._optimizer,
            iteration=self._current_iteration,
            run_name=self._run_name,
        )


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
