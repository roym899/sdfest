"""Modular SDF pose and shape estimation in depth images."""
import copy
import math
import os
import pickle
import random
import time
from typing import Optional, Tuple

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from sdf_vae import sdf_utils
from sdf_vae.sdf_vae import SDFVAE
from sdf_single_shot.sdf_pose_network import SDFPoseNet, SDFPoseHead
from sdf_single_shot.pointnet import VanillaPointNet
from sdf_single_shot import pointset_utils, quaternion_utils
from sdf_differentiable_renderer import Camera, render_depth_gpu
import torch

from sdf_estimation import synthetic, losses


INIT_MODULE_DICT = {c.__name__: c for c in [SDFPoseHead, VanillaPointNet]}


class SDFPipeline:
    """SDF pose and shape estimation pipeline."""

    def __init__(self, config: dict) -> None:
        """Load and initialize the pipeline.

        Args:
            config: Configuration dictionary.
        """
        self._parse_config(config)

        self.init_network = SDFPoseNet(
            INIT_MODULE_DICT[self.init_config["backbone_type"]](
                **self.init_config["backbone"]
            ),
            INIT_MODULE_DICT[self.init_config["head_type"]](
                shape_dimension=self.vae_config["latent_size"],
                **self.init_config["head"],
            ),
        ).to(self.device)
        state_dict = torch.load(self.init_config["model"], map_location=self.device)
        self.init_network.load_state_dict(state_dict)
        self.init_network.eval()

        self.resolution = 64
        self.vae = SDFVAE(
            sdf_size=64,
            latent_size=self.vae_config["latent_size"],
            encoder_dict=self.vae_config["encoder"],
            decoder_dict=self.vae_config["decoder"],
            device=self.device,
        ).to(self.device)
        state_dict = torch.load(self.vae_config["model"], map_location=self.device)
        self.vae.load_state_dict(state_dict)
        self.vae.eval()

        self.cam = Camera(**self.camera_config)
        self.render = lambda sdf, pos, quat, i_s: render_depth_gpu(
            sdf, pos, quat, i_s, None, None, None, config["threshold"], self.cam
        )
        self.config = config

        self.log_data = []

    def _parse_config(self, config: dict) -> None:
        """Parse config dict.

        This function makes sure that all required keys are available.
        """
        self.device = config["device"]
        self.init_config = config["init"]
        self.vae_config = config["vae"] if "vae" in config else self.init_config["vae"]
        self.camera_config = config["camera"]

        self.config = config

    @staticmethod
    def _compute_gradients(loss: torch.Tensor) -> None:
        loss.backward()

    def _compute_losses(
        self,
        depth_input: torch.Tensor,
        depth_estimate: torch.Tensor,
        position: torch.Tensor,
        orientation: torch.Tensor,
        scale: torch.Tensor,
        sdf: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # depth l1
        overlap_mask = (depth_input > 0) & (depth_estimate > 0)
        depth_error = torch.abs(depth_estimate - depth_input)
        # max_depth_error = 0.05
        # depth_outlier_mask = depth_error > max_depth_error
        # depth_mask = overlap_mask & ~depth_outlier_mask
        # depth_error[~overlap_mask] = 0
        loss_depth = torch.mean(depth_error[overlap_mask])

        # pointcloud l1
        pointcloud_obs = pointset_utils.depth_to_pointcloud(
            depth_input, self.cam, normalize=False
        )
        pointcloud_error = losses.pc_loss(
            pointcloud_obs,
            position,
            orientation,
            scale,
            sdf,
        )
        loss_pc = torch.mean(torch.abs(pointcloud_error))

        # nearest neighbor l1
        # pointcloud_outliers = pointset_utils.depth_to_pointcloud(
        #     depth_estimate, self.cam, normalize=False, mask=depth_outlier_mask
        # )

        loss_nn = 0

        # if pointcloud_outliers.shape[0] != 0:
        # pass
        # loss_nn += 0
        # # TODO different gradients for point cloud (not derived by renderer)
        # outlier_nn_d = losses.nn_loss(pointcloud_outliers, pointcloud_obs)
        # # only use positive, because sqrt is not differentiable at 0
        # outlier_nn_d = outlier_nn_d[outlier_nn_d > 0]
        # loss_nn = loss_nn + torch.mean(torch.sqrt(outlier_nn_d))

        return loss_depth, loss_pc, loss_nn

    def __call__(
        self,
        depth_images: torch.Tensor,
        masks: torch.Tensor,
        color_images: torch.Tensor,
        visualize: bool = False,
        camera_positions: Optional[torch.Tensor] = None,
        camera_orientations: Optional[torch.Tensor] = None,
        log_path: Optional[str] = None,
        shape_optimization: bool = True,
        animation_path: Optional[str] = None,
        prior_orientation_distribution: Optional[torch.Tensor] = None,
        training_orientation_distribution: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Infer pose, size and latent representation from depth and mask.

        If multiple images are passed the cameras are assumed to be fixed.
        All tensors should be on the same device as the pipeline.

        Batch dimension N must be provided either for all or none of the arguments.

        Args:
            depth_images:
                the depth map containing the distance along the camera's z-axis,
                does not have to be masked, necessary preprocessing is done by pipeline,
                will be masked and preprocessed in-place
                (pass copy if full depth is used afterwards)
                shape (N, H, W) or (H, W) for a single depth image
            masks:
                binary mask of the object to estimate, same shape as depth_images
            color_images:
                the color image (currently only used in visualization),
                shape (N, H, W, 3) or (H, W, 3)
            visualize: Whether to visualize the intermediate steps and final result.
            camera_positions:
                position of camera in world coordinates for each image,
                if None, (0,0,0) will be assumed for all images,
                shape (N, 3) or (3,)
            camera_orientations:
                orientation of camera in world-frame as normalized quaternion,
                quaternion is in scalar-last convention,
                note, that this is the quaternion that transforms a point from camera
                to world-frame
                if None, (0,0,0,1) will be assumed for all images,
                shape (N, 4) or (4,)
            log_path:
                file path to write timestamps and intermediate steps to,
                no logging is performed if None
            shape_optimization:
                enable or disable shape optimization during iterative optimization
            animation_path:
                file path to write rendering and error visualizations to
            prior_orientation_distribution:
                Prior distribution of orientations used for initialization.
                If None, distribution of initialization network will not be modified.
                Only supported for initialization network with discretized orientation
                representation.
                Output distribution of initialization network will be adjusted by
                multiplying with
                prior_orientation_distribution / training_orientation_distribution
                and renormalizing.
                Tensor of shape (N,C,) or (C,) for single image.
                C being the number of grid cells in the SO3Grid used by the
                initialization network.
            training_orientation_distribution:
                Distribution of orientations used for training initialization network.
                If None, equal probability for each cell will be assumed.
                Note this is only approximately the same as a uniform distribution.
                Only used if prior_orientation_distribution is provided.
                Tensor of shape (C,). C being the number of grid cells in the SO3Grid
                used by the initialization network. N not supported, since same
                training network (and hence distribution) is used independent of view.

        Returns:
            - 3D pose of SDF center in world frame, shape (1,3,)
            - Orientation as normalized quaternion, scalar-last convention, shape (1,4,)
            - Size of SDF as length of half-width, shape (1,)
            - Latent shape representation of the object, shape (1,latent_size,).
        """
        if animation_path is not None:
            self._create_animation_folders(animation_path)

        start_time = time.time()  # for logging

        # Add batch dimension if necessary
        if depth_images.dim() == 2:
            depth_images = depth_images.unsqueeze(0)
            masks = masks.unsqueeze(0)
            color_images = color_images.unsqueeze(0)
            if camera_positions is not None:
                camera_positions.unsqueeze(0)
            if camera_orientations is not None:
                camera_orientations.unsqueeze(0)
            if prior_orientation_distribution is not None:
                camera_orientations.unsqueeze(0)

        if animation_path is not None:
            self._save_inputs(animation_path, depth_images, color_images, masks)

        n_imgs = depth_images.shape[0]

        if camera_positions is None:
            camera_positions = torch.zeros(n_imgs, 3, device=self.device)
        if camera_orientations is None:
            camera_orientations = torch.zeros(n_imgs, 4, device=self.device)
            camera_orientations[:, 3] = 1.0

        if log_path is not None:
            torch.cuda.synchronize()
            self._log_data(
                {
                    "timestamp": time.time() - start_time,
                    "depth_images": depth_images,
                    "masks": masks,
                    "color_images": color_images,
                    "camera_positions": camera_positions,
                    "camera_orientations": camera_orientations,
                },
            )

        # Initialization
        with torch.no_grad():
            self._preprocess_depth(depth_images, masks)
            latent_shape, position, scale, orientation = self._nn_init(
                depth_images,
                camera_positions,
                camera_orientations,
                prior_orientation_distribution,
                training_orientation_distribution,
            )
            scale_inv = 1 / scale

        if log_path is not None:
            torch.cuda.synchronize()
            self._log_data(
                {
                    "timestamp": time.time() - start_time,
                    "depth_images": depth_images,
                    "camera_positions": camera_positions,
                    "camera_orientations": camera_orientations,
                    "latent_shape": latent_shape,
                    "position": position,
                    "scale_inv": scale_inv,
                    "orientation": orientation,
                },
            )

        if animation_path is not None:
            self._save_preprocessed_inputs(animation_path, depth_images)

        # Iterative optimization
        self._current_iteration = 1

        position.requires_grad_()
        scale_inv.requires_grad_()
        orientation.requires_grad_()
        latent_shape.requires_grad_()

        if visualize:
            plt.ion()
            fig_vis, axes = plt.subplots(
                2, 3, sharex=True, sharey=True, figsize=(12, 8)
            )
            fig_loss, loss_ax = plt.subplots(1, 1)
            vmin, vmax = None, None

        depth_losses = []
        pointcloud_losses = []
        nn_losses = []
        total_losses = []

        opt_vars = [
            {"params": position, "lr": 1e-3},
            {"params": orientation, "lr": 1e-2},
            {"params": scale_inv, "lr": 1e-2},
            {"params": latent_shape, "lr": 1e-2},
        ]
        optimizer = torch.optim.Adam(opt_vars)

        while self._current_iteration <= self.config["max_iterations"]:
            optimizer.zero_grad()

            norm_orientation = orientation / torch.sqrt(torch.sum(orientation ** 2))

            with torch.set_grad_enabled(shape_optimization):
                sdf = self.vae.decode(latent_shape)

            loss_depth = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_pc = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_nn = torch.tensor(0.0, device=self.device, requires_grad=True)

            for depth_image, camera_position, camera_orientation in zip(
                depth_images, camera_positions, camera_orientations
            ):
                # transform object to camera frame
                q_w2c = quaternion_utils.quaternion_invert(camera_orientation)
                position_c = quaternion_utils.quaternion_apply(
                    q_w2c, position - camera_position
                )
                orientation_c = quaternion_utils.quaternion_multiply(
                    q_w2c, norm_orientation
                )

                depth_estimate = self.render(
                    sdf[0, 0], position_c[0], orientation_c[0], scale_inv[0]
                )

                view_loss_depth, view_loss_pc, view_loss_nn = self._compute_losses(
                    depth_image,
                    depth_estimate,
                    position_c[0],
                    orientation_c[0],
                    1 / scale_inv[0],
                    sdf[0, 0],
                )
                loss_depth = loss_depth + view_loss_depth
                loss_pc = loss_pc + view_loss_pc
                loss_nn = loss_nn + view_loss_nn

            loss = (
                self.config["depth_weight"] * loss_depth
                + self.config["pc_weight"] * loss_pc
                + self.config["nn_weight"] * loss_nn
            )

            self._compute_gradients(loss)

            optimizer.step()
            optimizer.zero_grad()

            if visualize:
                depth_losses.append(loss_depth.item())
                pointcloud_losses.append(loss_pc.item())
                nn_losses.append(loss_nn.item())
                total_losses.append(loss.item())

            with torch.no_grad():
                orientation /= torch.sqrt(torch.sum(orientation ** 2))

            if log_path is not None:
                torch.cuda.synchronize()
                self._log_data(
                    {
                        "timestamp": time.time() - start_time,
                        "latent_shape": latent_shape,
                        "position": position,
                        "scale_inv": scale_inv,
                        "orientation": orientation,
                    },
                )

            with torch.no_grad():
                if animation_path is not None:
                    self._save_current_state(
                        depth_images,
                        animation_path,
                        camera_positions,
                        camera_orientations,
                        position,
                        orientation,
                        scale_inv,
                        sdf,
                    )

                if visualize and (
                    self._current_iteration % 10 == 1
                    or self._current_iteration == self.config["max_iterations"]
                ):
                    q_w2c = quaternion_utils.quaternion_invert(camera_orientations[0])
                    position_c = quaternion_utils.quaternion_apply(
                        q_w2c, position - camera_positions[0]
                    )
                    orientation_c = quaternion_utils.quaternion_multiply(
                        q_w2c, orientation
                    )

                    current_depth = self.render(
                        sdf[0, 0], position_c, orientation_c, scale_inv
                    )

                    depth_image = depth_images[0]
                    color_image = color_images[0]

                    if self._current_iteration == 1:
                        vmin = depth_image[depth_image != 0].min() * 0.9
                        vmax = depth_image[depth_image != 0].max()
                        # show input image
                        axes[0, 0].clear()
                        axes[0, 0].imshow(depth_image.cpu(), vmin=vmin, vmax=vmax)
                        axes[0, 1].imshow(color_image.cpu())

                        # show initial estimate
                        axes[1, 0].clear()
                        axes[1, 0].imshow(
                            current_depth.detach().cpu(), vmin=vmin, vmax=vmax
                        )
                        axes[1, 0].set_title(f"loss {loss.item()}")

                    # update iterative estimate
                    # axes[0, 2].clear()
                    # axes[0, 2].imshow(rendered_error.detach().cpu())
                    # axes[0, 2].set_title("depth_loss")
                    # axes[1, 2].clear()
                    # axes[1, 2].imshow(error_pc.detach().cpu())
                    # axes[1, 2].set_title("pointcloud_loss")

                    loss_ax.clear()
                    loss_ax.plot(depth_losses, label="Depth")
                    loss_ax.plot(pointcloud_losses, label="Pointcloud")
                    loss_ax.plot(nn_losses, label="Nearest Neighbor")
                    loss_ax.plot(total_losses, label="Total")
                    loss_ax.legend()

                    axes[1, 1].clear()
                    axes[1, 1].imshow(
                        current_depth.detach().cpu(), vmin=vmin, vmax=vmax
                    )
                    axes[1, 1].set_title(f"loss {loss.item()}")
                    plt.draw()
                    plt.pause(0.001)

            self._current_iteration += 1

        if visualize:
            plt.ioff()
            plt.show()
            plt.close(fig_loss)
            plt.close(fig_vis)

        if log_path is not None:
            self._write_log_data(log_path)

        if animation_path is not None:
            self._create_animations(animation_path)

        return position, orientation, scale, latent_shape

    def _log_data(self, data: dict) -> None:
        """Add dictionary with associated timestamp to log data list."""
        new_log_data = copy.deepcopy(data)
        self.log_data.append(new_log_data)

    def _write_log_data(self, file_path: str) -> None:
        """Write current list of log data to file."""
        with open(file_path, "wb") as f:
            pickle.dump({"config": self.config, "log": self.log_data}, f)
        self.log_data = []  # reset log

    def generate_depth(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor,
        scale: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Generate depth image representing positioned object."""
        sdf = self.vae.decode(latent)
        depth = self.render(sdf[0, 0], position, orientation, 1 / scale)
        return depth

    def generate_mesh(
        self, latent: torch.tensor, scale: torch.tensor, complete_mesh: bool = False
    ) -> synthetic.Mesh:
        """Generate mesh without pose.

        Currently only supports batch size 1.

        Args:
            latent: latent shape descriptor, shape (1,L).
            scale:
                relative scale of the signed distance field, (i.e., half-width),
                shape (1,).
            complete_mesh:
                if True, the SDF will be padded with positive values prior
                to converting it to a mesh. This ensures a watertight mesh is created.

        Returns:
            Generate mesh by decoding latent shape descriptor and scaling it.
        """
        with torch.no_grad():
            sdf = self.vae.decode(latent)
            if complete_mesh:
                inc = 2
                sdf = torch.nn.functional.pad(sdf, (1, 1, 1, 1, 1, 1), value=1.0)
            else:
                inc = 0
            try:
                sdf = sdf.cpu().numpy()
                s = 2.0 / (self.resolution - 1)
                vertices, faces, _, _ = marching_cubes(
                    sdf[0, 0],
                    spacing=(
                        s,
                        s,
                        s,
                    ),
                    level=self.config["iso_threshold"],
                )

                c = s * (self.resolution + inc - 1) / 2.0  # move origin to center
                vertices -= np.array([[c, c, c]])

                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(vertices),
                    triangles=o3d.utility.Vector3iVector(faces),
                )
            except KeyError:
                return None
            return synthetic.Mesh(mesh=mesh, scale=scale.item(), rel_scale=True)

    @staticmethod
    def _preprocess_depth(depth_images: torch.Tensor, masks: torch.Tensor) -> None:
        """Preprocesses depth image based on segmentation mask.

        Args:
            depth_images:
                the depth images to preprocess, will be modified in place,
                shape (N, H, W)
            masks: the masks used for preprocessing, same shape as depth_images
        """
        # shrink mask
        # masks = (
        #     -torch.nn.functional.max_pool2d(
        #         -masks.double(), kernel_size=9, stride=1, padding=4
        #     )
        # ).bool()

        depth_images[~masks] = 0  # set outside of depth to 0

        # only consider available depth values for outlier detection
        # masks = torch.logical_and(masks, depth_images != 0)

        # depth_images =

        # remove outliers based on median
        # plt.imshow(depth_images[0].cpu().numpy())
        # plt.show()
        # for mask, depth_image in zip(masks, depth_images):
        #     median = torch.median(depth_image[mask])
        #     errors = torch.abs(depth_image[mask] - median)

        #     bins = 100
        #     hist = torch.histc(errors, bins=bins)
        #     print(hist)
        #     zero_indices = torch.nonzero(hist == 0)
        #     if len(zero_indices):
        #         threshold = zero_indices[0] / bins * errors.max()
        #         print(threshold)
        #         depth_image[torch.abs(depth_image - median) > threshold] = 0
        # plt.imshow(depth_images[0].cpu().numpy())
        # plt.show()

    def _nn_init(
        self,
        depth_images: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
        prior_orientation_distribution: Optional[torch.Tensor] = None,
        training_orientation_distribution: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Estimate shape, pose, scale and orientation using initialization network.

        Args:
            depth_images: the preprocessed depth images, shape (N, H, W)
            camera_positions:
                position of camera in world coordinates for each image, shape (N, 3)
            camera_orientations:
                orientation of camera in world-frame as normalized quaternion,
                quaternion is in scalar-last convention, shape (N, 4)
            strategy:
                how to handle multiple depth images
                "first": return single state based on first depth image
            prior_orientation_distribution:
                Prior distribution of orientations used for initialization.
                If None, distribution of initialization network will not be modified.
                Only supported for initialization network with discretized orientation
                representation.
                Output distribution of initialization network will be adjusted by
                multiplying with
                prior_orientation_distribution / training_orientation_distribution
                and renormalizing.
                Tensor of shape (N,C,). C being the number of grid cells in the SO3Grid
                used by the initialization network.
            training_orientation_distribution:
                Distribution of orientations used for training initialization network.
                If None, equal probability for each cell will be assumed.
                Note this is only approximately the same as a uniform distribution.
                Only used if prior_orientation_distribution is provided.
                Tensor of shape (C,). C being the number of grid cells in the SO3Grid
                used by the initialization network.

        Returns:
            Tuple comprised of:
            - Latent shape representation of the object, shape (1, latent_size)
            - 3D pose of SDF center in camera frame, shape (1, 3)
            - Size of SDF as length of half-width, (1,)
            - Orientation of SDF as normalized quaternion (1,4)
        """
        if (
            prior_orientation_distribution is not None
            and self.init_config["head"]["orientation_repr"] != "discretized"
        ):
            raise ValueError(
                "prior_orientation_distribution only supported for discretized "
                "orientation representation."
            )

        best = 0
        best_result = None
        for i, (depth_image, camera_orientation, camera_position) in enumerate(
            zip(depth_images, camera_orientations, camera_positions)
        ):
            centroid = None
            if self.init_config["backbone_type"] == "VanillaPointNet":
                inp = pointset_utils.depth_to_pointcloud(
                    depth_image, self.cam, normalize=False
                )
                if self.init_config["normalize_pose"]:
                    inp, centroid = pointset_utils.normalize_points(inp)
            else:
                inp = depth_image

            inp = inp.unsqueeze(0)
            latent_shape, position, scale, orientation_repr = self.init_network(inp)

            if self.config["mean_shape"]:
                latent_shape = latent_shape.new_zeros(latent_shape.shape)

            if centroid is not None:
                position += centroid

            if self.init_config["head"]["orientation_repr"] == "discretized":
                posterior_orientation_dist = torch.softmax(orientation_repr, -1)

                if prior_orientation_distribution is not None:
                    posterior_orientation_dist = self._adjust_categorical_posterior(
                        posterior=posterior_orientation_dist,
                        prior=prior_orientation_distribution[i],
                        train_prior=training_orientation_distribution,
                    )

                orientation_camera = torch.tensor(
                    self.init_network._head._grid.index_to_quat(
                        posterior_orientation_dist.argmax().item()
                    ),
                    dtype=torch.float,
                    device=self.device,
                ).unsqueeze(0)
            elif self.init_config["head"]["orientation_repr"] == "quaternion":
                orientation_camera = orientation_repr
            else:
                raise NotImplementedError("Orientation representation is not supported")

            # output are in camera frame, transform to world frame
            position_world = (
                quaternion_utils.quaternion_apply(camera_orientation, position)
                + camera_position
            )
            orientation_world = quaternion_utils.quaternion_multiply(
                camera_orientation, orientation_camera
            )

            if self.config["init_view"] == "first":
                return latent_shape, position_world, scale, orientation_world
            elif self.config["init_view"] == "best":
                if self.init_config["head"]["orientation_repr"] != "discretized":
                    raise NotImplementedError(
                        '"best" init strategy only supported with discretized '
                        "orientation representation"
                    )
                maximum = posterior_orientation_dist.max()
                if maximum > best:
                    best = maximum
                    best_result = latent_shape, position_world, scale, orientation_world
            else:
                raise NotImplementedError(
                    'Only "first" and "best" strategies are currently supported'
                )

        return best_result

    def _generate_uniform_quaternion(self) -> torch.tensor:
        """Generate a uniform quaternion.

        Following the method from K. Shoemake, Uniform Random Rotations, 1992.

        See: http://planning.cs.uiuc.edu/node198.html

        Returns:
            Uniformly distributed unit quaternion on the estimator's device.
        """
        u1, u2, u3 = random.random(), random.random(), random.random()
        return (
            torch.tensor(
                [
                    math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
                    math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                    math.sqrt(u1) * math.sin(2 * math.pi * u3),
                    math.sqrt(u1) * math.cos(2 * math.pi * u3),
                ]
            )
            .unsqueeze(0)
            .to(self.device)
        )

    def _create_animation_folders(self, animation_path: str) -> None:
        """Create subfolders to store animation frames."""
        os.makedirs(animation_path)
        depth_path = os.path.join(animation_path, "depth")
        os.makedirs(depth_path)
        error_path = os.path.join(animation_path, "depth_error")
        os.makedirs(error_path)
        sdf_path = os.path.join(animation_path, "sdf")
        os.makedirs(sdf_path)

    def _save_inputs(
        self,
        animation_path: str,
        color_images: torch.Tensor,
        depth_images: torch.Tensor,
        instance_masks: torch.Tensor,
    ) -> None:
        color_path = os.path.join(animation_path, "color_input.png")
        plt.imshow(color_images[0].cpu().numpy())
        plt.savefig(color_path)
        plt.close()
        depth_path = os.path.join(animation_path, "depth_input.png")
        plt.imshow(depth_images[0].cpu().numpy())
        plt.savefig(depth_path)
        plt.close()
        mask_path = os.path.join(animation_path, "mask.png")
        plt.imshow(instance_masks[0].cpu().numpy())
        plt.savefig(mask_path)
        plt.close()

    def _save_preprocessed_inputs(
        self,
        animation_path: str,
        depth_images: torch.Tensor,
    ) -> None:
        depth_path = os.path.join(animation_path, "preprocessed_depth_input.png")
        plt.imshow(depth_images[0].cpu().numpy())
        plt.savefig(depth_path)
        plt.close()

    def _save_current_state(
        self,
        depth_images: torch.Tensor,
        animation_path: str,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
        position: torch.Tensor,
        orientation: torch.Tensor,
        scale_inv: torch.Tensor,
        sdf: torch.Tensor,
    ) -> None:
        q_w2c = quaternion_utils.quaternion_invert(camera_orientations[0])
        position_c = quaternion_utils.quaternion_apply(
            q_w2c, position - camera_positions[0]
        )
        orientation_c = quaternion_utils.quaternion_multiply(q_w2c, orientation)
        current_depth = self.render(sdf[0, 0], position_c, orientation_c, scale_inv)
        depth_path = os.path.join(
            animation_path, "depth", f"{self._current_iteration:06}.png"
        )
        plt.imshow(current_depth.cpu().numpy(), interpolation="none")
        plt.savefig(depth_path)
        plt.close()

        error_image = torch.abs(current_depth - depth_images[0])
        error_image[depth_images[0] == 0] = 0
        error_image[current_depth == 0] = 0
        error_path = os.path.join(
            animation_path, "depth_error", f"{self._current_iteration:06}.png"
        )
        plt.imshow(error_image.cpu().numpy(), interpolation="none")
        plt.savefig(error_path)
        plt.close()

        unscaled_threshold = self.config["threshold"] * scale_inv.item()
        mesh = sdf_utils.mesh_from_sdf(
            sdf[0, 0].cpu().numpy(),
            unscaled_threshold,
            complete_mesh=True,
        )
        sdf_path = os.path.join(
            animation_path, "sdf", f"{self._current_iteration:06}.png"
        )

        # map y -> z; z -> y
        transform = np.eye(4)
        transform[0:3, 0:3] = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        sdf_utils.plot_mesh(mesh, transform=transform)
        plt.savefig(sdf_path)
        plt.close()

    def _create_animations(self, animation_path: str) -> None:
        names = ["sdf", "depth", "depth_error"]
        for name in names:
            frame_folder = os.path.join(animation_path, name)
            video_name = os.path.join(animation_path, f"{name}.mp4")
            ffmpeg.input(
                os.path.join(frame_folder, "*.png"), pattern_type="glob", framerate=30
            ).output(video_name).run()

    @staticmethod
    def _adjust_categorical_posterior(
        posterior: torch.Tensor, prior: torch.Tensor, train_prior: torch.Tensor
    ) -> torch.Tensor:
        """Adjust categorical posterior distribution.

        Posterior is calculated with a train_prior

        Args:
            posterior:
                Posterior distribution computed assuming train_prior.
                Shape (..., K). K being number of categories.
            prior:
                The desired new prior distribution.
                Same shape as posterior.
            train_prior:
                The prior distribution used to compute the posterior.
                If None, equal probability for each category will be assumed.
                Same shape as posterior.

        Returns:
            The categorical posterior, adjusted such that prior is prior, instead of
            train_prior.
            Same shape as posterior.
        """
        adjusted_posterior = posterior.clone()
        # adjust if prior different from training
        adjusted_posterior *= prior
        if train_prior is not None:
            adjusted_posterior /= train_prior
        adjusted_posterior = torch.nn.functional.normalize(
            adjusted_posterior, p=1, dim=-1
        )
        return adjusted_posterior
