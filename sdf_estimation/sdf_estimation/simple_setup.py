"""Modular SDF pose and shape estimation in depth images."""
import argparse
import copy
import math
import pickle
import random
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from sdf_vae.sdf_vae import SDFVAE
from sdf_single_shot.sdf_pose_network import SDFPoseNet, SDFPoseHead
from sdf_single_shot.pointnet import VanillaPointNet
from sdf_single_shot import pointset_utils, quaternion_utils
from sdf_differentiable_renderer import Camera, render_depth_gpu
import torch
import yoco

from sdf_estimation import synthetic, losses


INIT_MODULE_DICT = {c.__name__: c for c in [SDFPoseHead, VanillaPointNet]}


class SDFPipeline:
    """SDF pose and shape estimation pipeline."""

    def __init__(self, config: dict) -> None:
        """Load and initialize the pipeline.

        Args:
            config: Configuration dictionary.
        """
        self.device = config["device"]

        self.init_network = SDFPoseNet(
            INIT_MODULE_DICT[config["init"]["backbone_type"]](
                **config["init"]["backbone"]
            ),
            INIT_MODULE_DICT[config["init"]["head_type"]](
                shape_dimension=config["vae"]["latent_size"], **config["init"]["head"]
            ),
        ).to(self.device)
        state_dict = torch.load(config["init"]["model"], map_location=self.device)
        self.init_network.load_state_dict(state_dict)
        self.init_network.eval()

        self.resolution = 64
        self.vae = SDFVAE(
            sdf_size=64,
            latent_size=config["vae"]["latent_size"],
            encoder_dict=config["vae"]["encoder"],
            decoder_dict=config["vae"]["decoder"],
            device=self.device,
        ).to(self.device)
        state_dict = torch.load(config["vae"]["model"], map_location=self.device)
        self.vae.load_state_dict(state_dict)
        self.vae.eval()

        self.cam = Camera(**config["camera"])
        self.render = lambda sdf, pos, quat, i_s: render_depth_gpu(
            sdf, pos, quat, i_s, None, None, None, config["threshold"], self.cam
        )
        self.config = config

        self.log_data = []

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
    ) -> tuple:
        """Infer pose, size and latent representation from depth and mask.

        If multiple images are passed the cameras are assumed to be fixed.
        All tensors should be on the same device as the pipeline.

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

        Returns:
            3D pose of SDF center in world frame, shape (3,)
            orientation as normalized quaternion, scalar-last convention, shape (4,)
            size of SDF as length of half-width, scalar
            latent shape representation of the object, shape (latent_size,)
        """
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
                depth_images, camera_positions, camera_orientations
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

        # Iterative optimization
        current_iteration = 1

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
            {"params": latent_shape, "lr": 2e-1},
        ]
        optimizer = torch.optim.Adam(opt_vars)

        while current_iteration <= self.config["max_iterations"]:
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
                if visualize and (
                    current_iteration % 10 == 1
                    or current_iteration == self.config["max_iterations"]
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

                    if current_iteration == 1:
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

            current_iteration += 1

        if visualize:
            plt.ioff()
            plt.show()
            plt.close(fig_loss)
            plt.close(fig_vis)

        if log_path is not None:
            self._write_log_data(log_path)

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
        masks = (
            -torch.nn.functional.max_pool2d(
                -masks.double(), kernel_size=5, stride=1, padding=2
            )
        ).bool()

        depth_images[~masks] = 0  # set outside of depth to 0

        # only consider available depth values for outlier detection
        # masks = torch.logical_and(masks, depth_images != 0)

        # depth_images =

        # remove outliers based on median
        # for mask, depth_image in zip(masks, depth_images):
        #     median = torch.median(depth_image[mask])
        #     errors = torch.abs(depth_image[mask] - median)

        #     bins = 20
        #     hist = torch.histc(errors, bins=bins)
        #     print(hist)
        #     zero_indices = torch.nonzero(hist == 0)
        #     if len(zero_indices):
        #         threshold = zero_indices[0] / bins * errors.max()
        #         depth_image[torch.abs(depth_image - median) > threshold] = 0

    def _nn_init(
        self,
        depth_images: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_orientations: torch.Tensor,
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

        Returns:
            Tuple comprised of:
            - Latent shape representation of the object, shape (N, latent_size)
            - 3D pose of SDF center in camera frame, shape (N, 3)
            - Size of SDF as length of half-width, (N,)
            - Orientation of SDF as normalized quaternion (N,4)
        """
        best = 0
        best_result = None
        for depth_image, camera_orientation, camera_position in zip(
            depth_images, camera_orientations, camera_positions
        ):
            centroid = None
            if self.config["init"]["backbone_type"] == "VanillaPointNet":
                inp = pointset_utils.depth_to_pointcloud(
                    depth_image, self.cam, normalize=False
                )
                if self.config["init"]["generated_dataset"]["normalize_pose"]:
                    inp, centroid = pointset_utils.normalize_points(inp)
            else:
                inp = depth_image

            inp = inp.unsqueeze(0)
            latent_shape, position, scale, orientation_repr = self.init_network(inp)

            if self.config["mean_shape"]:
                latent_shape = latent_shape.new_zeros(latent_shape.shape)

            if centroid is not None:
                position += centroid

            if self.config["init"]["head"]["orientation_repr"] == "discretized":
                orientation_repr = torch.softmax(orientation_repr, 1)
                orientation_camera = torch.tensor(
                    self.init_network._head._grid.index_to_quat(
                        orientation_repr.argmax().item()
                    ),
                    dtype=torch.float,
                    device=self.device,
                ).unsqueeze(0)
            elif self.config["init"]["head"]["orientation_repr"] == "quaternion":
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
                if self.config["init"]["head"]["orientation_repr"] != "discretized":
                    raise NotImplementedError(
                        '"Best" init strategy only supported with discretized '
                        "orientation representation"
                    )
                maximum = orientation_repr.max()
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


def main() -> None:
    """Entry point of the program."""
    # define the arguments
    parser = argparse.ArgumentParser(description="SDF pose estimation in depth images")

    # parse arguments
    parser.add_argument("--device")
    parser.add_argument("--config", default="configs/default.yaml", nargs="+")
    args = parser.parse_args()
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    config = yoco.load_config(config_dict)

    pipeline = SDFPipeline(config)

    # generate test depth with same camera params
    cam = Camera(**config["camera"])
    mesh = synthetic.Mesh(
        path="/home/leo/datasets/shapenet/mug_03797390/61c10dccfa8e508e2d66cbf6a91063/"
        "models/model_normalized.obj",
        scale=0.08,
    )
    mesh.position = np.array([-0.0, 0.1, 0.7])
    mesh.orientation = np.array([1.0, 0.5, 0.0, -1.5])
    mesh.orientation /= np.linalg.norm(mesh.orientation)

    # render depth image
    test_depth = torch.tensor(
        synthetic.draw_depth_geometry(mesh, cam), device=config["device"]
    )
    test_mask = test_depth != 0

    position, orientation, scale, shape = pipeline(
        test_depth, test_mask, test_depth, visualize=True
    )

    with torch.no_grad():
        # visualize pipeline output
        out = pipeline.generate_depth(position, orientation, scale, shape)
        plt.imshow(out.cpu())
        plt.show()


if __name__ == "__main__":
    main()
