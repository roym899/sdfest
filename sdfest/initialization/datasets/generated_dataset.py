"""Module which provides SDFDataset class."""
import math
from typing import Tuple, Optional, Iterator, TypedDict
import random

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torchvision.transforms as T
import yoco

from sdfest.differentiable_renderer.sdf_renderer import render_depth_gpu, Camera
from sdfest.vae.sdf_vae import SDFVAE
from sdfest.initialization import pointset_utils, so3grid

# TODO support opencv / opengl conventions
# TODO support different scale conventions


class SDFVAEViewDataset(torch.utils.data.IterableDataset):
    """Dataset of SDF views generated by VAE and renderer from a random view."""
    class Config(TypedDict, total=False):
        """Configuration dictionary for SDFVAEViewDataset.

        Attributes:
            width: The width of the generated images in px.
            height: The height of the generated images in px.
            fov_deg: The horizontal fov in deg.
            z_min:
                Minimum z value (i.e., distance from camera) for the SDF.
                Note that positive z means in front of the camera, hence z_sampler
                should in most cases return positive values.
            z_max:
                Maximum z value (i.e., distance from camera) for the SDF.
            extent_mean:
                Mean extent of the SDF.
                Extent is the total side length of an SDF.
            extent_std:
                Standard deviation of the SDF scale.
            pointcloud: Whether to generate pointcloud or depth image.
            normalize_pose:
                Whether to center the augmented pointcloud at 0,0,0.
                Ignored if pointcloud=False
            orientation_repr:
                Which orientation representation is used. One of:
                    "quaternion"
                    "discretized"
            orientation_grid_resolution:
                Resolution of the orientation grid.
                Only used if orientation_repr is "discretized".
            mask_noise:
                Whether the mask should be perturbed to simulate noisy segmentation.
                If True a random, small, affine transform will be applied to the correct
                mask. The outliers will be filled with a random value sampled between
                mask_noise_min, and mask_noise_max.
            mask_noise_min:
                Minimum value to fill in for noisy mask.
                Only used if mask_noise is True.
            mask_noise_max:
                Maximum value to fill in for noisy mask.
                Only used if mask_noise is True.
            gaussian_noise_probability:
                Probability to apply gaussian noise filter on depth image.
            gaussian_noise_kernel_size:
                Size of the Gaussian kernel.
                Only used if Gaussian noise probability > 0.0.
            gausian_noise_kernel_std:
                Standard deviation of the Gaussian kernel.
                Only used if Gaussian noise probability > 0.0.
        """

        width: int
        height: int
        fov_deg: float
        z_min: float
        z_max: float
        extent_mean: float
        extent_std: float
        pointcloud: bool
        normalize_pose: Optional[bool]
        render_threshold: float
        orientation_repr: str
        orientation_grid_resolution: Optional[int]
        mask_noise: bool
        mask_noise_min: Optional[float]
        mask_noise_max: Optional[float]
        norm_noise: bool
        norm_noise_min: Optional[float]
        norm_noise_max: Optional[float]
        scale_to_unit_ball: bool
        gaussian_noise_probability: float
        gaussian_noise_kernel_size: Optional[int]
        gausian_noise_kernel_std: Optional[float]

    default_config: Config = {
        "device": "cuda",
        "width": 640,
        "height": 480,
        "fov_deg": 90,
        "render_threshold": 0.004,
        "normalize_pose": None,
        "orientation_repr": "quaternion",
        "orientation_grid_resolution": None,
        "mask_noise": False,
        "mask_noise_min": 0.1,
        "mask_noise_max": 2.0,
        "norm_noise": False,
        "norm_noise_min": -0.2,
        "norm_noise_max": 0.2,
        "scale_to_unit_ball": False,
        "gaussian_noise_probability": 0.0,
        "gaussian_noise_kernel_size": 5,
        "gaussian_noise_kernel_std": 1,
    }

    def __init__(
        self,
        config: dict,
        vae: SDFVAE,
    ) -> None:
        """Initialize the dataset.

        Args:
            config:
                Configuration dictionary of dataset. Provided dictionary will be merged
                with default_dict. See SDFVAEViewDataset.Config for supported keys.
            vae: The variational autoencoder used to create training samples.
        """
        config = yoco.load_config(config, current_dict=SDFVAEViewDataset.default_config)
        self._vae = vae
        self._vae.eval()
        self._device = next(self._vae.parameters()).device
        f = config["width"] / math.tan(config["fov_deg"] * math.pi / 180.0 / 2.0) / 2
        self._camera = Camera(
            width=config["width"],
            height=config["height"],
            fx=f,
            fy=f,
            cx=config["width"] / 2,
            cy=config["height"] / 2,
            pixel_center=0.5,
        )
        self._fov_deg = config["fov_deg"]
        self._z_min = config["z_min"]
        self._z_max = config["z_max"]
        self._z_sampler = lambda: random.uniform(self._z_min, self._z_max)
        self._extent_mean = config["extent_mean"]
        self._extent_std = config["extent_std"]
        self._scale_sampler = (
            lambda: random.gauss(self._extent_mean, self._extent_std) / 2.0
        )
        self._mask_noise = config["mask_noise"]
        self._mask_noise_min = config["mask_noise_min"]
        self._mask_noise_max = config["mask_noise_max"]
        self._mask_noise_sampler = lambda: random.uniform(
            config["mask_noise_min"], config["mask_noise_max"]
        )
        self._norm_noise = config["norm_noise"]
        self._norm_noise_min = config["norm_noise_min"]
        self._norm_noise_max = config["norm_noise_max"]
        self._norm_noise_sampler = lambda: random.uniform(
            config["norm_noise_min"], config["norm_noise_max"]
        )
        self._scale_to_unit_ball = config["scale_to_unit_ball"]
        self._pointcloud = config["pointcloud"]
        self._normalize_pose = config["normalize_pose"]
        self._render_threshold = config["render_threshold"]
        self._orientation_repr = config["orientation_repr"]
        if self._orientation_repr == "discretized":
            self._orientation_grid = so3grid.SO3Grid(
                config["orientation_grid_resolution"]
            )
        self._gaussian_noise_probability = config["gaussian_noise_probability"]
        self._create_gaussian_kernel(
            config["gaussian_noise_kernel_std"], config["gaussian_noise_kernel_size"]
        )

    def __iter__(self) -> Iterator:
        """Return SDF volume at a specific index.

        Returns:
            Infinite iterator, generating sample dictionaries.
            See SDFVAEViewDataset._generate_sample for more details about returned
            dictionaries.
        """
        # this is an infinite iterator as the sentinel False will never be returned
        while True:
            yield self._generate_valid_sample()

    def _generate_uniform_quaternion(self) -> torch.Tensor:
        """Generate a uniform quaternion.

        Following the method from K. Shoemake, Uniform Random Rotations, 1992.

        See: http://planning.cs.uiuc.edu/node198.html

        Returns:
            Uniformly distributed unit quaternion on the dataset's device.
        """
        u1, u2, u3 = random.random(), random.random(), random.random()
        return torch.tensor(
            [
                math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
                math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                math.sqrt(u1) * math.sin(2 * math.pi * u3),
                math.sqrt(u1) * math.cos(2 * math.pi * u3),
            ]
        ).to(self._device)

    def _is_valid(self, sample: dict) -> bool:
        """Check whether a generated sample is valid.

        A valid sample contains at least one valid point in the depth image and hence
        in the pointcloud.
        """
        if sample["depth"].max() == 0:
            return False
        return True

    def _generate_valid_sample(self) -> dict:
        """Generate a single non-zero sample.

        Returns:
            See _generate_sample.
        """
        sample = self._generate_sample()
        while not self._is_valid(sample):
            sample = self._generate_sample()
            print("Warning: invalid sample, this should only happen very infrequently.")
            # if this happens often, either the SDF does not have any zero crossings
            # or the object pose is completely outside the frustum
        return sample

    def _perturb_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Perturb mask by applying small random affine transform to it.

        Args:
            mask: The mask to perturb.
        Returns:
            The perturbed mask. Same shape as mask.
        """
        affine_transfomer = T.RandomAffine(
            degrees=(0, 1), translate=(0.00, 0.01), scale=(0.999, 1.001)
        )
        return affine_transfomer(mask.unsqueeze(0))[0]

    def _generate_sample(self) -> Tuple:
        """Generate a single sample. Possibly (albeit very unlikely) zero / empty.

        Return:
            Sample containing the following items:
                "depth"
                "pointset"
                "latent_shape"
                "position"
                "orientation"
                "quaternion"
                "scale"
        """
        sample = {}

        latent = self._vae.sample()  # will be 1xlatent_size (batch size 1 here)
        with torch.no_grad():
            sdf = self._vae.decode(latent)
        # generate x, y, z s.t. center is inside frustum
        z = self._z_sampler()
        x_pix = random.uniform(-self._camera.width / 2, self._camera.height / 2)
        x = x_pix / self._camera.fx * z
        y_pix = random.uniform(-self._camera.height / 2, self._camera.height / 2)
        y = y_pix / self._camera.fy * z
        position = torch.tensor([x, y, -z]).to(self._device)
        quaternion = self._generate_uniform_quaternion()
        scale = torch.tensor(self._scale_sampler()).to(self._device)
        inv_scale = 1.0 / scale
        orientation = self._quat_to_orientation_repr(quaternion)

        depth = render_depth_gpu(
            sdf[0, 0],
            position,
            quaternion,
            inv_scale,
            threshold=self._render_threshold,
            camera=self._camera,
        )

        exact_mask = depth != 0
        if self._mask_noise:
            final_mask = self._perturb_mask(exact_mask)
            depth[~exact_mask] = self._mask_noise_sampler()
        else:
            final_mask = exact_mask

        if self._gaussian_noise_probability > 0.0:
            if random.random() < self._gaussian_noise_probability:
                invalid_depth_mask = depth == 0
                depth[invalid_depth_mask] = torch.nan
                depth_filtered = torch.nn.functional.conv2d(
                    depth[None, None], self._gaussian_kernel, padding="same"
                )[0, 0]
                # nan might become inf be preserved
                # https://github.com/pytorch/pytorch/issues/12484
                mask = torch.logical_or(depth_filtered.isnan(), depth_filtered.isinf())
                depth[~mask] = depth_filtered[~mask]
                depth[depth.isnan()] = 0.0

        depth[~final_mask] = 0

        if self._pointcloud:
            pointset = pointset_utils.depth_to_pointcloud(
                depth, self._camera, convention="opengl"
            )

            if self._normalize_pose:
                pointset, centroid = pointset_utils.normalize_points(pointset)
                position -= centroid  # adjust target

                if self._norm_noise:
                    noise = position.new_tensor(
                        [
                            self._norm_noise_sampler(),
                            self._norm_noise_sampler(),
                            self._norm_noise_sampler(),
                        ]
                    )
                    position += noise
                    pointset += noise

                if self._scale_to_unit_ball:
                    max_distance = torch.max(torch.linalg.norm(pointset))
                    pointset /= max_distance
                    scale /= max_distance

            sample["pointset"] = pointset

        sample["depth"] = depth
        sample["latent_shape"] = latent.squeeze()
        sample["position"] = position
        sample["orientation"] = orientation
        sample["quaternion"] = quaternion
        sample["scale"] = scale

        return sample

        # TODO: add augmentation

    def _quat_to_orientation_repr(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to selected orientation representation.

        Args:
            quaternion:
                The quaternion to convert, scalar-last, shape (4,).
        Returns:
            The same orientation as represented by the quaternion in the chosen
            orientation representation.
        """
        if self._orientation_repr == "quaternion":
            return quaternion
        elif self._orientation_repr == "discretized":
            index = self._orientation_grid.quat_to_index(quaternion.cpu().numpy())
            return torch.tensor(
                index,
                device=self._device,
                dtype=torch.long,
            )
        else:
            raise NotImplementedError(
                f"Orientation representation {self._orientation_repr} is not supported."
            )

    def _create_gaussian_kernel(self, std: float, kernel_size: int) -> None:
        """Create and set Gaussian noise kernel used for smoothing the depth image."""
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size should be odd.")
        impulse = np.zeros((kernel_size, kernel_size))
        impulse[kernel_size // 2, kernel_size // 2] = 1
        kernel = gaussian_filter(impulse, std)
        self._gaussian_kernel = torch.Tensor(kernel[None, None]).to(self._device)
