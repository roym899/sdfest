"""Wrapper for pose and shape estimation methods."""
from abc import ABC
from typing import Optional, TypedDict

import numpy as np
import torch
import torchvision.transforms.functional as TF

from sdf_differentiable_renderer import Camera
from sdf_single_shot import pointset_utils

import yoco
from cass.lib.models import CASS
from cass.datasets.dataset import get_bbox


class PredictionDict(TypedDict):
    """Pose and shape prediction.

    Attributes:
        position:
            Position of object center in camera frame. OpenCV convention. Shape (3,).
        orientation:
            Orientation of object in camera frame. OpenCV convention.
            Scalar-last quaternion, shape (4,).
        extents:
            Bounding box side lengths., shape (3,).
        reconstructed_pointcloud:
            Reconstructed pointcloud in object frame.
            None if method does not perform reconstruction.
    """

    position: torch.Tensor
    orientation: torch.Tensor
    extents: torch.Tensor
    reconstructed_pointcloud: Optional[torch.Tensor]


class MethodWrapper(ABC):
    """Interface class for pose and shape estimation methods."""

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_id: int,
    ) -> PredictionDict:
        """Run a method to predict pose and shape of an object.

        Args:
            color_image: The color image, shape (H, W, 3), RGB, 0-1, float.
            depth_image: The depth image, shape (H, W), meters, float.
            instance_mask: Mask of object of interest. (H, W), bool.
            category_id: Category of object, represented as an integer.
        """
        pass


class CASSWrapper(MethodWrapper):
    """Wrapper class for CASS."""

    class Config(TypedDict):
        """Configuration dictionary for CASS.

        Attributes:
            model: Path to model.
        """

        model: str

    default_config: Config = {
        "model": None,
    }

    def __init__(self, config: Config, camera: Camera) -> None:
        """Initialize and load CASS model.

        Args:
            config: CASS configuration. See CASSWrapper.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, default_dict=CASSWrapper.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._cass = CASS(
            num_points=config["num_points"], num_obj=config["num_objects"]
        )
        self._num_points = config["num_points"]
        self._cass.load_state_dict(torch.load(config["model"]), strict=True)
        self._cass.to(config["device"])

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_id: int,
    ) -> PredictionDict:
        """See MethodWrapper.inference.

        Based on cass.tools.eval.
        """
        # get bounding box
        valid_mask = (depth_image != 0) * instance_mask
        rmin, rmax, cmin, cmax = get_bbox(valid_mask.numpy())
        bb_mask = torch.zeros_like(depth_image)
        bb_mask[rmin:rmax, cmin:cmax] = 1.0

        # prepare image crop
        color_input = torch.flip(color_image, (2,)).permute([2, 0, 1])  # RGB -> BGR
        color_input = color_input[:, rmin:rmax, cmin:cmax]  # bb crop
        color_input = color_input.unsqueeze(0)  # add batch dim
        color_input = TF.normalize(
            color_input, mean=[0.51, 0.47, 0.44], std=[0.29, 0.27, 0.28]
        )

        # prepare points (fixed number of points, randomly picked)
        point_indices = valid_mask.nonzero()
        if len(point_indices) > self._num_points:
            subset = np.random.choice(
                len(point_indices), replace=False, size=self._num_points
            )
            point_indices = point_indices[subset]
        depth_mask = torch.zeros_like(depth_image)
        depth_mask[point_indices[:, 0], point_indices[:, 1]] = 1.0
        cropped_depth_mask = depth_mask[rmin:rmax, cmin:cmax]
        point_indices_input = cropped_depth_mask.flatten().nonzero()[:, 0]

        # prepare pointcloud
        points = pointset_utils.depth_to_pointcloud(
            depth_image,
            self._camera,
            normalize=False,
            mask=depth_mask,
            convention="opencv",
        )
        if len(points) < self._num_points:
            print(len(points))
            wrap_indices = np.pad(
                np.arange(len(points)), (0, self._num_points - len(points)), mode="wrap"
            )
            points = points[wrap_indices]
            point_indices_input = point_indices_input[wrap_indices]

        # x, y inverted for some reason...
        points[:,0] *= -1
        points[:,1] *= -1
        points = points.unsqueeze(0)
        point_indices_input = point_indices_input.unsqueeze(0)

        # move inputs to device
        color_input = color_input.to(self._device)
        points = points.to(self._device)
        point_indices_input = point_indices_input.to(self._device)

        # TODO: compare inputs to running eval script to make sure they are equal

        # CASS model uses 0-indexed categories, same order as NOCSDataset
        category_index = torch.tensor([category_id - 1], device=self._device)

        folding_encode = self._cass.foldingnet.encode(
            color_input, points, point_indices_input
        )
        posenet_encode = self._cass.estimator.encode(
            color_input, points, point_indices_input
        )
        pred_r, pred_t, pred_c = self._cass.estimator.pose(
            torch.cat([posenet_encode, folding_encode], dim=1), category_index
        )

        # print(pred_r.shape, pred_t.shape, pred_c.shape)
        # # what is pred_c (probably confidence, similar to DenseFusion)
        # reconstructed_points = self._cass.foldingnet.recon(folding_encode)[0]
        # print(pred_t)
        # pointset_utils.visualize_pointset(reconstructed_points[0])
        # exit()
        return {
            "position": torch.tensor([0, 0, 0]),
            "orientation": torch.tensor([0, 0, 0, 1]),
            "extents": torch.tensor([1, 1, 1]),
            "reconstructed_pointcloud": torch.tensor([[0, 0, 0]]),
        }


class NOCSWrapper:
    """Wrapper class for NOCS."""

    def __init__(self, config: dict, camera: Camera) -> None:
        """Initialize and load NOCS model."""
        pass

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_id: int,
    ) -> PredictionDict:
        return {
            "position": torch.tensor([0, 0, 0]),
            "orientation": torch.tensor([0, 0, 0, 1]),
            "extents": torch.tensor([1, 1, 1]),
            "reconstructed_pointcloud": torch.tensor([[0, 0, 0]]),
        }
