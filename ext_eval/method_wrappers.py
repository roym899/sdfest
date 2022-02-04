"""Wrapper for pose and shape estimation methods."""
from abc import ABC
import copy
from typing import Optional, TypedDict

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation

from sdf_differentiable_renderer import Camera
from sdf_single_shot import pointset_utils, quaternion_utils
from sdf_estimation.simple_setup import SDFPipeline

import yoco
from cass.cass.lib.models import CASS
from cass.cass.datasets.dataset import get_bbox as cass_get_bbox
from spd.lib.network import DeformNet
import spd.lib.utils
import spd.lib.align


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
        reconstructed_mesh:
            Reconstructed mesh in object frame.
            None if method does not perform reconstruction.
    """

    position: torch.Tensor
    orientation: torch.Tensor
    extents: torch.Tensor
    reconstructed_pointcloud: Optional[torch.Tensor]
    reconstructed_mesh: Optional[o3d.geometry.TriangleMesh]


class MethodWrapper(ABC):
    """Interface class for pose and shape estimation methods."""

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """Run a method to predict pose and shape of an object.

        Args:
            color_image: The color image, shape (H, W, 3), RGB, 0-1, float.
            depth_image: The depth image, shape (H, W), meters, float.
            instance_mask: Mask of object of interest. (H, W), bool.
            category_str: The category of the object.
        """
        pass


class SPDWrapper(MethodWrapper):
    """Wrapper class for Shape Prior Deformation (SPD)."""

    class Config(TypedDict):
        """Configuration dictionary for SPD.

        Attributes:
            model: Path to model.
            num_categories: Number of categories used by model.
            num_shape_points: Number of points in shape prior.
        """

        model: str
        num_categories: int

    default_config: Config = {
        "model": None,
        "num_categories": None,
        "num_shape_points": None,
    }

    def __init__(self, config: Config, camera: Camera) -> None:
        """Initialize and load SPD model.

        Args:
            config: SPD configuration. See SPDWrapper.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, default_dict=SPDWrapper.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._spd_net = DeformNet(config["num_categories"], config["num_shape_points"])
        self._spd_net.to(self._device)
        self._spd_net.load_state_dict(torch.load(config["model"]))
        self._spd_net.eval()
        self._mean_shape_pointsets = np.load(config["mean_shape_pointsets"])
        self._num_input_points = config["num_input_points"]
        self._image_size = config["image_size"]

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See MethodWrapper.inference.

        Based on spd.evalute.
        """
        category_str_to_id = {
            "bottle": 0,
            "bowl": 1,
            "camera": 2,
            "can": 3,
            "laptop": 4,
            "mug": 5,
        }
        category_id = category_str_to_id[category_str]
        mean_shape_pointset = self._mean_shape_pointsets[category_id]

        # get bounding box
        x1 = min(instance_mask.nonzero()[:, 1]).item()
        y1 = min(instance_mask.nonzero()[:, 0]).item()
        x2 = max(instance_mask.nonzero()[:, 1]).item()
        y2 = max(instance_mask.nonzero()[:, 0]).item()
        rmin, rmax, cmin, cmax = spd.lib.utils.get_bbox([y1, x1, y2, x2])
        bb_mask = torch.zeros_like(depth_image)
        bb_mask[rmin:rmax, cmin:cmax] = 1.0

        valid_mask = (depth_image != 0) * instance_mask

        # prepare image crop
        color_input = color_image[rmin:rmax, cmin:cmax, :].numpy()  # bb crop
        color_input = cv2.resize(
            color_input,
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        color_input = TF.normalize(
            TF.to_tensor(color_input),  # (H, W, C) -> (C, H, W), RGB
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        color_input = color_input.unsqueeze(0)  # add batch dim

        # convert depth to pointcloud
        fx, fy, cx, cy, _ = self._camera.get_pinhole_camera_parameters(pixel_center=0.0)
        width = self._camera.width
        height = self._camera.height
        point_indices = valid_mask[rmin:rmax, cmin:cmax].numpy().flatten().nonzero()[0]
        xmap = np.array([[i for i in range(width)] for _ in range(height)])
        ymap = np.array([[j for _ in range(width)] for j in range(height)])
        if len(point_indices) > self._num_input_points:
            # take subset of points if two many depth points
            point_indices_mask = np.zeros(len(point_indices), dtype=int)
            point_indices_mask[: self._num_input_points] = 1
            np.random.shuffle(point_indices_mask)
            point_indices = point_indices[point_indices_mask.nonzero()]
        else:
            point_indices = np.pad(
                point_indices, (0, self._num_input_points - len(point_indices)), "wrap"
            )  # repeat points if not enough depth observation
        depth_masked = depth_image[rmin:rmax, cmin:cmax].flatten()[point_indices][
            :, None
        ]
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[point_indices][:, None]
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[point_indices][:, None]
        pt2 = depth_masked.numpy()
        pt0 = (xmap_masked - cx) * pt2 / fx
        pt1 = (ymap_masked - cy) * pt2 / fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)
        # adjust indices for resizing of color image
        crop_w = rmax - rmin
        ratio = self._image_size / crop_w
        col_idx = point_indices % crop_w
        row_idx = point_indices // crop_w
        point_indices = (
            np.floor(row_idx * ratio) * self._image_size + np.floor(col_idx * ratio)
        ).astype(np.int64)

        # move inputs to device
        color_input = color_input.to(self._device)
        points = torch.Tensor(points).unsqueeze(0).to(self._device)
        point_indices = torch.LongTensor(point_indices).unsqueeze(0).to(self._device)
        category_id = torch.LongTensor([category_id]).to(self._device)
        mean_shape_pointset = (
            torch.Tensor(mean_shape_pointset).unsqueeze(0).to(self._device)
        )

        # Call SPD network
        assign_matrix, deltas = self._spd_net(
            points, color_input, point_indices, category_id, mean_shape_pointset
        )

        # Postprocess outputs
        inst_shape = mean_shape_pointset + deltas
        assign_matrix = torch.softmax(assign_matrix, dim=2)
        coords = torch.bmm(assign_matrix, inst_shape)  # (1, n_pts, 3)

        point_indices = point_indices[0].cpu().numpy()
        _, point_indices = np.unique(point_indices, return_index=True)
        nocs_coords = coords[0, point_indices, :].detach().cpu().numpy()
        extents = 2 * np.amax(np.abs(inst_shape[0].detach().cpu().numpy()), axis=0)
        points = points[0, point_indices, :].cpu().numpy()
        scale, orientation_m, position, _ = spd.lib.align.estimateSimilarityTransform(
            nocs_coords, points
        )
        orientation_q = torch.Tensor(Rotation.from_matrix(orientation_m).as_quat())

        reconstructed_points = inst_shape[0].detach().cpu() * scale

        # NOCS Object -> ShapeNet Object convention
        obj_fix = torch.tensor(
            [0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]
        )  # CASS object to ShapeNet object
        orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)
        reconstructed_points = quaternion_utils.quaternion_apply(
            quaternion_utils.quaternion_invert(obj_fix),
            reconstructed_points,
        )
        extents, _ = reconstructed_points.abs().max(dim=0)
        extents *= 2.0

        return {
            "position": torch.Tensor(position),
            "orientation": orientation_q,
            "extents": torch.Tensor(extents),
            "reconstructed_pointcloud": reconstructed_points,
            "reconstructed_mesh": None,
        }


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
        self._cass.load_state_dict(torch.load(config["model"]))
        self._cass.to(config["device"])
        self._cass.eval()

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See MethodWrapper.inference.

        Based on cass.tools.eval.
        """
        # get bounding box
        valid_mask = (depth_image != 0) * instance_mask
        rmin, rmax, cmin, cmax = cass_get_bbox(valid_mask.numpy())
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
            wrap_indices = np.pad(
                np.arange(len(points)), (0, self._num_points - len(points)), mode="wrap"
            )
            points = points[wrap_indices]
            point_indices_input = point_indices_input[wrap_indices]

        # x, y inverted for some reason...
        points[:, 0] *= -1
        points[:, 1] *= -1
        points = points.unsqueeze(0)
        point_indices_input = point_indices_input.unsqueeze(0)

        # move inputs to device
        color_input = color_input.to(self._device)
        points = points.to(self._device)
        point_indices_input = point_indices_input.to(self._device)

        category_str_to_id = {
            "bottle": 0,
            "bowl": 1,
            "camera": 2,
            "can": 3,
            "laptop": 4,
            "mug": 5,
        }
        category_id = category_str_to_id[category_str]

        # CASS model uses 0-indexed categories, same order as NOCSDataset
        category_index = torch.tensor([category_id], device=self._device)

        # Call CASS network
        folding_encode = self._cass.foldingnet.encode(
            color_input, points, point_indices_input
        )
        posenet_encode = self._cass.estimator.encode(
            color_input, points, point_indices_input
        )
        pred_r, pred_t, pred_c = self._cass.estimator.pose(
            torch.cat([posenet_encode, folding_encode], dim=1), category_index
        )
        reconstructed_points = self._cass.foldingnet.recon(folding_encode)[0]

        # Postprocess outputs
        reconstructed_points = reconstructed_points.view(-1, 3).cpu()
        pred_c = pred_c.view(1, self._num_points)
        _, max_index = torch.max(pred_c, 1)
        pred_t = pred_t.view(self._num_points, 1, 3)
        orientation_q = pred_r[0][max_index[0]].view(-1).cpu()
        points = points.view(self._num_points, 1, 3)
        position = (points + pred_t)[max_index[0]].view(-1).cpu()
        # output is scalar-first -> scalar-last
        orientation_q = torch.tensor([*orientation_q[1:], orientation_q[0]])

        # Flip x and y axis of position and orientation (undo flipping of points)
        # (x-left, y-up, z-forward) convention -> OpenCV convention
        position[0] *= -1
        position[1] *= -1
        cam_fix = torch.tensor([0.0, 0.0, 1.0, 0.0])
        # NOCS Object -> ShapeNet Object convention
        obj_fix = torch.tensor(
            [0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]
        )  # CASS object to ShapeNet object
        orientation_q = quaternion_utils.quaternion_multiply(cam_fix, orientation_q)
        orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)
        reconstructed_points = quaternion_utils.quaternion_apply(
            quaternion_utils.quaternion_invert(obj_fix),
            reconstructed_points,
        )

        # TODO refinement code from cass.tools.eval? (not mentioned in paper??)

        extents, _ = reconstructed_points.abs().max(dim=0)
        extents *= 2.0

        # pointset_utils.visualize_pointset(reconstructed_points)
        return {
            "position": position.detach(),
            "orientation": orientation_q.detach(),
            "extents": extents.detach(),
            "reconstructed_pointcloud": reconstructed_points.detach(),
            "reconstructed_mesh": None,
        }


class NOCSWrapper:
    """Wrapper class for NOCS."""

    def __init__(self, config: dict, camera: Camera) -> None:
        """Initialize and load NOCS models."""
        pass

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See MethodWrapper.inference."""
        return {
            "position": torch.tensor([0, 0, 0]),
            "orientation": torch.tensor([0, 0, 0, 1]),
            "extents": torch.tensor([1, 1, 1]),
            "reconstructed_pointcloud": torch.tensor([[0, 0, 0]]),
            "reconstructed_mesh": None,
        }


class SDFEstWrapper:
    """Wrapper class for SDFEst."""

    def __init__(self, config: dict, camera: Camera) -> None:
        """Initialize and load SDFEst models."""
        self._pipeline_dict = {}  # maps category to category-specific pipeline
        self._device = config["device"]
        self._visualize_optimization = config["visualize_optimization"]
        self._num_points = config["num_points"]

        # create per-categry models
        for category_str in config["category_configs"].keys():
            category_config = yoco.load_config(
                config["category_configs"][category_str], copy.deepcopy(config)
            )
            self._pipeline_dict[category_str] = SDFPipeline(category_config)
            self._pipeline_dict[category_str].cam = camera

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See MethodWrapper.inference."""
        # skip unsupported category
        if category_str not in self._pipeline_dict:
            return {
                "position": torch.tensor([0, 0, 0]),
                "orientation": torch.tensor([0, 0, 0, 1]),
                "extents": torch.tensor([1, 1, 1]),
                "reconstructed_pointcloud": torch.tensor([[0, 0, 0]]),
                "reconstructed_mesh": None,
            }

        pipeline = self._pipeline_dict[category_str]

        # move inputs to device
        color_image = color_image.to(self._device)
        depth_image = depth_image.to(self._device, copy=True)
        instance_mask = instance_mask.to(self._device)

        position, orientation, scale, shape = pipeline(
            depth_image,
            instance_mask,
            color_image,
            visualize=self._visualize_optimization,
        )

        # outputs of SDFEst are OpenGL camera, ShapeNet object convention
        position_cv = pointset_utils.change_position_camera_convention(
            position[0], "opengl", "opencv"
        )
        orientation_cv = pointset_utils.change_orientation_camera_convention(
            orientation[0], "opengl", "opencv"
        )

        # reconstruction + extent
        mesh = pipeline.generate_mesh(shape, scale, True).get_transformed_o3d_geometry()
        reconstructed_points = torch.from_numpy(
            np.asarray(mesh.sample_points_uniformly(self._num_points).points)
        )
        extents, _ = reconstructed_points.abs().max(dim=0)
        extents *= 2.0

        return {
            "position": position_cv.detach().cpu(),
            "orientation": orientation_cv.detach().cpu(),
            "extents": extents,
            "reconstructed_pointcloud": reconstructed_points,
            "reconstructed_mesh": mesh,
        }
