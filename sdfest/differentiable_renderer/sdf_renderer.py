"""PyTorch interface for diffferentiable renderer.

This module provides two functions:
    - render_depth: numpy-based CPU implementation
    (not recommended, only for development)
    - render_depth_gpu: CUDA implementation (fast)
"""
import math
import os

import numpy as np
import open3d as o3d
import torch
from torch.utils.cpp_extension import load
from typing import Optional, Tuple

from .simple_renderer import render_depth as _render_depth
from .simple_renderer import SDFObject

# TODO: allow JIT compilation by pip argument??
directory = os.path.dirname(__file__)
sdf_renderer_cpp = load(
    name="sdf_renderer_cpp",
    sources=[
        os.path.join(directory, "./csrc/sdf_renderer.cpp"),
        os.path.join(directory, "./csrc/sdf_renderer_cuda.cu"),
    ],
)


class Camera:
    """Pinhole camera parameters.

    This class allows conversion between different pixel conventions, i.e., pixel
    center at (0.5, 0.5) (as common in computer graphics), and (0, 0) as common in
    computer vision.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        s: float = 0.0,
        pixel_center: float = 0.0,
    ):
        """Initialize camera parameters.

        Note that the principal point is only fully defined in combination with
        pixel_center.

        The pixel_center defines the relation between continuous image plane
        coordinates and discrete pixel coordinates.

        A discrete image coordinate (x, y) will correspond to the continuous
        image coordinate (x + pixel_center, y + pixel_center). Normally pixel_center
        will be either 0 or 0.5. During calibration it depends on the convention
        the point features used to compute the calibration matrix.

        Note that if pixel_center == 0, the corresponding continuous coordinate
        interval for a pixel are [x-0.5, x+0.5). I.e., proper rounding has to be done
        to convert from continuous coordinate to the corresponding discrete coordinate.

        For pixel_center == 0.5, the corresponding continuous coordinate interval for a
        pixel are [x, x+1). I.e., floor is sufficient to convert from continuous
        coordinate to the corresponding discrete coordinate.

        Args:
            width: Number of pixels in horizontal direction.
            height: Number of pixels in vertical direction.
            fx: Horizontal focal length.
            fy: Vertical focal length.
            cx: Principal point x-coordinate.
            cy: Principal point y-coordinate.
            s: Skew.
            pixel_center: The center offset for the provided principal point.
        """
        # focal length
        self.fx = fx
        self.fy = fy

        # principal point
        self.cx = cx
        self.cy = cy

        self.pixel_center = pixel_center

        # skew
        self.s = s

        # image dimensions
        self.width = width
        self.height = height

    def get_o3d_pinhole_camera_parameters(self) -> o3d.camera.PinholeCameraParameters():
        """Convert camera to Open3D pinhole camera parameters.

        Open3D camera is at (0,0,0) looking along positive z axis (i.e., positive z
        values are in front of camera). Open3D expects camera with pixel_center = 0
        and does not support skew.

        Returns:
            The pinhole camera parameters.
        """
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(0)
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)
        params.extrinsic = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        return params

    def get_pinhole_camera_parameters(self, pixel_center: float) -> Tuple:
        """Convert camera to general camera parameters.

        Args:
            pixel_center:
                At which ratio of a square the pixel center should be for the resulting
                parameters. Typically 0 or 0.5. See class documentation for more info.
        Returns:
            - fx, fy: The horizontal and vertical focal length
            - cx, cy:
                The position of the principal point in continuous image plane
                coordinates considering the provided pixel center and the pixel center
                specified during the construction.
            - s: The skew.
        """
        cx_corrected = self.cx - self.pixel_center + pixel_center
        cy_corrected = self.cy - self.pixel_center + pixel_center
        return self.fx, self.fy, cx_corrected, cy_corrected, self.s


class SDFRendererFunction(torch.autograd.Function):
    """Renderer function for signed distance fields."""

    @staticmethod
    def forward(
        ctx,
        sdf: torch.Tensor,
        position: torch.Tensor,
        orientation: torch.Tensor,
        inv_scale: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fov_deg: Optional[float] = None,
        threshold: Optional[float] = 0.0,
        camera: Optional[Camera] = None,
    ) -> torch.Tensor:
        """Render depth image of a 7-DOF discrete signed distance field.

        The SDF position is assumed to be in the camera frame under OpenGL convention.

        That is, camera looks along negative z-axis, y pointing upwards and x to the
        right. Note that the rendered image will still follow the classical computer
        vision convention, of first row being up in the camera frame.

        This function internally usese numpy and is very slow due to the fully serial
        implementation. This is only for testing purposes. Use the GPU version for
        practical performance.

        Camera can be specified either via camera parameter giving most
        flexbility or alternatively by providing width, height and fov_deg.

        Args:
            ctx:
                Context object to stash information.
                See https://pytorch.org/docs/stable/notes/extending.html.
            sdf:
                Discrete signed distance field with shape (M, M, M).
                Arbitrary (but uniform) resolutions are supported.
            position:
                The position of the signed distance field origin in the camera frame.
            orientation:
                The orientation of the SDF as a normalized quaternion.
            inv_scale:
                The inverted scale of the SDF. The scale of an SDF the half-width of the
                full SDF volume.
            width:
                Number of pixels in x direction. Recommended to use camera instead.
            height:
                Number of pixels in y direction. Recommended to use camera instead.
            fov_deg:
                The horizontal field of view (i.e., in x direction).
                Pixels are assumed to be square, i.e., fx=fy, computed based on width
                and fov_deg.
                Recommended to use camera instead.
            threshold:
                The distance threshold at which sphere tracing should be stopped.
                Smaller value will be more accurate, but slower and might potentially
                lead to holes in the rendering for thin structures in the SDF.
                Larger values will be faster, but will overestimate the thickness.

                Should always be positive to guarantee convergence.
            camera:
                Camera parameters (not supported right now).
        Returns:
            The rendered depth image.
        """
        if None not in [width, height, fov_deg] and camera is not None:
            raise ValueError("Either width+height+fov_dev or camera must be provided.")
        if camera is not None:
            raise NotImplementedError(
                "Only width+height+fov_dev currently supported for CPU"
            )
        # for simplicity use numpy internally
        ctx.save_for_backward(sdf, position, orientation, inv_scale)
        sdf = sdf.detach().numpy()
        position = position.detach().numpy()
        orientation = orientation.detach().numpy()
        inv_scale = inv_scale.detach().numpy()
        sdf_object = SDFObject(sdf)
        image, derivatives = _render_depth(
            sdf_object,
            width,
            height,
            fov_deg,
            "d",
            threshold,
            position,
            orientation,
            inv_scale,
        )
        ctx.derivatives = derivatives
        return torch.from_numpy(image)

    @staticmethod
    def backward(ctx, inp: torch.Tensor):
        """Compute gradients of inputs with respect to the provided gradients.

        Normally called by PyTorch as part of a call to backward() on a loss.

        Args:
            grad_depth_image:
        Returns:
            Gradients of
                discretized signed distance field, position, orientation, inverted scale
                followed by None for all the non-supported variables passed to forward.
        """
        derivatives = ctx.derivatives
        sdf, pos, quat, inv_s = ctx.saved_tensors
        g_image = inp.numpy()
        g_sdf = g_p = g_q = g_is = g_w = g_h = g_fov = g_thresh = g_camera = None
        g_sdf = torch.zeros_like(sdf)
        g_p = torch.empty_like(pos)
        g_q = torch.empty_like(quat)
        g_is = torch.empty_like(inv_s)
        g_p[0] = torch.tensor(np.sum(derivatives["x"] * g_image))
        g_p[1] = torch.tensor(np.sum(derivatives["y"] * g_image))
        g_p[2] = torch.tensor(np.sum(derivatives["z"] * g_image))
        g_q[0] = torch.tensor(np.sum(derivatives["qx"] * g_image))
        g_q[1] = torch.tensor(np.sum(derivatives["qy"] * g_image))
        g_q[2] = torch.tensor(np.sum(derivatives["qz"] * g_image))
        g_q[3] = torch.tensor(np.sum(derivatives["qw"] * g_image))
        g_is = torch.tensor(np.sum(derivatives["s_inv"] * g_image))
        if "sdf" in derivatives:
            for k, v in derivatives["sdf"].items():
                g_sdf[k] = torch.tensor(np.sum(v * g_image))
        return g_sdf, g_p, g_q, g_is, g_w, g_h, g_fov, g_thresh, g_camera


render_depth = SDFRendererFunction.apply


class SDFRendererFunctionGPU(torch.autograd.Function):
    """Renderer function for signed distance fields."""

    @staticmethod
    def forward(
        ctx,
        sdf: torch.Tensor,
        position: torch.Tensor,
        orientation: torch.Tensor,
        inv_scale: torch.Tensor,
        threshold: Optional[float] = 0.0,
        camera: Optional[Camera] = None,
    ) -> torch.Tensor:
        """Render depth image of a 7-DOF discrete signed distance field on the GPU.

        Also see render_depth_gpu for documentation.

        Args:
            ctx:
                Context object to stash information.
                See https://pytorch.org/docs/stable/notes/extending.html.
            sdf:
                Discrete signed distance field with shape (M, M, M).
                Arbitrary (but uniform) resolutions are supported.
            position:
                The position of the signed distance field origin in the camera frame.
            orientation:
                The orientation of the SDF as a normalized quaternion.
            inv_scale:
                The inverted scale of the SDF. The scale of an SDF the half-width of the
                full SDF volume.
            threshold:
                The distance threshold at which sphere tracing should be stopped.
                Smaller value will be more accurate, but slower and might potentially
                lead to holes in the rendering for thin structures in the SDF.
                Larger values will be faster, but will overestimate the thickness.

                Should always be positive to guarantee convergence.
            camera:
                Camera parameters.
        Returns:
            The rendered depth image.
        """
        fx, fy, cx, cy, _ = camera.get_pinhole_camera_parameters(0.5)
        (image,) = sdf_renderer_cpp.forward(
            sdf,
            position,
            orientation,
            inv_scale,
            camera.width,
            camera.height,
            cx,
            cy,
            fx,
            fy,
            threshold,
        )
        ctx.save_for_backward(image, sdf, position, orientation, inv_scale)
        ctx.width = camera.width
        ctx.height = camera.height
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy
        return image

    @staticmethod
    def backward(ctx, grad_depth_image: torch.Tensor):
        """Compute gradients of inputs with respect to the provided gradients.

        Normally called by PyTorch as part of a call to backward() on a loss.

        Args:
            grad_depth_image:
        Returns:
            Gradients of
                discretized signed distance field, position, orientation, inverted scale
                followed by None for all the non-supported variables passed to forward.
        """
        g_sdf = g_p = g_q = g_is = g_thresh = g_camera = None
        g_sdf, g_p, g_q, g_is = sdf_renderer_cpp.backward(
            grad_depth_image,
            *ctx.saved_tensors,
            ctx.width,
            ctx.height,
            ctx.cx,
            ctx.cy,
            ctx.fx,
            ctx.fy
        )
        return g_sdf, g_p, g_q, g_is, g_thresh, g_camera


def render_depth_gpu(
    sdf: torch.Tensor,
    position: torch.Tensor,
    orientation: torch.Tensor,
    inv_scale: torch.Tensor,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fov_deg: Optional[float] = None,
    threshold: Optional[float] = 0.0,
    camera: Optional[Camera] = None,
):
    """Render depth image of a 7-DOF discrete signed distance field on the GPU.

    The SDF position is assumed to be in the camera frame under OpenGL convention.

    That is, camera looks along negative z-axis, y pointing upwards and x to the
    right. Note that the rendered image will still follow the classical computer
    vision convention, of first row being up in the camera frame.

    Camera can be specified either via camera parameter giving most
    flexbility or alternatively by providing width, height and fov_deg.

    All provided tensors must reside on the GPU.

    Args:
        sdf:
            Discrete signed distance field with shape (M, M, M).
            Arbitrary (but uniform) resolutions are supported.
        position:
            The position of the signed distance field origin in the camera frame.
        orientation:
            The orientation of the SDF as a normalized quaternion.
        inv_scale:
            The inverted scale of the SDF. The scale of an SDF the half-width of the
            full SDF volume.
        width:
            Number of pixels in x direction. Recommended to use camera instead.
        height:
            Number of pixels in y direction. Recommended to use camera instead.
        fov_deg:
            The horizontal field of view (i.e., in x direction).
            Pixels are assumed to be square, i.e., fx=fy, computed based on width
            and fov_deg.
            Recommended to use camera instead.
        threshold:
            The distance threshold at which sphere tracing should be stopped.
            Smaller value will be more accurate, but slower and might potentially
            lead to holes in the rendering for thin structures in the SDF.
            Larger values will be faster, but will overestimate the thickness.

            Should always be positive to guarantee convergence.
        camera:
            Camera parameters.
    Returns:
        The rendered depth image.
    """
    if None not in [width, height, fov_deg] and camera is not None:
        raise ValueError("Either width+height+fov_dev or camera must be provided.")
    if camera is None:
        f = width / math.tan(fov_deg * math.pi / 180.0 / 2.0) / 2
        camera = Camera(width, height, f, f, width / 2, height / 2, pixel_center=0.5)

    return SDFRendererFunctionGPU.apply(
        sdf, position, orientation, inv_scale, threshold, camera
    )
