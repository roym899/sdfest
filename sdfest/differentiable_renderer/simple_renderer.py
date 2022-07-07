"""Experimental differentiable RGBD renderer.

This module is not optimized for performance.
"""
from typing import List, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import time
from functools import partial
from collections import defaultdict


class Ray:
    """A single ray defined by its origin and direction."""

    def __init__(self, origin: np.array, direction: np.array, row: int, col: int):
        """Construct a single ray.

        Args:
            origin: The origin of the ray.
            direction: The direction of the ray. Will be normalized.
            row: The row of the pixel this ray corresponds to.
            col: The col of the pixel this ray corresponds to.
        """
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)
        self.row = row
        self.col = col

    def z_from_t(self, t: float):
        """Compute the z value in camera coordinates for a given t value.

        Note: the origin has no influence on this value.

        Args:
            t: The Euclidean distance from the origin.

        Returns:
            Z coordinate at distance t.
        """
        return t * self.direction[2]


class SDFObject:
    """An object represented as a voxelized signed distance field."""

    def __init__(self, sdf: np.array):
        """Initialize SDF based object.

        Args:
            sdf: The array containing the signed distance to the closest surface.
        """
        self._sdf = sdf
        self._grid_size = 2.0 / (sdf.shape[0] - 1)
        self._grid_size_inv = 1.0 / self._grid_size
        self._resolution = sdf.shape[0]
        # bounding box center
        self._ac = np.array([0, 0, 0, 1])
        # bounding box axes
        self._a = {}
        self._a["x"] = np.array([1, 0, 0])
        self._a["y"] = np.array([0, 1, 0])
        self._a["z"] = np.array([0, 0, 1])
        # nominal bounding box half-lengths (i.e., before scaling)
        self._h = {}
        self._h["x"] = 1
        self._h["y"] = 1
        self._h["z"] = 1

    def find_obb_ray_t(
        self, ray: Ray, o2w: np.array, scale: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find t value for which a ray hits the oriented bounding box.

        This follows ray/box intersection algorithm described in Real-Time Rendering,
        Akenine-Möller, 2018.

        Returns:
            tuple(None, None) if there is no intersection.
            tuple(t_min, t_max) if there is an intersection. If the ray starts inside
            the obb t_min will be 0. t_max indicates the t value for which the ray
            exits the obb.
        """
        t_min = float("-inf")
        t_max = float("inf")

        # transform object
        a_c = (o2w @ self._ac)[0:3]  # rotate and translate
        a = {
            k: o2w[0:3, 0:3] @ v for k, v in self._a.items()
        }  # only rotate the axis vectors

        # ?
        p = a_c - ray.origin[0:3]

        for ax in self._a.keys():
            e = a[ax] @ p
            f = a[ax] @ ray.direction
            if (
                abs(f) > 1e-20
            ):  # ray is perpendicular to slab normal (i.e., parallel to slab)
                t_1 = (e + self._h[ax] * scale) / f
                t_2 = (e - self._h[ax] * scale) / f
                if t_1 > t_2:
                    t_1, t_2 = t_2, t_1
                t_min = max(t_min, t_1)
                t_max = min(t_max, t_2)
                if t_min > t_max or t_max < 0:
                    return None, None
            elif (
                -e > self._h[ax] * scale or -e < -self._h[ax] * scale
            ):  # if parallel, check if outside
                return None, None

        if t_min > 0:
            return t_min, t_max
        return 0, t_max

    def sphere_trace(
        self,
        ray: Ray,
        t_start: float,
        t_end: float,
        threshold: float,
        w2o: np.array,
        inv_scale: float,
    ) -> Optional[float]:
        """Find intersection of ray with SDF using sphere tracing.

        Args:
            ray: Ray in world coordinates.
            t_start: Minimum t value for ray to intersect with obb.
            t_end: Maximum t value for ray to intersect with obb.
            w2o: Matrix to transform world coordinates to object coordinates.

        Returns:
            None if no intersection is found.
            t value at the intersection otherwise.
        """
        origin = w2o @ ray.origin
        direction = w2o[0:3, 0:3] @ ray.direction
        count = 0
        scale = 1.0 / inv_scale

        t = t_start
        while t < t_end:
            point = origin[0:3] + t * direction
            count += 1
            val = self.trilinear(point, inv_scale) * scale
            # print(f"{ray.row} {ray.col} {t} {val}")
            if val < threshold * t:
                return t, count
            else:
                t += val
        return None, count

    def point_2_index(self, point: np.array) -> Tuple[int]:
        """Convert a 3D point to voxel indices.

        The point is assumed to be within [-1, 1) and the cell is the first index
        such that index * self._grid_size + 0.5 > point.
        """
        return tuple(
            max(
                0,
                min(
                    self._resolution - 2, int((x + 1.0) * (self._resolution - 1) / 2.0)
                ),
            )
            for x in point
        )

    def index_2_point(self, index: Tuple[int]) -> np.array:
        """Convert voxel index to 3D point.

        Args:
            index: The 3-tuple index to convert.

        Returns:
            The 3D point assuming a 2x2x2 sdf volume with (0,0,0) at the origin.
        """
        return np.array([i * self._grid_size - 1.0 for i in index])

    def trilinear(self, point: np.array, inv_scale: float):
        """Trilinear interpolation of the signed distance field at a local 3D point.

        Args:
            point: The 3D point in object coordinates to query.
            inv_scale: The inverted scale of the object.

        Returns:
            The distance to the closest surface, computed by trilinear interpolation.
        """
        point *= inv_scale
        base_index = self.point_2_index(point)
        n_indices = [
            tuple(i + j for i, j in zip(base_index, x))
            for x in [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        ]
        pos_0 = self.index_2_point(base_index)
        off = (point - pos_0) * self._grid_size_inv
        c00 = self._sdf[n_indices[0]] * (1 - off[0]) + self._sdf[n_indices[4]] * off[0]
        c01 = self._sdf[n_indices[1]] * (1 - off[0]) + self._sdf[n_indices[5]] * off[0]
        c10 = self._sdf[n_indices[2]] * (1 - off[0]) + self._sdf[n_indices[6]] * off[0]
        c11 = self._sdf[n_indices[3]] * (1 - off[0]) + self._sdf[n_indices[7]] * off[0]
        c0 = c00 * (1 - off[1]) + c10 * off[1]
        c1 = c01 * (1 - off[1]) + c11 * off[1]

        return c0 * (1 - off[2]) + c1 * off[2]


def generate_rays(width: int, height: int, fov_deg: float, c2w: np.array) -> List[Ray]:
    """Generate rays in world coordinates.

    The logic mostly follows scratchapixel's Generating Camera Rays chapter.

    Args:
        width: The width of the image.
        height: The height of the image.
        fov_deg: The field of the view of the camera.
        c2w: Camera-to-world transform matrix with shape (4,4).
    Returns:
        The list of rays each representing one pixel of the image.
    """
    rays = []
    aspect_ratio = width / height
    fov_rad = fov_deg * np.pi / 180
    origin = c2w @ np.array([0, 0, 0, 1])
    for row in range(0, height):
        for col in range(0, width):
            ndc_x = (col + 0.5) / width
            ndc_y = (row + 0.5) / height
            screen_x = (2 * ndc_x - 1) * np.tan(fov_rad / 2)
            screen_y = (1 - 2 * ndc_y) * np.tan(fov_rad / 2) / aspect_ratio
            pixel_pos_camera = np.array([screen_x, screen_y, -1, 1])
            pixel_pos_world = c2w @ pixel_pos_camera
            direction = pixel_pos_world[0:3] - origin[0:3]
            rays.append(Ray(origin=origin, direction=direction, row=row, col=col))

    return rays


def render_depth(
    sdf_object: SDFObject,
    width: int,
    height: int,
    fov_deg: float,
    value: str,
    threshold: float,
    p_o2w: np.array,
    q_o2w: np.array,
    inv_scale: float,
) -> np.array:
    """Render depth image of the SDF."""
    start_time = time.time()
    derivative_time = 0
    # fmt: off
    c2w = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float)
    o2w = np.array([[1, 0, 0, p_o2w[0]],
                    [0, 1, 0, p_o2w[1]],
                    [0, 0, 1, p_o2w[2]],
                    [0, 0, 0, 1]], dtype=np.float)
    o2w[0:3, 0:3] = Rotation.from_quat(q_o2w).as_matrix()
    # fmt: on

    rays = generate_rays(width, height, fov_deg, c2w)
    image = np.zeros(shape=(height, width))

    derivatives = defaultdict(partial(np.zeros_like, image))

    scale = 1.0 / inv_scale

    for ray in rays:
        t_min, t_max = sdf_object.find_obb_ray_t(ray, o2w, scale)
        # print(ray.direction, ray.row, ray.col, t_min, t_max)
        if t_min is not None:
            # print(f"{t_min} {t_max}, {ray.row} {ray.col}")
            t, c = sdf_object.sphere_trace(
                ray, t_min, t_max, threshold, np.linalg.inv(o2w), inv_scale
            )

            if t is None:
                continue
            elif value == "t":
                image[ray.row, ray.col] = t
            elif value == "z":
                image[ray.row, ray.col] = ray.z_from_t(t)
            elif value == "c":
                image[ray.row, ray.col] = c
            elif value == "d":
                image[ray.row, ray.col] = abs(ray.z_from_t(t))
                derivative_start_time = time.time()
                compute_z_derivatives(
                    derivatives, ray, t, p_o2w, q_o2w, inv_scale, sdf_object
                )
                derivative_time += time.time() - derivative_start_time
    print(
        f"Render time: {time.time()-start_time-derivative_time}"
        f"Derivative time: {derivative_time}"
    )
    return image, derivatives


def compute_z_derivatives(
    derivatives: Dict,
    ray: Ray,
    t: float,
    p_o2w: np.array,
    q_o2w: np.array,
    inv_scale: float,
    sdf_object: SDFObject,
):
    """Compute derivatives of depth value of a single pixel."""
    # see notes for derivation
    # notation: frame component var index
    x = ray.origin[0:3] + t * ray.direction
    scale = 1.0 / inv_scale

    wxx, wyx, wzx = x
    wxxo, wyxo, wzxo = p_o2w
    wxqo, wyqo, wzqo, wwqo = q_o2w
    dx, dy, dz = wxx - wxxo, wyx - wyxo, wzx - wzxo
    oxx = (
        (1 - 2 * (wyqo ** 2 + wzqo ** 2)) * dx
        + 2 * (wxqo * wyqo + wwqo * wzqo) * dy
        + 2 * (wxqo * wzqo - wwqo * wyqo) * dz
    )
    oyx = (
        2 * (wxqo * wyqo - wwqo * wzqo) * dx
        + (1 - 2 * (wxqo ** 2 + wzqo ** 2)) * dy
        + 2 * (wyqo * wzqo + wwqo * wxqo) * dz
    )
    ozx = (
        2 * (wxqo * wzqo + wwqo * wyqo) * dx
        + 2 * (wyqo * wzqo - wwqo * wxqo) * dy
        + (1 - 2 * (wxqo ** 2 + wyqo ** 2)) * dz
    )

    # retrieve cell where point terminated in
    base_index = sdf_object.point_2_index(
        np.array([oxx * inv_scale, oyx * inv_scale, ozx * inv_scale])
    )
    oxxnc, oyxnc, ozxnc = sdf_object.index_2_point(base_index)
    cxx = (-oxxnc + oxx * inv_scale) * sdf_object._grid_size_inv
    cyx = (-oyxnc + oyx * inv_scale) * sdf_object._grid_size_inv
    czx = (-ozxnc + ozx * inv_scale) * sdf_object._grid_size_inv

    n_indices = [
        tuple(i + j for i, j in zip(base_index, x))
        for x in [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    ]
    c000 = sdf_object._sdf[n_indices[0]]
    c001 = sdf_object._sdf[n_indices[1]]
    c010 = sdf_object._sdf[n_indices[2]]
    c011 = sdf_object._sdf[n_indices[3]]
    c100 = sdf_object._sdf[n_indices[4]]
    c101 = sdf_object._sdf[n_indices[5]]
    c110 = sdf_object._sdf[n_indices[6]]
    c111 = sdf_object._sdf[n_indices[7]]

    c00 = c000 * (1 - cxx) + c100 * cxx
    c01 = c001 * (1 - cxx) + c101 * cxx
    c10 = c010 * (1 - cxx) + c110 * cxx
    c11 = c011 * (1 - cxx) + c111 * cxx

    c0 = c00 * (1 - cyx) + c10 * cyx
    c1 = c01 * (1 - cyx) + c11 * cyx

    t_diff = c0 * (1 - czx) + c1 * czx

    # now compute the derivatives
    # most of what is done before, could have been cached also
    # derivatives with respect to sdf values
    if "sdf" not in derivatives:
        derivatives["sdf"] = defaultdict(derivatives.default_factory)
    f = scale * abs(ray.direction[2])
    derivatives["sdf"][n_indices[0]][ray.row, ray.col] = (
        (1 - cxx) * (1 - cyx) * (1 - czx)
    ) * f
    derivatives["sdf"][n_indices[1]][ray.row, ray.col] = (1 - cxx) * (1 - cyx) * czx * f
    derivatives["sdf"][n_indices[2]][ray.row, ray.col] = (1 - cxx) * cyx * (1 - czx) * f
    derivatives["sdf"][n_indices[3]][ray.row, ray.col] = (1 - cxx) * cyx * czx * f
    derivatives["sdf"][n_indices[4]][ray.row, ray.col] = cxx * (1 - cyx) * (1 - czx) * f
    derivatives["sdf"][n_indices[5]][ray.row, ray.col] = cxx * (1 - cyx) * czx * f
    derivatives["sdf"][n_indices[6]][ray.row, ray.col] = cxx * cyx * (1 - czx) * f
    derivatives["sdf"][n_indices[7]][ray.row, ray.col] = cxx * cyx * czx * f

    if ray.row == 20 and ray.col == 20:
        # print("Test: ", ray.direction[0], t, wxx, oxx, c000)
        print("Test: ", oxxnc, cxx)

    # derivatives with respect to object params
    s = inv_scale * sdf_object._grid_size_inv
    dcxx, dcyx, dczx = {}, {}, {}
    dcxx["x"] = (2 * (wyqo ** 2 + wzqo ** 2) - 1) * s
    dcyx["x"] = 2 * (wwqo * wzqo - wxqo * wyqo) * s
    dczx["x"] = -2 * (wxqo * wzqo + wwqo * wyqo) * s
    dcxx["y"] = -2 * (wxqo * wyqo + wwqo * wzqo) * s
    dcyx["y"] = (2 * (wxqo ** 2 + wzqo ** 2) - 1) * s
    dczx["y"] = 2 * (wwqo * wxqo - wyqo * wzqo) * s
    dcxx["z"] = 2 * (wwqo * wyqo - wxqo * wzqo) * s
    dcyx["z"] = -2 * (wyqo * wzqo + wwqo * wxqo) * s
    dczx["z"] = (2 * (wxqo ** 2 + wyqo ** 2) - 1) * s
    dcxx["qw"] = (2 * wwqo * dx + 2 * wzqo * dy - 2 * wyqo * dz - 2 * wwqo * oxx) * s
    dcyx["qw"] = (-2 * wzqo * dx + 2 * wwqo * dy + 2 * wxqo * dz - 2 * wwqo * oyx) * s
    dczx["qw"] = (2 * wyqo * dx - 2 * wxqo * dy + 2 * wwqo * dz - 2 * wwqo * ozx) * s
    dcxx["qx"] = (2 * wxqo * dx + 2 * wyqo * dy + 2 * wzqo * dz - 2 * wxqo * oxx) * s
    dcyx["qx"] = (2 * wyqo * dx - 2 * wxqo * dy + 2 * wwqo * dz - 2 * wxqo * oyx) * s
    dczx["qx"] = (2 * wzqo * dx - 2 * wwqo * dy - 2 * wxqo * dz - 2 * wxqo * ozx) * s
    dcxx["qy"] = (-2 * wyqo * dx + 2 * wxqo * dy - 2 * wwqo * dz - 2 * wyqo * oxx) * s
    dcyx["qy"] = (2 * wxqo * dx + 2 * wyqo * dy + 2 * wzqo * dz - 2 * wyqo * oyx) * s
    dczx["qy"] = (2 * wwqo * dx + 2 * wzqo * dy - 2 * wyqo * dz - 2 * wyqo * ozx) * s
    dcxx["qz"] = (-2 * wzqo * dx + 2 * wwqo * dy + 2 * wxqo * dz - 2 * wzqo * oxx) * s
    dcyx["qz"] = (-2 * wwqo * dx - 2 * wzqo * dy + 2 * wyqo * dz - 2 * wzqo * oyx) * s
    dczx["qz"] = (2 * wxqo * dx + 2 * wyqo * dy + 2 * wzqo * dz - 2 * wzqo * ozx) * s
    dcxx["s_inv"] = oxx * sdf_object._grid_size_inv
    dcyx["s_inv"] = oyx * sdf_object._grid_size_inv
    dczx["s_inv"] = ozx * sdf_object._grid_size_inv

    for k in dcxx.keys():
        dc00 = -c000 * dcxx[k] + c100 * dcxx[k]
        dc01 = -c001 * dcxx[k] + c101 * dcxx[k]
        dc10 = -c010 * dcxx[k] + c110 * dcxx[k]
        dc11 = -c011 * dcxx[k] + c111 * dcxx[k]

        dc0 = dc00 * (1 - cyx) - c00 * dcyx[k] + dc10 * cyx + c10 * dcyx[k]
        dc1 = dc01 * (1 - cyx) - c01 * dcyx[k] + dc11 * cyx + c11 * dcyx[k]

        dtdiff = dc0 * (1 - czx) - c0 * dczx[k] + dc1 * czx + c1 * dczx[k]

        derivatives[k][ray.row, ray.col] = scale * dtdiff * abs(ray.direction[2])

    # product rule for scale needed additionally
    derivatives["s_inv"][ray.row, ray.col] -= (t_diff * scale ** 2) * abs(
        ray.direction[2]
    )


def plot_derivative(derivative: np.array, cap: Optional[float] = None) -> float:
    """Plot derivative image.

    Optionally the color scale can be capped to allow multiple plots with consistent
    color scales.

    Args:
        derivative:
            HxW array containing derivative of pixel with respect to some parameter.
        cap:
            The absolute value used for the negative and positive limit of the color
            scale.
    Returns:
        The utilized maximum absolute value.
    """
    plt.figure()
    if cap is None:
        cap = abs(max(derivative.min(), derivative.max(), key=abs))
    plt.imshow(derivative, cmap="seismic", vmin=-cap, vmax=cap)
    return cap


# TODO: add click
if __name__ == "__main__":
    object = SDFObject(sdf=np.load("./data/00001.npy"))

    p_o2w = np.array([0, 0, -2.0])
    q_o2w = Rotation.from_euler("XYZ", [0, 120, 0], True).as_quat()
    inv_scale = 1.0 / 1.0
    res = 100
    image, derivatives = render_depth(
        object, res, res, 90, "d", 0.001, p_o2w, q_o2w, inv_scale=inv_scale
    )
    delta = 1e-10
    image2, _ = render_depth(
        object,
        res,
        res,
        90,
        "d",
        0.001,
        p_o2w + np.array([0, 0, 0]),
        q_o2w + np.array([0, 0, delta, 0]),
        inv_scale=inv_scale,
    )
    num_d = (image2 - image) / delta
    plt.figure()
    plt.imshow(image, cmap="gist_heat_r")
    cap = plot_derivative(derivatives["qz"])
    plot_derivative(num_d, cap)
    plt.figure()
    plt.imshow(num_d - derivatives["qz"])
    # plot_derivative(derivatives["y"])
    # plot_derivative(derivatives["z"])
    # plot_derivative(derivatives["qw"])
    # plot_derivative(derivatives["qx"])
    # plot_derivative(derivatives["qy"])
    # plot_derivative(derivatives["qz"])
    # plot_derivative(list(derivatives["sdf"].values())[500])
    plt.show()
