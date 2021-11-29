"""This module provides utility functions for working with SDF volumes."""
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from skimage.measure import marching_cubes
import torch
import trimesh
from trimesh import Trimesh
from trimesh.visual.material import SimpleMaterial
import mesh_to_sdf as mts
import pyrender  # has to be reported after mesh_to_sdf
from pyrender.constants import RenderFlags


def mesh_to_sdf(mesh: Trimesh, cells_per_dim: int, padding: Optional[int] = 0):
    """Convert mesh to discretized signed distance field.

    The mesh will be stretched, so that its longest extend fills out the unit cube
    leaving the specified padding empty.

    Args:
        mesh: The mesh to convert.
        cells_per_dim: The number of cells along each dimension.
        padding: Number of empty space cells.
    Returns:
        The discretized signed distance field.
    """
    surface_point_method = "scan"
    sign_method = "depth"
    scaled_mesh = mts.utils.scale_to_unit_cube(mesh)
    scaled_mesh.vertices *= (cells_per_dim - 2 * padding) / cells_per_dim
    surface_point_cloud = mts.get_surface_point_cloud(
        scaled_mesh, surface_point_method, calculate_normals=sign_method == "normal"
    )
    try:
        return surface_point_cloud.get_voxels(
            cells_per_dim, check_result=True, use_depth_buffer=sign_method == "depth"
        )
    except mts.BadMeshException:
        print("Bad mesh detected. Skipping.")
        return None


def mesh_from_sdf(
    sdf_volume: np.array, level: Optional[float] = 0, complete_mesh: bool = False
) -> Trimesh:
    """Compute mesh from sdf using marching cubes algorithm.

    Args:
        sdf_volume: the SDF volume to convert, shape (M, M, M)
        level: the isosurface level to extract the mesh for
        complete_mesh:
            if True, the SDF will be padded with positive values prior to converting it
            to a mesh. This ensures a watertight mesh is created.
    Returns:
        The resulting mesh.
    """
    try:
        if complete_mesh:
            sdf_volume = np.pad(sdf_volume, pad_width=1, constant_values=1.0)
        sdf_volume.shape
        vertices, faces, normals, _ = marching_cubes(
            sdf_volume, spacing=2 / np.array(sdf_volume.shape), level=level
        )
        vertices -= 1
    except ValueError:
        return None
    return Trimesh(
        vertices,
        faces,
        vertex_normals=normals,
        visual=trimesh.visual.TextureVisuals(material=SimpleMaterial()),
    )


def plot_mesh(
    mesh: Trimesh,
    polar_angle=np.pi / 4,
    azimuth=0,
    camera_distance=2.5,
    plot_object: Optional[Axes] = None,
    transform: Optional[np.array] = None,
):
    """Render a mesh with camera pointing at its center.

    Note that in pyrender z-axis is up, x,y form the polar_angle=0 plane.

    Args:
        mesh: The mesh to render.
        polar_angle:
            Polar angle of the camera.
            For 0 the camera will look down the z-axis.
        azimuth:
            Azimuth of the camera.
            For 0, polar_anlge=pi/2 the camera will look down the x axis.
        camera_distance:
            Distance of camera to the origin.
        plot_object:
            Axis to plot the image. Will use plt if not provided.
        transform: Transform of the object. Identity by default.
    """
    if plot_object is None:
        plot_object = plt
    if transform is None:
        transform = np.eye(4, 4)[None]
    elif transform.ndim == 2:
        transform = transform[None]
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, poses=transform, smooth=False)
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    # position camera on sphere centered and pointing at centroid
    camera_unit_vector = np.array(
        [
            np.sin(polar_angle) * np.cos(azimuth),
            np.sin(polar_angle) * np.sin(azimuth),
            np.cos(polar_angle),
        ]
    )
    camera_position = camera_distance * camera_unit_vector

    camera_pose = np.array(
        [
            [
                -np.sin(azimuth),
                -np.cos(azimuth) * np.cos(polar_angle),
                np.cos(azimuth) * np.sin(polar_angle),
                camera_position[0],
            ],
            [
                np.cos(azimuth),
                -np.sin(azimuth) * np.cos(polar_angle),
                np.sin(azimuth) * np.sin(polar_angle),
                camera_position[1],
            ],
            [0, np.sin(polar_angle), np.cos(polar_angle), camera_position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene.add(camera, pose=camera_pose)
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=45.0)
    light_pose = np.array(
        [
            [
                np.cos(azimuth) * np.cos(polar_angle),
                -np.sin(azimuth),
                np.cos(azimuth) * np.sin(polar_angle),
                camera_position[0],
            ],
            [
                np.sin(azimuth) * np.cos(polar_angle),
                np.cos(azimuth),
                np.sin(azimuth) * np.sin(polar_angle),
                camera_position[1],
            ],
            [-np.sin(polar_angle), 0, np.cos(polar_angle), camera_position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # scene.add(light, pose=camera_pose)
    scene.add_node(pyrender.Node(light=light, matrix=light_pose))
    flags = RenderFlags.RGBA | RenderFlags.ALL_SOLID
    r = pyrender.OffscreenRenderer(1000, 1000, flags)
    color, _ = r.render(scene)
    plot_object.axis("off")
    plot_object.imshow(color, interpolation="none")


def visualize_sdf_reconstruction(
    sdf: np.array, sdf_reconstruction: np.array, show: bool = False
):
    fig = plt.figure()

    level = 1.0 / sdf.shape[-1]
    mesh = mesh_from_sdf(sdf, level=level)
    mesh_reconstruction = mesh_from_sdf(sdf_reconstruction, level=level)

    min = np.min(sdf)
    max = np.max(sdf)

    center = np.array(sdf.shape)[0] // 2

    if mesh is not None:
        plt.subplot(4, 2, 1)
        plot_mesh(mesh)
    if mesh_reconstruction is not None:
        plt.subplot(4, 2, 2)
        plot_mesh(mesh_reconstruction)

    plt.subplot(4, 2, 3)
    plt.imshow(sdf[center, :, :].T, origin="lower", vmin=min, vmax=max)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.subplot(4, 2, 4)
    plt.imshow(sdf_reconstruction[center, :, :].T, origin="lower", vmin=min, vmax=max)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.subplot(4, 2, 5)
    plt.imshow(sdf[:, center, :].T, origin="lower", vmin=min, vmax=max)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.subplot(4, 2, 6)
    plt.imshow(sdf_reconstruction[:, center, :].T, origin="lower", vmin=min, vmax=max)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.subplot(4, 2, 7)
    plt.imshow(sdf[:, :, center].T, origin="lower", vmin=min, vmax=max)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(4, 2, 8)
    plt.imshow(sdf_reconstruction[:, :, center].T, origin="lower", vmin=min, vmax=max)
    plt.xlabel("x")
    plt.ylabel("y")

    if show:
        plt.show()

    return fig


def visualize_sdf_batch(sdfs: np.array, show: bool = False):
    fig = plt.figure()
    num_sdfs = sdfs.shape[0]

    level = 1.0 / sdfs.shape[-1]

    # find nice layout
    cols = 1
    rows = 1
    while rows * cols < num_sdfs:
        if cols / rows > 4 / 3:
            rows += 1
        else:
            cols += 1

    for c in range(num_sdfs):
        plt.subplot(rows, cols, c + 1)
        mesh = mesh_from_sdf(sdfs[c], level=level)
        if mesh is not None:
            plot_mesh(mesh)

    if show:
        plt.show()

    return fig


def visualize_sdf_batch_columns(sdfs: np.array, show: bool = False):
    """Visualize batch of sdfs, with one per column (mesh + cross-views)."""
    fig = plt.figure()
    num_sdfs = sdfs.shape[0]

    center = np.array(sdfs.shape)[2] // 2

    level = 1.0 / sdfs.shape[-1]

    for c in range(num_sdfs):
        min = np.min(sdfs[c])
        max = np.max(sdfs[c])

        plt.subplot(4, num_sdfs, 0 * num_sdfs + c + 1)
        mesh = mesh_from_sdf(sdfs[c], level=level)
        if mesh is not None:
            plot_mesh(mesh)
        plt.subplot(4, num_sdfs, 1 * num_sdfs + c + 1)
        plt.imshow(sdfs[c, center, :, :].T, origin="lower", vmin=min, vmax=max)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.subplot(4, num_sdfs, 2 * num_sdfs + c + 1)
        plt.imshow(sdfs[c, :, center, :].T, origin="lower", vmin=min, vmax=max)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.subplot(4, num_sdfs, 3 * num_sdfs + c + 1)
        plt.imshow(sdfs[c, :, :, center].T, origin="lower", vmin=min, vmax=max)
        plt.xlabel("x")
        plt.ylabel("y")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    # test rendering
    sdf_np = np.load("./data/shapenet_mug_filtered/00000.npy")
    sdf_np2 = np.load("./data/shapenet_mug_filtered/00001.npy")
    sdf_np = np.transpose(sdf_np, (0, 2, 1))
    sdf_np2 = np.transpose(sdf_np2, (0, 2, 1))
    sdfs = np.array([sdf_np, sdf_np2])
    print(np.min(sdf_np), np.max(sdf_np))
    mesh = mesh_from_sdf(sdf_np)
    visualize_sdf_reconstruction(sdf_np, sdf_np2, show=True)
    visualize_sdf_batch(sdfs, True)
    visualize_sdf_batch_columns(sdfs, True)
