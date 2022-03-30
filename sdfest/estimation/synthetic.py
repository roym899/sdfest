"""Module for synthetic data generation."""

from abc import ABC
from typing import Optional

import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np

from sdfest.differentiable_renderer import Camera


class Object(ABC):
    """Generic positioned object representation.

    Each object has a 6-DOF pose, stored as a 3D translation vector and
    a normalized quaternion representing its orientation.
    """

    def __init__(self, position=None, orientation=None):
        """Initialize object position and orientation."""
        if position is None:
            position = np.array([0, 0, 0])
        if orientation is None:
            orientation = np.array([0, 0, 0, 1])
        self.position = position
        self.orientation = orientation


class Mesh(Object):
    """Object with associated mesh.

    This class maintains two meshes, the original mesh and the scaled mesh.

    Updating the scale will always be relative to the original mesh. I.e., two times
    setting the relative scale by 0.1 will not yield a final scale of 0.01 as it will
    always be relative to the original mesh.
    """

    def __init__(
        self,
        mesh: Optional[o3d.geometry.TriangleMesh] = None,
        path: Optional[str] = None,
        scale: float = 1,
        rel_scale: bool = False,
        center: bool = False,
        position: Optional[np.array] = None,
        orientation: Optional[np.array] = None,
    ):
        """Initialize mesh.

        Must provide either mesh or path. If path is given, mesh will be loaded from
        specified file. If mesh is given, it will be used as the original mesh.

        Args:
            mesh: The original (i.e., unscaled mesh).
            path: The path of the mesh to load.
            scale: See Mesh.update_scale.
            rel_scale: See Mesh.update_scale.
            center: whether to center the mesh upon loading.
            position: See Object.__init__.
            orientation:  See Object.__init__.
        """
        super().__init__(position=position, orientation=orientation)
        if mesh is not None and path is not None:
            raise ValueError("Only one of mesh or path can be specified")
        if mesh is not None:
            self._original_mesh = mesh
        if path is not None:
            self._original_mesh = o3d.io.read_triangle_mesh(path)
        if center:
            self._original_mesh.translate([0, 0, 0], relative=False)

        self.update_scale(scale, rel_scale)

    def load_mesh_from_file(
        self, path: str, scale: float = 1, rel_scale: bool = False
    ) -> None:
        """Load mesh from file.

        Args:
            path: Path of the obj file.
            scale: See Mesh.update_scale.
            rel_scale: See Mesh.update_scale.
        """
        self._original_mesh = o3d.io.read_triangle_mesh(path)
        self.update_scale(scale, rel_scale)

    def update_scale(self, scale: float = 1, rel_scale: bool = False) -> None:
        """Update relative or absolute scale of mesh.

        Absolute scale represents half the largest extent in x, y, or z direction.
        Relative scale represents the scale factor from original mesh.

        Args:
            scale: The desired absolute or relative scale of the object.
            rel_scale:
                If true, scale will be relative to original mesh.
                Otherwise, scale will be the resulting absolute scale.
        """
        # self._scale will always be absolute scale
        if rel_scale:
            # copy construct mesh
            self._scaled_mesh = o3d.geometry.TriangleMesh(self._original_mesh)

            original_scale = self._get_original_scale()
            self._scaled_mesh.scale(scale, [0, 0, 0])

            self._scale = original_scale * scale
        else:
            # copy construct mesh
            self._scaled_mesh = o3d.geometry.TriangleMesh(self._original_mesh)

            # scale original mesh s.t. output has the provided scale
            original_scale = self._get_original_scale()
            scale_factor = scale / original_scale
            self._scaled_mesh.scale(scale_factor, [0, 0, 0])

            self._scale = scale

    def _get_original_scale(self) -> float:
        """Compute current scale of original mesh.

        Scale is largest x/y/z extent over 2.
        """
        mins = np.amin(self._original_mesh.vertices, axis=0)
        maxs = np.amax(self._original_mesh.vertices, axis=0)
        ranges = maxs - mins
        return np.max(ranges) / 2

    def get_transformed_o3d_geometry(self) -> o3d.geometry.TriangleMesh:
        """Get o3d mesh at current pose."""
        transformed_mesh = o3d.geometry.TriangleMesh(self._scaled_mesh)
        R = Rotation.from_quat(self.orientation).as_matrix()
        transformed_mesh.rotate(R, center=np.array([0, 0, 0]))
        transformed_mesh.translate(self.position)
        transformed_mesh.compute_vertex_normals()
        return transformed_mesh


def draw_depth_geometry(obj: Object, camera: Camera):
    """Render an object given a camera."""
    # see http://www.open3d.org/docs/latest/tutorial/visualization/customized_visualization.html
    # rend = o3d.visualization.rendering.OffscreenRenderer()
    # img = rend.render_to_image()

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera.width, height=camera.height, visible=False)

    # Add mesh in correct position
    vis.add_geometry(obj.get_transformed_o3d_geometry(), True)

    options = vis.get_render_option()
    options.mesh_show_back_face = True

    # Set camera at fixed position (i.e., at 0,0,0, looking along z axis)
    view_control = vis.get_view_control()
    o3d_cam = camera.get_o3d_pinhole_camera_parameters()
    o3d_cam.extrinsic = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    view_control.convert_from_pinhole_camera_parameters(o3d_cam, True)

    # Generate the depth image
    vis.poll_events()
    vis.update_renderer()
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))

    return depth


if __name__ == "__main__":
    f = 320
    width = 640
    height = 480
    cam = Camera(
        width,
        height,
        fx=f,
        fy=f,
        cx=(width - 1) / 2,
        cy=(height - 1) / 2,
        pixel_center=0.5,
    )
    mesh = Mesh(path="/home/leo/datasets/shapenet_mug_filtered/00000.obj", scale=0.06)
    mesh.position = np.array([0, 0, 0.15])

    # render depth image
    depth = draw_depth_geometry(mesh, cam)

    # show depth image
    plt.imshow(depth)
    plt.show()

    # primitives
    # see: http://www.open3d.org/docs/latest/tutorial/visualization/visualization.html#Geometry-primitives
