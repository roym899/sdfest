"""This module provides a deterministic low-dispersion grid on SO3."""
from typing import Tuple

import numpy as np
import healpy as hp


class SO3Grid:
    """Low-dispersion SO3 grid.

    This approach was introduced by Generating Uniform Incremental Grids on SO3 Using
    the Hopf Fibration, Yershova, 2010. We only generate the base grid (i.e., up to and
    including Section 5.2 of the paper), since we only need a fixed size set.

    Implementation roughly based on https://github.com/zhonge/cryodrgn
    """

    def __init__(self, resol: int):
        """Construct the SO3 grid.

        Args:
            resol: The resolution of the grid. Coarsest possible grid for 0.
        """
        self._resol = resol
        self._s1 = self._grid_s1(resol)
        self._s2_theta, self._s2_phi = self._grid_s2(resol)

    def num_cells(self) -> int:
        """Return the number of points in the grid."""
        return len(self._s1) * len(self._s2_theta)

    def hopf_to_index(self, psi, theta, phi):
        """Convert hopf coordinate to index.

        Args:
            phi: [0, 2pi)
            theta: [0, pi]
            psi: [0, 2pi)
        Returns:
            Grid index of closest point in grid.
        """
        s1_index = int(psi // (2 * np.pi / len(self._s1)))
        s2_index = hp.ang2pix(2 ** self._resol, theta, phi, nest=True)
        return s1_index * len(self._s2_theta) + s2_index

    def index_to_hopf(self, index: int) -> Tuple[float, float, float]:
        """Convert index to hopf coordinates.

        Psi: [0,2*pi)
        Theta: [0, pi]
        Phi: [0, 2*pi)

        Args:
            index: The index of the grid point.
        Returns:
            Tuple of psi, theta, phi.
        """
        s1_index = index // len(self._s2_theta)
        s2_index = index % len(self._s2_theta)
        psi = self._s1[s1_index]
        theta = self._s2_theta[s2_index]
        phi = self._s2_phi[s2_index]
        return psi, theta, phi

    def quat_to_index(self, quaternion: np.array) -> int:
        """Convert quaternion to index.

        Will convert quaternion to Hopf coordinates and look up closest Hopf coordinate.
        Closest means, closest in Hopf coordinates.

        Args:
            quaternion:
                Array of shape (4,), containing a normalized quaternion.
                The order of the quaternion is (x, y, z, w).
        Returns:
            The index of the closest (in Hopf coordinates) point.
        """
        hopf = SO3Grid._quat_to_hopf(quaternion)
        return self.hopf_to_index(*hopf)

    def index_to_quat(self, index: int) -> np.array:
        """Convert index to quaternion.

        Returns:
            Array of shape (4,), containing the normalized quaternion corresponding
            to the index.
        """
        hopf = self.index_to_hopf(index)
        return SO3Grid._hopf_to_quat(*hopf)

    @staticmethod
    def _quat_to_hopf(quaternion: np.array) -> Tuple[float, float, float]:
        """Convert quaternion to hopf coordinates.

        Args:
            quaternion:
                Array of shape (4,), containing a normalized quaternion.
                The order of the quaternion is (x, y, z, w).
        Returns:
            Tuple of psi, theta, phi.

            With psi, theta, phi in [0,2pi), [0,pi], [0,2pi) respectively.
        """
        x, y, z, w = quaternion
        psi = 2 * np.arctan2(x, w)
        theta = 2 * np.arctan2(np.sqrt(z ** 2 + y ** 2), np.sqrt(w ** 2 + x ** 2))
        phi = np.arctan2(z * w - x * y, y * w + x * z)

        # Note for the following correction use while instead of if, to support
        # float32, because atan2 range for float32 ([-np.float32(np.pi),
        # np.float32(np.pi)]) is larger than for float64 ([-np.pi,np.pi]).

        # Psi must be [0, 2pi) and wraps around at 4*pi, so this correction changes the
        # the half-sphere
        while psi < 0:
            psi += 2 * np.pi
        while psi >= 2 * np.pi:
            psi -= 2 * np.pi

        # Phi must be [0, 2pi) and wraps around at 2*pi, so this correction just makes
        # sure the angle is in the expected range
        while phi < 0:
            phi += 2 * np.pi
        while phi >= 2 * np.pi:
            phi -= 2 * np.pi
        return psi, theta, phi

    @staticmethod
    def _hopf_to_quat(psi, theta, phi):
        """Convert quaternion to hopf coordinates.

        Args:
            phi: [0, 2pi)
            theta: [0, pi]
            psi: [0, 2pi)
        Returns:
            Array of shape (4,), containing the normalized quaternion corresponding
            to the index.
        """
        quaternion = np.array(
            [
                np.cos(theta / 2) * np.sin(psi / 2),  # x
                np.sin(theta / 2) * np.cos(phi + psi / 2),  # y
                np.sin(theta / 2) * np.sin(phi + psi / 2),  # z
                np.cos(theta / 2) * np.cos(psi / 2),  # w
            ]
        )
        if quaternion[0] < 0:
            quaternion *= -1
        return quaternion

    @staticmethod
    def _grid_s1(resol):
        """Compute equidistant grid on 1-sphere.

        Args:
            resol: Resolution of grid.
        Returns:
            Center points of the grid cells."""
        points = 6 * 2 ** resol
        grid = np.linspace(0, 2 * np.pi, points, endpoint=False) + np.pi / points
        return grid

    @staticmethod
    def _grid_s2(resol):
        """Compute HEALpix coordinates of 2-sphere.

        Args:
            resol: Resolution of grid.
        Returns:
            Center points of the grid cells."""
        points_per_side = 2 ** resol
        points = 12 * points_per_side * points_per_side
        theta, phi = hp.pix2ang(points_per_side, np.arange(points), nest=True)
        return theta, phi
