"""Tests for SO3Grid class."""
import numpy as np
from sdf_single_shot.so3grid import SO3Grid


def test_num_cells():
    """Test whether the grid has the expected number of points.

    Expected number is 6 * 12 * 2^(3*resolution)
    """
    grid_0 = SO3Grid(0)
    assert grid_0.num_cells() == 6 * 12

    grid_1 = SO3Grid(1)
    assert grid_1.num_cells() == 6 * 12 * 2 ** 3

    grid_2 = SO3Grid(2)
    assert grid_2.num_cells() == 6 * 12 * 2 ** 6


def test_hopf_to_quat_conversion():
    """Test Hopf to quaternion conversion.

    Check if this implements equation (4) from Generating Uniform Incremental Grids on
    SO3 Using the Hopf Fibration, Yershova, 2010.
    """
    hopf = (0, 0, 0)
    quat = SO3Grid._hopf_to_quat(*hopf)
    assert (quat == np.array([0, 0, 0, 1])).all()

    hopf = (0.3, 0.4, 0.2)
    quat = SO3Grid._hopf_to_quat(*hopf)
    np.testing.assert_allclose(
        quat, np.array([0.1464593191, 0.1866245482, 0.06812327794, 0.9690614866])
    )

    # Hopf coordinate outside of [0,2pi)x[0,pi]x[0,2pi) should still yield quaternion
    # on x>0 half-sphere.
    hopf = (0.3, 4, 0.2)
    quat = SO3Grid._hopf_to_quat(*hopf)
    # just applying (4) yields quaternion on x<0 half-sphere.
    quat_direct = np.array([-0.06218820609, 0.8541691906, 0.311796094, -0.4114739562])
    # expected is the antipodal point on the three sphere, i.e., the negative quaternion
    np.testing.assert_allclose(quat, -quat_direct)


def test_quat_hopf_conversions():
    """Test quaternion to Hopf and Hopf to quaternion conversions by inversion.

    This is tested by checking whether we can recover the original quaternion after
    quaternion to Hopf conversion.
    """
    # Test 1: quat -> Hopf -> quat
    # Note that this is only expected to hold if x-component of quaternion is > 0
    # This is because quaternions are a double cover of SO3.
    quat = np.array([0.3, 0.2, 0.6, 1])
    quat /= np.linalg.norm(quat)  # normalize quaternion
    hopf = SO3Grid._quat_to_hopf(quat)
    quat_2 = SO3Grid._hopf_to_quat(*hopf)
    np.testing.assert_allclose(quat, quat_2)

    # Test 2: Hopf -> quat -> Hopf
    # Note that this only expected to hold if psi, theta, phi (Hopf coordinates)
    # in [0, 2pi), [0,pi], [0,2pi), respectively.
    hopf = (0.3, 0.1, 0.2)
    quat = SO3Grid._hopf_to_quat(*hopf)
    hopf_2 = SO3Grid._quat_to_hopf(quat)
    np.testing.assert_allclose(hopf, hopf_2)


def test_hopf_indices():
    """Test that a grid point in Hopf coordinates yields its index.

    Computing a grid point for a given index and computing the index for that point
    should give the same index.
    """
    grid = SO3Grid(0)
    hopf = grid.index_to_hopf(10)
    index = grid.hopf_to_index(*hopf)
    assert index == 10


def test_quaternion_indices():
    """Test that a grid point specified as a quaternion yields its index.

    Computing a quaternion for a given index and computing the index for that quternion
    should give the same index.
    """
    grid = SO3Grid(0)
    quaternion = grid.index_to_quat(30)
    index = grid.quat_to_index(quaternion)
    assert index == 30


def test_noisy_mapping():
    """Test that a noisy Hopf coordinate is mapped to the closest index.

    Computing a grid point for a given index, adding some noise and computing the index
    for the noisy point should recover the index.
    """
    grid = SO3Grid(0)
    psi, theta, phi = grid.index_to_hopf(15)
    index = grid.hopf_to_index(psi + 0.11, theta - 0.11, phi + 0.12)
    assert index == 15
