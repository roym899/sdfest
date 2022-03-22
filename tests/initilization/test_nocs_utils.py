"""Tests for NOCS utils module."""
import numpy as np
from scipy.spatial.transform import Rotation

from sdfest.initialization.datasets import nocs_utils


def test_estimate_similarity_transform() -> None:
    """Test estimation of similarity transform."""
    a = np.random.rand(4, 10)
    a[3, :] = 1
    r = Rotation.from_euler("xyz", np.array([100, 70, -30]), degrees=True).as_matrix()
    s = 0.3
    t = np.array([0.3, 1.0, 10.0])
    transform = np.eye(4)
    transform[:3, :3] = s * r
    transform[:3, 3] = t
    b = transform @ a
    a_inhom = a[:3, :].T
    b_inhom = b[:3, :].T
    t_est, r_est, s_est, transform_est = nocs_utils.estimate_similarity_transform(
        a_inhom, b_inhom
    )
    np.testing.assert_allclose(t, t_est, atol=1e-10)
    np.testing.assert_allclose(r, r_est, atol=1e-10)
    np.testing.assert_allclose(s, s_est, atol=1e-10)
    np.testing.assert_allclose(transform, transform_est, atol=1e-10)


def test_estimate_similarity_umeyama() -> None:
    """Test estimation of similarity transform."""
    a = np.random.rand(4, 10)
    a[3, :] = 1
    s, r, t, transform = nocs_utils._estimate_similarity_umeyama(a, a)
    np.testing.assert_allclose(np.array([1, 1, 1]), s, atol=1e-10)
    np.testing.assert_allclose(np.diag([1, 1, 1]), r, atol=1e-10)
    np.testing.assert_allclose(np.array([0, 0, 0]), t, atol=1e-10)

    r = Rotation.from_euler("xyz", np.array([100, 70, -30]), degrees=True).as_matrix()
    s = 0.3
    t = np.array([0.3, 1.0, 10.0])
    transform = np.eye(4)
    transform[:3, :3] = s * r
    transform[:3, 3] = t
    b = transform @ a
    s_est, r_est, t_est, transform_est = nocs_utils._estimate_similarity_umeyama(a, b)
    np.testing.assert_allclose(r, r_est, atol=1e-10)
    np.testing.assert_allclose(s, s_est, atol=1e-10)
    np.testing.assert_allclose(t, t_est, atol=1e-10)
    np.testing.assert_allclose(transform, transform_est, atol=1e-10)
    np.testing.assert_allclose(transform_est @ a, b, atol=1e-10)
