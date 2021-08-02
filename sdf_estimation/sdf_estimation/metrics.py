"""Metrics for shape evaluation."""
import numpy as np
import scipy.spatial


def mean_accuracy(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute accuracy metric.

    Accuracy metric is the same as asymmetric chamfer distance from rec to gt.

    See, for example, Occupancy Networks Learning 3D Reconstruction in Function Space,
    Mescheder et al., 2019.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide result by Euclidean extent of points_gt
    Returns:
        Arithmetic mean of p-norm from reconstructed points to closest (in p-norm)
        ground truth points.
    """
    kd_tree = scipy.spatial.KDTree(points_gt)
    d, _ = kd_tree.query(points_rec, p=p_norm)
    if normalize:
        return np.mean(d) / extent(points_gt)
    else:
        return np.mean(d)


def mean_completeness(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute completeness metric.

    Completeness metric is the same as asymmetric chamfer distance from gt to rec.

    See, for example, Occupancy Networks Learning 3D Reconstruction in Function Space,
    Mescheder et al., 2019.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide result by Euclidean extent of points_gt
    Returns:
        Arithmetic mean of p-norm from ground truth points to closest (in p-norm)
        reconstructed points.
    """
    kd_tree = scipy.spatial.KDTree(points_rec)
    d, _ = kd_tree.query(points_gt, p=p_norm)
    if normalize:
        return np.mean(d) / extent(points_gt)
    else:
        return np.mean(d)


def symmetric_chamfer(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute symmetric chamfer distance.

    There are various slightly different definitions for the chamfer distance.

    Note that completeness and accuracy are themselves sometimes referred to as
    chamfer distances, with symmetric chamfer distance being the combination of the two.

    Chamfer L1 in the literature (see, for example, Occupancy Networks Learning 3D
    Reconstruction in Function Space, Mescheder et al., 2019) refers to using
    arithmetic mean (note that this is actually differently scaled from L1) when
    combining accuracy and completeness.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide result by Euclidean extent of points_gt
    Returns:
        Arithmetic mean of accuracy and completeness metrics using the specified p-norm.
    """
    return (
        mean_completeness(points_gt, points_rec, p_norm=p_norm, normalize=normalize)
        + mean_accuracy(points_gt, points_rec, p_norm=p_norm, normalize=normalize)
    ) / 2


def completeness_thresh(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    threshold: float,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute thresholded completion metric.

    See FroDO: From Detections to 3D Objects, Rünz et al., 2020.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        threshold: distance threshold to count a point as correct
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide distances by Euclidean extent of points_gt
    Returns:
        Ratio of ground truth points with closest reconstructed point closer than
        threshold (in p-norm).
    """
    kd_tree = scipy.spatial.KDTree(points_rec)
    d, _ = kd_tree.query(points_gt, p=p_norm)
    if normalize:
        return np.sum(d / extent(points_gt) < threshold) / points_gt.shape[0]
    else:
        return np.sum(d < threshold) / points_gt.shape[0]


def accuracy_thresh(
    points_gt: np.ndarray,
    points_rec: np.ndarray,
    threshold: float,
    p_norm: int = 2,
    normalize: bool = False,
) -> float:
    """Compute thresholded accuracy metric.

    See FroDO: From Detections to 3D Objects, Rünz et al., 2020.

    Args:
        points_gt: set of true points, expected shape (N,3)
        points_rec: set of reconstructed points, expected shape (M,3)
        threshold: distance threshold to count a point as correct
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
        normalize: whether to divide distances by Euclidean extent of points_gt
    Returns:
        Ratio of reconstructed points with closest ground truth point closer than
        threshold (in p-norm).
    """
    kd_tree = scipy.spatial.KDTree(points_gt)
    d, _ = kd_tree.query(points_rec, p=p_norm)
    if normalize:
        return np.sum(d / extent(points_gt) < threshold) / points_rec.shape[0]
    else:
        return np.sum(d < threshold) / points_rec.shape[0]


def extent(points: np.ndarray) -> float:
    """Compute largest Euclidean distance between any two points.

    Args:
        points_gt: set of true
        p_norm: which Minkowski p-norm is used for distance and nearest neighbor query
    Returns:
        Ratio of reconstructed points with closest ground truth point closer than
        threshold (in p-norm).
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except scipy.spatial.qhull.QhullError:
        # fallback to brute force distance matrix
        return np.max(scipy.spatial.distance_matrix(points, points))

    # this is wasteful, if too slow implement rotating caliper method
    return np.max(
        scipy.spatial.distance_matrix(points[hull.vertices], points[hull.vertices])
    )
