"""Module for utility function related to NOCS dataset.

This module contains functions to find similarity transform from NOCS maps and
evaluation function for typical metrics on the NOCS datasets.

Aligning code by Srinath Sridhar:
    https://raw.githubusercontent.com/hughw19/NOCS_CVPR2019/master/aligning.py

Evaluation code by ... TODO
"""
import numpy as np
import cv2
import itertools


def estimate_similarity_transform(
    source: np.ndarray, target: np.ndarray, verbose: bool = False
) -> tuple:
    """Estimate similarity transform from source to target from point correspondences.

    Source and target are pairwise correponding pointsets, i.e., they include same
    number of points and the first point of source corresponds to the first point of
    target. RANSAC is used for outlier-robust estimation.

    A similarity transform is estimated (i.e., isotropic scale, rotation and
    translation) that transforms source points onto the target points.

    Args:
        source: Source points that will be transformed, shape (N,3).
        target: Target points to which source will be aligned to, same shape as source.
        verbose: If true additional information will be printed.
    Returns:
        scales (np.ndarray):
        rotation (np.ndarray):
        translation (np.ndarray):
        transform (np.ndarray):
    """
    # make points homogeneous
    source_hom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))  # 4,N
    target_hom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))  # 4,N

    # Auto-parameter selection based on source-target heuristics
    target_norm = np.mean(np.linalg.norm(target, axis=1))  # mean distance from origin
    source_norm = np.mean(np.linalg.norm(source, axis=1))
    ratio_ts = target_norm / source_norm
    ratio_st = source_norm / target_norm
    pass_t = ratio_st if (ratio_st > ratio_ts) else ratio_ts
    stop_t = pass_t / 100
    n_iter = 100
    if verbose:
        print("Pass threshold: ", pass_t)
        print("Stop threshold: ", stop_t)
        print("Number of iterations: ", n_iter)

    source_inliers_hom, target_inliers_hom, best_inlier_ratio = _get_ransac_inliers(
        source_hom,
        target_hom,
        max_iterations=n_iter,
        pass_threshold=pass_t,
        stop_threshold=stop_t,
    )

    if best_inlier_ratio < 0.1:
        print(
            "[ WARN ] - Something is wrong. Small BestInlierRatio: ", best_inlier_ratio
        )
        return None, None, None, None

    scales, rotation, translation, out_transform = _estimate_similarity_umeyama(
        source_inliers_hom, target_inliers_hom
    )

    if verbose:
        print("BestInlierRatio:", best_inlier_ratio)
        print("Rotation:\n", rotation)
        print("Translation:\n", translation)
        print("Scales:", scales)

    return scales, rotation, translation, out_transform


def _get_ransac_inliers(
    source_hom: np.ndarray,
    target_hom: np.ndarray,
    max_iterations: int = 100,
    pass_threshold: float = 200,
    stop_threshold: float = 1,
) -> tuple:
    """Apply RANSAC and return set of inliers.

    Args:
        source_hom: Homogeneous coordinates of source points, shape (4,N).
        target_hom: Homogeneous coordinates of target points, shape (4,N).
        max_iterations: Maximum number of RANSAC iterations.
        pass_threshold: Threshold at which a point correspondence is considered good.
        stop_threshold: If residual is below this threshold, RANSAC will stop early.
    Returns:
        source_inliers (np.ndarray):
            Homogeneous coordinates of inlier source points, shape (4,M).
        target_inliers (np.ndarray):
            Homogeneous coordinates of inlier target points, shape (4,M).
        inlier_ratio (float): Ratio of inliers and outliers.
    """
    best_residual = 1e10
    best_inlier_ratio = 0
    best_inlier_idx = np.arange(source_hom.shape[1])
    for _ in range(0, max_iterations):
        # Pick 5 random (but corresponding) points from source and target
        rand_idx = np.random.randint(source_hom.shape[1], size=5)
        _, _, _, out_transform = _estimate_similarity_umeyama(
            source_hom[:, rand_idx], target_hom[:, rand_idx]
        )
        residual, inlier_ratio, inlier_idx = _evaluate_model(
            out_transform, source_hom, target_hom, pass_threshold
        )
        if residual < best_residual:
            best_residual = residual
            best_inlier_ratio = inlier_ratio
            best_inlier_idx = inlier_idx
        if best_residual < stop_threshold:
            break

        # print('Iteration: ', i)
        # print('Residual: ', Residual)
        # print('Inlier ratio: ', InlierRatio)

    return (
        source_hom[:, best_inlier_idx],
        target_hom[:, best_inlier_idx],
        best_inlier_ratio,
    )


def _evaluate_model(
    out_transform: np.ndarray,
    source_hom: np.ndarray,
    target_hom: np.ndarray,
    pass_threshold: float,
) -> tuple:
    """Evaluate transformation from source to target points.

    Args:
        out_transform:
            Transformation which will be applied to source points, shape (4,4).
        source_hom: Homogeneous coordinates of source points, shape (4,N).
        target_hom: Homogeneous coordinates of target points, shape (4,N).
        pass_threshold: Threshold at which a point correspondence is considered good.
    Returns:
        residual (float): The mean error between transformed source and target.
        inlier_ratio (float):
            Ratio between inliers and number of correspondences (i.e., N).
        inlier_idx (np.ndarray): Array containing the indices of inliers, shape (M,).
    """
    diff = target_hom - np.matmul(out_transform, source_hom)
    residual_vec = np.linalg.norm(diff[:3, :], axis=0)  # shape (N,)
    residual = np.linalg.norm(residual_vec)
    inlier_idx = np.where(residual_vec < pass_threshold)
    n_inliers = np.count_nonzero(inlier_idx)
    inlier_ratio = n_inliers / source_hom.shape[1]
    return residual, inlier_ratio, inlier_idx[0]


def _estimate_similarity_umeyama(
    source_hom: np.ndarray, target_hom: np.ndarray
) -> tuple:
    """Estimate similarity transform from 3D point correspondences.

    A similarity transform is estimated (i.e., isotropic scale, rotation and
    translation) that transforms source points onto the target points.

    Original algorithm from Least-squares estimation of transformation parameters
    between two point patterns, Umeyama, 1991.
    http://web.stanford.edu/class/cs273/refs/umeyama.pdf

    Args:
        source_hom: Homogeneous coordinates of 5 source points, shape (4,5).
        target_hom: Homogeneous coordinates of 5 target points, shape (4,5).
    Returns:
        scales (np.ndarray):
            Scaling factors along each axis, to scale source to target, shape (3,).
            This will always be three times the same value, since similarity transforms
            only include isotropic scaling.
        rotation (np.ndarray): Rotation to rotate source to target, shape (3,).
        translation (np.ndarray): Translation to translate source to target, shape (3,).
        transform (np.ndarray): Homogeneous transformation matrix, shape (4,4).
    """
    source_centroid = np.mean(source_hom[:3, :], axis=1)
    target_centroid = np.mean(target_hom[:3, :], axis=1)
    n_points = source_hom.shape[1]

    centered_source = (
        source_hom[:3, :] - np.tile(source_centroid, (n_points, 1)).transpose()
    )
    centered_target = (
        target_hom[:3, :] - np.tile(target_centroid, (n_points, 1)).transpose()
    )

    cov_matrix = np.matmul(centered_target, np.transpose(centered_source)) / n_points

    if np.isnan(cov_matrix).any():
        print("nPoints:", n_points)
        print(source_hom.shape)
        print(target_hom.shape)
        raise RuntimeError("There are NANs in the input.")

    u, diag, v = np.linalg.svd(cov_matrix, full_matrices=True)
    d = (np.linalg.det(u) * np.linalg.det(v)) < 0.0
    if d:
        diag[-1] = -diag[-1]
        u[:, -1] = -u[:, -1]

    rotation = np.matmul(u, v).T  # Transpose is the one that works

    var_p = np.var(source_hom[:3, :], axis=1).sum()
    scale_fact = 1 / var_p * np.sum(diag)  # scale factor
    scales = np.array([scale_fact, scale_fact, scale_fact])
    scale_matrix = np.diag(scales)

    translation = target_hom[:3, :].mean(axis=1) - source_hom[:3, :].mean(axis=1).dot(
        scale_fact * rotation
    )

    out_transform = np.identity(4)
    out_transform[:3, :3] = scale_matrix @ rotation
    out_transform[:3, 3] = translation

    return scales, rotation, translation, out_transform


# TODO following functions are only used for affine transform, delete if not needed

# TODO this function doesn't seems to enforce zero shear in its current form
# TODO remove if not needed anywhere
def estimate_restricted_affine_transform(
    source: np.ndarray, target: np.ndarray, verbose: bool = False
) -> tuple:
    """Estimate affine transform from source to target from point correspondences.

    Source and target are pairwise correponding pointsets, i.e., they include same
    number of points and the first point of source corresponds to the first point of
    target. RANSAC is used for outlier-robust estimation.

    A restricted (no shear) affine transform is estimated (i.e., anisotropic scale,
    rotation and translation) that transforms source points onto the target points.

    Args:
        source: Source points that will be transformed, shape (N,3).
        target: Target points to which source will be aligned to, same shape as source.
        verbose: If true additional information will be printed.
    Returns:
        scales:
        translation:
        rotation:
        transform:
    """
    source_hom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    target_hom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    ret_val, affine_trans, inliers = cv2.estimateAffine3D(source, target)

    # We assume no shear in the affine matrix and decompose into rotation, non-uniform
    # scales, and translation
    translation = affine_trans[:3, 3]
    nuscale_rot_mat = affine_trans[:3, :3]
    # nuscale_rot_mat should be the matrix SR, where S is a diagonal scale matrix and R
    # is the rotation matrix (equivalently RS)

    # Let us do the SVD of nuscale_rot_mat to obtain R1*S*R2 and then R = R1 * R2
    _, scales_sorted, _ = np.linalg.svd(nuscale_rot_mat, full_matrices=True)

    if verbose:
        print("-----------------------------------------------------------------------")

    # Now, the scales are sort in ascending order which is painful because we don't know
    # the x, y, z scales; Let's figure that out by evaluating all 6 possible
    # permutations of the scales
    scale_permutations = list(itertools.permutations(scales_sorted))
    min_residual = 1e8
    scales = scale_permutations[0]
    out_transform = np.identity(4)
    rotation = np.identity(3)
    for scale_cand in scale_permutations:
        curr_scale = np.asarray(scale_cand)
        curr_transform = np.identity(4)
        curr_rotation = (np.diag(1 / curr_scale) @ nuscale_rot_mat).transpose()
        curr_transform[:3, :3] = np.diag(curr_scale) @ curr_rotation
        curr_transform[:3, 3] = translation
        # Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
        residual = _evaluate_model_non_hom(
            source, target, curr_scale, curr_rotation, translation
        )
        if verbose:
            # print('CurrTransform:\n', CurrTransform)
            print("CurrScale:", curr_scale)
            print("Residual:", residual)
            print(
                "AltRes:",
                _evaluate_model_no_thresh(curr_transform, source_hom, target_hom),
            )
        if residual < min_residual:
            min_residual = residual
            scales = curr_scale
            rotation = curr_rotation
            out_transform = curr_transform

    if verbose:
        print("Best Scale:", scales)

    if verbose:
        print("Affine Scales:", scales)
        print("Affine Translation:", translation)
        print("Affine Rotation:\n", rotation)
        print("-----------------------------------------------------------------------")

    return scales, rotation, translation, out_transform


# TODO delete if estimate_restricted_affine_transform is not required
def _evaluate_model_no_thresh(out_transform, source_hom, target_hom):
    Diff = target_hom - np.matmul(out_transform, source_hom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual


# TODO delete if estimate_restricted_affine_transform is not required
def _evaluate_model_non_hom(source, target, scales, rotation, translation):
    RepTrans = np.tile(translation, (source.shape[0], 1))
    TransSource = (
        np.diag(scales) @ rotation @ source.transpose() + RepTrans.transpose()
    ).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual
