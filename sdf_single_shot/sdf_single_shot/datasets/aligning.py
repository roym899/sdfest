"""Module to compute similarity transform from NOCS maps.

Original code by Srinath Sridhar:
    https://raw.githubusercontent.com/hughw19/NOCS_CVPR2019/master/aligning.py
"""
import numpy as np
import cv2
import itertools


def estimate_similarity_transform(
    source: np.ndarray, target: np.ndarray, verbose: bool = False
) -> tuple:
    """Given two 3D pointclouds, estimate the transform to transform source on target.

    A similarity transform is estimated, i.e., isotropic scale, rotation and
    translation.

    Args:
        source: Source pointcloud that will be transformed, shape (N,3)
        target: Target pointcloud to which source will be aligned to, shape (M,3)
        verbose: If true additional information will be printed.
    Returns:
        scale:
        translation:
        rotation:
        transform:
    """
    # make points homogeneous
    source_hom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))  # 4,N
    target_hom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))  # 4,M

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

    scales, rotation, translation, out_transform = estimate_similarity_umeyama(
        source_inliers_hom, target_inliers_hom
    )

    if verbose:
        print("BestInlierRatio:", best_inlier_ratio)
        print("Rotation:\n", rotation)
        print("Translation:\n", translation)
        print("Scales:", scales)

    return scales, rotation, translation, out_transform


def estimate_restricted_affine_transform(
    source: np.ndarray, target: np.ndarray, verbose=False
):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    RetVal, AffineTrans, Inliers = cv2.estimateAffine3D(source, target)
    # We assume no shear in the affine matrix and decompose into rotation, non-uniform scales, and translation
    Translation = AffineTrans[:3, 3]
    NUScaleRotMat = AffineTrans[:3, :3]
    # NUScaleRotMat should be the matrix SR, where S is a diagonal scale matrix and R is the rotation matrix (equivalently RS)
    # Let us do the SVD of NUScaleRotMat to obtain R1*S*R2 and then R = R1 * R2
    R1, ScalesSorted, R2 = np.linalg.svd(NUScaleRotMat, full_matrices=True)

    if verbose:
        print("-----------------------------------------------------------------------")
    # Now, the scales are sort in ascending order which is painful because we don't know the x, y, z scales
    # Let's figure that out by evaluating all 6 possible permutations of the scales
    ScalePermutations = list(itertools.permutations(ScalesSorted))
    MinResidual = 1e8
    Scales = ScalePermutations[0]
    OutTransform = np.identity(4)
    Rotation = np.identity(3)
    for ScaleCand in ScalePermutations:
        CurrScale = np.asarray(ScaleCand)
        CurrTransform = np.identity(4)
        CurrRotation = (np.diag(1 / CurrScale) @ NUScaleRotMat).transpose()
        CurrTransform[:3, :3] = np.diag(CurrScale) @ CurrRotation
        CurrTransform[:3, 3] = Translation
        # Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
        Residual = evaluate_model_non_hom(
            source, target, CurrScale, CurrRotation, Translation
        )
        if verbose:
            # print('CurrTransform:\n', CurrTransform)
            print("CurrScale:", CurrScale)
            print("Residual:", Residual)
            print(
                "AltRes:", evaluate_model_no_thresh(CurrTransform, SourceHom, TargetHom)
            )
        if Residual < MinResidual:
            MinResidual = Residual
            Scales = CurrScale
            Rotation = CurrRotation
            OutTransform = CurrTransform

    if verbose:
        print("Best Scale:", Scales)

    if verbose:
        print("Affine Scales:", Scales)
        print("Affine Translation:", Translation)
        print("Affine Rotation:\n", Rotation)
        print("-----------------------------------------------------------------------")

    return Scales, Rotation, Translation, OutTransform


def _get_ransac_inliers(
    source_hom: np.ndarray,
    target_hom: np.ndarray,
    max_iterations: int = 100,
    pass_threshold: float = 200,
    stop_threshold: float = 1,
) -> tuple:
    """

    Args:
        source_hom
        target_hom
        max_iterations:
        pass_threshold:
        stop_threshold:
    Returns:
        source_inliers:
        target_inliers:
    """
    best_residual = 1e10
    best_inlier_ratio = 0
    best_inlier_idx = np.arange(source_hom.shape[1])
    for _ in range(0, max_iterations):
        # Pick 5 random (but corresponding) points from source and target
        rand_idx = np.random.randint(source_hom.shape[1], size=5)
        _, _, _, out_transform = estimate_similarity_umeyama(
            source_hom[:, rand_idx], target_hom[:, rand_idx]
        )
        residual, inlier_ratio, inlier_idx = evaluate_model(
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


def evaluate_model(out_transform, source_hom, target_hom, pass_threshold):
    diff = target_hom - np.matmul(out_transform, source_hom)
    residual_vec = np.linalg.norm(diff[:3, :], axis=0)
    residual = np.linalg.norm(residual_vec)
    inlier_idx = np.where(residual_vec < pass_threshold)
    n_inliers = np.count_nonzero(inlier_idx)
    inlier_ratio = n_inliers / source_hom.shape[1]
    return residual, inlier_ratio, inlier_idx[0]


def evaluate_model_no_thresh(OutTransform, SourceHom, TargetHom):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual


def evaluate_model_non_hom(source, target, Scales, Rotation, Translation):
    RepTrans = np.tile(Translation, (source.shape[0], 1))
    TransSource = (
        np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()
    ).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual


def test_non_uniform_scale(SourceHom, TargetHom):
    OutTransform = np.matmul(TargetHom, np.linalg.pinv(SourceHom))
    ScaledRotation = OutTransform[:3, :3]
    Translation = OutTransform[:3, 3]
    Sx = np.linalg.norm(ScaledRotation[0, :])
    Sy = np.linalg.norm(ScaledRotation[1, :])
    Sz = np.linalg.norm(ScaledRotation[2, :])
    Rotation = np.vstack(
        [
            ScaledRotation[0, :] / Sx,
            ScaledRotation[1, :] / Sy,
            ScaledRotation[2, :] / Sz,
        ]
    )
    print("Rotation matrix norm:", np.linalg.norm(Rotation))
    Scales = np.array([Sx, Sy, Sz])

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform


def estimate_similarity_umeyama(source_hom, target_hom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(source_hom[:3, :], axis=1)
    TargetCentroid = np.mean(target_hom[:3, :], axis=1)
    nPoints = source_hom.shape[1]

    CenteredSource = (
        source_hom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    )
    CenteredTarget = (
        target_hom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    )

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print("nPoints:", nPoints)
        print(source_hom.shape)
        print(target_hom.shape)
        raise RuntimeError("There are NANs in the input.")

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = np.matmul(U, Vh).T  # Transpose is the one that works

    varP = np.var(source_hom[:3, :], axis=1).sum()
    ScaleFact = 1 / varP * np.sum(D)  # scale factor
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation = target_hom[:3, :].mean(axis=1) - source_hom[:3, :].mean(axis=1).dot(
        ScaleFact * Rotation
    )

    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleMatrix @ Rotation
    OutTransform[:3, 3] = Translation

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform
