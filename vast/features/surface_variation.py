"""Multi-scale surface variation features."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_surface_variation(
    points: np.ndarray,
    k_scales: list[int] | None = None,
    fusion: str = "max",
) -> np.ndarray:
    """Compute fused multi-scale absolute surface variation.

    For each scale ``k``, local PCA is performed on the ``k`` nearest
    neighbors and the smallest-eigenvalue ratio is computed as:

        c = lambda_0 / (lambda_0 + lambda_1 + lambda_2 + 1e-8)

    The per-scale values are fused into a single array ``c_abs`` using
    ``max``, ``mean``, or ``p75``.

    Args:
        points: Point cloud with shape ``(N, 3)``.
        k_scales: Neighborhood sizes for multi-scale PCA.
        fusion: Fusion strategy, one of ``{'max', 'mean', 'p75'}``.

    Returns:
        Fused absolute surface variation with shape ``(N,)``.
    """
    points = _validate_points(points)
    if k_scales is None:
        k_scales = [12, 24, 48]

    fusion = str(fusion).lower()
    if fusion not in {"max", "mean", "p75"}:
        raise ValueError("fusion must be one of {'max', 'mean', 'p75'}.")

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("points must not be empty.")

    normalized_scales = sorted({int(k) for k in k_scales if int(k) > 0})
    if not normalized_scales:
        raise ValueError("k_scales must contain at least one positive integer.")

    if num_points < 3:
        return np.zeros(num_points, dtype=np.float64)

    max_k = min(max(normalized_scales), num_points)
    k_query = min(max_k + 1, num_points)
    tree = cKDTree(points)
    try:
        _, neighbor_indices = tree.query(points, k=k_query, workers=-1)
    except TypeError:
        _, neighbor_indices = tree.query(points, k=k_query)

    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if k_query == 1:
        return np.zeros(num_points, dtype=np.float64)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices.reshape(-1, 1)
    if k_query > 1:
        neighbor_indices = neighbor_indices[:, 1:]

    per_scale = np.stack(
        [
            _surface_variation_from_neighbors(points, neighbor_indices[:, : min(k, neighbor_indices.shape[1])])
            for k in normalized_scales
        ],
        axis=1,
    )
    return _fuse_variation(per_scale, fusion=fusion)


def _surface_variation_from_neighbors(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Compute surface variation from precomputed neighbor indices."""
    num_points = points.shape[0]
    effective_k = int(neighbor_indices.shape[1])
    if effective_k < 3:
        return np.zeros(num_points, dtype=np.float64)

    local_points = points[neighbor_indices]
    local_mean = local_points.mean(axis=1, keepdims=True)
    centered = local_points - local_mean

    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True)
    covariance /= float(max(effective_k - 1, 1))

    eigenvalues = np.linalg.eigh(covariance)[0]
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    eigen_sum = eigenvalues.sum(axis=1) + 1e-8
    return (eigenvalues[:, 0] / eigen_sum).astype(np.float64)


def _fuse_variation(values: np.ndarray, fusion: str) -> np.ndarray:
    """Fuse per-scale surface variation values into one array."""
    if fusion == "max":
        return values.max(axis=1).astype(np.float64)
    if fusion == "mean":
        return values.mean(axis=1).astype(np.float64)
    return np.percentile(values, 75.0, axis=1).astype(np.float64)


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Validate point cloud shape and numeric values."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")
    return points
