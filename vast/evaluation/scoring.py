"""Mahalanobis distance scoring for multi-dimensional feature vectors."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

def compute_mahalanobis_score(features: np.ndarray) -> np.ndarray:
    """Compute squared Mahalanobis distance for each feature vector.

    Args:
        features: Feature matrix with shape ``(N, D)``.

    Returns:
        One-dimensional squared Mahalanobis distance scores with shape ``(N,)``.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D).")
    if features.shape[0] < 2:
        raise ValueError("features must contain at least two samples.")

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values.")

    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov += np.eye(features.shape[1], dtype=np.float64) * 1e-6

    inv_cov = np.linalg.inv(cov)
    diff = features - mu
    scores = np.einsum("ni,ij,nj->n", diff, inv_cov, diff, optimize=True)
    return scores.astype(np.float64, copy=False)


def compute_calibrated_mahalanobis_score(
    test_features: np.ndarray,
    baseline_features: np.ndarray,
) -> np.ndarray:
    """Compute squared Mahalanobis distance using a normal-sample baseline.

    The mean and covariance are estimated from ``baseline_features``, then
    applied to each row of ``test_features``.

    Args:
        test_features: Feature matrix for the test cloud with shape ``(N, D)``.
        baseline_features: Feature matrix for normal baseline samples with shape
            ``(M, D)``.

    Returns:
        One-dimensional squared Mahalanobis distance scores with shape ``(N,)``.
    """
    test_features = np.asarray(test_features, dtype=np.float64)
    baseline_features = np.asarray(baseline_features, dtype=np.float64)

    if test_features.ndim != 2 or baseline_features.ndim != 2:
        raise ValueError("test_features and baseline_features must have shape (N, D).")
    if test_features.shape[1] != baseline_features.shape[1]:
        raise ValueError("test_features and baseline_features must share feature dimension D.")
    if baseline_features.shape[0] < 2:
        raise ValueError("baseline_features must contain at least two samples.")
    if test_features.shape[0] < 1:
        raise ValueError("test_features must contain at least one sample.")

    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_features = np.nan_to_num(baseline_features, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(test_features).all() or not np.isfinite(baseline_features).all():
        raise ValueError("feature matrices must contain only finite values.")

    mu = np.mean(baseline_features, axis=0)
    cov = np.cov(baseline_features, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov += np.eye(baseline_features.shape[1], dtype=np.float64) * 1e-6

    inv_cov = np.linalg.inv(cov)
    diff = test_features - mu
    scores = np.einsum("ni,ij,nj->n", diff, inv_cov, diff, optimize=True)
    return scores.astype(np.float64, copy=False)


def compute_feature_knn_score(
    test_features: np.ndarray,
    baseline_features: np.ndarray,
) -> np.ndarray:
    """Compute anomaly scores via feature-space nearest-neighbor distance.

    Features are z-score normalized using baseline statistics before querying
    the nearest baseline neighbor in Euclidean space.

    Args:
        test_features: Feature matrix for the test cloud with shape ``(N, D)``.
        baseline_features: Feature matrix for normal baseline samples with shape
            ``(M, D)``.

    Returns:
        One-dimensional squared nearest-neighbor distance scores with shape ``(N,)``.
    """
    test_features = np.asarray(test_features, dtype=np.float64)
    baseline_features = np.asarray(baseline_features, dtype=np.float64)

    if test_features.ndim != 2 or baseline_features.ndim != 2:
        raise ValueError("test_features and baseline_features must have shape (N, D).")
    if test_features.shape[1] != baseline_features.shape[1]:
        raise ValueError("test_features and baseline_features must share feature dimension D.")
    if baseline_features.shape[0] < 1:
        raise ValueError("baseline_features must contain at least one sample.")
    if test_features.shape[0] < 1:
        raise ValueError("test_features must contain at least one sample.")

    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_features = np.nan_to_num(baseline_features, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(test_features).all() or not np.isfinite(baseline_features).all():
        raise ValueError("feature matrices must contain only finite values.")

    mu = np.mean(baseline_features, axis=0)
    std = np.std(baseline_features, axis=0) + 1e-6
    base_norm = (baseline_features - mu) / std
    test_norm = (test_features - mu) / std

    tree = cKDTree(base_norm)
    distances, _ = tree.query(test_norm, k=1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    return (distances ** 2).astype(np.float64, copy=False)
