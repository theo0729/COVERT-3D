"""Mahalanobis anomaly score cleaning with HKS structural priors."""

from __future__ import annotations

from typing import Any

import numpy as np

from vast.features.hks_fingerprint import compute_annulus_mean

_EPS = 1e-12


def _as_1d_float_array(
    values: np.ndarray,
    name: str,
    n_expected: int | None = None,
) -> np.ndarray:
    """Convert ``values`` to a finite one-dimensional float64 array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if n_expected is not None and arr.shape[0] != int(n_expected):
        raise ValueError(f"{name} length must be {n_expected}, got {arr.shape[0]}.")
    if arr.size == 0 and n_expected not in (0, None):
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def score_summary(values: np.ndarray) -> dict[str, float]:
    """Return compact percentile-aware summary statistics for one score vector."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
    }


def robust_normalize_score(
    values: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Percentile-based robust normalization to ``[0, 1]``."""
    arr = _as_1d_float_array(values, name="values")
    p_low = float(p_low)
    p_high = float(p_high)
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_low >= p_high:
        raise ValueError("p_low and p_high must be finite with p_low < p_high.")

    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo:
        return np.zeros(arr.shape[0], dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo + _EPS), 0.0, 1.0)


def compute_body_surround_from_topo(
    points: np.ndarray,
    topo_prior: np.ndarray,
    r_inner: float = 3.0,
    r_outer: float = 12.0,
    min_neighbors: int = 8,
) -> np.ndarray:
    """Compute annulus-averaged topo support around each point."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")

    topo_prior = _as_1d_float_array(
        topo_prior,
        name="topo_prior",
        n_expected=points.shape[0],
    )
    body_surround = compute_annulus_mean(
        points,
        topo_prior,
        r_inner=r_inner,
        r_outer=r_outer,
        min_neighbors=min_neighbors,
    )
    return np.clip(body_surround, 0.0, 1.0)


def compute_mahalanobis_hks_structural_filter(
    points: np.ndarray,
    mahalanobis_score: np.ndarray,
    topo_score: np.ndarray,
    boundary_layer_score: np.ndarray,
    body_inner_radius: float = 3.0,
    body_outer_radius: float = 12.0,
    boundary_strength: float = 0.75,
    body_floor: float = 0.55,
    body_strength: float = 0.45,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> dict[str, Any]:
    """Clean Mahalanobis anomaly scores using HKS topo and boundary structural priors."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")

    num_points = points.shape[0]
    mahalanobis_score = _as_1d_float_array(
        mahalanobis_score,
        name="mahalanobis_score",
        n_expected=num_points,
    )
    topo_score = _as_1d_float_array(
        topo_score,
        name="topo_score",
        n_expected=num_points,
    )
    boundary_layer_score = _as_1d_float_array(
        boundary_layer_score,
        name="boundary_layer_score",
        n_expected=num_points,
    )

    boundary_strength = float(boundary_strength)
    body_floor = float(body_floor)
    body_strength = float(body_strength)
    if not 0.0 <= boundary_strength <= 1.0:
        raise ValueError("boundary_strength must be in [0, 1].")
    if not 0.0 <= body_floor <= 1.0:
        raise ValueError("body_floor must be in [0, 1].")
    if not 0.0 <= body_strength <= 1.0:
        raise ValueError("body_strength must be in [0, 1].")

    m_score = robust_normalize_score(mahalanobis_score, p_low=p_low, p_high=p_high)
    boundary_prior = robust_normalize_score(
        boundary_layer_score,
        p_low=p_low,
        p_high=p_high,
    )
    topo_prior = robust_normalize_score(topo_score, p_low=p_low, p_high=p_high)

    body_surround = compute_body_surround_from_topo(
        points,
        topo_prior,
        r_inner=body_inner_radius,
        r_outer=body_outer_radius,
    )
    body_gate = np.clip(body_floor + body_strength * body_surround, 0.0, 1.0)
    boundary_gate = np.clip(1.0 - boundary_strength * boundary_prior, 0.0, 1.0)
    mahalanobis_clean_score = np.clip(
        m_score * boundary_gate * body_gate,
        0.0,
        1.0,
    )

    return {
        "base_score_norm": m_score,
        "boundary_prior": boundary_prior,
        "topo_prior": topo_prior,
        "body_surround": body_surround,
        "body_gate": body_gate,
        "boundary_gate": boundary_gate,
        "mahalanobis_clean_score": mahalanobis_clean_score,
        "summary": {
            "base_score_norm": score_summary(m_score),
            "boundary_prior": score_summary(boundary_prior),
            "topo_prior": score_summary(topo_prior),
            "body_surround": score_summary(body_surround),
            "body_gate": score_summary(body_gate),
            "boundary_gate": score_summary(boundary_gate),
            "mahalanobis_clean_score": score_summary(mahalanobis_clean_score),
            "params": {
                "body_inner_radius": float(body_inner_radius),
                "body_outer_radius": float(body_outer_radius),
                "boundary_strength": boundary_strength,
                "body_floor": body_floor,
                "body_strength": body_strength,
                "p_low": float(p_low),
                "p_high": float(p_high),
            },
        },
    }
