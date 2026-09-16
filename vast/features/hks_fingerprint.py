"""HKS Thermal Spectral Fingerprint (TSF) feature extraction."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

_TARGET_TIMES = (1.0, 100.0, 1000.0, 10000.0)
_ROBUST_Z_SCALE = 1.4826


def robust_normalize(
    values: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Percentile-based robust normalization to ``[0, 1]``."""
    values = _validate_1d_finite(values, name="values")
    p_low = float(p_low)
    p_high = float(p_high)
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_low >= p_high:
        raise ValueError("p_low and p_high must be finite with p_low < p_high.")

    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    return np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("relu input must contain only finite values.")
    return np.maximum(x, 0.0)


def compute_annulus_mean(
    points: np.ndarray,
    values: np.ndarray,
    r_inner: float,
    r_outer: float,
    min_neighbors: int = 8,
) -> np.ndarray:
    """Mean of ``values`` over neighbors in the annulus ``[r_inner, r_outer]``."""
    points = _validate_points(points)
    values = _validate_1d_finite(values, name="values", length=points.shape[0])
    r_inner = float(r_inner)
    r_outer = float(r_outer)
    min_neighbors = int(min_neighbors)
    if r_inner < 0.0 or r_outer <= 0.0 or r_inner >= r_outer:
        raise ValueError("Require 0 <= r_inner < r_outer.")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be positive.")

    num_points = points.shape[0]
    if num_points == 0:
        return np.zeros(0, dtype=np.float64)

    tree = cKDTree(points)
    try:
        neighbor_lists = tree.query_ball_point(points, r=r_outer, workers=-1)
    except TypeError:
        neighbor_lists = tree.query_ball_point(points, r=r_outer)

    means = np.empty(num_points, dtype=np.float64)
    for point_index, neighbors in enumerate(neighbor_lists):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        if neighbors.size == 0:
            means[point_index] = values[point_index]
            continue

        deltas = points[neighbors] - points[point_index]
        distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        annulus_mask = (distances > r_inner) & (distances <= r_outer)
        annulus_indices = neighbors[annulus_mask]
        if annulus_indices.size < min_neighbors:
            means[point_index] = values[point_index]
        else:
            means[point_index] = float(np.mean(values[annulus_indices]))
    return means


def compute_ball_mean(
    points: np.ndarray,
    values: np.ndarray,
    radius: float,
    min_neighbors: int = 4,
) -> np.ndarray:
    """Mean of ``values`` over neighbors within ``radius``."""
    points = _validate_points(points)
    values = _validate_1d_finite(values, name="values", length=points.shape[0])
    radius = float(radius)
    min_neighbors = int(min_neighbors)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be positive.")

    num_points = points.shape[0]
    if num_points == 0:
        return np.zeros(0, dtype=np.float64)

    tree = cKDTree(points)
    try:
        neighbor_lists = tree.query_ball_point(points, r=radius, workers=-1)
    except TypeError:
        neighbor_lists = tree.query_ball_point(points, r=radius)

    means = np.empty(num_points, dtype=np.float64)
    for point_index, neighbors in enumerate(neighbor_lists):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        if neighbors.size < min_neighbors:
            means[point_index] = values[point_index]
        else:
            means[point_index] = float(np.mean(values[neighbors]))
    return means


def compute_local_robust_z(
    points: np.ndarray,
    values: np.ndarray,
    r_inner: float,
    r_outer: float,
    min_neighbors: int = 8,
) -> np.ndarray:
    """Local robust z-score using annulus neighborhood median and MAD."""
    points = _validate_points(points)
    values = _validate_1d_finite(values, name="values", length=points.shape[0])
    r_inner = float(r_inner)
    r_outer = float(r_outer)
    min_neighbors = int(min_neighbors)
    if r_inner < 0.0 or r_outer <= 0.0 or r_inner >= r_outer:
        raise ValueError("Require 0 <= r_inner < r_outer.")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be positive.")

    num_points = points.shape[0]
    if num_points == 0:
        return np.zeros(0, dtype=np.float64)

    tree = cKDTree(points)
    try:
        neighbor_lists = tree.query_ball_point(points, r=r_outer, workers=-1)
    except TypeError:
        neighbor_lists = tree.query_ball_point(points, r=r_outer)

    z_scores = np.zeros(num_points, dtype=np.float64)
    for point_index, neighbors in enumerate(neighbor_lists):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        if neighbors.size == 0:
            continue

        deltas = points[neighbors] - points[point_index]
        distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        annulus_mask = (distances > r_inner) & (distances <= r_outer)
        annulus_indices = neighbors[annulus_mask]
        if annulus_indices.size < min_neighbors:
            continue

        local_values = values[annulus_indices]
        local_median = float(np.median(local_values))
        mad = float(np.median(np.abs(local_values - local_median)))
        z_scores[point_index] = (values[point_index] - local_median) / (
            _ROBUST_Z_SCALE * mad + 1e-12
        )
    return z_scores


def graph_hop_distance_to_mask(
    seed_mask: np.ndarray,
    neighbor_indices: list[np.ndarray] | np.ndarray,
    max_hops: int = 5,
) -> np.ndarray:
    """Multi-source graph hop distance from ``seed_mask`` points."""
    seed_mask = np.asarray(seed_mask, dtype=bool).reshape(-1)
    max_hops = int(max_hops)
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative.")
    if seed_mask.size == 0:
        return np.zeros(0, dtype=np.float64)

    neighbor_lists = _coerce_neighbor_indices(neighbor_indices, seed_mask.size)
    unreachable = float(max_hops + 1)
    distances = np.full(seed_mask.size, unreachable, dtype=np.float64)

    queue: deque[int] = deque()
    for seed_index in np.flatnonzero(seed_mask):
        distances[seed_index] = 0.0
        queue.append(int(seed_index))

    while queue:
        center = queue.popleft()
        next_distance = distances[center] + 1.0
        if next_distance > float(max_hops):
            continue
        for neighbor in neighbor_lists[center]:
            if distances[neighbor] > next_distance:
                distances[neighbor] = next_distance
                queue.append(int(neighbor))
    return distances


def compute_hks_tsf(
    points: np.ndarray,
    hks_features: np.ndarray,
    time_scales: np.ndarray,
    c_abs: np.ndarray | None = None,
    boundary_mask: np.ndarray | None = None,
    neighbor_indices: list[np.ndarray] | np.ndarray | None = None,
    core_radius: float = 3.0,
    rim_inner_radius: float = 3.0,
    rim_outer_radius: float = 8.0,
    z_inner_radius: float = 3.0,
    z_outer_radius: float = 12.0,
) -> dict[str, Any]:
    """Compute HKS Thermal Spectral Fingerprint scores for each point.

    Fusion policy (V0.6.7): Surface Variation is the primary detector; HKS topo
    and boundary scores act as structural priors. ``proposal_score`` is driven
    by ``var_clean_score``, with weak boosts from gated cold/ring candidates.
    """
    points = _validate_points(points)
    hks_features = _validate_hks_features(hks_features, num_points=points.shape[0])
    time_scales = _validate_time_scales(time_scales, num_columns=hks_features.shape[1])

    num_points = points.shape[0]
    num_times = hks_features.shape[1]

    z_hks = np.zeros((num_points, num_times), dtype=np.float64)
    for time_index in range(num_times):
        z_hks[:, time_index] = compute_local_robust_z(
            points,
            hks_features[:, time_index],
            r_inner=z_inner_radius,
            r_outer=z_outer_radius,
        )

    selected_columns = {
        int(target_t): _nearest_time_column(time_scales, target_t)
        for target_t in _TARGET_TIMES
    }
    z_1 = z_hks[:, selected_columns[1]]
    z_100 = z_hks[:, selected_columns[100]]
    z_1000 = z_hks[:, selected_columns[1000]]
    z_10000 = z_hks[:, selected_columns[10000]]

    cold_score = 0.7 * relu(-z_100) + 1.0 * relu(-z_1000)

    core_100 = compute_ball_mean(points, z_100, radius=core_radius)
    rim_100 = compute_annulus_mean(
        points,
        z_100,
        r_inner=rim_inner_radius,
        r_outer=rim_outer_radius,
    )
    ring_100 = relu(core_100 - rim_100) * relu(-rim_100)

    core_1000 = compute_ball_mean(points, z_1000, radius=core_radius)
    rim_1000 = compute_annulus_mean(
        points,
        z_1000,
        r_inner=rim_inner_radius,
        r_outer=rim_outer_radius,
    )
    ring_1000 = relu(core_1000 - rim_1000) * relu(-rim_1000)
    ring_score = ring_100 + 0.5 * ring_1000

    topo_score = relu(z_10000)

    if boundary_mask is not None and neighbor_indices is not None:
        boundary_mask_arr = np.asarray(boundary_mask, dtype=bool).reshape(-1)
        if boundary_mask_arr.shape[0] != num_points:
            raise ValueError("boundary_mask length must match points.")
        hop_dist = graph_hop_distance_to_mask(
            boundary_mask_arr,
            neighbor_indices,
            max_hops=5,
        )
        near_boundary = hop_dist <= 3.0
        boundary_layer_score = near_boundary * (relu(z_1) + relu(-z_1000))
    else:
        boundary_layer_score = np.zeros(num_points, dtype=np.float64)

    if c_abs is not None:
        c_abs_arr = _validate_1d_finite(c_abs, name="c_abs", length=num_points)
        c_abs_input = c_abs_arr
        var_raw_score = robust_normalize(c_abs_arr)
        z_c_abs = compute_local_robust_z(
            points,
            c_abs_arr,
            r_inner=z_inner_radius,
            r_outer=z_outer_radius,
        )
        var_local_score = robust_normalize(relu(z_c_abs))
        var_base_score = 0.85 * var_raw_score + 0.15 * var_local_score
        var_base_score = np.clip(var_base_score, 0.0, 1.0)
    else:
        c_abs_input = None
        z_c_abs = np.zeros(num_points, dtype=np.float64)
        var_raw_score = np.zeros(num_points, dtype=np.float64)
        var_local_score = np.zeros(num_points, dtype=np.float64)
        var_base_score = np.zeros(num_points, dtype=np.float64)

    norm_topo = robust_normalize(topo_score)
    norm_boundary = robust_normalize(boundary_layer_score)

    body_support = norm_topo
    body_surround = compute_annulus_mean(
        points,
        body_support,
        r_inner=z_inner_radius,
        r_outer=z_outer_radius,
    )
    body_surround = np.clip(body_surround, 0.0, 1.0)

    var_clean_score = var_base_score.copy()
    var_clean_score *= 1.0 - 0.70 * norm_boundary
    var_clean_score *= 0.5 + 0.5 * body_surround
    var_clean_score = np.clip(var_clean_score, 0.0, 1.0)

    norm_cold = robust_normalize(cold_score)
    norm_ring = robust_normalize(ring_score)

    cold_candidate = norm_cold * body_surround * (1.0 - 0.5 * body_support)
    cold_candidate *= 1.0 - 0.70 * norm_boundary
    cold_candidate = np.clip(cold_candidate, 0.0, 1.0)

    ring_candidate = norm_ring * body_surround * (1.0 - 0.50 * norm_boundary)
    ring_candidate = np.clip(ring_candidate, 0.0, 1.0)

    proposal_score = np.maximum.reduce(
        [
            var_clean_score,
            0.20 * cold_candidate,
            0.25 * ring_candidate,
        ],
    )
    proposal_score = np.clip(proposal_score, 0.0, 1.0)

    var_score = var_clean_score

    summary = {
        "num_points": int(num_points),
        "num_time_scales": int(num_times),
        "time_scales": [float(t) for t in time_scales],
        "selected_columns": {
            str(int(target_t)): {
                "column": int(column),
                "time_scale": float(time_scales[column]),
            }
            for target_t, column in selected_columns.items()
        },
        "cold_score": _score_summary(cold_score),
        "ring_score": _score_summary(ring_score),
        "topo_score": _score_summary(topo_score),
        "boundary_layer_score": _score_summary(boundary_layer_score),
        "norm_topo": _score_summary(norm_topo),
        "norm_boundary": _score_summary(norm_boundary),
        "var_raw_score": _score_summary(var_raw_score),
        "var_local_score": _score_summary(var_local_score),
        "var_base_score": _score_summary(var_base_score),
        "var_clean_score": _score_summary(var_clean_score),
        "var_score": _score_summary(var_score),
        "body_support": _score_summary(body_support),
        "body_surround": _score_summary(body_surround),
        "cold_candidate": _score_summary(cold_candidate),
        "ring_candidate": _score_summary(ring_candidate),
        "proposal_score": _score_summary(proposal_score),
        "has_c_abs": c_abs is not None,
        "has_boundary": boundary_mask is not None and neighbor_indices is not None,
    }

    return {
        "cold_score": cold_score,
        "ring_score": ring_score,
        "topo_score": topo_score,
        "boundary_layer_score": boundary_layer_score,
        "c_abs_input": c_abs_input,
        "z_c_abs": z_c_abs,
        "var_raw_score": var_raw_score,
        "var_local_score": var_local_score,
        "var_base_score": var_base_score,
        "var_clean_score": var_clean_score,
        "var_score": var_score,
        "body_support": body_support,
        "body_surround": body_surround,
        "cold_candidate": cold_candidate,
        "ring_candidate": ring_candidate,
        "norm_boundary": norm_boundary,
        "norm_topo": norm_topo,
        "proposal_score": proposal_score,
        "z_hks": z_hks,
        "summary": summary,
    }


def _nearest_time_column(time_scales: np.ndarray, target_t: float) -> int:
    """Return the column index whose time scale is nearest to ``target_t`` in log space."""
    target_t = float(target_t)
    if target_t <= 0.0 or not np.isfinite(target_t):
        raise ValueError("target_t must be a positive finite value.")
    log_scales = np.log(np.asarray(time_scales, dtype=np.float64).reshape(-1))
    log_target = np.log(target_t)
    return int(np.argmin(np.abs(log_scales - log_target)))


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Validate point cloud shape and numeric values."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")
    return points


def _validate_1d_finite(
    values: np.ndarray,
    name: str,
    length: int | None = None,
) -> np.ndarray:
    """Validate a one-dimensional finite numeric array."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if length is not None and values.shape[0] != int(length):
        raise ValueError(f"{name} length must match points ({length}).")
    if values.size == 0 and length not in (0, None):
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _validate_hks_features(hks_features: np.ndarray, num_points: int) -> np.ndarray:
    """Validate HKS feature matrix."""
    hks_features = np.asarray(hks_features, dtype=np.float64)
    if hks_features.ndim != 2:
        raise ValueError("hks_features must have shape (N, T).")
    if hks_features.shape[0] != num_points:
        raise ValueError("hks_features row count must match points.")
    if hks_features.shape[1] == 0:
        raise ValueError("hks_features must contain at least one time column.")
    if not np.isfinite(hks_features).all():
        raise ValueError("hks_features must contain only finite values.")
    return hks_features


def _validate_time_scales(time_scales: np.ndarray, num_columns: int) -> np.ndarray:
    """Validate diffusion time scales."""
    time_scales = np.asarray(time_scales, dtype=np.float64).reshape(-1)
    if time_scales.size != num_columns:
        raise ValueError("time_scales length must match hks_features column count.")
    if time_scales.size == 0:
        raise ValueError("time_scales must contain at least one value.")
    if not np.isfinite(time_scales).all():
        raise ValueError("time_scales must contain only finite values.")
    if np.any(time_scales <= 0.0):
        raise ValueError("time_scales must be strictly positive.")
    if not np.all(np.diff(time_scales) > 0.0):
        raise ValueError("time_scales must be strictly increasing.")
    return time_scales


def _coerce_neighbor_indices(
    neighbor_indices: list[np.ndarray] | np.ndarray,
    num_points: int,
) -> list[np.ndarray]:
    """Normalize neighbor indices to a per-point list of int64 arrays."""
    if isinstance(neighbor_indices, list):
        if len(neighbor_indices) != num_points:
            raise ValueError("neighbor_indices length must match num_points.")
        neighbor_lists = [
            np.asarray(entry, dtype=np.int64).reshape(-1) for entry in neighbor_indices
        ]
    else:
        arr = np.asarray(neighbor_indices)
        if arr.ndim != 2 or arr.shape[0] != num_points:
            raise ValueError("neighbor_indices must have shape (N, K).")
        neighbor_lists = [row.astype(np.int64, copy=False) for row in arr]

    filtered_lists: list[np.ndarray] = []
    for neighbors in neighbor_lists:
        if neighbors.size == 0:
            filtered_lists.append(np.empty(0, dtype=np.int64))
            continue
        valid_mask = (neighbors >= 0) & (neighbors < num_points)
        filtered_lists.append(neighbors[valid_mask])
    return filtered_lists


def _score_summary(values: np.ndarray) -> dict[str, float]:
    """Compact summary statistics for one score vector."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p95": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
    }
