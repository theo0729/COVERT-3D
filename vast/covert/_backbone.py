"""Private production runtime mechanically migrated from the frozen reference.

Only symbols reachable from the active consolidated call root are retained.
"""
from __future__ import annotations
import heapq
import hashlib
import json
import math
import sys
import time
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Sequence
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from vast.data import io as data_io
from vast.data import preprocess
from vast.evaluation import scoring
from vast.features import hks
from vast.features.surface_variation import compute_surface_variation
from vast.graph import edge_weight, laplacian
from vast.graph.mutual_knn import build_mutual_knn_graph
try:
    import numba
except ModuleNotFoundError:
    numba = None

class _NullRuntimeProfiler:
    """Disabled profiler stub so exp14 stays usable without exp17."""
    enabled = False

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[None]:
        yield

    def begin_stage(self, name: str, **meta: Any) -> None:
        return None

    def end_stage(self, token: Any) -> None:
        return None

def _resolve_exp14_profiler(profiler: Any | None) -> Any:
    if profiler is None:
        return _NullRuntimeProfiler()
    return profiler

def _exp14_begin_stage(profiler: Any, name: str, *, include: bool=True, **meta: Any) -> Any:
    if not include:
        return None
    begin = getattr(profiler, 'begin_stage', None)
    if callable(begin):
        return begin(name, **meta)
    cm = profiler.stage(name, **meta)
    cm.__enter__()
    return {'_cm': cm, 'name': name}

def _exp14_end_stage(profiler: Any, token: Any) -> None:
    if token is None:
        return
    if isinstance(token, dict) and '_cm' in token:
        token['_cm'].__exit__(None, None, None)
        return
    end = getattr(profiler, 'end_stage', None)
    if callable(end):
        end(token)
NUM_EIGENVALUES = 100
LOCAL_RADII = [3.0, 6.0, 12.0]
LOCAL_CONTRAST_QUERY_WORKERS = -1
KNN_SMOOTH_ITERS = 2
HEATMAP_P_LOW = 1.0
HEATMAP_P_HIGH = 99.0
PURE_HKS_WEIGHT_MODE = 'gaussian_distance'
PURE_HKS_SIGMA_MODE = 'median_edge_distance'
SGCR_GROWTH_MODE = 'energy_descent'
SGCR_SEED_RELATIVE_GROW_RATIO = 0.49
SGCR_SEED_RELATIVE_GROW_USE_SUPPORT_MIN_NORM = True
SGCR_SEED_RELATIVE_GROW_KEEP_EQUAL = True
SGCR_ENERGY_DESCENT_REQUIRE_LOWER_THAN_CURRENT = True
SGCR_ENERGY_DESCENT_EPS = 1e-06
SGCR_ENERGY_DESCENT_ALLOW_PLATEAU = False
SGCR_ENERGY_DESCENT_FLOOR_MODE = 'seed_relative'
SGCR_ENERGY_DESCENT_USE_PRIORITY_QUEUE = True

def _ablation_name(variant: str, gamma: float) -> str:
    """Build the canonical ablation result name used in exp13."""
    return f'{variant}_gamma_{gamma:g}'

def _enhance_surface_variation(c_abs: np.ndarray) -> np.ndarray:
    """Apply a stable nonlinear boost to surface variation for flat-region sensitivity."""
    c_abs_safe = np.clip(np.asarray(c_abs, dtype=np.float64).reshape(-1), 0.0, None)
    c_abs_enhanced = np.power(c_abs_safe, 0.5)
    return np.nan_to_num(c_abs_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

def _identity_surface_variation(c_abs: np.ndarray) -> np.ndarray:
    """Return raw non-enhanced surface variation feature for semantic gating/debug."""
    c_abs_safe = np.clip(np.asarray(c_abs, dtype=np.float64).reshape(-1), 0.0, None)
    c_abs_raw_feature = np.power(c_abs_safe, 1.0)
    return np.nan_to_num(c_abs_raw_feature, nan=0.0, posinf=0.0, neginf=0.0)

def _score_summary(values: np.ndarray) -> dict[str, float]:
    """Build min/max/mean/percentile summary for one score vector."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0}
    return {'min': float(np.min(values)), 'max': float(np.max(values)), 'mean': float(np.mean(values)), 'p50': float(np.percentile(values, 50.0)), 'p90': float(np.percentile(values, 90.0)), 'p95': float(np.percentile(values, 95.0)), 'p99': float(np.percentile(values, 99.0))}

class LocalContrastNeighborCache:
    """Shared deterministic radius-neighborhood cache for local contrast."""

    def __init__(self, *, radii: tuple[float, ...], neighbor_indices_by_radius: dict[int, list[np.ndarray]], num_points: int, include_self: bool, points_signature: str | None, num_radius_queries: int, primary_num_radius_queries: int, reference_num_radius_queries: int, actual_num_radius_queries: int, kdtree_build_count: int, build_wall_time_sec: float, query_workers_requested: int, query_workers_effective: int, query_workers_fallback_used: bool, query_workers_fallback_reason: str | None, neighbor_order_exact_vs_workers1: bool | None, num_different_neighbor_lists_vs_workers1: int | None) -> None:
        self.radii = radii
        self.neighbor_indices_by_radius = neighbor_indices_by_radius
        self.num_points = int(num_points)
        self.include_self = bool(include_self)
        self.points_signature = points_signature
        self.num_radius_queries = int(num_radius_queries)
        self.primary_num_radius_queries = int(primary_num_radius_queries)
        self.reference_num_radius_queries = int(reference_num_radius_queries)
        self.actual_num_radius_queries = int(actual_num_radius_queries)
        self.kdtree_build_count = int(kdtree_build_count)
        self.build_wall_time_sec = float(build_wall_time_sec)
        self.query_workers_requested = int(query_workers_requested)
        self.query_workers_effective = int(query_workers_effective)
        self.query_workers_fallback_used = bool(query_workers_fallback_used)
        self.query_workers_fallback_reason = query_workers_fallback_reason
        self.neighbor_order_exact_vs_workers1 = None if neighbor_order_exact_vs_workers1 is None else bool(neighbor_order_exact_vs_workers1)
        self.num_different_neighbor_lists_vs_workers1 = None if num_different_neighbor_lists_vs_workers1 is None else int(num_different_neighbor_lists_vs_workers1)
        self.cache_hit_views: list[str] = []
        self.fallback_used = False
        self.fallback_reason: str | None = None

    def register_view_hit(self, view_name: str) -> None:
        if view_name not in self.cache_hit_views:
            self.cache_hit_views.append(view_name)

    def mark_fallback(self, reason: str) -> None:
        self.fallback_used = True
        if self.fallback_reason is None:
            self.fallback_reason = str(reason)

def _points_signature(points: np.ndarray) -> str:
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    return hashlib.md5(points.tobytes()).hexdigest()

def _normalize_neighbor_lists(raw_neighbor_lists: Sequence[Any]) -> list[np.ndarray]:
    return [np.asarray(neighbors, dtype=np.int64) for neighbors in raw_neighbor_lists]

def _count_different_neighbor_lists(lhs_lists: Sequence[np.ndarray], rhs_lists: Sequence[np.ndarray]) -> int:
    if len(lhs_lists) != len(rhs_lists):
        return abs(len(lhs_lists) - len(rhs_lists))
    num_different = 0
    for (lhs, rhs) in zip(lhs_lists, rhs_lists, strict=True):
        if not np.array_equal(np.asarray(lhs, dtype=np.int64).reshape(-1), np.asarray(rhs, dtype=np.int64).reshape(-1)):
            num_different += 1
    return int(num_different)

def build_local_contrast_neighbor_cache(points: np.ndarray, radii: Sequence[float], *, include_self: bool=True, query_workers: int=LOCAL_CONTRAST_QUERY_WORKERS, verify_against_workers1: bool=False, profiler: Any | None=None, include_substages: bool=True) -> LocalContrastNeighborCache:
    """Build shared local-contrast radius neighborhoods once for all base views."""
    profiler = _resolve_exp14_profiler(profiler)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')
    if not radii:
        raise ValueError('radii must contain at least one radius value.')
    if not include_self:
        raise ValueError('Local-contrast cache currently requires include_self=True.')
    normalized_radii = tuple((float(r) for r in radii))
    requested_workers = int(query_workers)
    effective_workers = int(requested_workers)
    query_workers_fallback_used = False
    query_workers_fallback_reason: str | None = None
    verify_with_workers1 = bool(verify_against_workers1)
    wall_start = time.perf_counter()
    _tok = _exp14_begin_stage(profiler, 'local_contrast_kdtree_build', include=include_substages, view_name='shared_local_contrast_cache')
    tree = cKDTree(points)
    _exp14_end_stage(profiler, _tok)

    def _query_neighbors_for_radius(radius: float, *, workers: int, view_name: str) -> list[np.ndarray]:
        _tok_local = _exp14_begin_stage(profiler, 'local_contrast_radius_query', include=include_substages, view_name=view_name, radius=float(radius), workers=int(workers))
        raw_lists = None
        try:
            try:
                raw_lists = tree.query_ball_point(points, r=float(radius), workers=int(workers))
            except TypeError:
                if int(workers) != 1:
                    raise
                raw_lists = tree.query_ball_point(points, r=float(radius))
        finally:
            _exp14_end_stage(profiler, _tok_local)
        return _normalize_neighbor_lists(raw_lists)
    neighbor_indices_by_radius: dict[int, list[np.ndarray]] = {}
    workers1_reference_by_radius: dict[int, list[np.ndarray]] = {}
    primary_num_radius_queries = 0
    reference_num_radius_queries = 0
    for (radius_index, radius) in enumerate(normalized_radii):
        try:
            neighbor_indices_by_radius[radius_index] = _query_neighbors_for_radius(float(radius), workers=int(effective_workers), view_name='shared_local_contrast_cache')
            primary_num_radius_queries += 1
        except TypeError as exc:
            if int(effective_workers) == 1:
                raise
            effective_workers = 1
            query_workers_fallback_used = True
            query_workers_fallback_reason = f'query_ball_point workers unsupported, fallback to workers=1: {exc}'
            neighbor_indices_by_radius[radius_index] = _query_neighbors_for_radius(float(radius), workers=1, view_name='shared_local_contrast_cache')
            primary_num_radius_queries += 1
        if verify_with_workers1 and int(effective_workers) != 1:
            workers1_reference_by_radius[radius_index] = _query_neighbors_for_radius(float(radius), workers=1, view_name='shared_local_contrast_cache_workers1_reference')
            reference_num_radius_queries += 1
    neighbor_order_exact_vs_workers1: bool | None = None
    num_different_neighbor_lists_vs_workers1: int | None = None
    if verify_with_workers1 and int(effective_workers) != 1:
        diff_total = 0
        for radius_index in range(len(normalized_radii)):
            diff_total += _count_different_neighbor_lists(neighbor_indices_by_radius.get(radius_index, []), workers1_reference_by_radius.get(radius_index, []))
        num_different_neighbor_lists_vs_workers1 = int(diff_total)
        neighbor_order_exact_vs_workers1 = bool(diff_total == 0)
    actual_num_radius_queries = int(primary_num_radius_queries + reference_num_radius_queries)
    return LocalContrastNeighborCache(radii=normalized_radii, neighbor_indices_by_radius=neighbor_indices_by_radius, num_points=int(points.shape[0]), include_self=bool(include_self), points_signature=_points_signature(points), num_radius_queries=int(actual_num_radius_queries), primary_num_radius_queries=int(primary_num_radius_queries), reference_num_radius_queries=int(reference_num_radius_queries), actual_num_radius_queries=int(actual_num_radius_queries), kdtree_build_count=1, build_wall_time_sec=float(time.perf_counter() - wall_start), query_workers_requested=int(requested_workers), query_workers_effective=int(effective_workers), query_workers_fallback_used=bool(query_workers_fallback_used), query_workers_fallback_reason=query_workers_fallback_reason, neighbor_order_exact_vs_workers1=neighbor_order_exact_vs_workers1, num_different_neighbor_lists_vs_workers1=num_different_neighbor_lists_vs_workers1)

def _validate_local_contrast_neighbor_cache(*, points: np.ndarray, radii: Sequence[float], include_self: bool, neighbor_cache: LocalContrastNeighborCache) -> tuple[bool, str | None]:
    if not include_self:
        return (False, 'include_self_mismatch')
    if int(points.shape[0]) != int(neighbor_cache.num_points):
        return (False, 'num_points_mismatch')
    expected_radii = tuple((float(r) for r in radii))
    if len(expected_radii) != len(neighbor_cache.radii):
        return (False, 'radii_length_mismatch')
    for (idx, radius) in enumerate(expected_radii):
        if float(radius) != float(neighbor_cache.radii[idx]):
            return (False, 'radius_value_mismatch')
    points_sig = _points_signature(points)
    if neighbor_cache.points_signature is not None and points_sig != neighbor_cache.points_signature:
        return (False, 'points_signature_mismatch')
    return (True, None)

def _compute_local_contrast_from_neighbors(scores: np.ndarray, neighbor_lists: list[np.ndarray]) -> np.ndarray:
    """Compute clipped local contrast (score minus neighborhood mean) using given neighbors."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mu_local = np.empty(scores.shape[0], dtype=np.float64)
    for (point_index, neighbors) in enumerate(neighbor_lists):
        neighbor_scores = scores[np.asarray(neighbors, dtype=np.int64)]
        mu_local[point_index] = float(np.mean(neighbor_scores))
    contrast_score = scores - mu_local
    contrast_score = np.nan_to_num(contrast_score, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(contrast_score, 0.0, None).astype(np.float64, copy=False)

def _compute_local_contrast_at_radius(points: np.ndarray, scores: np.ndarray, radius: float, *, neighbor_lists: list[np.ndarray] | None=None, profiler: Any | None=None, include_substages: bool=True, view_name: str | None=None) -> np.ndarray:
    """Compute clipped local contrast (score minus neighborhood mean) within radius R."""
    profiler = _resolve_exp14_profiler(profiler)
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if points.shape[0] != scores.shape[0]:
        raise ValueError('points and scores must contain the same number of samples.')
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3).')
    if radius <= 0.0:
        raise ValueError('radius must be positive.')
    if neighbor_lists is None:
        _tok = _exp14_begin_stage(profiler, 'local_contrast_kdtree_build', include=include_substages, view_name=view_name, radius=float(radius))
        tree = cKDTree(points)
        _exp14_end_stage(profiler, _tok)
        _tok = _exp14_begin_stage(profiler, 'local_contrast_radius_query', include=include_substages, view_name=view_name, radius=float(radius), workers=1)
        try:
            raw_neighbor_lists = tree.query_ball_point(points, r=float(radius), workers=1)
        except TypeError:
            raw_neighbor_lists = tree.query_ball_point(points, r=float(radius))
        _exp14_end_stage(profiler, _tok)
        neighbor_lists = _normalize_neighbor_lists(raw_neighbor_lists)
    _tok = _exp14_begin_stage(profiler, 'local_contrast_apply_neighbors', include=include_substages, view_name=view_name, radius=float(radius))
    contrast = _compute_local_contrast_from_neighbors(scores, neighbor_lists)
    _exp14_end_stage(profiler, _tok)
    return contrast

def _aggregate_multiscale_local_contrast(points: np.ndarray, scores: np.ndarray, radii: list[float] | tuple[float, ...]=LOCAL_RADII, *, neighbor_cache: LocalContrastNeighborCache | None=None, profiler: Any | None=None, include_substages: bool=True, view_name: str='unknown_view') -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate multi-radius local contrast via per-point max consensus."""
    profiler = _resolve_exp14_profiler(profiler)
    if not radii:
        raise ValueError('radii must contain at least one radius value.')
    use_cache = False
    cache_fallback_reason: str | None = None
    if neighbor_cache is not None:
        (cache_ok, cache_reason) = _validate_local_contrast_neighbor_cache(points=np.asarray(points, dtype=np.float64), radii=radii, include_self=True, neighbor_cache=neighbor_cache)
        if cache_ok:
            use_cache = True
            neighbor_cache.register_view_hit(view_name)
        else:
            cache_fallback_reason = cache_reason
            neighbor_cache.mark_fallback(str(cache_reason))
    _tok_total = _exp14_begin_stage(profiler, 'base_view_local_contrast_total', include=include_substages, view_name=view_name, used_cache=bool(use_cache))
    contrast_per_radius: list[np.ndarray] = []
    radius_summaries: dict[str, dict[str, float]] = {}
    for (radius_index, radius) in enumerate(radii):
        _tok = _exp14_begin_stage(profiler, 'base_view_local_contrast_radius_query', include=include_substages, view_name=view_name, radius=float(radius), used_cache=bool(use_cache))
        cached_neighbors = None
        if use_cache and neighbor_cache is not None:
            cached_neighbors = neighbor_cache.neighbor_indices_by_radius.get(radius_index)
        _exp14_end_stage(profiler, _tok)
        _tok = _exp14_begin_stage(profiler, 'base_view_local_contrast_score_apply', include=include_substages, view_name=view_name, radius=float(radius), used_cache=bool(cached_neighbors is not None))
        contrast_radius = _compute_local_contrast_at_radius(points, scores, radius=radius, neighbor_lists=cached_neighbors, profiler=profiler, include_substages=include_substages, view_name=view_name)
        _exp14_end_stage(profiler, _tok)
        contrast_per_radius.append(contrast_radius)
        radius_summaries[f'radius_{radius:g}'] = _score_summary(contrast_radius)
    final_scores = np.max(np.stack(contrast_per_radius, axis=0), axis=0)
    _exp14_end_stage(profiler, _tok_total)
    aggregate_summary = {'local_radii': [float(r) for r in radii], 'aggregation': 'max', 'used_neighbor_cache': bool(use_cache), 'cache_fallback_reason': cache_fallback_reason, 'per_radius': radius_summaries, 'final': _score_summary(final_scores)}
    return (final_scores.astype(np.float64, copy=False), aggregate_summary)

def _knn_spatial_smooth_scores(scores: np.ndarray, neighbor_indices: list[np.ndarray], iterations: int=2) -> np.ndarray:
    """Low-pass filter scores by averaging each point with its mutual-KNN neighbors."""
    smoothed = np.asarray(scores, dtype=np.float64).reshape(-1)
    num_points = smoothed.shape[0]
    for _ in range(iterations):
        next_scores = np.empty(num_points, dtype=np.float64)
        for (point_index, neighbors) in enumerate(neighbor_indices):
            hood = np.concatenate(([point_index], np.asarray(neighbors, dtype=np.int64).reshape(-1)))
            next_scores[point_index] = float(np.mean(smoothed[hood]))
        smoothed = next_scores
    return smoothed

def _safe_zscore_features(features: np.ndarray) -> np.ndarray:
    """Column-wise z-score with stable handling for near-constant feature columns."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.asarray(std, dtype=np.float64).copy()
    std[std < 1e-12] = 1.0
    return (features - mean) / std

def _percentile_minmax_normalize(values: np.ndarray, p_low: float=HEATMAP_P_LOW, p_high: float=HEATMAP_P_HIGH) -> np.ndarray:
    """
    Normalize scores to [0, 1] using the same percentile min-max rule as exp13 heatmaps.

    This is the normalized score map used for component thresholds and penalty levels.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values.astype(np.float64, copy=False)
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    if hi <= lo:
        return np.zeros(values.shape[0], dtype=np.float64)
    normalized = (values - lo) / (hi - lo + 1e-12)
    return np.clip(normalized, 0.0, 1.0).astype(np.float64, copy=False)

def _build_gaussian_distance_weights(adjacency_dist: sparse.csr_matrix) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build pure distance Gaussian edge weights without using c_abs."""
    adjacency_dist = adjacency_dist.tocsr()
    coo = adjacency_dist.tocoo()
    if coo.nnz == 0:
        empty = sparse.csr_matrix(adjacency_dist.shape, dtype=np.float64)
        summary = {'weight_mode': PURE_HKS_WEIGHT_MODE, 'sigma_mode': PURE_HKS_SIGMA_MODE, 'sigma': 1.0, 'num_edges': 0, 'weight_min': 0.0, 'weight_max': 0.0, 'weight_mean': 0.0}
        return (empty, summary)
    distances = coo.data.astype(np.float64, copy=False)
    positive_distances = distances[distances > 0.0]
    if positive_distances.size > 0:
        sigma = float(np.median(positive_distances))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.mean(positive_distances))
    else:
        sigma = 1.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    weights = np.exp(-distances ** 2 / (2.0 * sigma ** 2))
    distance_weight = sparse.coo_matrix((weights, (coo.row, coo.col)), shape=adjacency_dist.shape, dtype=np.float64).tocsr()
    distance_weight.eliminate_zeros()
    pure_graph_summary = {'weight_mode': PURE_HKS_WEIGHT_MODE, 'sigma_mode': PURE_HKS_SIGMA_MODE, 'sigma': float(sigma), 'num_edges': int(distance_weight.nnz), 'weight_min': float(weights.min()) if weights.size else 0.0, 'weight_max': float(weights.max()) if weights.size else 0.0, 'weight_mean': float(weights.mean()) if weights.size else 0.0}
    return (distance_weight, pure_graph_summary)

def _eigen_summary(eigenvalues: np.ndarray) -> dict[str, Any]:
    """Build a compact eigenvalue summary."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if eigenvalues.size == 0:
        return {'num_eigenvalues': 0, 'lambda_min': 0.0, 'lambda_max': 0.0}
    return {'num_eigenvalues': int(eigenvalues.size), 'lambda_min': float(eigenvalues[0]), 'lambda_max': float(eigenvalues[-1])}

def _compute_pure_hks(adjacency_dist: sparse.csr_matrix, profiler: Any | None=None, *, include_substages: bool=True, num_eigenvalues: int=NUM_EIGENVALUES) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any], dict[str, Any]]:
    """Compute HKS on a pure Gaussian distance graph."""
    profiler = _resolve_exp14_profiler(profiler)
    _tok = _exp14_begin_stage(profiler, '03_03_hks_graph_weights', include=include_substages, graph='pure')
    (distance_weight, pure_graph_summary) = _build_gaussian_distance_weights(adjacency_dist)
    laplacian_pure = laplacian.build_normalized_laplacian(distance_weight)
    _exp14_end_stage(profiler, _tok)
    _tok = _exp14_begin_stage(profiler, '03_04_hks_eigendecomposition', include=include_substages, graph='pure')
    (eigenvalues_pure, eigenvectors_pure) = hks.compute_eigen_decomposition(laplacian_pure, num_eigenvalues=int(num_eigenvalues))
    _exp14_end_stage(profiler, _tok)
    _tok = _exp14_begin_stage(profiler, '03_05_hks_feature_maps', include=include_substages, graph='pure')
    (t_small_pure, t_mid_pure, t_large_pure) = hks.compute_adaptive_time_scales(eigenvalues_pure)
    hks_pure = hks.compute_hks(eigenvalues_pure, eigenvectors_pure, time_scales=(t_small_pure, t_mid_pure, t_large_pure))
    time_scales_pure = (float(t_small_pure), float(t_mid_pure), float(t_large_pure))
    eigen_summary_pure = _eigen_summary(eigenvalues_pure)
    _exp14_end_stage(profiler, _tok)
    return (hks_pure, time_scales_pure, pure_graph_summary, eigen_summary_pure)

def _compute_variation_aware_hks(adjacency_dist: sparse.csr_matrix, c_abs: np.ndarray, gamma: float, profiler: Any | None=None, *, include_substages: bool=True, num_eigenvalues: int=NUM_EIGENVALUES) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any], dict[str, Any]]:
    """Compute HKS on a variation-aware graph for one gamma."""
    profiler = _resolve_exp14_profiler(profiler)
    _tok = _exp14_begin_stage(profiler, '03_03_hks_graph_weights', include=include_substages, graph='variation_aware', gamma=float(gamma))
    adjacency_weight_va = edge_weight.compute_variation_aware_weights(adjacency_dist, c_abs, gamma=gamma)
    laplacian_va = laplacian.build_normalized_laplacian(adjacency_weight_va)
    _exp14_end_stage(profiler, _tok)
    _tok = _exp14_begin_stage(profiler, '03_04_hks_eigendecomposition', include=include_substages, graph='variation_aware', gamma=float(gamma))
    (eigenvalues_va, eigenvectors_va) = hks.compute_eigen_decomposition(laplacian_va, num_eigenvalues=int(num_eigenvalues))
    _exp14_end_stage(profiler, _tok)
    _tok = _exp14_begin_stage(profiler, '03_05_hks_feature_maps', include=include_substages, graph='variation_aware', gamma=float(gamma))
    (t_small_va, t_mid_va, t_large_va) = hks.compute_adaptive_time_scales(eigenvalues_va)
    hks_va = hks.compute_hks(eigenvalues_va, eigenvectors_va, time_scales=(t_small_va, t_mid_va, t_large_va))
    time_scales_va = (float(t_small_va), float(t_mid_va), float(t_large_va))
    graph_summary_va = {'gamma': float(gamma), 'num_edges': int(adjacency_weight_va.nnz), 'weight_min': float(adjacency_weight_va.data.min()) if adjacency_weight_va.nnz else 0.0, 'weight_max': float(adjacency_weight_va.data.max()) if adjacency_weight_va.nnz else 0.0, 'weight_mean': float(adjacency_weight_va.data.mean()) if adjacency_weight_va.nnz else 0.0}
    eigen_summary_va = _eigen_summary(eigenvalues_va)
    _exp14_end_stage(profiler, _tok)
    return (hks_va, time_scales_va, graph_summary_va, eigen_summary_va)

def _compute_ablation_score(name: str, view_name: str, features: np.ndarray, points: np.ndarray, neighbor_indices: list[np.ndarray], gamma: float, gamma_effective: bool, feature_source: str, *, local_contrast_neighbor_cache: LocalContrastNeighborCache | None=None, profiler: Any | None=None, include_substages: bool=True, knn_smooth_iterations: int=KNN_SMOOTH_ITERS, local_radii: Sequence[float]=LOCAL_RADII) -> dict[str, Any]:
    """Run the unified z-score -> Mahalanobis -> smooth -> local-contrast scoring chain."""
    profiler = _resolve_exp14_profiler(profiler)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    _tok = _exp14_begin_stage(profiler, 'base_view_zscore', include=include_substages, view_name=view_name)
    features_z = _safe_zscore_features(features)
    _exp14_end_stage(profiler, _tok)
    _tok = _exp14_begin_stage(profiler, 'base_view_score_raw', include=include_substages, view_name=view_name)
    if features_z.shape[1] == 1:
        try:
            score_raw = scoring.compute_mahalanobis_score(features_z)
            if not np.isfinite(score_raw).all():
                raise ValueError('non-finite Mahalanobis scores for 1D features')
        except (ValueError, np.linalg.LinAlgError):
            score_raw = np.abs(features_z[:, 0])
    else:
        score_raw = scoring.compute_mahalanobis_score(features_z)
    _exp14_end_stage(profiler, _tok)
    _tok = _exp14_begin_stage(profiler, 'base_view_knn_smooth', include=include_substages, view_name=view_name)
    score_smooth = _knn_spatial_smooth_scores(score_raw, neighbor_indices, iterations=int(knn_smooth_iterations))
    _exp14_end_stage(profiler, _tok)
    (score_local, local_contrast_summary) = _aggregate_multiscale_local_contrast(points, score_smooth, radii=tuple((float(value) for value in local_radii)), neighbor_cache=local_contrast_neighbor_cache, profiler=profiler, include_substages=include_substages, view_name=view_name)
    return {'name': name, 'view_name': view_name, 'gamma': float(gamma), 'gamma_effective': bool(gamma_effective), 'feature_source': feature_source, 'features_raw': features, 'features_z': features_z, 'score_raw': score_raw, 'score_smooth': score_smooth, 'score_local': score_local, 'summary': {'name': name, 'gamma': float(gamma), 'gamma_effective': bool(gamma_effective), 'feature_source': feature_source, 'num_features': int(features.shape[1]), 'raw': _score_summary(score_raw), 'smooth': _score_summary(score_smooth), 'local': _score_summary(score_local), 'local_contrast': local_contrast_summary}}

def _connected_components_from_mask(mask: np.ndarray, neighbor_indices: list[np.ndarray]) -> list[np.ndarray]:
    """Find connected components inside a boolean mask on the mutual-KNN graph."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    visited = np.zeros(mask.shape[0], dtype=bool)
    components: list[np.ndarray] = []
    for start_index in np.flatnonzero(mask):
        start_index = int(start_index)
        if visited[start_index]:
            continue
        queue: deque[int] = deque([start_index])
        visited[start_index] = True
        component = [start_index]
        while queue:
            point_index = queue.popleft()
            for neighbor_index in np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1):
                neighbor_index = int(neighbor_index)
                if not mask[neighbor_index] or visited[neighbor_index]:
                    continue
                visited[neighbor_index] = True
                component.append(neighbor_index)
                queue.append(neighbor_index)
        components.append(np.asarray(component, dtype=np.int64))
    return components

def _graph_dilate_indices(seed_indices: np.ndarray, neighbor_indices: list[np.ndarray], num_points: int, hops: int) -> np.ndarray:
    """Return all point indices within given graph hops from seed_indices."""
    seed_indices = np.unique(np.asarray(seed_indices, dtype=np.int64).reshape(-1))
    hops = int(hops)
    num_points = int(num_points)
    if seed_indices.size == 0:
        return seed_indices
    if hops <= 0:
        return seed_indices
    visited = np.zeros(num_points, dtype=bool)
    visited[seed_indices] = True
    frontier = seed_indices.tolist()
    for _ in range(hops):
        if not frontier:
            break
        next_frontier: list[int] = []
        for point_index in frontier:
            for neighbor_index in np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1):
                neighbor_index = int(neighbor_index)
                if visited[neighbor_index]:
                    continue
                visited[neighbor_index] = True
                next_frontier.append(neighbor_index)
        frontier = next_frontier
    return np.flatnonzero(visited).astype(np.int64, copy=False)

def _grow_one_sgcr_component(seed_indices: np.ndarray, sv_norm: np.ndarray, neighbor_indices: list[np.ndarray], max_grow_hops: int, local_bg_percentile: float, seed_level_percentile: float, alpha_candidates: tuple[float, ...], support_min_norm: float, max_component_size: int, growth_mode: str=SGCR_GROWTH_MODE, seed_relative_grow_ratio: float=SGCR_SEED_RELATIVE_GROW_RATIO, seed_relative_use_support_min_norm: bool=SGCR_SEED_RELATIVE_GROW_USE_SUPPORT_MIN_NORM, seed_relative_keep_equal: bool=SGCR_SEED_RELATIVE_GROW_KEEP_EQUAL, energy_descent_require_lower_than_current: bool=SGCR_ENERGY_DESCENT_REQUIRE_LOWER_THAN_CURRENT, energy_descent_eps: float=SGCR_ENERGY_DESCENT_EPS, energy_descent_allow_plateau: bool=SGCR_ENERGY_DESCENT_ALLOW_PLATEAU, energy_descent_floor_mode: str=SGCR_ENERGY_DESCENT_FLOOR_MODE, energy_descent_use_priority_queue: bool=SGCR_ENERGY_DESCENT_USE_PRIORITY_QUEUE, compute_inactive_local_background: bool=True) -> dict[str, Any]:
    """
    Grow one seed component in the normalized enhanced-SV map.

    J only provides seed_indices.
    sv_norm provides the growth support.
    """
    seed_indices = np.unique(np.asarray(seed_indices, dtype=np.int64).reshape(-1))
    sv_norm = np.asarray(sv_norm, dtype=np.float64).reshape(-1)
    num_points = int(sv_norm.shape[0])
    max_grow_hops = int(max_grow_hops)
    max_component_size = int(max_component_size)
    local_window_indices = _graph_dilate_indices(seed_indices, neighbor_indices, num_points=num_points, hops=max_grow_hops)
    local_window_mask = np.zeros(num_points, dtype=bool)
    local_window_mask[local_window_indices] = True
    seed_level = float(np.percentile(sv_norm[seed_indices], seed_level_percentile))
    local_bg_level: float | None = None
    if compute_inactive_local_background or growth_mode != 'energy_descent':
        seed_index_set = set(seed_indices.tolist())
        local_bg_indices = np.asarray([index for index in local_window_indices if index not in seed_index_set], dtype=np.int64)
        if local_bg_indices.size > 0:
            local_bg_level = float(np.percentile(sv_norm[local_bg_indices], local_bg_percentile))
        else:
            local_bg_level = float(np.percentile(sv_norm, local_bg_percentile))

    def _bfs_grow(support_threshold: float, keep_equal: bool) -> np.ndarray:
        grown_mask = np.zeros(num_points, dtype=bool)
        hop_distance = np.full(num_points, -1, dtype=np.int32)
        queue: deque[int] = deque()
        for seed_index in seed_indices:
            seed_index = int(seed_index)
            grown_mask[seed_index] = True
            hop_distance[seed_index] = 0
            queue.append(seed_index)
        while queue:
            point_index = queue.popleft()
            current_hop = int(hop_distance[point_index])
            if current_hop >= max_grow_hops:
                continue
            for neighbor_index in np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1):
                neighbor_index = int(neighbor_index)
                if grown_mask[neighbor_index]:
                    continue
                if not local_window_mask[neighbor_index]:
                    continue
                if current_hop + 1 > max_grow_hops:
                    continue
                neighbor_score = float(sv_norm[neighbor_index])
                if keep_equal:
                    if neighbor_score < support_threshold:
                        continue
                elif neighbor_score <= support_threshold:
                    continue
                grown_mask[neighbor_index] = True
                hop_distance[neighbor_index] = current_hop + 1
                queue.append(neighbor_index)
        return np.flatnonzero(grown_mask).astype(np.int64, copy=False)

    def _neighbor_passes_energy_descent(current_score: float, neighbor_score: float, support_threshold: float) -> bool:
        if neighbor_score < support_threshold:
            return False
        if not energy_descent_require_lower_than_current:
            return True
        if energy_descent_allow_plateau:
            return neighbor_score <= current_score + float(energy_descent_eps)
        return neighbor_score < current_score - float(energy_descent_eps)

    def _energy_descent_grow(support_threshold: float) -> np.ndarray:
        grown_mask = np.zeros(num_points, dtype=bool)
        hop_distance = np.full(num_points, -1, dtype=np.int32)
        for seed_index in seed_indices:
            seed_index = int(seed_index)
            grown_mask[seed_index] = True
            hop_distance[seed_index] = 0
        if energy_descent_use_priority_queue:
            frontier_heap: list[tuple[float, int, int]] = []
            for seed_index in seed_indices:
                seed_index = int(seed_index)
                heapq.heappush(frontier_heap, (-float(sv_norm[seed_index]), 0, seed_index))
            while frontier_heap:
                (_, current_hop, point_index) = heapq.heappop(frontier_heap)
                if not grown_mask[point_index]:
                    continue
                if int(hop_distance[point_index]) != current_hop:
                    continue
                if current_hop >= max_grow_hops:
                    continue
                current_score = float(sv_norm[point_index])
                for neighbor_index in np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1):
                    neighbor_index = int(neighbor_index)
                    if grown_mask[neighbor_index]:
                        continue
                    if not local_window_mask[neighbor_index]:
                        continue
                    neighbor_score = float(sv_norm[neighbor_index])
                    if not _neighbor_passes_energy_descent(current_score, neighbor_score, support_threshold):
                        continue
                    next_hop = current_hop + 1
                    if next_hop > max_grow_hops:
                        continue
                    grown_mask[neighbor_index] = True
                    hop_distance[neighbor_index] = next_hop
                    heapq.heappush(frontier_heap, (-neighbor_score, next_hop, neighbor_index))
        else:
            queue: deque[int] = deque((int(seed_index) for seed_index in seed_indices))
            while queue:
                point_index = queue.popleft()
                current_hop = int(hop_distance[point_index])
                if current_hop >= max_grow_hops:
                    continue
                current_score = float(sv_norm[point_index])
                for neighbor_index in np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1):
                    neighbor_index = int(neighbor_index)
                    if grown_mask[neighbor_index]:
                        continue
                    if not local_window_mask[neighbor_index]:
                        continue
                    neighbor_score = float(sv_norm[neighbor_index])
                    if not _neighbor_passes_energy_descent(current_score, neighbor_score, support_threshold):
                        continue
                    next_hop = current_hop + 1
                    if next_hop > max_grow_hops:
                        continue
                    grown_mask[neighbor_index] = True
                    hop_distance[neighbor_index] = next_hop
                    queue.append(neighbor_index)
        return np.flatnonzero(grown_mask).astype(np.int64, copy=False)

    def _resolve_energy_descent_support_threshold(seed_self_energy: float) -> float:
        floor_mode = str(energy_descent_floor_mode).strip().lower()
        if floor_mode == 'seed_relative':
            support_threshold = seed_self_energy * float(seed_relative_grow_ratio)
            if seed_relative_use_support_min_norm:
                support_threshold = max(float(support_min_norm), support_threshold)
            return float(support_threshold)
        raise ValueError("energy_descent_floor_mode must be 'seed_relative' in this version.")
    if growth_mode == 'energy_descent':
        seed_self_energy = float(np.percentile(sv_norm[seed_indices], seed_level_percentile))
        support_threshold = _resolve_energy_descent_support_threshold(seed_self_energy)
        recovered_indices = _energy_descent_grow(support_threshold)
        return {'indices': recovered_indices, 'growth_mode': 'energy_descent', 'support_threshold': float(support_threshold), 'seed_self_energy': float(seed_self_energy), 'seed_relative_grow_ratio': float(seed_relative_grow_ratio), 'energy_descent_require_lower_than_current': bool(energy_descent_require_lower_than_current), 'energy_descent_eps': float(energy_descent_eps), 'energy_descent_allow_plateau': bool(energy_descent_allow_plateau), 'energy_descent_floor_mode': str(energy_descent_floor_mode), 'energy_descent_use_priority_queue': bool(energy_descent_use_priority_queue), 'seed_level': seed_level, 'local_bg_level': local_bg_level, 'alpha': None, 'size': int(recovered_indices.size), 'seed_size': int(seed_indices.size), 'oversized': bool(recovered_indices.size > max_component_size)}
    if growth_mode == 'seed_relative':
        seed_self_energy = float(np.percentile(sv_norm[seed_indices], seed_level_percentile))
        support_threshold = seed_self_energy * float(seed_relative_grow_ratio)
        if seed_relative_use_support_min_norm:
            support_threshold = max(float(support_min_norm), support_threshold)
        recovered_indices = _bfs_grow(support_threshold, keep_equal=seed_relative_keep_equal)
        return {'indices': recovered_indices, 'growth_mode': 'seed_relative', 'support_threshold': float(support_threshold), 'seed_self_energy': float(seed_self_energy), 'seed_relative_grow_ratio': float(seed_relative_grow_ratio), 'seed_level': seed_level, 'local_bg_level': local_bg_level, 'alpha': None, 'size': int(recovered_indices.size), 'seed_size': int(seed_indices.size), 'oversized': bool(recovered_indices.size > max_component_size)}
    if growth_mode != 'adaptive_alpha':
        raise ValueError("growth_mode must be 'seed_relative', 'energy_descent', or 'adaptive_alpha'.")
    if local_bg_level is None:
        raise AssertionError('adaptive-alpha growth requires a local background level.')
    best_oversized_report: dict[str, Any] | None = None
    for alpha in alpha_candidates:
        support_threshold = local_bg_level + float(alpha) * (seed_level - local_bg_level)
        support_threshold = max(float(support_min_norm), support_threshold)
        support_threshold = min(support_threshold, max(seed_level * 0.95, float(support_min_norm)))
        recovered_indices = _bfs_grow(support_threshold, keep_equal=True)
        report = {'indices': recovered_indices, 'growth_mode': 'adaptive_alpha', 'support_threshold': float(support_threshold), 'seed_self_energy': float(seed_level), 'seed_relative_grow_ratio': None, 'alpha': float(alpha), 'seed_level': seed_level, 'local_bg_level': local_bg_level, 'size': int(recovered_indices.size), 'seed_size': int(seed_indices.size), 'oversized': bool(recovered_indices.size > max_component_size)}
        if not report['oversized']:
            return report
        if best_oversized_report is None or report['size'] < best_oversized_report['size']:
            best_oversized_report = report
    if best_oversized_report is None:
        return {'indices': seed_indices.copy(), 'growth_mode': 'adaptive_alpha', 'support_threshold': float(support_min_norm), 'seed_self_energy': float(seed_level), 'seed_relative_grow_ratio': None, 'alpha': float(alpha_candidates[0]) if alpha_candidates else 0.0, 'seed_level': seed_level, 'local_bg_level': local_bg_level, 'size': int(seed_indices.size), 'seed_size': int(seed_indices.size), 'oversized': False}
    return best_oversized_report

def _components_to_label_array(components: list[np.ndarray], num_points: int) -> np.ndarray:
    labels = np.zeros(num_points, dtype=np.int32)
    for (component_id, indices) in enumerate(components, start=1):
        indices = np.asarray(indices, dtype=np.int64)
        labels[indices] = int(component_id)
    return labels

def _build_mainline_base_views(*, c_abs_enhanced: np.ndarray, hks_pure: np.ndarray, hks_va: np.ndarray, points: np.ndarray, neighbor_indices: list[np.ndarray], feature_config: Any, query_workers: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build only the three score maps consumed by the paper mainline."""
    radii = tuple((float(value) for value in feature_config.local_contrast_radii))
    neighbor_cache = None
    if bool(feature_config.use_local_contrast_neighbor_cache):
        neighbor_cache = build_local_contrast_neighbor_cache(points=points, radii=radii, include_self=True, query_workers=int(query_workers), profiler=None, include_substages=False)
    specs = (('c_abs_only', c_abs_enhanced.reshape(-1, 1), False, 'c_abs_enhanced'), ('pure_hks_only', np.column_stack((hks_pure[:, 0], hks_pure[:, 1], hks_pure[:, 2])), False, 'pure_hks'), ('variation_aware_hks_only', np.column_stack((hks_va[:, 0], hks_va[:, 1], hks_va[:, 2])), True, 'variation_aware_hks'))
    views: dict[str, dict[str, Any]] = {}
    for (variant, features, gamma_effective, source) in specs:
        name = _ablation_name(variant, float(feature_config.va_hks_gamma))
        result = _compute_ablation_score(name=name, view_name=variant, features=features, points=points, neighbor_indices=neighbor_indices, gamma=float(feature_config.va_hks_gamma), gamma_effective=gamma_effective, feature_source=source, local_contrast_neighbor_cache=neighbor_cache, profiler=None, include_substages=False, knn_smooth_iterations=int(feature_config.knn_smooth_iterations), local_radii=radii)
        score_local = np.asarray(result['score_local'], dtype=np.float64).reshape(-1)
        views[variant] = {'variant': variant, 'name': name, 'score_local': score_local, 'score_norm': _percentile_minmax_normalize(score_local), 'ablation_result': result}
    cache_summary = {'enabled': bool(feature_config.use_local_contrast_neighbor_cache), 'used': bool(neighbor_cache is not None), 'radii': list(radii), 'query_workers_requested': int(query_workers)}
    return (views, cache_summary)

def run_covert_backbone(raw_points: np.ndarray, config: Any, *, query_workers: int=1, sample_id: str='sample') -> dict[str, Any]:
    """Compute the minimal geometric and spectral evidence for Covert."""
    (p, f) = (config.preprocess, config.features)
    raw_points = np.asarray(raw_points, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] != 3:
        raise ValueError('raw_points must have shape (N, 3).')
    if bool(p.use_sor):
        raise NotImplementedError('SOR is outside the frozen Covert mainline.')
    cleaned_points = raw_points.copy()
    sor_indices = np.arange(raw_points.shape[0], dtype=np.int64)
    fps_downsample_start = time.perf_counter()
    (sampled_points, fps_indices, fps_summary) = preprocess.fps_downsample(cleaned_points, num_samples=int(p.fps_num_samples), gt_mask=None, fps_mode=str(p.fps_mode), fps_prefilter_num_points=int(p.fps_prefilter_num_points), seed=int(p.fps_seed), return_fps_summary=True)
    fps_downsample_seconds = time.perf_counter() - fps_downsample_start
    (points, d_max, center, scale) = preprocess.isotropic_scale(sampled_points, cube_size=float(p.cube_size))
    raw_points_aligned = ((raw_points - center) * scale).astype(np.float64, copy=False)
    scale_summary = {'cube_size': float(p.cube_size), 'd_max': float(d_max), 'scale': float(scale), 'center': np.asarray(center, dtype=np.float64).tolist(), 'num_points': int(points.shape[0])}
    (adjacency_dist, neighbor_indices, graph_summary) = build_mutual_knn_graph(points, k=int(f.graph_knn_k), radius_cap_factor=float(f.graph_radius_cap_factor))
    c_abs = compute_surface_variation(points, k_scales=tuple((int(value) for value in f.surface_variation_scales)), fusion='max')
    c_abs_enhanced = _enhance_surface_variation(c_abs)
    c_abs_raw_feature = _identity_surface_variation(c_abs)
    (hks_pure, pure_times, pure_graph, pure_eigen) = _compute_pure_hks(adjacency_dist, profiler=None, include_substages=False, num_eigenvalues=int(f.hks_num_eigenvalues))
    (hks_va, va_times, va_graph, va_eigen) = _compute_variation_aware_hks(adjacency_dist, c_abs, gamma=float(f.va_hks_gamma), profiler=None, include_substages=False, num_eigenvalues=int(f.hks_num_eigenvalues))
    (base_views, cache_summary) = _build_mainline_base_views(c_abs_enhanced=c_abs_enhanced, hks_pure=hks_pure, hks_va=hks_va, points=points, neighbor_indices=neighbor_indices, feature_config=f, query_workers=query_workers)
    return {'raw_points': raw_points, 'raw_points_aligned': raw_points_aligned, 'sor_indices': sor_indices, 'fps_indices': np.asarray(fps_indices, dtype=np.int64), 'points': points, 'gt_mask': None, 'adjacency_dist': adjacency_dist, 'neighbor_indices': neighbor_indices, 'c_abs': c_abs, 'c_abs_enhanced': c_abs_enhanced, 'c_abs_raw_feature': c_abs_raw_feature, 'semantic_sv_base_feature': c_abs_raw_feature, 'hks_pure': hks_pure, 'hks_va_gamma_10': hks_va, 'base_views': base_views, 'summaries': {'load': {'sample_id': str(sample_id), 'num_raw_points': int(raw_points.shape[0])}, 'preprocess': {'fps': fps_summary, 'fps_downsample_seconds': float(fps_downsample_seconds), 'scale': scale_summary}, 'graph': graph_summary, 'pure_distance_graph': pure_graph, 'pure_hks': {'time_scales': list(pure_times), 'eigen': pure_eigen}, 'variation_aware_graph_gamma_10': va_graph, 'variation_aware_hks_gamma_10': {'time_scales': list(va_times), 'eigen': va_eigen}, 'local_contrast_neighbor_cache': cache_summary}}
