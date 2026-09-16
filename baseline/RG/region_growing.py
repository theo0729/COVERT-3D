"""Index-preserving Python reimplementation of PCL-style Region Growing.

PCL is not installed in the project environment.  This module follows the PCL
theoretical primer: points are visited in ascending curvature order, neighbours
are admitted by a normal-angle smoothness constraint, and admitted points below
the curvature threshold become new seeds.  The largest final region is normal;
all remaining points are anomalies, as described by the 3D-SONAR RG baseline.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import tempfile
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.spatial import cKDTree

from baseline.common import BaselineResult


CURVATURE_THEORETICAL_UPPER_BOUND = 1.0 / 3.0


def _validate_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("points must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError("points contain NaN or Inf.")
    return array


def _knn_indices(points: np.ndarray, k: int, workers: int) -> np.ndarray:
    if k < 3:
        raise ValueError("k must be at least 3.")
    if k >= points.shape[0]:
        raise ValueError("k must be smaller than the number of points.")
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k, workers=workers)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape != (points.shape[0], k):
        raise RuntimeError(f"Unexpected KNN shape: {indices.shape}")
    return indices


def _pca_from_neighbours(
    points: np.ndarray,
    neighbour_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbourhoods = points[neighbour_indices]
    centroids = neighbourhoods.mean(axis=1)
    centered = neighbourhoods - centroids[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True)
    covariance /= float(neighbour_indices.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normals = eigenvectors[:, :, 0]
    normal_norms = np.linalg.norm(normals, axis=1)
    eigenvalue_sum = eigenvalues.sum(axis=1)
    invalid = (
        (~np.isfinite(eigenvalues).all(axis=1))
        | (~np.isfinite(normals).all(axis=1))
        | (normal_norms <= 1e-12)
        | (eigenvalue_sum <= 1e-15)
    )
    if np.any(invalid):
        raise RuntimeError(
            f"Local PCA produced {int(np.count_nonzero(invalid))} invalid points."
        )
    normals = normals / normal_norms[:, None]
    curvature = eigenvalues[:, 0] / eigenvalue_sum
    return normals, curvature, centroids


def estimate_normals_and_curvature(
    points: np.ndarray,
    k: int = 50,
    query_workers: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate PCA normals/curvature and return the exact KNN indices used."""

    array = _validate_points(points)
    k = int(k)
    query_workers = int(query_workers)
    started = time.perf_counter()
    neighbour_indices = _knn_indices(array, k=k, workers=query_workers)
    knn_sec = float(time.perf_counter() - started)
    started = time.perf_counter()
    normals, curvature, _ = _pca_from_neighbours(array, neighbour_indices)
    pca_sec = float(time.perf_counter() - started)
    diagnostics = {
        "normal_k": k,
        "query_workers": query_workers,
        "normal_knn_sec": knn_sec,
        "normal_pca_sec": pca_sec,
        "curvature_min": float(curvature.min()),
        "curvature_max": float(curvature.max()),
        "curvature_mean": float(curvature.mean()),
        "curvature_p50": float(np.quantile(curvature, 0.50)),
        "curvature_p95": float(np.quantile(curvature, 0.95)),
    }
    return normals, curvature, neighbour_indices, diagnostics


def local_plane_projection_smooth(
    points: np.ndarray,
    k: int = 30,
    iterations: int = 1,
    query_workers: int = -1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply index-preserving local-plane projection smoothing.

    This is deliberately not described as Moving Least Squares: it performs a
    single PCA-plane projection per input index and does not fit an MLS surface.
    """

    original = _validate_points(points)
    k = int(k)
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if iterations == 0:
        return original.copy(), {
            "smoothing": "none",
            "smoothing_k": k,
            "smoothing_iterations": 0,
            "smoothing_sec": 0.0,
            "smoothing_mean_displacement": 0.0,
            "smoothing_max_displacement": 0.0,
        }

    started = time.perf_counter()
    smoothed = original.copy()
    for _ in range(iterations):
        neighbour_indices = _knn_indices(smoothed, k=k, workers=query_workers)
        normals, _, centroids = _pca_from_neighbours(smoothed, neighbour_indices)
        signed_distance = np.einsum(
            "ni,ni->n", smoothed - centroids, normals, optimize=True
        )
        smoothed = smoothed - signed_distance[:, None] * normals
    elapsed = float(time.perf_counter() - started)
    displacement = np.linalg.norm(smoothed - original, axis=1)
    return smoothed, {
        "smoothing": "index_preserving_local_plane_projection",
        "smoothing_k": k,
        "smoothing_iterations": iterations,
        "smoothing_sec": elapsed,
        "smoothing_mean_displacement": float(displacement.mean()),
        "smoothing_max_displacement": float(displacement.max()),
    }


def curvature_gate_diagnostics(
    curvature: np.ndarray,
    curvature_threshold: float,
    *,
    emit_warning: bool = True,
) -> dict[str, Any]:
    """Describe whether the PCA-curvature seed gate can discriminate points."""

    values = np.asarray(curvature, dtype=np.float64).reshape(-1)
    threshold = float(curvature_threshold)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("curvature must be a non-empty finite array.")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("curvature_threshold must be finite and non-negative.")
    gate_active = threshold < CURVATURE_THEORETICAL_UPPER_BOUND - 1e-12
    fraction_below = float(np.mean(values < threshold))
    if not gate_active and emit_warning:
        warnings.warn(
            "curvature_threshold is at or above the theoretical 1/3 upper "
            "bound of lambda_min / sum(lambda); curvature seed gating is inactive.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "curvature_gate_active": bool(gate_active),
        "curvature_gate_effective_on_sample": bool(fraction_below < 1.0),
        "curvature_theoretical_upper_bound": CURVATURE_THEORETICAL_UPPER_BOUND,
        "fraction_below_curvature_threshold": fraction_below,
    }


def rg_feature_cache_key(
    *,
    sample_identity: str,
    preprocess_config_hash: str,
    smoothing_k: int,
    smoothing_iterations: int,
    normal_k: int,
    region_neighbor_k: int,
    query_workers: int,
    fps_selected_index_hash: str | None = None,
) -> str:
    """Return a deterministic key for all threshold-independent RG features."""

    payload = {
        "schema": "rg_feature_cache_v1",
        "sample_identity": str(sample_identity),
        "preprocess_config_hash": str(preprocess_config_hash),
        "smoothing": "index_preserving_local_plane_projection",
        "smoothing_k": int(smoothing_k),
        "smoothing_iterations": int(smoothing_iterations),
        "normal_estimation": "local_pca",
        "normal_k": int(normal_k),
        "region_neighbor_k": int(region_neighbor_k),
        "query_workers": int(query_workers),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    if fps_selected_index_hash is not None:
        payload["fps_selected_index_hash"] = str(fps_selected_index_hash)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def compute_region_growing_features(
    points: np.ndarray,
    *,
    smoothing_k: int = 30,
    smoothing_iterations: int = 1,
    normal_k: int = 50,
    region_neighbor_k: int = 30,
    query_workers: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute threshold-independent smoothed normals, curvature, and KNN."""

    array = _validate_points(points)
    total_started = time.perf_counter()
    smoothed, smoothing_diagnostics = local_plane_projection_smooth(
        array,
        k=smoothing_k,
        iterations=smoothing_iterations,
        query_workers=query_workers,
    )
    normals, curvature, _, normal_diagnostics = estimate_normals_and_curvature(
        smoothed,
        k=normal_k,
        query_workers=query_workers,
    )
    started = time.perf_counter()
    region_neighbours = _knn_indices(
        smoothed,
        k=int(region_neighbor_k) + 1,
        workers=int(query_workers),
    )[:, 1:]
    region_knn_sec = float(time.perf_counter() - started)
    diagnostics = {
        **smoothing_diagnostics,
        **normal_diagnostics,
        "region_neighbor_k": int(region_neighbor_k),
        "region_knn_sec": region_knn_sec,
        "feature_preparation_sec": float(time.perf_counter() - total_started),
    }
    return normals, curvature, region_neighbours, diagnostics


def load_or_compute_region_growing_features(
    points: np.ndarray,
    *,
    cache_dir: str | Path,
    sample_identity: str,
    preprocess_config_hash: str,
    smoothing_k: int = 30,
    smoothing_iterations: int = 1,
    normal_k: int = 50,
    region_neighbor_k: int = 30,
    query_workers: int = -1,
    fps_selected_index_hash: str | None = None,
    fallback_cache_dirs: Iterable[str | Path] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load or atomically create a GT-free RG intermediate-feature cache."""

    array = _validate_points(points)
    key = rg_feature_cache_key(
        sample_identity=sample_identity,
        preprocess_config_hash=preprocess_config_hash,
        smoothing_k=smoothing_k,
        smoothing_iterations=smoothing_iterations,
        normal_k=normal_k,
        region_neighbor_k=region_neighbor_k,
        query_workers=query_workers,
        fps_selected_index_hash=fps_selected_index_hash,
    )
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"{key}.npz"
    candidates: list[tuple[Path, str, str]] = [(cache_path, key, "v2")]
    fallback_roots = [Path(item).expanduser().resolve() for item in fallback_cache_dirs]
    for fallback_root in fallback_roots:
        candidates.append((fallback_root / f"{key}.npz", key, "parent_v1"))
    if fps_selected_index_hash is not None:
        legacy_key = rg_feature_cache_key(
            sample_identity=sample_identity,
            preprocess_config_hash=preprocess_config_hash,
            smoothing_k=smoothing_k,
            smoothing_iterations=smoothing_iterations,
            normal_k=normal_k,
            region_neighbor_k=region_neighbor_k,
            query_workers=query_workers,
            fps_selected_index_hash=None,
        )
        for fallback_root in fallback_roots:
            candidates.append((fallback_root / f"{legacy_key}.npz", legacy_key, "parent_v1"))

    selected = next(
        ((path, expected_key, source) for path, expected_key, source in candidates if path.is_file()),
        None,
    )
    if selected is not None:
        selected_path, expected_key, source = selected
        with np.load(selected_path, allow_pickle=False) as data:
            cached_key = str(np.asarray(data["cache_key"]).reshape(()).item())
            normals = np.asarray(data["normals"], dtype=np.float64)
            curvature = np.asarray(data["curvature"], dtype=np.float64)
            neighbours = np.asarray(data["region_neighbours"], dtype=np.int64)
            diagnostics = json.loads(
                str(np.asarray(data["diagnostics_json"]).reshape(()).item())
            )
        if cached_key != expected_key:
            raise RuntimeError(f"RG feature cache-key mismatch: {selected_path}")
        if normals.shape != array.shape or curvature.shape != (array.shape[0],):
            raise RuntimeError("Cached RG feature shapes do not match the input cloud.")
        diagnostics.update(
            {
                "feature_cache_key": expected_key,
                "feature_cache_path": str(selected_path),
                "feature_cache_hit": True,
                "feature_cache_source": source,
                "fps_selected_index_hash": fps_selected_index_hash or "legacy_unspecified",
            }
        )
        return normals, curvature, neighbours, diagnostics

    normals, curvature, neighbours, diagnostics = compute_region_growing_features(
        array,
        smoothing_k=smoothing_k,
        smoothing_iterations=smoothing_iterations,
        normal_k=normal_k,
        region_neighbor_k=region_neighbor_k,
        query_workers=query_workers,
    )
    payload_diagnostics = dict(diagnostics)
    payload_diagnostics.update(
        {
            "feature_cache_key": key,
            "feature_cache_path": str(cache_path),
            "feature_cache_hit": False,
            "feature_cache_source": "recomputed",
            "fps_selected_index_hash": fps_selected_index_hash or "legacy_unspecified",
        }
    )
    with tempfile.NamedTemporaryFile(
        dir=root, prefix=f".{key}.", suffix=".tmp.npz", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        np.savez_compressed(
            temporary_path,
            normals=normals,
            curvature=curvature,
            region_neighbours=neighbours,
            cache_key=np.asarray(key),
            diagnostics_json=np.asarray(
                json.dumps(payload_diagnostics, sort_keys=True, separators=(",", ":"))
            ),
        )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return normals, curvature, neighbours, payload_diagnostics


def region_grow_from_features(
    normals: np.ndarray,
    curvature: np.ndarray,
    neighbour_indices: np.ndarray,
    smoothness_threshold_deg: float,
    curvature_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the PCL theoretical-primer region-growing state machine."""

    normals = np.asarray(normals, dtype=np.float64)
    curvature = np.asarray(curvature, dtype=np.float64).reshape(-1)
    neighbour_indices = np.asarray(neighbour_indices, dtype=np.int64)
    num_points = curvature.shape[0]
    if normals.shape != (num_points, 3):
        raise ValueError("normals must have shape (N, 3).")
    if neighbour_indices.ndim != 2 or neighbour_indices.shape[0] != num_points:
        raise ValueError("neighbour_indices must have shape (N, K).")
    if not np.isfinite(normals).all() or not np.isfinite(curvature).all():
        raise ValueError("normals and curvature must be finite.")
    if neighbour_indices.size:
        if neighbour_indices.min() < 0 or neighbour_indices.max() >= num_points:
            raise ValueError("neighbour_indices contain out-of-range values.")
    smoothness_threshold_deg = float(smoothness_threshold_deg)
    curvature_threshold = float(curvature_threshold)
    if not 0.0 < smoothness_threshold_deg <= 180.0:
        raise ValueError("smoothness_threshold_deg must be within (0, 180].")
    if not np.isfinite(curvature_threshold) or curvature_threshold < 0.0:
        raise ValueError("curvature_threshold must be finite and non-negative.")

    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(normal_norms <= 1e-12):
        raise ValueError("normals contain zero-length vectors.")
    unit_normals = normals / normal_norms[:, None]
    cos_threshold = math.cos(math.radians(smoothness_threshold_deg))
    order = np.argsort(curvature, kind="stable")
    labels = np.full(num_points, -1, dtype=np.int32)
    region_sizes: list[int] = []

    for initial_seed in order:
        initial_seed = int(initial_seed)
        if labels[initial_seed] >= 0:
            continue
        region_id = len(region_sizes)
        labels[initial_seed] = region_id
        queue: deque[int] = deque([initial_seed])
        region_size = 1
        while queue:
            seed = queue.popleft()
            neighbours = neighbour_indices[seed]
            candidates = neighbours[labels[neighbours] < 0]
            if candidates.size == 0:
                continue
            dots = np.abs(unit_normals[candidates] @ unit_normals[seed])
            accepted = candidates[dots >= cos_threshold]
            if accepted.size == 0:
                continue
            labels[accepted] = region_id
            region_size += int(accepted.size)
            new_seeds = accepted[curvature[accepted] < curvature_threshold]
            queue.extend(int(value) for value in new_seeds)
        region_sizes.append(region_size)

    sizes = np.asarray(region_sizes, dtype=np.int64)
    if np.any(labels < 0):
        raise RuntimeError("Region growing left unassigned points.")
    if sizes.sum() != num_points:
        raise RuntimeError("Region sizes do not sum to the number of input points.")
    return labels, sizes


def classify_region_growing_features(
    normals: np.ndarray,
    curvature: np.ndarray,
    region_neighbours: np.ndarray,
    *,
    smoothness_threshold_deg: float,
    curvature_threshold: float,
    min_cluster_size: int = 1,
    emit_curvature_warning: bool = True,
) -> BaselineResult:
    """Grow cached features and mark every non-largest valid region anomalous."""

    curvature = np.asarray(curvature, dtype=np.float64).reshape(-1)
    min_cluster_size = int(min_cluster_size)
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive.")
    gate_diagnostics = curvature_gate_diagnostics(
        curvature, curvature_threshold, emit_warning=emit_curvature_warning
    )
    started = time.perf_counter()
    labels, region_sizes = region_grow_from_features(
        normals,
        curvature,
        region_neighbours,
        smoothness_threshold_deg=smoothness_threshold_deg,
        curvature_threshold=curvature_threshold,
    )
    region_growing_sec = float(time.perf_counter() - started)
    valid_region_ids = np.flatnonzero(region_sizes >= min_cluster_size)
    if valid_region_ids.size == 0:
        normal_region_id = -1
        pred_mask = np.ones(curvature.shape[0], dtype=bool)
    else:
        valid_sizes = region_sizes[valid_region_ids]
        normal_region_id = int(valid_region_ids[int(np.argmax(valid_sizes))])
        pred_mask = labels != normal_region_id

    sorted_sizes = np.sort(region_sizes)[::-1]
    largest_size = int(sorted_sizes[0]) if sorted_sizes.size else 0
    second_size = int(sorted_sizes[1]) if sorted_sizes.size > 1 else 0
    diagnostics = {
        "implementation": "RG (PCL-style Python reimplementation)",
        "source_algorithm": "PCL RegionGrowing theoretical primer",
        **gate_diagnostics,
        "smoothness_threshold_deg": float(smoothness_threshold_deg),
        "curvature_threshold": float(curvature_threshold),
        "normal_angle_rule": "acos(abs(dot(n_seed, n_neighbor)))",
        "min_cluster_size": min_cluster_size,
        "normal_region_rule": "largest_valid_cluster",
        "unassigned_points_rule": "anomaly",
        "region_growing_sec": region_growing_sec,
        "num_regions": int(region_sizes.size),
        "num_valid_regions": int(valid_region_ids.size),
        "num_regions_below_min_size": int(
            np.count_nonzero(region_sizes < min_cluster_size)
        ),
        "num_singleton_regions": int(np.count_nonzero(region_sizes == 1)),
        "normal_region_id": normal_region_id,
        "largest_region_points": largest_size,
        "largest_region_ratio": float(largest_size / curvature.shape[0]),
        "second_largest_region_points": second_size,
        "second_largest_region_ratio": float(second_size / curvature.shape[0]),
        "top10_region_sizes": [int(value) for value in sorted_sizes[:10]],
        "num_unassigned_points": int(np.count_nonzero(labels < 0)),
        "num_predicted_defect": int(np.count_nonzero(pred_mask)),
        "predicted_defect_ratio": float(np.mean(pred_mask)),
    }
    return BaselineResult(
        pred_mask=np.asarray(pred_mask, dtype=bool),
        runtime_sec=region_growing_sec,
        diagnostics=diagnostics,
    )


def run_region_growing_baseline(
    points: np.ndarray,
    smoothing_k: int = 30,
    smoothing_iterations: int = 1,
    normal_k: int = 50,
    region_neighbor_k: int = 30,
    smoothness_threshold_deg: float = 3.0,
    curvature_threshold: float = 1.0,
    min_cluster_size: int = 1,
    query_workers: int = -1,
) -> BaselineResult:
    """Run the complete RG baseline and mark all non-largest regions anomalous."""

    array = _validate_points(points)
    total_started = time.perf_counter()
    normals, curvature, region_neighbours, feature_diagnostics = (
        compute_region_growing_features(
            array,
            smoothing_k=smoothing_k,
            smoothing_iterations=smoothing_iterations,
            normal_k=normal_k,
            region_neighbor_k=region_neighbor_k,
            query_workers=query_workers,
        )
    )
    classification = classify_region_growing_features(
        normals,
        curvature,
        region_neighbours,
        smoothness_threshold_deg=smoothness_threshold_deg,
        curvature_threshold=curvature_threshold,
        min_cluster_size=min_cluster_size,
        emit_curvature_warning=True,
    )
    runtime_sec = float(time.perf_counter() - total_started)
    diagnostics = {
        **feature_diagnostics,
        **classification.diagnostics,
        "region_neighbor_k": int(region_neighbor_k),
    }
    return BaselineResult(
        pred_mask=classification.pred_mask,
        runtime_sec=runtime_sec,
        diagnostics=diagnostics,
    )
