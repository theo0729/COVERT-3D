"""Point cloud preprocessing utilities for Phase 1."""

from __future__ import annotations

import hashlib
import time
import warnings
from typing import Any

import numpy as np

o3d = None


def _require_open3d():
    """Load Open3D only when the optional SOR path is selected."""

    global o3d
    if o3d is None:
        try:
            import open3d as loaded_open3d
        except ModuleNotFoundError as exc:  # pragma: no cover - local dependency.
            raise RuntimeError("weak_sor requires Open3D.") from exc
        o3d = loaded_open3d
    return o3d

try:
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    njit = None

FPS_MODE_EXACT_NUMPY = "exact_numpy"
FPS_MODE_EXACT_NUMBA = "exact_numba"
FPS_MODE_VOXEL_THEN_EXACT = "voxel_then_exact"
FPS_MODE_RANDOM_THEN_EXACT = "random_then_exact"
FPS_VALID_MODES = (
    FPS_MODE_EXACT_NUMPY,
    FPS_MODE_EXACT_NUMBA,
    FPS_MODE_VOXEL_THEN_EXACT,
    FPS_MODE_RANDOM_THEN_EXACT,
)

FPS_PREFILTER_NUM_POINTS_DEFAULT = 20_000
FPS_VOXEL_TARGET_RATIO_DEFAULT = 2.0
FPS_RANDOM_SEED_DEFAULT = 0
FPS_FALLBACK_TO_EXACT_DEFAULT = True

_NUMBA_AVAILABLE = njit is not None
_NUMBA_FALLBACK_WARNED = False


if njit is not None:

    @njit(cache=True)
    def _farthest_point_sample_exact_numba_kernel(
        points: np.ndarray,
        num_samples: int,
        start_index: int,
    ) -> np.ndarray:
        """Numba kernel: exact FPS with deterministic first-max tie-breaking."""
        num_points = points.shape[0]
        selected = np.empty(num_samples, dtype=np.int64)
        min_dist_sq = np.empty(num_points, dtype=np.float64)
        for j in range(num_points):
            min_dist_sq[j] = np.inf

        farthest = int(start_index)
        for i in range(num_samples):
            selected[i] = farthest
            cx = points[farthest, 0]
            cy = points[farthest, 1]
            cz = points[farthest, 2]

            for j in range(num_points):
                dx = points[j, 0] - cx
                dy = points[j, 1] - cy
                dz = points[j, 2] - cz
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq < min_dist_sq[j]:
                    min_dist_sq[j] = dist_sq

            best_idx = 0
            best_val = min_dist_sq[0]
            for j in range(1, num_points):
                val = min_dist_sq[j]
                if val > best_val:
                    best_val = val
                    best_idx = j
            farthest = best_idx

        return selected


def fps_downsample(
    points: np.ndarray,
    num_samples: int,
    gt_mask: np.ndarray | None = None,
    seed: int | None = FPS_RANDOM_SEED_DEFAULT,
    fps_mode: str = FPS_MODE_EXACT_NUMPY,
    fps_prefilter_num_points: int = FPS_PREFILTER_NUM_POINTS_DEFAULT,
    fps_seed: int | None = None,
    return_fps_summary: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Downsample a point cloud with configurable FPS backends.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        num_samples: Target number of sampled points.
        gt_mask: Optional per-point binary GT mask aligned with ``points``.
        seed: Legacy seed argument for backward compatibility.
        fps_mode: One of ``exact_numpy``, ``exact_numba``, ``voxel_then_exact``,
            ``random_then_exact``.
        fps_prefilter_num_points: Candidate count for prefilter-based modes.
        fps_seed: New explicit FPS seed; when provided it overrides ``seed``.
        return_fps_summary: If ``True``, return ``(points, indices, summary)``.
            Otherwise return legacy ``(points, indices)``.
    """
    points = _validate_points(points)
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("points must not be empty.")

    if gt_mask is not None:
        gt_mask = _validate_binary_mask(gt_mask, num_points=num_points)

    effective_seed = seed if fps_seed is None else fps_seed
    fps_mode_normalized = str(fps_mode).strip().lower()
    if fps_mode_normalized not in FPS_VALID_MODES:
        raise ValueError(
            f"Unsupported fps_mode='{fps_mode}'. Valid modes: {list(FPS_VALID_MODES)}"
        )

    wall_start = time.perf_counter()
    fps_start_index = 0

    if num_points <= num_samples:
        indices = np.arange(num_points, dtype=np.int64)
        sampled_points = points.copy()
        mode_summary: dict[str, Any] = {
            "fps_mode": fps_mode_normalized,
            "backend": fps_mode_normalized,
            "num_raw_points": int(num_points),
            "num_samples": int(num_samples),
            "prefilter_num_points": int(fps_prefilter_num_points),
            "num_prefiltered_points": int(num_points),
            "selected_index_hash": _selected_index_hash(indices),
            "fallback_used": False,
            "fallback_reason": None,
        }
    elif fps_mode_normalized == FPS_MODE_EXACT_NUMPY:
        indices, fps_start_index = farthest_point_sample_exact_numpy(
            points,
            num_samples=num_samples,
            seed=effective_seed,
        )
        sampled_points = points[indices]
        mode_summary = {
            "fps_mode": FPS_MODE_EXACT_NUMPY,
            "backend": FPS_MODE_EXACT_NUMPY,
            "num_raw_points": int(num_points),
            "num_samples": int(num_samples),
            "prefilter_num_points": int(fps_prefilter_num_points),
            "num_prefiltered_points": int(num_points),
            "selected_index_hash": _selected_index_hash(indices),
            "fallback_used": False,
            "fallback_reason": None,
        }
    elif fps_mode_normalized == FPS_MODE_EXACT_NUMBA:
        indices, fps_start_index, backend_used, fallback_used, fallback_reason = (
            _run_exact_backend_indices(
                points=points,
                num_samples=num_samples,
                seed=effective_seed,
                backend=FPS_MODE_EXACT_NUMBA,
            )
        )
        sampled_points = points[indices]
        mode_summary = {
            "fps_mode": FPS_MODE_EXACT_NUMBA,
            "backend": backend_used,
            "num_raw_points": int(num_points),
            "num_samples": int(num_samples),
            "prefilter_num_points": int(fps_prefilter_num_points),
            "num_prefiltered_points": int(num_points),
            "selected_index_hash": _selected_index_hash(indices),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
        }
    elif fps_mode_normalized == FPS_MODE_VOXEL_THEN_EXACT:
        sampled_points, indices, mode_summary = voxel_then_exact_fps_downsample(
            points=points,
            num_samples=num_samples,
            prefilter_num_points=fps_prefilter_num_points,
            seed=effective_seed,
            exact_backend=FPS_MODE_EXACT_NUMBA,
        )
        fps_start_index = int(mode_summary.get("fps_start_index", 0))
    else:
        sampled_points, indices, mode_summary = random_then_exact_fps_downsample(
            points=points,
            num_samples=num_samples,
            prefilter_num_points=fps_prefilter_num_points,
            seed=effective_seed,
            exact_backend=FPS_MODE_EXACT_NUMBA,
        )
        fps_start_index = int(mode_summary.get("fps_start_index", 0))

    fps_wall_time_sec = float(time.perf_counter() - wall_start)
    mode_summary["fps_wall_time_sec"] = fps_wall_time_sec

    summary: dict[str, Any] = {
        "num_points_before": int(num_points),
        "num_points_after": int(sampled_points.shape[0]),
        "fps_target_num_samples": int(num_samples),
        "reduction_ratio": float(sampled_points.shape[0] / num_points),
        "fps_seed": effective_seed,
        "fps_start_index": int(fps_start_index),
    }
    summary.update(mode_summary)

    if gt_mask is not None:
        original_gt_count = int(np.sum(gt_mask > 0))
        retained_gt_count = int(np.sum(gt_mask[indices] > 0))
        summary["original_gt_count"] = original_gt_count
        summary["retained_gt_count"] = retained_gt_count
        summary["gt_retention_ratio"] = (
            float(retained_gt_count / original_gt_count)
            if original_gt_count > 0
            else 1.0
        )

    if return_fps_summary:
        return sampled_points, indices, summary
    return sampled_points, indices


def isotropic_scale(
    points: np.ndarray,
    cube_size: float = 64.0,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Center a point cloud and scale it isotropically into a target cube.

    The cloud is centered by its axis-aligned bounding box (AABB) center,
    then scaled by ``cube_size / D_max`` where
    ``D_max = max(extent_x, extent_y, extent_z)``.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        cube_size: Target cube edge length used to compute the scale factor.

    Returns:
        A tuple of ``(scaled_points, d_max, center, scale)`` where
        ``scale = cube_size / d_max`` and
        ``scaled_points = (points - center) * scale``.
    """
    points = _validate_points(points)
    if points.shape[0] == 0:
        raise ValueError("points must not be empty.")

    cube_size = float(cube_size)
    if cube_size <= 0.0:
        raise ValueError("cube_size must be positive.")

    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    center = ((xyz_min + xyz_max) / 2.0).astype(np.float64)
    extent = xyz_max - xyz_min
    d_max = float(np.max(extent))
    if d_max <= 0.0:
        raise ValueError(
            "Invalid point cloud extent: D_max <= 0. "
            "All points may be identical or collapsed on a single location."
        )

    scale = cube_size / d_max
    scaled_points = ((points - center) * scale).astype(np.float64)
    return scaled_points, d_max, center, scale


def weak_sor(
    points: np.ndarray,
    k: int = 30,
    alpha: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove obvious statistical outliers with a weak SOR pass.

    Uses Open3D ``remove_statistical_outlier`` and keeps only mild filtering
    suitable for defect-preserving preprocessing.

    Args:
        points: Input point cloud with shape ``(N, 3)``.
        k: Number of neighbors used by statistical outlier removal.
        alpha: Standard-deviation ratio threshold passed to Open3D as
            ``std_ratio``.

    Returns:
        A tuple of ``(cleaned_points, valid_indices, summary)`` where
        ``valid_indices`` maps cleaned points back to the input cloud.
    """
    backend = _require_open3d()

    points = _validate_points(points)
    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("points must not be empty.")

    k = int(k)
    alpha = float(alpha)
    if k <= 0:
        raise ValueError("k must be positive.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    point_cloud = backend.geometry.PointCloud()
    point_cloud.points = backend.utility.Vector3dVector(points.astype(np.float64))
    filtered_cloud, inlier_indices = point_cloud.remove_statistical_outlier(
        nb_neighbors=k,
        std_ratio=alpha,
    )

    valid_indices = np.asarray(inlier_indices, dtype=np.int64)
    cleaned_points = np.asarray(filtered_cloud.points, dtype=np.float64)
    removed_points_count = int(num_points - cleaned_points.shape[0])
    removed_points_ratio = float(removed_points_count / num_points)

    summary: dict[str, Any] = {
        "num_points_before": int(num_points),
        "num_points_after": int(cleaned_points.shape[0]),
        "removed_points_count": removed_points_count,
        "removed_points_ratio": removed_points_ratio,
        "sor_k": int(k),
        "sor_alpha": float(alpha),
    }
    return cleaned_points, valid_indices, summary


def farthest_point_sample_exact_numpy(
    points: np.ndarray,
    num_samples: int,
    seed: int | None = 0,
) -> tuple[np.ndarray, int]:
    """Legacy exact FPS implementation (NumPy), unchanged behavior."""
    points = _validate_points(points)
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    num_points = points.shape[0]
    if num_points <= num_samples:
        return np.arange(num_points, dtype=np.int64), 0

    rng = np.random.default_rng(seed)
    selected = np.empty(num_samples, dtype=np.int64)
    min_dist_sq = np.full(num_points, np.inf, dtype=np.float64)

    farthest = int(rng.integers(0, num_points))
    start_index = farthest
    for i in range(num_samples):
        selected[i] = farthest
        current = points[farthest]
        diff = points - current
        dist_sq = np.einsum("ij,ij->i", diff, diff, optimize=True)
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
        farthest = int(np.argmax(min_dist_sq))

    return selected, start_index


def farthest_point_sample_exact_numba(
    points: np.ndarray,
    num_samples: int,
    seed: int | None = 0,
) -> tuple[np.ndarray, int, str, bool, str | None]:
    """Exact FPS with optional numba acceleration and deterministic fallback."""
    points = _validate_points(points)
    points = np.ascontiguousarray(points, dtype=np.float64)
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    num_points = points.shape[0]
    if num_points <= num_samples:
        return (
            np.arange(num_points, dtype=np.int64),
            0,
            FPS_MODE_EXACT_NUMBA if _NUMBA_AVAILABLE else FPS_MODE_EXACT_NUMPY,
            False,
            None,
        )

    rng = np.random.default_rng(seed)
    start_index = int(rng.integers(0, num_points))
    if _NUMBA_AVAILABLE:
        selected = _farthest_point_sample_exact_numba_kernel(
            points,
            num_samples,
            start_index,
        )
        return selected.astype(np.int64), start_index, FPS_MODE_EXACT_NUMBA, False, None

    _warn_numba_fallback_once()
    selected, start_index = farthest_point_sample_exact_numpy(
        points,
        num_samples=num_samples,
        seed=seed,
    )
    return (
        selected,
        start_index,
        FPS_MODE_EXACT_NUMPY,
        True,
        "numba_not_available",
    )


def voxel_then_exact_fps_downsample(
    points: np.ndarray,
    num_samples: int,
    prefilter_num_points: int = FPS_PREFILTER_NUM_POINTS_DEFAULT,
    seed: int | None = FPS_RANDOM_SEED_DEFAULT,
    exact_backend: str = FPS_MODE_EXACT_NUMBA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deterministic voxel prefilter followed by exact FPS."""
    points = _validate_points(points)
    num_samples = int(num_samples)
    num_points = int(points.shape[0])
    prefilter_num_points = int(prefilter_num_points)
    if prefilter_num_points <= 0:
        raise ValueError("prefilter_num_points must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if num_points <= num_samples:
        selected = np.arange(num_points, dtype=np.int64)
        summary = {
            "fps_mode": FPS_MODE_VOXEL_THEN_EXACT,
            "backend": exact_backend,
            "num_raw_points": num_points,
            "num_samples": int(num_samples),
            "prefilter_num_points": prefilter_num_points,
            "num_prefiltered_points": num_points,
            "num_voxel_representatives": num_points,
            "selected_index_hash": _selected_index_hash(selected),
            "fps_start_index": 0,
            "fallback_used": False,
            "fallback_reason": None,
        }
        return points.copy(), selected, summary

    effective_prefilter_num_points = int(
        max(
            num_samples,
            prefilter_num_points,
            int(np.ceil(num_samples * FPS_VOXEL_TARGET_RATIO_DEFAULT)),
        )
    )

    voxel_size, xyz_min = _estimate_voxel_size_for_target(
        points=points,
        target_num_voxels=effective_prefilter_num_points,
    )
    representative_indices = _select_deterministic_voxel_representatives(
        points=points,
        xyz_min=xyz_min,
        voxel_size=voxel_size,
    )
    num_representatives = int(representative_indices.shape[0])

    fallback_used = False
    fallback_reason: str | None = None
    backend_used = exact_backend
    fps_start_index = 0

    candidate_indices = representative_indices
    if num_representatives > effective_prefilter_num_points:
        rep_points = points[representative_indices]
        rep_selected, _, backend_used, rep_fallback_used, rep_fallback_reason = (
            _run_exact_backend_indices(
                points=rep_points,
                num_samples=effective_prefilter_num_points,
                seed=seed,
                backend=exact_backend,
            )
        )
        candidate_indices = representative_indices[rep_selected]
        if rep_fallback_used:
            fallback_used = True
            fallback_reason = rep_fallback_reason

    if candidate_indices.shape[0] < num_samples:
        fallback_used = True
        fallback_reason = (
            f"prefiltered_candidates_lt_num_samples:"
            f"{int(candidate_indices.shape[0])}<{int(num_samples)}"
        )
        if FPS_FALLBACK_TO_EXACT_DEFAULT:
            selected, fps_start_index, backend_used, exact_fallback_used, exact_reason = (
                _run_exact_backend_indices(
                    points=points,
                    num_samples=num_samples,
                    seed=seed,
                    backend=exact_backend,
                )
            )
            if exact_fallback_used and fallback_reason is None:
                fallback_reason = exact_reason
            sampled_points = points[selected]
            summary = {
                "fps_mode": FPS_MODE_VOXEL_THEN_EXACT,
                "backend": backend_used,
                "num_raw_points": num_points,
                "num_samples": int(num_samples),
                "prefilter_num_points": int(effective_prefilter_num_points),
                "num_prefiltered_points": int(num_points),
                "num_voxel_representatives": num_representatives,
                "voxel_size": float(voxel_size),
                "selected_index_hash": _selected_index_hash(selected),
                "fps_start_index": int(fps_start_index),
                "fallback_used": True,
                "fallback_reason": fallback_reason,
            }
            return sampled_points, selected, summary

    candidate_points = points[candidate_indices]
    selected_in_candidates, fps_start_index, backend_used, exact_fallback_used, exact_reason = (
        _run_exact_backend_indices(
            points=candidate_points,
            num_samples=num_samples,
            seed=seed,
            backend=exact_backend,
        )
    )
    selected = candidate_indices[selected_in_candidates]
    sampled_points = points[selected]
    if exact_fallback_used:
        fallback_used = True
        if fallback_reason is None:
            fallback_reason = exact_reason

    summary = {
        "fps_mode": FPS_MODE_VOXEL_THEN_EXACT,
        "backend": backend_used,
        "num_raw_points": num_points,
        "num_samples": int(num_samples),
        "prefilter_num_points": int(effective_prefilter_num_points),
        "num_prefiltered_points": int(candidate_indices.shape[0]),
        "num_voxel_representatives": num_representatives,
        "voxel_size": float(voxel_size),
        "selected_index_hash": _selected_index_hash(selected),
        "fps_start_index": int(fps_start_index),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
    }
    return sampled_points, selected.astype(np.int64), summary


def random_then_exact_fps_downsample(
    points: np.ndarray,
    num_samples: int,
    prefilter_num_points: int = FPS_PREFILTER_NUM_POINTS_DEFAULT,
    seed: int | None = FPS_RANDOM_SEED_DEFAULT,
    exact_backend: str = FPS_MODE_EXACT_NUMBA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Random candidate prefilter followed by exact FPS (ablation baseline)."""
    points = _validate_points(points)
    num_samples = int(num_samples)
    num_points = int(points.shape[0])
    prefilter_num_points = int(prefilter_num_points)
    if prefilter_num_points <= 0:
        raise ValueError("prefilter_num_points must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if num_points <= num_samples:
        selected = np.arange(num_points, dtype=np.int64)
        summary = {
            "fps_mode": FPS_MODE_RANDOM_THEN_EXACT,
            "backend": exact_backend,
            "num_raw_points": num_points,
            "num_samples": int(num_samples),
            "prefilter_num_points": prefilter_num_points,
            "num_prefiltered_points": num_points,
            "selected_index_hash": _selected_index_hash(selected),
            "fps_start_index": 0,
            "fallback_used": False,
            "fallback_reason": None,
        }
        return points.copy(), selected, summary

    candidate_count = int(max(num_samples, min(prefilter_num_points, num_points)))
    rng = np.random.default_rng(seed)
    if candidate_count < num_points:
        candidate_indices = rng.choice(num_points, size=candidate_count, replace=False)
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    else:
        candidate_indices = np.arange(num_points, dtype=np.int64)

    candidate_points = points[candidate_indices]
    selected_in_candidates, fps_start_index, backend_used, fallback_used, fallback_reason = (
        _run_exact_backend_indices(
            points=candidate_points,
            num_samples=num_samples,
            seed=seed,
            backend=exact_backend,
        )
    )
    selected = candidate_indices[selected_in_candidates]
    sampled_points = points[selected]

    summary = {
        "fps_mode": FPS_MODE_RANDOM_THEN_EXACT,
        "backend": backend_used,
        "num_raw_points": num_points,
        "num_samples": int(num_samples),
        "prefilter_num_points": int(candidate_count),
        "num_prefiltered_points": int(candidate_count),
        "selected_index_hash": _selected_index_hash(selected),
        "fps_start_index": int(fps_start_index),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
    }
    return sampled_points, selected.astype(np.int64), summary


def _farthest_point_sampling_indices(
    points: np.ndarray,
    num_samples: int,
    seed: int | None = 0,
) -> tuple[np.ndarray, int]:
    """Backward-compatible alias of legacy exact NumPy FPS."""
    return farthest_point_sample_exact_numpy(
        points=points,
        num_samples=num_samples,
        seed=seed,
    )


def _run_exact_backend_indices(
    points: np.ndarray,
    num_samples: int,
    seed: int | None,
    backend: str,
) -> tuple[np.ndarray, int, str, bool, str | None]:
    backend_normalized = str(backend).strip().lower()
    if backend_normalized == FPS_MODE_EXACT_NUMPY:
        selected, start_index = farthest_point_sample_exact_numpy(
            points=points,
            num_samples=num_samples,
            seed=seed,
        )
        return selected, start_index, FPS_MODE_EXACT_NUMPY, False, None
    if backend_normalized == FPS_MODE_EXACT_NUMBA:
        return farthest_point_sample_exact_numba(
            points=points,
            num_samples=num_samples,
            seed=seed,
        )
    raise ValueError(
        f"Unsupported exact backend='{backend}'. "
        f"Valid: {[FPS_MODE_EXACT_NUMPY, FPS_MODE_EXACT_NUMBA]}"
    )


def _estimate_voxel_size_for_target(
    points: np.ndarray,
    target_num_voxels: int,
) -> tuple[float, np.ndarray]:
    xyz_min = np.min(points, axis=0)
    xyz_max = np.max(points, axis=0)
    extent = xyz_max - xyz_min
    max_extent = float(np.max(extent))
    target_num_voxels = int(max(1, min(target_num_voxels, points.shape[0])))

    if max_extent <= 0.0:
        return 1.0, xyz_min.astype(np.float64)

    extent_clamped = np.maximum(extent, max_extent / 1024.0)
    volume = float(np.prod(extent_clamped))
    voxel_size = float((volume / float(target_num_voxels)) ** (1.0 / 3.0))
    voxel_size = max(voxel_size, max_extent / 4096.0)

    for _ in range(8):
        occupied = _count_occupied_voxels(points, xyz_min, voxel_size)
        if occupied <= 0:
            voxel_size *= 0.5
            continue
        ratio = float(occupied) / float(target_num_voxels)
        if 0.90 <= ratio <= 1.10:
            break
        voxel_size *= ratio ** (1.0 / 3.0)
        voxel_size = max(voxel_size, max_extent / 4096.0)

    return float(voxel_size), xyz_min.astype(np.float64)


def _count_occupied_voxels(
    points: np.ndarray,
    xyz_min: np.ndarray,
    voxel_size: float,
) -> int:
    voxel_size = float(max(voxel_size, np.finfo(np.float64).eps))
    coords = np.floor((points - xyz_min.reshape(1, 3)) / voxel_size).astype(np.int64)
    return int(np.unique(coords, axis=0).shape[0])


def _select_deterministic_voxel_representatives(
    points: np.ndarray,
    xyz_min: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    voxel_size = float(max(voxel_size, np.finfo(np.float64).eps))
    inv_size = 1.0 / voxel_size
    best_by_voxel: dict[tuple[int, int, int], tuple[float, int]] = {}

    for idx in range(points.shape[0]):
        px = float(points[idx, 0])
        py = float(points[idx, 1])
        pz = float(points[idx, 2])
        vx = int(np.floor((px - xyz_min[0]) * inv_size))
        vy = int(np.floor((py - xyz_min[1]) * inv_size))
        vz = int(np.floor((pz - xyz_min[2]) * inv_size))
        key = (vx, vy, vz)

        cx = xyz_min[0] + (float(vx) + 0.5) * voxel_size
        cy = xyz_min[1] + (float(vy) + 0.5) * voxel_size
        cz = xyz_min[2] + (float(vz) + 0.5) * voxel_size
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        dist_sq = dx * dx + dy * dy + dz * dz

        prev = best_by_voxel.get(key)
        if prev is None:
            best_by_voxel[key] = (dist_sq, idx)
        else:
            prev_dist_sq, prev_idx = prev
            if (dist_sq < prev_dist_sq) or (dist_sq == prev_dist_sq and idx < prev_idx):
                best_by_voxel[key] = (dist_sq, idx)

    representative_indices = np.array(
        sorted(item[1] for item in best_by_voxel.values()),
        dtype=np.int64,
    )
    return representative_indices


def _selected_index_hash(indices: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _warn_numba_fallback_once() -> None:
    global _NUMBA_FALLBACK_WARNED
    if _NUMBA_FALLBACK_WARNED:
        return
    warnings.warn(
        "numba is not available; exact_numba FPS falls back to exact_numpy.",
        RuntimeWarning,
        stacklevel=3,
    )
    _NUMBA_FALLBACK_WARNED = True


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Validate point tensor shape and dtype."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")
    return points


def _validate_binary_mask(mask: np.ndarray, num_points: int) -> np.ndarray:
    """Validate a one-dimensional binary mask aligned with points."""
    mask = np.asarray(mask)
    if mask.ndim != 1:
        raise ValueError("gt_mask must be one-dimensional.")
    if mask.shape[0] != num_points:
        raise ValueError("gt_mask must have the same length as points.")
    return mask
