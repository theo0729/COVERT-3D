"""Angular criterion (AC) boundary seed extraction."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_points(points: np.ndarray) -> np.ndarray:
    """Validate a point cloud and return it as float64 array."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if points.shape[0] < 3:
        raise ValueError("points must contain at least 3 points.")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")
    return points


def validate_neighbor_indices(
    neighbor_indices: list[np.ndarray],
    num_points: int,
) -> list[np.ndarray]:
    """Validate neighbor list format and index bounds."""
    if not isinstance(neighbor_indices, list):
        raise ValueError("neighbor_indices must be a list of numpy arrays.")
    if len(neighbor_indices) != int(num_points):
        raise ValueError("neighbor_indices length must equal num_points.")

    validated: list[np.ndarray] = []
    for neighbors in neighbor_indices:
        arr = np.asarray(neighbors)
        if arr.ndim != 1:
            raise ValueError("Each neighbor index entry must be one-dimensional.")
        if arr.size == 0:
            validated.append(np.empty(0, dtype=np.int64))
            continue

        if not np.issubdtype(arr.dtype, np.integer):
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError("Neighbor indices must be integer arrays.")
            if not np.isfinite(arr).all() or not np.all(arr == np.floor(arr)):
                raise ValueError("Neighbor indices must contain finite integers only.")
        arr = arr.astype(np.int64, copy=False)
        if np.any(arr < 0) or np.any(arr >= num_points):
            raise ValueError("Neighbor index out of bounds.")
        validated.append(arr)
    return validated


def coerce_neighbor_indices(
    neighbor_indices: list[np.ndarray] | np.ndarray,
    num_points: int,
) -> list[np.ndarray]:
    """Normalize neighbor indices from a list or array into a validated list."""
    if isinstance(neighbor_indices, list):
        return validate_neighbor_indices(neighbor_indices, num_points)

    array = np.asarray(neighbor_indices)
    if array.ndim == 2:
        if array.shape[0] != num_points:
            raise ValueError("neighbor_indices rows must equal num_points.")
        neighbor_list = [np.asarray(array[i], dtype=np.int64).reshape(-1) for i in range(num_points)]
        return validate_neighbor_indices(neighbor_list, num_points)

    if array.ndim == 1 and array.dtype == object:
        if array.shape[0] != num_points:
            raise ValueError("neighbor_indices length must equal num_points.")
        neighbor_list = [np.asarray(entry, dtype=np.int64).reshape(-1) for entry in array]
        return validate_neighbor_indices(neighbor_list, num_points)

    raise ValueError(
        "neighbor_indices must be a list of 1D arrays or a 2D/object ndarray of shape (N, K)."
    )


def estimate_normals_pca(
    points: np.ndarray,
    neighbor_indices: list[np.ndarray] | np.ndarray,
    min_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate local normals via PCA for each point."""
    points = validate_points(points)
    num_points = points.shape[0]
    neighbor_indices = coerce_neighbor_indices(neighbor_indices, num_points)
    min_neighbors = int(min_neighbors)
    if min_neighbors < 0:
        raise ValueError("min_neighbors must be non-negative.")

    normals = np.full((num_points, 3), np.nan, dtype=np.float64)
    normal_valid_mask = np.zeros(num_points, dtype=bool)

    low_neighbor_count = 0
    insufficient_for_pca_count = 0

    for point_index, neighbors in enumerate(neighbor_indices):
        if neighbors.size < min_neighbors:
            low_neighbor_count += 1
        if neighbors.size < 3:
            insufficient_for_pca_count += 1
            continue

        local_points = points[neighbors]
        centered = local_points - np.mean(local_points, axis=0, keepdims=True)
        covariance = centered.T @ centered
        try:
            _, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            continue
        normal = eigenvectors[:, 0]
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm <= 1e-12:
            continue

        normals[point_index] = normal / normal_norm
        normal_valid_mask[point_index] = True

    normal_meta = {
        "num_points": int(num_points),
        "min_neighbors": int(min_neighbors),
        "low_neighbor_count": int(low_neighbor_count),
        "insufficient_for_pca_count": int(insufficient_for_pca_count),
        "normal_invalid_count": int(np.sum(~normal_valid_mask)),
        "normal_valid_ratio": float(np.mean(normal_valid_mask)) if num_points else 0.0,
    }
    return normals, normal_valid_mask, normal_meta


def _tangent_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a numerically stable orthonormal tangent basis from a unit normal."""
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,):
        raise ValueError("normal must have shape (3,).")
    if not np.isfinite(normal).all():
        raise ValueError("normal must contain only finite values.")

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("normal magnitude must be positive.")
    n = normal / normal_norm

    reference_axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, reference_axis)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        reference_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(n[0]) > 0.9:
            reference_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(n, reference_axis)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-12:
            raise ValueError("Failed to build tangent basis from normal.")
    u = u / u_norm

    v = np.cross(n, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12:
        raise ValueError("Failed to build tangent basis from normal.")
    v = v / v_norm
    return u, v


def extract_ac_boundary_seed(
    points: np.ndarray,
    neighbor_indices: list[np.ndarray] | np.ndarray,
    normals: np.ndarray | None = None,
    angle_threshold: float = np.pi / 2,
    min_neighbors: int = 8,
    low_neighbor_policy: str = "boundary",
) -> dict[str, Any]:
    """Extract AC boundary seed points by the max-angle-gap criterion."""
    points = validate_points(points)
    num_points = points.shape[0]
    neighbor_indices = coerce_neighbor_indices(neighbor_indices, num_points)
    min_neighbors = int(min_neighbors)
    if min_neighbors < 0:
        raise ValueError("min_neighbors must be non-negative.")
    if low_neighbor_policy not in {"boundary", "ignore"}:
        raise ValueError("low_neighbor_policy must be either 'boundary' or 'ignore'.")

    angle_threshold = float(angle_threshold)
    if not np.isfinite(angle_threshold):
        raise ValueError("angle_threshold must be finite.")
    if angle_threshold <= 0.0 or angle_threshold > 2.0 * np.pi:
        raise ValueError("angle_threshold must be in (0, 2*pi].")

    if normals is None:
        normals, normal_valid_mask, _ = estimate_normals_pca(
            points,
            neighbor_indices,
            min_neighbors=min_neighbors,
        )
    else:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != (num_points, 3):
            raise ValueError("normals must have shape (N, 3).")
        normal_norms = np.linalg.norm(normals, axis=1)
        normal_valid_mask = np.isfinite(normals).all(axis=1) & (normal_norms > 1e-12)
        normals = normals.copy()
        normals[normal_valid_mask] = (
            normals[normal_valid_mask] / normal_norms[normal_valid_mask, None]
        )

    degree = np.asarray([neighbors.size for neighbors in neighbor_indices], dtype=np.int64)
    boundary_mask = np.zeros(num_points, dtype=bool)
    max_angle_gap = np.full(num_points, np.nan, dtype=np.float64)

    low_neighbor_count = int(np.sum(degree < min_neighbors))
    two_pi = float(2.0 * np.pi)

    for point_index in range(num_points):
        if not normal_valid_mask[point_index]:
            continue

        neighbors = neighbor_indices[point_index]
        if neighbors.size < min_neighbors:
            if low_neighbor_policy == "boundary":
                boundary_mask[point_index] = True
            continue

        tangent_u, tangent_v = _tangent_basis_from_normal(normals[point_index])
        relative = points[neighbors] - points[point_index]
        x_coords = relative @ tangent_u
        y_coords = relative @ tangent_v

        projection_norm_sq = x_coords * x_coords + y_coords * y_coords
        valid_projection = projection_norm_sq > 1e-18
        if np.count_nonzero(valid_projection) < 2:
            continue

        angles = np.arctan2(y_coords[valid_projection], x_coords[valid_projection])
        angles = np.mod(angles, two_pi)
        angles = np.sort(angles)

        gaps = np.diff(angles)
        wrap_gap = two_pi - angles[-1] + angles[0]
        all_gaps = np.concatenate([gaps, np.asarray([wrap_gap], dtype=np.float64)])
        max_gap = float(np.max(all_gaps))
        max_angle_gap[point_index] = max_gap
        if max_gap > angle_threshold:
            boundary_mask[point_index] = True

    finite_gaps = max_angle_gap[np.isfinite(max_angle_gap)]
    summary = {
        "num_points": int(num_points),
        "num_boundary_points": int(np.sum(boundary_mask)),
        "boundary_ratio": float(np.mean(boundary_mask)) if num_points else 0.0,
        "angle_threshold": float(angle_threshold),
        "angle_threshold_degrees": float(np.degrees(angle_threshold)),
        "min_neighbors": int(min_neighbors),
        "low_neighbor_policy": low_neighbor_policy,
        "normal_invalid_count": int(np.sum(~normal_valid_mask)),
        "max_angle_gap": _summary_stats(finite_gaps),
        "degree": _summary_stats(degree.astype(np.float64)),
        "low_neighbor_count": int(low_neighbor_count),
    }
    return {
        "boundary_mask": boundary_mask,
        "max_angle_gap": max_angle_gap,
        "normals": normals,
        "normal_valid_mask": normal_valid_mask,
        "degree": degree,
        "summary": summary,
    }


def _summary_stats(values: np.ndarray) -> dict[str, float | None]:
    """Return min/mean/median/max for finite numeric arrays."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }
