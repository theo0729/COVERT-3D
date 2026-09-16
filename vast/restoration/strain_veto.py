"""Topological strain veto for rigid structures such as fins and sharp tips."""

from __future__ import annotations

import numpy as np


def apply_strain_veto(
    points: np.ndarray,
    inpainted_points: np.ndarray,
    neighbor_indices: list[np.ndarray],
    geometry_anomaly_mask: np.ndarray,
    strain_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject high-strain rigid structures and keep low-strain soft defects.

    Args:
        points: Raw coordinates with shape ``(N, 3)``.
        inpainted_points: Restored coordinates with shape ``(N, 3)``.
        neighbor_indices: Per-point mutual neighbor index lists.
        geometry_anomaly_mask: Candidate geometry anomaly mask with shape ``(N,)``.
        strain_threshold: Maximum mean edge strain retained as a soft defect.

    Returns:
        A tuple of ``(final_defect_mask, point_strains)``.
    """
    points = np.asarray(points, dtype=np.float64)
    inpainted_points = np.asarray(inpainted_points, dtype=np.float64)
    geometry_anomaly_mask = np.asarray(geometry_anomaly_mask, dtype=bool).reshape(-1)
    strain_threshold = float(strain_threshold)

    if points.shape != inpainted_points.shape:
        raise ValueError("points and inpainted_points must have the same shape.")
    if geometry_anomaly_mask.shape[0] != points.shape[0]:
        raise ValueError("geometry_anomaly_mask must match points length.")
    if len(neighbor_indices) != points.shape[0]:
        raise ValueError("neighbor_indices length must match points length.")

    point_strains = np.zeros(points.shape[0], dtype=np.float64)
    candidate_indices = np.flatnonzero(geometry_anomaly_mask)

    for center_index in candidate_indices:
        neighbors = np.asarray(neighbor_indices[int(center_index)], dtype=np.int64).reshape(-1)
        if neighbors.size == 0:
            point_strains[center_index] = np.inf
            continue

        old_vectors = points[neighbors] - points[center_index]
        new_vectors = inpainted_points[neighbors] - inpainted_points[center_index]
        old_lengths = np.linalg.norm(old_vectors, axis=1)
        new_lengths = np.linalg.norm(new_vectors, axis=1)
        edge_strains = np.abs(old_lengths - new_lengths) / (old_lengths + 1e-8)
        point_strains[center_index] = float(np.mean(edge_strains))

    final_defect_mask = geometry_anomaly_mask & (point_strains < strain_threshold)
    return np.asarray(final_defect_mask, dtype=bool), point_strains
