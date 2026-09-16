"""Geometric displacement between raw and inpainted coordinates."""

from __future__ import annotations

import numpy as np


def compute_geometry_displacement(
    points: np.ndarray,
    inpainted_points: np.ndarray,
    internal_mask: np.ndarray,
    depth_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point displacement magnitude and depth-based anomaly mask.

    Args:
        points: Raw coordinates with shape ``(N, 3)``.
        inpainted_points: Restored coordinates with shape ``(N, 3)``.
        internal_mask: Boolean mask selecting ROI interior vertices.
        depth_threshold: Minimum displacement to mark a geometric anomaly.

    Returns:
        A tuple of ``(displacements, geometry_anomaly_mask)``.
    """
    points = np.asarray(points, dtype=np.float64)
    inpainted_points = np.asarray(inpainted_points, dtype=np.float64)
    internal_mask = np.asarray(internal_mask, dtype=bool).reshape(-1)
    depth_threshold = float(depth_threshold)

    if points.shape != inpainted_points.shape:
        raise ValueError("points and inpainted_points must have the same shape.")
    if internal_mask.shape[0] != points.shape[0]:
        raise ValueError("internal_mask must match points length.")

    displacements = np.zeros(points.shape[0], dtype=np.float64)
    if np.any(internal_mask):
        delta = points[internal_mask] - inpainted_points[internal_mask]
        displacements[internal_mask] = np.linalg.norm(delta, axis=1)

    geometry_anomaly_mask = np.zeros(points.shape[0], dtype=bool)
    geometry_anomaly_mask[internal_mask] = displacements[internal_mask] > depth_threshold
    return displacements, geometry_anomaly_mask
