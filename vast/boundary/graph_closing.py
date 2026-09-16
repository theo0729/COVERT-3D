"""Graph-based morphological closing for boundary masks."""

from __future__ import annotations

from typing import Any

import numpy as np

from vast.boundary.ac_boundary import coerce_neighbor_indices
from vast.boundary.boundary_expand import expand_boundary_mask


def close_gaps(
    boundary_mask: np.ndarray,
    neighbor_indices: list[np.ndarray] | np.ndarray,
    dilation_hops: int = 1,
    erosion_hops: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Close small boundary gaps via graph dilation followed by erosion.

    Dilation expands the boundary mask outward over ``dilation_hops`` graph
    hops. Erosion then removes boundary points that have at least one
    non-boundary neighbor, repeated for ``erosion_hops`` hops.

    Args:
        boundary_mask: Boolean boundary mask with shape ``(N,)``.
        neighbor_indices: Per-point neighbor index list or ``(N, K)`` array.
        dilation_hops: Number of outward expansion hops.
        erosion_hops: Number of inward shrink hops.

    Returns:
        A tuple of ``(closed_mask, summary)`` for audit and visualization.
    """
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    num_points = boundary_mask.shape[0]
    if boundary_mask.ndim != 1:
        raise ValueError("boundary_mask must be one-dimensional.")
    neighbor_indices = coerce_neighbor_indices(neighbor_indices, num_points)

    dilation_hops = int(dilation_hops)
    erosion_hops = int(erosion_hops)
    if dilation_hops < 0:
        raise ValueError("dilation_hops must be non-negative.")
    if erosion_hops < 0:
        raise ValueError("erosion_hops must be non-negative.")

    count_before = int(np.sum(boundary_mask))

    if dilation_hops > 0:
        dilated_mask = expand_boundary_mask(
            boundary_mask,
            neighbor_indices,
            hops=dilation_hops,
        )
    else:
        dilated_mask = boundary_mask.copy()
    count_after_dilation = int(np.sum(dilated_mask))

    if erosion_hops > 0:
        closed_mask = _erode_boundary_mask(
            dilated_mask,
            neighbor_indices,
            hops=erosion_hops,
        )
    else:
        closed_mask = dilated_mask.copy()
    count_after_closing = int(np.sum(closed_mask))

    summary: dict[str, Any] = {
        "num_points": int(num_points),
        "num_boundary_before": count_before,
        "num_boundary_after_dilation": count_after_dilation,
        "num_boundary_after_closing": count_after_closing,
        "boundary_delta_dilation": int(count_after_dilation - count_before),
        "boundary_delta_closing": int(count_after_closing - count_before),
        "boundary_ratio_before": float(count_before / num_points) if num_points else 0.0,
        "boundary_ratio_after_dilation": float(count_after_dilation / num_points) if num_points else 0.0,
        "boundary_ratio_after_closing": float(count_after_closing / num_points) if num_points else 0.0,
        "dilation_hops": dilation_hops,
        "erosion_hops": erosion_hops,
    }
    return closed_mask, summary


def _erode_boundary_mask(
    boundary_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    hops: int = 1,
) -> np.ndarray:
    """Erode a boundary mask on a graph neighborhood."""
    eroded = np.asarray(boundary_mask, dtype=bool).copy()
    hops = int(hops)
    if hops <= 0:
        return eroded

    for _ in range(hops):
        if not np.any(eroded):
            break

        remove_indices: list[int] = []
        for point_index in np.flatnonzero(eroded):
            neighbors = neighbor_indices[point_index]
            if neighbors.size == 0:
                continue
            if np.any(~eroded[neighbors]):
                remove_indices.append(int(point_index))

        if not remove_indices:
            break
        eroded[np.asarray(remove_indices, dtype=np.int64)] = False

    return eroded
