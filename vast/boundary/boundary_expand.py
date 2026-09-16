"""Boundary band expansion over KNN neighbor connectivity."""

from __future__ import annotations

import numpy as np

from vast.boundary.ac_boundary import coerce_neighbor_indices


def expand_boundary_mask(
    boundary_mask: np.ndarray,
    neighbor_indices: list[np.ndarray] | np.ndarray,
    hops: int = 1,
) -> np.ndarray:
    """Expand boundary mask by graph hops over neighbor connectivity."""
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    num_points = boundary_mask.shape[0]
    neighbor_indices = coerce_neighbor_indices(neighbor_indices, num_points)

    hops = int(hops)
    if hops < 0:
        raise ValueError("hops must be non-negative.")
    if hops == 0:
        return boundary_mask.copy()

    expanded = boundary_mask.copy()
    frontier = np.flatnonzero(boundary_mask)
    for _ in range(hops):
        if frontier.size == 0:
            break
        neighbor_parts = [neighbor_indices[index] for index in frontier if neighbor_indices[index].size > 0]
        if not neighbor_parts:
            break
        candidates = np.unique(np.concatenate(neighbor_parts))
        new_points = candidates[~expanded[candidates]]
        expanded[new_points] = True
        frontier = new_points
    return expanded
