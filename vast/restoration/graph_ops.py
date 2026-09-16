"""Graph morphology and connected-component helpers for GTR restoration."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def graph_dilate(mask: np.ndarray, adjacency: sparse.csr_matrix, hops: int) -> np.ndarray:
    """Expand a boolean mask outward along graph edges for ``hops`` steps."""
    hops = int(hops)
    expanded = np.asarray(mask, dtype=bool).reshape(-1)
    if hops <= 0:
        return expanded.copy()
    if adjacency.shape[0] != expanded.shape[0]:
        raise ValueError("adjacency row count must match mask length.")

    for _ in range(hops):
        neighbor_active = np.asarray(adjacency.dot(expanded.astype(np.float64)) > 0, dtype=bool)
        expanded = expanded | neighbor_active
    return expanded


def graph_erode(mask: np.ndarray, adjacency: sparse.csr_matrix, hops: int) -> np.ndarray:
    """Erode a boolean mask by dilating its complement, then inverting."""
    hops = int(hops)
    if hops <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return ~graph_dilate(~np.asarray(mask, dtype=bool), adjacency, hops)


def graph_close(mask: np.ndarray, adjacency: sparse.csr_matrix, hops: int = 1) -> np.ndarray:
    """Apply graph closing (dilate then erode)."""
    hops = int(hops)
    if hops <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return graph_erode(graph_dilate(mask, adjacency, hops), adjacency, hops)


def label_connected_components(
    active_mask: np.ndarray,
    adjacency: sparse.csr_matrix,
) -> tuple[np.ndarray, int]:
    """Label connected components among active vertices on an undirected graph."""
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    labels = np.zeros(active_mask.shape[0], dtype=np.int32)
    component_id = 0

    for start in np.flatnonzero(active_mask):
        if labels[start] != 0:
            continue
        component_id += 1
        stack = [int(start)]
        labels[start] = component_id
        while stack:
            node = stack.pop()
            begin = int(adjacency.indptr[node])
            end = int(adjacency.indptr[node + 1])
            for neighbor in adjacency.indices[begin:end]:
                if active_mask[neighbor] and labels[neighbor] == 0:
                    labels[neighbor] = component_id
                    stack.append(int(neighbor))
    return labels, component_id


def remove_small_components(
    mask: np.ndarray,
    adjacency: sparse.csr_matrix,
    min_size: int,
) -> np.ndarray:
    """Drop connected components smaller than ``min_size``."""
    mask = np.asarray(mask, dtype=bool).copy()
    min_size = int(min_size)
    if min_size <= 1 or not np.any(mask):
        return mask

    labels, num_components = label_connected_components(mask, adjacency)
    if num_components == 0:
        return mask

    for component_id in range(1, num_components + 1):
        component_mask = labels == component_id
        if int(np.sum(component_mask)) < min_size:
            mask[component_mask] = False
    return mask


def _dilate_mask_neighbor_indices(
    mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    hops: int,
) -> np.ndarray:
    """Expand a boolean mask along graph edges defined by per-point neighbor lists."""
    hops = int(hops)
    expanded = np.asarray(mask, dtype=bool).reshape(-1).copy()
    if hops <= 0:
        return expanded
    if len(neighbor_indices) != expanded.shape[0]:
        raise ValueError("neighbor_indices length must match mask length.")

    for _ in range(hops):
        next_expanded = expanded.copy()
        for center in np.flatnonzero(expanded):
            for neighbor in np.asarray(neighbor_indices[center], dtype=np.int64).reshape(-1):
                next_expanded[int(neighbor)] = True
        expanded = next_expanded
    return expanded


def fill_enclosed_holes(
    component_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    roi_hops: int = 4,
    max_growth_ratio: float = 3.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fill low-energy interior holes enclosed by a closed high-energy component ring."""
    component_mask = np.asarray(component_mask, dtype=bool).reshape(-1)
    num_closed_points = int(np.sum(component_mask))
    summary: dict[str, object] = {
        "num_closed_points": num_closed_points,
        "num_hole_points": 0,
        "num_filled_points": num_closed_points,
        "hole_fill_growth_ratio": 1.0,
        "hole_fill_used": False,
        "hole_fill_rejection_reason": "",
    }
    if num_closed_points == 0:
        summary["hole_fill_rejection_reason"] = "empty_component"
        return component_mask.copy(), summary

    roi_mask = _dilate_mask_neighbor_indices(component_mask, neighbor_indices, hops=int(roi_hops))
    barrier = component_mask
    passable = roi_mask & ~barrier

    outside_reachable = np.zeros(component_mask.shape[0], dtype=bool)
    queue: list[int] = []
    for center in np.flatnonzero(passable):
        neighbors = np.asarray(neighbor_indices[int(center)], dtype=np.int64).reshape(-1)
        if neighbors.size == 0 or np.any(~roi_mask[neighbors]):
            outside_reachable[int(center)] = True
            queue.append(int(center))

    while queue:
        center = queue.pop()
        for neighbor in np.asarray(neighbor_indices[center], dtype=np.int64).reshape(-1):
            neighbor = int(neighbor)
            if passable[neighbor] and not outside_reachable[neighbor]:
                outside_reachable[neighbor] = True
                queue.append(neighbor)

    hole_points = passable & ~outside_reachable
    num_hole_points = int(np.sum(hole_points))
    summary["num_hole_points"] = num_hole_points

    if num_hole_points == 0:
        summary["hole_fill_rejection_reason"] = "no_enclosed_holes"
        return component_mask.copy(), summary

    filled_mask = component_mask | hole_points
    num_filled_points = int(np.sum(filled_mask))
    growth_ratio = float(num_filled_points / max(num_closed_points, 1))
    summary["num_filled_points"] = num_filled_points
    summary["hole_fill_growth_ratio"] = growth_ratio

    if growth_ratio > float(max_growth_ratio):
        summary["hole_fill_rejection_reason"] = "hole_growth_too_large"
        return component_mask.copy(), summary

    summary["hole_fill_used"] = True
    return filled_mask, summary


def refine_component_by_displacement(
    component_mask: np.ndarray,
    displacements: np.ndarray,
    neighbor_indices: list[np.ndarray],
    high_percentile: float = 70.0,
    low_ratio: float = 0.4,
    min_points: int = 10,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    """Refine a filled component mask via hysteresis-connected displacement seeds."""
    component_mask = np.asarray(component_mask, dtype=bool).reshape(-1)
    displacements = np.asarray(displacements, dtype=np.float64).reshape(-1)
    min_points = int(min_points)
    low_ratio = float(low_ratio)

    summary: dict[str, object] = {
        "point_refine_used": False,
        "refine_high_threshold": 0.0,
        "refine_low_threshold": 0.0,
        "num_refined_points": int(np.sum(component_mask)),
        "refinement_rejection_reason": "",
    }

    component_displacements = displacements[component_mask]
    if component_displacements.size == 0:
        summary["refinement_rejection_reason"] = "empty_component"
        return component_mask.copy(), summary

    high_threshold = max(float(np.percentile(component_displacements, high_percentile)), float(eps))
    low_threshold = high_threshold * low_ratio
    summary["refine_high_threshold"] = high_threshold
    summary["refine_low_threshold"] = low_threshold

    strong_mask = component_mask & (displacements >= high_threshold)
    weak_mask = component_mask & (displacements >= low_threshold)
    if not np.any(strong_mask):
        summary["refinement_rejection_reason"] = "no_strong_displacement_seed"
        return component_mask.copy(), summary

    refined_mask = np.zeros(component_mask.shape[0], dtype=bool)
    visited = np.zeros(component_mask.shape[0], dtype=bool)
    queue = [int(index) for index in np.flatnonzero(strong_mask)]
    for index in queue:
        refined_mask[index] = True
        visited[index] = True

    while queue:
        center = queue.pop()
        for neighbor in np.asarray(neighbor_indices[center], dtype=np.int64).reshape(-1):
            neighbor = int(neighbor)
            if weak_mask[neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                refined_mask[neighbor] = True
                queue.append(neighbor)

    num_refined_points = int(np.sum(refined_mask))
    summary["num_refined_points"] = num_refined_points
    if num_refined_points < min_points:
        summary["refinement_rejection_reason"] = "too_few_refined_points"
        return component_mask.copy(), summary

    summary["point_refine_used"] = True
    return refined_mask, summary


def hysteresis_connected(
    displacements: np.ndarray,
    roi_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    depth_threshold: float,
    low_ratio: float,
) -> tuple[np.ndarray | None, float, float, str | None]:
    """Grow a geometry mask from strong displacement seeds through weak neighbors."""
    displacements = np.asarray(displacements, dtype=np.float64).reshape(-1)
    roi_mask = np.asarray(roi_mask, dtype=bool).reshape(-1)
    low_ratio = float(low_ratio)

    roi_displacements = displacements[roi_mask]
    if roi_displacements.size == 0:
        return None, 0.0, 0.0, "empty_unknown_roi"

    t_high = max(float(depth_threshold), float(np.percentile(roi_displacements, 90)))
    t_low = low_ratio * t_high

    strong_mask = roi_mask & (displacements >= t_high)
    weak_mask = roi_mask & (displacements >= t_low)
    if not np.any(strong_mask):
        return None, t_high, t_low, "no_strong_displacement_seed"

    geometry_mask = np.zeros(displacements.shape[0], dtype=bool)
    visited = np.zeros(displacements.shape[0], dtype=bool)
    queue = [int(index) for index in np.flatnonzero(strong_mask)]
    for index in queue:
        geometry_mask[index] = True
        visited[index] = True

    while queue:
        center = queue.pop()
        for neighbor in np.asarray(neighbor_indices[center], dtype=np.int64).reshape(-1):
            if weak_mask[neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                geometry_mask[neighbor] = True
                queue.append(int(neighbor))

    return geometry_mask, t_high, t_low, None
