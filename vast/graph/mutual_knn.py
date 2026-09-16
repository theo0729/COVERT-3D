"""Mutual KNN graph construction with radius cap."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def build_mutual_knn_graph(
    points: np.ndarray,
    k: int = 30,
    radius_cap_factor: float = 2.5,
    query_workers: int = -1,
) -> tuple[sparse.csr_matrix, list[np.ndarray], dict[str, Any]]:
    """Build a mutual KNN graph with radius capping on directed KNN edges.

    Steps:
        1. Query ``k`` nearest neighbors per point with ``cKDTree``.
        2. Estimate ``r_max = mean_edge_len * radius_cap_factor``.
        3. Build a directed adjacency matrix keeping edges with length ``<= r_max``.
        4. Intersect with the transpose to obtain the mutual KNN graph.

    Args:
        points: Point cloud with shape ``(N, 3)``.
        k: Number of nearest neighbors per point.
        radius_cap_factor: Multiplier applied to the mean KNN edge length.
        query_workers: Worker count forwarded to ``cKDTree.query``.  This only
            controls execution parallelism and cannot change graph semantics.

    Returns:
        A tuple of ``(adjacency_mutual, neighbor_indices, summary)`` where
        ``adjacency_mutual`` is a weighted CSR matrix and ``neighbor_indices``
        lists mutual neighbors for each point.
    """
    points = _validate_points(points)
    num_points = points.shape[0]
    k = int(k)
    radius_cap_factor = float(radius_cap_factor)
    query_workers = int(query_workers)

    if num_points == 0:
        raise ValueError("points must not be empty.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if radius_cap_factor <= 0.0:
        raise ValueError("radius_cap_factor must be positive.")
    if query_workers == 0 or query_workers < -1:
        raise ValueError("query_workers must be -1 or a positive integer.")

    if num_points == 1:
        empty = sparse.csr_matrix((1, 1), dtype=np.float64)
        summary = _empty_graph_summary(
            num_points=num_points,
            k=k,
            radius_cap_factor=radius_cap_factor,
        )
        return empty, [np.empty(0, dtype=np.int64)], summary

    k_query = min(k + 1, num_points)
    tree = cKDTree(points)
    try:
        distances, indices = tree.query(
            points, k=k_query, workers=query_workers
        )
    except TypeError:
        distances, indices = tree.query(points, k=k_query)

    if k_query == 1:
        neighbor_distances = np.empty((num_points, 0), dtype=np.float64)
        neighbor_indices_knn = np.empty((num_points, 0), dtype=np.int64)
    else:
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float64)
        neighbor_indices_knn = np.asarray(indices[:, 1:], dtype=np.int64)

    num_candidate_edges = int(neighbor_distances.size)
    if num_candidate_edges == 0:
        mean_edge_len = 0.0
        r_max = 0.0
        kept_mask = np.zeros_like(neighbor_distances, dtype=bool)
    else:
        positive_distances = neighbor_distances[neighbor_distances > 0.0]
        mean_edge_len = float(np.mean(positive_distances)) if positive_distances.size else 0.0
        r_max = float(mean_edge_len * radius_cap_factor)
        kept_mask = neighbor_distances <= r_max

    row_idx = np.repeat(np.arange(num_points, dtype=np.int64), neighbor_indices_knn.shape[1])
    col_idx = neighbor_indices_knn.reshape(-1)
    edge_distances = neighbor_distances.reshape(-1)
    kept_mask_flat = kept_mask.reshape(-1)

    num_edges_cut = int(num_candidate_edges - np.count_nonzero(kept_mask_flat))
    if np.any(kept_mask_flat):
        row_kept = row_idx[kept_mask_flat]
        col_kept = col_idx[kept_mask_flat]
        data_kept = edge_distances[kept_mask_flat]
        adjacency = sparse.coo_matrix(
            (data_kept, (row_kept, col_kept)),
            shape=(num_points, num_points),
            dtype=np.float64,
        ).tocsr()
    else:
        adjacency = sparse.csr_matrix((num_points, num_points), dtype=np.float64)
   
    # 原做法，downstream模块会把这个值当作真实欧式距离使用 
    # adjacency_mutual = adjacency.multiply(adjacency.transpose()).tocsr()

    # 修复版，只保留双向存在的边，但边权仍然是原始距离d
    adjacency_mutual = adjacency.minimum(adjacency.transpose()).tocsr()
    neighbor_indices = _neighbor_indices_from_csr(adjacency_mutual)

    undirected = adjacency_mutual + adjacency_mutual.transpose()
    undirected.eliminate_zeros()
    num_components, labels = connected_components(
        csgraph=undirected,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels, minlength=max(int(num_components), 1))
    largest_component_size = int(component_sizes.max()) if component_sizes.size else 0
    degrees = np.diff(adjacency_mutual.indptr)
    isolated_node_count = int(np.sum(degrees == 0))

    summary: dict[str, Any] = {
        "num_points": int(num_points),
        "k": int(k),
        "radius_cap_factor": float(radius_cap_factor),
        "mean_edge_len": float(mean_edge_len),
        "r_max": float(r_max),
        "num_candidate_directed_edges": int(num_candidate_edges),
        "num_directed_edges_after_cap": int(adjacency.nnz),
        "num_edges_cut_by_radius": int(num_edges_cut),
        "num_mutual_directed_edges": int(adjacency_mutual.nnz),
        "num_components": int(num_components),
        "largest_component_size": largest_component_size,
        "largest_component_ratio": float(largest_component_size / num_points),
        "isolated_node_count": isolated_node_count,
        "isolated_node_ratio": float(isolated_node_count / num_points),
        "degree_min": int(np.min(degrees)) if degrees.size else 0,
        "degree_max": int(np.max(degrees)) if degrees.size else 0,
        "degree_mean": float(np.mean(degrees)) if degrees.size else 0.0,
    }
    return adjacency_mutual, neighbor_indices, summary


def _neighbor_indices_from_csr(adjacency: sparse.csr_matrix) -> list[np.ndarray]:
    """Extract per-point neighbor index arrays from a CSR adjacency matrix."""
    adjacency = adjacency.tocsr()
    num_points = adjacency.shape[0]
    neighbor_indices: list[np.ndarray] = []
    for point_index in range(num_points):
        start = int(adjacency.indptr[point_index])
        end = int(adjacency.indptr[point_index + 1])
        neighbors = np.asarray(adjacency.indices[start:end], dtype=np.int64)
        neighbor_indices.append(neighbors)
    return neighbor_indices


def _empty_graph_summary(
    num_points: int,
    k: int,
    radius_cap_factor: float,
) -> dict[str, Any]:
    """Return a summary dictionary for degenerate single-point graphs."""
    return {
        "num_points": int(num_points),
        "k": int(k),
        "radius_cap_factor": float(radius_cap_factor),
        "mean_edge_len": 0.0,
        "r_max": 0.0,
        "num_candidate_directed_edges": 0,
        "num_directed_edges_after_cap": 0,
        "num_edges_cut_by_radius": 0,
        "num_mutual_directed_edges": 0,
        "num_components": int(num_points),
        "largest_component_size": int(num_points),
        "largest_component_ratio": 1.0,
        "isolated_node_count": int(num_points),
        "isolated_node_ratio": 1.0,
        "degree_min": 0,
        "degree_max": 0,
        "degree_mean": 0.0,
    }


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Validate point cloud shape and numeric values."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")
    return points
