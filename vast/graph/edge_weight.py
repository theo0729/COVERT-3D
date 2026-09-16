"""Variation-aware edge weight computation for sparse graphs."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def compute_variation_aware_weights(
    adjacency_dist: sparse.csr_matrix,
    c_abs: np.ndarray,
    gamma: float = 5.0,
) -> sparse.csr_matrix:
    """Convert a distance-weighted adjacency matrix into variation-aware weights.

    Each edge weight is computed with the VAST L1 penalty form:

        W = exp( - (gamma * max(c_u, c_v) * dist) / (mean_dist + 1e-8) )

    Args:
        adjacency_dist: Sparse CSR adjacency matrix storing Euclidean edge lengths.
        c_abs: Per-node absolute surface variation with shape ``(N,)``.
        gamma: Penalty strength for high-variation edges.

    Returns:
        Sparse CSR adjacency matrix with variation-aware edge weights.
    """
    adjacency_dist = _validate_adjacency(adjacency_dist)
    c_abs = _validate_c_abs(c_abs, num_nodes=adjacency_dist.shape[0])
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")

    coo = adjacency_dist.tocoo()
    if coo.nnz == 0:
        return sparse.csr_matrix(adjacency_dist.shape, dtype=np.float64)

    row = coo.row.astype(np.int64, copy=False)
    col = coo.col.astype(np.int64, copy=False)
    dist = coo.data.astype(np.float64, copy=False)

    off_diagonal = row != col
    row = row[off_diagonal]
    col = col[off_diagonal]
    dist = dist[off_diagonal]

    positive_mask = dist > 0.0
    mean_dist = float(np.mean(dist[positive_mask])) if np.any(positive_mask) else 0.0
    mean_dist_denom = mean_dist + 1e-8

    c_u = c_abs[row]
    c_v = c_abs[col]
    penalty = np.maximum(c_u, c_v)
    weights = np.exp(-(gamma * penalty * dist) / mean_dist_denom)

    adjacency_weight = sparse.coo_matrix(
        (weights, (row, col)),
        shape=adjacency_dist.shape,
        dtype=np.float64,
    ).tocsr()
    adjacency_weight.eliminate_zeros()
    return adjacency_weight


def _validate_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Validate adjacency matrix shape and sparsity format."""
    if not sparse.isspmatrix(adjacency):
        raise TypeError("adjacency_dist must be a scipy sparse matrix.")
    adjacency = adjacency.tocsr()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_dist must be a square 2D sparse matrix.")
    if adjacency.shape[0] == 0:
        raise ValueError("adjacency_dist must not be empty.")
    return adjacency


def _validate_c_abs(c_abs: np.ndarray, num_nodes: int) -> np.ndarray:
    """Validate per-node surface variation array."""
    c_abs = np.asarray(c_abs, dtype=np.float64).reshape(-1)
    if c_abs.shape[0] != num_nodes:
        raise ValueError("c_abs must have the same length as the number of nodes.")
    if not np.isfinite(c_abs).all():
        raise ValueError("c_abs must contain only finite values.")
    return c_abs
