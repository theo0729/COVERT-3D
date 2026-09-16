"""Normalized Laplacian construction for sparse weighted graphs."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def build_normalized_laplacian(adjacency_weight: sparse.csr_matrix) -> sparse.csr_matrix:
    """Build the symmetric normalized graph Laplacian.

    Computes:

        L_sym = I - D^{-1/2} W D^{-1/2}

    where ``D`` is the degree matrix of ``W``. For isolated nodes, the
    corresponding inverse-square-root degree is set to zero.

    Args:
        adjacency_weight: Sparse CSR weighted adjacency matrix.

    Returns:
        Sparse CSR normalized Laplacian ``L_sym``.
    """
    adjacency_weight = _validate_adjacency(adjacency_weight)
    num_nodes = adjacency_weight.shape[0]

    degree = np.asarray(adjacency_weight.sum(axis=1)).reshape(-1)
    degree = degree.astype(np.float64, copy=False)
    valid_degree = degree > 1e-12
    degree_inv_sqrt = np.zeros(num_nodes, dtype=np.float64)
    degree_inv_sqrt[valid_degree] = 1.0 / np.sqrt(degree[valid_degree])

    degree_inv_sqrt_diag = sparse.diags(degree_inv_sqrt, format="csr", dtype=np.float64)
    identity = sparse.identity(num_nodes, format="csr", dtype=np.float64)

    normalized_adjacency = degree_inv_sqrt_diag @ adjacency_weight @ degree_inv_sqrt_diag
    laplacian = identity - normalized_adjacency
    return laplacian.tocsr()


def _validate_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Validate adjacency matrix shape and sparsity format."""
    if not sparse.isspmatrix(adjacency):
        raise TypeError("adjacency_weight must be a scipy sparse matrix.")
    adjacency = adjacency.tocsr()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_weight must be a square 2D sparse matrix.")
    if adjacency.shape[0] == 0:
        raise ValueError("adjacency_weight must not be empty.")
    return adjacency
