"""Dirichlet Laplacian inpainting for soap-film surface restoration."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def build_restoration_adjacency(
    adjacency_csr: sparse.csr_matrix,
    weight_mode: str = "inverse_distance",
    eps: float = 1e-12,
) -> sparse.csr_matrix:
    """Build a symmetric restoration adjacency with mild distance-based weights."""
    adjacency = sparse.csr_matrix(adjacency_csr).tocsr()
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    weight_mode = weight_mode.strip().lower()

    if weight_mode == "inverse_distance":
        weights = 1.0 / (adjacency.data + float(eps))
    elif weight_mode == "gaussian":
        edge_distances = adjacency.data
        sigma = float(np.median(edge_distances)) if edge_distances.size else 1.0
        sigma = max(sigma, float(eps))
        weights = np.exp(-(edge_distances ** 2) / (2.0 * sigma ** 2))
    else:
        raise ValueError("weight_mode must be one of: inverse_distance, gaussian")

    restoration = adjacency.copy()
    restoration.data = np.asarray(weights, dtype=np.float64)
    return restoration.tocsr()


def build_weighted_laplacian(adjacency_csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """Build the weighted graph Laplacian ``L = D - W`` from a weighted adjacency."""
    adjacency = sparse.csr_matrix(adjacency_csr).tocsr()
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    degree = np.asarray(adjacency.sum(axis=1), dtype=np.float64).reshape(-1)
    degree_matrix = sparse.diags(degree, format="csr", dtype=np.float64)
    return (degree_matrix - adjacency).tocsr()


def _build_combinatorial_laplacian(adjacency_csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """Build the combinatorial graph Laplacian ``L = D - A`` from a binary adjacency."""
    adjacency = sparse.csr_matrix(adjacency_csr).tocsr()
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.float64)
    adjacency.eliminate_zeros()
    return build_weighted_laplacian(adjacency)


def inpaint_global_dirichlet(
    points: np.ndarray,
    laplacian: sparse.csr_matrix,
    unknown_mask: np.ndarray,
    fixed_mask: np.ndarray | None = None,
    eps_reg: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    """Solve ``L_II X_I = -L_IB X_B`` for all unknown vertices at once."""
    points = np.asarray(points, dtype=np.float64)
    unknown_mask = np.asarray(unknown_mask, dtype=bool).reshape(-1)
    if fixed_mask is None:
        fixed_mask = ~unknown_mask
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)

    restored_points = points.copy()
    status: dict[str, float | str | bool] = {
        "solver_ok": True,
        "solver_status": "skipped_empty_unknown",
        "residual_norm": 0.0,
    }

    unknown_indices = np.flatnonzero(unknown_mask)
    fixed_indices = np.flatnonzero(fixed_mask)
    if unknown_indices.size == 0:
        status["solver_status"] = "empty_unknown"
        return restored_points, status
    if fixed_indices.size == 0:
        status["solver_ok"] = False
        status["solver_status"] = "empty_fixed"
        return restored_points, status

    laplacian = sparse.csr_matrix(laplacian).tocsr()
    laplacian_ii = laplacian[unknown_indices][:, unknown_indices].tocsc()
    laplacian_ib = laplacian[unknown_indices][:, fixed_indices].tocsc()
    if eps_reg > 0.0:
        laplacian_ii = laplacian_ii + float(eps_reg) * sparse.eye(
            unknown_indices.size,
            format="csc",
            dtype=np.float64,
        )

    fixed_coords = points[fixed_indices]
    solved_internal = np.empty((unknown_indices.size, 3), dtype=np.float64)
    residual_norm = 0.0
    try:
        for axis in range(3):
            rhs = -(laplacian_ib @ fixed_coords[:, axis])
            solved_axis = spsolve(laplacian_ii, rhs)
            solved_internal[:, axis] = solved_axis
            residual = laplacian_ii @ solved_axis + (laplacian_ib @ fixed_coords[:, axis])
            residual_norm = max(residual_norm, float(np.linalg.norm(residual)))
        restored_points[unknown_indices] = solved_internal
        status["solver_status"] = "ok"
        status["residual_norm"] = residual_norm
    except Exception as exc:  # pragma: no cover - solver failures are environment-specific.
        status["solver_ok"] = False
        status["solver_status"] = f"solver_failed:{type(exc).__name__}"
        status["residual_norm"] = residual_norm

    return restored_points, status


def inpaint_laplacian_surface(
    points: np.ndarray,
    adjacency_csr: sparse.csr_matrix,
    internal_mask: np.ndarray,
    boundary_mask: np.ndarray,
) -> np.ndarray:
    """Solve the Dirichlet problem ``L_II X_I = -L_IB X_B`` for internal coordinates.

    Legacy combinatorial inpainting interface kept for backward compatibility.
    """
    points = np.asarray(points, dtype=np.float64)
    internal_mask = np.asarray(internal_mask, dtype=bool).reshape(-1)
    boundary_mask = np.asarray(boundary_mask, dtype=bool).reshape(-1)

    laplacian = _build_combinatorial_laplacian(adjacency_csr)
    restored_points, _status = inpaint_global_dirichlet(
        points=points,
        laplacian=laplacian,
        unknown_mask=internal_mask,
        fixed_mask=boundary_mask,
    )
    return restored_points
