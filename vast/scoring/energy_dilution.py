"""Energy-dilution filtering with PCA elongation-aware joint rejection."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse import csgraph


def _compute_elongation(points: np.ndarray) -> float:
    """Compute 3D connected-component elongation as lambda_1 / lambda_2 from PCA."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return 1.0

    centered = points - np.mean(points, axis=0)
    cov = np.dot(centered.T, centered) / points.shape[0]
    evals = eigh(cov, eigvals_only=True)
    evals = np.sort(evals)[::-1]
    if evals[1] < 1e-6:
        return float("inf")
    return float(evals[0] / evals[1])


def filter_candidates_by_energy_dilution(
    scores: np.ndarray,
    adjacency_matrix: sp.csr_matrix,
    points: np.ndarray,
    base_threshold: float,
    mean_energy_threshold: float,
    min_points: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Filter candidate components via mean energy and PCA elongation joint rules."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    points = np.asarray(points, dtype=np.float64)
    adjacency_matrix = sp.csr_matrix(adjacency_matrix).tocsr()
    if points.shape[0] != scores.shape[0]:
        raise ValueError("points must have the same length as scores.")

    mask_base = scores >= base_threshold
    num_points = scores.shape[0]

    diag_mask = sp.diags(mask_base.astype(int))
    adj_sub = diag_mask.dot(adjacency_matrix).dot(diag_mask)
    num_components, labels = csgraph.connected_components(adj_sub, directed=False)

    candidate_mask = np.zeros(num_points, dtype=bool)
    filtered_labels = np.zeros(num_points, dtype=np.int32)

    for comp_id in range(num_components):
        comp_indices = np.flatnonzero((labels == comp_id) & mask_base)
        if comp_indices.size < min_points:
            continue

        mean_energy = float(np.mean(scores[comp_indices]))
        comp_points = points[comp_indices]
        elongation = _compute_elongation(comp_points)

        is_elongated = elongation > 3.5
        required_energy = mean_energy_threshold * 1.5 if is_elongated else mean_energy_threshold

        if mean_energy >= required_energy:
            candidate_mask[comp_indices] = True
            filtered_labels[comp_indices] = comp_id + 1

    return candidate_mask, filtered_labels, num_components
