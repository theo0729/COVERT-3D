"""ROI construction: thermodynamic seed graph closing (dilate + erode) and 1-hop boundary moat."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _graph_dilate(mask: np.ndarray, adjacency: sparse.csr_matrix, hops: int) -> np.ndarray:
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


def _graph_erode(mask: np.ndarray, adjacency: sparse.csr_matrix, hops: int) -> np.ndarray:
    """Erode a boolean mask inward by dilating the background."""
    # Graph erosion is the complement of dilating the background.
    background = ~np.asarray(mask, dtype=bool).reshape(-1)
    dilated_background = _graph_dilate(background, adjacency, hops)
    return ~dilated_background


def build_restoration_roi(
    scores: np.ndarray,
    adjacency_csr: sparse.csr_matrix,
    seed_percentile: float = 90.0,
    dilation_hops: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build internal ROI using Graph Closing (Dilate + Erode) to fill holes without expanding boundaries."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    adjacency_csr = sparse.csr_matrix(adjacency_csr).tocsr()
    
    seed_threshold = float(np.percentile(scores, seed_percentile))
    seed_mask = scores >= seed_threshold

    # Dilate to fill enclosed gaps and connect nearby fragments.
    dilated_mask = _graph_dilate(seed_mask, adjacency_csr, hops=int(dilation_hops))
    
    # Erode back to the original outer boundary while retaining filled gaps.
    internal_mask = _graph_erode(dilated_mask, adjacency_csr, hops=int(dilation_hops))
    
    # Use the one-hop exterior ring as the fixed reference boundary.
    expanded_once = _graph_dilate(internal_mask, adjacency_csr, hops=1)
    boundary_mask = np.logical_xor(expanded_once, internal_mask)

    return (
        np.asarray(seed_mask, dtype=bool),
        np.asarray(internal_mask, dtype=bool),
        np.asarray(boundary_mask, dtype=bool),
    )
