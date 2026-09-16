"""VAST-GTR restoration pipeline assembly."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from vast.restoration.displacement import compute_geometry_displacement
from vast.restoration.laplacian_inpaint import inpaint_laplacian_surface
from vast.restoration.roi_builder import build_restoration_roi
from vast.restoration.strain_veto import apply_strain_veto


def extract_defects_gtr(
    points: np.ndarray,
    scores: np.ndarray,
    adjacency_csr: sparse.csr_matrix,
    neighbor_indices: list[np.ndarray],
    seed_percentile: float = 90.0,
    dilation_hops: int = 5,
    depth_threshold: float = 0.2,
    strain_threshold: float = 0.15,
) -> dict[str, Any]:
    """Run the full Geometry-Thermodynamic Restoration defect extraction pipeline.

    Args:
        points: Scaled point coordinates with shape ``(N, 3)``.
        scores: Thermodynamic anomaly scores with shape ``(N,)``.
        adjacency_csr: Sparse graph adjacency with shape ``(N, N)``.
        neighbor_indices: Per-point mutual neighbor index lists.
        seed_percentile: Percentile for high-energy seed extraction.
        dilation_hops: Graph hops used to grow the internal ROI.
        depth_threshold: Minimum inpainting displacement for geometry anomalies.
        strain_threshold: Maximum mean edge strain retained as a soft defect.

    Returns:
        Dictionary containing masks, restored geometry, displacements, and strains.
    """
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    adjacency_csr = sparse.csr_matrix(adjacency_csr).tocsr()
    if adjacency_csr.shape[0] != points.shape[0] or scores.shape[0] != points.shape[0]:
        raise ValueError("points, scores, and adjacency_csr must share the same length.")
    if len(neighbor_indices) != points.shape[0]:
        raise ValueError("neighbor_indices length must match points length.")

    seed_mask, internal_mask, boundary_mask = build_restoration_roi(
        scores=scores,
        adjacency_csr=adjacency_csr,
        seed_percentile=seed_percentile,
        dilation_hops=dilation_hops,
    )
    inpainted_points = inpaint_laplacian_surface(
        points=points,
        adjacency_csr=adjacency_csr,
        internal_mask=internal_mask,
        boundary_mask=boundary_mask,
    )
    displacements, geometry_anomaly_mask = compute_geometry_displacement(
        points=points,
        inpainted_points=inpainted_points,
        internal_mask=internal_mask,
        depth_threshold=depth_threshold,
    )
    final_defect_mask, point_strains = apply_strain_veto(
        points=points,
        inpainted_points=inpainted_points,
        neighbor_indices=neighbor_indices,
        geometry_anomaly_mask=geometry_anomaly_mask,
        strain_threshold=strain_threshold,
    )

    return {
        "final_defect_mask": final_defect_mask,
        "inpainted_points": inpainted_points,
        "displacements": displacements,
        "point_strains": point_strains,
        "seed_mask": seed_mask,
        "internal_mask": internal_mask,
        "boundary_mask": boundary_mask,
        "geometry_anomaly_mask": geometry_anomaly_mask,
    }
