"""VAST-GTR v2: per-component graph closing + global Laplacian inpainting."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from vast.restoration.graph_ops import (
    fill_enclosed_holes as fill_enclosed_holes_graph,
    graph_close,
    label_connected_components,
    refine_component_by_displacement,
    remove_small_components,
)
from vast.restoration.laplacian_inpaint import (
    build_restoration_adjacency,
    build_weighted_laplacian,
    inpaint_global_dirichlet,
)


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize a 1D array with robust order statistics."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _compute_component_displacements(
    points: np.ndarray,
    restored_points: np.ndarray,
    component_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute displacement magnitudes inside one component mask."""
    component_mask = np.asarray(component_mask, dtype=bool).reshape(-1)
    comp_displacements = np.linalg.norm(
        points[component_mask] - restored_points[component_mask],
        axis=1,
    )
    return comp_displacements, _distribution_summary(comp_displacements)


def _compute_component_strains(
    points: np.ndarray,
    restored_points: np.ndarray,
    component_mask: np.ndarray,
    adjacency_csr: sparse.csr_matrix,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute edge strains for component-internal and 1-hop neighborhood edges."""
    component_mask = np.asarray(component_mask, dtype=bool).reshape(-1)
    adjacency = sparse.csr_matrix(adjacency_csr).tocsr()
    edge_strains: list[float] = []
    point_strain_acc = np.zeros(points.shape[0], dtype=np.float64)
    point_strain_count = np.zeros(points.shape[0], dtype=np.int32)
    seen_edges: set[tuple[int, int]] = set()

    for center in np.flatnonzero(component_mask):
        begin = int(adjacency.indptr[center])
        end = int(adjacency.indptr[center + 1])
        for neighbor in adjacency.indices[begin:end]:
            neighbor = int(neighbor)
            edge_key = (min(center, neighbor), max(center, neighbor))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            old_len = float(np.linalg.norm(points[center] - points[neighbor]))
            new_len = float(np.linalg.norm(restored_points[center] - restored_points[neighbor]))
            strain = abs(old_len - new_len) / (old_len + float(eps))
            edge_strains.append(strain)
            point_strain_acc[center] += strain
            point_strain_acc[neighbor] += strain
            point_strain_count[center] += 1
            point_strain_count[neighbor] += 1

    point_strains = np.zeros(points.shape[0], dtype=np.float64)
    active = point_strain_count > 0
    point_strains[active] = point_strain_acc[active] / point_strain_count[active]
    return point_strains, _distribution_summary(np.asarray(edge_strains, dtype=np.float64))


def _new_component_report(component_id: int, num_seed_points: int) -> dict[str, Any]:
    """Create a component report with default hole-fill and refinement fields."""
    return {
        "component_id": int(component_id),
        "num_seed_points": int(num_seed_points),
        "num_closed_points": 0,
        "num_filled_points": 0,
        "num_hole_points": 0,
        "closed_growth_ratio": 0.0,
        "hole_fill_used": False,
        "hole_fill_growth_ratio": 1.0,
        "hole_fill_rejection_reason": "",
        "accepted": False,
        "rejection_reason": "",
        "D_mean": 0.0,
        "D_median": 0.0,
        "D_p90": 0.0,
        "D_max": 0.0,
        "strain_mean": 0.0,
        "strain_median": 0.0,
        "strain_p90": 0.0,
        "strain_max": 0.0,
        "point_refine_used": False,
        "refine_high_threshold": 0.0,
        "refine_low_threshold": 0.0,
        "num_refined_points": 0,
        "refinement_rejection_reason": "",
    }


def _print_gtr_v2_audit(summary: dict[str, Any], component_reports: list[dict[str, Any]]) -> None:
    """Print concise GTR v2 audit lines to the console."""
    print(f"[GTR v2] score_source={summary['gtr_score_source']}")
    print(f"[GTR v2] seed_threshold={summary['seed_threshold']:.6f}")
    print(f"[GTR v2] seed_points={summary['num_seed_points']}")
    print(
        "[GTR v2] seed_components_before_filter="
        f"{summary['num_seed_components_before_filter']}"
    )
    print(
        "[GTR v2] seed_components_after_filter="
        f"{summary['num_seed_components_after_filter']}"
    )
    print(f"[GTR v2] closed_points={summary['num_closed_points']}")
    print(f"[GTR v2] filled_points={summary['num_filled_points']}")
    print(f"[GTR v2] hole_points={summary['num_hole_points']}")
    print(f"[GTR v2] unknown_points={summary['num_unknown_points']}")
    print(f"[GTR v2] refined_points={summary['num_refined_points']}")
    print(f"[GTR v2] accepted_components={summary['num_accepted_components']}")
    print(f"[GTR v2] rejected_components={summary['num_rejected_components']}")
    print(f"[GTR v2] final_defect_points={summary['num_defect_points']}")

    for report in component_reports:
        print(
            "[GTR v2][component] "
            f"id={report['component_id']} "
            f"seed={report['num_seed_points']} "
            f"closed={report['num_closed_points']} "
            f"filled={report['num_filled_points']} "
            f"hole={report['num_hole_points']} "
            f"refined={report['num_refined_points']} "
            f"D_p90={report['D_p90']:.6f} "
            f"strain_p90={report['strain_p90']:.6f} "
            f"accepted={report['accepted']} "
            f"reason={report['rejection_reason'] or 'ok'}"
        )


def _build_summary(
    *,
    gtr_score_source: str,
    seed_percentile: float,
    seed_threshold: float,
    seed_mask: np.ndarray,
    num_seed_components_before: int,
    num_seed_components_after: int,
    closing_hops: int,
    min_seed_component_size: int,
    min_closed_component_size: int,
    closed_component_id: int,
    closed_mask: np.ndarray,
    filled_mask: np.ndarray,
    hole_mask: np.ndarray,
    unknown_mask: np.ndarray,
    refined_mask: np.ndarray,
    final_defect_mask: np.ndarray,
    displacement_threshold: float,
    use_strain_veto: bool,
    strain_threshold: float,
    restoration_weight: str,
    solver_status: dict[str, float | str | bool],
    num_accepted: int,
    num_rejected: int,
    fill_enclosed_holes: bool,
    hole_fill_roi_hops: int,
    max_hole_growth_ratio: float,
    use_point_refinement: bool,
    num_points: int,
) -> dict[str, Any]:
    """Assemble the global GTR v2 summary dictionary."""
    return {
        "method": "vast_gtr_v2",
        "gtr_score_source": gtr_score_source,
        "seed_percentile": float(seed_percentile),
        "seed_threshold": float(seed_threshold),
        "seed_ratio": float(np.mean(seed_mask)),
        "num_seed_points": int(np.sum(seed_mask)),
        "num_seed_components_before_filter": int(num_seed_components_before),
        "num_seed_components_after_filter": int(num_seed_components_after),
        "closing_hops": int(closing_hops),
        "min_seed_component_size": int(min_seed_component_size),
        "min_closed_component_size": int(min_closed_component_size),
        "num_closed_components": int(closed_component_id),
        "fill_enclosed_holes": bool(fill_enclosed_holes),
        "hole_fill_roi_hops": int(hole_fill_roi_hops),
        "max_hole_growth_ratio": float(max_hole_growth_ratio),
        "num_closed_points": int(np.sum(closed_mask)),
        "num_filled_points": int(np.sum(filled_mask)),
        "num_hole_points": int(np.sum(hole_mask)),
        "num_unknown_points": int(np.sum(unknown_mask)),
        "num_fixed_points": int(num_points - int(np.sum(unknown_mask))),
        "unknown_ratio": float(np.mean(unknown_mask)),
        "use_point_refinement": bool(use_point_refinement),
        "num_refined_points": int(np.sum(refined_mask)),
        "displacement_threshold": float(displacement_threshold),
        "use_strain_veto": bool(use_strain_veto),
        "strain_threshold": float(strain_threshold),
        "restoration_weight": restoration_weight,
        "solver_status": str(solver_status["solver_status"]),
        "solver_residual_norm": float(solver_status["residual_norm"]),
        "num_accepted_components": int(num_accepted),
        "num_rejected_components": int(num_rejected),
        "num_defect_points": int(np.sum(final_defect_mask)),
        "defect_ratio": float(np.mean(final_defect_mask)),
    }


def _build_empty_result(
    points: np.ndarray,
    seed_mask: np.ndarray,
    seed_threshold: float,
    seed_percentile: float,
    num_seed_components_before: int,
    num_seed_components_after: int,
    closing_hops: int,
    min_seed_component_size: int,
    min_closed_component_size: int,
    displacement_threshold: float,
    use_strain_veto: bool,
    strain_threshold: float,
    restoration_weight: str,
    gtr_score_source: str,
    component_reports: list[dict[str, Any]],
    fill_enclosed_holes: bool,
    hole_fill_roi_hops: int,
    max_hole_growth_ratio: float,
    use_point_refinement: bool,
    solver_status: dict[str, float | str | bool] | None = None,
) -> dict[str, Any]:
    """Assemble a consistent empty-or-failed GTR v2 result dictionary."""
    if solver_status is None:
        solver_status = {
            "solver_ok": True,
            "solver_status": "empty_unknown",
            "residual_norm": 0.0,
        }

    empty_mask = np.zeros(points.shape[0], dtype=bool)
    summary = _build_summary(
        gtr_score_source=gtr_score_source,
        seed_percentile=seed_percentile,
        seed_threshold=seed_threshold,
        seed_mask=seed_mask,
        num_seed_components_before=num_seed_components_before,
        num_seed_components_after=num_seed_components_after,
        closing_hops=closing_hops,
        min_seed_component_size=min_seed_component_size,
        min_closed_component_size=min_closed_component_size,
        closed_component_id=0,
        closed_mask=empty_mask,
        filled_mask=empty_mask,
        hole_mask=empty_mask,
        unknown_mask=empty_mask,
        refined_mask=empty_mask,
        final_defect_mask=empty_mask,
        displacement_threshold=displacement_threshold,
        use_strain_veto=use_strain_veto,
        strain_threshold=strain_threshold,
        restoration_weight=restoration_weight,
        solver_status=solver_status,
        num_accepted=0,
        num_rejected=len(component_reports),
        fill_enclosed_holes=fill_enclosed_holes,
        hole_fill_roi_hops=hole_fill_roi_hops,
        max_hole_growth_ratio=max_hole_growth_ratio,
        use_point_refinement=use_point_refinement,
        num_points=points.shape[0],
    )
    _print_gtr_v2_audit(summary, component_reports)
    return {
        "final_defect_mask": empty_mask,
        "seed_mask": seed_mask,
        "closed_mask": empty_mask,
        "filled_mask": empty_mask,
        "hole_mask": empty_mask,
        "unknown_mask": empty_mask,
        "refined_mask": empty_mask,
        "restored_points": points.copy(),
        "displacements": np.zeros(points.shape[0], dtype=np.float64),
        "point_strains": np.zeros(points.shape[0], dtype=np.float64),
        "component_id_map": np.zeros(points.shape[0], dtype=np.int32),
        "component_reports": component_reports,
        "summary": summary,
    }


def extract_defects_gtr_v2(
    points: np.ndarray,
    gtr_scores: np.ndarray,
    adjacency_csr: sparse.csr_matrix,
    neighbor_indices: list[np.ndarray] | None = None,
    raw_scores: np.ndarray | None = None,
    local_scores: np.ndarray | None = None,
    seed_percentile: float = 92.0,
    closing_hops: int = 1,
    min_seed_component_size: int = 10,
    min_closed_component_size: int = 20,
    displacement_threshold: float = 0.15,
    use_strain_veto: bool = False,
    strain_threshold: float = 0.35,
    restoration_weight: str = "inverse_distance",
    fill_enclosed_holes: bool = True,
    hole_fill_roi_hops: int = 4,
    max_hole_growth_ratio: float = 3.0,
    use_point_refinement: bool = True,
    point_refine_high_percentile: float = 70.0,
    point_refine_low_ratio: float = 0.4,
    point_refine_min_points: int = 10,
    eps: float = 1e-12,
    eps_reg: float = 1e-8,
    gtr_score_source: str = "fused_max_raw_local",
) -> dict[str, Any]:
    """Run VAST-GTR v2 defect extraction with global Laplacian inpainting."""
    del raw_scores, local_scores

    points = np.asarray(points, dtype=np.float64)
    gtr_scores = np.asarray(gtr_scores, dtype=np.float64).reshape(-1)
    adjacency_csr = sparse.csr_matrix(adjacency_csr).tocsr()
    adjacency_csr = adjacency_csr.maximum(adjacency_csr.transpose()).tocsr()

    if points.shape[0] != gtr_scores.shape[0] or adjacency_csr.shape[0] != points.shape[0]:
        raise ValueError("points, gtr_scores, and adjacency_csr must share the same length.")
    if (fill_enclosed_holes or use_point_refinement) and neighbor_indices is None:
        raise ValueError("neighbor_indices is required when hole filling or point refinement is enabled.")
    if neighbor_indices is not None and len(neighbor_indices) != points.shape[0]:
        raise ValueError("neighbor_indices length must match points length.")

    topology_adjacency = adjacency_csr.copy()
    topology_adjacency.data = np.ones_like(topology_adjacency.data, dtype=np.float64)
    topology_adjacency.eliminate_zeros()

    seed_threshold = float(np.percentile(gtr_scores, seed_percentile))
    seed_mask = gtr_scores >= seed_threshold

    _seed_labels, num_seed_components_before = label_connected_components(seed_mask, topology_adjacency)
    filtered_seed_mask = remove_small_components(
        seed_mask,
        topology_adjacency,
        min_size=min_seed_component_size,
    )
    filtered_seed_labels, num_seed_components_after = label_connected_components(
        filtered_seed_mask,
        topology_adjacency,
    )

    component_reports: list[dict[str, Any]] = []
    closed_mask = np.zeros(points.shape[0], dtype=bool)
    filled_mask = np.zeros(points.shape[0], dtype=bool)
    component_id_map = np.zeros(points.shape[0], dtype=np.int32)
    closed_component_id = 0
    pending_components: list[tuple[int, np.ndarray, np.ndarray, dict[str, Any]]] = []

    for component_id in range(1, num_seed_components_after + 1):
        component_seed_mask = filtered_seed_labels == component_id
        num_seed_points = int(np.sum(component_seed_mask))
        report = _new_component_report(component_id, num_seed_points)

        if num_seed_points < int(min_seed_component_size):
            report["rejection_reason"] = "too_small_seed_component"
            component_reports.append(report)
            continue

        component_closed = graph_close(component_seed_mask, topology_adjacency, hops=int(closing_hops))
        num_closed_points = int(np.sum(component_closed))
        report["num_closed_points"] = num_closed_points
        report["closed_growth_ratio"] = float(num_closed_points / max(num_seed_points, 1))

        if num_closed_points < int(min_closed_component_size):
            report["rejection_reason"] = "too_small_closed_component"
            component_reports.append(report)
            continue

        if fill_enclosed_holes and neighbor_indices is not None:
            component_filled, fill_report = fill_enclosed_holes_graph(
                component_closed,
                neighbor_indices,
                roi_hops=hole_fill_roi_hops,
                max_growth_ratio=max_hole_growth_ratio,
            )
        else:
            component_filled = component_closed.copy()
            fill_report = {
                "num_closed_points": num_closed_points,
                "num_hole_points": 0,
                "num_filled_points": num_closed_points,
                "hole_fill_growth_ratio": 1.0,
                "hole_fill_used": False,
                "hole_fill_rejection_reason": "disabled",
            }

        report["num_filled_points"] = int(fill_report["num_filled_points"])
        report["num_hole_points"] = int(fill_report["num_hole_points"])
        report["hole_fill_used"] = bool(fill_report["hole_fill_used"])
        report["hole_fill_growth_ratio"] = float(fill_report["hole_fill_growth_ratio"])
        report["hole_fill_rejection_reason"] = str(fill_report.get("hole_fill_rejection_reason", ""))

        closed_component_id += 1
        closed_mask |= component_closed
        filled_mask |= component_filled
        component_id_map[component_filled] = closed_component_id
        report["component_id"] = int(closed_component_id)
        pending_components.append((closed_component_id, component_closed, component_filled, report))

    unknown_mask = filled_mask.copy()
    hole_mask = filled_mask & ~closed_mask
    num_unknown_points = int(np.sum(unknown_mask))

    empty_kwargs = {
        "fill_enclosed_holes": fill_enclosed_holes,
        "hole_fill_roi_hops": hole_fill_roi_hops,
        "max_hole_growth_ratio": max_hole_growth_ratio,
        "use_point_refinement": use_point_refinement,
    }

    if num_unknown_points == 0:
        return _build_empty_result(
            points=points,
            seed_mask=seed_mask,
            seed_threshold=seed_threshold,
            seed_percentile=seed_percentile,
            num_seed_components_before=num_seed_components_before,
            num_seed_components_after=num_seed_components_after,
            closing_hops=closing_hops,
            min_seed_component_size=min_seed_component_size,
            min_closed_component_size=min_closed_component_size,
            displacement_threshold=displacement_threshold,
            use_strain_veto=use_strain_veto,
            strain_threshold=strain_threshold,
            restoration_weight=restoration_weight,
            gtr_score_source=gtr_score_source,
            component_reports=component_reports,
            **empty_kwargs,
        )

    restoration_adjacency = build_restoration_adjacency(
        adjacency_csr,
        weight_mode=restoration_weight,
        eps=eps,
    )
    restoration_laplacian = build_weighted_laplacian(restoration_adjacency)
    restored_points, solver_status = inpaint_global_dirichlet(
        points=points,
        laplacian=restoration_laplacian,
        unknown_mask=unknown_mask,
        fixed_mask=~unknown_mask,
        eps_reg=eps_reg,
    )

    if not bool(solver_status.get("solver_ok", False)):
        for _cid, _closed, _filled, report in pending_components:
            report["rejection_reason"] = "solver_failed"
            component_reports.append(report)
        return _build_empty_result(
            points=points,
            seed_mask=seed_mask,
            seed_threshold=seed_threshold,
            seed_percentile=seed_percentile,
            num_seed_components_before=num_seed_components_before,
            num_seed_components_after=num_seed_components_after,
            closing_hops=closing_hops,
            min_seed_component_size=min_seed_component_size,
            min_closed_component_size=min_closed_component_size,
            displacement_threshold=displacement_threshold,
            use_strain_veto=use_strain_veto,
            strain_threshold=strain_threshold,
            restoration_weight=restoration_weight,
            gtr_score_source=gtr_score_source,
            component_reports=component_reports,
            solver_status=solver_status,
            **empty_kwargs,
        )

    final_defect_mask = np.zeros(points.shape[0], dtype=bool)
    refined_mask = np.zeros(points.shape[0], dtype=bool)
    displacements = np.zeros(points.shape[0], dtype=np.float64)
    point_strains = np.zeros(points.shape[0], dtype=np.float64)

    for _cid, _component_closed, component_filled, report in pending_components:
        comp_displacements, disp_stats = _compute_component_displacements(
            points,
            restored_points,
            component_filled,
        )
        comp_point_strains, strain_stats = _compute_component_strains(
            points,
            restored_points,
            component_filled,
            topology_adjacency,
            eps=eps,
        )

        displacements[component_filled] = comp_displacements
        point_strains[component_filled] = comp_point_strains[component_filled]
        report.update(
            {
                "D_mean": disp_stats["mean"],
                "D_median": disp_stats["median"],
                "D_p90": disp_stats["p90"],
                "D_max": disp_stats["max"],
                "strain_mean": strain_stats["mean"],
                "strain_median": strain_stats["median"],
                "strain_p90": strain_stats["p90"],
                "strain_max": strain_stats["max"],
            }
        )

        if disp_stats["p90"] < float(displacement_threshold):
            report["accepted"] = False
            report["rejection_reason"] = "low_displacement_p90"
        elif use_strain_veto and strain_stats["p90"] > float(strain_threshold):
            report["accepted"] = False
            report["rejection_reason"] = "high_component_strain"
        else:
            report["accepted"] = True
            report["rejection_reason"] = ""
            if use_point_refinement and neighbor_indices is not None:
                component_refined, refine_report = refine_component_by_displacement(
                    component_filled,
                    displacements,
                    neighbor_indices,
                    high_percentile=point_refine_high_percentile,
                    low_ratio=point_refine_low_ratio,
                    min_points=point_refine_min_points,
                    eps=eps,
                )
            else:
                component_refined = component_filled.copy()
                refine_report = {
                    "point_refine_used": False,
                    "refine_high_threshold": 0.0,
                    "refine_low_threshold": 0.0,
                    "num_refined_points": int(np.sum(component_filled)),
                    "refinement_rejection_reason": "disabled",
                }

            report["point_refine_used"] = bool(refine_report["point_refine_used"])
            report["refine_high_threshold"] = float(refine_report["refine_high_threshold"])
            report["refine_low_threshold"] = float(refine_report["refine_low_threshold"])
            report["num_refined_points"] = int(refine_report["num_refined_points"])
            report["refinement_rejection_reason"] = str(
                refine_report.get("refinement_rejection_reason", "")
            )
            refined_mask |= component_refined
            final_defect_mask[component_refined] = True

        component_reports.append(report)

    num_accepted = int(sum(1 for report in component_reports if report["accepted"]))
    num_rejected = len(component_reports) - num_accepted
    summary = _build_summary(
        gtr_score_source=gtr_score_source,
        seed_percentile=seed_percentile,
        seed_threshold=seed_threshold,
        seed_mask=seed_mask,
        num_seed_components_before=num_seed_components_before,
        num_seed_components_after=num_seed_components_after,
        closing_hops=closing_hops,
        min_seed_component_size=min_seed_component_size,
        min_closed_component_size=min_closed_component_size,
        closed_component_id=closed_component_id,
        closed_mask=closed_mask,
        filled_mask=filled_mask,
        hole_mask=hole_mask,
        unknown_mask=unknown_mask,
        refined_mask=refined_mask,
        final_defect_mask=final_defect_mask,
        displacement_threshold=displacement_threshold,
        use_strain_veto=use_strain_veto,
        strain_threshold=strain_threshold,
        restoration_weight=restoration_weight,
        solver_status=solver_status,
        num_accepted=num_accepted,
        num_rejected=num_rejected,
        fill_enclosed_holes=fill_enclosed_holes,
        hole_fill_roi_hops=hole_fill_roi_hops,
        max_hole_growth_ratio=max_hole_growth_ratio,
        use_point_refinement=use_point_refinement,
        num_points=points.shape[0],
    )
    _print_gtr_v2_audit(summary, component_reports)

    return {
        "final_defect_mask": final_defect_mask,
        "seed_mask": seed_mask,
        "closed_mask": closed_mask,
        "filled_mask": filled_mask,
        "hole_mask": hole_mask,
        "unknown_mask": unknown_mask,
        "refined_mask": refined_mask,
        "restored_points": restored_points,
        "displacements": displacements,
        "point_strains": point_strains,
        "component_id_map": component_id_map,
        "component_reports": component_reports,
        "summary": summary,
    }
