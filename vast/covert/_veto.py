"""Frozen defect-veto and Legacy-HKS barrier used by the paper pipeline."""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from vast.covert import _frontend as frontend


STRATEGY_NAME = "new_ac_legacy_hks_defect_veto"


def _outside_shells(
    component_mask: np.ndarray, neighbors: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    component = np.asarray(component_mask, dtype=bool).reshape(-1)
    distance = np.full(component.size, -1, dtype=np.int8)
    queue: deque[int] = deque()
    for value in np.flatnonzero(component):
        distance[int(value)] = 0
        queue.append(int(value))
    while queue:
        current = queue.popleft()
        if distance[current] >= 2:
            continue
        for value in neighbors[current]:
            neighbor = int(value)
            if distance[neighbor] < 0:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance == 1, distance == 2


def _graph_distance(
    neighbors: Sequence[np.ndarray], source_mask: np.ndarray
) -> np.ndarray:
    sources = np.asarray(source_mask, dtype=bool).reshape(-1)
    distance = np.full(sources.size, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for value in np.flatnonzero(sources):
        source = int(value)
        distance[source] = 0
        queue.append(source)
    while queue:
        current = queue.popleft()
        for value in neighbors[current]:
            neighbor = int(value)
            if distance[neighbor] < 0:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


def _boundary_neutral_widest_path(
    landscape: np.ndarray,
    neighbors: Sequence[np.ndarray],
    final_ac: np.ndarray,
) -> dict[str, np.ndarray]:
    response = np.asarray(landscape, dtype=np.float64).reshape(-1)
    if (
        response.size == 0
        or not np.isfinite(response).all()
        or np.any(response < 0.0)
        or np.any(response > 1.0)
    ):
        raise ValueError("Original-SV landscape must be finite and within [0, 1].")
    capacity = np.full(response.size, -np.inf, dtype=np.float64)
    heap: list[tuple[float, int]] = []
    for value in np.flatnonzero(final_ac):
        source = int(value)
        capacity[source] = 1.0
        heapq.heappush(heap, (-1.0, source))
    while heap:
        negative_capacity, current = heapq.heappop(heap)
        current_capacity = -negative_capacity
        if current_capacity < capacity[current]:
            continue
        for value in neighbors[current]:
            neighbor = int(value)
            candidate = min(current_capacity, float(response[neighbor]))
            if candidate > capacity[neighbor]:
                capacity[neighbor] = candidate
                heapq.heappush(heap, (-candidate, neighbor))

    # Deterministic shortest path among all maximum-capacity paths.
    path_length = np.full(response.size, -1, dtype=np.int32)
    queue2: list[tuple[int, int]] = []
    for value in np.flatnonzero(final_ac):
        source = int(value)
        path_length[source] = 0
        heapq.heappush(queue2, (0, source))
    while queue2:
        current_length, current = heapq.heappop(queue2)
        if current_length != int(path_length[current]):
            continue
        for value in neighbors[current]:
            neighbor = int(value)
            if min(capacity[current], float(response[neighbor])) != capacity[neighbor]:
                continue
            candidate_length = current_length + 1
            if path_length[neighbor] < 0 or candidate_length < path_length[neighbor]:
                path_length[neighbor] = candidate_length
                heapq.heappush(queue2, (candidate_length, neighbor))
    return {"capacity": capacity, "path_length_edges": path_length}


def _path_efficiency(
    graph_distance: float, path_length: float, *, reachable: bool
) -> tuple[float, bool]:
    if (
        not reachable
        or not math.isfinite(graph_distance)
        or not math.isfinite(path_length)
        or path_length <= 0.0
    ):
        return float("nan"), False
    value = float(graph_distance / path_length)
    if value <= 0.0 or value > 1.0 + 1e-9:
        raise ValueError("Inconsistent structural-ridge path efficiency.")
    return min(value, 1.0), True


def _normalized(values: Any, size: int) -> tuple[np.ndarray, float, float]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    if raw.shape != (size,) or not np.isfinite(raw).all():
        raise ValueError("Veto response must be finite and match the point count.")
    norm, q01, q99 = frontend.compute_display_normalized_response(
        raw, lower_percentile=1.0, upper_percentile=99.0, eps=1e-12
    )
    return np.asarray(norm, dtype=np.float64), float(q01), float(q99)


def _decide(context: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    forbidden = {"gt", "gt_mask", "category", "sample_id", "filename", "pc_path", "gt_path"}
    leaked = forbidden.intersection(context)
    if leaked:
        raise ValueError(f"Defect veto received forbidden metadata: {sorted(leaked)}")
    pure = np.asarray(context["pure_hks_response"], dtype=np.float64).reshape(-1)
    size = int(pure.size)
    candidate = np.asarray(context["pure_hks_normal_candidate_mask"], dtype=bool).reshape(-1)
    final_ac = np.asarray(context["final_ac_suppression_mask"], dtype=bool).reshape(-1)
    neighbors = frontend.coerce_neighbor_indices(context["neighbor_indices"], size)
    semantic = frontend.classify_pure_hks_components_by_final_ac(
        candidate, final_ac, neighbors
    )
    sv_norm, sv_q01, sv_q99 = _normalized(context["sv_response"], size)
    va_norm, va_q01, va_q99 = _normalized(context["va_hks_response"], size)
    joint = np.minimum(sv_norm, va_norm)
    original = np.asarray(context["original_sv"], dtype=np.float64).reshape(-1)
    c_q01, c_q99 = float(np.percentile(original, 1.0)), float(np.percentile(original, 99.0))
    c_norm = frontend.compute_reference_display_norm(original, c_q01, c_q99, eps=1e-12)
    graph_distance = _graph_distance(neighbors, final_ac)
    paths = _boundary_neutral_widest_path(c_norm, neighbors, final_ac)
    path_length = paths["path_length_edges"]
    labels = np.asarray(semantic["pure_hks_candidate_component_labels"], dtype=np.int32)
    protected = np.zeros(size, dtype=bool)
    records: list[dict[str, Any]] = []
    final_ac_empty = not bool(np.any(final_ac))
    for source in semantic["component_records"]:
        if source["component_class"] != "normal_hks_component":
            continue
        component_id = int(source["component_id"])
        component = labels == component_id
        _shell1, shell2 = _outside_shells(component, neighbors)
        component_p90 = float(np.percentile(joint[component], 90.0))
        shell_available = bool(np.any(shell2))
        shell_p90 = float(np.percentile(joint[shell2], 90.0)) if shell_available else float("nan")
        distances = graph_distance[component]
        distances = distances[distances >= 0]
        reachable = bool(not final_ac_empty and distances.size)
        d_geo = float(np.min(distances)) if reachable else float("nan")
        lengths = path_length[component]
        lengths = lengths[lengths >= 0]
        path_available = bool(not final_ac_empty and lengths.size)
        best_length = float(np.min(lengths)) if path_available else float("nan")
        eta, eta_available = _path_efficiency(
            d_geo, best_length, reachable=reachable and path_available
        )
        anomaly = bool(
            shell_available
            and component_p90 >= float(thresholds["joint_component_p90_min"])
            and shell_p90 >= float(thresholds["joint_shell2_p90_min"])
        )
        boundary = bool(
            reachable and d_geo <= float(thresholds["boundary_max_graph_distance"])
        )
        long_direct = bool(
            eta_available
            and d_geo >= float(thresholds["long_ridge_min_graph_distance"])
            and eta >= float(thresholds["long_ridge_min_path_efficiency"])
        )
        complete = bool(shell_available and reachable and eta_available)
        protect = bool(anomaly and complete and not boundary and not long_direct)
        if protect:
            protected[component] = True
        records.append(
            {
                "component_id": component_id,
                "component_size": int(np.sum(component)),
                "joint_component_p90": component_p90,
                "joint_shell2_p90": shell_p90,
                "shell2_available": shell_available,
                "final_ac_reachable": reachable,
                "d_geo": d_geo,
                "c_widest_path_available": eta_available,
                "c_best_path_length": best_length,
                "c_ridge_path_efficiency": eta,
                "anomaly_supported": anomaly,
                "boundary_normal_explanation": boundary,
                "long_direct_c_normal_explanation": long_direct,
                "required_evidence_complete": complete,
                "defect_protected": protect,
            }
        )
    return {
        "protected": protected,
        "records": records,
        "semantic": semantic,
        "quantiles": {
            "sv_a_q01": sv_q01,
            "sv_a_q99": sv_q99,
            "va_q01": va_q01,
            "va_q99": va_q99,
            "original_sv_c_q01": c_q01,
            "original_sv_c_q99": c_q99,
        },
    }


def build_legacy_hks_defect_veto_override(
    context: dict[str, Any], *, thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Return the exact New-AC/Legacy-HKS suppression with veto barriers."""

    decision = _decide(context, thresholds)
    protected = decision["protected"]
    response = np.asarray(context["pure_hks_response"], dtype=np.float64).reshape(-1)
    display = np.asarray(context["pure_hks_display_level"], dtype=np.int32).reshape(-1)
    final_ac = np.asarray(context["final_ac_suppression_mask"], dtype=bool).reshape(-1)
    neighbors = frontend.coerce_neighbor_indices(context["neighbor_indices"], response.size)
    base = dict(context["base_selected_suppression"])
    base_roots = np.asarray(base["pure_hks_root_mask"], dtype=bool)
    if np.any(protected & final_ac) or np.any(protected & ~base_roots):
        raise AssertionError("Defect protection must be disjoint from Final-AC and inside Legacy roots.")

    if not np.any(protected):
        selected = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in base.items()
        }
        remaining_roots = base_roots.copy()
        pure_support = np.asarray(base["pure_hks_suppression_mask"], dtype=bool).copy()
    else:
        remaining_roots = base_roots & ~protected
        budgets = tuple(int(v) for v in np.asarray(context["display_level_hop_budgets"]))
        growth = frontend._grow_from_roots(
            response=response,
            neighbor_indices=neighbors,
            display_level=display,
            root_mask=remaining_roots,
            level_hop_budgets=budgets,
            normal_threshold=float(context["pure_hks_normal_threshold"]),
            blocked_mask=protected,
        )
        pure_support = np.asarray(growth["suppression_mask"], dtype=bool) & ~protected
        pure_growth = frontend._restrict_growth_result(
            growth["growth_result"], pure_support, remaining_roots
        )
        ac_expanded = np.asarray(base["ac_expanded_mask"], dtype=bool)
        total_roots = ac_expanded | remaining_roots
        total_mask = final_ac | pure_support
        total_growth = frontend._merge_partitioned_growth_results(
            ac_growth_result=base["ac_growth_result"],
            normal_growth_result=pure_growth,
            final_ac_suppression_mask=final_ac,
            normal_structure_support_mask=pure_support & ~final_ac,
            root_mask=total_roots,
        )
        pure_factor = np.ones(response.size, dtype=np.float64)
        pure_factor[pure_support] = 0.0
        combined_factor = np.ones(response.size, dtype=np.float64)
        combined_factor[total_mask] = 0.0
        selected = dict(base)
        selected.update(
            {
                "pure_hks_root_mask": remaining_roots.copy(),
                "growth_root_mask": total_roots,
                "root_hop_budget": frontend.build_root_hop_budget(
                    display, total_roots, level_hop_budgets=budgets
                ),
                "growth_result": total_growth,
                "pure_hks_growth_result": pure_growth,
                "pure_hks_suppression_mask": pure_support,
                "total_suppression_mask": total_mask,
                "pure_hks_hard_component_mask": np.asarray(
                    base["pure_hks_hard_component_mask"], dtype=bool
                ) & ~protected,
                "pure_hks_attenuation_factor": pure_factor,
                "combined_attenuation_factor": combined_factor,
            }
        )
    total_mask = np.asarray(selected["total_suppression_mask"], dtype=bool)
    selected.update(
        {
            "suppression_combination_name": STRATEGY_NAME,
            "sv_suppression_mask": total_mask.copy(),
            "va_hks_suppression_mask": total_mask.copy(),
            "va_hks_allow_mask_root_growth": True,
            "va_hks_growth_blocked_mask": protected.copy(),
            "allow_pure_hks_ac_overlap": True,
        }
    )
    semantic = decision["semantic"]
    return {
        "suppression_combination_name": STRATEGY_NAME,
        "selected_suppression": selected,
        "audit": {
            **semantic,
            "defect_veto_component_records": decision["records"],
            "defect_protected_component_mask": protected.copy(),
            "legacy_hks_root_mask_before_veto": base_roots.copy(),
            "legacy_hks_root_mask_after_veto": remaining_roots.copy(),
            "legacy_hks_suppression_mask_after_veto": pure_support.copy(),
            "legacy_hks_growth_blocked_mask": protected.copy(),
            "sv_active_suppression_mask": total_mask.copy(),
            "va_branch_root_mask": total_mask.copy(),
            "va_hks_growth_blocked_mask": protected.copy(),
            "defect_veto_normalization_quantiles": decision["quantiles"],
            "defect_veto_gate_uses_gt_or_sample_metadata": False,
            "defect_veto_fixed_thresholds": dict(thresholds),
            "summary": {
                "strategy_name": STRATEGY_NAME,
                "suppression_morphology": "legacy_hks",
                "veto_enabled": True,
                "no_veto_exact_baseline_fast_path": not bool(np.any(protected)),
            },
        },
    }

