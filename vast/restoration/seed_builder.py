"""Hysteresis peak-guided candidate builder for VAST-GTR seed generation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from vast.restoration.graph_ops import label_connected_components


@dataclass
class HysteresisCandidate:
    """One hysteresis-grown candidate region anchored by a strong core."""

    label: int
    core_mask: np.ndarray
    candidate_mask: np.ndarray
    peak_index: int
    peak_score: float
    core_size: int
    candidate_size: int
    growth_ratio: float
    prominence: float
    accepted: bool
    rejection_reason: str | None
    growth_too_large: bool = False


@dataclass
class HysteresisCandidateResult:
    """Aggregated output of hysteresis peak-guided candidate building."""

    candidate_masks: list[np.ndarray]
    core_mask: np.ndarray
    weak_mask: np.ndarray
    assigned_mask: np.ndarray
    watershed_boundary_mask: np.ndarray
    conflict_mask: np.ndarray
    smoothed_scores: np.ndarray
    candidates: list[HysteresisCandidate]
    summary: dict[str, Any] = field(default_factory=dict)


def smooth_scores_on_graph(
    scores: np.ndarray,
    neighbor_indices: list[np.ndarray],
    num_iters: int = 1,
    include_self: bool = True,
) -> np.ndarray:
    """Low-pass filter scores by averaging each point with graph neighbors."""
    smoothed = np.nan_to_num(
        np.asarray(scores, dtype=np.float64).reshape(-1).copy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    num_iters = int(num_iters)
    if num_iters <= 0:
        return smoothed

    num_points = smoothed.shape[0]
    if len(neighbor_indices) != num_points:
        raise ValueError("neighbor_indices length must match scores length.")

    for _ in range(num_iters):
        next_scores = np.empty(num_points, dtype=np.float64)
        for point_index, neighbors in enumerate(neighbor_indices):
            hood = [point_index] if include_self else []
            hood.extend(int(neighbor) for neighbor in np.asarray(neighbors, dtype=np.int64).reshape(-1))
            next_scores[point_index] = float(np.mean(smoothed[hood]))
        smoothed = next_scores
    return smoothed


def dilate_mask_hops(
    mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    hops: int,
) -> np.ndarray:
    """Expand a boolean mask outward along neighbor lists for ``hops`` steps."""
    hops = int(hops)
    expanded = np.asarray(mask, dtype=bool).reshape(-1).copy()
    if hops <= 0:
        return expanded
    if len(neighbor_indices) != expanded.shape[0]:
        raise ValueError("neighbor_indices length must match mask length.")

    for _ in range(hops):
        next_expanded = expanded.copy()
        for center in np.flatnonzero(expanded):
            for neighbor in np.asarray(neighbor_indices[int(center)], dtype=np.int64).reshape(-1):
                next_expanded[int(neighbor)] = True
        expanded = next_expanded
    return expanded


def _neighbor_indices_to_csr(neighbor_indices: list[np.ndarray]) -> sparse.csr_matrix:
    """Build a symmetric binary adjacency matrix from per-point neighbor lists."""
    num_points = len(neighbor_indices)
    rows: list[int] = []
    cols: list[int] = []
    for center, neighbors in enumerate(neighbor_indices):
        for neighbor in np.asarray(neighbors, dtype=np.int64).reshape(-1):
            rows.append(int(center))
            cols.append(int(neighbor))
    if not rows:
        return sparse.csr_matrix((num_points, num_points), dtype=np.float64)
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(num_points, num_points))
    return adjacency.maximum(adjacency.transpose()).tocsr()


def _compute_core_prominence(
    scores_smooth: np.ndarray,
    peak_index: int,
    core_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    prominence_hops: int,
) -> float:
    """Estimate peak prominence as peak score minus local background median."""
    peak_seed = np.zeros(scores_smooth.shape[0], dtype=bool)
    peak_seed[int(peak_index)] = True
    neighborhood = dilate_mask_hops(peak_seed, neighbor_indices, hops=int(prominence_hops))
    background_mask = neighborhood & ~core_mask
    background_scores = scores_smooth[background_mask]
    if background_scores.size == 0:
        local_background = float(np.median(scores_smooth))
    else:
        local_background = float(np.median(background_scores))
    return float(scores_smooth[int(peak_index)] - local_background)


def extract_strong_cores(
    scores_smooth: np.ndarray,
    neighbor_indices: list[np.ndarray],
    strong_percentile: float,
    core_min_size: int,
    prominence_hops: int,
    min_prominence: float,
    nms_hops: int,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Extract strong score cores with prominence filtering and peak NMS."""
    scores_smooth = np.asarray(scores_smooth, dtype=np.float64).reshape(-1)
    strong_threshold = float(np.percentile(scores_smooth, strong_percentile))
    strong_mask = scores_smooth >= strong_threshold

    topology = _neighbor_indices_to_csr(neighbor_indices)
    labels, num_components = label_connected_components(strong_mask, topology)

    raw_cores: list[dict[str, Any]] = []
    rejected_small = 0
    rejected_prominence = 0

    for component_id in range(1, num_components + 1):
        core_mask = labels == component_id
        core_size = int(np.sum(core_mask))
        if core_size < int(core_min_size):
            rejected_small += 1
            continue

        component_indices = np.flatnonzero(core_mask)
        peak_index = int(component_indices[np.argmax(scores_smooth[component_indices])])
        peak_score = float(scores_smooth[peak_index])
        prominence = _compute_core_prominence(
            scores_smooth,
            peak_index,
            core_mask,
            neighbor_indices,
            prominence_hops,
        )
        if prominence < float(min_prominence):
            rejected_prominence += 1
            continue

        raw_cores.append(
            {
                "label": len(raw_cores) + 1,
                "core_mask": core_mask.copy(),
                "peak_index": peak_index,
                "peak_score": peak_score,
                "core_size": core_size,
                "prominence": prominence,
            }
        )

    raw_cores.sort(key=lambda item: item["peak_score"], reverse=True)
    kept_cores: list[dict[str, Any]] = []
    nms_suppressed = 0
    kept_peak_regions = np.zeros(scores_smooth.shape[0], dtype=bool)

    for core in raw_cores:
        peak_index = int(core["peak_index"])
        peak_seed = np.zeros(scores_smooth.shape[0], dtype=bool)
        peak_seed[peak_index] = True
        peak_neighborhood = dilate_mask_hops(peak_seed, neighbor_indices, hops=int(nms_hops))
        if np.any(kept_peak_regions & peak_neighborhood):
            nms_suppressed += 1
            continue
        kept_peak_regions |= peak_neighborhood
        core["label"] = len(kept_cores) + 1
        kept_cores.append(core)

    final_core_mask = np.zeros(scores_smooth.shape[0], dtype=bool)
    for core in kept_cores:
        final_core_mask |= np.asarray(core["core_mask"], dtype=bool)

    summary = {
        "strong_threshold": strong_threshold,
        "num_strong_points": int(np.sum(strong_mask)),
        "num_strong_components_before_filter": int(num_components),
        "num_cores_after_filter": len(kept_cores),
        "rejected_small_cores": int(rejected_small),
        "rejected_low_prominence_cores": int(rejected_prominence),
        "nms_suppressed_cores": int(nms_suppressed),
    }
    return kept_cores, final_core_mask, summary


def _multi_source_hysteresis_growth(
    scores_smooth: np.ndarray,
    weak_mask: np.ndarray,
    cores: list[dict[str, Any]],
    neighbor_indices: list[np.ndarray],
    conflict_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grow multiple core labels inside weak_mask with watershed conflict handling."""
    num_points = scores_smooth.shape[0]
    labels = np.zeros(num_points, dtype=np.int32)
    conflict_mask = np.zeros(num_points, dtype=bool)
    core_peak_scores = {int(core["label"]): float(core["peak_score"]) for core in cores}

    for core in cores:
        labels[np.asarray(core["core_mask"], dtype=bool)] = int(core["label"])

    heap: list[tuple[float, int]] = []
    for core in cores:
        for index in np.flatnonzero(core["core_mask"]):
            heapq.heappush(heap, (-float(scores_smooth[int(index)]), int(index)))

    conflict_policy = conflict_policy.strip().lower()
    while heap:
        _neg_score, center = heapq.heappop(heap)
        source_label = int(labels[center])
        if source_label <= 0:
            continue

        for neighbor in np.asarray(neighbor_indices[center], dtype=np.int64).reshape(-1):
            neighbor = int(neighbor)
            if not weak_mask[neighbor]:
                continue

            neighbor_label = int(labels[neighbor])
            if neighbor_label == 0:
                labels[neighbor] = source_label
                heapq.heappush(heap, (-float(scores_smooth[neighbor]), neighbor))
                continue

            if neighbor_label == source_label:
                continue

            if conflict_policy == "higher_score":
                if core_peak_scores.get(source_label, 0.0) > core_peak_scores.get(neighbor_label, 0.0):
                    labels[neighbor] = source_label
                    heapq.heappush(heap, (-float(scores_smooth[neighbor]), neighbor))
                else:
                    conflict_mask[neighbor] = True
            else:
                conflict_mask[neighbor] = True
                conflict_mask[center] = True

    assigned_mask = labels > 0
    watershed_boundary_mask = conflict_mask.copy()
    return labels, assigned_mask, watershed_boundary_mask


def _fallback_weak_components(
    scores_smooth: np.ndarray,
    weak_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    min_component_size: int,
) -> list[HysteresisCandidate]:
    """Fallback to connected components on weak_mask when no strong cores survive."""
    topology = _neighbor_indices_to_csr(neighbor_indices)
    labels, num_components = label_connected_components(weak_mask, topology)
    candidates: list[HysteresisCandidate] = []

    for component_id in range(1, num_components + 1):
        candidate_mask = labels == component_id
        candidate_size = int(np.sum(candidate_mask))
        if candidate_size < int(min_component_size):
            continue
        component_indices = np.flatnonzero(candidate_mask)
        peak_index = int(component_indices[np.argmax(scores_smooth[component_indices])])
        candidates.append(
            HysteresisCandidate(
                label=len(candidates) + 1,
                core_mask=candidate_mask.copy(),
                candidate_mask=candidate_mask.copy(),
                peak_index=peak_index,
                peak_score=float(scores_smooth[peak_index]),
                core_size=candidate_size,
                candidate_size=candidate_size,
                growth_ratio=1.0,
                prominence=0.0,
                accepted=True,
                rejection_reason=None,
                growth_too_large=False,
            )
        )
    return candidates


def build_hysteresis_peak_candidates(
    scores: np.ndarray,
    neighbor_indices: list[np.ndarray],
    strong_percentile: float = 95.0,
    weak_percentile: float = 85.0,
    score_smoothing_iters: int = 1,
    core_min_size: int = 3,
    core_prominence_hops: int = 3,
    core_min_prominence: float = 0.0,
    peak_nms_hops: int = 3,
    conflict_policy: str = "boundary",
    min_component_size: int = 10,
    max_growth_ratio: float = 20.0,
    fallback_to_single_threshold: bool = True,
) -> HysteresisCandidateResult:
    """Build peak-guided hysteresis candidates with watershed conflict boundaries."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(neighbor_indices) != scores.shape[0]:
        raise ValueError("neighbor_indices length must match scores length.")

    summary: dict[str, Any] = {
        "fallback_used": False,
        "conflict_policy": conflict_policy,
        "strong_percentile": float(strong_percentile),
        "weak_percentile": float(weak_percentile),
        "score_smoothing_iters": int(score_smoothing_iters),
        "percentile_auto_corrected": False,
    }

    if float(strong_percentile) <= float(weak_percentile):
        strong_percentile = float(weak_percentile) + 1.0
        summary["percentile_auto_corrected"] = True
        summary["strong_percentile_corrected"] = float(strong_percentile)

    scores_smooth = smooth_scores_on_graph(
        scores,
        neighbor_indices,
        num_iters=score_smoothing_iters,
    )
    weak_threshold = float(np.percentile(scores_smooth, weak_percentile))
    weak_mask = scores_smooth >= weak_threshold
    summary["weak_threshold"] = weak_threshold
    summary["num_weak_seed_points"] = int(np.sum(weak_mask))

    cores, core_mask, core_summary = extract_strong_cores(
        scores_smooth=scores_smooth,
        neighbor_indices=neighbor_indices,
        strong_percentile=strong_percentile,
        core_min_size=core_min_size,
        prominence_hops=core_prominence_hops,
        min_prominence=core_min_prominence,
        nms_hops=peak_nms_hops,
    )
    summary.update(core_summary)
    summary["num_strong_core_points"] = int(np.sum(core_mask))

    candidates: list[HysteresisCandidate] = []
    labels = np.zeros(scores.shape[0], dtype=np.int32)
    assigned_mask = np.zeros(scores.shape[0], dtype=bool)
    watershed_boundary_mask = np.zeros(scores.shape[0], dtype=bool)
    conflict_mask = np.zeros(scores.shape[0], dtype=bool)

    if not cores:
        if fallback_to_single_threshold:
            summary["fallback_used"] = True
            candidates = _fallback_weak_components(
                scores_smooth,
                weak_mask,
                neighbor_indices,
                min_component_size=min_component_size,
            )
        return HysteresisCandidateResult(
            candidate_masks=[candidate.candidate_mask.copy() for candidate in candidates if candidate.accepted],
            core_mask=core_mask,
            weak_mask=weak_mask,
            assigned_mask=assigned_mask,
            watershed_boundary_mask=watershed_boundary_mask,
            conflict_mask=conflict_mask,
            smoothed_scores=scores_smooth,
            candidates=candidates,
            summary=summary,
        )

    labels, assigned_mask, watershed_boundary_mask = _multi_source_hysteresis_growth(
        scores_smooth,
        weak_mask,
        cores,
        neighbor_indices,
        conflict_policy=conflict_policy,
    )
    conflict_mask = watershed_boundary_mask.copy()

    for core in cores:
        label = int(core["label"])
        candidate_mask = (labels == label) & ~watershed_boundary_mask
        candidate_size = int(np.sum(candidate_mask))
        core_size = int(core["core_size"])
        growth_ratio = float(candidate_size / max(core_size, 1))
        growth_too_large = growth_ratio > float(max_growth_ratio)

        accepted = candidate_size >= int(min_component_size) and candidate_size > 0
        rejection_reason: str | None = None
        if candidate_size == 0:
            accepted = False
            rejection_reason = "empty_candidate"
        elif candidate_size < int(min_component_size):
            accepted = False
            rejection_reason = "too_small_candidate"
        elif growth_too_large:
            rejection_reason = "growth_too_large"

        candidates.append(
            HysteresisCandidate(
                label=label,
                core_mask=np.asarray(core["core_mask"], dtype=bool).copy(),
                candidate_mask=candidate_mask,
                peak_index=int(core["peak_index"]),
                peak_score=float(core["peak_score"]),
                core_size=core_size,
                candidate_size=candidate_size,
                growth_ratio=growth_ratio,
                prominence=float(core["prominence"]),
                accepted=accepted,
                rejection_reason=rejection_reason,
                growth_too_large=growth_too_large,
            )
        )

    accepted_candidates = [candidate for candidate in candidates if candidate.accepted]
    summary["num_hysteresis_candidates"] = len(accepted_candidates)
    summary["num_hysteresis_candidates_total"] = len(candidates)
    summary["num_watershed_boundary_points"] = int(np.sum(watershed_boundary_mask))

    return HysteresisCandidateResult(
        candidate_masks=[candidate.candidate_mask.copy() for candidate in accepted_candidates],
        core_mask=core_mask,
        weak_mask=weak_mask,
        assigned_mask=assigned_mask,
        watershed_boundary_mask=watershed_boundary_mask,
        conflict_mask=conflict_mask,
        smoothed_scores=scores_smooth,
        candidates=candidates,
        summary=summary,
    )
