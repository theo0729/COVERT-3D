"""ABLATION-LOCAL COVERT pipeline adapters.

DO NOT USE FOR PRODUCTION INFERENCE.

This module deliberately reuses the frozen production numerical primitives and
changes only the data passed across one named conceptual boundary.  Production
source files are never edited.  The temporary frontend function substitution is
process-local, restored in ``finally``, and batch workers each use a separate
process.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Callable, Iterator

import numpy as np

from ablations.ablation_modes import get_ablation_spec, validate_semantic_diagnostics
from vast.covert import _backbone, _frontend, _gtr, _pipeline, _score_compat


_PRODUCTION_BUILD_MAINLINE_CONFIDENCE = _frontend.build_mainline_confidence
_PRODUCTION_CLASSIFY_GTR_COMPONENTS = _gtr._classify_gtr_components
_RELATIVE_NORMAL_DISABLED_SENTINEL = -1.0
_FIXED_ABSOLUTE_MODE = "fixed_absolute_threshold_verification"


def _validate_absolute_threshold(value: float | None) -> float:
    if value is None:
        raise ValueError(
            "fixed_absolute_threshold_verification requires an explicit "
            "absolute_threshold."
        )
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("absolute_threshold must be finite and non-negative.")
    return threshold


def build_effective_gtr_config(
    config: Any,
    ablation_mode: str,
    *,
    absolute_threshold: float | None = None,
) -> Any:
    """Return an immutable ablation-local config without mutating production input."""

    mode = get_ablation_spec(ablation_mode).mode
    if mode == "laplacian_restoration":
        return replace(config, gtr=replace(config.gtr, restoration_mode="laplacian"))
    if mode == "wo_local_control_calibration":
        return replace(
            config,
            gtr=replace(
                config.gtr,
                adaptive_relative_normal_max_ratio=_RELATIVE_NORMAL_DISABLED_SENTINEL,
            ),
        )
    if mode == _FIXED_ABSOLUTE_MODE:
        # Keep the adaptive route underneath so object scale and Local Normal
        # Reference are computed exactly as in Full.  A scoped classifier hook
        # replaces only the normal-like Boolean decision.
        return replace(
            config,
            gtr=replace(
                config.gtr,
                displacement_threshold=_validate_absolute_threshold(
                    absolute_threshold
                ),
            ),
        )
    return config


def _classify_fixed_absolute_components(
    closed_labels: np.ndarray,
    seed_labels: np.ndarray,
    gtr_scores: np.ndarray,
    displacement_magnitude: np.ndarray,
    stress_proxy: np.ndarray,
    min_closed_component_size: int,
    displacement_threshold: float,
    use_stress_veto: bool,
    stress_threshold: float,
    *,
    classifier_mode: str,
    local_normal_reference_result: dict[str, Any] | None,
    object_scale: float | None,
    adaptive_relative_normal_max_ratio: float,
    adaptive_min_large_fraction: float,
    adaptive_max_shallow_displacement_normalized: float,
) -> dict[str, Any]:
    """Replace only relative-normal rejection with strict ``D_p90 < tau``.

    The production adaptive classifier is first evaluated to obtain exactly the
    same component statistics, Local Normal diagnostics, and large-shallow
    predicate.  Its decision is then rebuilt without consuming either local
    candidate/reference ratio.
    """

    threshold = _validate_absolute_threshold(displacement_threshold)
    diagnostic = _PRODUCTION_CLASSIFY_GTR_COMPONENTS(
        closed_labels=closed_labels,
        seed_labels=seed_labels,
        gtr_scores=gtr_scores,
        displacement_magnitude=displacement_magnitude,
        stress_proxy=stress_proxy,
        min_closed_component_size=min_closed_component_size,
        displacement_threshold=threshold,
        use_stress_veto=use_stress_veto,
        stress_threshold=stress_threshold,
        classifier_mode="adaptive_normal_rejection",
        local_normal_reference_result=local_normal_reference_result,
        object_scale=object_scale,
        adaptive_relative_normal_max_ratio=adaptive_relative_normal_max_ratio,
        adaptive_min_large_fraction=adaptive_min_large_fraction,
        adaptive_max_shallow_displacement_normalized=(
            adaptive_max_shallow_displacement_normalized
        ),
    )
    labels = np.asarray(closed_labels, dtype=np.int32).reshape(-1)
    final_labels = np.zeros_like(labels)
    rejected_labels = np.zeros_like(labels)
    reports: list[dict[str, Any]] = []
    for raw_report in diagnostic["component_reports"]:
        report = dict(raw_report)
        component_id = int(report["component_id"])
        too_small = bool(report["too_small"])
        absolute_triggered = bool(float(report["D_p90"]) < threshold)
        large_shallow_triggered = bool(report["large_shallow_triggered"])
        stress_triggered = bool(
            use_stress_veto and float(report["stress_p90_report"]) > stress_threshold
        )
        accepted = True
        rejection_reason: str | None = None
        if too_small:
            accepted = False
            rejection_reason = "too_small"
        elif absolute_triggered and large_shallow_triggered:
            accepted = False
            rejection_reason = "absolute_normal_and_large_shallow"
        elif absolute_triggered:
            accepted = False
            rejection_reason = "absolute_normal"
        elif large_shallow_triggered:
            accepted = False
            rejection_reason = "large_shallow"
        elif stress_triggered:
            accepted = False
            rejection_reason = "stress_veto"
        report.update(
            {
                "classifier_mode": _FIXED_ABSOLUTE_MODE,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "absolute_statistic": "candidate_D_p90",
                "absolute_threshold": threshold,
                "absolute_keep_equal": True,
                "absolute_normal_triggered": absolute_triggered,
                "relative_normal_consumed": False,
                "large_shallow_consumed": True,
                "stress_veto": stress_triggered,
            }
        )
        component_mask = labels == component_id
        if accepted:
            final_labels[component_mask] = component_id
        else:
            rejected_labels[component_mask] = component_id
        reports.append(report)

    rejected_reasons = [
        str(report["rejection_reason"])
        for report in reports
        if not bool(report["accepted"])
    ]
    accepted_count = sum(bool(report["accepted"]) for report in reports)
    return {
        "classifier_mode": _FIXED_ABSOLUTE_MODE,
        "final_defect_labels": final_labels,
        "rejected_labels": rejected_labels,
        "final_defect_mask": final_labels > 0,
        "rejected_mask": rejected_labels > 0,
        "component_reports": reports,
        "num_accepted_components": int(accepted_count),
        "num_rejected_components": int(len(reports) - accepted_count),
        "num_candidate_too_small_rejected": int(
            sum(reason == "too_small" for reason in rejected_reasons)
        ),
        "num_candidate_relative_normal_rejected": 0,
        "num_candidate_large_shallow_rejected": int(
            sum(reason == "large_shallow" for reason in rejected_reasons)
        ),
        "num_candidate_both_adaptive_rejected": 0,
        "num_candidate_stress_veto_rejected": int(
            sum(reason == "stress_veto" for reason in rejected_reasons)
        ),
        "num_candidate_legacy_low_displacement_rejected": 0,
        "num_candidate_absolute_normal_rejected": int(
            sum(reason == "absolute_normal" for reason in rejected_reasons)
        ),
        "num_candidate_absolute_and_large_shallow_rejected": int(
            sum(
                reason == "absolute_normal_and_large_shallow"
                for reason in rejected_reasons
            )
        ),
        "num_candidate_accepted": int(accepted_count),
    }


@contextmanager
def _gtr_component_classifier_hook(mode: str) -> Iterator[None]:
    if mode != _FIXED_ABSOLUTE_MODE:
        yield
        return
    previous = _gtr._classify_gtr_components
    _gtr._classify_gtr_components = _classify_fixed_absolute_components
    try:
        yield
    finally:
        _gtr._classify_gtr_components = previous


def _mark_fixed_absolute_semantics(
    gtr_result: dict[str, Any], *, threshold: float
) -> None:
    """Correct generic adaptive-route labels to the effective ablation strategy."""

    local_reference = gtr_result["local_normal_reference_result"]
    local_reference["consumed_by_candidate_classifier"] = False
    local_reference["diagnostic_only"] = True
    classification = gtr_result["candidate_classifier_result"]
    classification["classifier_mode"] = _FIXED_ABSOLUTE_MODE
    gtr_result["gtr_candidate_classifier_mode"] = _FIXED_ABSOLUTE_MODE
    summary = gtr_result["summary"]
    summary["gtr_candidate_classifier_mode"] = _FIXED_ABSOLUTE_MODE
    summary["resolved_gtr_params"][
        "candidate_classifier_mode"
    ] = _FIXED_ABSOLUTE_MODE
    summary["local_normal_reference"].update(
        {
            "diagnostic_only": True,
            "consumed_by_candidate_classifier": False,
        }
    )
    summary["candidate_classifier"].update(
        {
            "mode": _FIXED_ABSOLUTE_MODE,
            "strategy": _FIXED_ABSOLUTE_MODE,
            "absolute_statistic": "candidate_D_p90",
            "absolute_threshold": float(threshold),
            "absolute_rule": "D_p90 < tau_abs",
            "absolute_keep_equal": True,
            "relative_normal_decision_consumed": False,
            "large_shallow_rejection_enabled": True,
            "num_candidate_absolute_normal_rejected": int(
                classification["num_candidate_absolute_normal_rejected"]
            ),
            "num_candidate_absolute_and_large_shallow_rejected": int(
                classification[
                    "num_candidate_absolute_and_large_shallow_rejected"
                ]
            ),
        }
    )


def _replace_semantic_evidence(
    result: dict[str, Any],
    *,
    anomaly_support: np.ndarray,
    ac_seed_removal_mask: np.ndarray,
    config: Any,
    source: str,
) -> dict[str, Any]:
    """Recompute only the downstream seed-confidence chain."""

    conf = config.confidence
    support = np.asarray(anomaly_support, dtype=np.float64).reshape(-1)
    original = np.asarray(result["original_sv_gated"], dtype=np.float64).reshape(-1)
    ac_mask = np.asarray(ac_seed_removal_mask, dtype=bool).reshape(-1)
    if not support.shape == original.shape == ac_mask.shape:
        raise ValueError("Ablation confidence vectors must have identical shapes.")

    semantic_threshold = _frontend.compute_nonzero_response_percentile_threshold(
        support, percentile=float(conf.semantic_gate_nonzero_percentile)
    )
    semantic_pass = np.asarray(support >= semantic_threshold, dtype=bool)
    semantic_gated = np.where(
        semantic_pass, support, float(conf.semantic_gate_replacement_value)
    ).astype(np.float64, copy=False)
    semantic_powered = np.power(
        semantic_gated, float(conf.semantic_confidence_exponent)
    ).astype(np.float64, copy=False)
    seed = (original * semantic_powered).astype(np.float64, copy=False)
    seed_norm, q01, raw_q99, effective_q99, fallback = (
        _frontend.normalize_response_q1_q99_with_nonzero_upper_fallback(
            seed,
            p_low=float(conf.seed_display_lower_percentile),
            p_high=float(conf.seed_display_upper_percentile),
        )
    )
    seed_before_ac = np.asarray(
        seed_norm >= float(conf.high_confidence_seed_threshold), dtype=bool
    )
    removed_by_ac = np.asarray(seed_before_ac & ac_mask, dtype=bool)
    seed_mask = np.asarray(seed_before_ac & ~removed_by_ac, dtype=bool)

    updated = dict(result)
    updated.update(
        {
            "fusion_both_masked_min": support,
            "fusion_va_hks_only_masked_min": support,
            "active_fusion_summary_key": source,
            "active_fusion_min": support,
            "semantic_confidence": support,
            "semantic_gate_threshold": semantic_threshold,
            "semantic_gate_pass_mask": semantic_pass,
            "semantic_confidence_gated": semantic_gated,
            "semantic_confidence_powered": semantic_powered,
            "seed_confidence": seed,
            "seed_confidence_norm": seed_norm,
            "seed_confidence_normalization_q01": q01,
            "seed_confidence_normalization_raw_q99": raw_q99,
            "seed_confidence_normalization_effective_q99": effective_q99,
            "seed_confidence_normalization_fallback_applied": fallback,
            "high_confidence_seed_mask_before_boundary_filter": seed_before_ac,
            "high_confidence_seed_boundary_removed_mask": removed_by_ac,
            "high_confidence_seed_mask": seed_mask,
        }
    )
    return updated


def _sv_only_confidence(**kwargs: Any) -> dict[str, Any]:
    """Production confidence pipeline with SV-only anomaly support."""

    result = _PRODUCTION_BUILD_MAINLINE_CONFIDENCE(**kwargs)
    return _replace_semantic_evidence(
        result,
        anomaly_support=np.asarray(result["sv_suppressed_minmax"], dtype=np.float64),
        ac_seed_removal_mask=np.asarray(
            kwargs["suppression"]["ac_suppression_mask"], dtype=bool
        ),
        config=kwargs["config"],
        source="sv_only",
    )


def _without_structural_suppression_confidence(**kwargs: Any) -> dict[str, Any]:
    """Run unchanged SV-VA consensus with all front-end suppression masks empty."""

    suppression = dict(kwargs["suppression"])
    num_points = int(np.asarray(kwargs["sv_response"]).size)
    zeros = np.zeros(num_points, dtype=bool)
    suppression.update(
        {
            "total_suppression_mask": zeros.copy(),
            "ac_suppression_mask": zeros.copy(),
            "pure_hks_suppression_mask": zeros.copy(),
        }
    )
    forwarded = dict(kwargs)
    forwarded["suppression"] = suppression
    return _PRODUCTION_BUILD_MAINLINE_CONFIDENCE(**forwarded)


@contextmanager
def _frontend_confidence_hook(mode: str) -> Iterator[None]:
    replacement: Callable[..., dict[str, Any]] | None = None
    if mode == "wo_va_hks":
        replacement = _sv_only_confidence
    elif mode == "wo_structural_suppression":
        replacement = _without_structural_suppression_confidence
    if replacement is None:
        yield
        return
    previous = _frontend.build_mainline_confidence
    _frontend.build_mainline_confidence = replacement
    try:
        yield
    finally:
        _frontend.build_mainline_confidence = previous


def _component_records(labels: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    return [
        np.flatnonzero(values == component_id).astype(np.int64, copy=False)
        for component_id in sorted(int(v) for v in np.unique(values) if int(v) > 0)
    ]


def _seed_only_frontend(frontend: dict[str, Any]) -> dict[str, Any]:
    """Replace growth-derived candidate fields with cleaned Q seed components."""

    result = dict(frontend)
    labels = np.asarray(frontend["seed_size_filtered_component_labels"], dtype=np.int32)
    components = _component_records(labels)
    mask = labels > 0
    identity_groups = [
        {
            "merged_component_id": component_id,
            "source_seed_component_ids": [component_id],
        }
        for component_id in range(1, len(components) + 1)
    ]
    result.update(
        {
            "seed_grown_components": components,
            "seed_grown_component_labels": labels.copy(),
            "seed_grown_mask": mask.copy(),
            "seed_label_aware_merged_components": components,
            "seed_label_aware_merged_component_labels": labels.copy(),
            "seed_label_aware_merged_mask": mask.copy(),
            "seed_label_aware_merge_summary": {
                "enabled": False,
                "method": "ablation_identity_cleaned_seed_components",
                "merged_source_groups": identity_groups,
                "input_component_count": len(components),
                "output_component_count": len(components),
            },
            "seed_grow_reports": [
                {
                    "source_seed_component_id": component_id,
                    "size": int(component.size),
                    "growth_bypassed": True,
                }
                for component_id, component in enumerate(components, start=1)
            ],
        }
    )
    summaries = dict(frontend.get("summaries", {}))
    growth_summary = dict(summaries.get("seed_component_growth", {}))
    growth_summary.update(
        {
            "growth_mode": "disabled_ablation",
            "max_grow_hops": None,
            "grown_union_point_count": int(np.sum(mask)),
            "grown_union_added_point_count": 0,
            "label_aware_merge_applied": False,
            "ablation_candidate_source": "cleaned_seed_components",
        }
    )
    summaries["seed_component_growth"] = growth_summary
    result["summaries"] = summaries
    return result


def _disabled_component_rule(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    component_ids = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    return {
        "enabled": False,
        "classification_performed": False,
        "reason": "bypassed_with_growth_dependent_scale_consistency",
        "seed_reports": [],
        "y_reports": [],
        "normal_y_ids": [],
        "defect_y_ids": [],
        "unknown_y_ids": component_ids,
        "filtered_candidate_labels": labels.copy(),
        "filtered_candidate_mask": labels > 0,
        "overall": {
            "enabled": False,
            "original_candidate_point_count": int(np.sum(labels > 0)),
            "retained_candidate_point_count": int(np.sum(labels > 0)),
            "removed_candidate_point_count": 0,
        },
    }


def _disabled_local_normal_u_cleanup(num_points: int) -> dict[str, Any]:
    zeros = np.zeros(num_points, dtype=bool)
    return {
        "enabled": False,
        "effective_enabled": False,
        "mode": "final_mask_only",
        "reason": "bypassed_with_growth_attribution",
        "local_normal_u_cleanup_mask": zeros.copy(),
        "local_strong_normal_u_mask": zeros.copy(),
        "local_weak_normal_u_mask": zeros.copy(),
        "overall": {"enabled": False, "effective_enabled": False},
    }


def _run_pre_verification_gtr(
    *,
    points: np.ndarray,
    neighbor_indices: list[np.ndarray],
    candidate_labels: np.ndarray,
    adapter: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Return the exact restoration unknown support without running restoration."""

    params = _gtr.build_covert_gtr_params(config)
    closing = _gtr._run_covert_gtr_closing(
        points=points,
        neighbor_indices=neighbor_indices,
        sgcr_candidate_labels=np.asarray(candidate_labels, dtype=np.int32),
        tscb_result=adapter,
        exp06=_score_compat,
        gtr_params_override=params,
    )
    closed_labels = np.asarray(closing["closed_labels"], dtype=np.int32).reshape(-1)
    closed_mask = closed_labels > 0
    support_base = np.zeros(points.shape[0], dtype=bool)
    support = support_base.copy()
    support_expand_summary: dict[str, Any] = {
        "skipped": True,
        "reason": "boundary_support_disabled",
        "expand_hops": int(params["hks_open_boundary_support_expand_hops"]),
    }
    if bool(params["use_hks_open_boundary_as_restoration_support"]):
        support_base = _gtr._resolve_exp14_hks_open_boundary_support_mask(
            tscb_result=adapter,
            num_points=points.shape[0],
            use_expanded=bool(params["hks_open_boundary_support_use_expanded"]),
        )
        support, support_expand_summary = _gtr._expand_hks_open_boundary_support_mask(
            support_base,
            neighbor_indices,
            expand_hops=int(params["hks_open_boundary_support_expand_hops"]),
        )
    if bool(params["exclude_open_boundary_support_from_unknown"]):
        pre_verification_mask = closed_mask & ~support
    else:
        pre_verification_mask = closed_mask.copy()
    final_labels = np.where(pre_verification_mask, closed_labels, 0).astype(
        np.int32, copy=False
    )
    return {
        **closing,
        "unknown_mask": pre_verification_mask,
        "restoration_unknown_mask": pre_verification_mask,
        "hks_open_boundary_support_mask_base": support_base,
        "hks_open_boundary_support_mask": support,
        "pre_verification_candidate_support": pre_verification_mask,
        "final_defect_mask": pre_verification_mask,
        "final_defect_labels": final_labels,
        "summary": {
            "method": "ablation_pre_verification_candidate_support",
            "restoration_mode": "not_executed",
            "restoration_executed": False,
            "counterfactual_decision_executed": False,
            "point_refinement_executed": False,
            "final_mask_source": "pre_verification_candidate_support",
            "num_closed_points": int(np.sum(closed_mask)),
            "num_boundary_support_points": int(np.sum(support)),
            "num_boundary_support_excluded_from_candidate": int(
                np.sum(closed_mask & support)
            ),
            "num_final_defect_points": int(np.sum(pre_verification_mask)),
            "closing": closing["summary"],
            "boundary_support_expansion": support_expand_summary,
            "resolved_gtr_params": params,
        },
    }


def _scale_thresholds(config: Any) -> dict[str, int]:
    fields = (
        "small_defect_max_multiple",
        "large_defect_min_multiple",
        "late_gtr_over_y_min_multiple",
        "late_gtr_over_y_min_numerator",
        "late_gtr_over_y_min_denominator",
        "late_w_over_y_max_numerator",
        "late_w_over_y_max_denominator",
        "trusted_z_over_q_min_multiple",
        "trusted_gtr_over_w_min_multiple",
        "trusted_y_over_u_max_numerator",
        "trusted_y_over_u_max_denominator",
        "trusted_gtr_over_u_min_multiple",
        "weak_y_over_u_min_multiple",
        "normal_mass_majority_numerator",
        "normal_mass_majority_denominator",
    )
    return {field: int(getattr(config.scale_consistency, field)) for field in fields}


def run_ablation_pipeline(
    raw_points: np.ndarray,
    config: Any,
    *,
    ablation_mode: str,
    query_workers: int = 1,
    sample_id: str = "sample",
    suppression_override_hook: Any | None = None,
    absolute_threshold: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run exactly one registered semantic variant."""

    spec = get_ablation_spec(ablation_mode)
    mode = spec.mode
    if mode == _FIXED_ABSOLUTE_MODE:
        resolved_absolute_threshold = _validate_absolute_threshold(
            absolute_threshold
        )
    else:
        if absolute_threshold is not None:
            raise ValueError(
                "absolute_threshold is valid only for "
                "fixed_absolute_threshold_verification."
            )
        resolved_absolute_threshold = None
    if mode == "full":
        result = _pipeline.run_covert_pipeline(
            raw_points,
            config,
            query_workers=query_workers,
            sample_id=sample_id,
            suppression_override_hook=suppression_override_hook,
            verbose=verbose,
        )
        diagnostics = {"production_delegate": True}
        validate_semantic_diagnostics(mode, diagnostics)
        result["ablation_diagnostics"] = diagnostics
        return result

    base = _backbone.run_covert_backbone(
        raw_points,
        config,
        query_workers=query_workers,
        sample_id=sample_id,
    )
    frontend_hook = (
        None if mode == "wo_structural_suppression" else suppression_override_hook
    )
    with _frontend_confidence_hook(mode):
        frontend = _frontend.run_covert_frontend(
            base,
            config,
            suppression_override_hook=frontend_hook,
            verbose=verbose,
        )

    if mode == "wo_seed_growth":
        frontend = _seed_only_frontend(frontend)
    adapter = _pipeline._build_exp21_gtr_adapter(frontend)
    points = np.asarray(frontend["points"], dtype=np.float64)
    neighbor_indices = adapter["neighbor_indices"]
    adjacency_dist = adapter["adjacency_dist"]
    original_labels = np.asarray(
        frontend["seed_label_aware_merged_component_labels"], dtype=np.int32
    ).reshape(-1)

    preview = None
    preview_labels = None
    if mode == "wo_seed_growth":
        component_rule = _disabled_component_rule(original_labels)
    else:
        component_rule_enabled = bool(
            frontend["seed_confidence_p80_component_extraction_enable"]
        )
        if component_rule_enabled:
            preview = _gtr._run_covert_gtr_closing(
                points=points,
                neighbor_indices=neighbor_indices,
                sgcr_candidate_labels=original_labels,
                tscb_result=adapter,
                exp06=_score_compat,
                gtr_params_override=_gtr.build_covert_gtr_params(config),
            )
            preview_labels = np.asarray(preview["closed_labels_core"], dtype=np.int32)
        component_rule = _pipeline._build_component_rule_result(
            frontend,
            enabled=component_rule_enabled,
            preview_closed_labels_core=preview_labels,
            thresholds=_scale_thresholds(config),
        )
    filtered_labels = np.asarray(
        component_rule["filtered_candidate_labels"], dtype=np.int32
    )

    if mode == "wo_seed_growth":
        cleanup = _disabled_local_normal_u_cleanup(points.shape[0])
    else:
        cleanup = _pipeline._build_local_normal_u_cleanup_request(
            frontend,
            component_rule,
            enabled=bool(config.switches.final_local_normal_u_cleanup),
            mode=str(config.gtr.local_normal_u_cleanup_mode),
        )

    effective_gtr_config = build_effective_gtr_config(
        config,
        mode,
        absolute_threshold=resolved_absolute_threshold,
    )

    with _gtr_component_classifier_hook(mode):
        if mode == "wo_counterfactual_verification":
            gtr_result = _run_pre_verification_gtr(
                points=points,
                neighbor_indices=neighbor_indices,
                candidate_labels=filtered_labels,
                adapter=adapter,
                config=config,
            )
        else:
            gtr_result = _gtr.run_covert_gtr(
                points=points,
                adjacency_dist=adjacency_dist,
                neighbor_indices=neighbor_indices,
                candidate_labels=filtered_labels,
                frontend_adapter=adapter,
                config=effective_gtr_config,
                score_compat=_score_compat,
                local_normal_u_cleanup_request=cleanup,
            )
    if mode == _FIXED_ABSOLUTE_MODE:
        assert resolved_absolute_threshold is not None
        _mark_fixed_absolute_semantics(
            gtr_result, threshold=resolved_absolute_threshold
        )

    diagnostics: dict[str, Any] = {
        "production_delegate": False,
        "base_production_config_unchanged": True,
        "gt_used_by_inference": False,
    }
    if mode == "laplacian_restoration":
        diagnostics.update(
            {
                "restoration_operator": "laplacian",
                "local_normal_restoration_operator": "laplacian",
                "same_operator_for_candidate_and_local_normal": True,
            }
        )
    elif mode == "wo_va_hks":
        diagnostics.update(
            {
                "seed_anomaly_source": "sv_only",
                "defect_veto_anomaly_source": "sv_only",
                "va_hks_participates_in_seed_consensus": False,
                "va_hks_participates_in_defect_veto": False,
                "va_hks_computed_but_unused_as_anomaly_evidence": True,
                "defect_veto_mechanism_retained": True,
            }
        )
    elif mode == "wo_structural_suppression":
        support_enabled = bool(
            gtr_result["summary"]["hks_open_boundary_restoration_support"]["enabled"]
        )
        diagnostics.update(
            {
                "frontend_structural_suppression": False,
                "suppression_specific_veto": False,
                "suppression_specific_seed_removal": False,
                "new_ac_computed": True,
                "gtr_new_ac_boundary_support": support_enabled,
            }
        )
    elif mode == "wo_seed_growth":
        diagnostics.update(
            {
                "descending_growth": False,
                "candidate_source": "cleaned_seed_components",
                "growth_dependent_scale_consistency": False,
                "growth_attributed_local_normal_u_cleanup": False,
            }
        )
    elif mode == "wo_counterfactual_verification":
        diagnostics.update(
            {
                "gtr_restoration_executed": False,
                "gtr_candidate_decision_executed": False,
                "gtr_point_refinement_executed": False,
                "final_mask_source": "pre_verification_candidate_support",
                "new_ac_boundary_support_excluded": bool(
                    config.gtr.exclude_boundary_support_from_unknown
                ),
            }
        )
    elif mode == "wo_local_control_calibration":
        summary = gtr_result["summary"]
        local_reference = summary["local_normal_reference"]
        classifier = summary["candidate_classifier"]
        relative_rejected = int(summary["num_candidate_relative_normal_rejected"])
        both_adaptive_rejected = int(summary["num_candidate_both_adaptive_rejected"])
        if relative_rejected != 0 or both_adaptive_rejected != 0:
            raise AssertionError(
                "Local-control calibration ablation unexpectedly rejected a candidate "
                f"by the disabled relative rule: relative={relative_rejected}, "
                f"both={both_adaptive_rejected}."
            )
        diagnostics.update(
            {
                "local_normal_reference_enabled": bool(local_reference["enabled"]),
                "local_normal_reference_computed": bool(
                    isinstance(gtr_result.get("local_normal_reference_result"), dict)
                    and local_reference["enabled"]
                ),
                "relative_normal_rejection_enabled": False,
                "relative_normal_threshold_sentinel": float(
                    classifier["relative_normal_max_ratio"]
                ),
                "large_shallow_rejection_retained": bool(
                    classifier["min_large_fraction"]
                    == config.gtr.adaptive_min_large_fraction
                    and classifier["max_shallow_d_norm"]
                    == config.gtr.adaptive_max_shallow_displacement_normalized
                ),
                "stress_veto_retained": bool(
                    classifier["use_stress_veto"] == config.gtr.stress_veto_enabled
                    and classifier["stress_threshold"] == config.gtr.stress_threshold
                ),
                "point_refinement_retained": bool(
                    summary["point_displacement_filter"]["enabled"]
                    == config.switches.gtr_point_refinement
                    and summary["displacement_stress_quadrants"][
                        "displacement_norm_threshold"
                    ]
                    == config.gtr.ds_displacement_normalized_threshold
                    and summary["displacement_stress_quadrants"][
                        "stress_norm_threshold"
                    ]
                    == config.gtr.ds_stress_normalized_threshold
                ),
                "local_normal_u_cleanup_retained": bool(
                    summary["local_normal_u_cleanup"]["enabled"]
                    == config.switches.final_local_normal_u_cleanup
                    and summary["local_normal_u_cleanup"]["mode"]
                    == config.gtr.local_normal_u_cleanup_mode
                ),
                "restoration_operator": str(summary["restoration_mode"]),
                "num_candidate_relative_normal_rejected": relative_rejected,
                "num_candidate_both_adaptive_rejected": both_adaptive_rejected,
            }
        )
    elif mode == _FIXED_ABSOLUTE_MODE:
        assert resolved_absolute_threshold is not None
        summary = gtr_result["summary"]
        local_reference = summary["local_normal_reference"]
        classifier = summary["candidate_classifier"]
        diagnostics.update(
            {
                "classifier_strategy": _FIXED_ABSOLUTE_MODE,
                "absolute_statistic": "candidate_D_p90",
                "absolute_threshold": float(resolved_absolute_threshold),
                "absolute_rule": "D_p90 < tau_abs",
                "absolute_keep_equal": True,
                "relative_normal_decision_consumed": False,
                "large_shallow_rejection_retained": bool(
                    classifier["large_shallow_rejection_enabled"]
                    and classifier["min_large_fraction"]
                    == config.gtr.adaptive_min_large_fraction
                    and classifier["max_shallow_d_norm"]
                    == config.gtr.adaptive_max_shallow_displacement_normalized
                ),
                "stress_veto_retained": bool(
                    classifier["use_stress_veto"]
                    == config.gtr.stress_veto_enabled
                    and classifier["stress_threshold"]
                    == config.gtr.stress_threshold
                ),
                "minimum_size_rejection_retained": bool(
                    summary["resolved_gtr_params"]["min_closed_component_size"]
                    == config.gtr.minimum_closed_component_size
                ),
                "point_refinement_retained": bool(
                    summary["point_displacement_filter"]["enabled"]
                    == config.switches.gtr_point_refinement
                    and summary["displacement_stress_quadrants"][
                        "displacement_norm_threshold"
                    ]
                    == config.gtr.ds_displacement_normalized_threshold
                    and summary["displacement_stress_quadrants"][
                        "stress_norm_threshold"
                    ]
                    == config.gtr.ds_stress_normalized_threshold
                ),
                "final_cleanup_retained": bool(
                    summary["local_normal_u_cleanup"]["enabled"]
                    == config.switches.final_local_normal_u_cleanup
                    and summary["local_normal_u_cleanup"]["mode"]
                    == config.gtr.local_normal_u_cleanup_mode
                ),
                "biharmonic_restoration_retained": bool(
                    summary["restoration_mode"] == "biharmonic"
                ),
                "local_control_computation_retained": bool(
                    local_reference["enabled"]
                    and isinstance(
                        gtr_result.get("local_normal_reference_result"), dict
                    )
                ),
                "local_control_diagnostic_only": bool(
                    local_reference["diagnostic_only"]
                    and not local_reference["consumed_by_candidate_classifier"]
                ),
                "num_candidate_absolute_normal_rejected": int(
                    classifier["num_candidate_absolute_normal_rejected"]
                ),
                "num_candidate_absolute_and_large_shallow_rejected": int(
                    classifier[
                        "num_candidate_absolute_and_large_shallow_rejected"
                    ]
                ),
            }
        )
    validate_semantic_diagnostics(mode, diagnostics)
    if np.asarray(gtr_result["final_defect_mask"]).shape != (points.shape[0],):
        raise AssertionError("Ablation final mask shape disagrees with the working cloud.")
    return {
        "sample_id": str(sample_id),
        "points": points,
        "exp21_result": frontend,
        "tscb_result": adapter,
        "sgcr_candidate_mask": original_labels > 0,
        "sgcr_candidate_labels": original_labels,
        "component_rule_result": component_rule,
        "component_rule_preview_result": preview,
        "component_rule_preview_closed_labels_core": preview_labels,
        "component_rule_filtered_candidate_labels": filtered_labels,
        "component_rule_filtered_candidate_mask": filtered_labels > 0,
        "component_rule_normal_y_ids": list(component_rule["normal_y_ids"]),
        "component_rule_defect_y_ids": list(component_rule["defect_y_ids"]),
        "component_rule_unknown_y_ids": list(component_rule["unknown_y_ids"]),
        "local_normal_u_cleanup_request": cleanup,
        "gtr_result": gtr_result,
        "ablation_diagnostics": diagnostics,
        "invariants": {
            "final_mask_shape_valid": True,
            "base_production_config_unchanged": True,
        },
    }
