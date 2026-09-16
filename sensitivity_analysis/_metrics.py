"""Metric and stage-diagnostic extraction for sensitivity analysis v2."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from sensitivity_analysis._config import MAIN8


METRIC_NAMES = ("precision", "recall", "f1", "iou")
CONFUSION_NAMES = ("tp", "fp", "fn", "tn")

COMMON_SAMPLE_FIELDS = [
    "config_id",
    "is_production_default",
    "dataset",
    "category",
    "sample_id",
    "physical_group_key",
    "is_dev10",
    "sample_identity_hash",
    "status",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "iou",
    "predicted_positive_points",
    "gt_positive_points",
    "working_point_count",
    "inference_total_seconds",
    "worker_wall_seconds",
    "config_hash",
    "implementation_hash",
    "implementation_fingerprint_schema",
    "implementation_file_count",
    "run_fingerprint",
    "effective_parameters_json",
    "failure_type",
    "failure_message",
    "resumed",
]

STAGE_DIAGNOSTIC_FIELDS = [
    "config_id",
    "dataset",
    "category",
    "sample_id",
    "is_dev10",
    "graph_directed_edge_count",
    "mean_graph_degree",
    "va_hks_response_mean",
    "va_hks_response_p95",
    "raw_ac_point_count",
    "retained_closed_ac_point_count",
    "expanded_ac_point_count",
    "final_ac_suppression_point_count",
    "structurally_suppressed_point_count",
    "high_confidence_seed_point_count",
    "seed_point_count",
    "seed_component_count",
    "grown_u_point_count",
    "grown_u_component_count",
    "candidate_point_count",
    "candidate_component_count",
    "merged_y_point_count",
    "merged_y_component_count",
    "gtr_candidate_point_count",
    "gtr_closed_candidate_point_count",
    "gtr_closed_candidate_component_count",
    "local_normal_reference_valid_candidate_count",
    "relative_normal_rejected_candidate_count",
    "large_shallow_rejected_candidate_count",
    "stress_veto_rejected_candidate_count",
    "accepted_candidate_count",
    "final_positive_point_count",
    "unavailable_diagnostics",
]


def _positive_count(values: Any) -> int:
    return int(np.sum(np.asarray(values).reshape(-1) > 0))


def _component_count(values: Any) -> int:
    labels = np.asarray(values, dtype=np.int64).reshape(-1)
    return int(np.unique(labels[labels > 0]).size)


def complete_binary_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Validate production final-mask metrics and add the requested IoU."""

    result = dict(metrics)
    missing = [name for name in CONFUSION_NAMES if name not in result]
    if missing:
        raise RuntimeError(f"Production result lacks confusion metrics: {missing}")
    tp, fp, fn, tn = (int(result[name]) for name in CONFUSION_NAMES)
    if any(value < 0 for value in (tp, fp, fn, tn)):
        raise RuntimeError("Production confusion counts must be non-negative")
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    iou = float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0
    for name, expected in (
        ("precision", precision),
        ("recall", recall),
        ("f1", f1),
    ):
        if name not in result or not math.isclose(
            float(result[name]), expected, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(
                f"Production metric {name} is inconsistent with TP/FP/FN"
            )
    if not all(math.isfinite(value) for value in (precision, recall, f1, iou)):
        raise RuntimeError("Derived segmentation metrics must be finite")
    working_count = int(result.get("working_point_count", tp + fp + fn + tn))
    if tp + fp + fn + tn != working_count:
        raise RuntimeError("Confusion counts do not cover the working cloud")
    result.update(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "working_point_count": working_count,
        }
    )
    return result


def extract_stage_diagnostics(result: Any) -> dict[str, Any]:
    """Read only arrays and audit objects already returned by production."""

    degrees = np.asarray(
        [np.asarray(neighbors).size for neighbors in result.neighbor_indices],
        dtype=np.int64,
    )
    gtr = result.audit.get("gtr_summary")
    if not isinstance(gtr, Mapping):
        raise RuntimeError("Production result lacks the GTR summary")
    local_reference = gtr.get("local_normal_reference")
    classifier = gtr.get("candidate_classifier")
    if not isinstance(local_reference, Mapping):
        local_reference = {}
    if not isinstance(classifier, Mapping):
        classifier = {}

    va_response = np.asarray(result.va_hks_response, dtype=np.float64).reshape(-1)
    final_ac = np.asarray(result.final_ac_suppression_mask, dtype=bool).reshape(-1)
    hks_suppression = np.asarray(result.pure_hks_suppression_mask, dtype=bool).reshape(-1)
    q_labels = np.asarray(result.q_component_labels, dtype=np.int32).reshape(-1)
    u_labels = np.asarray(result.u_component_labels, dtype=np.int32).reshape(-1)
    y_labels = np.asarray(result.y_component_labels, dtype=np.int32).reshape(-1)

    # CovertSampleResult does not currently expose the distinct AC closed and
    # expanded masks.  Their absence is explicit rather than guessed from the
    # final suppression mask.
    unavailable = [
        "retained_closed_ac_point_count",
        "expanded_ac_point_count",
        "variation_aware_graph_weight_summary",
    ]
    return {
        "graph_directed_edge_count": int(np.sum(degrees)),
        "mean_graph_degree": float(np.mean(degrees)) if degrees.size else 0.0,
        "va_hks_response_mean": float(np.mean(va_response)) if va_response.size else 0.0,
        "va_hks_response_p95": (
            float(np.percentile(va_response, 95.0)) if va_response.size else 0.0
        ),
        "raw_ac_point_count": int(np.sum(result.raw_ac_mask)),
        "retained_closed_ac_point_count": None,
        "expanded_ac_point_count": None,
        "final_ac_suppression_point_count": int(np.sum(final_ac)),
        "structurally_suppressed_point_count": int(
            np.sum(final_ac | hks_suppression)
        ),
        "high_confidence_seed_point_count": int(
            np.sum(result.high_confidence_seed_mask)
        ),
        "seed_point_count": _positive_count(q_labels),
        "seed_component_count": _component_count(q_labels),
        "grown_u_point_count": _positive_count(u_labels),
        "grown_u_component_count": _component_count(u_labels),
        "candidate_point_count": _positive_count(y_labels),
        "candidate_component_count": _component_count(y_labels),
        "merged_y_point_count": _positive_count(y_labels),
        "merged_y_component_count": _component_count(y_labels),
        "gtr_candidate_point_count": int(np.sum(result.gtr_candidate_mask)),
        "gtr_closed_candidate_point_count": int(
            gtr.get("num_closed_points_after_post_expand", 0)
        ),
        "gtr_closed_candidate_component_count": int(
            gtr.get("num_closed_components", 0)
        ),
        "local_normal_reference_valid_candidate_count": int(
            local_reference.get("num_candidates_reference_valid", 0)
        ),
        "relative_normal_rejected_candidate_count": int(
            classifier.get("num_candidate_relative_normal_rejected", 0)
        ),
        "large_shallow_rejected_candidate_count": int(
            classifier.get("num_candidate_large_shallow_rejected", 0)
        ),
        "stress_veto_rejected_candidate_count": int(
            classifier.get("num_candidate_stress_veto_rejected", 0)
        ),
        "accepted_candidate_count": int(
            classifier.get("num_candidate_accepted", 0)
        ),
        "final_positive_point_count": int(np.sum(result.final_mask)),
        "unavailable_diagnostics": json.dumps(unavailable, separators=(",", ":")),
    }


def sample_fieldnames(parameter_columns: Sequence[str]) -> list[str]:
    return ["config_id", "is_production_default", *parameter_columns, *COMMON_SAMPLE_FIELDS[2:]]


def stage_fieldnames(parameter_columns: Sequence[str]) -> list[str]:
    return ["config_id", *parameter_columns, *STAGE_DIAGNOSTIC_FIELDS[1:]]


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(statistics.fmean(float(row[field]) for row in rows))


def _category_row(
    *,
    config_id: str,
    scope: str,
    category: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") != "success"]
    datasets = sorted({str(row.get("dataset") or "") for row in rows})
    if len(datasets) > 1:
        raise RuntimeError(f"Category {category} maps to multiple datasets: {datasets}")
    tp = sum(int(row["tp"]) for row in successful)
    fp = sum(int(row["fp"]) for row in successful)
    fn = sum(int(row["fn"]) for row in successful)
    tn = sum(int(row["tn"]) for row in successful)
    precision_micro = float(tp / (tp + fp)) if tp + fp else 0.0
    recall_micro = float(tp / (tp + fn)) if tp + fn else 0.0
    f1_micro = (
        float(2 * precision_micro * recall_micro / (precision_micro + recall_micro))
        if precision_micro + recall_micro
        else 0.0
    )
    iou_micro = float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0
    return {
        "config_id": config_id,
        "evaluation_scope": scope,
        "dataset": datasets[0] if datasets else "",
        "category": category,
        "sample_count": len(successful),
        "failed_sample_count": len(failed),
        "precision_macro": _mean(successful, "precision") if successful else None,
        "recall_macro": _mean(successful, "recall") if successful else None,
        "f1_macro": _mean(successful, "f1") if successful else None,
        "iou_macro": _mean(successful, "iou") if successful else None,
        "pooled_tp": tp,
        "pooled_fp": fp,
        "pooled_fn": fn,
        "pooled_tn": tn,
        "precision_micro_diagnostic": precision_micro,
        "recall_micro_diagnostic": recall_micro,
        "f1_micro_diagnostic": f1_micro,
        "iou_micro_diagnostic": iou_micro,
    }


CATEGORY_FIELDS = [
    "config_id",
    "evaluation_scope",
    "dataset",
    "category",
    "sample_count",
    "failed_sample_count",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "iou_macro",
    "pooled_tp",
    "pooled_fp",
    "pooled_fn",
    "pooled_tn",
    "precision_micro_diagnostic",
    "recall_micro_diagnostic",
    "f1_micro_diagnostic",
    "iou_micro_diagnostic",
]


def aggregate_configuration(
    *,
    config_id: str,
    rows: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate sample macro within category, then balance categories."""

    category_rows: list[dict[str, Any]] = []
    scoped_rows = {
        "all_main8": list(rows),
        "heldout_excluding_dev10": [
            row for row in rows if not bool(row.get("is_dev10"))
        ],
    }
    for scope, values in scoped_rows.items():
        for category in categories:
            category_rows.append(
                _category_row(
                    config_id=config_id,
                    scope=scope,
                    category=category,
                    rows=[row for row in values if row.get("category") == category],
                )
            )

    summary: dict[str, Any] = {
        "sample_count_all": sum(
            row.get("status") == "success" for row in scoped_rows["all_main8"]
        ),
        "sample_count_heldout": sum(
            row.get("status") == "success"
            for row in scoped_rows["heldout_excluding_dev10"]
        ),
        "failed_sample_count": sum(row.get("status") != "success" for row in rows),
    }
    canonical_main8 = tuple(categories) == MAIN8
    for scope, prefix in (
        ("all_main8", "all_main8"),
        ("heldout_excluding_dev10", "heldout"),
    ):
        selected = [
            row
            for row in category_rows
            if row["evaluation_scope"] == scope and int(row["sample_count"]) > 0
        ]
        complete_categories = len(selected) == len(categories)
        for metric in METRIC_NAMES:
            key = f"{prefix}_{metric}_cb"
            if canonical_main8 and complete_categories:
                summary[key] = float(
                    statistics.fmean(float(row[f"{metric}_macro"]) for row in selected)
                )
            else:
                summary[key] = None
    return category_rows, summary


SUMMARY_METRIC_FIELDS = [
    "all_main8_precision_cb",
    "all_main8_recall_cb",
    "all_main8_f1_cb",
    "all_main8_iou_cb",
    "heldout_precision_cb",
    "heldout_recall_cb",
    "heldout_f1_cb",
    "heldout_iou_cb",
    "sample_count_all",
    "sample_count_heldout",
    "failed_sample_count",
    "runtime_seconds",
]


__all__ = [
    "CATEGORY_FIELDS",
    "COMMON_SAMPLE_FIELDS",
    "CONFUSION_NAMES",
    "METRIC_NAMES",
    "STAGE_DIAGNOSTIC_FIELDS",
    "SUMMARY_METRIC_FIELDS",
    "aggregate_configuration",
    "complete_binary_metrics",
    "extract_stage_diagnostics",
    "sample_fieldnames",
    "stage_fieldnames",
]
