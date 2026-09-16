"""Pure helpers for the fixed absolute-threshold verification experiment."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


MODE = "fixed_absolute_threshold_verification"
DEVELOPMENT_GROUPS: tuple[tuple[str, str], ...] = (
    ("fish", "275_bulge"),
    ("fish", "309_sink"),
    ("diamond", "427_bulge"),
    ("diamond", "470_bulge"),
)
DEVELOPMENT_VARIANTS: tuple[tuple[str, str, str], ...] = tuple(
    (category, base_sample_id, sample_id)
    for category, base_sample_id in DEVELOPMENT_GROUPS
    for sample_id in (base_sample_id, f"{base_sample_id}_cut")
)
METRIC_FIELDS = ("precision", "recall", "f1", "iou")


def validate_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tau_abs must be finite and non-negative.")
    return threshold


def exact_breakpoint_grid(component_d90_values: Sequence[float]) -> list[float]:
    """Return 0 plus each finite development D90 crossed toward +infinity."""

    finite = sorted({float(value) for value in component_d90_values if math.isfinite(float(value))})
    grid = {0.0}
    grid.update(float(np.nextafter(value, np.inf)) for value in finite)
    return sorted(grid)


def replay_fixed_absolute_final_mask(
    final_labels_at_zero: np.ndarray,
    component_reports: Sequence[Mapping[str, Any]],
    threshold: float,
) -> np.ndarray:
    """Replay a threshold from a tau=0 final mask without rerunning restoration.

    The tau=0 labels have already passed minimum-size, large-shallow, stress,
    point refinement, boundary handling, and final cleanup.  Raising tau can
    therefore only remove whole component IDs whose D90 lies below tau.
    """

    tau = validate_threshold(threshold)
    labels = np.asarray(final_labels_at_zero, dtype=np.int32).reshape(-1)
    retained_ids = {
        int(report["component_id"])
        for report in component_reports
        if not bool(report.get("too_small", False))
        and not bool(report.get("large_shallow_triggered", False))
        and not bool(report.get("stress_veto", False))
        and float(report["D_p90"]) >= tau
    }
    if not retained_ids:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, np.fromiter(sorted(retained_ids), dtype=np.int32))


def aggregate_development_sweep(
    sample_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate variants -> physical groups -> categories with equal weights."""

    by_threshold_group: dict[tuple[float, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        key = (float(row["threshold"]), str(row["category"]), str(row["group_key"]))
        by_threshold_group[key].append(row)

    group_rows: list[dict[str, Any]] = []
    for (threshold, category, group_key), rows in sorted(by_threshold_group.items()):
        if len(rows) != 2:
            raise AssertionError(
                f"Development group {category}/{group_key} has {len(rows)} variants; expected 2."
            )
        aggregate = {
            "threshold": threshold,
            "category": category,
            "group_key": group_key,
            "variant_count": len(rows),
        }
        aggregate.update(
            {
                metric: float(np.mean([float(row[metric]) for row in rows]))
                for metric in METRIC_FIELDS
            }
        )
        group_rows.append(aggregate)

    by_threshold_category: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in group_rows:
        by_threshold_category[(float(row["threshold"]), str(row["category"]))].append(row)
    category_rows: list[dict[str, Any]] = []
    for (threshold, category), rows in sorted(by_threshold_category.items()):
        if len(rows) != 2:
            raise AssertionError(
                f"Development category {category} has {len(rows)} physical groups; expected 2."
            )
        aggregate = {
            "threshold": threshold,
            "category": category,
            "group_count": len(rows),
        }
        aggregate.update(
            {
                metric: float(np.mean([float(row[metric]) for row in rows]))
                for metric in METRIC_FIELDS
            }
        )
        category_rows.append(aggregate)

    by_threshold: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in category_rows:
        by_threshold[float(row["threshold"])].append(row)
    objective_rows: list[dict[str, Any]] = []
    for threshold, rows in sorted(by_threshold.items()):
        if {str(row["category"]) for row in rows} != {"fish", "diamond"}:
            raise AssertionError(
                f"Threshold {threshold!r} does not contain exactly Fish and Diamond."
            )
        objective = {"threshold": threshold, "category_count": 2}
        objective.update(
            {
                f"category_balanced_{metric}": float(
                    np.mean([float(row[metric]) for row in rows])
                )
                for metric in METRIC_FIELDS
            }
        )
        objective_rows.append(objective)
    return group_rows, category_rows, objective_rows


def select_threshold(objective_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply F1, then IoU, then smaller-threshold deterministic selection."""

    if not objective_rows:
        raise ValueError("Cannot select a threshold from an empty sweep.")
    ranked = sorted(
        (dict(row) for row in objective_rows),
        key=lambda row: (
            -float(row["category_balanced_f1"]),
            -float(row["category_balanced_iou"]),
            float(row["threshold"]),
        ),
    )
    selected = dict(ranked[0])
    best_f1 = float(selected["category_balanced_f1"])
    f1_ties = [row for row in ranked if float(row["category_balanced_f1"]) == best_f1]
    best_iou = max(float(row["category_balanced_iou"]) for row in f1_ties)
    iou_ties = [row for row in f1_ties if float(row["category_balanced_iou"]) == best_iou]
    selected["selection_trace"] = {
        "primary": "maximum category-balanced development F1",
        "secondary": "maximum category-balanced development IoU",
        "tertiary": "smaller numerical threshold",
        "f1_tie_count": len(f1_ties),
        "f1_iou_tie_count": len(iou_ties),
        "tertiary_tie_break_used": len(iou_ties) > 1,
        "historical_anchor_used": False,
        "heldout_metrics_used": False,
    }
    return selected

