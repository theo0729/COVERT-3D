"""Component-aware filtering from cross-path scale consistency evidence.

The functions in this module are deliberately experiment-agnostic.  Callers
provide authoritative integer point counts for the Q/U/Y/Z/W/GTR stages; this
module applies the fixed scale rules and performs semantic seed-mass voting.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


# Fixed integer thresholds. Comparisons are written as cross-products below so
# boundary decisions do not depend on floating-point rounding.
SMALL_DEFECT_MAX_MULTIPLE = 4
LARGE_DEFECT_MIN_MULTIPLE = 8
LATE_GTR_OVER_Y_MIN_MULTIPLE = 2
LATE_GTR_OVER_Y_MIN_NUMERATOR = 3
LATE_GTR_OVER_Y_MIN_DENOMINATOR = 2
LATE_W_OVER_Y_MAX_NUMERATOR = 1
LATE_W_OVER_Y_MAX_DENOMINATOR = 4
TRUSTED_Z_OVER_Q_MIN_MULTIPLE = 4
TRUSTED_GTR_OVER_W_MIN_MULTIPLE = 4
TRUSTED_Y_OVER_U_MAX_NUMERATOR = 3
TRUSTED_Y_OVER_U_MAX_DENOMINATOR = 2
TRUSTED_GTR_OVER_U_MIN_MULTIPLE = 4
WEAK_Y_OVER_U_MIN_MULTIPLE = 8
NORMAL_MASS_MAJORITY_NUMERATOR = 1
NORMAL_MASS_MAJORITY_DENOMINATOR = 2

SEED_STATE_DEFECT_ANCHOR = "defect_anchor"
SEED_STATE_STRONG_NORMAL = "strong_normal"
SEED_STATE_WEAK_NORMAL = "weak_normal"
SEED_STATE_UNKNOWN = "unknown"

Y_STATE_NORMAL = "normal"
Y_STATE_DEFECT = "defect"
Y_STATE_UNKNOWN = "unknown"


def _positive_count(value: Any, *, name: str) -> int:
    """Return one strictly positive integer point count."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer point count.")
    count = int(value)
    if count != value or count <= 0:
        raise ValueError(f"{name} must be a positive integer point count.")
    return count


def _positive_id(value: Any, *, name: str) -> int:
    """Return one strictly positive component ID."""
    return _positive_count(value, name=name)


def _ratio(numerator: int, denominator: int) -> float:
    """Compute an audit-only ratio after integer validation."""
    if denominator <= 0:
        raise ZeroDivisionError("Scale-ratio denominator must be positive.")
    return float(numerator) / float(denominator)


def classify_seed_component_evidence(
    *,
    q_id: int,
    y_id: int,
    q_size: int,
    u_size: int,
    y_size: int,
    z_size: int,
    w_size: int,
    gtr_size: int,
    view_z_component_id: int | None = None,
    view_w_component_id: int | None = None,
    thresholds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Classify one Q seed using fixed cross-path scale-consistency rules.

    ``w_size`` must be the independent morphology proposal size, rather than a
    display-label approximation. Defect evidence has priority over strong
    normal evidence, which has priority over weak normal evidence.
    """
    q_id = _positive_id(q_id, name="q_id")
    y_id = _positive_id(y_id, name="y_id")
    q_size = _positive_count(q_size, name="q_size")
    u_size = _positive_count(u_size, name="u_size")
    y_size = _positive_count(y_size, name="y_size")
    z_size = _positive_count(z_size, name="z_size")
    w_size = _positive_count(w_size, name="w_size")
    gtr_size = _positive_count(gtr_size, name="gtr_size")
    resolved = {
        "small_defect_max_multiple": SMALL_DEFECT_MAX_MULTIPLE,
        "large_defect_min_multiple": LARGE_DEFECT_MIN_MULTIPLE,
        "late_gtr_over_y_min_multiple": LATE_GTR_OVER_Y_MIN_MULTIPLE,
        "late_gtr_over_y_min_numerator": LATE_GTR_OVER_Y_MIN_NUMERATOR,
        "late_gtr_over_y_min_denominator": LATE_GTR_OVER_Y_MIN_DENOMINATOR,
        "late_w_over_y_max_numerator": LATE_W_OVER_Y_MAX_NUMERATOR,
        "late_w_over_y_max_denominator": LATE_W_OVER_Y_MAX_DENOMINATOR,
        "trusted_z_over_q_min_multiple": TRUSTED_Z_OVER_Q_MIN_MULTIPLE,
        "trusted_gtr_over_w_min_multiple": TRUSTED_GTR_OVER_W_MIN_MULTIPLE,
        "trusted_y_over_u_max_numerator": TRUSTED_Y_OVER_U_MAX_NUMERATOR,
        "trusted_y_over_u_max_denominator": TRUSTED_Y_OVER_U_MAX_DENOMINATOR,
        "trusted_gtr_over_u_min_multiple": TRUSTED_GTR_OVER_U_MIN_MULTIPLE,
        "weak_y_over_u_min_multiple": WEAK_Y_OVER_U_MIN_MULTIPLE,
        "normal_mass_majority_numerator": NORMAL_MASS_MAJORITY_NUMERATOR,
        "normal_mass_majority_denominator": NORMAL_MASS_MAJORITY_DENOMINATOR,
    }
    if thresholds is not None:
        unknown = sorted(set(thresholds) - set(resolved))
        if unknown:
            raise ValueError(f"Unknown component-filter threshold(s): {unknown}")
        resolved.update({key: int(value) for key, value in thresholds.items()})
    t = resolved

    is_small_defect_anchor = (
        w_size <= t["small_defect_max_multiple"] * z_size
        and y_size <= t["small_defect_max_multiple"] * q_size
    )
    is_large_defect_anchor = (
        w_size >= t["large_defect_min_multiple"] * z_size
        and u_size >= t["large_defect_min_multiple"] * q_size
        and y_size >= t["large_defect_min_multiple"] * q_size
        and gtr_size >= t["large_defect_min_multiple"] * q_size
        and z_size >= t["large_defect_min_multiple"] * q_size
    )
    is_defect_anchor = is_small_defect_anchor or is_large_defect_anchor

    late_condition_a = (
        gtr_size >= t["late_gtr_over_y_min_multiple"] * y_size
        and w_size <= u_size
    )
    late_condition_c = (
        t["late_gtr_over_y_min_denominator"] * gtr_size
        >= t["late_gtr_over_y_min_numerator"] * y_size
        and t["late_w_over_y_max_denominator"] * w_size
        <= t["late_w_over_y_max_numerator"] * y_size
    )
    late_inconsistency = late_condition_a or late_condition_c

    trusted_condition_n3 = (
        z_size >= t["trusted_z_over_q_min_multiple"] * q_size
        and gtr_size >= t["trusted_gtr_over_w_min_multiple"] * w_size
    )
    trusted_condition_n4 = (
        t["trusted_y_over_u_max_denominator"] * y_size
        <= t["trusted_y_over_u_max_numerator"] * u_size
        and gtr_size >= t["trusted_gtr_over_u_min_multiple"] * u_size
    )
    trusted_support_inconsistency = trusted_condition_n3 or trusted_condition_n4
    strong_normal_rule_matched = late_inconsistency or trusted_support_inconsistency

    weak_singleton = (
        q_size == 1
        and t["late_gtr_over_y_min_denominator"] * gtr_size
        >= t["late_gtr_over_y_min_numerator"] * y_size
    )
    weak_hitchhiker = (
        y_size >= t["weak_y_over_u_min_multiple"] * u_size
        and t["late_gtr_over_y_min_denominator"] * gtr_size
        >= t["late_gtr_over_y_min_numerator"] * w_size
    )
    weak_normal_rule_matched = weak_singleton or weak_hitchhiker

    # Mutually exclusive final flags enforce the requested evidence priority.
    # A defect anchor that also matches inconsistency contributes no normal vote.
    strong_normal = bool(strong_normal_rule_matched and not is_defect_anchor)
    weak_normal = bool(
        weak_normal_rule_matched and not is_defect_anchor and not strong_normal
    )
    if is_defect_anchor:
        seed_state = SEED_STATE_DEFECT_ANCHOR
    elif strong_normal:
        seed_state = SEED_STATE_STRONG_NORMAL
    elif weak_normal:
        seed_state = SEED_STATE_WEAK_NORMAL
    else:
        seed_state = SEED_STATE_UNKNOWN

    report: dict[str, Any] = {
        "q_id": q_id,
        "y_id": y_id,
        "q_size": q_size,
        "u_size": u_size,
        "y_size": y_size,
        "z_size": z_size,
        "w_size": w_size,
        "gtr_size": gtr_size,
        # Exp25-compatible basic coefficients.
        "m": _ratio(w_size, z_size),
        "u": _ratio(u_size, q_size),
        "y": _ratio(y_size, q_size),
        "z": _ratio(z_size, q_size),
        "g": _ratio(gtr_size, q_size),
        # Explicit cross-path ratios used for audit and ablation.
        "w_over_u": _ratio(w_size, u_size),
        "w_over_y": _ratio(w_size, y_size),
        "gtr_over_y": _ratio(gtr_size, y_size),
        "gtr_over_w": _ratio(gtr_size, w_size),
        "gtr_over_u": _ratio(gtr_size, u_size),
        "y_over_u": _ratio(y_size, u_size),
        "is_small_defect_anchor": bool(is_small_defect_anchor),
        "is_large_defect_anchor": bool(is_large_defect_anchor),
        "is_defect_anchor": bool(is_defect_anchor),
        "late_condition_a": bool(late_condition_a),
        "late_condition_c": bool(late_condition_c),
        "late_inconsistency": bool(late_inconsistency),
        "trusted_condition_n3": bool(trusted_condition_n3),
        "trusted_condition_n4": bool(trusted_condition_n4),
        "trusted_support_inconsistency": bool(trusted_support_inconsistency),
        "strong_normal_rule_matched": bool(strong_normal_rule_matched),
        "strong_normal": strong_normal,
        "weak_singleton": bool(weak_singleton),
        "weak_hitchhiker": bool(weak_hitchhiker),
        "weak_normal_rule_matched": bool(weak_normal_rule_matched),
        "weak_normal": weak_normal,
        "seed_state": seed_state,
        "final_seed_state": seed_state,
    }
    if view_z_component_id is not None:
        report["view_z_component_id"] = _positive_id(
            view_z_component_id,
            name="view_z_component_id",
        )
    if view_w_component_id is not None:
        report["view_w_component_id"] = _positive_id(
            view_w_component_id,
            name="view_w_component_id",
        )
    return report


def aggregate_seed_evidence_to_y(
    seed_reports: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate mutually exclusive seed evidence by Q-point mass for each Y."""
    reports = [dict(report) for report in seed_reports]
    normal_numerator = int(
        NORMAL_MASS_MAJORITY_NUMERATOR
        if thresholds is None
        else thresholds.get(
            "normal_mass_majority_numerator", NORMAL_MASS_MAJORITY_NUMERATOR
        )
    )
    normal_denominator = int(
        NORMAL_MASS_MAJORITY_DENOMINATOR
        if thresholds is None
        else thresholds.get(
            "normal_mass_majority_denominator", NORMAL_MASS_MAJORITY_DENOMINATOR
        )
    )
    by_y: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_q_ids: set[int] = set()
    for report in reports:
        q_id = _positive_id(report["q_id"], name="q_id")
        if q_id in seen_q_ids:
            raise ValueError(f"Duplicate q_id in seed reports: {q_id}.")
        seen_q_ids.add(q_id)
        y_id = _positive_id(report["y_id"], name="y_id")
        _positive_count(report["q_size"], name="q_size")
        if report.get("seed_state") not in {
            SEED_STATE_DEFECT_ANCHOR,
            SEED_STATE_STRONG_NORMAL,
            SEED_STATE_WEAK_NORMAL,
            SEED_STATE_UNKNOWN,
        }:
            raise ValueError(f"Invalid seed_state for Q{q_id}.")
        by_y[y_id].append(report)

    y_reports: list[dict[str, Any]] = []
    for y_id in sorted(by_y):
        members = sorted(by_y[y_id], key=lambda report: int(report["q_id"]))
        total_seed_mass = sum(int(report["q_size"]) for report in members)
        strong_normal_seed_mass = sum(
            int(report["q_size"])
            for report in members
            if report["seed_state"] == SEED_STATE_STRONG_NORMAL
        )
        contains_defect_anchor = any(
            report["seed_state"] == SEED_STATE_DEFECT_ANCHOR
            for report in members
        )
        is_normal = (
            normal_denominator * strong_normal_seed_mass
            > normal_numerator * total_seed_mass
        )
        if is_normal:
            y_state = Y_STATE_NORMAL
        elif contains_defect_anchor:
            y_state = Y_STATE_DEFECT
        else:
            y_state = Y_STATE_UNKNOWN
        y_reports.append(
            {
                "y_id": int(y_id),
                "constituent_q_ids": [int(report["q_id"]) for report in members],
                "total_seed_mass": int(total_seed_mass),
                "strong_normal_seed_mass": int(strong_normal_seed_mass),
                "normal_fraction": _ratio(strong_normal_seed_mass, total_seed_mass),
                "contains_defect_anchor": bool(contains_defect_anchor),
                "y_state": y_state,
                "final_y_state": y_state,
                "removed_before_gtr": bool(is_normal),
            }
        )

    normal_y_ids = [
        int(report["y_id"])
        for report in y_reports
        if report["y_state"] == Y_STATE_NORMAL
    ]
    defect_y_ids = [
        int(report["y_id"])
        for report in y_reports
        if report["y_state"] == Y_STATE_DEFECT
    ]
    unknown_y_ids = [
        int(report["y_id"])
        for report in y_reports
        if report["y_state"] == Y_STATE_UNKNOWN
    ]
    return {
        "y_reports": y_reports,
        "normal_y_ids": normal_y_ids,
        "defect_y_ids": defect_y_ids,
        "unknown_y_ids": unknown_y_ids,
    }


def build_component_filter_result(
    seed_component_inputs: Iterable[Mapping[str, Any]],
    *,
    enabled: bool = True,
    thresholds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Classify all Q components and aggregate their evidence to Y components."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be True or False.")
    if not enabled:
        return {
            "enabled": False,
            "seed_reports": [],
            "y_reports": [],
            "normal_y_ids": [],
            "defect_y_ids": [],
            "unknown_y_ids": [],
        }

    seed_reports = [
        classify_seed_component_evidence(
            **dict(component_input), thresholds=thresholds
        )
        for component_input in seed_component_inputs
    ]
    aggregation = aggregate_seed_evidence_to_y(
        seed_reports, thresholds=thresholds
    )
    return {
        "enabled": True,
        "seed_reports": seed_reports,
        **aggregation,
    }
