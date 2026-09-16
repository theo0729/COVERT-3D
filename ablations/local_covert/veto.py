"""ABLATION-LOCAL defect-veto adapters. DO NOT USE IN PRODUCTION."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vast.covert._veto import build_legacy_hks_defect_veto_override


def build_sv_only_veto_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a production veto context with VA anomaly evidence replaced by SV.

    The production veto computes ``min(normalized(SV), normalized(VA-HKS))``.
    Supplying the same SV vector to both branches preserves the entire veto
    mechanism while making its anomaly support strictly SV-only.
    """

    adapted = dict(context)
    sv_response = np.asarray(context["sv_response"], dtype=np.float64).copy()
    adapted["sv_response"] = sv_response
    adapted["va_hks_response"] = sv_response.copy()
    return adapted


def build_sv_only_defect_veto_override(
    context: dict[str, Any], *, thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Run the frozen production veto algorithm with SV-only anomaly evidence."""

    result = build_legacy_hks_defect_veto_override(
        build_sv_only_veto_context(context), thresholds=thresholds
    )
    audit = dict(result.get("audit", {}))
    summary = dict(audit.get("summary", {}))
    summary.update(
        {
            "defect_veto_anomaly_source": "sv_only",
            "va_hks_participates_in_defect_veto": False,
            "production_veto_algorithm_retained": True,
            "production_veto_thresholds_retained": True,
        }
    )
    audit["summary"] = summary
    audit["defect_veto_anomaly_source"] = "sv_only"
    audit["va_hks_participates_in_defect_veto"] = False
    result = dict(result)
    result["audit"] = audit
    return result

