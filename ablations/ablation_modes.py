"""Scientific contracts for six main and supplementary COVERT ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


MAIN_ABLATION_MODES = (
    "full",
    "laplacian_restoration",
    "wo_va_hks",
    "wo_structural_suppression",
    "wo_seed_growth",
    "wo_counterfactual_verification",
)

SUPPLEMENTARY_ABLATION_MODES = (
    "wo_local_control_calibration",
    "fixed_absolute_threshold_verification",
)

SUPPORTED_ABLATION_MODES = (
    *MAIN_ABLATION_MODES,
    *SUPPLEMENTARY_ABLATION_MODES,
)


@dataclass(frozen=True)
class AblationSpec:
    mode: str
    paper_name: str
    changed_module: str
    description: str
    retained_mechanisms: tuple[str, ...]
    expected_diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COMMON_GTR = (
    "New AC physical-boundary support",
    "candidate morphological closing",
    "local-normal counterfactual calibration",
    "displacement and pseudo-stress screening",
    "four-quadrant point refinement",
)


ABLATION_SPECS: dict[str, AblationSpec] = {
    "full": AblationSpec(
        mode="full",
        paper_name="Full COVERT",
        changed_module="none (production control)",
        description=(
            "Delegates directly to the frozen production COVERT sample path; no "
            "configuration or decision is changed."
        ),
        retained_mechanisms=("all frozen production mechanisms",),
        expected_diagnostics={"production_delegate": True},
    ),
    "laplacian_restoration": AblationSpec(
        mode="laplacian_restoration",
        paper_name="First-order Laplacian restoration",
        changed_module="GTR restoration operator only",
        description=(
            "Replaces biharmonic continuation with first-order Laplacian "
            "continuation for both candidates and local-normal controls."
        ),
        retained_mechanisms=(
            "SV-VA-HKS joint evidence",
            "structural suppression",
            "descending candidate recovery",
            *_COMMON_GTR,
        ),
        expected_diagnostics={
            "restoration_operator": "laplacian",
            "local_normal_restoration_operator": "laplacian",
        },
    ),
    "wo_va_hks": AblationSpec(
        mode="wo_va_hks",
        paper_name="w/o VA-HKS evidence",
        changed_module="VA-HKS anomaly evidence",
        description=(
            "Removes VA-HKS from anomaly evidence used for both seed consensus "
            "and defect-preserving structural suppression, while retaining the "
            "structural suppression and veto mechanisms themselves."
        ),
        retained_mechanisms=(
            "New AC and Pure-HKS structure processing",
            "structural suppression and defect-preservation veto mechanisms",
            "confidence gating and original-SV gating",
            "descending candidate recovery and consolidation",
            *_COMMON_GTR,
        ),
        expected_diagnostics={
            "seed_anomaly_source": "sv_only",
            "defect_veto_anomaly_source": "sv_only",
            "va_hks_participates_in_seed_consensus": False,
            "va_hks_participates_in_defect_veto": False,
            "va_hks_computed_but_unused_as_anomaly_evidence": True,
        },
    ),
    "wo_structural_suppression": AblationSpec(
        mode="wo_structural_suppression",
        paper_name="w/o structural suppression",
        changed_module="front-end structural attenuation and associated vetoes",
        description=(
            "Removes New-AC/Pure-HKS attenuation, suppression propagation effects, "
            "suppression-specific defect veto, and AC seed removal. New AC remains "
            "computed and is retained as GTR physical-boundary support."
        ),
        retained_mechanisms=(
            "SV and VA-HKS computation and minimum consensus",
            "seed formation, growth, and consolidation",
            "New AC computation for GTR boundary support",
            *_COMMON_GTR,
        ),
        expected_diagnostics={
            "frontend_structural_suppression": False,
            "gtr_new_ac_boundary_support": True,
        },
    ),
    "wo_seed_growth": AblationSpec(
        mode="wo_seed_growth",
        paper_name="w/o candidate recovery",
        changed_module="descending SGCR candidate recovery",
        description=(
            "Sends cleaned high-confidence seed components directly to GTR and "
            "bypasses descending growth, growth-derived merging/scale rules, and "
            "Local Normal-U cleanup. This is not an Hmax=0 emulation."
        ),
        retained_mechanisms=(
            "evidence encoding and structural suppression",
            "joint seed confidence and seed cleaning",
            *_COMMON_GTR,
        ),
        expected_diagnostics={
            "descending_growth": False,
            "candidate_source": "cleaned_seed_components",
        },
    ),
    "wo_counterfactual_verification": AblationSpec(
        mode="wo_counterfactual_verification",
        paper_name="w/o counterfactual verification",
        changed_module="GTR restoration and verification",
        description=(
            "Accepts the exact candidate support after production closing and New "
            "AC physical-boundary exclusion, before restoration or GTR decisions."
        ),
        retained_mechanisms=(
            "full evidence, suppression, seeds, growth, and consolidation",
            "candidate-growth scale-consistency safeguard",
            "candidate morphological closing",
            "New AC physical-boundary support exclusion",
        ),
        expected_diagnostics={
            "gtr_restoration_executed": False,
            "final_mask_source": "pre_verification_candidate_support",
        },
    ),
    "wo_local_control_calibration": AblationSpec(
        mode="wo_local_control_calibration",
        paper_name="w/o local-control calibration",
        changed_module="candidate-to-control relative rejection only",
        description=(
            "Keeps nearby noncandidate control construction, restoration, and "
            "candidate/reference discrepancy ratios, but removes only the "
            "region-level relative-normal rejection based on those ratios. "
            "Minimum-size, large-shallow, pseudo-stress, point-refinement, and "
            "final local-cleanup decisions remain unchanged."
        ),
        retained_mechanisms=(
            "full geometric-spectral evidence encoding",
            "structural suppression and defect protection",
            "seed formation",
            "descending candidate recovery",
            "candidate merging and scale consistency",
            "candidate closing",
            "boundary-constrained biharmonic restoration",
            "local noncandidate control construction and restoration",
            "displacement and pseudo-stress computation",
            "minimum-size candidate rejection",
            "large-shallow candidate rejection",
            "pseudo-stress instability rejection",
            "four-quadrant point-level refinement",
            "final local cleanup",
        ),
        expected_diagnostics={
            "production_delegate": False,
            "base_production_config_unchanged": True,
            "gt_used_by_inference": False,
            "local_normal_reference_enabled": True,
            "local_normal_reference_computed": True,
            "relative_normal_rejection_enabled": False,
            "relative_normal_threshold_sentinel": -1.0,
            "large_shallow_rejection_retained": True,
            "stress_veto_retained": True,
            "point_refinement_retained": True,
            "local_normal_u_cleanup_retained": True,
            "restoration_operator": "biharmonic",
        },
    ),
    "fixed_absolute_threshold_verification": AblationSpec(
        mode="fixed_absolute_threshold_verification",
        paper_name="Fixed absolute-threshold verification",
        changed_module="relative-normal candidate rejection only",
        description=(
            "Replaces only the locally calibrated candidate/reference normal-like "
            "decision with the strict global rule D_p90 < tau_abs. Local controls "
            "are still restored for diagnostics, while large-shallow rejection and "
            "all other frozen Full COVERT mechanisms remain active."
        ),
        retained_mechanisms=(
            "full geometric-spectral evidence encoding",
            "structural suppression and defect protection",
            "seed formation and descending candidate recovery",
            "candidate merging, scale consistency, and closing",
            "boundary-constrained biharmonic restoration",
            "local noncandidate control construction and restoration (diagnostic only)",
            "minimum-size candidate rejection",
            "large-shallow candidate rejection",
            "pseudo-stress instability rejection",
            "four-quadrant point-level refinement",
            "final Local Normal-U cleanup",
        ),
        expected_diagnostics={
            "production_delegate": False,
            "base_production_config_unchanged": True,
            "gt_used_by_inference": False,
            "classifier_strategy": "fixed_absolute_threshold_verification",
            "absolute_statistic": "candidate_D_p90",
            "absolute_rule": "D_p90 < tau_abs",
            "absolute_keep_equal": True,
            "relative_normal_decision_consumed": False,
            "large_shallow_rejection_retained": True,
            "stress_veto_retained": True,
            "minimum_size_rejection_retained": True,
            "point_refinement_retained": True,
            "final_cleanup_retained": True,
            "biharmonic_restoration_retained": True,
            "local_control_computation_retained": True,
            "local_control_diagnostic_only": True,
        },
    ),
}


def get_ablation_spec(mode: str) -> AblationSpec:
    """Return the immutable contract for a supported mode."""

    normalized = str(mode).strip()
    if normalized not in ABLATION_SPECS:
        allowed = ", ".join(SUPPORTED_ABLATION_MODES)
        raise ValueError(f"Unsupported ablation mode {mode!r}; expected one of: {allowed}.")
    return ABLATION_SPECS[normalized]


def validate_semantic_diagnostics(mode: str, diagnostics: Mapping[str, Any]) -> None:
    """Fail loudly when a requested hook did not report its contract."""

    spec = get_ablation_spec(mode)
    mismatches = {
        key: {"expected": expected, "actual": diagnostics.get(key)}
        for key, expected in spec.expected_diagnostics.items()
        if diagnostics.get(key) != expected
    }
    if mismatches:
        raise AssertionError(f"Ablation semantic diagnostics failed for {mode}: {mismatches}")
