"""Strict configuration construction for COVERT sensitivity analysis v2.

Every experimental configuration starts from ``covert_sample.DEFAULT_CONFIG``
and changes only the declared frozen dataclass leaves.  The declarations below
are also the static 37-configuration audit used by the runner and validation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from covert_sample import (
    DEFAULT_CONFIG,
    CovertPipelineConfig,
    config_sha256,
    config_to_dict,
)
from sensitivity_analysis._fingerprint import (
    IMPLEMENTATION_FINGERPRINT_SCHEMA,
    implementation_files,
    implementation_metadata,
    implementation_sha256,
    production_file_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN8 = (
    "candybar",
    "diamond",
    "fish",
    "toffees",
    "bagel",
    "carrot",
    "peach",
    "potato",
)
REAL3D_MAIN4 = MAIN8[:4]
MVTEC_MAIN4 = MAIN8[4:]
DEFAULT_DATASET_ROOT = Path("data/real3d_ad")
DEFAULT_MVTEC_ROOT = Path("data/mvtec_3d_ad_converted")
DEFAULT_SAMPLE_WORKERS = 4
DEFAULT_QUERY_WORKERS = 1
SAFE_CONFIG_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

TARGET_FIELDS = frozenset(
    {
        "features.graph_knn_k",
        "features.surface_variation_scales",
        "features.va_hks_gamma",
        "new_ac.angle_threshold_deg",
        "confidence.high_confidence_seed_threshold",
        "sgcr.max_grow_hops",
        "gtr.adaptive_relative_normal_max_ratio",
    }
)


def _entry(
    config_id: str,
    parameters: Mapping[str, Any],
    changes: Mapping[str, Any],
    *,
    is_default: bool = False,
) -> dict[str, Any]:
    return {
        "id": config_id,
        "parameters": dict(parameters),
        "changes": dict(changes),
        "is_production_default": bool(is_default),
    }


EXPERIMENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "graph_geometry_coupling": {
        "directory": "01_graph_geometry_coupling",
        "paper_role": "main_text",
        "sweep_fields": (
            "features.graph_knn_k",
            "features.surface_variation_scales",
        ),
        "parameter_columns": ("graph_knn_k", "surface_variation_scales"),
        "configurations": (
            _entry(
                "small_k10_L8-16-32",
                {"graph_knn_k": 10, "surface_variation_scales": [8, 16, 32]},
                {
                    "features.graph_knn_k": 10,
                    "features.surface_variation_scales": [8, 16, 32],
                },
            ),
            _entry(
                "local_k14_L12-24-48",
                {"graph_knn_k": 14, "surface_variation_scales": [12, 24, 48]},
                {
                    "features.graph_knn_k": 14,
                    "features.surface_variation_scales": [12, 24, 48],
                },
            ),
            _entry(
                "default_k16_L12-24-48",
                {
                    "graph_knn_k": 16,
                    "surface_variation_scales": [12, 24, 48],
                },
                {
                    "features.graph_knn_k": 16,
                    "features.surface_variation_scales": [12, 24, 48],
                },
                is_default=True,
            ),
            _entry(
                "local_k18_L12-24-48",
                {"graph_knn_k": 18, "surface_variation_scales": [12, 24, 48]},
                {
                    "features.graph_knn_k": 18,
                    "features.surface_variation_scales": [12, 24, 48],
                },
            ),
            _entry(
                "scale_k24_L18-36-72",
                {"graph_knn_k": 24, "surface_variation_scales": [18, 36, 72]},
                {
                    "features.graph_knn_k": 24,
                    "features.surface_variation_scales": [18, 36, 72],
                },
            ),
            _entry(
                "large_k32_L24-48-96",
                {
                    "graph_knn_k": 32,
                    "surface_variation_scales": [24, 48, 96],
                },
                {
                    "features.graph_knn_k": 32,
                    "features.surface_variation_scales": [24, 48, 96],
                },
            ),
        ),
    },
    "vahks_modulation": {
        "directory": "02_vahks_modulation",
        "paper_role": "main_text",
        "sweep_fields": ("features.va_hks_gamma",),
        "parameter_columns": ("gamma",),
        "configurations": tuple(
            _entry(
                f"gamma_{gamma}",
                {"gamma": gamma},
                {"features.va_hks_gamma": gamma},
                is_default=gamma == 10,
            )
            for gamma in (0, 2, 5, 10, 20, 50)
        ),
    },
    "ac_boundary": {
        "directory": "03_ac_boundary",
        "paper_role": "main_text",
        "sweep_fields": ("new_ac.angle_threshold_deg",),
        "parameter_columns": ("theta_ac_deg",),
        "configurations": tuple(
            _entry(
                f"ac_{angle}",
                {"theta_ac_deg": angle},
                {"new_ac.angle_threshold_deg": angle},
                is_default=angle == 110,
            )
            for angle in (90, 100, 110, 120, 130)
        ),
    },
    "candidate_recovery": {
        "directory": "04_candidate_recovery",
        "paper_role": "main_text",
        "sweep_fields": (
            "confidence.high_confidence_seed_threshold",
            "sgcr.max_grow_hops",
        ),
        "parameter_columns": ("tau_q", "h_max"),
        "configurations": tuple(
            _entry(
                f"tauQ_{int(round(tau_q * 100)):03d}_H{hops}",
                {"tau_q": tau_q, "h_max": hops},
                {
                    "confidence.high_confidence_seed_threshold": tau_q,
                    "sgcr.max_grow_hops": hops,
                },
                is_default=math.isclose(tau_q, 0.50) and hops == 9,
            )
            for tau_q in (0.30, 0.40, 0.50, 0.60, 0.70)
            for hops in (5, 9, 13)
        ),
    },
    "gtr_local_calibration": {
        "directory": "05_gtr_local_calibration",
        "paper_role": "supplementary",
        "sweep_fields": ("gtr.adaptive_relative_normal_max_ratio",),
        "parameter_columns": ("tau_rel",),
        "configurations": tuple(
            _entry(
                f"tauRel_{int(round(tau_rel * 100)):03d}",
                {"tau_rel": tau_rel},
                {"gtr.adaptive_relative_normal_max_ratio": tau_rel},
                is_default=math.isclose(tau_rel, 0.50),
            )
            for tau_rel in (0.25, 0.40, 0.50, 0.60, 0.75)
        ),
    },
}

EXPECTED_COUNTS = {
    "graph_geometry_coupling": 6,
    "vahks_modulation": 6,
    "ac_boundary": 5,
    "candidate_recovery": 15,
    "gtr_local_calibration": 5,
}
EXPECTED_TOTAL_CONFIGURATIONS = 37

# Compatibility export for validators and reports.  The schema-2 list is
# discovered from the production entrypoint rather than copied from
# ``covert_batch.py``.
PRODUCTION_IMPLEMENTATION_FILES = implementation_files()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {item.name: _canonical_value(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    return value


def config_value(config: CovertPipelineConfig, dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not hasattr(value, part):
            raise ValueError(f"Unknown production config field: {dotted_path}")
        value = getattr(value, part)
    return _canonical_value(value)


def snapshot_value(snapshot: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = snapshot
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Production result snapshot lacks {dotted_path}")
        value = value[part]
    return _canonical_value(value)


def _replace_leaf(
    config: CovertPipelineConfig,
    dotted_path: str,
    requested_value: Any,
) -> CovertPipelineConfig:
    parts = dotted_path.split(".")
    if len(parts) != 2:
        raise ValueError(f"Only one-level nested dataclass leaves are supported: {dotted_path}")
    section_name, leaf_name = parts
    section = getattr(config, section_name, None)
    if section is None or not dataclasses.is_dataclass(section):
        raise ValueError(f"Unknown dataclass section in {dotted_path}")
    field_names = {item.name for item in dataclasses.fields(section)}
    if leaf_name not in field_names:
        raise ValueError(f"Unknown dataclass field: {dotted_path}")
    current_value = getattr(section, leaf_name)
    value = requested_value
    if isinstance(current_value, tuple):
        if not isinstance(value, (tuple, list)):
            raise TypeError(f"{dotted_path} must be a sequence")
        value = tuple(value)
    elif isinstance(current_value, bool):
        value = bool(value)
    elif isinstance(current_value, int):
        value = int(value)
    elif isinstance(current_value, float):
        value = float(value)
    new_section = dataclasses.replace(section, **{leaf_name: value})
    return dataclasses.replace(config, **{section_name: new_section})


def _flatten_config(config: CovertPipelineConfig) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for section in dataclasses.fields(config):
        section_value = getattr(config, section.name)
        if dataclasses.is_dataclass(section_value):
            for leaf in dataclasses.fields(section_value):
                flattened[f"{section.name}.{leaf.name}"] = _canonical_value(
                    getattr(section_value, leaf.name)
                )
        else:
            flattened[section.name] = _canonical_value(section_value)
    return flattened


def build_sensitivity_config(
    changes: Mapping[str, Any],
    *,
    allowed_fields: Sequence[str],
) -> tuple[CovertPipelineConfig, dict[str, Any]]:
    """Create and audit an immutable config copy rooted at DEFAULT_CONFIG."""

    declared = tuple(str(field) for field in allowed_fields)
    if len(set(declared)) != len(declared):
        raise ValueError("Duplicate allowed sensitivity field")
    if set(declared) != set(changes):
        raise ValueError(
            f"Declared sweep fields {list(declared)} do not match changes {sorted(changes)}"
        )
    unsupported = sorted(set(declared) - TARGET_FIELDS)
    if unsupported:
        raise ValueError(f"Forbidden sensitivity field(s): {unsupported}")

    production_hash_before = config_sha256(DEFAULT_CONFIG)
    config = DEFAULT_CONFIG
    for dotted_path in declared:
        config = _replace_leaf(config, dotted_path, changes[dotted_path])
    if config is DEFAULT_CONFIG:
        raise AssertionError("Sensitivity construction unexpectedly returned DEFAULT_CONFIG itself")

    default_flat = _flatten_config(DEFAULT_CONFIG)
    sensitivity_flat = _flatten_config(config)
    differing = sorted(
        field for field in default_flat if default_flat[field] != sensitivity_flat[field]
    )
    unexpected = sorted(set(differing) - set(declared))
    if unexpected:
        raise AssertionError(f"Non-swept production fields changed: {unexpected}")
    for field in declared:
        requested = _canonical_value(changes[field])
        if sensitivity_flat[field] != requested:
            raise AssertionError(
                f"Requested {field}={requested!r}, constructed {sensitivity_flat[field]!r}"
            )
    if config_sha256(DEFAULT_CONFIG) != production_hash_before:
        raise AssertionError("Global covert_sample.DEFAULT_CONFIG was mutated")

    audit = {
        "production_default_config_hash": production_hash_before,
        "sensitivity_config_hash": config_sha256(config),
        "changed_config_fields": list(declared),
        "actual_nondefault_fields": differing,
        "production_default_values": {
            field: default_flat[field] for field in declared
        },
        "requested_values": {
            field: _canonical_value(changes[field]) for field in declared
        },
        "effective_values": {
            field: sensitivity_flat[field] for field in declared
        },
        "all_non_swept_fields_equal_production_default": True,
    }
    return config, audit


def read_experiment_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _normalized(value: Any) -> Any:
    return json.loads(canonical_json(value))


def validate_experiment_config(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    required = {
        "experiment_name",
        "paper_role",
        "sweep_parameters",
        "fixed_parameter_policy",
        "production_default",
        "metric_semantics",
        "pipeline",
        "configurations",
        "evaluation_scopes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{source}: missing required keys {missing}")
    if "launched_at" in payload:
        raise ValueError(f"{source}: launched_at belongs only in a runtime snapshot")

    experiment_name = str(payload["experiment_name"])
    if experiment_name not in EXPERIMENT_DEFINITIONS:
        raise ValueError(f"{source}: unknown experiment {experiment_name!r}")
    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    if source.parent.name != definition["directory"]:
        raise ValueError(
            f"{source}: expected directory {definition['directory']!r} for {experiment_name}"
        )
    if payload["paper_role"] != definition["paper_role"]:
        raise ValueError(f"{source}: incorrect paper_role")

    pipeline = payload["pipeline"]
    if not isinstance(pipeline, Mapping):
        raise TypeError(f"{source}: pipeline must be an object")
    if pipeline.get("entrypoint") != "covert_sample.run_covert_sample":
        raise ValueError(f"{source}: pipeline must call production run_covert_sample")
    if pipeline.get("run_good_samples") is not False:
        raise ValueError(f"{source}: sensitivity must not run good samples")
    if tuple(pipeline.get("categories") or ()) != MAIN8:
        raise ValueError(f"{source}: categories must be the canonical Main8 in order")

    scopes = payload["evaluation_scopes"]
    if list(scopes) != ["all_main8", "heldout_excluding_dev10"]:
        raise ValueError(f"{source}: invalid evaluation scopes")
    metric_semantics = payload["metric_semantics"]
    if not isinstance(metric_semantics, Mapping):
        raise TypeError(f"{source}: metric_semantics must be an object")
    if metric_semantics.get("within_category") != "sample_macro_mean":
        raise ValueError(f"{source}: within-category aggregation must be sample macro")
    if metric_semantics.get("across_categories") != "category_balanced_mean":
        raise ValueError(f"{source}: across-category aggregation must be balanced")

    expected = list(definition["configurations"])
    observed = payload["configurations"]
    if not isinstance(observed, list) or len(observed) != len(expected):
        raise ValueError(
            f"{source}: expected {len(expected)} configurations, got "
            f"{len(observed) if isinstance(observed, list) else 'non-list'}"
        )
    seen: set[str] = set()
    for index, (item, required_item) in enumerate(zip(observed, expected)):
        if not isinstance(item, Mapping):
            raise TypeError(f"{source}: configuration {index} must be an object")
        config_id = str(item.get("id", ""))
        if not SAFE_CONFIG_ID.fullmatch(config_id):
            raise ValueError(f"{source}: unsafe configuration id {config_id!r}")
        if config_id in seen:
            raise ValueError(f"{source}: duplicate configuration id {config_id}")
        seen.add(config_id)
        for key in ("id", "parameters", "changes", "is_production_default"):
            if _normalized(item.get(key)) != _normalized(required_item[key]):
                raise ValueError(
                    f"{source}: configuration {index} {key} differs from the frozen v2 declaration"
                )
        build_sensitivity_config(
            item["changes"], allowed_fields=definition["sweep_fields"]
        )

    defaults = [item for item in observed if item.get("is_production_default") is True]
    if len(defaults) != 1:
        raise ValueError(f"{source}: exactly one production-default configuration is required")
    declared_defaults = payload["production_default"]
    expected_defaults = {
        field: config_value(DEFAULT_CONFIG, field)
        for field in definition["sweep_fields"]
    }
    if _normalized(declared_defaults) != _normalized(expected_defaults):
        raise ValueError(f"{source}: production_default does not match DEFAULT_CONFIG")
    return dict(payload)


def validate_global_config_count() -> dict[str, int]:
    counts = {
        name: len(definition["configurations"])
        for name, definition in EXPERIMENT_DEFINITIONS.items()
    }
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"Sensitivity configuration counts changed: {counts}")
    if sum(counts.values()) != EXPECTED_TOTAL_CONFIGURATIONS:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_CONFIGURATIONS} total configurations, got {sum(counts.values())}"
        )
    return counts


def default_configuration(payload: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in payload["configurations"]
        if item.get("is_production_default") is True
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one production-default configuration")
    return matches[0]


def assert_result_effective_values(
    result: Any,
    *,
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate effective values emitted by the production result.

    All fields are checked against ``CovertSampleResult.config_snapshot`` and
    the GTR threshold is additionally checked against the GTR runtime audit,
    which is the only target parameter currently repeated in a public stage
    audit.  Missing or divergent values fail the current configuration.
    """

    sources: dict[str, Any] = {}
    for field, requested_raw in changes.items():
        requested = _canonical_value(requested_raw)
        effective = snapshot_value(result.config_snapshot, field)
        if effective != requested:
            raise RuntimeError(
                f"Effective parameter mismatch for {field}: requested={requested!r}, "
                f"production_result={effective!r}"
            )
        sources[field] = {
            "requested": requested,
            "effective": effective,
            "source": f"CovertSampleResult.config_snapshot.{field}",
        }

    gtr_field = "gtr.adaptive_relative_normal_max_ratio"
    if gtr_field in changes:
        summary = result.audit.get("gtr_summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError("Production GTR summary is missing")
        classifier = summary.get("candidate_classifier")
        if not isinstance(classifier, Mapping):
            raise RuntimeError("Production GTR candidate-classifier summary is missing")
        audit_value = float(classifier.get("relative_normal_max_ratio"))
        requested = float(changes[gtr_field])
        if not math.isclose(audit_value, requested, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"GTR runtime audit mismatch: requested={requested}, effective={audit_value}"
            )
        sources[gtr_field]["runtime_audit"] = audit_value
        sources[gtr_field]["runtime_audit_source"] = (
            "CovertSampleResult.audit.gtr_summary.candidate_classifier."
            "relative_normal_max_ratio"
        )
    return sources


validate_global_config_count()


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_MVTEC_ROOT",
    "DEFAULT_QUERY_WORKERS",
    "DEFAULT_SAMPLE_WORKERS",
    "EXPECTED_COUNTS",
    "EXPECTED_TOTAL_CONFIGURATIONS",
    "EXPERIMENT_DEFINITIONS",
    "MAIN8",
    "MVTEC_MAIN4",
    "PROJECT_ROOT",
    "PRODUCTION_IMPLEMENTATION_FILES",
    "IMPLEMENTATION_FINGERPRINT_SCHEMA",
    "REAL3D_MAIN4",
    "TARGET_FIELDS",
    "assert_result_effective_values",
    "build_sensitivity_config",
    "canonical_json",
    "config_value",
    "default_configuration",
    "implementation_sha256",
    "implementation_metadata",
    "implementation_files",
    "production_file_sha256",
    "read_experiment_config",
    "sha256_json",
    "snapshot_value",
    "validate_experiment_config",
    "validate_global_config_count",
]
