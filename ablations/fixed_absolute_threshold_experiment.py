"""Fit, freeze, smoke, and run the fixed absolute-threshold verification.

The formal ``run`` command first fits one global threshold on the frozen four
development groups, verifies exact replay against a direct run, freezes the
selection artifacts, and only then starts the Main8 held-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations.covert_ablation_batch import (  # noqa: E402
    ablation_implementation_hash,
    run_ablation_batch,
)
from ablations.covert_ablation_sample import run_ablation_sample  # noqa: E402
from ablations.fixed_absolute_threshold import (  # noqa: E402
    DEVELOPMENT_VARIANTS,
    METRIC_FIELDS,
    MODE,
    aggregate_development_sweep,
    exact_breakpoint_grid,
    replay_fixed_absolute_final_mask,
    select_threshold,
    validate_threshold,
)
from covert_batch import (  # noqa: E402
    DATASET_ROOT,
    MVTEC_ROOT,
    SampleSpec,
    _atomic_write_csv,
    _atomic_write_json,
    build_aggregate_tables,
    discover_samples,
)
from covert_sample import DEFAULT_CONFIG, _binary_metrics, config_hash  # noqa: E402
from experiments.benchmark_categories import BENCHMARK_CATEGORIES  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / MODE
FULL_DEVELOPMENT_MANIFEST = (
    PROJECT_ROOT
    / "experiments"
    / "dev10_physical_groups.csv"
)
FULL_CONTROL_DIR = Path(__file__).resolve().parent / "full"
REFERENCE_FULL = {
    "precision": 0.651,
    "recall": 0.736,
    "f1": 0.641,
    "iou": 0.519,
    "mean_fpr": 0.039,
}
REFERENCE_TOLERANCE = 0.001


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_progress(output_dir: Path, **payload: Any) -> None:
    _atomic_write_json(
        output_dir / "experiment_progress.json",
        {"mode": MODE, "updated_at_utc": _utc_now(), **payload},
    )


def _resolve_development_samples(
    *, dataset_root: Path, mvtec_root: Path
) -> list[tuple[SampleSpec, str]]:
    specs, failures = discover_samples(
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
        categories=("fish", "diamond"),
    )
    if failures:
        raise RuntimeError(f"Development discovery failed: {failures}")
    by_identity = {(sample.category, sample.sample_id): sample for sample in specs}
    resolved: list[tuple[SampleSpec, str]] = []
    for category, group_key, sample_id in DEVELOPMENT_VARIANTS:
        sample = by_identity.get((category, sample_id))
        if sample is None:
            raise FileNotFoundError(
                f"Missing frozen development sample {category}/{sample_id}."
            )
        resolved.append((sample, group_key))
    if len(resolved) != 8:
        raise AssertionError("The fixed development manifest must contain 8 variants.")
    return resolved


def materialize_development_manifest(
    *, output_dir: Path, dataset_root: Path, mvtec_root: Path
) -> tuple[list[tuple[SampleSpec, str]], str]:
    """Freeze the exact eight identities before any fitting inference."""

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _resolve_development_samples(
        dataset_root=dataset_root, mvtec_root=mvtec_root
    )
    production_hash = config_hash(DEFAULT_CONFIG)
    implementation = ablation_implementation_hash()
    rows = [
        {
            "dataset": sample.dataset,
            "category": sample.category,
            "physical_group": f"{sample.category}/{group_key}",
            "group_key": group_key,
            "sample_id": sample.sample_id,
            "variant": "cut" if sample.sample_id.endswith("_cut") else "original",
            "pc_path": str(Path(sample.pc_path).resolve()),
            "gt_path": str(Path(str(sample.gt_path)).resolve()),
            "production_config_hash": production_hash,
            "experiment_implementation_hash": implementation,
            "git_commit": "unavailable_not_a_git_checkout",
        }
        for sample, group_key in samples
    ]
    manifest_path = output_dir / "development_manifest.csv"
    _atomic_write_csv(manifest_path, rows)
    manifest_hash = _sha256(manifest_path)
    _atomic_write_json(
        output_dir / "development_manifest.json",
        {
            "schema": "fixed-absolute-development-manifest-v1",
            "created_at_utc": _utc_now(),
            "physical_group_count": 4,
            "variant_count": 8,
            "production_config_hash": production_hash,
            "experiment_implementation_hash": implementation,
            "csv_path": str(manifest_path.resolve()),
            "csv_sha256": manifest_hash,
            "samples": rows,
        },
    )
    return samples, manifest_hash


def _development_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    sample = SampleSpec(**payload["sample"])
    result = run_ablation_sample(
        sample.pc_path,
        sample.gt_path,
        ablation_mode=MODE,
        config=DEFAULT_CONFIG,
        query_workers=int(payload["query_workers"]),
        absolute_threshold=0.0,
    )
    replay = result.audit["fixed_absolute_replay"]
    return {
        "dataset": sample.dataset,
        "category": sample.category,
        "group_key": str(payload["group_key"]),
        "sample_id": sample.sample_id,
        "pc_path": sample.pc_path,
        "gt_path": sample.gt_path,
        "production_config_hash": result.production_config_hash,
        "experiment_implementation_hash": str(payload["implementation_hash"]),
        "metrics_at_zero": dict(result.metrics),
        "diagnostics": dict(result.ablation_metadata["diagnostics"]),
        "component_reports": [dict(report) for report in replay["component_reports"]],
        "final_labels_at_zero": np.asarray(
            replay["final_defect_labels"], dtype=np.int32
        ),
        "gt_mask": np.asarray(result.gt_mask, dtype=bool),
        "inference_total_seconds": float(result.timings.inference_total_seconds),
    }


def _development_artifact_dir(output_dir: Path, row: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "development_sample_artifacts"
        / str(row["category"])
        / str(row["sample_id"])
    )


def _save_development_row(output_dir: Path, row: Mapping[str, Any]) -> None:
    artifact_dir = _development_artifact_dir(output_dir, row)
    _write_npz_atomic(
        artifact_dir / "replay_state.npz",
        final_labels_at_zero=np.asarray(row["final_labels_at_zero"], dtype=np.int32),
        gt_mask=np.asarray(row["gt_mask"], dtype=np.uint8),
    )
    serialized = {
        key: value
        for key, value in row.items()
        if key not in {"final_labels_at_zero", "gt_mask"}
    }
    _atomic_write_json(artifact_dir / "metrics.json", serialized)


def _load_development_row(
    output_dir: Path, sample: SampleSpec, group_key: str
) -> dict[str, Any] | None:
    identity = {
        "category": sample.category,
        "sample_id": sample.sample_id,
    }
    artifact_dir = _development_artifact_dir(output_dir, identity)
    metrics_path = artifact_dir / "metrics.json"
    state_path = artifact_dir / "replay_state.npz"
    if not metrics_path.is_file() or not state_path.is_file():
        return None
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        raw.get("production_config_hash") != config_hash(DEFAULT_CONFIG)
        or raw.get("experiment_implementation_hash")
        != ablation_implementation_hash()
        or raw.get("group_key") != group_key
        or raw.get("pc_path") != sample.pc_path
        or raw.get("gt_path") != sample.gt_path
    ):
        return None
    with np.load(state_path) as state:
        raw["final_labels_at_zero"] = np.asarray(
            state["final_labels_at_zero"], dtype=np.int32
        )
        raw["gt_mask"] = np.asarray(state["gt_mask"], dtype=bool)
    return raw


def _run_development_samples(
    *,
    output_dir: Path,
    samples: Sequence[tuple[SampleSpec, str]],
    sample_workers: int,
    query_workers: int,
    resume: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: list[tuple[SampleSpec, str]] = []
    for sample, group_key in samples:
        previous = (
            _load_development_row(output_dir, sample, group_key) if resume else None
        )
        if previous is None:
            pending.append((sample, group_key))
        else:
            rows.append(previous)
    started = time.time()
    resumed_count = len(rows)

    def publish(status: str) -> None:
        _atomic_write_json(
            output_dir / "development_progress.json",
            {
                "status": status,
                "discovered": len(samples),
                "completed": len(rows),
                "resumed": resumed_count,
                "completed_this_run": max(len(rows) - resumed_count, 0),
                "pending": len(samples) - len(rows),
                "started_at_utc": datetime.fromtimestamp(
                    started, tz=timezone.utc
                ).isoformat(),
                "updated_at_utc": _utc_now(),
                "elapsed_seconds": max(time.time() - started, 0.0),
            },
        )

    def record(row: dict[str, Any]) -> None:
        _save_development_row(output_dir, row)
        rows.append(row)
        publish("running")
        print(
            f"[development {len(rows):02d}/{len(samples):02d}] "
            f"{row['category']}/{row['sample_id']} success",
            flush=True,
        )

    publish("running")
    implementation = ablation_implementation_hash()
    payloads = [
        {
            "sample": dataclasses.asdict(sample),
            "group_key": group_key,
            "query_workers": query_workers,
            "implementation_hash": implementation,
        }
        for sample, group_key in pending
    ]
    if sample_workers == 1:
        for payload in payloads:
            record(_development_worker(payload))
    elif payloads:
        with ProcessPoolExecutor(max_workers=sample_workers) as executor:
            futures = [executor.submit(_development_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                record(future.result())
    rows.sort(key=lambda row: (str(row["category"]), str(row["sample_id"])))
    publish("completed")
    return rows


def _verify_direct_replay(
    *,
    output_dir: Path,
    sample: SampleSpec,
    cached_row: Mapping[str, Any],
    threshold: float,
    query_workers: int,
) -> dict[str, Any]:
    replay_mask = replay_fixed_absolute_final_mask(
        np.asarray(cached_row["final_labels_at_zero"], dtype=np.int32),
        cached_row["component_reports"],
        threshold,
    )
    direct = run_ablation_sample(
        sample.pc_path,
        sample.gt_path,
        ablation_mode=MODE,
        config=DEFAULT_CONFIG,
        query_workers=query_workers,
        absolute_threshold=threshold,
    )
    exact = bool(np.array_equal(replay_mask, direct.final_mask))
    payload = {
        "passed": exact,
        "sample": f"{sample.category}/{sample.sample_id}",
        "threshold": float(threshold),
        "replay_positive_points": int(np.sum(replay_mask)),
        "direct_positive_points": int(np.sum(direct.final_mask)),
        "differing_points": int(np.sum(replay_mask != direct.final_mask)),
        "checked_at_utc": _utc_now(),
    }
    _atomic_write_json(output_dir / "replay_equivalence.json", payload)
    if not exact:
        raise AssertionError(f"Threshold replay did not match the direct run: {payload}")
    return payload


def fit_threshold(
    *,
    output_dir: Path,
    dataset_root: Path,
    mvtec_root: Path,
    sample_workers: int,
    query_workers: int,
    resume: bool,
) -> dict[str, Any]:
    _write_progress(output_dir, phase="development_manifest", status="running")
    samples, manifest_hash = materialize_development_manifest(
        output_dir=output_dir,
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
    )
    _write_progress(output_dir, phase="development_fit", status="running")
    cached = _run_development_samples(
        output_dir=output_dir,
        samples=samples,
        sample_workers=sample_workers,
        query_workers=query_workers,
        resume=resume,
    )
    all_d90 = [
        float(report["D_p90"])
        for row in cached
        for report in row["component_reports"]
        if np.isfinite(float(report["D_p90"]))
    ]
    grid = exact_breakpoint_grid(all_d90)
    _atomic_write_json(
        output_dir / "threshold_grid.json",
        {
            "construction": "0.0 plus nextafter(x, +inf) for each unique finite development candidate D_p90",
            "source_scope": "four frozen physical development groups / eight variants only",
            "finite_candidate_d90_count": len(all_d90),
            "unique_candidate_d90_count": len(set(all_d90)),
            "threshold_count": len(grid),
            "thresholds": grid,
            "thresholds_float_hex": [float(value).hex() for value in grid],
            "historical_anchor_values_added": False,
            "heldout_values_used": False,
        },
    )
    sample_rows: list[dict[str, Any]] = []
    for threshold in grid:
        for row in cached:
            mask = replay_fixed_absolute_final_mask(
                np.asarray(row["final_labels_at_zero"], dtype=np.int32),
                row["component_reports"],
                threshold,
            )
            metrics = _binary_metrics(mask, np.asarray(row["gt_mask"], dtype=bool))
            denominator = int(metrics["tp"]) + int(metrics["fp"]) + int(metrics["fn"])
            metrics["iou"] = (
                float(int(metrics["tp"]) / denominator) if denominator else 0.0
            )
            sample_rows.append(
                {
                    "threshold": float(threshold),
                    "threshold_float_hex": float(threshold).hex(),
                    "category": row["category"],
                    "group_key": row["group_key"],
                    "sample_id": row["sample_id"],
                    **metrics,
                }
            )
    group_rows, category_rows, objective_rows = aggregate_development_sweep(sample_rows)
    selected = select_threshold(objective_rows)
    _atomic_write_csv(output_dir / "threshold_sweep_sample_metrics.csv", sample_rows)
    _atomic_write_csv(output_dir / "threshold_sweep_group_metrics.csv", group_rows)
    _atomic_write_csv(output_dir / "threshold_sweep_category_metrics.csv", category_rows)
    _atomic_write_csv(output_dir / "threshold_sweep_objective.csv", objective_rows)
    selection_path = output_dir / "threshold_selection.json"
    selection_payload = {
        "schema": "fixed-absolute-threshold-selection-v1",
        "frozen_at_utc": _utc_now(),
        "status": "frozen_before_heldout_evaluation",
        "classifier_strategy": MODE,
        "absolute_rule": "D_p90 < tau_abs",
        "absolute_keep_equal": True,
        "selected_threshold": float(selected["threshold"]),
        "selected_threshold_float_hex": float(selected["threshold"]).hex(),
        "selected_category_balanced_f1": float(
            selected["category_balanced_f1"]
        ),
        "selected_category_balanced_iou": float(
            selected["category_balanced_iou"]
        ),
        "selection_trace": selected["selection_trace"],
        "threshold_grid": grid,
        "threshold_grid_float_hex": [float(value).hex() for value in grid],
        "threshold_sweep_sample_metrics": sample_rows,
        "threshold_sweep_group_metrics": group_rows,
        "threshold_sweep_category_metrics": category_rows,
        "threshold_sweep_objective": objective_rows,
        "development_manifest_sha256": manifest_hash,
        "production_config_hash": config_hash(DEFAULT_CONFIG),
        "experiment_implementation_hash": ablation_implementation_hash(),
        "development_physical_group_count": 4,
        "development_variant_count": 8,
        "defect_free_samples_used": False,
        "heldout_samples_used": False,
        "category_specific_threshold": False,
    }
    _atomic_write_json(selection_path, selection_payload)

    first_sample, first_group = samples[0]
    first_cached = next(
        row
        for row in cached
        if row["category"] == first_sample.category
        and row["sample_id"] == first_sample.sample_id
        and row["group_key"] == first_group
    )
    replay_sensitive_d90 = [
        float(report["D_p90"])
        for row in cached
        for report in row["component_reports"]
        if np.isfinite(float(report["D_p90"]))
        and not bool(report.get("too_small", False))
        and not bool(report.get("large_shallow_triggered", False))
        and not bool(report.get("stress_veto", False))
    ]
    replay_test_threshold = (
        float(np.nextafter(min(replay_sensitive_d90), np.inf))
        if replay_sensitive_d90
        else float(grid[1] if len(grid) > 1 else grid[0])
    )
    replay_validation = _verify_direct_replay(
        output_dir=output_dir,
        sample=first_sample,
        cached_row=first_cached,
        threshold=replay_test_threshold,
        query_workers=query_workers,
    )
    protected_files = (
        "development_manifest.csv",
        "threshold_grid.json",
        "threshold_sweep_sample_metrics.csv",
        "threshold_sweep_group_metrics.csv",
        "threshold_sweep_category_metrics.csv",
        "threshold_sweep_objective.csv",
        "threshold_selection.json",
        "replay_equivalence.json",
    )
    lock_payload = {
        "schema": "fixed-absolute-threshold-lock-v1",
        "frozen_at_utc": selection_payload["frozen_at_utc"],
        "selected_threshold": float(selected["threshold"]),
        "production_config_hash": config_hash(DEFAULT_CONFIG),
        "experiment_implementation_hash": ablation_implementation_hash(),
        "files": {
            name: _sha256(output_dir / name) for name in protected_files
        },
        "replay_equivalence_passed": bool(replay_validation["passed"]),
    }
    _atomic_write_json(output_dir / "threshold_lock.json", lock_payload)
    _write_progress(
        output_dir,
        phase="threshold_frozen",
        status="completed",
        selected_threshold=float(selected["threshold"]),
    )
    return selection_payload


def verify_threshold_lock(output_dir: Path) -> dict[str, Any]:
    lock_path = output_dir / "threshold_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"Frozen threshold lock not found: {lock_path}. Run the fit phase first."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("production_config_hash") != config_hash(DEFAULT_CONFIG):
        raise RuntimeError("Production config changed after threshold fitting.")
    if lock.get("experiment_implementation_hash") != ablation_implementation_hash():
        raise RuntimeError("Experiment implementation changed after threshold fitting.")
    for name, expected in lock.get("files", {}).items():
        path = output_dir / str(name)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Frozen threshold artifact changed or is missing: {path}")
    if lock.get("replay_equivalence_passed") is not True:
        raise RuntimeError("Frozen threshold did not pass replay equivalence.")
    return lock


def _load_excluded_dev10_identities() -> tuple[set[tuple[str, str]], str]:
    if not FULL_DEVELOPMENT_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Frozen held-out exclusion manifest not found: {FULL_DEVELOPMENT_MANIFEST}"
        )
    identities: set[tuple[str, str]] = set()
    with FULL_DEVELOPMENT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            category = str(row["category"])
            members = str(row["member_sample_ids"]).split("|")
            identities.update((category, member) for member in members if member)
    if len(identities) != 20:
        raise AssertionError(
            f"Held-out exclusion manifest resolved {len(identities)} variants; expected 20."
        )
    return identities, _sha256(FULL_DEVELOPMENT_MANIFEST)


def _category_balanced_metrics(
    rows: Sequence[Mapping[str, Any]], good_payload: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    per_category: dict[str, dict[str, Any]] = {}
    for category in BENCHMARK_CATEGORIES:
        selected = [row for row in rows if row.get("category") == category]
        successful = [
            row
            for row in selected
            if row.get("success") is True or row.get("status") == "success"
        ]
        metrics = {
            metric: float(np.mean([float(row[metric]) for row in successful]))
            if successful
            else 0.0
            for metric in METRIC_FIELDS
        }
        good_category = good_payload["categories"][category]
        per_category[category] = {
            "heldout_defective_count": len(selected),
            "heldout_defective_success_count": len(successful),
            "defect_free_count": int(good_category["sample_count"]),
            "defect_free_success_count": int(good_category["success_count"]),
            **metrics,
            "mean_fpr": float(good_category["mean_fpr"]),
        }
    balanced = {
        metric: float(np.mean([row[metric] for row in per_category.values()]))
        for metric in (*METRIC_FIELDS, "mean_fpr")
    }
    return per_category, balanced


def _read_sample_artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "sample_artifacts").glob("*/*/*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _write_report(
    *,
    output_dir: Path,
    selection: Mapping[str, Any],
    paper: Mapping[str, Any],
) -> None:
    full = paper["full_control"]
    fixed = paper["fixed_absolute"]
    lines = [
        "# Fixed Absolute-Threshold Verification Report",
        "",
        "## Scientific question",
        "",
        "This experiment compares locally calibrated candidate verification with one global absolute displacement threshold fitted only on four physical development groups.",
        "",
        "## Variant definition",
        "",
        "Only the relative-normal decision is replaced by the strict rule `D_p90 < tau_abs`; equality is retained. Large-shallow rejection, stress veto, minimum-size rejection, point-level refinement, final cleanup, and biharmonic restoration remain active. Local controls are computed for diagnostics only.",
        "",
        "## Development fitting and freeze",
        "",
        f"- Development scope: 4 physical groups / 8 variants.",
        f"- Exact breakpoint-grid size: {len(selection['threshold_grid'])}.",
        f"- Selected threshold: `{float(selection['selected_threshold']):.17g}`.",
        f"- Development category-balanced F1: {float(selection['selected_category_balanced_f1']):.6f}.",
        f"- Development category-balanced IoU: {float(selection['selected_category_balanced_iou']):.6f}.",
        f"- Manifest SHA-256: `{selection['development_manifest_sha256']}`.",
        f"- Production config hash: `{selection['production_config_hash']}`.",
        f"- Experiment implementation hash: `{selection['experiment_implementation_hash']}`.",
        "- Threshold selection was frozen before held-out inference; exact replay matched a direct run.",
        "",
        "## Formal held-out result",
        "",
        "| Variant | Precision | Recall | F1 | IoU | Mean FPR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Full COVERT | {full['precision']:.6f} | {full['recall']:.6f} | {full['f1']:.6f} | {full['iou']:.6f} | {full['mean_fpr']:.6f} |",
        f"| Fixed absolute-threshold verification | {fixed['precision']:.6f} | {fixed['recall']:.6f} | {fixed['f1']:.6f} | {fixed['iou']:.6f} | {fixed['mean_fpr']:.6f} |",
        "",
        "## Per-category diagnosis",
        "",
        "| Category | Precision | Recall | F1 | IoU | Mean FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category in BENCHMARK_CATEGORIES:
        row = paper["per_category_metrics"][category]
        lines.append(
            f"| {category} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['iou']:.6f} | {row['mean_fpr']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Leakage guard",
            "",
            "No held-out or defect-free result was used to alter the threshold grid, objective, tie-break rule, selected threshold, or any other COVERT parameter. A single global threshold was used for every category.",
            "",
        ]
    )
    (output_dir / "FIXED_ABSOLUTE_THRESHOLD_VERIFICATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def finalize_paper_outputs(
    *,
    output_dir: Path,
    selection: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    excluded, exclusion_hash = _load_excluded_dev10_identities()
    heldout = [
        dict(row)
        for row in all_rows
        if (str(row.get("category")), str(row.get("sample_id"))) not in excluded
    ]
    if len(all_rows) != 618 or len(heldout) != 598:
        raise AssertionError(
            f"Expected 618 discovered / 598 held-out defective rows; got {len(all_rows)} / {len(heldout)}."
        )
    failures = [row for row in heldout if row.get("success") is not True]
    if failures:
        raise RuntimeError(f"Held-out defective failures prevent reporting: {failures[:3]}")
    good_path = output_dir / "good_metrics.json"
    good_payload = json.loads(good_path.read_text(encoding="utf-8"))
    if int(good_payload["overall"]["sample_count"]) != 297 or int(
        good_payload["overall"]["success_count"]
    ) != 297:
        raise RuntimeError("Expected 297 / 297 successful defect-free samples.")

    all_sample_path = output_dir / "all_discovered_sample_metrics.csv"
    all_category_path = output_dir / "all_discovered_category_metrics.csv"
    all_aggregate_path = output_dir / "all_discovered_aggregate_metrics.json"
    if (output_dir / "sample_metrics.csv").is_file():
        shutil.copy2(output_dir / "sample_metrics.csv", all_sample_path)
    if (output_dir / "category_metrics.csv").is_file():
        shutil.copy2(output_dir / "category_metrics.csv", all_category_path)
    if (output_dir / "aggregate_metrics.json").is_file():
        shutil.copy2(output_dir / "aggregate_metrics.json", all_aggregate_path)

    _, group_rows, _ = build_aggregate_tables(heldout, BENCHMARK_CATEGORIES)
    for row in group_rows:
        tp, fp, fn = int(row["pooled_tp"]), int(row["pooled_fp"]), int(row["pooled_fn"])
        row["iou"] = float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0
    _atomic_write_csv(output_dir / "sample_metrics.csv", heldout)
    _atomic_write_csv(output_dir / "group_metrics.csv", group_rows)

    per_category, fixed = _category_balanced_metrics(heldout, good_payload)
    paper_category_rows = [
        {"category": category, **per_category[category]}
        for category in BENCHMARK_CATEGORIES
    ]
    paper_aggregate = {
        "benchmark_scope": "Main8 held-out excluding frozen Dev10",
        "aggregation": "sample macro within category, then equal category weight",
        "selected_categories": list(BENCHMARK_CATEGORIES),
        "heldout_defective_count": len(heldout),
        "defect_free_count": int(good_payload["overall"]["sample_count"]),
        "category_balanced_precision": fixed["precision"],
        "category_balanced_recall": fixed["recall"],
        "category_balanced_f1": fixed["f1"],
        "category_balanced_iou": fixed["iou"],
        "category_balanced_mean_fpr": fixed["mean_fpr"],
    }
    _atomic_write_csv(output_dir / "category_metrics.csv", paper_category_rows)
    _atomic_write_json(output_dir / "aggregate_metrics.json", paper_aggregate)
    full_rows = _read_sample_artifact_rows(FULL_CONTROL_DIR)
    full_heldout = [
        row
        for row in full_rows
        if (str(row.get("category")), str(row.get("sample_id"))) not in excluded
    ]
    full_good = json.loads(
        (FULL_CONTROL_DIR / "good_metrics.json").read_text(encoding="utf-8")
    )
    if len(full_heldout) != 598:
        raise RuntimeError(
            f"Frozen Full control has {len(full_heldout)} held-out rows; expected 598."
        )
    _, full = _category_balanced_metrics(full_heldout, full_good)
    validation_deltas = {
        metric: abs(float(full[metric]) - reference)
        for metric, reference in REFERENCE_FULL.items()
    }
    validation_passed = all(
        delta <= REFERENCE_TOLERANCE for delta in validation_deltas.values()
    )
    if not validation_passed:
        raise AssertionError(
            f"Full paper-row validation failed: {validation_deltas}"
        )
    paper = {
        "ablation_mode": MODE,
        "paper_name": "Fixed absolute-threshold verification",
        "selected_threshold": float(selection["selected_threshold"]),
        "threshold_frozen_before_heldout": True,
        "heldout_defective_count": len(heldout),
        "heldout_defective_success_count": len(heldout),
        "all_discovered_defective_count": len(all_rows),
        "excluded_development_group_count": 10,
        "excluded_development_variant_count": len(excluded),
        "defect_free_count": int(good_payload["overall"]["sample_count"]),
        "defect_free_success_count": int(good_payload["overall"]["success_count"]),
        "fixed_absolute": fixed,
        "full_control": full,
        "delta_fixed_minus_full": {
            metric: float(fixed[metric] - full[metric]) for metric in fixed
        },
        "per_category_metrics": per_category,
        "full_paper_row_validation": {
            "passed": validation_passed,
            "absolute_tolerance": REFERENCE_TOLERANCE,
            "reference": REFERENCE_FULL,
            "absolute_deltas": validation_deltas,
        },
        "heldout_exclusion_manifest": {
            "path": str(FULL_DEVELOPMENT_MANIFEST.resolve()),
            "sha256": exclusion_hash,
        },
        "development_manifest_sha256": selection["development_manifest_sha256"],
        "threshold_selection_sha256": _sha256(
            output_dir / "threshold_selection.json"
        ),
        "threshold_lock_sha256": _sha256(output_dir / "threshold_lock.json"),
        "production_config_hash": selection["production_config_hash"],
        "experiment_implementation_hash": selection[
            "experiment_implementation_hash"
        ],
    }
    _atomic_write_json(output_dir / "paper_heldout_metrics.json", paper)
    _write_report(output_dir=output_dir, selection=selection, paper=paper)
    return paper


def run_formal(
    *,
    output_dir: Path,
    dataset_root: Path,
    mvtec_root: Path,
    sample_workers: int,
    query_workers: int,
    resume: bool,
) -> dict[str, Any]:
    lock = verify_threshold_lock(output_dir)
    selection = json.loads(
        (output_dir / "threshold_selection.json").read_text(encoding="utf-8")
    )
    threshold = validate_threshold(float(lock["selected_threshold"]))
    if threshold != float(selection["selected_threshold"]):
        raise RuntimeError("Threshold lock and selection artifact disagree.")
    _write_progress(
        output_dir,
        phase="formal_defective",
        status="running",
        selected_threshold=threshold,
    )
    result = run_ablation_batch(
        ablation_mode=MODE,
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
        categories=BENCHMARK_CATEGORIES,
        output_dir=output_dir,
        config=DEFAULT_CONFIG,
        sample_workers=sample_workers,
        query_workers=query_workers,
        absolute_threshold=threshold,
        resume=resume,
        retry_failed=True,
        save_final_masks=False,
        run_good_samples=True,
        overwrite=False,
        show_progress=True,
    )
    _write_progress(
        output_dir,
        phase="paper_finalize",
        status="running",
        selected_threshold=threshold,
    )
    paper = finalize_paper_outputs(
        output_dir=output_dir,
        selection=selection,
        all_rows=result["sample_rows"],
    )
    formal_provenance = {
        "paper_evaluation_scope": "598 held-out defective plus 297 defect-free",
        "all_discovered_defective_processed": 618,
        "heldout_exclusion_manifest": paper["heldout_exclusion_manifest"],
        "threshold_frozen_before_heldout": True,
        "threshold_selection_sha256": paper["threshold_selection_sha256"],
        "threshold_lock_sha256": paper["threshold_lock_sha256"],
    }
    for name in ("ablation_metadata.json", "config.json"):
        path = output_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["formal_protocol"] = formal_provenance
        _atomic_write_json(path, payload)
    _write_progress(
        output_dir,
        phase="completed",
        status="completed",
        selected_threshold=threshold,
        processed_defective=618,
        heldout_defective=598,
        good_completed=297,
    )
    return paper


def _smoke_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    sample = SampleSpec(**payload["sample"])
    started = time.perf_counter()
    try:
        result = run_ablation_sample(
            sample.pc_path,
            sample.gt_path,
            ablation_mode=MODE,
            config=DEFAULT_CONFIG,
            query_workers=int(payload["query_workers"]),
            absolute_threshold=float(payload["threshold"]),
        )
        diagnostics = dict(result.ablation_metadata["diagnostics"])
        required_true = (
            "large_shallow_rejection_retained",
            "stress_veto_retained",
            "minimum_size_rejection_retained",
            "point_refinement_retained",
            "final_cleanup_retained",
            "biharmonic_restoration_retained",
            "local_control_computation_retained",
            "local_control_diagnostic_only",
        )
        if any(diagnostics.get(key) is not True for key in required_true):
            raise AssertionError(f"Smoke semantic diagnostics failed: {diagnostics}")
        if diagnostics.get("relative_normal_decision_consumed") is not False:
            raise AssertionError("Smoke unexpectedly consumed local-control ratios.")
        return {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "success",
            "success": True,
            "predicted_positive_points": int(np.sum(result.final_mask)),
            "working_point_count": int(result.final_mask.size),
            "metrics": dict(result.metrics),
            "diagnostics": diagnostics,
            "wall_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "failure",
            "success": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "wall_seconds": float(time.perf_counter() - started),
        }


def run_smoke(
    *,
    output_dir: Path,
    dataset_root: Path,
    mvtec_root: Path,
    sample_workers: int,
    query_workers: int,
    threshold: float,
) -> dict[str, Any]:
    tau = validate_threshold(threshold)
    specs, failures = discover_samples(
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
        categories=BENCHMARK_CATEGORIES,
    )
    if failures:
        raise RuntimeError(f"Smoke discovery failed: {failures}")
    first_by_category: dict[str, SampleSpec] = {}
    for sample in specs:
        first_by_category.setdefault(sample.category, sample)
    if tuple(first_by_category) != tuple(BENCHMARK_CATEGORIES):
        raise AssertionError("Smoke could not resolve one deterministic sample per Main8 category.")
    smoke_dir = output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    selected = [first_by_category[category] for category in BENCHMARK_CATEGORIES]
    _atomic_write_csv(
        smoke_dir / "smoke_manifest.csv",
        [dataclasses.asdict(sample) for sample in selected],
    )
    payloads = [
        {
            "sample": dataclasses.asdict(sample),
            "query_workers": query_workers,
            "threshold": tau,
        }
        for sample in selected
    ]
    rows: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> None:
        rows.append(row)
        _atomic_write_json(
            smoke_dir / "smoke_progress.json",
            {
                "status": "running",
                "discovered": len(selected),
                "completed": len(rows),
                "pending": len(selected) - len(rows),
                "updated_at_utc": _utc_now(),
            },
        )
        print(
            f"[smoke {len(rows):02d}/{len(selected):02d}] "
            f"{row['category']}/{row['sample_id']} {row['status']}",
            flush=True,
        )

    if sample_workers == 1:
        for payload in payloads:
            record(_smoke_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=sample_workers) as executor:
            futures = [executor.submit(_smoke_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                record(future.result())
    rows.sort(
        key=lambda row: BENCHMARK_CATEGORIES.index(str(row["category"]))
    )
    success_count = sum(row.get("success") is True for row in rows)
    summary = {
        "status": "completed" if success_count == len(rows) else "failed",
        "mode": MODE,
        "purpose": "end-to-end smoke only; threshold is not fitted or frozen",
        "smoke_threshold": tau,
        "smoke_threshold_used_for_selection": False,
        "sample_selection": "lexicographically first discovered defective sample per Main8 category",
        "category_count": len(BENCHMARK_CATEGORIES),
        "success_count": success_count,
        "failure_count": len(rows) - success_count,
        "samples": rows,
        "completed_at_utc": _utc_now(),
    }
    _atomic_write_json(smoke_dir / "smoke_summary.json", summary)
    _atomic_write_json(
        smoke_dir / "smoke_progress.json",
        {
            "status": summary["status"],
            "discovered": len(rows),
            "completed": len(rows),
            "pending": 0,
            "success": success_count,
            "failure": len(rows) - success_count,
            "updated_at_utc": _utc_now(),
        },
    )
    if success_count != len(rows):
        raise RuntimeError("One or more Main8 smoke samples failed.")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "fit", "formal", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--mvtec-root", type=Path, default=MVTEC_ROOT)
    parser.add_argument("--sample-workers", type=int, default=4)
    parser.add_argument("--query-workers", type=int, default=1)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--smoke-threshold",
        type=float,
        default=0.123456789,
        help="Smoke-only threshold; never used for fitting or formal evaluation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.sample_workers < 1:
        raise ValueError("sample_workers must be at least one.")
    if args.query_workers == 0 or args.query_workers < -1:
        raise ValueError("query_workers must be -1 or a positive integer.")
    output_dir = args.output_dir.resolve()
    if args.command == "smoke":
        summary = run_smoke(
            output_dir=output_dir,
            dataset_root=args.dataset_root,
            mvtec_root=args.mvtec_root,
            sample_workers=args.sample_workers,
            query_workers=args.query_workers,
            threshold=args.smoke_threshold,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return 0
    selection: dict[str, Any] | None = None
    if args.command in {"fit", "run"}:
        existing_lock = output_dir / "threshold_lock.json"
        if args.command == "run" and args.resume and existing_lock.is_file():
            # A resumed formal run must keep the original freeze byte-for-byte.
            verify_threshold_lock(output_dir)
            selection = json.loads(
                (output_dir / "threshold_selection.json").read_text(
                    encoding="utf-8"
                )
            )
        else:
            selection = fit_threshold(
                output_dir=output_dir,
                dataset_root=args.dataset_root,
                mvtec_root=args.mvtec_root,
                sample_workers=args.sample_workers,
                query_workers=args.query_workers,
                resume=args.resume,
            )
        if args.command == "fit":
            print(json.dumps(selection, indent=2, ensure_ascii=False, default=str))
            return 0
    if args.command in {"formal", "run"}:
        paper = run_formal(
            output_dir=output_dir,
            dataset_root=args.dataset_root,
            mvtec_root=args.mvtec_root,
            sample_workers=args.sample_workers,
            query_workers=args.query_workers,
            resume=args.resume,
        )
        print(json.dumps(paper, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
