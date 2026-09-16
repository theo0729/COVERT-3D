"""Parallel, resumable Main8-defective runner for isolated COVERT ablations."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import statistics
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

from ablations.ablation_modes import (  # noqa: E402
    SUPPORTED_ABLATION_MODES,
    get_ablation_spec,
)
from ablations.covert_ablation_sample import run_ablation_sample  # noqa: E402
from covert_batch import (  # noqa: E402
    DATASET_ROOT,
    MVTEC_ROOT,
    SampleSpec,
    _atomic_write_csv,
    _atomic_write_json,
    _interrupt_process_pool,
    build_good_metrics,
    build_aggregate_tables,
    discover_good_samples,
    discover_samples,
    implementation_hash as production_implementation_hash,
)
from covert_sample import (  # noqa: E402
    DEFAULT_CONFIG,
    CovertPipelineConfig,
    config_from_dict,
    config_hash,
    config_to_dict,
)
from experiments.benchmark_categories import BENCHMARK_CATEGORIES  # noqa: E402


# ============================================================
# ABLATION BATCH RUN CONFIG -- edit only ABLATION_MODE if desired
# ============================================================
ABLATION_MODE = "full"
ABLATIONS_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ABLATIONS_ROOT / ABLATION_MODE
CATEGORIES = BENCHMARK_CATEGORIES
SAMPLE_WORKERS = 4  # formal ablation within-mode sample parallelism
QUERY_WORKERS = 1
RUN_GOOD_SAMPLES = True
SAVE_FINAL_MASKS = False
RESUME = False
RETRY_FAILED = False
SHOW_PROGRESS = True


def ablation_implementation_hash() -> str:
    digest = hashlib.sha256()
    digest.update(production_implementation_hash().encode("ascii"))
    for path in sorted(ABLATIONS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_fingerprint(
    mode: str,
    config: CovertPipelineConfig,
    *,
    sample_workers: int,
    query_workers: int,
    implementation_hash_value: str,
    absolute_threshold: float | None = None,
) -> str:
    payload = {
        "ablation_mode": mode,
        "production_config_hash": config_hash(config),
        "ablation_implementation_hash": implementation_hash_value,
        "sample_workers": int(sample_workers),
        "query_workers": int(query_workers),
    }
    if absolute_threshold is not None:
        payload["absolute_threshold"] = float(absolute_threshold)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sample_dir(output_dir: Path, sample: SampleSpec) -> Path:
    return output_dir / "sample_artifacts" / sample.dataset / sample.category / sample.sample_id


def _read_resume_row(
    output_dir: Path,
    sample: SampleSpec,
    fingerprint: str,
    *,
    retry_failed: bool,
) -> dict[str, Any] | None:
    path = _sample_dir(output_dir, sample) / "metrics.json"
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("run_fingerprint") != fingerprint:
        return None
    if row.get("success") is True or not retry_failed:
        row["resumed"] = True
        return row
    return None


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    sample = SampleSpec(**payload["sample"])
    config = config_from_dict(payload["config"])
    started = time.perf_counter()
    try:
        result = run_ablation_sample(
            sample.pc_path,
            sample.gt_path,
            ablation_mode=str(payload["ablation_mode"]),
            config=config,
            query_workers=int(payload["query_workers"]),
            absolute_threshold=payload.get("absolute_threshold"),
        )
        row: dict[str, Any] = {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "success",
            "success": True,
            "ablation_mode": result.ablation_mode,
            **dict(result.metrics),
            "working_point_count": int(result.points.shape[0]),
            "predicted_positive_points": int(np.sum(result.final_mask)),
            "input_load_seconds": result.timings.input_load_seconds,
            "pipeline_seconds": result.timings.pipeline_seconds,
            "inference_total_seconds": result.timings.inference_total_seconds,
            "failure_type": "",
            "failure_message": "",
            "production_config_hash": result.production_config_hash,
            "ablation_implementation_hash": payload["implementation_hash"],
            "run_fingerprint": payload["run_fingerprint"],
            "semantic_diagnostics": json.dumps(
                result.ablation_metadata["diagnostics"], sort_keys=True
            ),
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
        artifact_dir = _sample_dir(Path(payload["output_dir"]), sample)
        if bool(payload["save_final_masks"]):
            artifact_dir.mkdir(parents=True, exist_ok=True)
            temporary = artifact_dir / f".final_mask.{os.getpid()}.{uuid.uuid4().hex}.npz"
            np.savez_compressed(temporary, final_mask=result.final_mask.astype(np.uint8))
            os.replace(temporary, artifact_dir / "final_mask.npz")
    except Exception as exc:
        row = {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "failure",
            "success": False,
            "ablation_mode": str(payload["ablation_mode"]),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "production_config_hash": config_hash(config),
            "ablation_implementation_hash": payload["implementation_hash"],
            "run_fingerprint": payload["run_fingerprint"],
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
    _atomic_write_json(_sample_dir(Path(payload["output_dir"]), sample) / "metrics.json", row)
    return row


def _good_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run one label-free good sample through the selected ablation mode."""

    sample = SampleSpec(**payload["sample"])
    config = config_from_dict(payload["config"])
    mode = get_ablation_spec(str(payload["ablation_mode"])).mode
    expected_hash = config_hash(config)
    started = time.perf_counter()
    try:
        if sample.sample_kind != "good" or sample.gt_path is not None:
            raise ValueError("Good worker requires sample_kind='good' and gt_path=None.")
        result = run_ablation_sample(
            sample.pc_path,
            gt_path=None,
            ablation_mode=mode,
            config=config,
            query_workers=int(payload["query_workers"]),
            absolute_threshold=payload.get("absolute_threshold"),
            output_dir=None,
            verbose=False,
        )
        final_mask = np.asarray(result.final_mask, dtype=bool).reshape(-1)
        if final_mask.size == 0:
            raise RuntimeError("COVERT ablation returned an empty final mask.")
        if result.gt_mask is not None or result.metrics:
            raise RuntimeError("Good inference unexpectedly performed GT evaluation.")
        if result.ablation_mode != mode:
            raise AssertionError("Good worker executed a different ablation mode.")
        if result.production_config_hash != expected_hash:
            raise AssertionError("Good worker production config hash changed.")
        fp = int(np.count_nonzero(final_mask))
        num_points = int(final_mask.size)
        row: dict[str, Any] = {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "success",
            "success": True,
            "ablation_mode": mode,
            "num_points": num_points,
            "fp": fp,
            "tn": int(num_points - fp),
            "fpr": float(fp / num_points),
            "input_load_seconds": float(result.timings.input_load_seconds),
            "downsampling_seconds": float(
                getattr(result.timings, "downsampling_seconds", 0.0)
            ),
            "algorithm_seconds_excluding_downsampling": float(
                getattr(
                    result.timings,
                    "algorithm_seconds_excluding_downsampling",
                    result.timings.pipeline_seconds,
                )
            ),
            "pipeline_seconds": float(result.timings.pipeline_seconds),
            "inference_total_seconds": float(result.timings.inference_total_seconds),
            "failure_type": "",
            "failure_message": "",
            "production_config_hash": result.production_config_hash,
            "ablation_implementation_hash": str(payload["implementation_hash"]),
            "run_fingerprint": str(payload["run_fingerprint"]),
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
    except Exception as exc:
        row = {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "failure",
            "success": False,
            "ablation_mode": mode,
            "num_points": None,
            "fp": None,
            "tn": None,
            "fpr": None,
            "input_load_seconds": None,
            "downsampling_seconds": None,
            "algorithm_seconds_excluding_downsampling": None,
            "pipeline_seconds": None,
            "inference_total_seconds": None,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "production_config_hash": expected_hash,
            "ablation_implementation_hash": str(payload["implementation_hash"]),
            "run_fingerprint": str(payload["run_fingerprint"]),
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
    return row


def _read_good_resume_rows(
    path: Path,
    specs: Sequence[SampleSpec],
    expected_fingerprint: str,
    *,
    retry_failed: bool,
) -> list[dict[str, Any]]:
    """Read resumable good rows from this mode's isolated good_metrics.json."""

    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    allowed = {
        (sample.dataset, sample.category, sample.sample_id) for sample in specs
    }
    rows: list[dict[str, Any]] = []
    for raw in payload.get("samples", []):
        if not isinstance(raw, Mapping):
            continue
        identity = (
            str(raw.get("dataset", "")),
            str(raw.get("category", "")),
            str(raw.get("sample_id", "")),
        )
        if identity not in allowed:
            continue
        if raw.get("run_fingerprint") != expected_fingerprint:
            continue
        if raw.get("success") is not True and retry_failed:
            continue
        row = dict(raw)
        row["resumed"] = True
        rows.append(row)
    return rows


def _run_good_batch(
    *,
    ablation_mode: str,
    dataset_root: str | Path,
    mvtec_root: str | Path,
    categories: Sequence[str],
    output_dir: Path,
    config: CovertPipelineConfig,
    sample_workers: int,
    query_workers: int,
    resume: bool,
    retry_failed: bool,
    show_progress: bool,
    implementation_hash_value: str,
    run_fingerprint: str,
    absolute_threshold: float | None,
) -> tuple[dict[str, Any], bool]:
    """Run a production-discovered good cohort after defective finalization."""

    mode = get_ablation_spec(ablation_mode).mode
    specs, discovery_failures = discover_good_samples(
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
        categories=categories,
    )
    result_path = output_dir / "good_metrics.json"
    rows = (
        _read_good_resume_rows(
            result_path,
            specs,
            run_fingerprint,
            retry_failed=retry_failed,
        )
        if resume
        else []
    )
    resumed = {
        (str(row["dataset"]), str(row["category"]), str(row["sample_id"]))
        for row in rows
    }
    pending = [
        sample
        for sample in specs
        if (sample.dataset, sample.category, sample.sample_id) not in resumed
    ]
    good_resumed_count = len(rows)

    good_started_epoch = time.time()
    good_started_at = datetime.now(timezone.utc).isoformat()

    def publish(status: str) -> dict[str, Any]:
        rows.sort(
            key=lambda row: (
                str(row.get("dataset", "")),
                str(row.get("category", "")),
                str(row.get("sample_id", "")),
            )
        )
        payload = build_good_metrics(
            rows,
            categories,
            discovery_failures=discovery_failures,
            status=status,
        )
        payload["ablation_mode"] = mode
        payload["production_config_hash"] = config_hash(config)
        payload["ablation_implementation_hash"] = implementation_hash_value
        payload["run_fingerprint"] = run_fingerprint
        _atomic_write_json(result_path, payload)
        _atomic_write_json(
            output_dir / "good_batch_progress.json",
            {
                "status": status,
                "ablation_mode": mode,
                "discovered": len(specs),
                "completed": len(rows),
                "resumed": good_resumed_count,
                "completed_this_run": max(len(rows) - good_resumed_count, 0),
                "pending": max(len(specs) - len(rows), 0),
                "success": sum(row.get("success") is True for row in rows),
                "failure": sum(row.get("success") is not True for row in rows),
                "started_at_utc": good_started_at,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": max(time.time() - good_started_epoch, 0.0),
            },
        )
        return payload

    publish("running")
    worker_payloads = [
        {
            "sample": dataclasses.asdict(sample),
            "config": config_to_dict(config),
            "ablation_mode": mode,
            "query_workers": int(query_workers),
            "implementation_hash": implementation_hash_value,
            "run_fingerprint": run_fingerprint,
            "absolute_threshold": absolute_threshold,
        }
        for sample in pending
    ]
    completed = len(rows)

    def record(row: dict[str, Any]) -> None:
        nonlocal completed
        rows.append(row)
        completed += 1
        publish("running")
        if show_progress:
            print(
                f"[COVERT ablation good {completed:03d}/{len(specs):03d}] "
                f"{row['category']}/{row['sample_id']} {row['status']}",
                flush=True,
            )

    interrupted = False
    executor: ProcessPoolExecutor | None = None
    try:
        if sample_workers == 1:
            for payload in worker_payloads:
                record(_good_worker(payload))
        elif worker_payloads:
            executor = ProcessPoolExecutor(max_workers=sample_workers)
            futures = [executor.submit(_good_worker, payload) for payload in worker_payloads]
            for future in as_completed(futures):
                record(future.result())
            executor.shutdown(wait=True)
            executor = None
    except KeyboardInterrupt:
        interrupted = True
        if executor is not None:
            _interrupt_process_pool(executor)
            executor = None
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return publish("interrupted" if interrupted else "completed"), interrupted


def _add_iou(row: dict[str, Any]) -> dict[str, Any]:
    tp = int(row.get("pooled_tp", 0))
    fp = int(row.get("pooled_fp", 0))
    fn = int(row.get("pooled_fn", 0))
    denominator = tp + fp + fn
    row["iou"] = float(tp / denominator) if denominator else 0.0
    return row


def _enrich_aggregate_iou(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    if {"pooled_tp", "pooled_fp", "pooled_fn"} <= payload.keys():
        _add_iou(payload)
    for value in payload.values():
        _enrich_aggregate_iou(value)


def _assert_output_safe(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    entries = list(output_dir.iterdir())
    if entries and not resume and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty ablation output {output_dir}. "
            "Use --resume or an explicit --overwrite."
        )


def run_ablation_batch(
    *,
    ablation_mode: str = ABLATION_MODE,
    dataset_root: str | Path = DATASET_ROOT,
    mvtec_root: str | Path = MVTEC_ROOT,
    categories: Sequence[str] = CATEGORIES,
    output_dir: str | Path | None = None,
    config: CovertPipelineConfig = DEFAULT_CONFIG,
    sample_workers: int = SAMPLE_WORKERS,
    query_workers: int = QUERY_WORKERS,
    absolute_threshold: float | None = None,
    resume: bool = RESUME,
    retry_failed: bool = RETRY_FAILED,
    save_final_masks: bool = SAVE_FINAL_MASKS,
    run_good_samples: bool = RUN_GOOD_SAMPLES,
    overwrite: bool = False,
    show_progress: bool = SHOW_PROGRESS,
) -> dict[str, Any]:
    """Run defective samples, then an optional isolated good-sample cohort."""

    mode = get_ablation_spec(ablation_mode).mode
    if mode == "fixed_absolute_threshold_verification":
        if absolute_threshold is None:
            raise ValueError(
                "fixed_absolute_threshold_verification requires "
                "absolute_threshold."
            )
        if not np.isfinite(float(absolute_threshold)) or float(absolute_threshold) < 0:
            raise ValueError("absolute_threshold must be finite and non-negative.")
        absolute_threshold = float(absolute_threshold)
    elif absolute_threshold is not None:
        raise ValueError(
            "absolute_threshold is valid only for "
            "fixed_absolute_threshold_verification."
        )
    if sample_workers < 1:
        raise ValueError("sample_workers must be at least one.")
    if query_workers == 0 or query_workers < -1:
        raise ValueError("query_workers must be -1 or a positive integer.")
    destination = Path(output_dir) if output_dir is not None else ABLATIONS_ROOT / mode
    _assert_output_safe(destination, resume=resume, overwrite=overwrite)
    destination.mkdir(parents=True, exist_ok=True)

    implementation = ablation_implementation_hash()
    fingerprint = _run_fingerprint(
        mode,
        config,
        sample_workers=sample_workers,
        query_workers=query_workers,
        implementation_hash_value=implementation,
        absolute_threshold=absolute_threshold,
    )
    spec = get_ablation_spec(mode)
    metadata = {
        "ablation_mode": mode,
        "paper_name": spec.paper_name,
        "changed_mechanism": spec.changed_module,
        "ablation_spec": spec.to_dict(),
        "base_production_config_hash": config_hash(config),
        "production_implementation_hash": production_implementation_hash(),
        "ablation_implementation_hash": implementation,
        "run_fingerprint": fingerprint,
        "output_dir": str(destination.resolve()),
        "benchmark_scope": "Main8 defective plus separate good cohort when enabled",
        "run_good_samples": bool(run_good_samples),
        "parameters_tuned_per_ablation": False,
    }
    if mode == "wo_local_control_calibration":
        metadata.update(
            {
                "internal_disabling_sentinel": {
                    "field": "gtr.adaptive_relative_normal_max_ratio",
                    "value": -1.0,
                    "purpose": "disable candidate-to-control relative rejection only",
                    "tuned_hyperparameter": False,
                },
                "expected_semantic_diagnostics": dict(spec.expected_diagnostics),
                "category_specific_tuning": False,
            }
        )
    elif mode == "fixed_absolute_threshold_verification":
        metadata["parameters_tuned_per_ablation"] = True
        metadata.update(
            {
                "classifier_strategy": mode,
                "absolute_statistic": "candidate_D_p90",
                "absolute_threshold": absolute_threshold,
                "absolute_rule": "D_p90 < tau_abs",
                "absolute_keep_equal": True,
                "threshold_scope": "single_global_frozen_development_threshold",
                "parameter_tuning_scope": (
                    "four frozen physical development groups / eight variants"
                ),
                "relative_normal_decision_consumed": False,
                "local_control_computation": "diagnostics_only",
                "large_shallow_rejection_retained": True,
                "category_specific_tuning": False,
                "expected_semantic_diagnostics": dict(
                    spec.expected_diagnostics
                ),
            }
        )
    _atomic_write_json(destination / "ablation_metadata.json", metadata)
    _atomic_write_json(
        destination / "config.json",
        {
            **metadata,
            "source": "covert_sample.DEFAULT_CONFIG",
            "config": config_to_dict(config),
            "categories": list(categories),
            "execution": {
                "sample_workers": int(sample_workers),
                "query_workers": int(query_workers),
                "save_final_masks": bool(save_final_masks),
                "run_good_samples": bool(run_good_samples),
                "absolute_threshold": absolute_threshold,
            },
        },
    )

    specs, discovery_failures = discover_samples(
        dataset_root=dataset_root, mvtec_root=mvtec_root, categories=tuple(categories)
    )
    rows: list[dict[str, Any]] = []
    pending: list[SampleSpec] = []
    for sample in specs:
        previous = (
            _read_resume_row(destination, sample, fingerprint, retry_failed=retry_failed)
            if resume
            else None
        )
        if previous is None:
            pending.append(sample)
        else:
            rows.append(previous)
    resumed_count = len(rows)

    batch_started_epoch = time.time()
    batch_started_at = datetime.now(timezone.utc).isoformat()

    def publish_progress(status: str) -> None:
        _atomic_write_json(
            destination / "batch_progress.json",
            {
                "status": status,
                "ablation_mode": mode,
                "discovered": len(specs),
                "completed": len(rows),
                "resumed": resumed_count,
                "completed_this_run": max(len(rows) - resumed_count, 0),
                "pending": max(len(specs) - len(rows), 0),
                "success": sum(row.get("success") is True for row in rows),
                "failure": sum(row.get("success") is not True for row in rows),
                "started_at_utc": batch_started_at,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": max(time.time() - batch_started_epoch, 0.0),
            },
        )

    publish_progress("running")
    payloads = [
        {
            "sample": dataclasses.asdict(sample),
            "config": config_to_dict(config),
            "ablation_mode": mode,
            "output_dir": str(destination),
            "query_workers": int(query_workers),
            "save_final_masks": bool(save_final_masks),
            "implementation_hash": implementation,
            "run_fingerprint": fingerprint,
            "absolute_threshold": absolute_threshold,
        }
        for sample in pending
    ]
    batch_start = time.perf_counter()
    if sample_workers == 1:
        for payload in payloads:
            row = _worker(payload)
            rows.append(row)
            publish_progress("running")
            if show_progress:
                print(f"[{len(rows):03d}/{len(specs):03d}] {row['category']}/{row['sample_id']} {row['status']}", flush=True)
    elif payloads:
        with ProcessPoolExecutor(max_workers=sample_workers) as executor:
            future_map = {executor.submit(_worker, payload): payload for payload in payloads}
            for future in as_completed(future_map):
                row = future.result()
                rows.append(row)
                publish_progress("running")
                if show_progress:
                    print(f"[{len(rows):03d}/{len(specs):03d}] {row['category']}/{row['sample_id']} {row['status']}", flush=True)
    wall_seconds = float(time.perf_counter() - batch_start)

    rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("category", "")), str(row.get("sample_id", ""))))
    category_rows, group_rows, aggregate = build_aggregate_tables(rows, tuple(categories))
    for row in category_rows:
        _add_iou(row)
    for row in group_rows:
        _add_iou(row)
    _enrich_aggregate_iou(aggregate)
    failures = [dict(row) for row in discovery_failures] + [
        dict(row) for row in rows if row.get("success") is not True
    ]
    successful = [row for row in rows if row.get("success") is True]
    runtimes = [float(row["inference_total_seconds"]) for row in successful]
    runtime = {
        "ablation_mode": mode,
        "sample_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "sample_workers": int(sample_workers),
        "query_workers": int(query_workers),
        "batch_wall_seconds": wall_seconds,
        "mean_inference_seconds": float(statistics.mean(runtimes)) if runtimes else 0.0,
        "median_inference_seconds": float(statistics.median(runtimes)) if runtimes else 0.0,
        "production_config_hash": config_hash(config),
        "ablation_implementation_hash": implementation,
        "run_fingerprint": fingerprint,
    }
    _atomic_write_csv(destination / "sample_metrics.csv", rows)
    _atomic_write_csv(destination / "category_metrics.csv", category_rows)
    _atomic_write_csv(destination / "group_metrics.csv", group_rows)
    _atomic_write_csv(destination / "failures.csv", failures)
    _atomic_write_json(destination / "aggregate_metrics.json", aggregate)
    _atomic_write_json(destination / "runtime_summary.json", runtime)
    publish_progress("completed")
    result_payload = {
        "sample_rows": rows,
        "category_rows": category_rows,
        "group_rows": group_rows,
        "aggregate": aggregate,
        "runtime": runtime,
        "failures": failures,
        "ablation_metadata": metadata,
    }
    # Match production ordering: the defective reports above are fully finalized
    # before good discovery or execution begins. Any earlier KeyboardInterrupt
    # propagates before this point, so an interrupted defective run cannot start good.
    if run_good_samples:
        good_metrics, good_interrupted = _run_good_batch(
            ablation_mode=mode,
            dataset_root=dataset_root,
            mvtec_root=mvtec_root,
            categories=tuple(categories),
            output_dir=destination,
            config=config,
            sample_workers=sample_workers,
            query_workers=query_workers,
            resume=resume,
            retry_failed=retry_failed,
            show_progress=show_progress,
            implementation_hash_value=implementation,
            run_fingerprint=fingerprint,
            absolute_threshold=absolute_threshold,
        )
        result_payload["good_metrics"] = good_metrics
        result_payload["good_interrupted"] = bool(good_interrupted)
    return result_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-mode", choices=SUPPORTED_ABLATION_MODES, default=ABLATION_MODE
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--mvtec-root", type=Path, default=MVTEC_ROOT)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES),
        choices=BENCHMARK_CATEGORIES,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-workers", type=int, default=SAMPLE_WORKERS)
    parser.add_argument("--query-workers", type=int, default=QUERY_WORKERS)
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        help="Frozen tau_abs for fixed_absolute_threshold_verification.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME)
    parser.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=RETRY_FAILED)
    parser.add_argument("--save-final-masks", action=argparse.BooleanOptionalAction, default=SAVE_FINAL_MASKS)
    parser.add_argument(
        "--good-samples",
        action=argparse.BooleanOptionalAction,
        default=RUN_GOOD_SAMPLES,
        help="Run the separate Main8 good-sample FPR cohort after defective finalization.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=SHOW_PROGRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_ablation_batch(
            ablation_mode=args.ablation_mode,
            dataset_root=args.dataset_root,
            mvtec_root=args.mvtec_root,
            categories=tuple(args.categories),
            output_dir=args.output_dir,
            config=DEFAULT_CONFIG,
            sample_workers=args.sample_workers,
            query_workers=args.query_workers,
            absolute_threshold=args.absolute_threshold,
            resume=args.resume,
            retry_failed=args.retry_failed,
            save_final_masks=args.save_final_masks,
            run_good_samples=args.good_samples,
            overwrite=args.overwrite,
            show_progress=args.progress,
        )
    except KeyboardInterrupt:
        return 130
    if result.get("good_interrupted") is True:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
