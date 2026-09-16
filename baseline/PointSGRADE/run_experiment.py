"""Run PointSGRADE on selected frozen benchmark-cache samples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.PointSGRADE.pointsgrade import (
    POINTSGRADE_SOLVER_FILE,
    audit_pointsgrade_dependencies,
    hash_points,
    run_pointsgrade_baseline,
)
from baseline.common import (
    collect_environment,
    compute_binary_metrics,
    discover_defect_samples,
    list_cache_files,
    load_cache_file,
    run_good_evaluation,
    stable_config_hash,
    write_csv,
    write_csv_atomic,
)
from baseline.common.parallel import (
    config_without_execution_settings,
    iter_process_results,
    limited_inner_threads,
    limited_numba_threads,
    validate_execution_settings,
)


POINTSGRADE_NUMBA_THREAD_REASON = (
    "Numba was restricted to one thread because the released PointSGRADE "
    "reverse-mapping kernel contains overlapping indexed accumulations under "
    "prange; MKL/BLAS threading remained native."
)


PER_SAMPLE_FIELDS = [
    "category", "sample_id", "base_sample_id", "is_cut", "method", "status",
    "error", "first_attempt_error", "attempts", "traceback_path", "cache_path",
    "num_points", "points_dtype", "coordinate_min", "coordinate_max",
    "num_gt_positive", "num_pred_positive", "tp", "fp", "fn", "tn",
    "precision", "recall", "f1", "iou", "runtime_sec",
    "initialization_sec", "graph_matrix_sec", "optimization_sec",
    "vendor_stage_sum_sec", "predicted_anomaly_ratio", "random_state",
    "repetitions", "prediction_identical_across_repetitions",
    "numba_threads", "blas_openmp_threads",
    "input_points_sha256", "cache_points_sha256", "point_order_preserved",
    "config_hash", "preprocess_config_hash", "fps_selected_index_hash",
    "prediction_path",
    "sample_workers", "inner_thread_limit",
]

SUMMARY_FIELDS = [
    "method", "scope", "num_expected_samples", "num_samples",
    "num_failed_samples", "precision_mean", "precision_std", "recall_mean",
    "recall_std", "f1_mean", "f1_std", "iou_mean", "iou_std",
    "runtime_mean_sec", "runtime_std_sec", "runtime_median_sec",
    "runtime_min_sec", "runtime_max_sec", "runtime_p50_sec", "runtime_p90_sec",
    "initialization_mean_sec", "graph_matrix_mean_sec", "optimization_mean_sec",
    "vendor_stage_sum_mean_sec", "predicted_anomaly_fraction_mean", "config_hash",
]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _validate_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "paths", "pointsgrade", "evaluation"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing config mapping: {section}")
    experiment = config["experiment"]
    if int(experiment.get("num_working_points", 0)) <= 0:
        raise ValueError("experiment.num_working_points must be positive.")
    categories = experiment.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("experiment.categories must be a non-empty list.")
    validate_execution_settings(
        int(config["evaluation"].get("sample_workers", 1)),
        config["evaluation"].get("inner_thread_limit", 1),
    )
    if int(config["evaluation"].get("numba_threads", 1)) != 1:
        raise ValueError(
            "PointSGRADE final numba_threads must be 1 for race-safe execution."
        )


def _method_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "lambda0", "epsilon", "num_neighbor", "num_neighbor_max",
        "threshold_angle", "threshold_dist", "sigma", "random_state",
    )
    section = config["pointsgrade"]
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"Missing PointSGRADE parameters: {missing}")
    return {key: section[key] for key in keys}


def _save_prediction(
    path: Path,
    *,
    cache: Mapping[str, Any],
    pred_mask: np.ndarray,
    runtime_sec: float,
    diagnostics: Mapping[str, Any],
    config_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_mask=np.asarray(pred_mask, dtype=np.uint8),
        method=np.asarray("PointSGRADE"),
        sample_id=np.asarray(cache["sample_id"]),
        base_sample_id=np.asarray(_base_sample_id(str(cache["sample_id"]))),
        is_cut=np.asarray(_is_cut(str(cache["sample_id"]))),
        category=np.asarray(cache["category"]),
        runtime_sec=np.asarray(runtime_sec, dtype=np.float64),
        config_hash=np.asarray(config_hash),
        preprocess_config_hash=np.asarray(cache["preprocess_config_hash"]),
        fps_selected_index_hash=np.asarray(cache["fps_selected_index_hash"]),
        input_points_sha256=np.asarray(diagnostics["input_points_sha256"]),
        numba_threads=np.asarray(
            int(diagnostics["execution_controls"]["numba_threads"])
        ),
        blas_openmp_threads=np.asarray(
            int(diagnostics["execution_controls"]["blas_openmp_threads"])
        ),
        diagnostics_json=np.asarray(
            json.dumps(dict(diagnostics), ensure_ascii=False, sort_keys=True)
        ),
    )


def _mean_std(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not values.size:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def _base_sample_id(sample_id: str) -> str:
    return sample_id[:-4] if sample_id.endswith("_cut") else sample_id


def _is_cut(sample_id: str) -> bool:
    return sample_id.endswith("_cut")


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    scope: str,
    expected_samples: int,
    config_hash: str,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["status"] == "ok" and (scope == "overall" or row["category"] == scope)
    ]
    failures = [
        row for row in rows
        if row["status"] != "ok" and (scope == "overall" or row["category"] == scope)
    ]
    output: dict[str, Any] = {
        "method": "PointSGRADE", "scope": scope,
        "num_expected_samples": expected_samples, "num_samples": len(selected),
        "num_failed_samples": len(failures), "config_hash": config_hash,
    }
    for field in ("precision", "recall", "f1", "iou", "runtime_sec"):
        mean, std = _mean_std(selected, field)
        label = "runtime" if field == "runtime_sec" else field
        suffix = "_sec" if field == "runtime_sec" else ""
        output[f"{label}_mean{suffix}"] = mean
        output[f"{label}_std{suffix}"] = std
    runtime = np.asarray(
        [float(row["runtime_sec"]) for row in selected], dtype=np.float64
    )
    if runtime.size:
        output.update(
            {
                "runtime_median_sec": float(np.median(runtime)),
                "runtime_min_sec": float(np.min(runtime)),
                "runtime_max_sec": float(np.max(runtime)),
                "runtime_p50_sec": float(np.percentile(runtime, 50)),
                "runtime_p90_sec": float(np.percentile(runtime, 90)),
            }
        )
    else:
        output.update(
            {
                key: float("nan")
                for key in (
                    "runtime_median_sec", "runtime_min_sec", "runtime_max_sec",
                    "runtime_p50_sec", "runtime_p90_sec",
                )
            }
        )
    for source, target in (
        ("initialization_sec", "initialization_mean_sec"),
        ("graph_matrix_sec", "graph_matrix_mean_sec"),
        ("optimization_sec", "optimization_mean_sec"),
        ("vendor_stage_sum_sec", "vendor_stage_sum_mean_sec"),
        ("predicted_anomaly_ratio", "predicted_anomaly_fraction_mean"),
    ):
        output[target] = _mean_std(selected, source)[0]
    return output


def _select_cache_files(
    cache_files: Sequence[Path],
    categories: Sequence[str],
    requested_samples: Sequence[str],
    limit: int | None,
) -> list[Path]:
    if requested_samples and limit is not None:
        raise ValueError("Use either --sample or --limit, not both.")
    if requested_samples:
        by_key = {
            f"{path.parent.name.lower()}/{path.stem}": path for path in cache_files
        }
        selected = []
        for raw in requested_samples:
            key = raw.replace("\\", "/").strip()
            if "/" not in key:
                raise ValueError("--sample must use CATEGORY/SAMPLE_ID syntax.")
            category, sample_id = key.split("/", 1)
            normalized = f"{category.lower()}/{sample_id}"
            if normalized not in by_key:
                raise FileNotFoundError(f"Cached sample not found: {normalized}")
            selected.append(by_key[normalized])
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate --sample selections are not allowed.")
        return selected
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        selected = []
        for category in categories:
            selected.extend(
                [path for path in cache_files if path.parent.name == category][:limit]
            )
        return selected
    return list(cache_files)


def _run_repetitions(
    points: np.ndarray,
    parameters: Mapping[str, Any],
    repetitions: int,
    inner_thread_limit: int | None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    results = [
        run_pointsgrade_baseline(
            points,
            **parameters,
            inner_thread_limit=inner_thread_limit,
        )
        for _ in range(repetitions)
    ]
    first_mask = np.asarray(results[0].pred_mask, dtype=bool)
    identical = all(
        np.array_equal(first_mask, np.asarray(result.pred_mask, dtype=bool))
        for result in results[1:]
    )
    if not identical:
        raise RuntimeError(
            "PointSGRADE predictions differ across identical seeded repetitions."
        )
    runtime_values = [float(result.runtime_sec) for result in results]
    stage_keys = (
        "initialization_sec", "graph_matrix_sec", "optimization_sec",
        "vendor_stage_sum_sec", "wall_total_sec",
    )
    diagnostics = dict(results[0].diagnostics)
    diagnostics["reproducibility"] = {
        "repetitions": repetitions,
        "prediction_identical": identical,
        "runtime_sec_by_repetition": runtime_values,
    }
    for key in stage_keys:
        values = [float(result.diagnostics[key]) for result in results]
        diagnostics[key] = float(np.mean(values))
        diagnostics[f"{key}_by_repetition"] = values
    return first_mask, float(np.mean(runtime_values)), diagnostics


def _execute_cache_sample(
    cache_path: Path,
    *,
    num_working_points: int,
    parameters: Mapping[str, Any],
    repetitions: int,
    inner_thread_limit: int | None,
) -> tuple[dict[str, Any], np.ndarray, float, dict[str, Any], dict[str, Any], str]:
    """Load and execute one sample, enforcing all common-input invariants."""

    cache = load_cache_file(cache_path, expected_num_points=num_working_points)
    cache_points_hash = hash_points(cache["points"])
    pred_mask, runtime_sec, diagnostics = _run_repetitions(
        cache["points"],
        parameters,
        repetitions,
        inner_thread_limit,
    )
    if pred_mask.shape != cache["gt_mask"].shape:
        raise RuntimeError(
            f"PointSGRADE output shape mismatch: {pred_mask.shape} vs "
            f"{cache['gt_mask'].shape}"
        )
    if not np.all(np.isfinite(np.asarray(pred_mask, dtype=np.float64))):
        raise RuntimeError("PointSGRADE prediction contains NaN or Inf.")
    if diagnostics["input_points_sha256"] != cache_points_hash:
        raise RuntimeError("PointSGRADE did not receive the exact cached point array.")
    metrics = compute_binary_metrics(pred_mask, cache["gt_mask"])
    return cache, pred_mask, runtime_sec, diagnostics, metrics, cache_points_hash


def _pointsgrade_final_sample_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Retry one PointSGRADE sample inside its bounded worker process."""

    cache_path = Path(str(task["cache_path"]))
    messages: list[str] = []
    traces: list[str] = []
    for attempt in (1, 2):
        try:
            inner_thread_limit = task["inner_thread_limit"]
            numba_threads = int(task["numba_threads"])
            with limited_inner_threads(inner_thread_limit), limited_numba_threads(
                numba_threads
            ):
                payload = _execute_cache_sample(
                    cache_path,
                    num_working_points=int(task["num_working_points"]),
                    parameters=task["parameters"],
                    repetitions=int(task["repetitions"]),
                    inner_thread_limit=inner_thread_limit,
                )
            execution_controls = dict(payload[3]["execution_controls"])
            execution_controls.update(
                {
                    "numba_threads": numba_threads,
                    "reason": POINTSGRADE_NUMBA_THREAD_REASON,
                }
            )
            payload[3]["execution_controls"] = execution_controls
            return {
                "status": "ok",
                "payload": payload,
                "attempts": attempt,
                "first_attempt_error": messages[0] if messages else "",
                "traceback_text": "\n\n".join(traces),
            }
        except BaseException as exc:
            messages.append(f"{type(exc).__name__}: {exc}")
            traces.append(f"ATTEMPT {attempt}\n{traceback.format_exc()}")
    return {
        "status": "error",
        "error": messages[-1],
        "first_attempt_error": messages[0],
        "attempts": 2,
        "traceback_text": "\n\n".join(traces),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    parser.add_argument(
        "--sample", action="append", default=[],
        help="Run one explicit cached sample as CATEGORY/SAMPLE_ID; repeatable.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run at most N cached samples per category (smoke tests only).",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-full-benchmark", action="store_true",
        help="Required when neither --sample nor --limit is supplied.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.sample and args.limit is None and not args.allow_full_benchmark:
        raise RuntimeError(
            "Refusing an accidental full PointSGRADE run. Select --sample/--limit "
            "or explicitly pass --allow-full-benchmark."
        )
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive.")

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    _validate_config(config)
    experiment = config["experiment"]
    evaluation = config["evaluation"]
    parameters = _method_parameters(config)
    categories = [str(value).strip().lower() for value in experiment["categories"]]
    num_working_points = int(experiment["num_working_points"])
    cache_dir = _resolve_project_path(config["paths"]["cache_dir"])
    real3d_root = _resolve_project_path(config["paths"]["real3d_root"])
    mvtec_root = _resolve_project_path(config["paths"]["mvtec_root"])
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _resolve_project_path(config["paths"]["output_dir"])
    )
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite or choose --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dependency_audit = audit_pointsgrade_dependencies()
    if dependency_audit["status"] != "ok":
        raise RuntimeError(
            f"Missing PointSGRADE dependencies: {dependency_audit['missing_required']}"
        )
    sample_workers, inner_thread_limit = validate_execution_settings(
        int(evaluation.get("sample_workers", 2)),
        evaluation.get("inner_thread_limit", 1),
    )
    numba_threads = int(evaluation.get("numba_threads", 1))
    config_hash = stable_config_hash(config_without_execution_settings(config))
    all_cache_files = list_cache_files(cache_dir, categories)
    cache_files = _select_cache_files(
        all_cache_files, categories, args.sample, args.limit
    )
    counts = Counter(path.parent.name for path in cache_files)
    discovered_counts = Counter(
        str(sample["category"])
        for sample in discover_defect_samples(
            real3d_root, categories, mvtec_root=mvtec_root
        )
    )
    if (
        bool(evaluation.get("require_complete_cache", True))
        and not args.sample and args.limit is None
    ):
        for category in categories:
            if counts.get(category, 0) != discovered_counts.get(category, 0):
                raise RuntimeError(
                    f"Incomplete cache for {category}: {counts.get(category, 0)}"
                )

    resolved_config = dict(config)
    resolved_config["resolved"] = {
        "project_root": str(PROJECT_ROOT), "config_path": str(config_path),
        "cache_dir": str(cache_dir), "output_dir": str(output_dir),
        "real3d_root": str(real3d_root), "mvtec_root": str(mvtec_root),
        "config_hash": config_hash, "selected_cache_files": len(cache_files),
        "selected_samples": [f"{path.parent.name}/{path.stem}" for path in cache_files],
        "limit_per_category": args.limit, "repetitions": args.repetitions,
        "full_benchmark_explicitly_allowed": bool(args.allow_full_benchmark),
        "sample_workers": sample_workers,
        "inner_thread_limit": inner_thread_limit,
        "numba_threads": numba_threads,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    environment = collect_environment()
    environment["pointsgrade_dependency_audit"] = dependency_audit
    environment["pointSGRADE_solver_file"] = str(POINTSGRADE_SOLVER_FILE)
    environment["pointSGRADE_execution_controls"] = {
        "sample_workers": sample_workers,
        "numba_threads": numba_threads,
        "inner_thread_limit": inner_thread_limit,
        "reason": POINTSGRADE_NUMBA_THREAD_REASON,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    save_predictions = bool(evaluation.get("save_predictions", True))
    run_started = time.perf_counter()
    tasks = [
        {
            "cache_path": str(cache_path),
            "num_working_points": num_working_points,
            "parameters": parameters,
            "repetitions": args.repetitions,
            "inner_thread_limit": inner_thread_limit,
            "numba_threads": numba_threads,
        }
        for cache_path in cache_files
    ]
    completed = iter_process_results(
        tasks, _pointsgrade_final_sample_worker, sample_workers=sample_workers
    )
    for index, (task, worker_result, process_error) in enumerate(completed, start=1):
        cache_path = Path(str(task["cache_path"]))
        traceback_path = (
            output_dir / "errors" / cache_path.parent.name / f"{cache_path.stem}.txt"
        )
        traceback_text = (
            str(worker_result.get("traceback_text", ""))
            if worker_result is not None else ""
        )
        if traceback_text:
            traceback_path.parent.mkdir(parents=True, exist_ok=True)
            traceback_path.write_text(traceback_text, encoding="utf-8")
        if worker_result is not None and worker_result["status"] == "ok":
            cache, pred_mask, runtime_sec, diagnostics, metrics, cache_points_hash = (
                worker_result["payload"]
            )
            prediction_path = (
                output_dir / "predictions" / cache["category"]
                / f"{cache['sample_id']}.npz"
            )
            if save_predictions:
                _save_prediction(
                    prediction_path, cache=cache, pred_mask=pred_mask,
                    runtime_sec=runtime_sec, diagnostics=diagnostics,
                    config_hash=config_hash,
                )
            points = cache["points"]
            rows.append(
                {
                    "category": cache["category"], "sample_id": cache["sample_id"],
                    "base_sample_id": _base_sample_id(str(cache["sample_id"])),
                    "is_cut": _is_cut(str(cache["sample_id"])),
                    "method": "PointSGRADE", "status": "ok", "error": "",
                    "first_attempt_error": worker_result["first_attempt_error"],
                    "attempts": int(worker_result["attempts"]),
                    "traceback_path": str(traceback_path) if traceback_text else "",
                    "cache_path": str(cache_path), "num_points": int(points.shape[0]),
                    "points_dtype": str(points.dtype),
                    "coordinate_min": json.dumps(points.min(axis=0).tolist()),
                    "coordinate_max": json.dumps(points.max(axis=0).tolist()),
                    "num_gt_positive": int(np.count_nonzero(cache["gt_mask"])),
                    "num_pred_positive": int(np.count_nonzero(pred_mask)), **metrics,
                    "runtime_sec": runtime_sec,
                    "initialization_sec": diagnostics["initialization_sec"],
                    "graph_matrix_sec": diagnostics["graph_matrix_sec"],
                    "optimization_sec": diagnostics["optimization_sec"],
                    "vendor_stage_sum_sec": diagnostics["vendor_stage_sum_sec"],
                    "predicted_anomaly_ratio": float(np.mean(pred_mask)),
                    "random_state": int(parameters["random_state"]),
                    "repetitions": args.repetitions,
                    "prediction_identical_across_repetitions": diagnostics[
                        "reproducibility"
                    ]["prediction_identical"],
                    "numba_threads": diagnostics["execution_controls"]["numba_threads"],
                    "blas_openmp_threads": diagnostics["execution_controls"][
                        "blas_openmp_threads"
                    ],
                    "input_points_sha256": diagnostics["input_points_sha256"],
                    "cache_points_sha256": cache_points_hash,
                    "point_order_preserved": diagnostics["point_order_preserved"],
                    "config_hash": config_hash,
                    "preprocess_config_hash": cache["preprocess_config_hash"],
                    "fps_selected_index_hash": cache["fps_selected_index_hash"],
                    "prediction_path": str(prediction_path) if save_predictions else "",
                    "sample_workers": sample_workers,
                    "inner_thread_limit": inner_thread_limit,
                }
            )
        else:
            sample_id = cache_path.stem
            error_text = (
                str(worker_result["error"])
                if worker_result is not None
                else f"{type(process_error).__name__}: {process_error}"
            )
            rows.append(
                {
                    "category": cache_path.parent.name,
                    "sample_id": sample_id,
                    "base_sample_id": _base_sample_id(sample_id),
                    "is_cut": _is_cut(sample_id),
                    "method": "PointSGRADE", "status": "error",
                    "error": error_text,
                    "first_attempt_error": (
                        str(worker_result.get("first_attempt_error", ""))
                        if worker_result is not None else ""
                    ),
                    "attempts": int(worker_result.get("attempts", 1)) if worker_result is not None else 1,
                    "traceback_path": str(traceback_path),
                    "cache_path": str(cache_path), "config_hash": config_hash,
                    "random_state": int(parameters["random_state"]),
                    "repetitions": args.repetitions,
                    "numba_threads": numba_threads, "blas_openmp_threads": "",
                    "sample_workers": sample_workers,
                    "inner_thread_limit": inner_thread_limit,
                }
            )
        rows.sort(key=lambda row: (str(row["category"]), str(row["sample_id"])))
        write_csv_atomic(output_dir / "per_sample_metrics.csv", rows, PER_SAMPLE_FIELDS)
        elapsed = float(time.perf_counter() - run_started)
        eta = elapsed / index * (len(cache_files) - index)
        latest = next(
            row for row in rows
            if row["category"] == cache_path.parent.name and row["sample_id"] == cache_path.stem
        )
        current_runtime = latest.get("runtime_sec", "N/A")
        print(
            f"[PointSGRADE {index:03d}/{len(cache_files):03d}] "
            f"{cache_path.parent.name}/{cache_path.stem} "
            f"status={latest['status']} runtime={current_runtime} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    summary_rows = []
    for category in categories:
        expected = counts.get(category, 0)
        summary_rows.append(_metric_summary(rows, category, expected, config_hash))
    summary_rows.append(
        _metric_summary(rows, "overall", len(cache_files), config_hash)
    )
    write_csv(output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    manifest = {
        "experiment_name": experiment["name"], "method": "PointSGRADE",
        "config_hash": config_hash, "cache_dir": str(cache_dir),
        "output_dir": str(output_dir), "num_selected_samples": len(cache_files),
        "selected_samples": [f"{path.parent.name}/{path.stem}" for path in cache_files],
        "repetitions_per_sample": args.repetitions,
        "num_successful_samples": sum(row["status"] == "ok" for row in rows),
        "num_failed_samples": sum(row["status"] != "ok" for row in rows),
        "num_retried_samples": sum(int(row.get("attempts", 1)) > 1 for row in rows),
        "execution_controls": {
            "sample_workers": sample_workers,
            "inner_thread_limit": inner_thread_limit,
            "numba_threads": numba_threads,
            "blas_openmp_threads": inner_thread_limit,
            "parallel_unit": "independent_sample",
            "reason": POINTSGRADE_NUMBA_THREAD_REASON,
        },
        "invocation_wall_time_sec": float(time.perf_counter() - run_started),
        "summary": summary_rows,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if bool(evaluation.get("run_good_samples", False)):
        run_good_evaluation(
            method="PointSGRADE",
            inference_parameters=parameters,
            dataset_root=real3d_root,
            mvtec_root=mvtec_root,
            categories=categories,
            output_dir=output_dir,
            max_samples_per_category=args.limit,
            sample_workers=sample_workers,
            inner_thread_limit=inner_thread_limit,
            numba_threads=numba_threads,
        )
    return 0 if manifest["num_failed_samples"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
