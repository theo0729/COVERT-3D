"""Run the controlled FPFH + Isolation Forest baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    execute_final_cache_sample,
    iter_process_results,
    validate_execution_settings,
)
from baseline.FPFH_IF.fpfh_if import run_fpfh_if_baseline


PER_SAMPLE_FIELDS = [
    "category",
    "sample_id",
    "method",
    "status",
    "error",
    "cache_path",
    "num_points",
    "num_gt_positive",
    "num_pred_positive",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "iou",
    "runtime_sec",
    "predicted_defect_ratio",
    "normal_k",
    "fpfh_k",
    "orient_normals",
    "n_estimators",
    "max_samples",
    "contamination",
    "random_state",
    "num_invalid_fpfh",
    "normal_estimation_sec",
    "normal_orientation_sec",
    "fpfh_sec",
    "isolation_forest_sec",
    "anomaly_score_min",
    "anomaly_score_max",
    "anomaly_score_mean",
    "anomaly_score_std",
    "config_hash",
    "preprocess_config_hash",
    "fps_selected_index_hash",
    "prediction_path",
    "sample_workers",
    "inner_thread_limit",
    "configured_n_jobs",
    "effective_n_jobs",
]

SUMMARY_FIELDS = [
    "method",
    "scope",
    "num_expected_samples",
    "num_samples",
    "num_failed_samples",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "iou_mean",
    "iou_std",
    "runtime_mean_sec",
    "runtime_std_sec",
    "predicted_defect_ratio_mean",
    "invalid_fpfh_total",
    "config_hash",
]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return loaded


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _validate_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "paths", "fpfh_if", "evaluation"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing config mapping: {section}")
    experiment = config["experiment"]
    method = config["fpfh_if"]
    if int(experiment.get("num_working_points", 0)) <= 0:
        raise ValueError("experiment.num_working_points must be positive.")
    if not isinstance(experiment.get("categories"), list):
        raise ValueError("experiment.categories must be a list.")
    for name in ("normal_k", "fpfh_k", "n_estimators"):
        if int(method.get(name, 0)) <= 0:
            raise ValueError(f"fpfh_if.{name} must be positive.")
    if int(method.get("random_state", -1)) < 0:
        raise ValueError("fpfh_if.random_state must be non-negative.")
    validate_execution_settings(
        int(config["evaluation"].get("sample_workers", 1)),
        config["evaluation"].get("inner_thread_limit", 1),
    )


def _save_prediction(
    path: Path,
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
        method=np.asarray("FPFH_IF"),
        sample_id=np.asarray(cache["sample_id"]),
        category=np.asarray(cache["category"]),
        runtime_sec=np.asarray(runtime_sec, dtype=np.float64),
        config_hash=np.asarray(config_hash),
        preprocess_config_hash=np.asarray(cache["preprocess_config_hash"]),
        fps_selected_index_hash=np.asarray(cache["fps_selected_index_hash"]),
        diagnostics_json=np.asarray(
            json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)
        ),
    )


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    scope: str,
    expected_samples: int,
    config_hash: str,
) -> dict[str, Any]:
    in_scope = [
        row for row in rows if scope == "overall" or row["category"] == scope
    ]
    selected = [row for row in in_scope if row["status"] == "ok"]
    failures = [row for row in in_scope if row["status"] != "ok"]

    def mean_std(field: str) -> tuple[float, float]:
        values = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
        if values.size == 0:
            return float("nan"), float("nan")
        return float(values.mean()), float(values.std(ddof=0))

    precision_mean, precision_std = mean_std("precision")
    recall_mean, recall_std = mean_std("recall")
    f1_mean, f1_std = mean_std("f1")
    iou_mean, iou_std = mean_std("iou")
    runtime_mean, runtime_std = mean_std("runtime_sec")
    predicted_ratio_mean, _ = mean_std("predicted_defect_ratio")
    return {
        "method": "FPFH_IF",
        "scope": scope,
        "num_expected_samples": int(expected_samples),
        "num_samples": len(selected),
        "num_failed_samples": len(failures),
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "iou_mean": iou_mean,
        "iou_std": iou_std,
        "runtime_mean_sec": runtime_mean,
        "runtime_std_sec": runtime_std,
        "predicted_defect_ratio_mean": predicted_ratio_mean,
        "invalid_fpfh_total": sum(int(row["num_invalid_fpfh"]) for row in selected),
        "config_hash": config_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most N cached samples per category (smoke tests only).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    _validate_config(config)
    experiment = config["experiment"]
    method_config = config["fpfh_if"]
    evaluation = config["evaluation"]
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
                f"Output directory already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_workers, inner_thread_limit = validate_execution_settings(
        int(evaluation.get("sample_workers", 8)),
        evaluation.get("inner_thread_limit", 1),
    )
    configured_n_jobs = int(method_config["n_jobs"])
    effective_n_jobs = (
        configured_n_jobs if inner_thread_limit is None else inner_thread_limit
    )
    config_hash = stable_config_hash(config_without_execution_settings(config))
    cache_files = list_cache_files(cache_dir, categories)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        selected_files: list[Path] = []
        for category in categories:
            selected_files.extend(
                [path for path in cache_files if path.parent.name == category][
                    : args.limit
                ]
            )
        cache_files = selected_files

    counts = Counter(path.parent.name for path in cache_files)
    discovered_counts = Counter(
        str(sample["category"])
        for sample in discover_defect_samples(
            real3d_root,
            categories,
            mvtec_root=mvtec_root,
        )
    )
    if bool(evaluation.get("require_complete_cache", True)) and args.limit is None:
        for category in categories:
            expected = discovered_counts.get(category, 0)
            if counts.get(category, 0) != expected:
                raise RuntimeError(
                    f"Incomplete cache for {category}: found {counts.get(category, 0)}, "
                    f"expected {expected}."
                )

    resolved_config = dict(config)
    resolved_config["resolved"] = {
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path),
        "cache_dir": str(cache_dir),
        "real3d_root": str(real3d_root),
        "mvtec_root": str(mvtec_root),
        "output_dir": str(output_dir),
        "config_hash": config_hash,
        "selected_cache_files": len(cache_files),
        "limit_per_category": args.limit,
        "sample_workers": sample_workers,
        "inner_thread_limit": inner_thread_limit,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(collect_environment(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    save_predictions = bool(evaluation.get("save_predictions", True))
    parameters = {
        "normal_k": int(method_config["normal_k"]),
        "fpfh_k": int(method_config["fpfh_k"]),
        "orient_normals": bool(method_config["orient_normals"]),
        "n_estimators": int(method_config["n_estimators"]),
        "max_samples": method_config["max_samples"],
        "contamination": method_config["contamination"],
        "random_state": int(method_config["random_state"]),
        "n_jobs": int(method_config["n_jobs"]),
    }
    tasks = [
        {
            "method": "FPFH_IF",
            "cache_path": str(cache_path),
            "num_working_points": num_working_points,
            "parameters": parameters,
            "sample_workers": sample_workers,
            "inner_thread_limit": inner_thread_limit,
        }
        for cache_path in cache_files
    ]
    completed = iter_process_results(
        tasks, execute_final_cache_sample, sample_workers=sample_workers
    )
    for index, (task, payload, process_error) in enumerate(completed, start=1):
        cache_path = Path(str(task["cache_path"]))
        if payload is not None and payload["status"] == "ok":
            cache = payload["cache"]
            pred_mask = np.asarray(payload["pred_mask"], dtype=bool)
            gt_mask = np.asarray(payload["gt_mask"], dtype=bool)
            diagnostics = payload["diagnostics"]
            prediction_path = (
                output_dir / "predictions" / cache["category"] / f"{cache['sample_id']}.npz"
            )
            if save_predictions:
                _save_prediction(
                    prediction_path,
                    cache,
                    pred_mask,
                    float(payload["runtime_sec"]),
                    diagnostics,
                    config_hash,
                )
            rows.append(
                {
                    "category": cache["category"],
                    "sample_id": cache["sample_id"],
                    "method": "FPFH_IF",
                    "status": "ok",
                    "error": "",
                    "cache_path": str(cache_path),
                    "num_points": int(pred_mask.shape[0]),
                    "num_gt_positive": int(np.count_nonzero(gt_mask)),
                    "num_pred_positive": int(np.count_nonzero(pred_mask)),
                    **payload["metrics"],
                    "runtime_sec": float(payload["runtime_sec"]),
                    "predicted_defect_ratio": float(np.mean(pred_mask)),
                    "normal_k": int(method_config["normal_k"]),
                    "fpfh_k": int(method_config["fpfh_k"]),
                    "orient_normals": bool(method_config["orient_normals"]),
                    "n_estimators": int(method_config["n_estimators"]),
                    "max_samples": method_config["max_samples"],
                    "contamination": method_config["contamination"],
                    "random_state": int(method_config["random_state"]),
                    "num_invalid_fpfh": diagnostics["num_invalid_fpfh"],
                    "normal_estimation_sec": diagnostics["normal_estimation_sec"],
                    "normal_orientation_sec": diagnostics["normal_orientation_sec"],
                    "fpfh_sec": diagnostics["fpfh_sec"],
                    "isolation_forest_sec": diagnostics["isolation_forest_sec"],
                    "anomaly_score_min": diagnostics["anomaly_score_min"],
                    "anomaly_score_max": diagnostics["anomaly_score_max"],
                    "anomaly_score_mean": diagnostics["anomaly_score_mean"],
                    "anomaly_score_std": diagnostics["anomaly_score_std"],
                    "config_hash": config_hash,
                    "preprocess_config_hash": cache["preprocess_config_hash"],
                    "fps_selected_index_hash": cache["fps_selected_index_hash"],
                    "prediction_path": str(prediction_path) if save_predictions else "",
                    "sample_workers": sample_workers,
                    "inner_thread_limit": inner_thread_limit,
                    "configured_n_jobs": configured_n_jobs,
                    "effective_n_jobs": effective_n_jobs,
                }
            )
        else:
            error_text = (
                str(payload["error"]) if payload is not None
                else f"{type(process_error).__name__}: {process_error}"
            )
            rows.append(
                {
                    "category": cache_path.parent.name,
                    "sample_id": cache_path.stem,
                    "method": "FPFH_IF",
                    "status": "error",
                    "error": error_text,
                    "cache_path": str(cache_path),
                    "config_hash": config_hash,
                    "sample_workers": sample_workers,
                    "inner_thread_limit": inner_thread_limit,
                    "configured_n_jobs": configured_n_jobs,
                    "effective_n_jobs": effective_n_jobs,
                }
            )
        rows.sort(key=lambda row: (str(row["category"]), str(row["sample_id"])))
        write_csv_atomic(output_dir / "per_sample_metrics.csv", rows, PER_SAMPLE_FIELDS)
        latest = next(
            row for row in rows
            if row["category"] == cache_path.parent.name and row["sample_id"] == cache_path.stem
        )
        print(
            f"[FPFH_IF {index:03d}/{len(cache_files):03d}] "
            f"{cache_path.parent.name}/{cache_path.stem} {latest['status']}",
            flush=True,
        )

    summary_rows: list[dict[str, Any]] = []
    for category in categories:
        expected = counts.get(category, 0)
        summary_rows.append(_metric_summary(rows, category, expected, config_hash))
    overall_expected = sum(int(row["num_expected_samples"]) for row in summary_rows)
    summary_rows.append(_metric_summary(rows, "overall", overall_expected, config_hash))
    write_csv(output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)

    manifest = {
        "experiment_name": experiment["name"],
        "method": "FPFH_IF",
        "config_hash": config_hash,
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "num_selected_samples": len(cache_files),
        "num_successful_samples": sum(row["status"] == "ok" for row in rows),
        "num_failed_samples": sum(row["status"] != "ok" for row in rows),
        "execution": {
            "sample_workers": sample_workers,
            "inner_thread_limit": inner_thread_limit,
            "configured_n_jobs": configured_n_jobs,
            "effective_n_jobs": effective_n_jobs,
            "parallel_unit": "independent_sample",
        },
        "summary": summary_rows,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if bool(evaluation.get("run_good_samples", False)):
        run_good_evaluation(
            method="FPFH_IF",
            inference_parameters=parameters,
            dataset_root=real3d_root,
            mvtec_root=mvtec_root,
            categories=categories,
            output_dir=output_dir,
            max_samples_per_category=args.limit,
            sample_workers=sample_workers,
            inner_thread_limit=inner_thread_limit,
        )
    return 0 if manifest["num_failed_samples"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
