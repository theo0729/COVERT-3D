"""Run the controlled PCL-style Region Growing baseline."""

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
from baseline.RG.region_growing import run_region_growing_baseline


PER_SAMPLE_FIELDS = [
    "category", "sample_id", "method", "status", "error", "cache_path",
    "num_points", "num_gt_positive", "num_pred_positive", "tp", "fp", "fn", "tn",
    "precision", "recall", "f1", "iou", "runtime_sec", "predicted_defect_ratio",
    "smoothing_k", "smoothing_iterations", "normal_k", "region_neighbor_k",
    "smoothness_threshold_deg", "curvature_threshold", "min_cluster_size",
    "num_regions", "num_valid_regions", "num_singleton_regions",
    "largest_region_points", "largest_region_ratio", "second_largest_region_ratio",
    "smoothing_sec", "normal_knn_sec", "normal_pca_sec", "region_knn_sec",
    "region_growing_sec", "curvature_mean", "curvature_p95",
    "config_hash", "preprocess_config_hash", "fps_selected_index_hash", "prediction_path",
    "sample_workers", "inner_thread_limit", "configured_query_workers",
    "effective_query_workers",
]

SUMMARY_FIELDS = [
    "method", "scope", "num_expected_samples", "num_samples", "num_failed_samples",
    "precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std",
    "iou_mean", "iou_std", "runtime_mean_sec", "runtime_std_sec",
    "predicted_defect_ratio_mean", "num_regions_mean", "largest_region_ratio_mean",
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
    for section in ("experiment", "paths", "region_growing", "evaluation"):
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing config mapping: {section}")
    method = config["region_growing"]
    for name in (
        "smoothing_k", "normal_k", "region_neighbor_k", "min_cluster_size"
    ):
        if int(method.get(name, 0)) <= 0:
            raise ValueError(f"region_growing.{name} must be positive.")
    if int(method.get("smoothing_iterations", -1)) < 0:
        raise ValueError("smoothing_iterations must be non-negative.")
    if float(method.get("smoothness_threshold_deg", 0.0)) <= 0.0:
        raise ValueError("smoothness_threshold_deg must be positive.")
    if float(method.get("curvature_threshold", -1.0)) < 0.0:
        raise ValueError("curvature_threshold must be non-negative.")
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
        method=np.asarray("RG"),
        sample_id=np.asarray(cache["sample_id"]),
        category=np.asarray(cache["category"]),
        runtime_sec=np.asarray(runtime_sec, dtype=np.float64),
        config_hash=np.asarray(config_hash),
        preprocess_config_hash=np.asarray(cache["preprocess_config_hash"]),
        fps_selected_index_hash=np.asarray(cache["fps_selected_index_hash"]),
        diagnostics_json=np.asarray(json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)),
    )


def _summary(
    rows: Sequence[Mapping[str, Any]], scope: str, expected: int, config_hash: str
) -> dict[str, Any]:
    scoped = [row for row in rows if scope == "overall" or row["category"] == scope]
    valid = [row for row in scoped if row["status"] == "ok"]
    failed = [row for row in scoped if row["status"] != "ok"]

    def mean_std(field: str) -> tuple[float, float]:
        values = np.asarray([float(row[field]) for row in valid], dtype=np.float64)
        if values.size == 0:
            return float("nan"), float("nan")
        return float(values.mean()), float(values.std(ddof=0))

    values = {}
    for field in ("precision", "recall", "f1", "iou", "runtime_sec"):
        values[f"{field}_mean"], values[f"{field}_std"] = mean_std(field)
    ratio_mean, _ = mean_std("predicted_defect_ratio")
    regions_mean, _ = mean_std("num_regions")
    largest_mean, _ = mean_std("largest_region_ratio")
    return {
        "method": "RG", "scope": scope, "num_expected_samples": int(expected),
        "num_samples": len(valid), "num_failed_samples": len(failed),
        "precision_mean": values["precision_mean"], "precision_std": values["precision_std"],
        "recall_mean": values["recall_mean"], "recall_std": values["recall_std"],
        "f1_mean": values["f1_mean"], "f1_std": values["f1_std"],
        "iou_mean": values["iou_mean"], "iou_std": values["iou_std"],
        "runtime_mean_sec": values["runtime_sec_mean"],
        "runtime_std_sec": values["runtime_sec_std"],
        "predicted_defect_ratio_mean": ratio_mean,
        "num_regions_mean": regions_mean,
        "largest_region_ratio_mean": largest_mean,
        "config_hash": config_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    _validate_config(config)
    experiment = config["experiment"]
    method = config["region_growing"]
    evaluation = config["evaluation"]
    categories = [str(value).lower() for value in experiment["categories"]]
    num_working_points = int(experiment["num_working_points"])
    cache_dir = _resolve_project_path(config["paths"]["cache_dir"])
    real3d_root = _resolve_project_path(config["paths"]["real3d_root"])
    mvtec_root = _resolve_project_path(config["paths"]["mvtec_root"])
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else _resolve_project_path(config["paths"]["output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_workers, inner_thread_limit = validate_execution_settings(
        int(evaluation.get("sample_workers", 8)),
        evaluation.get("inner_thread_limit", 1),
    )
    configured_query_workers = int(method["query_workers"])
    effective_query_workers = (
        configured_query_workers
        if inner_thread_limit is None
        else inner_thread_limit
    )
    config_hash = stable_config_hash(config_without_execution_settings(config))

    cache_files = list_cache_files(cache_dir, categories)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        cache_files = [
            path for category in categories
            for path in [p for p in cache_files if p.parent.name == category][:args.limit]
        ]
    counts = Counter(path.parent.name for path in cache_files)
    discovered_counts = Counter(
        str(sample["category"])
        for sample in discover_defect_samples(
            real3d_root, categories, mvtec_root=mvtec_root
        )
    )
    if bool(evaluation.get("require_complete_cache", True)) and args.limit is None:
        for category in categories:
            if counts.get(category, 0) != discovered_counts.get(category, 0):
                raise RuntimeError(f"Incomplete cache for {category}.")

    resolved = dict(config)
    resolved["resolved"] = {
        "project_root": str(PROJECT_ROOT), "config_path": str(config_path),
        "cache_dir": str(cache_dir), "output_dir": str(output_dir),
        "real3d_root": str(real3d_root), "mvtec_root": str(mvtec_root),
        "config_hash": config_hash, "selected_cache_files": len(cache_files),
        "limit_per_category": args.limit, "sample_workers": sample_workers,
        "inner_thread_limit": inner_thread_limit,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(collect_environment(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    parameters = {
        "smoothing_k": int(method["smoothing_k"]),
        "smoothing_iterations": int(method["smoothing_iterations"]),
        "normal_k": int(method["normal_k"]),
        "region_neighbor_k": int(method["region_neighbor_k"]),
        "smoothness_threshold_deg": float(method["smoothness_threshold_deg"]),
        "curvature_threshold": float(method["curvature_threshold"]),
        "min_cluster_size": int(method["min_cluster_size"]),
        "query_workers": int(method["query_workers"]),
    }
    tasks = [
        {
            "method": "RG", "cache_path": str(cache_path),
            "num_working_points": num_working_points, "parameters": parameters,
            "sample_workers": sample_workers, "inner_thread_limit": inner_thread_limit,
        }
        for cache_path in cache_files
    ]
    completed = iter_process_results(
        tasks, execute_final_cache_sample, sample_workers=sample_workers
    )
    for index, (task, payload, process_error) in enumerate(completed, 1):
        cache_path = Path(str(task["cache_path"]))
        if payload is not None and payload["status"] == "ok":
            cache = payload["cache"]
            pred = np.asarray(payload["pred_mask"], dtype=bool)
            gt_mask = np.asarray(payload["gt_mask"], dtype=bool)
            diagnostics = payload["diagnostics"]
            prediction_path = output_dir / "predictions" / cache["category"] / f"{cache['sample_id']}.npz"
            if bool(evaluation.get("save_predictions", True)):
                _save_prediction(prediction_path, cache, pred, float(payload["runtime_sec"]), diagnostics, config_hash)
            rows.append({
                "category": cache["category"], "sample_id": cache["sample_id"], "method": "RG",
                "status": "ok", "error": "", "cache_path": str(cache_path),
                "num_points": int(pred.size), "num_gt_positive": int(gt_mask.sum()),
                "num_pred_positive": int(pred.sum()), **payload["metrics"],
                "runtime_sec": float(payload["runtime_sec"]), "predicted_defect_ratio": float(pred.mean()),
                "smoothing_k": method["smoothing_k"], "smoothing_iterations": method["smoothing_iterations"],
                "normal_k": method["normal_k"], "region_neighbor_k": method["region_neighbor_k"],
                "smoothness_threshold_deg": method["smoothness_threshold_deg"],
                "curvature_threshold": method["curvature_threshold"], "min_cluster_size": method["min_cluster_size"],
                "num_regions": diagnostics["num_regions"], "num_valid_regions": diagnostics["num_valid_regions"],
                "num_singleton_regions": diagnostics["num_singleton_regions"],
                "largest_region_points": diagnostics["largest_region_points"],
                "largest_region_ratio": diagnostics["largest_region_ratio"],
                "second_largest_region_ratio": diagnostics["second_largest_region_ratio"],
                "smoothing_sec": diagnostics["smoothing_sec"], "normal_knn_sec": diagnostics["normal_knn_sec"],
                "normal_pca_sec": diagnostics["normal_pca_sec"], "region_knn_sec": diagnostics["region_knn_sec"],
                "region_growing_sec": diagnostics["region_growing_sec"], "curvature_mean": diagnostics["curvature_mean"],
                "curvature_p95": diagnostics["curvature_p95"], "config_hash": config_hash,
                "preprocess_config_hash": cache["preprocess_config_hash"],
                "fps_selected_index_hash": cache["fps_selected_index_hash"], "prediction_path": str(prediction_path),
                "sample_workers": sample_workers, "inner_thread_limit": inner_thread_limit,
                "configured_query_workers": configured_query_workers,
                "effective_query_workers": effective_query_workers,
            })
        else:
            error_text = str(payload["error"]) if payload is not None else f"{type(process_error).__name__}: {process_error}"
            rows.append({
                "category": cache_path.parent.name, "sample_id": cache_path.stem,
                "method": "RG", "status": "error", "error": error_text,
                "cache_path": str(cache_path), "config_hash": config_hash,
                "sample_workers": sample_workers, "inner_thread_limit": inner_thread_limit,
                "configured_query_workers": configured_query_workers,
                "effective_query_workers": effective_query_workers,
            })
        rows.sort(key=lambda row: (str(row["category"]), str(row["sample_id"])))
        write_csv_atomic(output_dir / "per_sample_metrics.csv", rows, PER_SAMPLE_FIELDS)
        latest = next(row for row in rows if row["category"] == cache_path.parent.name and row["sample_id"] == cache_path.stem)
        print(f"[RG {index:03d}/{len(cache_files):03d}] {cache_path.parent.name}/{cache_path.stem} {latest['status']}", flush=True)

    summary_rows = []
    for category in categories:
        expected = counts.get(category, 0)
        summary_rows.append(_summary(rows, category, expected, config_hash))
    summary_rows.append(_summary(rows, "overall", sum(row["num_expected_samples"] for row in summary_rows), config_hash))
    write_csv(output_dir / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    manifest = {
        "experiment_name": experiment["name"], "method": "RG", "config_hash": config_hash,
        "cache_dir": str(cache_dir), "output_dir": str(output_dir), "num_selected_samples": len(cache_files),
        "num_successful_samples": sum(row["status"] == "ok" for row in rows),
        "num_failed_samples": sum(row["status"] != "ok" for row in rows), "summary": summary_rows,
        "execution": {
            "sample_workers": sample_workers,
            "inner_thread_limit": inner_thread_limit,
            "configured_query_workers": configured_query_workers,
            "effective_query_workers": effective_query_workers,
            "parallel_unit": "independent_sample",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if bool(evaluation.get("run_good_samples", False)):
        run_good_evaluation(
            method="RG",
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
