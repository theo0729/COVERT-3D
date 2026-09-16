"""Shared, sequential-by-configuration COVERT sensitivity runner v2."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import numpy as np

from sensitivity_analysis._config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MVTEC_ROOT,
    DEFAULT_QUERY_WORKERS,
    DEFAULT_SAMPLE_WORKERS,
    EXPERIMENT_DEFINITIONS,
    MAIN8,
    MVTEC_MAIN4,
    PROJECT_ROOT,
    assert_result_effective_values,
    build_sensitivity_config,
    canonical_json,
    config_sha256,
    implementation_metadata,
    production_file_sha256,
    read_experiment_config,
    sha256_json,
    validate_experiment_config,
    validate_global_config_count,
)
from sensitivity_analysis._dev10 import (
    DEFAULT_MANIFEST,
    is_dev10,
    load_dev10_groups,
    physical_group_key,
)
from sensitivity_analysis._metrics import (
    CATEGORY_FIELDS,
    STAGE_DIAGNOSTIC_FIELDS,
    SUMMARY_METRIC_FIELDS,
    aggregate_configuration,
    complete_binary_metrics,
    extract_stage_diagnostics,
    sample_fieldnames,
    stage_fieldnames,
)


POINT_SUFFIXES = (".pcd", ".ply", ".npy", ".txt", ".xyz", ".csv")


@dataclass(frozen=True)
class SampleSpec:
    dataset: str
    category: str
    sample_id: str
    pc_path: str
    gt_path: str
    sample_identity_hash: str


class _TeeStream(io.TextIOBase):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (list, tuple, dict)):
                    row[key] = canonical_json(value)
            writer.writerow(row)
    os.replace(temporary, path)


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sample_identity(
    dataset: str,
    category: str,
    sample_id: str,
    pc_path: Path,
    gt_path: Path,
) -> str:
    return sha256_json(
        {
            "dataset": dataset,
            "category": category,
            "sample_id": sample_id,
            "pc": _file_identity(pc_path),
            "gt": _file_identity(gt_path),
        }
    )


def _is_good(path: Path) -> bool:
    return "good" in path.stem.strip().lower().split("_")


def discover_samples(
    *,
    dataset_root: Path,
    mvtec_root: Path,
    categories: Sequence[str],
) -> tuple[list[SampleSpec], list[dict[str, str]]]:
    """Discover defective Main8 samples without importing covert_batch."""

    if any(category not in MAIN8 for category in categories):
        invalid = [category for category in categories if category not in MAIN8]
        raise ValueError(f"Sensitivity accepts only Main8 categories: {invalid}")
    specs: list[SampleSpec] = []
    failures: list[dict[str, str]] = []
    for category in categories:
        if category in MVTEC_MAIN4:
            candidates = (
                ("mvtec3d", mvtec_root / category),
                ("mvtec3d", dataset_root / "MvTec" / category),
            )
        else:
            candidates = (("real3d", dataset_root / category),)
        resolved = next(
            ((dataset, root) for dataset, root in candidates if root.is_dir()), None
        )
        if resolved is None:
            failures.append(
                {
                    "dataset": "mvtec3d" if category in MVTEC_MAIN4 else "real3d",
                    "category": category,
                    "sample_id": "",
                    "failure_type": "MissingCategoryDirectory",
                    "failure_message": " | ".join(str(root) for _, root in candidates),
                }
            )
            continue
        dataset, category_root = resolved
        test_root, gt_root = category_root / "test", category_root / "gt"
        if not test_root.is_dir() or not gt_root.is_dir():
            failures.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "sample_id": "",
                    "failure_type": "MissingTestOrGtDirectory",
                    "failure_message": str(category_root),
                }
            )
            continue
        gt_files = sorted(
            path
            for path in gt_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".npy"}
        )
        for gt_path in gt_files:
            if _is_good(gt_path):
                continue
            pc_path = next(
                (
                    test_root / f"{gt_path.stem}{suffix}"
                    for suffix in POINT_SUFFIXES
                    if (test_root / f"{gt_path.stem}{suffix}").is_file()
                ),
                None,
            )
            if pc_path is None:
                failures.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "sample_id": gt_path.stem,
                        "failure_type": "MissingPointCloud",
                        "failure_message": str(test_root),
                    }
                )
                continue
            specs.append(
                SampleSpec(
                    dataset=dataset,
                    category=category,
                    sample_id=pc_path.stem,
                    pc_path=str(pc_path.resolve()),
                    gt_path=str(gt_path.resolve()),
                    sample_identity_hash=_sample_identity(
                        dataset, category, pc_path.stem, pc_path, gt_path
                    ),
                )
            )
    order = {category: index for index, category in enumerate(categories)}
    specs.sort(key=lambda item: (order[item.category], item.sample_id, item.pc_path))
    return specs, failures


def select_quick_sample(
    samples: Sequence[SampleSpec],
    categories: Sequence[str],
    sample_key: str | None = None,
) -> SampleSpec:
    if sample_key is not None:
        normalized = sample_key.strip().replace("\\", "/").strip("/")
        parts = normalized.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("--sample-key must use CATEGORY/SAMPLE_ID syntax")
        category, sample_id = parts
        if category not in categories:
            raise ValueError(
                f"--sample-key category {category!r} is not in --categories"
            )
        exact = [
            sample
            for sample in samples
            if sample.category == category and sample.sample_id == sample_id
        ]
        if len(exact) != 1:
            raise RuntimeError(
                f"Expected exactly one defective sample for --sample-key {normalized!r}, "
                f"found {len(exact)}"
            )
        return exact[0]
    if "fish" in categories:
        exact = [
            sample
            for sample in samples
            if sample.category == "fish" and sample.sample_id == "275_bulge"
        ]
        if exact:
            return exact[0]
    if not samples:
        raise RuntimeError("No defective sample is available for --quick")
    return samples[0]


def _checkpoint_path(config_dir: Path, sample: SampleSpec) -> Path:
    return (
        config_dir
        / "samples"
        / sample.dataset
        / sample.category
        / sample.sample_id
        / "metrics.json"
    )


def _read_checkpoint(
    config_dir: Path,
    sample: SampleSpec,
    run_fingerprint: str,
) -> dict[str, Any] | None:
    path = _checkpoint_path(config_dir, sample)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    row = payload.get("sample_metrics")
    diagnostics = payload.get("stage_diagnostics")
    if not isinstance(row, dict) or not isinstance(diagnostics, dict):
        return None
    if row.get("run_fingerprint") != run_fingerprint:
        return None
    if row.get("sample_identity_hash") != sample.sample_identity_hash:
        return None
    if row.get("status") != "success":
        return None
    row["resumed"] = True
    return {"sample_metrics": row, "stage_diagnostics": diagnostics}


def _blank_stage_row(
    item: Mapping[str, Any], sample: SampleSpec, *, is_dev: bool
) -> dict[str, Any]:
    return {
        "config_id": item["id"],
        **dict(item["parameters"]),
        "dataset": sample.dataset,
        "category": sample.category,
        "sample_id": sample.sample_id,
        "is_dev10": is_dev,
        **{field: None for field in STAGE_DIAGNOSTIC_FIELDS[5:]},
    }


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run exactly one sample through production run_covert_sample."""

    from covert_sample import DEFAULT_CONFIG, config_sha256, run_covert_sample

    sample = SampleSpec(**payload["sample"])
    item = payload["configuration"]
    default_hash_before = config_sha256(DEFAULT_CONFIG)
    config, construction = build_sensitivity_config(
        item["changes"], allowed_fields=payload["sweep_fields"]
    )
    if construction["sensitivity_config_hash"] != payload["config_hash"]:
        raise AssertionError("Worker reconstructed a different sensitivity config")
    started = time.perf_counter()
    try:
        result = run_covert_sample(
            sample.pc_path,
            sample.gt_path,
            config=config,
            query_workers=int(payload["query_workers"]),
            visualize=False,
            output_dir=None,
            verbose=False,
            config_source=(
                f"sensitivity_analysis_v2:{payload['experiment_name']}:{item['id']}"
            ),
        )
        if result.config_hash != payload["config_hash"]:
            raise AssertionError("Production result config hash differs from requested config")
        effective = assert_result_effective_values(result, changes=item["changes"])
        metrics = complete_binary_metrics(result.metrics)
        final_mask = np.asarray(result.final_mask)
        if final_mask.ndim != 1:
            raise AssertionError("Production final mask must be one-dimensional")
        if final_mask.size != int(metrics["working_point_count"]):
            raise AssertionError("Production final-mask length differs from working cloud")
        if final_mask.dtype != np.bool_ and not bool(
            np.all((final_mask == 0) | (final_mask == 1))
        ):
            raise AssertionError("Production final mask is not bool/binary")
        if int(metrics["predicted_positive_points"]) != int(final_mask.sum()):
            raise AssertionError("Production predicted-positive metric differs from final mask")
        dev = bool(payload["is_dev10"])
        row = {
            "config_id": item["id"],
            "is_production_default": bool(item["is_production_default"]),
            **dict(item["parameters"]),
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "physical_group_key": physical_group_key(
                sample.category, sample.sample_id
            ),
            "is_dev10": dev,
            "sample_identity_hash": sample.sample_identity_hash,
            "status": "success",
            **{name: metrics[name] for name in ("tp", "fp", "fn", "tn")},
            **{name: metrics[name] for name in ("precision", "recall", "f1", "iou")},
            "predicted_positive_points": metrics["predicted_positive_points"],
            "gt_positive_points": metrics["gt_positive_points"],
            "working_point_count": metrics["working_point_count"],
            "inference_total_seconds": result.timings.inference_total_seconds,
            "worker_wall_seconds": float(time.perf_counter() - started),
            "config_hash": result.config_hash,
            "implementation_hash": payload["implementation_hash"],
            "implementation_fingerprint_schema": payload[
                "implementation_fingerprint_schema"
            ],
            "implementation_file_count": payload["implementation_file_count"],
            "run_fingerprint": payload["run_fingerprint"],
            "effective_parameters_json": canonical_json(effective),
            "failure_type": "",
            "failure_message": "",
            "resumed": False,
        }
        diagnostics = {
            "config_id": item["id"],
            **dict(item["parameters"]),
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "is_dev10": dev,
            **extract_stage_diagnostics(result),
        }
    except Exception as exc:
        dev = bool(payload["is_dev10"])
        row = {
            "config_id": item["id"],
            "is_production_default": bool(item["is_production_default"]),
            **dict(item["parameters"]),
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "physical_group_key": physical_group_key(
                sample.category, sample.sample_id
            ),
            "is_dev10": dev,
            "sample_identity_hash": sample.sample_identity_hash,
            "status": "failure",
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "iou": None,
            "predicted_positive_points": None,
            "gt_positive_points": None,
            "working_point_count": None,
            "inference_total_seconds": None,
            "worker_wall_seconds": float(time.perf_counter() - started),
            "config_hash": payload["config_hash"],
            "implementation_hash": payload["implementation_hash"],
            "implementation_fingerprint_schema": payload[
                "implementation_fingerprint_schema"
            ],
            "implementation_file_count": payload["implementation_file_count"],
            "run_fingerprint": payload["run_fingerprint"],
            "effective_parameters_json": "",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback": traceback.format_exc(),
            "resumed": False,
        }
        diagnostics = _blank_stage_row(item, sample, is_dev=dev)
    if config_sha256(DEFAULT_CONFIG) != default_hash_before:
        raise AssertionError("Worker observed mutation of covert_sample.DEFAULT_CONFIG")
    return {"sample_metrics": row, "stage_diagnostics": diagnostics}


def _select_configurations(
    payload: Mapping[str, Any], only: Sequence[str] | None, quick: bool
) -> list[dict[str, Any]]:
    configurations = [dict(item) for item in payload["configurations"]]
    by_id = {str(item["id"]): item for item in configurations}
    if only:
        requested = list(dict.fromkeys(str(value) for value in only))
        unknown = [value for value in requested if value not in by_id]
        if unknown:
            raise ValueError(f"Unknown --only configuration id(s): {unknown}")
        selected = [by_id[value] for value in requested]
    elif quick:
        selected = [
            item for item in configurations if item["is_production_default"] is True
        ]
    else:
        selected = configurations
    if quick and len(selected) != 1:
        raise ValueError("--quick executes exactly one configuration; pass one --only ID")
    return selected


def _result_base(
    experiment_dir: Path,
    *,
    quick: bool,
    categories: Sequence[str],
    output_root: Path | None,
) -> Path:
    if output_root is not None:
        return output_root
    if quick:
        return experiment_dir / "results" / "quick"
    if tuple(categories) != MAIN8:
        slug = "-".join(categories)
        return experiment_dir / "results" / "custom" / slug
    return experiment_dir / "results"


def _run_fingerprint(
    *,
    experiment_name: str,
    config_id: str,
    config_hash: str,
    implementation_hash: str,
    implementation_fingerprint_schema: int,
    implementation_file_count: int,
    cohort_hash: str,
    categories: Sequence[str],
    quick: bool,
    query_workers: int,
) -> str:
    return sha256_json(
        {
            "schema": "covert-sensitivity-v2",
            "experiment_name": experiment_name,
            "config_id": config_id,
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "implementation_fingerprint_schema": int(
                implementation_fingerprint_schema
            ),
            "implementation_file_count": int(implementation_file_count),
            "cohort_hash": cohort_hash,
            "categories": list(categories),
            "mode": "quick" if quick else "formal_or_custom",
            "query_workers": int(query_workers),
            "run_good_samples": False,
        }
    )


def _configuration_complete(config_dir: Path, run_fingerprint: str) -> bool:
    path = config_dir / "completed.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("run_fingerprint") == run_fingerprint
        and (config_dir / "sample_metrics.json").is_file()
    )


def _write_config_outputs(
    *,
    config_dir: Path,
    metrics_dir: Path,
    parameter_columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> None:
    _atomic_write_json(config_dir / "sample_metrics.json", list(rows))
    _atomic_write_json(config_dir / "stage_diagnostics.json", list(diagnostics))
    _atomic_write_csv(
        config_dir / "sample_metrics.csv", sample_fieldnames(parameter_columns), rows
    )
    _atomic_write_csv(
        config_dir / "stage_diagnostics.csv",
        stage_fieldnames(parameter_columns),
        diagnostics,
    )
    _atomic_write_csv(
        metrics_dir / f"{config_dir.name}.csv",
        sample_fieldnames(parameter_columns),
        rows,
    )


def _run_one_configuration(
    *,
    experiment_name: str,
    definition: Mapping[str, Any],
    item: Mapping[str, Any],
    samples: Sequence[SampleSpec],
    dev10_groups: frozenset[str],
    result_base: Path,
    implementation: Mapping[str, Any],
    cohort_hash: str,
    categories: Sequence[str],
    quick: bool,
    sample_workers: int,
    query_workers: int,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, str]:
    implementation_hash = str(implementation["implementation_hash"])
    implementation_schema = int(
        implementation["implementation_fingerprint_schema"]
    )
    implementation_file_count = int(implementation["implementation_file_count"])
    config, audit = build_sensitivity_config(
        item["changes"], allowed_fields=definition["sweep_fields"]
    )
    config_id = str(item["id"])
    run_fingerprint = _run_fingerprint(
        experiment_name=experiment_name,
        config_id=config_id,
        config_hash=audit["sensitivity_config_hash"],
        implementation_hash=implementation_hash,
        implementation_fingerprint_schema=implementation_schema,
        implementation_file_count=implementation_file_count,
        cohort_hash=cohort_hash,
        categories=categories,
        quick=quick,
        query_workers=query_workers,
    )
    config_dir = result_base / "raw" / config_id
    config_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "experiment_name": experiment_name,
        "config_id": config_id,
        "swept_parameters": dict(item["parameters"]),
        **audit,
        **dict(implementation),
        "is_production_default": bool(item["is_production_default"]),
        "sample_workers": int(sample_workers),
        "query_workers": int(query_workers),
        "run_fingerprint": run_fingerprint,
        "cohort_hash": cohort_hash,
        "categories": list(categories),
        "quick": bool(quick),
        "pipeline_entrypoint": "covert_sample.run_covert_sample",
        "run_good_samples": False,
    }
    _atomic_write_json(config_dir / "config.json", config_payload)
    if _configuration_complete(config_dir, run_fingerprint) and not force:
        rows = json.loads((config_dir / "sample_metrics.json").read_text(encoding="utf-8"))
        diagnostics = json.loads(
            (config_dir / "stage_diagnostics.json").read_text(encoding="utf-8")
        )
        return rows, diagnostics, 0.0, "skipped_complete"

    for stale in (config_dir / "completed.json", config_dir / "failure.json"):
        if stale.is_file():
            stale.unlink()

    resumed_by_id: dict[str, dict[str, Any]] = {}
    pending: list[SampleSpec] = []
    for sample in samples:
        checkpoint = None if force else _read_checkpoint(
            config_dir, sample, run_fingerprint
        )
        if checkpoint is None:
            pending.append(sample)
        else:
            resumed_by_id[sample.sample_identity_hash] = checkpoint

    started = time.perf_counter()
    produced_by_id: dict[str, dict[str, Any]] = dict(resumed_by_id)
    payloads = [
        {
            "experiment_name": experiment_name,
            "configuration": dict(item),
            "sweep_fields": list(definition["sweep_fields"]),
            "sample": asdict(sample),
            "is_dev10": is_dev10(sample.category, sample.sample_id, dev10_groups),
            "config_hash": config_sha256(config),
            "implementation_hash": implementation_hash,
            "implementation_fingerprint_schema": implementation_schema,
            "implementation_file_count": implementation_file_count,
            "run_fingerprint": run_fingerprint,
            "query_workers": int(query_workers),
        }
        for sample in pending
    ]

    if sample_workers == 1:
        iterator = ((_worker(payload), SampleSpec(**payload["sample"])) for payload in payloads)
        for output, sample in iterator:
            produced_by_id[sample.sample_identity_hash] = output
            _atomic_write_json(_checkpoint_path(config_dir, sample), output)
    elif payloads:
        with ProcessPoolExecutor(max_workers=sample_workers) as executor:
            future_to_sample = {
                executor.submit(_worker, payload): SampleSpec(**payload["sample"])
                for payload in payloads
            }
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                output = future.result()
                produced_by_id[sample.sample_identity_hash] = output
                _atomic_write_json(_checkpoint_path(config_dir, sample), output)

    rows = [
        produced_by_id[sample.sample_identity_hash]["sample_metrics"]
        for sample in samples
        if sample.sample_identity_hash in produced_by_id
    ]
    diagnostics = [
        produced_by_id[sample.sample_identity_hash]["stage_diagnostics"]
        for sample in samples
        if sample.sample_identity_hash in produced_by_id
    ]
    runtime = float(time.perf_counter() - started)
    _write_config_outputs(
        config_dir=config_dir,
        metrics_dir=result_base / "metrics",
        parameter_columns=definition["parameter_columns"],
        rows=rows,
        diagnostics=diagnostics,
    )

    failures = [row for row in rows if row.get("status") != "success"]
    if len(rows) != len(samples) or failures:
        failure_payload = {
            "status": "failed",
            "failed_at": datetime.now().astimezone().isoformat(),
            "configuration_id": config_id,
            "run_fingerprint": run_fingerprint,
            "implementation_hash": implementation_hash,
            "implementation_fingerprint_schema": implementation_schema,
            "implementation_file_count": implementation_file_count,
            "expected_sample_count": len(samples),
            "observed_sample_count": len(rows),
            "failed_sample_count": len(failures),
            "failed_samples": [
                f"{row.get('category')}/{row.get('sample_id')}" for row in failures
            ],
        }
        _atomic_write_json(config_dir / "failure.json", failure_payload)
        return rows, diagnostics, runtime, "failed"

    completion = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "configuration_id": config_id,
        "run_fingerprint": run_fingerprint,
        "config_hash": config_sha256(config),
        "implementation_hash": implementation_hash,
        "implementation_fingerprint_schema": implementation_schema,
        "implementation_file_count": implementation_file_count,
        "cohort_hash": cohort_hash,
        "categories": list(categories),
        "quick": bool(quick),
        "successful_samples": len(rows),
        "runtime_seconds": runtime,
    }
    _atomic_write_json(config_dir / "completed.json", completion)
    return rows, diagnostics, runtime, "completed"


def _rebuild_experiment_outputs(
    *,
    payload: Mapping[str, Any],
    definition: Mapping[str, Any],
    result_base: Path,
    implementation_hash: str,
    implementation_fingerprint_schema: int,
) -> dict[str, int]:
    all_sample_rows: list[dict[str, Any]] = []
    all_stage_rows: list[dict[str, Any]] = []
    all_category_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cohort_hashes: set[str] = set()

    for item in payload["configurations"]:
        config_dir = result_base / "raw" / str(item["id"])
        completed_path = config_dir / "completed.json"
        samples_path = config_dir / "sample_metrics.json"
        diagnostics_path = config_dir / "stage_diagnostics.json"
        if not completed_path.is_file() or not samples_path.is_file():
            continue
        completion = json.loads(completed_path.read_text(encoding="utf-8"))
        if completion.get("status") != "completed":
            continue
        if completion.get("implementation_hash") != implementation_hash:
            continue
        if (
            completion.get("implementation_fingerprint_schema")
            != implementation_fingerprint_schema
        ):
            continue
        cohort_hashes.add(str(completion.get("cohort_hash") or ""))
        rows = json.loads(samples_path.read_text(encoding="utf-8"))
        diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if diagnostics_path.is_file()
            else []
        )
        categories = tuple(completion.get("categories") or ())
        category_rows, aggregate = aggregate_configuration(
            config_id=str(item["id"]), rows=rows, categories=categories
        )
        all_sample_rows.extend(rows)
        all_stage_rows.extend(diagnostics)
        all_category_rows.extend(category_rows)
        summary_rows.append(
            {
                "config_id": item["id"],
                "is_production_default": bool(item["is_production_default"]),
                **dict(item["parameters"]),
                **aggregate,
                "runtime_seconds": completion.get("runtime_seconds"),
            }
        )

    if len(cohort_hashes) > 1:
        raise RuntimeError(
            "Completed configurations use different sample cohorts; refusing to combine them"
        )
    parameter_columns = definition["parameter_columns"]
    _atomic_write_csv(
        result_base / "per_sample_metrics.csv",
        sample_fieldnames(parameter_columns),
        all_sample_rows,
    )
    _atomic_write_csv(
        result_base / "per_category_metrics.csv",
        CATEGORY_FIELDS,
        all_category_rows,
    )
    _atomic_write_csv(
        result_base / "stage_diagnostics.csv",
        stage_fieldnames(parameter_columns),
        all_stage_rows,
    )
    _atomic_write_csv(
        result_base / "summary.csv",
        [
            "config_id",
            "is_production_default",
            *parameter_columns,
            *SUMMARY_METRIC_FIELDS,
        ],
        summary_rows,
    )
    return {
        "completed_configurations": len(summary_rows),
        "sample_rows": len(all_sample_rows),
        "category_rows": len(all_category_rows),
        "stage_rows": len(all_stage_rows),
    }


def _parser(experiment_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one production COVERT parameter-sensitivity experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment_dir / "experiment_config.json",
        help="Static experiment declaration.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mvtec-root", type=Path, default=DEFAULT_MVTEC_ROOT)
    parser.add_argument("--development-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--categories", nargs="+", help="Main8 category subset")
    parser.add_argument("--only", nargs="+", metavar="CONFIG_ID")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--sample-key",
        help="Exact defective quick sample in CATEGORY/SAMPLE_ID form.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Isolated output root for a quick validation run.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--sample-workers", type=int, default=DEFAULT_SAMPLE_WORKERS
    )
    parser.add_argument("--query-workers", type=int, default=DEFAULT_QUERY_WORKERS)
    return parser


def run_cli(experiment_dir: Path, argv: Sequence[str] | None = None) -> int:
    experiment_dir = Path(experiment_dir).resolve()
    args = _parser(experiment_dir).parse_args(argv)
    if args.sample_workers < 1:
        raise ValueError("--sample-workers must be at least 1")
    if args.query_workers == 0 or args.query_workers < -1:
        raise ValueError("--query-workers must be -1 or a positive integer")
    if args.force and not args.only:
        raise ValueError("--force requires explicit --only CONFIG_ID target(s)")
    if args.sample_key and not args.quick:
        raise ValueError("--sample-key is available only with --quick")
    if args.output_root is not None and not args.quick:
        raise ValueError("--output-root is available only with --quick")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else None
    )
    if output_root is not None:
        formal_root = (experiment_dir / "results").resolve()
        try:
            output_root.relative_to(formal_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Explicit --output-root must be outside the experiment's formal "
                "results directory"
            )

    source = args.config.resolve()
    payload = validate_experiment_config(read_experiment_config(source), source)
    experiment_name = str(payload["experiment_name"])
    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    validate_global_config_count()
    categories = tuple(args.categories or payload["pipeline"]["categories"])
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Categories must be a non-empty list without duplicates")
    if any(category not in MAIN8 for category in categories):
        raise ValueError("Only final benchmark categories are allowed")
    selected = _select_configurations(payload, args.only, args.quick)
    dataset_root = args.data_root.expanduser().resolve()
    mvtec_root = args.mvtec_root.expanduser().resolve()
    dev10_groups, dev10_metadata = load_dev10_groups(
        args.development_manifest.expanduser().resolve()
    )
    samples, discovery_failures = discover_samples(
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
        categories=categories,
    )
    if args.quick:
        samples = [select_quick_sample(samples, categories, args.sample_key)]
    if discovery_failures:
        details = "; ".join(
            f"{item['category']}/{item['sample_id']}:{item['failure_type']}"
            for item in discovery_failures
        )
        raise RuntimeError(f"Dataset discovery failed: {details}")
    if not samples:
        raise RuntimeError("No defective samples were discovered")
    if any(_is_good(Path(sample.pc_path)) for sample in samples):
        raise AssertionError("Good sample entered the defective sensitivity cohort")

    counts = {
        category: sum(sample.category == category for sample in samples)
        for category in categories
    }
    implementation = implementation_metadata()
    implementation_hash = str(implementation["implementation_hash"])
    default_hash = config_sha256(__import__("covert_sample").DEFAULT_CONFIG)
    cohort_hash = sha256_json(
        [
            {
                "dataset": sample.dataset,
                "category": sample.category,
                "sample_id": sample.sample_id,
                "sample_identity_hash": sample.sample_identity_hash,
            }
            for sample in samples
        ]
    )

    print(f"[sensitivity] experiment={experiment_name}")
    print(f"[sensitivity] pipeline=covert_sample.run_covert_sample")
    print(f"[sensitivity] mode={'quick' if args.quick else 'formal_or_custom'}")
    print(f"[sensitivity] categories={list(categories)}")
    print(f"[sensitivity] defective_sample_counts={counts}")
    print(f"[sensitivity] configurations={[item['id'] for item in selected]}")
    print(f"[sensitivity] sample_workers={args.sample_workers} query_workers={args.query_workers}")
    print(f"[sensitivity] production_default_config_hash={default_hash}")
    print(f"[sensitivity] implementation_hash={implementation_hash}")
    print(
        "[sensitivity] implementation_fingerprint_schema="
        f"{implementation['implementation_fingerprint_schema']}"
    )
    print(
        f"[sensitivity] implementation_file_count="
        f"{implementation['implementation_file_count']}"
    )
    print(f"[sensitivity] cohort_hash={cohort_hash}")
    print(f"[sensitivity] dev10_source={dev10_metadata['source']}")
    for item in selected:
        _, audit = build_sensitivity_config(
            item["changes"], allowed_fields=definition["sweep_fields"]
        )
        print(
            f"[sensitivity] plan {item['id']}: requested={audit['requested_values']} "
            f"config_hash={audit['sensitivity_config_hash']}"
        )
    if args.dry_run:
        print("[sensitivity] dry-run complete; no inference or output write performed")
        return 0

    result_base = _result_base(
        experiment_dir,
        quick=args.quick,
        categories=categories,
        output_root=output_root,
    )
    result_base.mkdir(parents=True, exist_ok=True)
    logs_dir = result_base / "logs" if output_root is not None else experiment_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    launched_at = datetime.now().astimezone().isoformat()
    log_path = logs_dir / f"run_{timestamp}.log"
    snapshot_path = logs_dir / f"run_snapshot_{timestamp}.json"
    snapshot = {
        "schema": "covert-sensitivity-v2-run",
        "experiment_name": experiment_name,
        "launched_at": launched_at,
        "static_config_path": str(source),
        "static_config_sha256": sha256_json(payload),
        "selected_configurations": selected,
        "categories": list(categories),
        "run_good_samples": False,
        "quick": bool(args.quick),
        "sample_workers": int(args.sample_workers),
        "query_workers": int(args.query_workers),
        "dataset_root": str(dataset_root),
        "mvtec_root": str(mvtec_root),
        "sample_counts": counts,
        "sample_count_total": len(samples),
        "cohort_hash": cohort_hash,
        "dev10": dev10_metadata,
        "production_default_config_hash": default_hash,
        **implementation,
        "production_file_hashes": production_file_sha256(),
        "result_base": str(result_base),
    }
    _atomic_write_json(snapshot_path, snapshot)

    exit_code = 0
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        stdout_tee = _TeeStream(sys.stdout, log_handle)
        stderr_tee = _TeeStream(sys.stderr, log_handle)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"[sensitivity] launched_at={launched_at}")
            print(f"[sensitivity] run_snapshot={snapshot_path}")
            for index, item in enumerate(selected, start=1):
                print(
                    f"[sensitivity] ({index}/{len(selected)}) starting {item['id']}"
                )
                try:
                    rows, _, runtime, status = _run_one_configuration(
                        experiment_name=experiment_name,
                        definition=definition,
                        item=item,
                        samples=samples,
                        dev10_groups=dev10_groups,
                        result_base=result_base,
                        implementation=implementation,
                        cohort_hash=cohort_hash,
                        categories=categories,
                        quick=args.quick,
                        sample_workers=args.sample_workers,
                        query_workers=args.query_workers,
                        force=args.force,
                    )
                    failures = sum(row.get("status") != "success" for row in rows)
                    print(
                        f"[sensitivity] {item['id']} status={status} "
                        f"samples={len(rows)} failures={failures} runtime={runtime:.3f}s"
                    )
                    if failures:
                        exit_code = 1
                        if args.fail_fast:
                            break
                except Exception as exc:
                    exit_code = 1
                    print(
                        f"[sensitivity] {item['id']} failed loudly: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    if args.fail_fast:
                        break

            output_counts = _rebuild_experiment_outputs(
                payload=payload,
                definition=definition,
                result_base=result_base,
                implementation_hash=implementation_hash,
                implementation_fingerprint_schema=int(
                    implementation["implementation_fingerprint_schema"]
                ),
            )
            print(f"[sensitivity] rebuilt_outputs={output_counts}")
            print(f"[sensitivity] result_base={result_base}")
            print(f"[sensitivity] log={log_path}")
    return exit_code


__all__ = [
    "SampleSpec",
    "discover_samples",
    "run_cli",
    "select_quick_sample",
]
