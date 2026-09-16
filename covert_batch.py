"""Parallel, resumable paper-benchmark runner for :mod:`covert_sample`."""

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Mapping, Sequence

import numpy as np

from covert_sample import (
    DEFAULT_CONFIG,
    CovertPipelineConfig,
    config_from_dict,
    config_hash,
    config_to_dict,
    run_covert_sample,
)
from experiments.benchmark_categories import (
    BENCHMARK_CATEGORIES,
    MVTEC_3D_AD_CATEGORIES,
    REAL3D_AD_CATEGORIES,
    is_mvtec_category,
    validate_benchmark_category,
)


# ============================================================
# BATCH RUN CONFIG
# ============================================================
# These values control only discovery/orchestration/output for
# ``python covert_batch.py``.  Detector parameters come exclusively from the
# imported covert_sample.DEFAULT_CONFIG.
SUPPORTED_CATEGORIES = BENCHMARK_CATEGORIES
CATEGORIES = BENCHMARK_CATEGORIES
DATASET_ROOT = Path("data/real3d_ad")
MVTEC_ROOT = Path("data/mvtec_3d_ad_converted")
OUTPUT_DIR = Path(r"outputs\reports\covert_main8_sobj_timing")
SAMPLE_WORKERS = 1
QUERY_WORKERS = 1
SAVE_FINAL_MASKS = False
RESUME = False
RETRY_FAILED = False
MAX_SAMPLES_PER_CATEGORY: int | None = None
RUN_GOOD_SAMPLES = True
SHOW_PROGRESS = True
PROGRESS_REFRESH_SECONDS = 5.0

# Backward-compatible names for callers of the first consolidation package.
DEFAULT_DATASET_ROOT = DATASET_ROOT
DEFAULT_MVTEC_ROOT = MVTEC_ROOT
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
POINT_SUFFIXES = (".pcd", ".ply", ".npy", ".txt", ".xyz", ".csv")

PRODUCTION_IMPLEMENTATION_FILES = (
    "covert_sample.py",
    "covert_batch.py",
    "vast/covert/_backbone.py",
    "vast/covert/_frontend.py",
    "vast/covert/_gtr.py",
    "vast/covert/_pipeline.py",
    "vast/covert/_point_io.py",
    "vast/covert/_score_compat.py",
    "vast/covert/_veto.py",
    "vast/boundary/ac_boundary.py",
    "vast/boundary/boundary_expand.py",
    "vast/boundary/graph_closing.py",
    "vast/scoring/component_filter.py",
    "vast/restoration/laplacian_inpaint.py",
)


def implementation_hash() -> str:
    """Hash the production detector and its direct in-repository dependencies."""

    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for relative in PRODUCTION_IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_fingerprint(
    config: CovertPipelineConfig,
    *,
    query_workers: int,
    sample_workers: int,
    timing_run: bool,
    resolved_implementation_hash: str | None = None,
) -> str:
    payload = {
        "config_hash": config_hash(config),
        "implementation_hash": (
            implementation_hash()
            if resolved_implementation_hash is None
            else str(resolved_implementation_hash)
        ),
        "execution": {
            "query_workers": int(query_workers),
            "sample_workers": int(sample_workers),
            "timing_run": bool(timing_run),
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SampleSpec:
    dataset: str
    category: str
    sample_id: str
    pc_path: str
    gt_path: str | None
    sample_kind: str = "defective"


def _is_good_sample(path: Path) -> bool:
    return "good" in path.stem.strip().lower().split("_")


def _validate_category(category: str) -> str:
    return validate_benchmark_category(category)


def _is_mvtec_category(category: str) -> bool:
    return is_mvtec_category(category)


def discover_samples(
    *,
    dataset_root: str | Path,
    mvtec_root: str | Path,
    categories: Sequence[str] = CATEGORIES,
) -> tuple[list[SampleSpec], list[dict[str, Any]]]:
    """Discover defective samples in deterministic selected-category order."""

    dataset_root, mvtec_root = Path(dataset_root), Path(mvtec_root)
    specs: list[SampleSpec] = []
    failures: list[dict[str, Any]] = []
    for category in categories:
        _validate_category(category)
        roots = (
            (
                ("mvtec3d", mvtec_root / category),
                ("mvtec3d", dataset_root / "MvTec" / category),
            )
            if _is_mvtec_category(category)
            else (("real3d", dataset_root / category),)
        )
        resolved = next(
            ((dataset, root) for dataset, root in roots if root.is_dir()),
            None,
        )
        if resolved is None:
            failures.append({
                "dataset": "mvtec3d" if _is_mvtec_category(category) else "real3d",
                "category": category, "sample_id": "",
                "status": "discovery_error", "failure_type": "MissingCategoryDirectory",
                "failure_message": f"{dataset_root} | {mvtec_root}",
            })
            continue
        dataset, category_root = resolved
        test_root, gt_root = category_root / "test", category_root / "gt"
        if not test_root.is_dir() or not gt_root.is_dir():
            failures.append({
                "dataset": dataset, "category": category, "sample_id": "",
                "status": "discovery_error", "failure_type": "MissingTestOrGtDirectory",
                "failure_message": str(category_root),
            })
            continue
        gt_files = sorted(
            path for path in gt_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".npy"}
        )
        for gt_path in gt_files:
            if _is_good_sample(gt_path):
                continue
            sample = next(
                (
                    test_root / f"{gt_path.stem}{suffix}"
                    for suffix in POINT_SUFFIXES
                    if (test_root / f"{gt_path.stem}{suffix}").is_file()
                ),
                None,
            )
            if sample is None:
                failures.append({
                    "dataset": dataset, "category": category,
                    "sample_id": gt_path.stem, "status": "discovery_error",
                    "failure_type": "MissingPointCloud",
                    "failure_message": str(test_root / f"{gt_path.stem}.pcd"),
                })
                continue
            specs.append(SampleSpec(
                dataset=dataset, category=category, sample_id=sample.stem,
                pc_path=str(sample), gt_path=str(gt_path),
            ))
    category_order = {category: index for index, category in enumerate(categories)}
    specs.sort(
        key=lambda item: (
            category_order[item.category], item.sample_id, item.pc_path
        )
    )
    return specs, failures


def _point_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in POINT_SUFFIXES
        ),
        key=lambda path: (path.name.lower(), str(path)),
    )


def _mvtec_manifest_good_names(category_root: Path) -> set[str]:
    """Return converted good filenames declared by the local MVTec manifest."""

    manifest_path = category_root / "conversion_manifest.json"
    if not manifest_path.is_file():
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid MVTec records list: {manifest_path}")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if str(record.get("defect_type", "")).strip().lower() != "good":
            continue
        output_pcd = str(record.get("output_pcd", "")).strip()
        if output_pcd:
            names.add(Path(output_pcd).name.lower())
    return names


def discover_good_samples(
    *,
    dataset_root: str | Path,
    mvtec_root: str | Path,
    categories: Sequence[str] = CATEGORIES,
) -> tuple[list[SampleSpec], list[dict[str, Any]]]:
    """Discover label-free good samples without admitting defective samples.

    The local Real3D conversion stores ``*_good*.pcd`` beside defective files
    in ``test`` even though other Real3D layouts use a sibling ``good``
    directory, so both verified layouts are supported.  The local MVTec
    conversion is identified first by ``conversion_manifest.json`` and then by
    its native ``*_good`` name.  Every accepted file must lack a same-stem GT.
    """

    dataset_root, mvtec_root = Path(dataset_root), Path(mvtec_root)
    category_order = {category: index for index, category in enumerate(categories)}
    specs: list[SampleSpec] = []
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for category in categories:
        _validate_category(category)
        is_mvtec = _is_mvtec_category(category)
        dataset = "mvtec3d" if is_mvtec else "real3d"
        roots = (
            (mvtec_root / category, dataset_root / "MvTec" / category)
            if is_mvtec
            else (dataset_root / category,)
        )
        category_root = next((root for root in roots if root.is_dir()), None)
        if category_root is None:
            failures.append({
                "dataset": dataset,
                "category": category,
                "sample_id": "",
                "status": "discovery_error",
                "failure_type": "MissingCategoryDirectory",
                "failure_message": " | ".join(str(root) for root in roots),
            })
            continue

        test_root = category_root / "test"
        gt_root = category_root / "gt"
        candidates: dict[str, Path] = {}
        good_root = category_root / "good"
        for path in _point_files(good_root):
            candidates[str(path.resolve()).lower()] = path

        test_files = _point_files(test_root)
        if is_mvtec:
            try:
                manifest_names = _mvtec_manifest_good_names(category_root)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                failures.append({
                    "dataset": dataset,
                    "category": category,
                    "sample_id": "",
                    "status": "discovery_error",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                })
                manifest_names = set()
            for path in test_files:
                manifest_good = path.name.lower() in manifest_names
                if manifest_good or _is_good_sample(path):
                    candidates[str(path.resolve()).lower()] = path
        else:
            for path in test_files:
                if _is_good_sample(path):
                    candidates[str(path.resolve()).lower()] = path

        accepted_in_category = 0
        for path in sorted(candidates.values(), key=lambda item: item.name.lower()):
            gt_candidates = [
                gt_root / f"{path.stem}{suffix}"
                for suffix in (".txt", ".csv", ".npy")
            ]
            matching_gt = next((item for item in gt_candidates if item.is_file()), None)
            if matching_gt is not None:
                failures.append({
                    "dataset": dataset,
                    "category": category,
                    "sample_id": path.stem,
                    "status": "discovery_error",
                    "failure_type": "GoodSampleHasGroundTruth",
                    "failure_message": str(matching_gt),
                })
                continue
            identity = (dataset, category, path.stem)
            if identity in seen:
                failures.append({
                    "dataset": dataset,
                    "category": category,
                    "sample_id": path.stem,
                    "status": "discovery_error",
                    "failure_type": "DuplicateGoodSample",
                    "failure_message": str(path),
                })
                continue
            seen.add(identity)
            accepted_in_category += 1
            specs.append(SampleSpec(
                dataset=dataset,
                category=category,
                sample_id=path.stem,
                pc_path=str(path),
                gt_path=None,
                sample_kind="good",
            ))
        if accepted_in_category == 0:
            failures.append({
                "dataset": dataset,
                "category": category,
                "sample_id": "",
                "status": "discovery_error",
                "failure_type": "NoGoodSamples",
                "failure_message": str(category_root),
            })

    specs.sort(key=lambda item: (
        category_order[item.category], item.sample_id, item.pc_path
    ))
    return specs, failures


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    error: OSError | None = None
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            error = exc
            time.sleep(0.025 * (2**attempt))
    try:
        temporary.unlink(missing_ok=True)
    finally:
        if error is not None:
            raise error


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str)
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0.0:
        return "calculating"
    total = int(seconds)
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    return f"{days:02d}.{hours:02d}:{minutes:02d}:{secs:02d}"


class _BatchProgressReporter:
    """Publish one live terminal line and an external-monitor JSON snapshot."""

    def __init__(
        self,
        *,
        path: Path,
        total: int,
        initial_rows: Sequence[Mapping[str, Any]],
        discovery_failures: Sequence[Mapping[str, Any]],
        show_terminal: bool,
        refresh_seconds: float,
    ) -> None:
        self.path = Path(path)
        self.total = int(total)
        self.rows = [dict(row) for row in initial_rows]
        self.discovery_failures = [dict(row) for row in discovery_failures]
        self.show_terminal = bool(show_terminal)
        self.refresh_seconds = float(refresh_seconds)
        self.started_perf = time.perf_counter()
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.status = "running"
        self.last_completed_sample: str | None = None
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._terminal_width = 0
        self._terminal_active = False
        self._write_warning_emitted = False

    @staticmethod
    def _sample_name(row: Mapping[str, Any]) -> str:
        values = tuple(str(row.get(name, "")).strip() for name in (
            "dataset", "category", "sample_id"
        ))
        return "/".join(value for value in values if value)

    def _snapshot_locked(self) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started_perf, 0.0)
        completed = len(self.rows)
        succeeded = sum(row.get("success") is True for row in self.rows)
        failed_rows = [row for row in self.rows if row.get("success") is not True]
        resumed = sum(bool(row.get("resumed", False)) for row in self.rows)
        executed_completed = completed - resumed
        pending = max(self.total - completed, 0)
        throughput = (
            float(60.0 * executed_completed / elapsed)
            if executed_completed and elapsed > 0.0
            else 0.0
        )
        eta_seconds = (
            float(60.0 * pending / throughput) if throughput > 0.0 else None
        )
        return {
            "status": self.status,
            "process_id": os.getpid(),
            "started_at": self.started_at,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "total": self.total,
            "completed": completed,
            "succeeded": int(succeeded),
            "failed": len(failed_rows),
            "pending": pending,
            "resumed": int(resumed),
            "executed_completed": int(executed_completed),
            "progress_percent": (
                float(100.0 * completed / self.total) if self.total else 100.0
            ),
            "elapsed_seconds": float(elapsed),
            "throughput_samples_per_minute": throughput,
            "eta_seconds": eta_seconds,
            "last_completed_sample": self.last_completed_sample,
            "failed_samples": [self._sample_name(row) for row in failed_rows],
            "discovery_failure_count": len(self.discovery_failures),
            "discovery_failed_samples": [
                self._sample_name(row) for row in self.discovery_failures
            ],
        }

    def _terminal_text(self, payload: Mapping[str, Any]) -> str:
        status_names = {
            "running": "running",
            "finalizing": "finalizing",
            "completed": "completed",
            "interrupted": "interrupted",
            "failed": "failed",
        }
        return (
            f"[Covert Batch] {status_names.get(str(payload['status']), payload['status'])} | "
            f"{payload['completed']}/{payload['total']} "
            f"({float(payload['progress_percent']):6.2f}%) | "
            f"ok={payload['succeeded']} fail={payload['failed']} "
            f"pending={payload['pending']} | "
            f"elapsed={_format_duration(float(payload['elapsed_seconds']))} | "
            f"{float(payload['throughput_samples_per_minute']):.2f} samples/min | "
            f"ETA={_format_duration(payload['eta_seconds'])}"
        )

    def publish(self, *, final: bool = False) -> dict[str, Any]:
        with self._lock:
            payload = self._snapshot_locked()
            try:
                _atomic_write_json(self.path, payload)
            except OSError as exc:
                if self.show_terminal and not self._write_warning_emitted:
                    if self._terminal_active:
                        print()
                        self._terminal_active = False
                    print(f"[Covert Batch] Progress-file update warning: {exc}")
                    self._write_warning_emitted = True
            if self.show_terminal:
                text = self._terminal_text(payload)
                self._terminal_width = max(self._terminal_width, len(text))
                print(
                    "\r" + text.ljust(self._terminal_width),
                    end="\n" if final else "",
                    flush=True,
                )
                self._terminal_active = not final
            return payload

    def _refresh_loop(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            self.publish()

    def start(self) -> None:
        if self.show_terminal:
            print("[Covert Batch] Starting paper batch")
            print(f"[Covert Batch] Progress file: {self.path.resolve()}")
            print("[Covert Batch] Press Ctrl+C to stop safely and keep completed results.")
        self.publish()
        self._thread = Thread(
            target=self._refresh_loop,
            name="covert-batch-progress",
            daemon=True,
        )
        self._thread.start()

    def record(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            copied = dict(row)
            self.rows.append(copied)
            self.last_completed_sample = self._sample_name(copied)
        self.publish()

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = str(status)

    def finish(self, status: str) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.refresh_seconds * 2.0, 1.0))
        self.set_status(status)
        return self.publish(final=True)


def _interrupt_process_pool(executor: ProcessPoolExecutor) -> None:
    """Terminate workers and return without waiting on Python 3.10 pool threads."""

    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):  # Python 3.14+
        terminate_workers()
        return
    processes = list(getattr(executor, "_processes", {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    try:
        # Do not call Future.cancel() here.  Python 3.10's manager thread marks
        # every pending future with BrokenProcessPool after a worker is
        # terminated; pre-cancelling those futures races set_exception() and
        # raises InvalidStateError.  Also do not wait for executor threads here:
        # real Windows workloads can leave them blocked after worker termination.
        # The CLI writes partial reports and then uses a flushed process-level
        # exit specifically for the controlled-interrupt path.
        executor.shutdown(wait=False, cancel_futures=False)
    except TypeError:  # pragma: no cover - Python < 3.9 compatibility.
        executor.shutdown(wait=False)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
    error: OSError | None = None
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            error = exc
            time.sleep(0.025 * (2**attempt))
    temporary.unlink(missing_ok=True)
    if error is not None:
        raise error


def _sample_dir(output_dir: Path, sample: SampleSpec) -> Path:
    return output_dir / "sample_artifacts" / sample.dataset / sample.category / sample.sample_id


def _read_resume_row(
    output_dir: Path,
    sample: SampleSpec,
    expected_fingerprint: str,
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
    if row.get("run_fingerprint") != expected_fingerprint:
        return None
    if row.get("success") is True:
        row["resumed"] = True
        return row
    if not retry_failed:
        row["resumed"] = True
        return row
    return None


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    sample = SampleSpec(**payload["sample"])
    config = config_from_dict(payload["config"])
    output_dir = Path(payload["output_dir"])
    expected_hash = config_hash(config)
    expected_implementation_hash = str(payload["implementation_hash"])
    expected_fingerprint = str(payload["run_fingerprint"])
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
        )
        row = {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "success",
            "success": True,
            **dict(getattr(result, "object_scores", {})),
            "candidate_structure_readouts": list(
                result.audit.get("object_screening", {}).get(
                    "candidate_readouts", ()
                )
            ),
            **dict(result.metrics),
            "input_load_seconds": result.timings.input_load_seconds,
            "downsampling_seconds": getattr(
                result.timings, "downsampling_seconds", 0.0
            ),
            "algorithm_seconds_excluding_downsampling": getattr(
                result.timings, "algorithm_seconds_excluding_downsampling",
                result.timings.pipeline_seconds,
            ),
            "pipeline_seconds": result.timings.pipeline_seconds,
            "inference_total_seconds": result.timings.inference_total_seconds,
            "gt_load_seconds": result.timings.gt_load_seconds,
            "evaluation_seconds": result.timings.evaluation_seconds,
            "failure_type": "",
            "failure_message": "",
            "config_hash": result.config_hash,
            "implementation_hash": expected_implementation_hash,
            "run_fingerprint": expected_fingerprint,
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
        if result.config_hash != expected_hash:
            raise AssertionError("Worker detector config hash changed.")
        artifact_dir = _sample_dir(output_dir, sample)
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
            "tp": "", "fp": "", "fn": "", "precision": "", "recall": "", "f1": "",
            "S_obj": "", "S_obj_max_candidate_Sp85": "",
            "S_obj_max_candidate_Sp90": "", "S_obj_max_candidate_Sp95": "",
            "n_restoration_attempts": "", "n_restored_candidates": "",
            "candidate_structure_readouts": [],
            "input_load_seconds": "", "downsampling_seconds": "",
            "algorithm_seconds_excluding_downsampling": "",
            "pipeline_seconds": "", "inference_total_seconds": "",
            "gt_load_seconds": "", "evaluation_seconds": "",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "config_hash": expected_hash,
            "implementation_hash": expected_implementation_hash,
            "run_fingerprint": expected_fingerprint,
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
    _atomic_write_json(_sample_dir(output_dir, sample) / "metrics.json", row)
    return row


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _aggregate_one(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("success") is True]
    tp = sum(int(row["tp"]) for row in success)
    fp = sum(int(row["fp"]) for row in success)
    fn = sum(int(row["fn"]) for row in success)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    micro_f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    latencies = [float(row["inference_total_seconds"]) for row in success]
    return {
        "name": name,
        "sample_count": len(rows),
        "success_count": len(success),
        "failure_count": len(rows) - len(success),
        "sample_macro_f1": _safe_mean([float(row["f1"]) for row in success]),
        "pooled_tp": tp,
        "pooled_fp": fp,
        "pooled_fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "mean_inference_seconds": _safe_mean(latencies),
        "median_inference_seconds": float(statistics.median(latencies)) if latencies else 0.0,
    }


def build_aggregate_tables(
    rows: Sequence[Mapping[str, Any]], categories: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    category_rows: list[dict[str, Any]] = []
    for category in categories:
        category_rows.append(_aggregate_one(
            category, [row for row in rows if row.get("category") == category]
        ))
        category_rows[-1]["category"] = category

    def group(name: str, members: Sequence[str]) -> dict[str, Any]:
        selected = [row for row in rows if row.get("category") in set(members)]
        value = _aggregate_one(name, selected)
        by_category = [row for row in category_rows if row["category"] in set(members)]
        value["class_balanced_sample_macro_f1"] = _safe_mean(
            [float(row["sample_macro_f1"]) for row in by_category]
        )
        value["categories"] = ",".join(members)
        return value

    full_benchmark = tuple(categories) == tuple(BENCHMARK_CATEGORIES)
    group_rows = [group("Selected", tuple(categories))]
    if full_benchmark:
        group_rows = [
            group("Real3D-AD", REAL3D_AD_CATEGORIES),
            group("MVTec 3D-AD", MVTEC_3D_AD_CATEGORIES),
            group("Eight-category benchmark", BENCHMARK_CATEGORIES),
        ]
    aggregate = {
        "selected_categories": list(categories),
        "category_count": len(categories),
        "selected": group("Selected", tuple(categories)),
        "benchmark_groups_emitted": full_benchmark,
        "benchmark_cohort": "eight_category" if full_benchmark else "subset",
    }
    if full_benchmark:
        aggregate["benchmark_groups"] = {
            "Real3D-AD": group("Real3D-AD", REAL3D_AD_CATEGORIES),
            "MVTec 3D-AD": group("MVTec 3D-AD", MVTEC_3D_AD_CATEGORIES),
            "Eight-category benchmark": group(
                "Eight-category benchmark", BENCHMARK_CATEGORIES
            ),
        }
    return category_rows, group_rows, aggregate


def _limit_per_category(
    specs: Sequence[SampleSpec], limit: int | None
) -> list[SampleSpec]:
    if limit is None:
        return list(specs)
    retained: list[SampleSpec] = []
    counts: dict[str, int] = {}
    for sample in specs:
        current = counts.get(sample.category, 0)
        if current < limit:
            retained.append(sample)
            counts[sample.category] = current + 1
    return retained


def _good_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run one label-free good sample and return only all-normal FPR fields."""

    sample = SampleSpec(**payload["sample"])
    config = config_from_dict(payload["config"])
    expected_hash = config_hash(config)
    expected_implementation_hash = str(payload["implementation_hash"])
    expected_fingerprint = str(payload["run_fingerprint"])
    started = time.perf_counter()
    try:
        if sample.sample_kind != "good" or sample.gt_path is not None:
            raise ValueError("Good worker requires sample_kind='good' and gt_path=None.")
        result = run_covert_sample(
            sample.pc_path,
            gt_path=None,
            config=config,
            query_workers=int(payload["query_workers"]),
            visualize=False,
            output_dir=None,
            verbose=False,
        )
        final_mask = np.asarray(result.final_mask, dtype=bool).reshape(-1)
        if final_mask.size == 0:
            raise RuntimeError("COVERT returned an empty final mask.")
        if result.gt_mask is not None or result.metrics:
            raise RuntimeError("Good inference unexpectedly performed GT evaluation.")
        fp = int(np.count_nonzero(final_mask))
        num_points = int(final_mask.size)
        tn = int(num_points - fp)
        row = {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "success",
            "success": True,
            "num_points": num_points,
            "fp": fp,
            "tn": tn,
            "fpr": float(fp / num_points),
            **dict(getattr(result, "object_scores", {})),
            "candidate_structure_readouts": list(
                result.audit.get("object_screening", {}).get(
                    "candidate_readouts", ()
                )
            ),
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
            "inference_total_seconds": float(
                result.timings.inference_total_seconds
            ),
            "failure_type": "",
            "failure_message": "",
            "config_hash": result.config_hash,
            "implementation_hash": expected_implementation_hash,
            "run_fingerprint": expected_fingerprint,
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
        if result.config_hash != expected_hash:
            raise AssertionError("Good worker detector config hash changed.")
    except Exception as exc:
        row = {
            "dataset": sample.dataset,
            "category": sample.category,
            "sample_id": sample.sample_id,
            "status": "failure",
            "success": False,
            "num_points": None,
            "fp": None,
            "tn": None,
            "fpr": None,
            "S_obj": None,
            "S_obj_max_candidate_Sp85": None,
            "S_obj_max_candidate_Sp90": None,
            "S_obj_max_candidate_Sp95": None,
            "n_restoration_attempts": None,
            "n_restored_candidates": None,
            "candidate_structure_readouts": [],
            "input_load_seconds": None,
            "downsampling_seconds": None,
            "algorithm_seconds_excluding_downsampling": None,
            "pipeline_seconds": None,
            "inference_total_seconds": None,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "config_hash": expected_hash,
            "implementation_hash": expected_implementation_hash,
            "run_fingerprint": expected_fingerprint,
            "worker_wall_seconds": float(time.perf_counter() - started),
            "resumed": False,
        }
    return row


def build_good_metrics(
    rows: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    *,
    discovery_failures: Sequence[Mapping[str, Any]] = (),
    status: str = "completed",
) -> dict[str, Any]:
    """Build good-only FPR aggregates; defective rows are rejected."""

    normalized_rows = [dict(row) for row in rows]
    category_payload: dict[str, dict[str, Any]] = {}
    pooled_fpr: list[float] = []
    valid_category_means: list[float] = []
    for category in categories:
        selected = [row for row in normalized_rows if row.get("category") == category]
        successful = [row for row in selected if row.get("success") is True]
        values = [float(row["fpr"]) for row in successful]
        inference_times = [
            float(row["inference_total_seconds"]) for row in successful
        ]
        downsampling_times = [
            float(row["downsampling_seconds"]) for row in successful
        ]
        algorithm_times = [
            float(row["algorithm_seconds_excluding_downsampling"])
            for row in successful
        ]
        pooled_fpr.extend(values)
        mean_fpr = float(statistics.mean(values)) if values else None
        p95_fpr = float(np.percentile(values, 95.0)) if values else None
        if mean_fpr is not None:
            valid_category_means.append(mean_fpr)
        category_payload[category] = {
            "sample_count": len(selected),
            "success_count": len(successful),
            "failure_count": len(selected) - len(successful),
            "mean_fpr": mean_fpr,
            "p95_fpr": p95_fpr,
            "mean_inference_total_seconds": _safe_mean(inference_times),
            "mean_downsampling_seconds": _safe_mean(downsampling_times),
            "mean_algorithm_seconds_excluding_downsampling": _safe_mean(
                algorithm_times
            ),
        }
    successes = [row for row in normalized_rows if row.get("success") is True]
    success_inference_times = [
        float(row["inference_total_seconds"]) for row in successes
    ]
    success_downsampling_times = [
        float(row["downsampling_seconds"]) for row in successes
    ]
    success_algorithm_times = [
        float(row["algorithm_seconds_excluding_downsampling"])
        for row in successes
    ]
    return {
        "status": status,
        "metric_definition": "predicted_defect_points / num_working_points",
        "object_score_definition": (
            "S_obj(Pq) = max candidate pseudo-stress Pq over restored "
            "candidates; 0.0 when none"
        ),
        "object_score_percentiles": [85, 90, 95],
        "canonical_object_score_field": "S_obj_max_candidate_Sp90",
        "gt_policy": "no GT passed to inference; all-normal truth is evaluation-only",
        "samples": normalized_rows,
        "categories": category_payload,
        "overall": {
            "category_count": len(categories),
            "valid_category_count": len(valid_category_means),
            "sample_count": len(normalized_rows),
            "success_count": len(successes),
            "failure_count": len(normalized_rows) - len(successes),
            "class_balanced_mean_fpr": (
                float(statistics.mean(valid_category_means))
                if valid_category_means else None
            ),
            "pooled_p95_fpr": (
                float(np.percentile(pooled_fpr, 95.0)) if pooled_fpr else None
            ),
            "mean_inference_total_seconds": _safe_mean(success_inference_times),
            "mean_downsampling_seconds": _safe_mean(
                success_downsampling_times
            ),
            "mean_algorithm_seconds_excluding_downsampling": _safe_mean(
                success_algorithm_times
            ),
        },
        "discovery_failures": [dict(item) for item in discovery_failures],
    }


def _build_object_score_timing_rows(
    defective_rows: Sequence[Mapping[str, Any]],
    good_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build one analysis-ready table for sample screening and runtime."""

    combined: list[dict[str, Any]] = []
    for source_rows, label, label_name in (
        (defective_rows, 1, "defective"),
        (good_rows, 0, "good"),
    ):
        for source in source_rows:
            if label == 1:
                predicted = source.get("predicted_positive_points", "")
                working = source.get("working_point_count", "")
            else:
                predicted = source.get("fp", "")
                working = source.get("num_points", "")
            ratio: float | str = ""
            if predicted not in (None, "") and working not in (None, ""):
                denominator = int(working)
                if denominator > 0:
                    ratio = float(int(predicted) / denominator)
            combined.append(
                {
                    "dataset": source.get("dataset", ""),
                    "category": source.get("category", ""),
                    "sample_id": source.get("sample_id", ""),
                    "label": label,
                    "label_name": label_name,
                    "status": source.get("status", ""),
                    "success": source.get("success", False),
                    "S_obj": source.get("S_obj", ""),
                    "S_obj_max_candidate_Sp85": source.get(
                        "S_obj_max_candidate_Sp85", ""
                    ),
                    "S_obj_max_candidate_Sp90": source.get(
                        "S_obj_max_candidate_Sp90", ""
                    ),
                    "S_obj_max_candidate_Sp95": source.get(
                        "S_obj_max_candidate_Sp95", ""
                    ),
                    "n_restoration_attempts": source.get(
                        "n_restoration_attempts", ""
                    ),
                    "n_restored_candidates": source.get(
                        "n_restored_candidates", ""
                    ),
                    "candidate_structure_readouts_json": json.dumps(
                        source.get("candidate_structure_readouts", ()),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "predicted_defect_points": predicted,
                    "working_point_count": working,
                    "predicted_defect_ratio": ratio,
                    "input_load_seconds": source.get("input_load_seconds", ""),
                    "downsampling_seconds": source.get(
                        "downsampling_seconds", ""
                    ),
                    "algorithm_seconds_excluding_downsampling": source.get(
                        "algorithm_seconds_excluding_downsampling", ""
                    ),
                    "pipeline_seconds": source.get("pipeline_seconds", ""),
                    "inference_total_seconds": source.get(
                        "inference_total_seconds", ""
                    ),
                    "worker_wall_seconds": source.get("worker_wall_seconds", ""),
                    "config_hash": source.get("config_hash", ""),
                    "implementation_hash": source.get("implementation_hash", ""),
                    "run_fingerprint": source.get("run_fingerprint", ""),
                }
            )
    combined.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["category"]),
            str(row["sample_id"]),
            int(row["label"]),
        )
    )
    return combined


def _read_good_resume_rows(
    path: Path,
    specs: Sequence[SampleSpec],
    expected_fingerprint: str,
    *,
    retry_failed: bool,
) -> list[dict[str, Any]]:
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
    dataset_root: str | Path,
    mvtec_root: str | Path,
    categories: Sequence[str],
    output_dir: Path,
    config: CovertPipelineConfig,
    sample_workers: int,
    query_workers: int,
    resume: bool,
    retry_failed: bool,
    max_samples_per_category: int | None,
    show_progress: bool,
    implementation_hash_value: str,
    run_fingerprint: str,
) -> tuple[dict[str, Any], bool]:
    specs, discovery_failures = discover_good_samples(
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
        categories=categories,
    )
    specs = _limit_per_category(specs, max_samples_per_category)
    result_path = output_dir / "good_metrics.json"
    rows = _read_good_resume_rows(
        result_path,
        specs,
        run_fingerprint,
        retry_failed=retry_failed,
    ) if resume else []
    resumed = {
        (str(row["dataset"]), str(row["category"]), str(row["sample_id"]))
        for row in rows
    }
    pending = [
        sample for sample in specs
        if (sample.dataset, sample.category, sample.sample_id) not in resumed
    ]

    def publish(status: str) -> dict[str, Any]:
        rows.sort(key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("category", "")),
            str(row.get("sample_id", "")),
        ))
        payload = build_good_metrics(
            rows,
            categories,
            discovery_failures=discovery_failures,
            status=status,
        )
        _atomic_write_json(result_path, payload)
        return payload

    publish("running")
    interrupted = False
    worker_payloads = [
        {
            "sample": dataclasses.asdict(sample),
            "config": config_to_dict(config),
            "query_workers": int(query_workers),
            "implementation_hash": implementation_hash_value,
            "run_fingerprint": run_fingerprint,
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
                f"[Covert good {completed:03d}/{len(specs):03d}] "
                f"{row['category']}/{row['sample_id']} {row['status']}",
                flush=True,
            )

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


def _runtime_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    discovered: int,
    sample_workers: int,
    batch_wall_seconds: float,
    expected_hash: str,
    executed_rows: Sequence[Mapping[str, Any]] | None = None,
    resolved_implementation_hash: str = "",
    run_fingerprint: str = "",
    timing_run: bool = False,
) -> dict[str, Any]:
    all_success = [row for row in rows if row.get("success") is True]
    executed = list(executed_rows) if executed_rows is not None else [
        row for row in rows if not bool(row.get("resumed", False))
    ]
    executed_success = [row for row in executed if row.get("success") is True]
    inference = [float(row["inference_total_seconds"]) for row in executed_success]
    pipelines = [float(row["pipeline_seconds"]) for row in executed_success]
    loads = [float(row["input_load_seconds"]) for row in executed_success]
    downsampling = [
        float(row.get("downsampling_seconds", 0.0)) for row in executed_success
    ]
    algorithm_excluding_downsampling = [
        float(
            row.get(
                "algorithm_seconds_excluding_downsampling",
                row["pipeline_seconds"],
            )
        )
        for row in executed_success
    ]
    all_inference = [float(row["inference_total_seconds"]) for row in all_success]
    percentile95 = float(np.percentile(inference, 95.0)) if inference else 0.0
    executed_count = len(executed)
    return {
        "total_discovered_samples": int(discovered),
        "success_count": len(all_success),
        "failure_count": len(rows) - len(all_success),
        "executed_this_run_count": executed_count,
        "resumed_count": int(sum(bool(row.get("resumed", False)) for row in rows)),
        "executed_success_count": len(executed_success),
        "executed_failure_count": executed_count - len(executed_success),
        "no_samples_executed_this_run": executed_count == 0,
        "sample_workers": int(sample_workers),
        "timing_run": bool(timing_run),
        "batch_wall_seconds": float(batch_wall_seconds),
        "sum_sample_inference_seconds": float(sum(inference)),
        "mean_sample_inference_seconds": _safe_mean(inference),
        "median_sample_inference_seconds": float(statistics.median(inference)) if inference else 0.0,
        "p95_sample_inference_seconds": percentile95,
        "min_sample_inference_seconds": float(min(inference)) if inference else 0.0,
        "max_sample_inference_seconds": float(max(inference)) if inference else 0.0,
        "mean_pipeline_seconds": _safe_mean(pipelines),
        "mean_input_load_seconds": _safe_mean(loads),
        "mean_downsampling_seconds": _safe_mean(downsampling),
        "mean_algorithm_seconds_excluding_downsampling": _safe_mean(
            algorithm_excluding_downsampling
        ),
        "all_rows_mean_sample_inference_seconds": _safe_mean(all_inference),
        "amortized_batch_wall_seconds_per_executed_sample": (
            float(batch_wall_seconds / executed_count) if executed_count else 0.0
        ),
        "amortized_batch_wall_seconds_per_successful_sample": (
            float(batch_wall_seconds / len(executed_success))
            if executed_success else 0.0
        ),
        "samples_per_second_wall_clock": (
            float(executed_count / batch_wall_seconds)
            if executed_count and batch_wall_seconds > 0 else 0.0
        ),
        "config_hash": expected_hash,
        "implementation_hash": resolved_implementation_hash,
        "run_fingerprint": run_fingerprint,
    }


def compare_reference(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def metric_path(root: Path, stem: str) -> Path:
        normal = root / f"{stem}_metrics.csv"
        paper = root / f"paper_{stem}_metrics.csv"
        if normal.is_file():
            return normal
        if paper.is_file():
            return paper
        raise FileNotFoundError(f"No {normal.name} or {paper.name} in {root}")

    current = read_rows(metric_path(output_dir, "sample"))
    reference = read_rows(metric_path(reference_dir, "sample"))
    key = lambda row: (row.get("dataset", ""), row.get("category", ""), row.get("sample_id", ""))
    current_index, reference_index = {key(row): row for row in current}, {key(row): row for row in reference}
    common = sorted(set(current_index) & set(reference_index))
    exact = 0
    mismatches: list[dict[str, Any]] = []
    for item in common:
        left, right = current_index[item], reference_index[item]
        fields = ("tp", "fp", "fn", "f1")
        equal = all(
            (
                int(float(left[field])) == int(float(right[field]))
                if field != "f1"
                else float(left[field]) == float(right[field])
            )
            for field in fields
        )
        if equal:
            exact += 1
        else:
            mismatches.append({"sample": item, **{f"current_{f}": left.get(f) for f in fields}, **{f"reference_{f}": right.get(f) for f in fields}})
    current_categories = {
        row.get("category") or row.get("name"): row
        for row in read_rows(metric_path(output_dir, "category"))
    }
    reference_categories = {
        row.get("category") or row.get("name"): row
        for row in read_rows(metric_path(reference_dir, "category"))
    }
    category_deltas: list[dict[str, Any]] = []
    for category in sorted(set(current_categories) & set(reference_categories)):
        left, right = current_categories[category], reference_categories[category]
        category_deltas.append(
            {
                "category": category,
                "sample_macro_f1_delta": float(left["sample_macro_f1"])
                - float(right["sample_macro_f1"]),
                "micro_f1_delta": float(left["micro_f1"])
                - float(right["micro_f1"]),
                "pooled_tp_delta": int(float(left["pooled_tp"]))
                - int(float(right["pooled_tp"])),
                "pooled_fp_delta": int(float(left["pooled_fp"]))
                - int(float(right["pooled_fp"])),
                "pooled_fn_delta": int(float(left["pooled_fn"]))
                - int(float(right["pooled_fn"])),
            }
        )

    def benchmark_row(root: Path) -> dict[str, str] | None:
        try:
            path = metric_path(root, "group")
        except FileNotFoundError:
            return None
        expected = ",".join(BENCHMARK_CATEGORIES)
        return next(
            (row for row in read_rows(path) if row.get("categories") == expected),
            None,
        )

    current_benchmark = benchmark_row(output_dir)
    reference_benchmark = benchmark_row(reference_dir)
    benchmark_delta = None
    if current_benchmark is not None and reference_benchmark is not None:
        benchmark_delta = {
            "class_balanced_sample_macro_f1_delta": float(
                current_benchmark["class_balanced_sample_macro_f1"]
            )
            - float(reference_benchmark["class_balanced_sample_macro_f1"]),
            "micro_f1_delta": float(current_benchmark["micro_f1"])
            - float(reference_benchmark["micro_f1"]),
        }

    result = {
        "common_sample_count": len(common),
        "exact_sample_count": exact,
        "missing_from_current": sorted(set(reference_index) - set(current_index)),
        "missing_from_reference": sorted(set(current_index) - set(reference_index)),
        "sample_mismatches": mismatches,
        "category_metric_deltas": category_deltas,
        "benchmark_delta": benchmark_delta,
    }
    _atomic_write_json(output_dir / "reference_comparison.json", result)
    return result


def run_covert_batch(
    *,
    dataset_root: str | Path = DATASET_ROOT,
    mvtec_root: str | Path = MVTEC_ROOT,
    categories: Sequence[str] = CATEGORIES,
    output_dir: str | Path = OUTPUT_DIR,
    config: CovertPipelineConfig = DEFAULT_CONFIG,
    sample_workers: int = SAMPLE_WORKERS,
    query_workers: int = QUERY_WORKERS,
    resume: bool = RESUME,
    retry_failed: bool = RETRY_FAILED,
    save_final_masks: bool = SAVE_FINAL_MASKS,
    max_samples_per_category: int | None = MAX_SAMPLES_PER_CATEGORY,
    run_good_samples: bool = RUN_GOOD_SAMPLES,
    timing_run: bool = False,
    show_progress: bool = SHOW_PROGRESS,
    progress_refresh_seconds: float = PROGRESS_REFRESH_SECONDS,
) -> dict[str, Any]:
    if sample_workers < 1:
        raise ValueError("sample_workers must be at least 1.")
    if max_samples_per_category is not None and max_samples_per_category < 1:
        raise ValueError("max_samples_per_category must be at least 1 or None.")
    if timing_run and resume:
        raise ValueError("--timing-run forbids resume; use a fresh output directory.")
    if not np.isfinite(progress_refresh_seconds) or progress_refresh_seconds <= 0.0:
        raise ValueError("progress_refresh_seconds must be a positive finite value.")
    if timing_run:
        save_final_masks = False
    categories = tuple(_validate_category(value) for value in categories)
    if len(set(categories)) != len(categories):
        raise ValueError("categories must not contain duplicates.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_hash = config_hash(config)
    resolved_implementation_hash = implementation_hash()
    run_fingerprint = build_run_fingerprint(
        config,
        query_workers=query_workers,
        sample_workers=sample_workers,
        timing_run=timing_run,
        resolved_implementation_hash=resolved_implementation_hash,
    )
    _atomic_write_json(output_dir / "config.json", {
        "source": (
            "covert_sample.DEFAULT_CONFIG"
            if config is DEFAULT_CONFIG
            else "explicit_config_object"
        ),
        "config_hash": expected_hash,
        "implementation_hash": resolved_implementation_hash,
        "run_fingerprint": run_fingerprint,
        "execution": {
            "query_workers": int(query_workers),
            "sample_workers": int(sample_workers),
            "timing_run": bool(timing_run),
        },
        "statistics_collection": {
            "object_score_definition": (
                "S_obj(Pq) = max candidate pseudo-stress Pq over restored "
                "candidates; 0.0 when none"
            ),
            "object_score_percentiles": [85, 90, 95],
            "canonical_paper_object_score": "S_obj_max_candidate_Sp90",
            "downsampling_timing": "wall time of the existing fps_downsample call",
            "algorithm_timing_excluding_downsampling": (
                "pipeline_seconds - downsampling_seconds"
            ),
        },
        "config": config_to_dict(config),
        "categories": list(categories),
    })
    specs, discovery_failures = discover_samples(
        dataset_root=dataset_root, mvtec_root=mvtec_root, categories=categories
    )
    specs = _limit_per_category(specs, max_samples_per_category)
    rows: list[dict[str, Any]] = []
    pending: list[SampleSpec] = []
    for sample in specs:
        previous = _read_resume_row(
            output_dir, sample, run_fingerprint, retry_failed=retry_failed
        ) if resume else None
        if previous is None:
            pending.append(sample)
        else:
            rows.append(previous)

    executed_rows: list[dict[str, Any]] = []
    batch_start = time.perf_counter()
    reporter = _BatchProgressReporter(
        path=output_dir / "batch_progress.json",
        total=len(specs),
        initial_rows=rows,
        discovery_failures=discovery_failures,
        show_terminal=show_progress,
        refresh_seconds=progress_refresh_seconds,
    )
    reporter.start()
    interrupted = False
    executor: ProcessPoolExecutor | None = None
    try:
        if pending:
            worker_payloads = [
                {
                    "sample": dataclasses.asdict(sample),
                    "config": config_to_dict(config),
                    "output_dir": str(output_dir),
                    "query_workers": int(query_workers),
                    "save_final_masks": bool(save_final_masks),
                    "implementation_hash": resolved_implementation_hash,
                    "run_fingerprint": run_fingerprint,
                }
                for sample in pending
            ]
            if sample_workers == 1:
                for payload in worker_payloads:
                    row = _worker(payload)
                    row.setdefault("implementation_hash", resolved_implementation_hash)
                    row.setdefault("run_fingerprint", run_fingerprint)
                    row.setdefault("resumed", False)
                    rows.append(row)
                    executed_rows.append(row)
                    reporter.record(row)
            else:
                executor = ProcessPoolExecutor(max_workers=sample_workers)
                future_to_payload: dict[Any, Mapping[str, Any]] = {}
                for payload in worker_payloads:
                    future = executor.submit(_worker, payload)
                    future_to_payload[future] = payload
                for future in as_completed(future_to_payload):
                    payload = future_to_payload[future]
                    try:
                        row = future.result()
                    except Exception as exc:  # _worker normally contains failures.
                        sample = payload["sample"]
                        row = {
                            "dataset": sample["dataset"], "category": sample["category"],
                            "sample_id": sample["sample_id"], "status": "failure", "success": False,
                            "failure_type": type(exc).__name__, "failure_message": str(exc),
                            "config_hash": expected_hash,
                            "implementation_hash": resolved_implementation_hash,
                            "run_fingerprint": run_fingerprint,
                            "resumed": False,
                        }
                    row.setdefault("implementation_hash", resolved_implementation_hash)
                    row.setdefault("run_fingerprint", run_fingerprint)
                    row.setdefault("resumed", False)
                    rows.append(row)
                    executed_rows.append(row)
                    reporter.record(row)
                executor.shutdown(wait=True)
                executor = None
    except KeyboardInterrupt:
        interrupted = True
        # Stop the periodic writer before touching the ProcessPool so the
        # terminal cannot continue refreshing while workers are being stopped.
        reporter.finish("interrupted")
        if executor is not None:
            _interrupt_process_pool(executor)
            executor = None
    except BaseException:
        if executor is not None:
            _interrupt_process_pool(executor)
            executor = None
        reporter.finish("failed")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    batch_wall_seconds = time.perf_counter() - batch_start
    if interrupted:
        reporter.set_status("interrupted")
    else:
        reporter.set_status("finalizing")
        reporter.publish()
    try:
        rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("category", "")), str(row.get("sample_id", ""))))
        category_rows, group_rows, aggregate = build_aggregate_tables(rows, categories)
        runtime = _runtime_summary(
            rows, discovered=len(specs), sample_workers=sample_workers,
            batch_wall_seconds=batch_wall_seconds, expected_hash=expected_hash,
            executed_rows=executed_rows,
            resolved_implementation_hash=resolved_implementation_hash,
            run_fingerprint=run_fingerprint,
            timing_run=timing_run,
        )
        runtime["interrupted"] = bool(interrupted)
        runtime["pending_count"] = max(len(specs) - len(rows), 0)
        runtime["progress_file"] = str(output_dir / "batch_progress.json")
        write_start = time.perf_counter()
        _atomic_write_csv(output_dir / "sample_metrics.csv", rows)
        _atomic_write_csv(
            output_dir / "sample_object_scores_and_timings.csv",
            _build_object_score_timing_rows(rows),
        )
        _atomic_write_csv(output_dir / "category_metrics.csv", category_rows)
        _atomic_write_csv(output_dir / "group_metrics.csv", group_rows)
        failures = [dict(row) for row in discovery_failures] + [dict(row) for row in rows if row.get("success") is not True]
        _atomic_write_csv(output_dir / "failures.csv", failures)
        _atomic_write_json(output_dir / "aggregate_metrics.json", aggregate)
        runtime["report_write_seconds"] = float(time.perf_counter() - write_start)
        _atomic_write_json(output_dir / "runtime_summary.json", runtime)
    except KeyboardInterrupt:
        interrupted = True
        reporter.finish("interrupted")
        raise
    except BaseException:
        reporter.finish("failed")
        raise
    reporter.finish("interrupted" if interrupted else "completed")
    result_payload = {
        "sample_rows": rows,
        "category_rows": category_rows,
        "group_rows": group_rows,
        "aggregate": aggregate,
        "runtime": runtime,
        "failures": failures,
        "interrupted": interrupted,
    }
    if run_good_samples and not interrupted:
        good_metrics, good_interrupted = _run_good_batch(
            dataset_root=dataset_root,
            mvtec_root=mvtec_root,
            categories=categories,
            output_dir=output_dir,
            config=config,
            sample_workers=sample_workers,
            query_workers=query_workers,
            resume=resume,
            retry_failed=retry_failed,
            max_samples_per_category=max_samples_per_category,
            show_progress=show_progress,
            implementation_hash_value=resolved_implementation_hash,
            run_fingerprint=run_fingerprint,
        )
        result_payload["good_metrics"] = good_metrics
        result_payload["interrupted"] = bool(good_interrupted)
        _atomic_write_csv(
            output_dir / "sample_object_scores_and_timings.csv",
            _build_object_score_timing_rows(
                rows, good_metrics.get("samples", ())
            ),
        )
    return result_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DATASET_ROOT,
        help="Override module-level DATASET_ROOT.",
    )
    parser.add_argument(
        "--mvtec-root", type=Path, default=MVTEC_ROOT,
        help="Override module-level MVTEC_ROOT.",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(CATEGORIES), choices=SUPPORTED_CATEGORIES,
        help="Override module-level CATEGORIES.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Override module-level OUTPUT_DIR.",
    )
    parser.add_argument(
        "--sample-workers", type=int, default=SAMPLE_WORKERS,
        help="Override module-level SAMPLE_WORKERS.",
    )
    parser.add_argument(
        "--query-workers", type=int, default=QUERY_WORKERS,
        help="Override module-level QUERY_WORKERS.",
    )
    parser.add_argument(
        "--max-samples-per-category", type=int,
        default=MAX_SAMPLES_PER_CATEGORY,
        help="Override module-level MAX_SAMPLES_PER_CATEGORY.",
    )
    parser.add_argument(
        "--good-samples", action=argparse.BooleanOptionalAction,
        default=RUN_GOOD_SAMPLES,
        help="Run the additional good-sample FPR cohort.",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help="Show live terminal progress; batch_progress.json is always written.",
    )
    parser.add_argument(
        "--progress-refresh-seconds", type=float,
        default=PROGRESS_REFRESH_SECONDS,
        help="Override module-level PROGRESS_REFRESH_SECONDS.",
    )
    parser.add_argument(
        "--save-final-masks", action=argparse.BooleanOptionalAction,
        default=SAVE_FINAL_MASKS,
        help="Override module-level SAVE_FINAL_MASKS.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=RESUME,
        help="Override module-level RESUME.",
    )
    parser.add_argument(
        "--retry-failed", action=argparse.BooleanOptionalAction,
        default=RETRY_FAILED,
        help="Override module-level RETRY_FAILED.",
    )
    parser.add_argument(
        "--timing-run", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--compare-reference-dir", type=Path, help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_covert_batch(
        dataset_root=args.dataset_root,
        mvtec_root=args.mvtec_root,
        categories=tuple(args.categories),
        output_dir=args.output_dir,
        config=DEFAULT_CONFIG,
        sample_workers=args.sample_workers,
        query_workers=args.query_workers,
        resume=args.resume,
        retry_failed=args.retry_failed,
        save_final_masks=args.save_final_masks,
        max_samples_per_category=args.max_samples_per_category,
        run_good_samples=args.good_samples,
        timing_run=args.timing_run,
        show_progress=args.progress,
        progress_refresh_seconds=args.progress_refresh_seconds,
    )
    if result.get("interrupted"):
        return 130
    if args.compare_reference_dir:
        result["reference_comparison"] = compare_reference(
            args.output_dir, args.compare_reference_dir
        )
    return 0


def _force_cli_interrupt_exit(exit_code: int = 130) -> None:
    """Exit after flushed partial reports when Python 3.10 pool threads linger."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError):
            pass
    os._exit(int(exit_code))


if __name__ == "__main__":
    resolved_exit_code = main()
    if resolved_exit_code == 130:
        _force_cli_interrupt_exit(resolved_exit_code)
    raise SystemExit(resolved_exit_code)
