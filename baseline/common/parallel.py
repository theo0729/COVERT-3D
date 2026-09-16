"""Process-based sample scheduling shared by baseline tuning and final runs."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


EXECUTION_ONLY_EVALUATION_KEYS = frozenset(
    {"sample_workers", "inner_thread_limit", "numba_threads"}
)


def _normalize_inner_thread_limit(inner_thread_limit: int | None) -> int | None:
    if inner_thread_limit is None:
        return None
    inner = int(inner_thread_limit)
    if inner <= 0:
        raise ValueError("inner_thread_limit must be positive when configured.")
    return inner


def _normalize_numba_threads(numba_threads: int | None) -> int | None:
    if numba_threads is None:
        return None
    threads = int(numba_threads)
    if threads <= 0:
        raise ValueError("numba_threads must be positive when configured.")
    return threads


def validate_execution_settings(
    sample_workers: int,
    inner_thread_limit: int | None,
) -> tuple[int, int | None]:
    workers = int(sample_workers)
    inner = _normalize_inner_thread_limit(inner_thread_limit)
    if workers <= 0:
        raise ValueError("sample_workers must be positive.")
    return workers, inner


def config_without_execution_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove scheduling-only fields before computing an algorithm config hash."""

    semantic = copy.deepcopy(dict(config))
    evaluation = semantic.get("evaluation")
    if isinstance(evaluation, dict):
        for key in EXECUTION_ONLY_EVALUATION_KEYS:
            evaluation.pop(key, None)
    semantic.pop("execution", None)
    return semantic


@contextmanager
def limited_inner_threads(limit: int | None):
    """Optionally cap BLAS/OpenMP/MKL pools inside one sample worker."""

    inner = _normalize_inner_thread_limit(limit)
    if inner is None:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError(
            "threadpoolctl is required for sample-level multiprocessing so nested "
            "BLAS/OpenMP pools can be bounded."
        ) from exc
    with threadpool_limits(limits=inner):
        yield


@contextmanager
def limited_numba_threads(limit: int | None):
    """Optionally limit only Numba workers and restore the previous setting."""

    threads = _normalize_numba_threads(limit)
    if threads is None:
        yield
        return
    import numba

    previous = int(numba.get_num_threads())
    try:
        numba.set_num_threads(threads)
        yield
    finally:
        numba.set_num_threads(previous)


def iter_process_results(
    tasks: Sequence[Mapping[str, Any]],
    worker: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    sample_workers: int,
) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any] | None, BaseException | None]]:
    """Yield completed independent sample tasks without imposing completion order."""

    workers, _ = validate_execution_settings(sample_workers, 1)
    if not tasks:
        return
    if workers == 1:
        for task in tasks:
            try:
                yield task, worker(task), None
            except BaseException as exc:  # preserve one error row per scheduled sample
                yield task, None, exc
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        future_to_task = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                yield task, future.result(), None
            except BaseException as exc:
                yield task, None, exc


def _run_pointsgrade_repetitions(
    points: np.ndarray,
    parameters: Mapping[str, Any],
    repetitions: int,
    inner_thread_limit: int | None = 1,
):
    from baseline.PointSGRADE.pointsgrade import run_pointsgrade_baseline

    results = [
        run_pointsgrade_baseline(
            points,
            lambda0=float(parameters["lambda0"]),
            epsilon=float(parameters["epsilon"]),
            num_neighbor=int(parameters["num_neighbor"]),
            num_neighbor_max=int(parameters["num_neighbor_max"]),
            threshold_angle=float(parameters["threshold_angle"]),
            threshold_dist=float(parameters["threshold_dist"]),
            sigma=float(parameters["sigma"]),
            random_state=int(parameters["random_state"]),
            inner_thread_limit=inner_thread_limit,
        )
        for _ in range(int(repetitions))
    ]
    first_mask = np.asarray(results[0].pred_mask, dtype=bool)
    if not all(
        np.array_equal(first_mask, np.asarray(result.pred_mask, dtype=bool))
        for result in results[1:]
    ):
        raise RuntimeError(
            "PointSGRADE predictions differ across identical seeded repetitions."
        )
    runtime_values = [float(result.runtime_sec) for result in results]
    diagnostics = dict(results[0].diagnostics)
    diagnostics["reproducibility"] = {
        "repetitions": int(repetitions),
        "prediction_identical": True,
        "runtime_sec_by_repetition": runtime_values,
    }
    for key in (
        "initialization_sec",
        "graph_matrix_sec",
        "optimization_sec",
        "vendor_stage_sum_sec",
        "wall_total_sec",
    ):
        values = [float(result.diagnostics[key]) for result in results]
        diagnostics[key] = float(np.mean(values))
        diagnostics[f"{key}_by_repetition"] = values
    from baseline.common.benchmark import BaselineResult

    return BaselineResult(
        pred_mask=first_mask,
        runtime_sec=float(np.mean(runtime_values)),
        diagnostics=diagnostics,
    )


def run_frozen_method(
    method: str,
    points: np.ndarray,
    parameters: Mapping[str, Any],
    *,
    inner_thread_limit: int | None,
    numba_threads: int | None = None,
    pointsgrade_repetitions: int = 1,
):
    """Dispatch one frozen method while separating configured/effective concurrency."""

    normalized = str(method)
    inner = _normalize_inner_thread_limit(inner_thread_limit)
    numba_limit = _normalize_numba_threads(numba_threads)
    if normalized == "SOR":
        from baseline.SOR.sor import run_sor_baseline

        result = run_sor_baseline(
            points,
            nb_neighbors=int(parameters["nb_neighbors"]),
            std_ratio=float(parameters["std_ratio"]),
        )
    elif normalized == "FPFH_IF":
        from baseline.FPFH_IF.fpfh_if import run_fpfh_if_baseline

        configured_n_jobs = int(parameters["n_jobs"])
        effective_n_jobs = configured_n_jobs if inner is None else inner
        result = run_fpfh_if_baseline(
            points,
            normal_k=int(parameters["normal_k"]),
            fpfh_k=int(parameters["fpfh_k"]),
            orient_normals=bool(parameters["orient_normals"]),
            n_estimators=int(parameters["n_estimators"]),
            max_samples=parameters["max_samples"],
            contamination=parameters["contamination"],
            random_state=int(parameters["random_state"]),
            n_jobs=effective_n_jobs,
        )
        result.diagnostics.update(
            {
                "configured_n_jobs": int(parameters["n_jobs"]),
                "effective_n_jobs": effective_n_jobs,
            }
        )
    elif normalized == "RG":
        from baseline.RG.region_growing import run_region_growing_baseline

        configured_query_workers = int(parameters["query_workers"])
        effective_query_workers = (
            configured_query_workers if inner is None else inner
        )
        result = run_region_growing_baseline(
            points,
            smoothing_k=int(parameters["smoothing_k"]),
            smoothing_iterations=int(parameters["smoothing_iterations"]),
            normal_k=int(parameters["normal_k"]),
            region_neighbor_k=int(parameters["region_neighbor_k"]),
            smoothness_threshold_deg=float(parameters["smoothness_threshold_deg"]),
            curvature_threshold=float(parameters["curvature_threshold"]),
            min_cluster_size=int(parameters["min_cluster_size"]),
            query_workers=effective_query_workers,
        )
        result.diagnostics.update(
            {
                "configured_query_workers": int(parameters["query_workers"]),
                "effective_query_workers": effective_query_workers,
            }
        )
    elif normalized == "PointSGRADE":
        with limited_numba_threads(numba_limit):
            result = _run_pointsgrade_repetitions(
                points,
                parameters,
                int(pointsgrade_repetitions),
                inner_thread_limit=inner,
            )
    else:
        raise ValueError(f"Unsupported baseline method: {method!r}.")
    controls = dict(result.diagnostics.get("execution_controls", {}))
    controls["inner_thread_limit"] = inner
    if normalized == "PointSGRADE" and numba_limit is not None:
        controls["numba_threads"] = numba_limit
        controls["reason"] = (
            "Numba restricted for race-safe PointSGRADE reverse mapping; "
            "BLAS/OpenMP/MKL governed separately by inner_thread_limit."
        )
    result.diagnostics["execution_controls"] = controls
    return result


def execute_final_cache_sample(task: Mapping[str, Any]) -> dict[str, Any]:
    """Load and evaluate one final-run defective cache sample in a worker."""

    from baseline.common.benchmark import compute_binary_metrics, load_cache_file

    cache_path = Path(str(task["cache_path"]))
    method = str(task["method"])
    inner = _normalize_inner_thread_limit(task["inner_thread_limit"])
    numba_threads = _normalize_numba_threads(task.get("numba_threads"))
    try:
        with limited_inner_threads(inner):
            cache = load_cache_file(
                cache_path,
                expected_num_points=int(task["num_working_points"]),
            )
            result = run_frozen_method(
                method,
                cache["points"],
                task["parameters"],
                inner_thread_limit=inner,
                numba_threads=numba_threads,
                pointsgrade_repetitions=int(task.get("pointsgrade_repetitions", 1)),
            )
        pred_mask = np.asarray(result.pred_mask, dtype=bool).reshape(-1)
        if pred_mask.shape != cache["gt_mask"].shape:
            raise RuntimeError(
                f"{method} output shape {pred_mask.shape} does not match cached GT "
                f"{cache['gt_mask'].shape}."
            )
        metrics = compute_binary_metrics(pred_mask, cache["gt_mask"])
        diagnostics = dict(result.diagnostics)
        diagnostics["sample_workers"] = int(task["sample_workers"])
        diagnostics["inner_thread_limit"] = inner
        payload = {
            "status": "ok",
            "error": "",
            "cache_path": str(cache_path),
            "cache": {
                "category": cache["category"],
                "sample_id": cache["sample_id"],
                "preprocess_config_hash": cache["preprocess_config_hash"],
                "fps_selected_index_hash": cache["fps_selected_index_hash"],
            },
            "pred_mask": pred_mask,
            "gt_mask": np.asarray(cache["gt_mask"], dtype=bool),
            "metrics": metrics,
            "runtime_sec": float(result.runtime_sec),
            "diagnostics": diagnostics,
            "points_dtype": str(cache["points"].dtype),
            "coordinate_min": np.asarray(cache["points"]).min(axis=0).tolist(),
            "coordinate_max": np.asarray(cache["points"]).max(axis=0).tolist(),
        }
        if method == "PointSGRADE":
            from baseline.PointSGRADE.pointsgrade import hash_points

            payload["cache_points_sha256"] = hash_points(cache["points"])
        return payload
    except BaseException as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "failure_type": type(exc).__name__,
            "cache_path": str(cache_path),
            "category": cache_path.parent.name,
            "sample_id": cache_path.stem,
        }


def execute_good_sample(task: Mapping[str, Any]) -> dict[str, Any]:
    """Run one label-free good sample in a worker process."""

    from baseline.common.benchmark import PreprocessConfig
    from vast.data import io as data_io
    from vast.data import preprocess

    sample = dict(task["sample"])
    config = PreprocessConfig(**dict(task["preprocess_config"]))
    config.validate()
    method = str(task["method"])
    inner = _normalize_inner_thread_limit(task["inner_thread_limit"])
    numba_threads = _normalize_numba_threads(task.get("numba_threads"))
    try:
        if sample.get("sample_kind") != "good" or sample.get("gt_path") is not None:
            raise ValueError("Good evaluation received a non-good SampleSpec.")
        with limited_inner_threads(inner):
            raw_points = np.asarray(
                data_io.load_point_cloud_file(Path(sample["pc_path"])),
                dtype=np.float64,
            )
            sampled_points, fps_indices, _ = preprocess.fps_downsample(
                raw_points,
                num_samples=config.num_working_points,
                gt_mask=None,
                fps_mode=config.fps_mode,
                fps_prefilter_num_points=config.fps_prefilter_num_points,
                seed=config.fps_seed,
                return_fps_summary=True,
            )
            points, _, _, _ = preprocess.isotropic_scale(
                sampled_points, cube_size=config.cube_size
            )
            points = np.asarray(points, dtype=np.float64)
            fps_indices = np.asarray(fps_indices, dtype=np.int64).reshape(-1)
            expected_shape = (config.num_working_points, 3)
            if points.shape != expected_shape:
                raise RuntimeError(
                    f"Good working cloud shape {points.shape}; expected {expected_shape}."
                )
            if np.unique(fps_indices).size != config.num_working_points:
                raise RuntimeError("Good-sample FPS returned duplicate indices.")
            result = run_frozen_method(
                method,
                points,
                task["parameters"],
                inner_thread_limit=inner,
                numba_threads=numba_threads,
            )
        pred_mask = np.asarray(result.pred_mask, dtype=bool).reshape(-1)
        if pred_mask.shape != (config.num_working_points,):
            raise RuntimeError(
                f"{method} output shape {pred_mask.shape}; expected "
                f"{(config.num_working_points,)}."
            )
        fp = int(np.count_nonzero(pred_mask))
        num_points = int(pred_mask.size)
        return {
            "dataset": sample["dataset"],
            "category": sample["category"],
            "sample_id": sample["sample_id"],
            "status": "success",
            "num_points": num_points,
            "fp": fp,
            "tn": int(num_points - fp),
            "fpr": float(fp / num_points),
            "inference_total_seconds": float(result.runtime_sec),
            "failure_type": "",
            "failure_message": "",
        }
    except BaseException as exc:
        return {
            "dataset": sample["dataset"],
            "category": sample["category"],
            "sample_id": sample["sample_id"],
            "status": "failure",
            "num_points": None,
            "fp": None,
            "tn": None,
            "fpr": None,
            "inference_total_seconds": None,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }


__all__ = [
    "config_without_execution_settings",
    "execute_final_cache_sample",
    "execute_good_sample",
    "iter_process_results",
    "limited_inner_threads",
    "limited_numba_threads",
    "run_frozen_method",
    "validate_execution_settings",
]
