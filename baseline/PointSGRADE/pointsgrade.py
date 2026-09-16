"""Thin, label-free adapter around the authors' PointSGRADE solver.

The adapter deliberately does not load a dataset, resample points, normalize
coordinates, or inspect ground truth.  Its only input is the frozen working
point cloud supplied by :mod:`baseline.common`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np
import numba
from threadpoolctl import threadpool_info, threadpool_limits

from baseline.common import BaselineResult


POINTSGRADE_ROOT = Path(__file__).resolve().parent
POINTSGRADE_VENDOR_ROOT = POINTSGRADE_ROOT / "vendor" / "pointSGRADE"
POINTSGRADE_SOURCE_ROOT = POINTSGRADE_VENDOR_ROOT / "Source code"
POINTSGRADE_SOLVER_FILE = (
    POINTSGRADE_SOURCE_ROOT / "utils" / "pointSGRADE_solution_mkl_float32.py"
)

_SOLVER_MODULE: ModuleType | None = None


def hash_points(points: np.ndarray) -> str:
    """Hash point coordinates and ordering in the canonical float64 form."""

    array = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}.")
    # Upstream initialization constructs 200-neighbour patch graphs.
    if array.shape[0] < 201:
        raise ValueError(
            "PointSGRADE requires at least 201 points because its released "
            "initialization uses a fixed 200-neighbour patch graph."
        )
    if not np.isfinite(array).all():
        raise ValueError("points contain NaN or Inf.")
    return np.ascontiguousarray(array)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def load_vendor_solver():
    """Import and return the unmodified upstream ``pointSGRADE_solver``."""

    global _SOLVER_MODULE
    if _SOLVER_MODULE is not None:
        return _SOLVER_MODULE.pointSGRADE_solver

    if not POINTSGRADE_SOLVER_FILE.is_file():
        raise FileNotFoundError(
            f"PointSGRADE solver not found at the frozen vendor path: "
            f"{POINTSGRADE_SOLVER_FILE}"
        )

    existing_utils = sys.modules.get("utils")
    if existing_utils is not None:
        existing_file = getattr(existing_utils, "__file__", None)
        existing_paths = list(getattr(existing_utils, "__path__", []))
        candidates = ([existing_file] if existing_file else []) + existing_paths
        if candidates and not all(
            _is_within(Path(candidate), POINTSGRADE_SOURCE_ROOT)
            for candidate in candidates
        ):
            raise RuntimeError(
                "Cannot safely import the upstream top-level 'utils' package: "
                "an unrelated module with that name is already loaded."
            )

    source_text = str(POINTSGRADE_SOURCE_ROOT)
    inserted = source_text not in sys.path
    old_dont_write_bytecode = sys.dont_write_bytecode
    try:
        # Do not generate __pycache__ files inside the frozen vendor snapshot.
        sys.dont_write_bytecode = True
        if inserted:
            sys.path.insert(0, source_text)
        module = importlib.import_module(
            "utils.pointSGRADE_solution_mkl_float32"
        )
    finally:
        if inserted:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass
        sys.dont_write_bytecode = old_dont_write_bytecode

    module_file = Path(str(module.__file__)).resolve()
    if module_file != POINTSGRADE_SOLVER_FILE.resolve():
        raise RuntimeError(
            f"Imported unexpected PointSGRADE solver: {module_file}"
        )
    solver = getattr(module, "pointSGRADE_solver", None)
    if not callable(solver):
        raise AttributeError("Upstream module does not expose pointSGRADE_solver.")
    _SOLVER_MODULE = module
    return solver


def _execution_controls(inner_thread_limit: int | None) -> dict[str, Any]:
    if inner_thread_limit is not None:
        inner = int(inner_thread_limit)
        return {
            "numba_threads": inner,
            "blas_openmp_threads": inner,
            "reason": "explicit execution-only inner thread limit",
        }
    pool_threads = [
        int(record["num_threads"])
        for record in threadpool_info()
        if record.get("num_threads") is not None
        and int(record["num_threads"]) > 0
    ]
    return {
        "numba_threads": int(numba.get_num_threads()),
        "blas_openmp_threads": max(
            pool_threads,
            default=int(numba.get_num_threads()),
        ),
        "reason": (
            "native/configured internal threading"
        ),
    }


@contextmanager
def _controlled_execution_state(
    seed: int,
    inner_thread_limit: int | None,
) -> Iterator[None]:
    """Control released global RNG state and optionally bound native threads.

    The released ``reverse_mapping`` kernel uses ``prange`` with overlapping
    indexed ``+=`` writes. With more than one Numba worker that is a data race.
    Callers may therefore set an explicit inner-thread limit; ``None`` retains
    native/configured Numba and BLAS/OpenMP/MKL threading without changing any
    vendor formula or source.
    """

    inner = None if inner_thread_limit is None else int(inner_thread_limit)
    if inner is not None and inner <= 0:
        raise ValueError("inner_thread_limit must be positive when configured.")
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    previous_numba_threads = numba.get_num_threads()
    try:
        np.random.seed(seed)
        random.seed(seed)
        if inner is None:
            yield
        else:
            numba.set_num_threads(inner)
            with threadpool_limits(limits=inner):
                yield
    finally:
        if inner is not None:
            numba.set_num_threads(previous_numba_threads)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def run_pointsgrade_baseline(
    points: np.ndarray,
    *,
    lambda0: float = 7.0e-3,
    epsilon: float = 1.0e-3,
    num_neighbor: int = 1500,
    num_neighbor_max: int = 800,
    threshold_angle: float = np.pi / 12.0,
    threshold_dist: float = 0.5,
    sigma: float = 1.0,
    random_state: int = 0,
    inner_thread_limit: int | None = 1,
) -> BaselineResult:
    """Run the official solver and convert upstream label 1 to ``True``.

    The label meaning is taken directly from the released solver: it creates
    an all-zero ``uint8`` label and sets entries to 1 where the recovered
    anomaly displacement norm exceeds ``1e-3``.
    """

    array = _validate_points(points)
    lambda0 = _positive_finite("lambda0", lambda0)
    epsilon = _positive_finite("epsilon", epsilon)
    threshold_angle = _positive_finite("threshold_angle", threshold_angle)
    threshold_dist = _positive_finite("threshold_dist", threshold_dist)
    sigma = _positive_finite("sigma", sigma)
    num_neighbor = int(num_neighbor)
    num_neighbor_max = int(num_neighbor_max)
    random_state = int(random_state)
    if num_neighbor <= 0 or num_neighbor >= array.shape[0]:
        raise ValueError("num_neighbor must be in [1, N - 1].")
    if num_neighbor_max <= 0 or num_neighbor_max > num_neighbor:
        raise ValueError("num_neighbor_max must be in [1, num_neighbor].")
    if random_state < 0:
        raise ValueError("random_state must be non-negative.")

    input_hash = hash_points(array)
    solver = load_vendor_solver()
    started = time.perf_counter()
    with _controlled_execution_state(
        random_state,
        inner_thread_limit,
    ):
        output = solver(
            array,
            lambda0=lambda0,
            epsilon=epsilon,
            num_neighbor=num_neighbor,
            num_neighbor_max=num_neighbor_max,
            threshold_angle=threshold_angle,
            threshold_dist=threshold_dist,
            sigma=sigma,
            show_result=False,
        )
    wall_total_sec = float(time.perf_counter() - started)
    execution_controls = _execution_controls(inner_thread_limit)

    if not isinstance(output, tuple) or len(output) != 6:
        raise RuntimeError(
            "Unexpected upstream return contract; expected six values from "
            "pointSGRADE_solver."
        )
    (
        estimated_label,
        estimated_reference,
        estimated_anomaly_part,
        initialization_sec,
        graph_matrix_sec,
        optimization_sec,
    ) = output

    labels = np.asarray(estimated_label).reshape(-1)
    if labels.shape != (array.shape[0],):
        raise RuntimeError(
            f"PointSGRADE returned label shape {labels.shape}; "
            f"expected {(array.shape[0],)}."
        )
    unique_labels = np.unique(labels)
    if not np.isin(unique_labels, [0, 1]).all():
        raise RuntimeError(
            f"PointSGRADE returned non-binary labels: {unique_labels.tolist()}"
        )
    pred_mask = labels.astype(bool, copy=False)

    reference = np.asarray(estimated_reference)
    anomaly_part = np.asarray(estimated_anomaly_part)
    for name, value in (
        ("estimated reference", reference),
        ("estimated anomaly part", anomaly_part),
    ):
        if value.shape != array.shape:
            raise RuntimeError(
                f"PointSGRADE {name} shape {value.shape}; expected {array.shape}."
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f"PointSGRADE {name} contains NaN or Inf.")

    timings = {
        "initialization_sec": float(initialization_sec),
        "graph_matrix_sec": float(graph_matrix_sec),
        "optimization_sec": float(optimization_sec),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in timings.values()):
        raise RuntimeError(f"PointSGRADE returned invalid stage timings: {timings}")
    vendor_stage_sum_sec = float(sum(timings.values()))

    if hash_points(array) != input_hash:
        raise RuntimeError("The upstream solver mutated input point coordinates/order.")

    parameters: dict[str, Any] = {
        "lambda0": lambda0,
        "epsilon": epsilon,
        "num_neighbor": num_neighbor,
        "num_neighbor_max": num_neighbor_max,
        "threshold_angle": threshold_angle,
        "threshold_dist": threshold_dist,
        "sigma": sigma,
        "random_state": random_state,
    }
    diagnostics = {
        "implementation": "authors_released_pointSGRADE_solver",
        "solver_file": str(POINTSGRADE_SOLVER_FILE),
        "upstream_source_modified": True,
        "upstream_algorithm_modified": False,
        "upstream_source_modification": (
            "CVXPY import made function-local in an inactive synthetic-data helper"
        ),
        "upstream_positive_label": 1,
        "canonical_positive_class": True,
        "source_continuous_output": "recovered anomaly displacement D (N x 3)",
        "source_scalar_magnitude": "norm(D, axis=1)",
        "label_rule": "norm(recovered anomaly displacement) > 1e-3",
        "binary_threshold": 1.0e-3,
        "binary_threshold_origin": "hard-coded native upstream solver rule",
        "binary_threshold_uses_gt": False,
        "wrapper_adds_posthoc_threshold": False,
        "input_points_sha256": input_hash,
        "input_shape": list(array.shape),
        "input_dtype": str(array.dtype),
        "input_coordinate_min": array.min(axis=0).tolist(),
        "input_coordinate_max": array.max(axis=0).tolist(),
        "input_coordinates_unchanged": True,
        "point_order_preserved": True,
        "num_predicted_anomaly": int(np.count_nonzero(pred_mask)),
        "predicted_anomaly_ratio": float(np.mean(pred_mask)),
        "estimated_anomaly_norm_min": float(
            np.linalg.norm(anomaly_part, axis=1).min()
        ),
        "estimated_anomaly_norm_max": float(
            np.linalg.norm(anomaly_part, axis=1).max()
        ),
        **timings,
        "vendor_stage_sum_sec": vendor_stage_sum_sec,
        "wall_total_sec": wall_total_sec,
        "parameters": parameters,
        "random_seed": random_state,
        "execution_controls": execution_controls,
    }
    return BaselineResult(
        pred_mask=pred_mask,
        runtime_sec=wall_total_sec,
        diagnostics=diagnostics,
    )


_DEPENDENCIES = (
    ("numpy", "numpy", "numerical solver", True),
    ("scipy", "scipy", "sparse matrices/eigensolver", True),
    ("numba", "numba", "released JIT kernels", True),
    ("sparse_dot_mkl", "sparse-dot-mkl", "released MKL sparse products", True),
    ("sklearn", "scikit-learn", "nearest-neighbour graph", True),
    ("tqdm", "tqdm", "released optimization progress iterator", True),
    ("threadpoolctl", "threadpoolctl", "adapter reproducibility control", True),
    ("open3d", "open3d", "upstream eager import; visualization disabled", True),
    (
        "cvxpy",
        "cvxpy",
        "optional synthetic-data coefficient-smoothing helper; lazy import",
        False,
    ),
    ("tifffile", "tifffile", "upstream eager import; dataset loader unused", True),
    ("PIL", "Pillow", "upstream eager import; dataset loader unused", True),
    ("bspline", "bspline", "synthetic-data generator only", False),
)


def audit_pointsgrade_dependencies() -> dict[str, Any]:
    """Return an import-path-aware dependency audit without importing vendor code."""

    records = []
    for module_name, distribution_name, role, required in _DEPENDENCIES:
        available = importlib.util.find_spec(module_name) is not None
        try:
            version = importlib.metadata.version(distribution_name) if available else None
        except importlib.metadata.PackageNotFoundError:
            version = "available_without_distribution_metadata" if available else None
        records.append(
            {
                "module": module_name,
                "distribution": distribution_name,
                "role": role,
                "required_for_adapter_import": required,
                "available": available,
                "version": version,
            }
        )
    missing_required = [
        row["distribution"]
        for row in records
        if row["required_for_adapter_import"] and not row["available"]
    ]
    return {
        "status": "ok" if not missing_required else "missing",
        "missing_required": missing_required,
        "dependencies": records,
    }
