"""Training-free FPFH + Isolation Forest point-cloud baseline.

This is our reimplementation following the high-level baseline description in
3D-SONAR; it is not an exact reproduction of an unpublished author pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d
from sklearn.ensemble import IsolationForest

from baseline.common import BaselineResult


def _validate_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("points must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError("points contain NaN or Inf.")
    return array


def compute_fpfh_features(
    points: np.ndarray,
    normal_k: int = 30,
    fpfh_k: int = 100,
    orient_normals: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate normals and return one 33-D Open3D FPFH vector per point."""

    array = _validate_points(points)
    normal_k = int(normal_k)
    fpfh_k = int(fpfh_k)
    if normal_k < 3:
        raise ValueError("normal_k must be at least 3.")
    if fpfh_k < 3:
        raise ValueError("fpfh_k must be at least 3.")
    if normal_k >= array.shape[0] or fpfh_k >= array.shape[0]:
        raise ValueError("normal_k and fpfh_k must be smaller than N.")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(array)

    started = time.perf_counter()
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
    )
    normal_estimation_sec = float(time.perf_counter() - started)

    started = time.perf_counter()
    if orient_normals:
        point_cloud.orient_normals_consistent_tangent_plane(normal_k)
    normal_orientation_sec = float(time.perf_counter() - started)

    normals = np.asarray(point_cloud.normals, dtype=np.float64)
    if normals.shape != array.shape:
        raise RuntimeError(
            f"Open3D returned invalid normal shape: {normals.shape}, expected {array.shape}."
        )
    normal_norms = np.linalg.norm(normals, axis=1)
    invalid_normals = (~np.isfinite(normals).all(axis=1)) | (normal_norms <= 1e-12)
    if np.any(invalid_normals):
        raise RuntimeError(
            f"Normal estimation produced {int(np.count_nonzero(invalid_normals))} "
            "invalid normals."
        )

    started = time.perf_counter()
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=fpfh_k),
    )
    fpfh_sec = float(time.perf_counter() - started)
    features = np.asarray(feature.data, dtype=np.float64).T
    expected_shape = (array.shape[0], 33)
    if features.shape != expected_shape:
        raise RuntimeError(
            f"Open3D returned FPFH shape {features.shape}; expected {expected_shape}."
        )

    diagnostics = {
        "normal_search": "knn",
        "normal_k": normal_k,
        "orient_normals": bool(orient_normals),
        "normal_orientation_method": (
            "consistent_tangent_plane" if orient_normals else "none"
        ),
        "fpfh_search": "knn",
        "fpfh_k": fpfh_k,
        "feature_shape": [int(value) for value in features.shape],
        "num_invalid_normals": int(np.count_nonzero(invalid_normals)),
        "normal_estimation_sec": normal_estimation_sec,
        "normal_orientation_sec": normal_orientation_sec,
        "fpfh_sec": fpfh_sec,
    }
    return features, diagnostics


def _normalize_contamination(value: str | float) -> str | float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized != "auto":
            raise ValueError("String contamination must be 'auto'.")
        return normalized
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 < numeric <= 0.5:
        raise ValueError("Numeric contamination must be within (0, 0.5].")
    return numeric


def fpfh_feature_cache_key(
    *,
    sample_identity: str,
    preprocess_config_hash: str,
    normal_k: int,
    orient_normals: bool,
    fpfh_k: int,
    fps_selected_index_hash: str | None = None,
    open3d_version: str | None = None,
) -> str:
    """Return an auditable key for a reusable per-cloud FPFH descriptor."""

    payload = {
        "schema": "fpfh_feature_cache_v1",
        "sample_identity": str(sample_identity),
        "preprocess_config_hash": str(preprocess_config_hash),
        "normal_k": int(normal_k),
        "normal_orientation_method": (
            "consistent_tangent_plane" if orient_normals else "none"
        ),
        "fpfh_k": int(fpfh_k),
        "open3d_version": str(open3d_version or o3d.__version__),
    }
    # Keep the legacy key byte-for-byte stable when no FPS hash is supplied so
    # the audited v1 intermediate cache remains reusable.  New studies always
    # supply the hash and therefore bind descriptors to the exact working cloud.
    if fps_selected_index_hash is not None:
        payload["fps_selected_index_hash"] = str(fps_selected_index_hash)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _prepare_fpfh_features(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and median-repair invalid descriptor rows explicitly."""

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 33:
        raise ValueError(f"features must have shape (N, 33), got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("features must not be empty.")
    valid_rows = np.isfinite(array).all(axis=1)
    if not np.any(valid_rows):
        raise RuntimeError("All FPFH descriptors are invalid.")
    repaired = array.copy()
    if not np.all(valid_rows):
        repaired[~valid_rows] = np.median(array[valid_rows], axis=0)
    if not np.isfinite(repaired).all():
        raise RuntimeError("FPFH repair left NaN or Inf values.")
    return repaired, valid_rows


def load_or_compute_fpfh_features(
    points: np.ndarray,
    *,
    cache_dir: str | Path,
    sample_identity: str,
    preprocess_config_hash: str,
    normal_k: int,
    fpfh_k: int,
    orient_normals: bool = True,
    fps_selected_index_hash: str | None = None,
    fallback_cache_dirs: Iterable[str | Path] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or atomically create a GT-free FPFH descriptor cache entry."""

    array = _validate_points(points)
    key = fpfh_feature_cache_key(
        sample_identity=sample_identity,
        preprocess_config_hash=preprocess_config_hash,
        normal_k=normal_k,
        orient_normals=orient_normals,
        fpfh_k=fpfh_k,
        fps_selected_index_hash=fps_selected_index_hash,
    )
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"{key}.npz"
    candidates: list[tuple[Path, str, str]] = [(cache_path, key, "v2")]
    fallback_roots = [Path(item).expanduser().resolve() for item in fallback_cache_dirs]
    for fallback_root in fallback_roots:
        candidates.append((fallback_root / f"{key}.npz", key, "parent_v1"))
    if fps_selected_index_hash is not None:
        legacy_key = fpfh_feature_cache_key(
            sample_identity=sample_identity,
            preprocess_config_hash=preprocess_config_hash,
            normal_k=normal_k,
            orient_normals=orient_normals,
            fpfh_k=fpfh_k,
            fps_selected_index_hash=None,
        )
        for fallback_root in fallback_roots:
            candidates.append((fallback_root / f"{legacy_key}.npz", legacy_key, "parent_v1"))

    selected = next(
        ((path, expected_key, source) for path, expected_key, source in candidates if path.is_file()),
        None,
    )
    if selected is not None:
        selected_path, expected_key, source = selected
        with np.load(selected_path, allow_pickle=False) as data:
            cached_key = str(np.asarray(data["cache_key"]).reshape(()).item())
            features = np.asarray(data["features"], dtype=np.float64)
            diagnostics = json.loads(
                str(np.asarray(data["diagnostics_json"]).reshape(()).item())
            )
        if cached_key != expected_key:
            raise RuntimeError(f"FPFH cache-key mismatch: {selected_path}")
        if features.shape != (array.shape[0], 33):
            raise RuntimeError(
                f"Cached FPFH shape {features.shape} does not match {(array.shape[0], 33)}."
            )
        diagnostics.update(
            {
                "feature_cache_key": expected_key,
                "feature_cache_path": str(selected_path),
                "feature_cache_hit": True,
                "feature_cache_source": source,
                "fps_selected_index_hash": fps_selected_index_hash or "legacy_unspecified",
            }
        )
        return features, diagnostics

    features, diagnostics = compute_fpfh_features(
        array,
        normal_k=normal_k,
        fpfh_k=fpfh_k,
        orient_normals=orient_normals,
    )
    payload_diagnostics = dict(diagnostics)
    payload_diagnostics.update(
        {
            "feature_cache_key": key,
            "feature_cache_path": str(cache_path),
            "feature_cache_hit": False,
            "feature_cache_source": "recomputed",
            "fps_selected_index_hash": fps_selected_index_hash or "legacy_unspecified",
        }
    )
    with tempfile.NamedTemporaryFile(
        dir=root, prefix=f".{key}.", suffix=".tmp.npz", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        np.savez_compressed(
            temporary_path,
            features=features,
            cache_key=np.asarray(key),
            diagnostics_json=np.asarray(
                json.dumps(payload_diagnostics, sort_keys=True, separators=(",", ":"))
            ),
        )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return features, payload_diagnostics


def run_isolation_forest_on_features(
    features: np.ndarray,
    *,
    n_estimators: int = 100,
    max_samples: str | int | float = "auto",
    contamination: str | float = "auto",
    random_state: int = 0,
    n_jobs: int | None = -1,
) -> BaselineResult:
    """Fit one Isolation Forest to one cloud's precomputed FPFH features."""

    n_estimators = int(n_estimators)
    random_state = int(random_state)
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if random_state < 0:
        raise ValueError("random_state must be non-negative.")
    contamination = _normalize_contamination(contamination)
    repaired, valid_rows = _prepare_fpfh_features(features)

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    started = time.perf_counter()
    labels = np.asarray(model.fit_predict(repaired), dtype=np.int8).reshape(-1)
    isolation_forest_sec = float(time.perf_counter() - started)
    if labels.shape != (repaired.shape[0],):
        raise RuntimeError(
            f"IsolationForest returned shape {labels.shape}; expected {(repaired.shape[0],)}."
        )
    unexpected_labels = sorted(set(np.unique(labels).tolist()) - {-1, 1})
    if unexpected_labels:
        raise RuntimeError(f"Unexpected IsolationForest labels: {unexpected_labels}")
    pred_mask = labels == -1

    decision_function = np.asarray(
        model.decision_function(repaired), dtype=np.float64
    ).reshape(-1)
    if decision_function.shape != (repaired.shape[0],):
        raise RuntimeError("IsolationForest decision_function shape mismatch.")
    anomaly_score = -decision_function
    if not np.isfinite(anomaly_score).all():
        raise RuntimeError("IsolationForest produced invalid anomaly scores.")
    if not np.array_equal(pred_mask, decision_function < 0.0):
        raise RuntimeError(
            "IsolationForest label direction disagrees with decision_function."
        )

    diagnostics = {
        "implementation": "scikit-learn IsolationForest",
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "contamination": contamination,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "num_invalid_fpfh": int(np.count_nonzero(~valid_rows)),
        "num_predicted_defect": int(np.count_nonzero(pred_mask)),
        "predicted_defect_ratio": float(np.mean(pred_mask)),
        "isolation_forest_sec": isolation_forest_sec,
        "binary_defect_rule": "isolation_forest_label_minus_one",
        "anomaly_score_direction": "higher_is_more_anomalous",
        "anomaly_score_min": float(anomaly_score.min()),
        "anomaly_score_max": float(anomaly_score.max()),
        "anomaly_score_mean": float(anomaly_score.mean()),
        "anomaly_score_std": float(anomaly_score.std(ddof=0)),
    }
    return BaselineResult(
        pred_mask=pred_mask,
        runtime_sec=isolation_forest_sec,
        diagnostics=diagnostics,
    )


def run_fpfh_if_baseline(
    points: np.ndarray,
    normal_k: int = 30,
    fpfh_k: int = 100,
    orient_normals: bool = True,
    n_estimators: int = 100,
    max_samples: str | int | float = "auto",
    contamination: str | float = "auto",
    random_state: int = 0,
    n_jobs: int | None = -1,
) -> BaselineResult:
    """Fit IF to one test cloud's FPFH vectors and return a binary mask.

    This function is deliberately single-cloud and training-free.  Ground truth
    is not accepted as an input.  Invalid FPFH rows, if any, are replaced by the
    coordinate-wise median of the finite descriptors before fitting.
    """

    array = _validate_points(points)
    total_started = time.perf_counter()
    features, feature_diagnostics = compute_fpfh_features(
        array,
        normal_k=normal_k,
        fpfh_k=fpfh_k,
        orient_normals=orient_normals,
    )
    if_result = run_isolation_forest_on_features(
        features,
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    runtime_sec = float(time.perf_counter() - total_started)
    diagnostics = {
        **feature_diagnostics,
        **if_result.diagnostics,
        "implementation": (
            "FPFH+IF, our reimplementation following the high-level "
            "baseline description in 3D-SONAR"
        ),
    }
    return BaselineResult(
        pred_mask=if_result.pred_mask,
        runtime_sec=runtime_sec,
        diagnostics=diagnostics,
    )
