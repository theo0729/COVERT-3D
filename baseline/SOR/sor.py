"""SOR baseline wrapper with a full-length binary output mask."""

from __future__ import annotations

import time

import numpy as np

from baseline.common import BaselineResult
from vast.data.preprocess import weak_sor


def _validate_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("points must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError("points contain NaN or Inf.")
    return array


def run_sor_baseline(
    points: np.ndarray,
    nb_neighbors: int = 50,
    std_ratio: float = 1.0,
) -> BaselineResult:
    """Run Open3D SOR and mark every removed input point as a defect.

    Ground truth is intentionally absent from this function.  The returned
    mask always has one entry per input point; SOR outliers are never deleted
    from the evaluation array.
    """

    array = _validate_points(points)
    nb_neighbors = int(nb_neighbors)
    std_ratio = float(std_ratio)
    if nb_neighbors <= 0:
        raise ValueError("nb_neighbors must be positive.")
    if nb_neighbors >= array.shape[0]:
        raise ValueError("nb_neighbors must be smaller than the number of points.")
    if not np.isfinite(std_ratio) or std_ratio <= 0.0:
        raise ValueError("std_ratio must be finite and positive.")

    started = time.perf_counter()
    _, inlier_indices, sor_summary = weak_sor(
        array,
        k=nb_neighbors,
        alpha=std_ratio,
    )
    runtime_sec = float(time.perf_counter() - started)

    inlier_indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
    if inlier_indices.size:
        if inlier_indices.min() < 0 or inlier_indices.max() >= array.shape[0]:
            raise RuntimeError("Open3D SOR returned an out-of-range input index.")
        if np.unique(inlier_indices).shape[0] != inlier_indices.shape[0]:
            raise RuntimeError("Open3D SOR returned duplicate inlier indices.")

    pred_mask = np.ones(array.shape[0], dtype=bool)
    pred_mask[inlier_indices] = False
    num_predicted_defect = int(np.count_nonzero(pred_mask))
    expected_removed = int(array.shape[0] - inlier_indices.shape[0])
    if num_predicted_defect != expected_removed:
        raise RuntimeError("SOR inlier-to-mask mapping failed its invariant check.")

    diagnostics = {
        "implementation": "Open3D remove_statistical_outlier",
        "open3d_positive_class": "removed point",
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
        "num_input_points": int(array.shape[0]),
        "num_inlier_points": int(inlier_indices.shape[0]),
        "num_predicted_defect": num_predicted_defect,
        "predicted_defect_ratio": float(np.mean(pred_mask)),
        "sor_summary": dict(sor_summary),
    }
    return BaselineResult(
        pred_mask=pred_mask,
        runtime_sec=runtime_sec,
        diagnostics=diagnostics,
    )

