"""Minimal score normalization contract used by the frozen GTR runtime."""

from __future__ import annotations

import numpy as np


def robust_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Match the legacy Q1--Q99 clipped normalization used by GTR scores."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo + float(eps):
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo + float(eps)), 0.0, 1.0)

