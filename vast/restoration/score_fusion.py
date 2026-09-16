"""Score normalization and fusion for GTR seeding."""

from __future__ import annotations

import numpy as np


def robust_normalize(
    values: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Percentile-based robust normalization to ``[0, 1]``."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    return np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def fuse_gtr_scores(
    raw_scores: np.ndarray,
    local_scores: np.ndarray,
    use_fused: bool = True,
) -> np.ndarray:
    """Fuse raw Mahalanobis and local-contrast scores for GTR proposals."""
    raw_scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    local_scores = np.asarray(local_scores, dtype=np.float64).reshape(-1)
    if raw_scores.shape[0] != local_scores.shape[0]:
        raise ValueError("raw_scores and local_scores must have the same length.")

    raw_norm = robust_normalize(raw_scores)
    if not use_fused:
        return raw_norm
    local_norm = robust_normalize(local_scores)
    return np.maximum(raw_norm, local_norm)
