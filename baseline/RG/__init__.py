"""PCL-style Region Growing controlled baseline."""

from .region_growing import (
    estimate_normals_and_curvature,
    region_grow_from_features,
    run_region_growing_baseline,
)

__all__ = [
    "estimate_normals_and_curvature",
    "region_grow_from_features",
    "run_region_growing_baseline",
]

