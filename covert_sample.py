"""Single-sample CLI for COVERT point-cloud defect segmentation."""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import functools
import hashlib
import itertools
import json
import platform
import pstats
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, get_type_hints

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    from numba import get_num_threads, njit, prange, set_num_threads
except ModuleNotFoundError:  # pragma: no cover - runtime environment dependent.
    njit = None
    prange = range
    get_num_threads = None
    set_num_threads = None

from vast.covert import _pipeline
from vast.covert._veto import build_legacy_hks_defect_veto_override


_farthest_point_sample_exact_numba_skip_selected_kernel = None
_farthest_point_sample_exact_numba_parallel_kernel = None
_csr_neighbor_means_numba_kernel = None
_filter_csr_by_radius_numba_kernel = None
_filter_csr_by_two_radii_numba_kernel = None
if njit is not None:

    @njit(cache=True)
    def _csr_neighbor_means_numba_kernel(
        scores: np.ndarray,
        offsets: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Mean each CSR row in its original neighbor order."""
        num_rows = offsets.shape[0] - 1
        means = np.empty(num_rows, dtype=np.float64)
        for row_index in range(num_rows):
            start = offsets[row_index]
            stop = offsets[row_index + 1]
            total = 0.0
            for position in range(start, stop):
                total += scores[indices[position]]
            means[row_index] = total / (stop - start)
        return means

    @njit(cache=True)
    def _filter_csr_by_radius_numba_kernel(
        points: np.ndarray,
        offsets: np.ndarray,
        indices: np.ndarray,
        radius_squared: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter ordered maximum-radius CSR rows without reordering entries."""
        num_rows = offsets.shape[0] - 1
        counts = np.zeros(num_rows, dtype=np.int64)
        for point_index in range(num_rows):
            px = points[point_index, 0]
            py = points[point_index, 1]
            pz = points[point_index, 2]
            count = 0
            for position in range(offsets[point_index], offsets[point_index + 1]):
                neighbor = indices[position]
                dx = points[neighbor, 0] - px
                dy = points[neighbor, 1] - py
                dz = points[neighbor, 2] - pz
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared <= radius_squared:
                    count += 1
            counts[point_index] = count

        filtered_offsets = np.empty(num_rows + 1, dtype=np.int64)
        filtered_offsets[0] = 0
        for point_index in range(num_rows):
            filtered_offsets[point_index + 1] = (
                filtered_offsets[point_index] + counts[point_index]
            )
        filtered_indices = np.empty(filtered_offsets[-1], dtype=np.int64)
        for point_index in range(num_rows):
            px = points[point_index, 0]
            py = points[point_index, 1]
            pz = points[point_index, 2]
            output_position = filtered_offsets[point_index]
            for position in range(offsets[point_index], offsets[point_index + 1]):
                neighbor = indices[position]
                dx = points[neighbor, 0] - px
                dy = points[neighbor, 1] - py
                dz = points[neighbor, 2] - pz
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared <= radius_squared:
                    filtered_indices[output_position] = neighbor
                    output_position += 1
        return filtered_offsets, filtered_indices

    @njit(cache=True)
    def _filter_csr_by_two_radii_numba_kernel(
        points: np.ndarray,
        offsets: np.ndarray,
        indices: np.ndarray,
        first_radius_squared: float,
        second_radius_squared: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Derive two ordered radius CSRs in one shared count/fill traversal."""
        num_rows = offsets.shape[0] - 1
        first_counts = np.zeros(num_rows, dtype=np.int64)
        second_counts = np.zeros(num_rows, dtype=np.int64)
        for point_index in range(num_rows):
            px = points[point_index, 0]
            py = points[point_index, 1]
            pz = points[point_index, 2]
            first_count = 0
            second_count = 0
            for position in range(offsets[point_index], offsets[point_index + 1]):
                neighbor = indices[position]
                dx = points[neighbor, 0] - px
                dy = points[neighbor, 1] - py
                dz = points[neighbor, 2] - pz
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared <= first_radius_squared:
                    first_count += 1
                if distance_squared <= second_radius_squared:
                    second_count += 1
            first_counts[point_index] = first_count
            second_counts[point_index] = second_count

        first_offsets = np.empty(num_rows + 1, dtype=np.int64)
        second_offsets = np.empty(num_rows + 1, dtype=np.int64)
        first_offsets[0] = 0
        second_offsets[0] = 0
        for point_index in range(num_rows):
            first_offsets[point_index + 1] = (
                first_offsets[point_index] + first_counts[point_index]
            )
            second_offsets[point_index + 1] = (
                second_offsets[point_index] + second_counts[point_index]
            )
        first_indices = np.empty(first_offsets[-1], dtype=np.int64)
        second_indices = np.empty(second_offsets[-1], dtype=np.int64)
        for point_index in range(num_rows):
            px = points[point_index, 0]
            py = points[point_index, 1]
            pz = points[point_index, 2]
            first_output = first_offsets[point_index]
            second_output = second_offsets[point_index]
            for position in range(offsets[point_index], offsets[point_index + 1]):
                neighbor = indices[position]
                dx = points[neighbor, 0] - px
                dy = points[neighbor, 1] - py
                dz = points[neighbor, 2] - pz
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared <= first_radius_squared:
                    first_indices[first_output] = neighbor
                    first_output += 1
                if distance_squared <= second_radius_squared:
                    second_indices[second_output] = neighbor
                    second_output += 1
        return first_offsets, first_indices, second_offsets, second_indices

    @njit(cache=True)
    def _farthest_point_sample_exact_numba_skip_selected_kernel(
        points: np.ndarray,
        num_samples: int,
        start_index: int,
    ) -> np.ndarray:
        """Exact FPS trial that omits distance work for fixed zero entries."""
        num_points = points.shape[0]
        selected = np.empty(num_samples, dtype=np.int64)
        selected_mask = np.zeros(num_points, dtype=np.bool_)
        min_dist_sq = np.empty(num_points, dtype=np.float64)
        for j in range(num_points):
            min_dist_sq[j] = np.inf

        farthest = int(start_index)
        for i in range(num_samples):
            selected[i] = farthest
            selected_mask[farthest] = True
            min_dist_sq[farthest] = 0.0
            cx = points[farthest, 0]
            cy = points[farthest, 1]
            cz = points[farthest, 2]

            for j in range(num_points):
                if selected_mask[j]:
                    continue
                dx = points[j, 0] - cx
                dy = points[j, 1] - cy
                dz = points[j, 2] - cz
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq < min_dist_sq[j]:
                    min_dist_sq[j] = dist_sq

            best_idx = 0
            best_val = min_dist_sq[0]
            for j in range(1, num_points):
                val = min_dist_sq[j]
                if val > best_val:
                    best_val = val
                    best_idx = j
            farthest = best_idx

        return selected

    @njit(cache=True, parallel=True)
    def _farthest_point_sample_exact_numba_parallel_kernel(
        points: np.ndarray,
        num_samples: int,
        start_index: int,
    ) -> np.ndarray:
        """Exact FPS with parallel pointwise distance updates and serial argmax."""
        num_points = points.shape[0]
        selected = np.empty(num_samples, dtype=np.int64)
        min_dist_sq = np.empty(num_points, dtype=np.float64)
        for j in prange(num_points):
            min_dist_sq[j] = np.inf

        farthest = int(start_index)
        for i in range(num_samples):
            selected[i] = farthest
            cx = points[farthest, 0]
            cy = points[farthest, 1]
            cz = points[farthest, 2]

            for j in prange(num_points):
                dx = points[j, 0] - cx
                dy = points[j, 1] - cy
                dz = points[j, 2] - cz
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq < min_dist_sq[j]:
                    min_dist_sq[j] = dist_sq

            best_idx = 0
            best_val = min_dist_sq[0]
            for j in range(1, num_points):
                val = min_dist_sq[j]
                if val > best_val:
                    best_val = val
                    best_idx = j
            farthest = best_idx

        return selected


@dataclass(frozen=True)
class PreprocessConfig:
    fps_num_samples: int = 10_000
    fps_mode: str = "exact_numba"
    fps_prefilter_num_points: int = 10_000
    fps_seed: int = 0
    cube_size: float = 64.0
    use_sor: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    graph_knn_k: int = 16
    graph_radius_cap_factor: float = 2.5
    surface_variation_scales: tuple[int, ...] = (12, 24, 48)
    hks_num_eigenvalues: int = 100
    local_contrast_radii: tuple[float, ...] = (3.0, 6.0, 12.0)
    knn_smooth_iterations: int = 2
    va_hks_gamma: float = 10.0
    use_local_contrast_neighbor_cache: bool = True
    use_batched_three_view_knn_smooth: bool = False
    use_batched_three_view_local_contrast: bool = False


@dataclass(frozen=True)
class FeatureSwitches:
    variation_aware_hks: bool = True
    new_ac: bool = True
    hks_normal_suppression: bool = True
    defect_veto: bool = True
    point_boundary_seed_removal: bool = True
    seed_local_graph: bool = True
    sgcr_growth: bool = True
    label_aware_merge: bool = True
    q_to_z_repair: bool = True
    scale_consistency_rules: bool = True
    gtr_reconstruction_refinement: bool = True
    gtr_point_refinement: bool = True
    final_local_normal_u_cleanup: bool = True


@dataclass(frozen=True)
class NewACConfig:
    angle_threshold_deg: float = 110.0
    min_neighbors: int = 8
    component_grouping_mode: str = "gap_repaired"
    component_group_max_gap_edges: int = 2
    component_min_relative_to_largest: float = 0.5
    closing_dilation_hops: int = 0
    closing_erosion_hops: int = 0
    expand_hops: int = 1


@dataclass(frozen=True)
class HKSConfig:
    pure_nonzero_growth_percentile: float = 10.0
    pure_display_lower_percentile: float = 1.0
    pure_display_upper_percentile: float = 99.0
    pure_structure_root_min_display_percent: float = 60.0
    pure_display_level_hop_budgets: tuple[int, ...] = (1, 2, 3, 4, 6)
    pure_normal_suppression_mode: str = "legacy_hard"
    va_nonzero_growth_percentile: float = 10.0
    va_display_lower_percentile: float = 1.0
    va_display_upper_percentile: float = 99.0
    va_display_level_hop_budgets: tuple[int, ...] = (1, 2, 3, 4, 6)
    va_mask_root_growth: bool = True
    boundary_keep_weight: float = 0.01


@dataclass(frozen=True)
class DefectVetoConfig:
    joint_component_p90_min: float = 0.80
    joint_shell2_p90_min: float = 0.40
    boundary_max_graph_distance: int = 2
    long_ridge_min_graph_distance: int = 16
    long_ridge_min_path_efficiency: float = 0.75


@dataclass(frozen=True)
class ConfidenceConfig:
    sv_fusion_mode: str = "mask_sv"
    semantic_gate_nonzero_percentile: float = 50.0
    semantic_gate_replacement_value: float = 0.0
    semantic_confidence_exponent: float = 2.0
    original_sv_gate_percentile: float = 50.0
    original_sv_gate_replacement_value: float = 0.0
    seed_display_lower_percentile: float = 1.0
    seed_display_upper_percentile: float = 99.0
    high_confidence_seed_threshold: float = 0.50


@dataclass(frozen=True)
class SeedGraphConfig:
    local_k: int = 8
    local_scale_k: int = 6
    local_radius_factor: float = 1.6
    min_component_size: int = 2
    red_sv_quantile: float = 0.98


@dataclass(frozen=True)
class SGCRConfig:
    growth_response_source: str = "enhanced_sv_q1_q99"
    growth_mode: str = "energy_descent"
    max_grow_hops: int = 9
    minimum_growth_component_size: int = 5
    maximum_component_ratio: float = 0.01
    seed_level_percentile: float = 90.0
    seed_relative_grow_ratio: float = 0.49
    use_absolute_support_floor: bool = True
    absolute_support_floor: float = 0.08
    keep_equal_to_floor: bool = True
    require_strict_descent: bool = True
    descent_epsilon: float = 1e-6
    allow_plateau: bool = False
    floor_mode: str = "seed_relative"
    use_priority_queue: bool = True
    label_percentile: float = 80.0
    minimum_energy_ratio: float = 0.70


@dataclass(frozen=True)
class MarkerConfig:
    raw_z_percentile: float = 97.0
    raw_z_strictly_above_percentile: bool = True
    w_dilation_hops: int = 8
    w_erosion_hops: int = 8
    w_keep_original: bool = True


@dataclass(frozen=True)
class ScaleConsistencyConfig:
    small_defect_max_multiple: int = 4
    large_defect_min_multiple: int = 8
    late_gtr_over_y_min_multiple: int = 2
    late_gtr_over_y_min_numerator: int = 3
    late_gtr_over_y_min_denominator: int = 2
    late_w_over_y_max_numerator: int = 1
    late_w_over_y_max_denominator: int = 4
    trusted_z_over_q_min_multiple: int = 4
    trusted_gtr_over_w_min_multiple: int = 4
    trusted_y_over_u_max_numerator: int = 3
    trusted_y_over_u_max_denominator: int = 2
    trusted_gtr_over_u_min_multiple: int = 4
    weak_y_over_u_min_multiple: int = 8
    normal_mass_majority_numerator: int = 1
    normal_mass_majority_denominator: int = 2


@dataclass(frozen=True)
class GTRConfig:
    dilation_hops: int = 8
    erosion_hops: int = 8
    keep_original_mask: bool = True
    block_other_original_masks: bool = True
    closing_mode: str = "independent_no_block_sv_priority"
    closing_overlap_resolution: str = "highest_seed_sv_mean"
    minimum_closed_component_size: int = 20
    post_closing_expand_hops: int = 0
    restoration_mode: str = "biharmonic"
    biharmonic_solver: str = "spsolve"
    biharmonic_fallback_to_laplacian: bool = True
    restore_per_component: bool = True
    use_hks_open_boundary_as_restoration_support: bool = True
    hks_open_boundary_support_use_expanded: bool = False
    restoration_weight: str = "inverse_distance"
    anchor_hops: int = 4
    minimum_anchor_points: int = 20
    displacement_threshold: float = 0.28
    stress_veto_enabled: bool = True
    stress_threshold: float = 0.90
    biharmonic_regularization_epsilon: float = 1e-8
    boundary_support_expand_hops: int = 1
    exclude_boundary_support_from_unknown: bool = True
    exclude_boundary_support_from_decision: bool = True
    restitute_boundary_support: bool = False
    point_filter_before_support_restitution: bool = True
    point_filter_mode: str = "ds_quadrant_remove_lowD_lowS"
    ds_scope: str = "all_points"
    ds_displacement_normalized_threshold: float = 0.50
    ds_stress_normalized_threshold: float = 0.50
    ds_keep_equal: bool = True
    ds_quadrant_enabled: bool = True
    candidate_classifier_mode: str = "adaptive_normal_rejection"
    local_normal_reference_enabled: bool = True
    local_normal_reference_max_hops: int = 3
    local_normal_reference_epsilon: float = 1e-12
    adaptive_relative_normal_max_ratio: float = 0.50
    adaptive_min_large_fraction: float = 0.02
    adaptive_max_shallow_displacement_normalized: float = 0.007
    local_normal_u_cleanup_mode: str = "final_mask_only"


@dataclass(frozen=True)
class CovertPipelineConfig:
    schema_version: str = "covert-paper-v1"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    switches: FeatureSwitches = field(default_factory=FeatureSwitches)
    new_ac: NewACConfig = field(default_factory=NewACConfig)
    hks: HKSConfig = field(default_factory=HKSConfig)
    defect_veto: DefectVetoConfig = field(default_factory=DefectVetoConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    seed_graph: SeedGraphConfig = field(default_factory=SeedGraphConfig)
    sgcr: SGCRConfig = field(default_factory=SGCRConfig)
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    scale_consistency: ScaleConsistencyConfig = field(
        default_factory=ScaleConsistencyConfig
    )
    gtr: GTRConfig = field(default_factory=GTRConfig)


DEFAULT_CONFIG = CovertPipelineConfig()
PRODUCTION_OPTIMIZATION_MODE = "r6_unsorted_nested_radius_fps8_ncv256"
PRODUCTION_QUERY_WORKERS = 16


# ============================================================
# SINGLE-SAMPLE RUN CONFIG
# ============================================================
# These values are used only by ``python covert_sample.py``.  They are not
# detector parameters and are deliberately ignored by covert_batch.py.
PC_PATH: str | Path | None = None
GT_PATH: str | Path | None = None
VISUALIZE = False
VISUALIZATION_STAGES = (
    "input",
    "sv",
    "pure-hks-response",
    "va-hks",
    "final-ac",
    "protected",
    "semantic",
    "seed",
    "q",
    "u",
    "y",
    "z",
    "w",
    "gtr",
    "final",
    "gt",
)
VISUALIZATION_STAGE_KEY_MAP = {
    "input": "I",
    "sv": "S",
    "enhanced-sv": "E",
    "pure-hks-response": "H",
    "va-hks": "V",
    "raw-ac": "A",
    "final-ac": "B",
    "hks-normal-prior": "N",
    "protected": "P",
    "semantic": "M",
    "seed": "C",
    "q": "Q",
    "u": "U",
    "y": "Y",
    "raw-z": "R",
    "z": "Z",
    "w": "W",
    "gtr": "G",
    "final": "F",
    "gt": "T",
}
VISUALIZATION_STAGE_TITLES = {
    "input": "Input point cloud",
    "sv": "Original SV",
    "enhanced-sv": "Enhanced SV",
    "pure-hks-response": "Pure-HKS response",
    "va-hks": "VA-HKS response",
    "raw-ac": "Raw AC",
    "final-ac": "Final AC",
    "hks-normal-prior": "Legacy HKS normal prior",
    "protected": "Defect-protected components",
    "semantic": "Semantic confidence",
    "seed": "Seed confidence",
    "q": "Q seed components",
    "u": "U grown components",
    "y": "Y merged components",
    "raw-z": "Raw Z",
    "z": "Authoritative Z",
    "w": "W morphology support",
    "gtr": "GTR candidate",
    "final": "Final prediction",
    "gt": "Ground truth",
}
SINGLE_SAMPLE_OUTPUT_DIR: str | Path | None = None
RUNTIME_PROFILE_OUTPUT_DIR: str | Path = Path("runtime/profile")
VERBOSE = True
OBJECT_PSEUDOSTRESS_PERCENTILES = (85.0, 90.0, 95.0)


@dataclass(frozen=True)
class CovertTiming:
    input_load_seconds: float
    pipeline_seconds: float
    inference_total_seconds: float
    gt_load_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    visualization_seconds: float = 0.0
    artifact_write_seconds: float = 0.0
    downsampling_seconds: float = 0.0
    algorithm_seconds_excluding_downsampling: float = 0.0


@dataclass(frozen=True)
class CovertSampleResult:
    sample_id: str
    pc_path: str
    gt_path: str | None
    config_snapshot: Mapping[str, Any]
    config_hash: str
    config_source: str
    points: np.ndarray
    fps_indices: np.ndarray
    raw_points_aligned: np.ndarray
    coordinate_scale: float
    neighbor_indices: tuple[np.ndarray, ...]
    pure_hks_features: np.ndarray
    va_hks_features: np.ndarray
    pure_hks_time_scales: np.ndarray
    va_hks_time_scales: np.ndarray
    original_sv: np.ndarray
    enhanced_sv: np.ndarray
    pure_hks_response: np.ndarray
    va_hks_response: np.ndarray
    raw_ac_mask: np.ndarray
    final_ac_suppression_mask: np.ndarray
    pure_hks_root_mask: np.ndarray
    pure_hks_suppression_mask: np.ndarray
    defect_protected_component_mask: np.ndarray
    sv_boundary_suppressed: np.ndarray
    va_hks_boundary_suppressed: np.ndarray
    semantic_confidence: np.ndarray
    seed_confidence: np.ndarray
    seed_confidence_normalized: np.ndarray
    high_confidence_seed_mask: np.ndarray
    q_component_labels: np.ndarray
    u_component_labels: np.ndarray
    y_component_labels: np.ndarray
    raw_z_component_labels: np.ndarray
    authoritative_z_component_labels: np.ndarray
    w_component_labels: np.ndarray
    gtr_candidate_labels: np.ndarray
    gtr_candidate_mask: np.ndarray
    closed_mask: np.ndarray
    restoration_unknown_mask: np.ndarray
    restoration_support_mask: np.ndarray
    restored_points: np.ndarray
    displacement_magnitude: np.ndarray
    stress_proxy: np.ndarray
    rejected_mask: np.ndarray
    final_mask: np.ndarray
    gt_mask: np.ndarray | None
    metrics: Mapping[str, Any]
    timings: CovertTiming
    audit: Mapping[str, Any]
    object_scores: Mapping[str, Any] = field(default_factory=dict)


class RuntimeStageCollector:
    """Collect lightweight wall/CPU timings without changing return values."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    @contextmanager
    def measure(self, group: str, name: str) -> Iterator[None]:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            yield
        finally:
            self._values[(str(group), str(name))].append(
                (time.perf_counter() - wall_start, time.process_time() - cpu_start)
            )

    def seconds(self, group: str, name: str) -> float:
        return float(sum(value[0] for value in self._values.get((group, name), ())))

    def table(self, group: str, denominator: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (row_group, name), values in self._values.items():
            if row_group != group:
                continue
            wall = float(sum(value[0] for value in values))
            cpu = float(sum(value[1] for value in values))
            rows.append(
                {
                    "name": name,
                    "wall_seconds": wall,
                    "cpu_seconds": cpu,
                    "calls": int(len(values)),
                    "wall_percent": float(100.0 * wall / denominator)
                    if denominator > 0.0
                    else 0.0,
                }
            )
        return sorted(rows, key=lambda row: float(row["wall_seconds"]), reverse=True)


class NeighborValidationCache:
    """Run-local cache for already validated neighbor-list objects.

    A strong reference prevents object-id reuse.  The cache does not copy,
    reorder, or otherwise transform any neighbor array; it only skips repeated
    validation of an object that the original function already accepted.
    """

    def __init__(self, original: Any) -> None:
        self.original = original
        self.entries: dict[tuple[int, int], tuple[Any, list[np.ndarray]]] = {}
        self.calls = 0
        self.hits = 0
        self.misses = 0
        self.original_validation_calls = 0

    def coerce(
        self, neighbor_indices: list[np.ndarray] | np.ndarray, num_points: int
    ) -> list[np.ndarray]:
        self.calls += 1
        key = (id(neighbor_indices), int(num_points))
        cached = self.entries.get(key)
        if cached is not None and cached[0] is neighbor_indices:
            self.hits += 1
            return cached[1]

        self.misses += 1
        self.original_validation_calls += 1
        validated = self.original(neighbor_indices, int(num_points))
        self.entries[key] = (neighbor_indices, validated)
        # Register the validated list itself.  The original implementation
        # allocates a new list even when every contained int64 array is reused.
        validated_key = (id(validated), int(num_points))
        self.entries[validated_key] = (validated, validated)
        return validated

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "cache_hits": int(self.hits),
            "cache_misses": int(self.misses),
            "original_validation_calls": int(self.original_validation_calls),
            "registered_objects": int(len(self.entries)),
        }


class TrustedNeighborValidationCache(NeighborValidationCache):
    """Trust neighbor lists produced inside the already-validated pipeline."""

    def __init__(self, original: Any) -> None:
        super().__init__(original)
        self.trusted_misses = 0
        self.fallback_calls = 0

    def coerce(
        self, neighbor_indices: list[np.ndarray] | np.ndarray, num_points: int
    ) -> list[np.ndarray]:
        self.calls += 1
        key = (id(neighbor_indices), int(num_points))
        cached = self.entries.get(key)
        if cached is not None and cached[0] is neighbor_indices:
            self.hits += 1
            return cached[1]

        self.misses += 1
        if isinstance(neighbor_indices, list) and len(neighbor_indices) == int(num_points):
            validated = neighbor_indices
            self.trusted_misses += 1
        else:
            self.original_validation_calls += 1
            self.fallback_calls += 1
            validated = self.original(neighbor_indices, int(num_points))
        self.entries[key] = (neighbor_indices, validated)
        return validated

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "trusted_pipeline_neighbor_lists": int(self.trusted_misses),
                "fallback_calls": int(self.fallback_calls),
                "validation_skipped": True,
            }
        )
        return result


class RepeatedIndexLoopCache:
    """Reuse immutable index views while preserving every NumPy reduction call."""

    def __init__(self) -> None:
        self.knn_entries: dict[tuple[int, int], tuple[Any, list[np.ndarray]]] = {}
        self.local_entries: dict[tuple[int, int], tuple[Any, list[np.ndarray]]] = {}
        self.knn_calls = 0
        self.knn_hits = 0
        self.local_calls = 0
        self.local_hits = 0

    @staticmethod
    def _key(values: Sequence[np.ndarray]) -> tuple[int, int]:
        return (id(values), len(values))

    def knn_smooth_scores(
        self,
        scores: np.ndarray,
        neighbor_indices: list[np.ndarray],
        iterations: int = 2,
    ) -> np.ndarray:
        self.knn_calls += 1
        key = self._key(neighbor_indices)
        cached = self.knn_entries.get(key)
        if cached is not None and cached[0] is neighbor_indices:
            self.knn_hits += 1
            hoods = cached[1]
        else:
            hoods = [
                np.concatenate(
                    ([point_index], np.asarray(neighbors, dtype=np.int64).reshape(-1))
                )
                for point_index, neighbors in enumerate(neighbor_indices)
            ]
            self.knn_entries[key] = (neighbor_indices, hoods)

        smoothed = np.asarray(scores, dtype=np.float64).reshape(-1)
        num_points = smoothed.shape[0]
        for _ in range(iterations):
            next_scores = np.empty(num_points, dtype=np.float64)
            for point_index, hood in enumerate(hoods):
                next_scores[point_index] = float(np.mean(smoothed[hood]))
            smoothed = next_scores
        return smoothed

    def local_contrast_from_neighbors(
        self, scores: np.ndarray, neighbor_lists: list[np.ndarray]
    ) -> np.ndarray:
        self.local_calls += 1
        key = self._key(neighbor_lists)
        cached = self.local_entries.get(key)
        if cached is not None and cached[0] is neighbor_lists:
            self.local_hits += 1
            normalized = cached[1]
        else:
            normalized = [
                np.asarray(neighbors, dtype=np.int64) for neighbors in neighbor_lists
            ]
            self.local_entries[key] = (neighbor_lists, normalized)

        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        mu_local = np.empty(scores.shape[0], dtype=np.float64)
        for point_index, neighbors in enumerate(normalized):
            neighbor_scores = scores[neighbors]
            mu_local[point_index] = float(np.mean(neighbor_scores))
        contrast_score = scores - mu_local
        contrast_score = np.nan_to_num(
            contrast_score, nan=0.0, posinf=0.0, neginf=0.0
        )
        return np.clip(contrast_score, 0.0, None).astype(np.float64, copy=False)

    def summary(self) -> dict[str, Any]:
        return {
            "knn_smooth_calls": int(self.knn_calls),
            "knn_index_cache_hits": int(self.knn_hits),
            "knn_registered_neighbor_lists": int(len(self.knn_entries)),
            "local_contrast_calls": int(self.local_calls),
            "local_index_cache_hits": int(self.local_hits),
            "local_registered_neighbor_lists": int(len(self.local_entries)),
        }


class LengthBucketedIndexLoopCache(RepeatedIndexLoopCache):
    """Batch equal-length rows while retaining each row's exact index order."""

    def __init__(
        self,
        max_block_elements: int = 500_000,
        *,
        verify_each_reduction: bool = False,
    ) -> None:
        super().__init__()
        self.max_block_elements = int(max_block_elements)
        self.verify_each_reduction = bool(verify_each_reduction)
        self.exact_reduction_checks = 0
        self.knn_bucket_entries: dict[
            tuple[int, int], tuple[Any, list[tuple[np.ndarray, np.ndarray]]]
        ] = {}
        self.local_bucket_entries: dict[
            tuple[int, int], tuple[Any, list[tuple[np.ndarray, np.ndarray]]]
        ] = {}
        self.knn_bucket_hits = 0
        self.local_bucket_hits = 0

    def _build_blocks(
        self, rows: Sequence[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        by_length: dict[int, list[int]] = {}
        for point_index, row in enumerate(rows):
            by_length.setdefault(int(row.size), []).append(int(point_index))

        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for row_length, point_values in by_length.items():
            if row_length <= 0:
                raise ValueError("Mean neighborhoods must not be empty.")
            rows_per_block = max(1, self.max_block_elements // row_length)
            for start in range(0, len(point_values), rows_per_block):
                selected = np.asarray(
                    point_values[start : start + rows_per_block], dtype=np.int64
                )
                index_matrix = np.stack(
                    [rows[int(point_index)] for point_index in selected], axis=0
                )
                blocks.append((selected, index_matrix))
        return blocks

    @staticmethod
    def _apply_blocks(
        scores: np.ndarray,
        blocks: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        means = np.empty(scores.shape[0], dtype=np.float64)
        for point_indices, index_matrix in blocks:
            gathered = scores[index_matrix]
            means[point_indices] = np.mean(gathered, axis=1)
        return means

    def _assert_original_means(
        self,
        scores: np.ndarray,
        rows: Sequence[np.ndarray],
        candidate: np.ndarray,
    ) -> None:
        if not self.verify_each_reduction:
            return
        reference = np.empty(scores.shape[0], dtype=np.float64)
        for point_index, row in enumerate(rows):
            reference[point_index] = float(np.mean(scores[row]))
        if not np.array_equal(reference, candidate):
            mismatch = np.flatnonzero(reference != candidate)
            first = int(mismatch[0]) if mismatch.size else -1
            raise AssertionError(
                "Length-bucketed mean differs from original per-point np.mean "
                f"at point {first}."
            )
        self.exact_reduction_checks += 1

    def knn_smooth_scores(
        self,
        scores: np.ndarray,
        neighbor_indices: list[np.ndarray],
        iterations: int = 2,
    ) -> np.ndarray:
        self.knn_calls += 1
        key = self._key(neighbor_indices)
        cached = self.knn_entries.get(key)
        if cached is not None and cached[0] is neighbor_indices:
            self.knn_hits += 1
            hoods = cached[1]
        else:
            hoods = [
                np.concatenate(
                    ([point_index], np.asarray(neighbors, dtype=np.int64).reshape(-1))
                )
                for point_index, neighbors in enumerate(neighbor_indices)
            ]
            self.knn_entries[key] = (neighbor_indices, hoods)

        bucket_cached = self.knn_bucket_entries.get(key)
        if bucket_cached is not None and bucket_cached[0] is neighbor_indices:
            self.knn_bucket_hits += 1
            blocks = bucket_cached[1]
        else:
            blocks = self._build_blocks(hoods)
            self.knn_bucket_entries[key] = (neighbor_indices, blocks)

        smoothed = np.asarray(scores, dtype=np.float64).reshape(-1)
        for _ in range(iterations):
            next_scores = self._apply_blocks(smoothed, blocks)
            self._assert_original_means(smoothed, hoods, next_scores)
            smoothed = next_scores
        return smoothed

    def local_contrast_from_neighbors(
        self, scores: np.ndarray, neighbor_lists: list[np.ndarray]
    ) -> np.ndarray:
        self.local_calls += 1
        key = self._key(neighbor_lists)
        cached = self.local_entries.get(key)
        if cached is not None and cached[0] is neighbor_lists:
            self.local_hits += 1
            normalized = cached[1]
        else:
            normalized = [
                np.asarray(neighbors, dtype=np.int64) for neighbors in neighbor_lists
            ]
            self.local_entries[key] = (neighbor_lists, normalized)

        bucket_cached = self.local_bucket_entries.get(key)
        if bucket_cached is not None and bucket_cached[0] is neighbor_lists:
            self.local_bucket_hits += 1
            blocks = bucket_cached[1]
        else:
            blocks = self._build_blocks(normalized)
            self.local_bucket_entries[key] = (neighbor_lists, blocks)

        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        mu_local = self._apply_blocks(scores, blocks)
        self._assert_original_means(scores, normalized, mu_local)
        contrast_score = scores - mu_local
        contrast_score = np.nan_to_num(
            contrast_score, nan=0.0, posinf=0.0, neginf=0.0
        )
        return np.clip(contrast_score, 0.0, None).astype(np.float64, copy=False)

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "length_bucketed_reduction": True,
                "max_block_elements": int(self.max_block_elements),
                "knn_bucket_blocks": int(
                    sum(len(entry[1]) for entry in self.knn_bucket_entries.values())
                ),
                "knn_bucket_cache_hits": int(self.knn_bucket_hits),
                "local_bucket_blocks": int(
                    sum(len(entry[1]) for entry in self.local_bucket_entries.values())
                ),
                "local_bucket_cache_hits": int(self.local_bucket_hits),
                "neighbor_order_changed": False,
                "verify_each_reduction": bool(self.verify_each_reduction),
                "exact_reduction_checks": int(self.exact_reduction_checks),
            }
        )
        return result


class CSRNeighborRows(Sequence[np.ndarray]):
    """Read-only row views over one ordered CSR index buffer."""

    def __init__(self, offsets: np.ndarray, indices: np.ndarray) -> None:
        self.offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        if self.offsets.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("CSR neighbor buffers must be one-dimensional.")
        if self.offsets.size < 1 or self.offsets[0] != 0:
            raise ValueError("CSR offsets must start at zero.")
        if self.offsets[-1] != self.indices.size:
            raise ValueError("CSR terminal offset must equal the index count.")

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def __getitem__(self, item: int | slice) -> Any:
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(len(self)))]
        index = int(item)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.indices[self.offsets[index] : self.offsets[index + 1]]


class NumbaCSRIndexLoopCache(RepeatedIndexLoopCache):
    """Compile neighborhood means over CSR rows without changing row order."""

    def __init__(self) -> None:
        super().__init__()
        if _csr_neighbor_means_numba_kernel is None:
            raise RuntimeError("Numba is required for the CSR neighborhood trial.")
        self.knn_csr_entries: dict[
            tuple[int, int], tuple[Any, tuple[np.ndarray, np.ndarray]]
        ] = {}
        self.local_csr_entries: dict[
            tuple[int, int], tuple[Any, tuple[np.ndarray, np.ndarray]]
        ] = {}
        self.knn_csr_hits = 0
        self.local_csr_hits = 0
        self.csr_build_seconds = 0.0
        self.kernel_seconds = 0.0
        self.kernel_calls = 0

    def _build_csr(
        self,
        rows: Sequence[Any],
        *,
        add_self: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter()
        if isinstance(rows, CSRNeighborRows) and not add_self:
            self.csr_build_seconds += time.perf_counter() - started
            return rows.offsets, rows.indices
        num_rows = len(rows)
        extra = 1 if add_self else 0
        lengths = np.fromiter(
            (len(row) + extra for row in rows),
            dtype=np.int64,
            count=num_rows,
        )
        if np.any(lengths <= 0):
            raise ValueError("Mean neighborhoods must not be empty.")
        offsets = np.empty(num_rows + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])
        if add_self:
            indices = np.empty(total, dtype=np.int64)
            for point_index, row in enumerate(rows):
                start = int(offsets[point_index])
                stop = int(offsets[point_index + 1])
                indices[start] = int(point_index)
                indices[start + 1 : stop] = row
        else:
            indices = np.fromiter(
                itertools.chain.from_iterable(rows),
                dtype=np.int64,
                count=total,
            )
        self.csr_build_seconds += time.perf_counter() - started
        return offsets, indices

    def _apply_csr(
        self,
        scores: np.ndarray,
        csr: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        started = time.perf_counter()
        offsets, indices = csr
        means = _csr_neighbor_means_numba_kernel(scores, offsets, indices)
        self.kernel_seconds += time.perf_counter() - started
        self.kernel_calls += 1
        return means

    def knn_smooth_scores(
        self,
        scores: np.ndarray,
        neighbor_indices: list[np.ndarray],
        iterations: int = 2,
    ) -> np.ndarray:
        self.knn_calls += 1
        key = self._key(neighbor_indices)
        cached = self.knn_csr_entries.get(key)
        if cached is not None and cached[0] is neighbor_indices:
            self.knn_hits += 1
            self.knn_csr_hits += 1
            csr = cached[1]
        else:
            csr = self._build_csr(neighbor_indices, add_self=True)
            self.knn_csr_entries[key] = (neighbor_indices, csr)

        smoothed = np.asarray(scores, dtype=np.float64).reshape(-1)
        for _ in range(iterations):
            smoothed = self._apply_csr(smoothed, csr)
        return smoothed

    def local_contrast_from_neighbors(
        self, scores: np.ndarray, neighbor_lists: list[np.ndarray]
    ) -> np.ndarray:
        self.local_calls += 1
        key = self._key(neighbor_lists)
        cached = self.local_csr_entries.get(key)
        if cached is not None and cached[0] is neighbor_lists:
            self.local_hits += 1
            self.local_csr_hits += 1
            csr = cached[1]
        else:
            csr = self._build_csr(neighbor_lists, add_self=False)
            self.local_csr_entries[key] = (neighbor_lists, csr)

        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        mu_local = self._apply_csr(scores, csr)
        contrast_score = scores - mu_local
        contrast_score = np.nan_to_num(
            contrast_score, nan=0.0, posinf=0.0, neginf=0.0
        )
        return np.clip(contrast_score, 0.0, None).astype(np.float64, copy=False)

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "numba_csr_reduction": True,
                "knn_csr_cache_hits": int(self.knn_csr_hits),
                "local_csr_cache_hits": int(self.local_csr_hits),
                "knn_csr_objects": int(len(self.knn_csr_entries)),
                "local_csr_objects": int(len(self.local_csr_entries)),
                "csr_build_seconds": float(self.csr_build_seconds),
                "kernel_seconds": float(self.kernel_seconds),
                "kernel_calls": int(self.kernel_calls),
                "neighbor_order_changed": False,
                "accumulator_dtype": "float64",
            }
        )
        return result


class TangentBasisConstantCache:
    """Exact AC tangent basis with constant reference-axis allocations reused."""

    def __init__(self) -> None:
        self.axes = np.eye(3, dtype=np.float64)
        self.fallback_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.fallback_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.calls = 0

    def tangent_basis(self, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        normal = np.asarray(normal, dtype=np.float64)
        if normal.shape != (3,):
            raise ValueError("normal must have shape (3,).")
        if not np.isfinite(normal).all():
            raise ValueError("normal must contain only finite values.")

        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            raise ValueError("normal magnitude must be positive.")
        n = normal / normal_norm

        reference_axis = self.axes[int(np.argmin(np.abs(n)))]
        u = np.cross(n, reference_axis)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-12:
            reference_axis = self.fallback_x
            if abs(n[0]) > 0.9:
                reference_axis = self.fallback_y
            u = np.cross(n, reference_axis)
            u_norm = float(np.linalg.norm(u))
            if u_norm <= 1e-12:
                raise ValueError("Failed to build tangent basis from normal.")
        u = u / u_norm

        v = np.cross(n, u)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-12:
            raise ValueError("Failed to build tangent basis from normal.")
        v = v / v_norm
        return u, v

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "reference_axis_allocations_per_run": 3,
            "baseline_reference_axis_allocations_estimate": int(self.calls + 2),
        }


class TrustedTangentBasisConstantCache(TangentBasisConstantCache):
    """Exact tangent basis without revalidating an already validated normal.

    ``extract_ac_boundary_seed`` calls this helper only after it has built
    ``normal_valid_mask`` and normalized every valid normal.  The production
    helper nevertheless repeats shape, finiteness, and magnitude checks for
    each valid point.  This runtime-only specialization removes those repeated
    guards while retaining the same NumPy norm/cross operations and constants.
    """

    def tangent_basis(self, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        normal_norm = float(np.linalg.norm(normal))
        n = normal / normal_norm

        reference_axis = self.axes[int(np.argmin(np.abs(n)))]
        u = np.cross(n, reference_axis)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-12:
            reference_axis = self.fallback_x
            if abs(n[0]) > 0.9:
                reference_axis = self.fallback_y
            u = np.cross(n, reference_axis)
            u_norm = float(np.linalg.norm(u))
            if u_norm <= 1e-12:
                raise ValueError("Failed to build tangent basis from normal.")
        u = u / u_norm

        v = np.cross(n, u)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-12:
            raise ValueError("Failed to build tangent basis from normal.")
        v = v / v_norm
        return u, v

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result["trusted_valid_normal_fast_path"] = True
        result["floating_point_operations_changed"] = False
        return result


class ScalarCrossTrustedTangentBasisCache(TrustedTangentBasisConstantCache):
    """Use scalar 3-D cross products while preserving their operation order."""

    def __init__(self, reference: Any | None = None, *, verify: bool = False) -> None:
        super().__init__()
        self.reference = reference
        self.verify = bool(verify)
        self.exact_basis_checks = 0

    def tangent_basis(self, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        normal_norm = float(np.linalg.norm(normal))
        n = normal / normal_norm

        axis_index = int(np.argmin(np.abs(n)))
        if axis_index == 0:
            u = np.array((0.0, n[2], -n[1]), dtype=np.float64)
        elif axis_index == 1:
            u = np.array((-n[2], 0.0, n[0]), dtype=np.float64)
        else:
            u = np.array((n[1], -n[0], 0.0), dtype=np.float64)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-12:
            reference_axis = self.fallback_x
            if abs(n[0]) > 0.9:
                reference_axis = self.fallback_y
            u = np.cross(n, reference_axis)
            u_norm = float(np.linalg.norm(u))
            if u_norm <= 1e-12:
                raise ValueError("Failed to build tangent basis from normal.")
        u = u / u_norm

        v = np.array(
            (
                n[1] * u[2] - n[2] * u[1],
                n[2] * u[0] - n[0] * u[2],
                n[0] * u[1] - n[1] * u[0],
            ),
            dtype=np.float64,
        )
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-12:
            raise ValueError("Failed to build tangent basis from normal.")
        v = v / v_norm
        if self.verify:
            if self.reference is None:
                raise AssertionError("Scalar tangent verification requires a reference.")
            reference_u, reference_v = self.reference(normal)
            if not np.array_equal(u, reference_u) or not np.array_equal(v, reference_v):
                raise AssertionError(
                    "Scalar tangent basis differs from the original NumPy cross result."
                )
            self.exact_basis_checks += 1
        return u, v

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result["scalar_cross_fast_path"] = True
        result["verify_original_basis"] = bool(self.verify)
        result["exact_basis_checks"] = int(self.exact_basis_checks)
        return result


class ParallelMainlineBaseViews:
    """Run the three independent mainline score views concurrently.

    Each view still executes the unmodified ``_compute_ablation_score`` chain.
    Only the scheduling of the three independent calls changes; their inputs,
    per-view reduction order, returned arrays, and final dict order are kept.
    """

    def __init__(self, backbone_module: Any, max_workers: int = 3) -> None:
        self.backbone = backbone_module
        self.max_workers = int(max_workers)
        self.calls = 0

    def build(
        self,
        *,
        c_abs_enhanced: np.ndarray,
        hks_pure: np.ndarray,
        hks_va: np.ndarray,
        points: np.ndarray,
        neighbor_indices: list[np.ndarray],
        feature_config: Any,
        query_workers: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        self.calls += 1
        b = self.backbone
        radii = tuple(float(value) for value in feature_config.local_contrast_radii)
        neighbor_cache = None
        if bool(feature_config.use_local_contrast_neighbor_cache):
            neighbor_cache = b.build_local_contrast_neighbor_cache(
                points=points,
                radii=radii,
                include_self=True,
                query_workers=int(query_workers),
                profiler=None,
                include_substages=False,
            )

        specs = (
            (
                "c_abs_only",
                c_abs_enhanced.reshape(-1, 1),
                False,
                "c_abs_enhanced",
            ),
            (
                "pure_hks_only",
                np.column_stack((hks_pure[:, 0], hks_pure[:, 1], hks_pure[:, 2])),
                False,
                "pure_hks",
            ),
            (
                "variation_aware_hks_only",
                np.column_stack((hks_va[:, 0], hks_va[:, 1], hks_va[:, 2])),
                True,
                "variation_aware_hks",
            ),
        )

        def compute(spec: tuple[str, np.ndarray, bool, str]) -> tuple[str, dict[str, Any]]:
            variant, features, gamma_effective, source = spec
            name = b._ablation_name(variant, float(feature_config.va_hks_gamma))
            result = b._compute_ablation_score(
                name=name,
                view_name=variant,
                features=features,
                points=points,
                neighbor_indices=neighbor_indices,
                gamma=float(feature_config.va_hks_gamma),
                gamma_effective=gamma_effective,
                feature_source=source,
                local_contrast_neighbor_cache=neighbor_cache,
                profiler=None,
                include_substages=False,
                knn_smooth_iterations=int(feature_config.knn_smooth_iterations),
                local_radii=radii,
            )
            return variant, result

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(specs)),
            thread_name_prefix="covert-base-view",
        ) as executor:
            ordered_results = list(executor.map(compute, specs))

        views: dict[str, dict[str, Any]] = {}
        for variant, result in ordered_results:
            score_local = np.asarray(result["score_local"], dtype=np.float64).reshape(-1)
            views[variant] = {
                "variant": variant,
                "name": result["name"],
                "score_local": score_local,
                "score_norm": b._percentile_minmax_normalize(score_local),
                "ablation_result": result,
            }
        cache_summary = {
            "enabled": bool(feature_config.use_local_contrast_neighbor_cache),
            "used": bool(neighbor_cache is not None),
            "radii": list(radii),
            "query_workers_requested": int(query_workers),
        }
        return views, cache_summary

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "max_workers": int(self.max_workers),
            "per_view_function_changed": False,
            "per_view_reduction_order_changed": False,
        }


class ParallelRadiusNeighborCacheBuilder:
    """Query the three independent radii concurrently with one worker each."""

    def __init__(self, backbone_module: Any, max_workers: int = 3) -> None:
        self.backbone = backbone_module
        self.max_workers = int(max_workers)
        self.calls = 0

    def build(
        self,
        points: np.ndarray,
        radii: Sequence[float],
        *,
        include_self: bool = True,
        query_workers: int = -1,
        verify_against_workers1: bool = False,
        profiler: Any | None = None,
        include_substages: bool = True,
    ) -> Any:
        from scipy.spatial import cKDTree

        self.calls += 1
        b = self.backbone
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        if not radii:
            raise ValueError("radii must contain at least one radius value.")
        if not include_self:
            raise ValueError("Local-contrast cache currently requires include_self=True.")
        normalized_radii = tuple(float(radius) for radius in radii)
        wall_start = time.perf_counter()
        tree = cKDTree(points)

        def query(radius: float) -> list[np.ndarray]:
            try:
                raw_lists = tree.query_ball_point(points, r=radius, workers=1)
            except TypeError:
                raw_lists = tree.query_ball_point(points, r=radius)
            return b._normalize_neighbor_lists(raw_lists)

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(normalized_radii)),
            thread_name_prefix="covert-radius",
        ) as executor:
            ordered_lists = list(executor.map(query, normalized_radii))
        neighbor_indices_by_radius = {
            radius_index: neighbors
            for radius_index, neighbors in enumerate(ordered_lists)
        }
        return b.LocalContrastNeighborCache(
            radii=normalized_radii,
            neighbor_indices_by_radius=neighbor_indices_by_radius,
            num_points=int(points.shape[0]),
            include_self=True,
            points_signature=b._points_signature(points),
            num_radius_queries=len(normalized_radii),
            primary_num_radius_queries=len(normalized_radii),
            reference_num_radius_queries=0,
            actual_num_radius_queries=len(normalized_radii),
            kdtree_build_count=1,
            build_wall_time_sec=float(time.perf_counter() - wall_start),
            query_workers_requested=int(query_workers),
            query_workers_effective=1,
            query_workers_fallback_used=False,
            query_workers_fallback_reason=None,
            neighbor_order_exact_vs_workers1=True,
            num_different_neighbor_lists_vs_workers1=0,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "radius_task_workers": int(self.max_workers),
            "workers_per_radius": 1,
            "neighbor_order": "workers_1_exact",
        }


class NestedRadiusNeighborCacheBuilder:
    """Query the largest radius once and derive ordered smaller-radius CSR rows."""

    def __init__(
        self,
        backbone_module: Any,
        *,
        verify: bool = False,
        return_sorted: bool | None = None,
        verify_order: bool = True,
    ) -> None:
        if (
            _filter_csr_by_radius_numba_kernel is None
            or _filter_csr_by_two_radii_numba_kernel is None
        ):
            raise RuntimeError("Numba is required for nested-radius filtering.")
        self.backbone = backbone_module
        self.verify = bool(verify)
        self.return_sorted = return_sorted
        self.verify_order = bool(verify_order)
        self.calls = 0
        self.query_seconds = 0.0
        self.csr_conversion_seconds = 0.0
        self.filter_seconds = 0.0
        self.verify_seconds = 0.0
        self.derived_radius_count = 0
        self.reference_radius_queries = 0
        self.different_neighbor_rows = 0
        self.different_neighbor_order_rows = 0

    @staticmethod
    def _rows_to_csr(rows: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        num_rows = len(rows)
        lengths = np.fromiter(
            (len(row) for row in rows), dtype=np.int64, count=num_rows
        )
        offsets = np.empty(num_rows + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        indices = np.fromiter(
            itertools.chain.from_iterable(rows),
            dtype=np.int64,
            count=int(offsets[-1]),
        )
        return offsets, indices

    def build(
        self,
        points: np.ndarray,
        radii: Sequence[float],
        *,
        include_self: bool = True,
        query_workers: int = 1,
        verify_against_workers1: bool = False,
        profiler: Any | None = None,
        include_substages: bool = True,
    ) -> Any:
        from scipy.spatial import cKDTree

        self.calls += 1
        b = self.backbone
        points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        normalized_radii = tuple(float(radius) for radius in radii)
        if not normalized_radii:
            raise ValueError("radii must contain at least one radius value.")
        if any(not np.isfinite(radius) or radius <= 0.0 for radius in normalized_radii):
            raise ValueError("radii must contain finite positive values.")
        if not include_self:
            raise ValueError("Local-contrast cache currently requires include_self=True.")
        requested_workers = int(query_workers)
        effective_workers = requested_workers
        fallback_used = False
        fallback_reason: str | None = None
        wall_started = time.perf_counter()
        tree = cKDTree(points)
        maximum_radius = max(normalized_radii)

        query_started = time.perf_counter()
        try:
            try:
                query_kwargs: dict[str, Any] = {"workers": effective_workers}
                if self.return_sorted is not None:
                    query_kwargs["return_sorted"] = self.return_sorted
                maximum_rows = tree.query_ball_point(
                    points, r=maximum_radius, **query_kwargs
                )
            except TypeError as exc:
                if effective_workers == 1:
                    maximum_rows = tree.query_ball_point(
                        points,
                        r=maximum_radius,
                        **(
                            {"return_sorted": self.return_sorted}
                            if self.return_sorted is not None
                            else {}
                        ),
                    )
                else:
                    effective_workers = 1
                    fallback_used = True
                    fallback_reason = f"workers unsupported; used workers=1: {exc}"
                    fallback_kwargs: dict[str, Any] = {"workers": 1}
                    if self.return_sorted is not None:
                        fallback_kwargs["return_sorted"] = self.return_sorted
                    maximum_rows = tree.query_ball_point(
                        points, r=maximum_radius, **fallback_kwargs
                    )
        finally:
            self.query_seconds += time.perf_counter() - query_started

        conversion_started = time.perf_counter()
        maximum_offsets, maximum_indices = self._rows_to_csr(maximum_rows)
        self.csr_conversion_seconds += time.perf_counter() - conversion_started
        rows_by_radius: dict[int, CSRNeighborRows] = {}
        derived = [
            (radius_index, radius)
            for radius_index, radius in enumerate(normalized_radii)
            if radius != maximum_radius
        ]
        if len(derived) == 2:
            filter_started = time.perf_counter()
            (
                first_offsets,
                first_indices,
                second_offsets,
                second_indices,
            ) = _filter_csr_by_two_radii_numba_kernel(
                points,
                maximum_offsets,
                maximum_indices,
                float(derived[0][1] * derived[0][1]),
                float(derived[1][1] * derived[1][1]),
            )
            self.filter_seconds += time.perf_counter() - filter_started
            self.derived_radius_count += 2
            rows_by_radius[derived[0][0]] = CSRNeighborRows(
                first_offsets, first_indices
            )
            rows_by_radius[derived[1][0]] = CSRNeighborRows(
                second_offsets, second_indices
            )
        else:
            for radius_index, radius in derived:
                filter_started = time.perf_counter()
                offsets, indices = _filter_csr_by_radius_numba_kernel(
                    points,
                    maximum_offsets,
                    maximum_indices,
                    float(radius * radius),
                )
                self.filter_seconds += time.perf_counter() - filter_started
                self.derived_radius_count += 1
                rows_by_radius[radius_index] = CSRNeighborRows(offsets, indices)
        for radius_index, radius in enumerate(normalized_radii):
            if radius == maximum_radius:
                rows_by_radius[radius_index] = CSRNeighborRows(
                    maximum_offsets, maximum_indices
                )

        exact_vs_reference: bool | None = None
        reference_queries_this_call = 0
        differences_this_call: int | None = None
        order_differences_this_call: int | None = None
        if self.verify or verify_against_workers1:
            verify_started = time.perf_counter()
            differences = 0
            order_differences = 0
            for radius_index, radius in enumerate(normalized_radii):
                try:
                    reference = tree.query_ball_point(points, r=radius, workers=1)
                except TypeError:
                    reference = tree.query_ball_point(points, r=radius)
                self.reference_radius_queries += 1
                reference_queries_this_call += 1
                candidate = rows_by_radius[radius_index]
                for point_index, reference_row in enumerate(reference):
                    reference_array = np.asarray(reference_row, dtype=np.int64)
                    candidate_array = candidate[point_index]
                    order_equal = np.array_equal(reference_array, candidate_array)
                    if not order_equal:
                        order_differences += 1
                    if self.verify_order:
                        set_equal = order_equal
                    else:
                        set_equal = bool(
                            reference_array.size == candidate_array.size
                            and np.array_equal(
                                reference_array,
                                np.sort(candidate_array),
                            )
                        )
                    if not set_equal:
                        differences += 1
            self.different_neighbor_rows += differences
            self.different_neighbor_order_rows += order_differences
            differences_this_call = differences
            order_differences_this_call = order_differences
            exact_vs_reference = order_differences == 0
            self.verify_seconds += time.perf_counter() - verify_started
            if differences:
                raise AssertionError(
                    "Nested-radius derivation differs from independent radius "
                    f"queries in {differences} neighbor sets."
                )

        return b.LocalContrastNeighborCache(
            radii=normalized_radii,
            neighbor_indices_by_radius=rows_by_radius,
            num_points=int(points.shape[0]),
            include_self=True,
            points_signature=b._points_signature(points),
            num_radius_queries=1 + reference_queries_this_call,
            primary_num_radius_queries=1,
            reference_num_radius_queries=reference_queries_this_call,
            actual_num_radius_queries=1 + reference_queries_this_call,
            kdtree_build_count=1,
            build_wall_time_sec=float(time.perf_counter() - wall_started),
            query_workers_requested=requested_workers,
            query_workers_effective=effective_workers,
            query_workers_fallback_used=fallback_used,
            query_workers_fallback_reason=fallback_reason,
            neighbor_order_exact_vs_workers1=exact_vs_reference,
            num_different_neighbor_lists_vs_workers1=order_differences_this_call,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "primary_radius_queries": int(self.calls),
            "derived_radius_count": int(self.derived_radius_count),
            "query_seconds": float(self.query_seconds),
            "csr_conversion_seconds": float(self.csr_conversion_seconds),
            "filter_seconds": float(self.filter_seconds),
            "verify_reference_queries": int(self.reference_radius_queries),
            "verify_seconds": float(self.verify_seconds),
            "different_neighbor_rows": int(self.different_neighbor_rows),
            "different_neighbor_order_rows": int(
                self.different_neighbor_order_rows
            ),
            "neighbor_set_verified": bool(self.verify),
            "verify_order": bool(self.verify_order),
            "query_return_sorted": self.return_sorted,
            "neighbor_order_preserved": self.return_sorted is not False,
            "distance_dtype": "float64",
        }


class ParallelACNormalEstimator:
    """Estimate independent per-point PCA normals in deterministic chunks."""

    def __init__(self, ac_module: Any, max_workers: int = 8) -> None:
        self.ac = ac_module
        self.max_workers = int(max_workers)
        self.calls = 0

    def estimate(
        self,
        points: np.ndarray,
        neighbor_indices: list[np.ndarray] | np.ndarray,
        min_neighbors: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self.calls += 1
        ac = self.ac
        points = ac.validate_points(points)
        num_points = int(points.shape[0])
        neighbor_indices = ac.coerce_neighbor_indices(neighbor_indices, num_points)
        min_neighbors = int(min_neighbors)
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be non-negative.")

        worker_count = min(self.max_workers, max(1, num_points))
        bounds = np.linspace(0, num_points, worker_count + 1, dtype=np.int64)

        def process(chunk_index: int) -> tuple[int, np.ndarray, np.ndarray, int, int]:
            start = int(bounds[chunk_index])
            stop = int(bounds[chunk_index + 1])
            chunk_normals = np.full((stop - start, 3), np.nan, dtype=np.float64)
            chunk_valid = np.zeros(stop - start, dtype=bool)
            low_count = 0
            insufficient_count = 0
            for local_index, point_index in enumerate(range(start, stop)):
                neighbors = neighbor_indices[point_index]
                if neighbors.size < min_neighbors:
                    low_count += 1
                if neighbors.size < 3:
                    insufficient_count += 1
                    continue
                local_points = points[neighbors]
                centered = local_points - np.mean(local_points, axis=0, keepdims=True)
                covariance = centered.T @ centered
                try:
                    _, eigenvectors = np.linalg.eigh(covariance)
                except np.linalg.LinAlgError:
                    continue
                normal = eigenvectors[:, 0]
                normal_norm = float(np.linalg.norm(normal))
                if not np.isfinite(normal_norm) or normal_norm <= 1e-12:
                    continue
                chunk_normals[local_index] = normal / normal_norm
                chunk_valid[local_index] = True
            return start, chunk_normals, chunk_valid, low_count, insufficient_count

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="covert-ac-normal",
        ) as executor:
            chunks = list(executor.map(process, range(worker_count)))

        normals = np.full((num_points, 3), np.nan, dtype=np.float64)
        normal_valid_mask = np.zeros(num_points, dtype=bool)
        low_neighbor_count = 0
        insufficient_for_pca_count = 0
        for start, chunk_normals, chunk_valid, low_count, insufficient_count in chunks:
            stop = start + int(chunk_valid.size)
            normals[start:stop] = chunk_normals
            normal_valid_mask[start:stop] = chunk_valid
            low_neighbor_count += int(low_count)
            insufficient_for_pca_count += int(insufficient_count)
        normal_meta = {
            "num_points": int(num_points),
            "min_neighbors": int(min_neighbors),
            "low_neighbor_count": int(low_neighbor_count),
            "insufficient_for_pca_count": int(insufficient_for_pca_count),
            "normal_invalid_count": int(np.sum(~normal_valid_mask)),
            "normal_valid_ratio": float(np.mean(normal_valid_mask)) if num_points else 0.0,
        }
        return normals, normal_valid_mask, normal_meta

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "max_workers": int(self.max_workers),
            "per_point_pca_operations_changed": False,
            "result_assembly_order": "original_point_index",
        }


class ExactGrowthFastPath:
    """Inline invariant response gates while preserving both BFS traversals."""

    def __init__(self, frontend_module: Any, original: Any, *, verify: bool = False) -> None:
        self.frontend = frontend_module
        self.original = original
        self.verify = bool(verify)
        self.calls = 0
        self.exact_result_checks = 0

    def grow(
        self,
        response: np.ndarray,
        neighbor_indices: list[np.ndarray] | np.ndarray,
        root_mask: np.ndarray,
        root_hop_budget: np.ndarray,
        normal_threshold: float,
        eps: float = 1e-12,
        blocked_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        self.calls += 1
        f = self.frontend
        response = np.asarray(response, dtype=np.float64).reshape(-1)
        num_points = int(response.shape[0])
        if num_points == 0:
            raise ValueError("response must contain at least one point.")
        if not np.isfinite(response).all():
            raise ValueError("response must contain only finite values.")
        neighbor_list = f.coerce_neighbor_indices(neighbor_indices, num_points)
        root_mask = f._as_bool_mask(root_mask, name="root_mask", length=num_points)
        blocked = (
            np.zeros(num_points, dtype=bool)
            if blocked_mask is None
            else f._as_bool_mask(blocked_mask, name="blocked_mask", length=num_points)
        )
        root_hop_budget = f._as_vector(
            root_hop_budget,
            name="root_hop_budget",
            length=num_points,
            dtype=np.int32,
        )
        normal_threshold = float(normal_threshold)
        eps = float(eps)
        if not np.isfinite(normal_threshold):
            raise ValueError("normal_threshold must be finite.")
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("eps must be finite and non-negative.")
        if np.any(root_hop_budget[~root_mask] != -1):
            raise ValueError("Non-root hop budgets must equal -1.")
        if np.any(root_hop_budget[root_mask] < 0):
            raise ValueError("Root hop budgets must be non-negative.")
        if np.any(root_mask & blocked):
            raise ValueError("root_mask and blocked_mask must be disjoint.")

        # These two invariant expressions are identical to the original edge
        # predicate, but are evaluated once per response instead of once per
        # visited edge.
        upper_response = response + eps
        lower_response = normal_threshold - eps
        lower_gate = response >= lower_response

        best_remaining = np.full(num_points, -1, dtype=np.int32)
        growth_parent = np.full(num_points, -1, dtype=np.int64)
        growth_parent_remaining = np.full(num_points, -1, dtype=np.int32)
        propagation_queue: deque[int] = deque()
        root_indices = np.flatnonzero(root_mask)
        for root_index in root_indices:
            budget = int(root_hop_budget[root_index])
            best_remaining[root_index] = max(int(best_remaining[root_index]), budget)
            if budget > 0:
                propagation_queue.append(int(root_index))
        while propagation_queue:
            current_index = propagation_queue.popleft()
            remaining = int(best_remaining[current_index])
            if remaining <= 0:
                continue
            current_upper = upper_response[current_index]
            for next_value in neighbor_list[current_index]:
                next_index = int(next_value)
                if blocked[next_index]:
                    continue
                if response[next_index] > current_upper or not lower_gate[next_index]:
                    continue
                new_remaining = remaining - 1
                if new_remaining <= int(best_remaining[next_index]):
                    continue
                best_remaining[next_index] = new_remaining
                growth_parent[next_index] = current_index
                growth_parent_remaining[next_index] = remaining
                if new_remaining > 0:
                    propagation_queue.append(next_index)

        suppression_region_mask = best_remaining >= 0
        grown_only_mask = suppression_region_mask & ~root_mask
        max_budget = int(np.max(root_hop_budget[root_mask])) if np.any(root_mask) else 0
        unseen_depth = np.iinfo(np.int32).max
        state_min_depth = np.full(
            (num_points, max_budget + 1), unseen_depth, dtype=np.int32
        )
        min_growth_depth = np.full(num_points, -1, dtype=np.int32)
        depth_queue: deque[tuple[int, int]] = deque()
        for root_index in root_indices:
            budget = int(root_hop_budget[root_index])
            state_min_depth[root_index, budget] = 0
            min_growth_depth[root_index] = 0
            if budget > 0:
                depth_queue.append((int(root_index), budget))
        while depth_queue:
            current_index, remaining = depth_queue.popleft()
            current_depth = int(state_min_depth[current_index, remaining])
            if remaining <= 0:
                continue
            new_remaining = remaining - 1
            current_upper = upper_response[current_index]
            for next_value in neighbor_list[current_index]:
                next_index = int(next_value)
                if blocked[next_index]:
                    continue
                if response[next_index] > current_upper or not lower_gate[next_index]:
                    continue
                new_depth = current_depth + 1
                if new_depth >= int(state_min_depth[next_index, new_remaining]):
                    continue
                state_min_depth[next_index, new_remaining] = new_depth
                if (
                    min_growth_depth[next_index] < 0
                    or new_depth < int(min_growth_depth[next_index])
                ):
                    min_growth_depth[next_index] = new_depth
                if new_remaining > 0:
                    depth_queue.append((next_index, new_remaining))

        if not np.array_equal(min_growth_depth >= 0, suppression_region_mask):
            raise AssertionError(
                "Depth-state reachability disagrees with best_remaining growth."
            )
        reached_by_parent = np.flatnonzero(growth_parent >= 0)
        for child_index in reached_by_parent:
            parent_index = int(growth_parent[child_index])
            if not np.any(neighbor_list[parent_index] == child_index):
                raise AssertionError(
                    "Recorded growth parent edge is not present in the shared graph."
                )
            if (
                response[int(child_index)] > upper_response[parent_index]
                or not lower_gate[int(child_index)]
            ):
                raise AssertionError(
                    "Recorded growth parent edge violates a response constraint."
                )
            if (
                int(best_remaining[child_index])
                != int(growth_parent_remaining[child_index]) - 1
            ):
                raise AssertionError(
                    "Recorded growth parent edge violates remaining-hop accounting."
                )
        result = {
            "best_remaining": best_remaining,
            "min_growth_depth": min_growth_depth,
            "grown_only_mask": grown_only_mask.astype(bool, copy=False),
            "suppression_region_mask": suppression_region_mask.astype(bool, copy=False),
            "growth_parent": growth_parent,
            "growth_parent_remaining": growth_parent_remaining,
        }
        if self.verify:
            reference = self.original(
                response=response,
                neighbor_indices=neighbor_indices,
                root_mask=root_mask,
                root_hop_budget=root_hop_budget,
                normal_threshold=normal_threshold,
                eps=eps,
                blocked_mask=blocked_mask,
            )
            mismatches = [
                name
                for name in result
                if not np.array_equal(result[name], reference[name])
            ]
            if mismatches:
                raise AssertionError(
                    "Exact growth fast path differs from original arrays: "
                    + ", ".join(mismatches)
                )
            self.exact_result_checks += 1
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "verify_original_result": bool(self.verify),
            "exact_result_checks": int(self.exact_result_checks),
            "queue_order_changed": False,
            "neighbor_order_changed": False,
        }


class DirectCanonicalHKSWeightBuilders:
    """Avoid COO reconstruction only when the input CSR is canonical."""

    def __init__(
        self,
        backbone_module: Any,
        edge_weight_module: Any,
        original_pure: Any,
        original_va: Any,
        *,
        verify: bool = False,
    ) -> None:
        self.backbone = backbone_module
        self.edge_weight = edge_weight_module
        self.original_pure = original_pure
        self.original_va = original_va
        self.verify = bool(verify)
        self.pure_calls = 0
        self.va_calls = 0
        self.exact_matrix_checks = 0
        self.fallback_calls = 0

    @staticmethod
    def _same_csr(lhs: Any, rhs: Any) -> bool:
        return bool(
            lhs.shape == rhs.shape
            and np.array_equal(lhs.indptr, rhs.indptr)
            and np.array_equal(lhs.indices, rhs.indices)
            and np.array_equal(lhs.data, rhs.data)
        )

    def pure(self, adjacency_dist: Any) -> tuple[Any, dict[str, Any]]:
        from scipy import sparse

        self.pure_calls += 1
        adjacency_dist = adjacency_dist.tocsr()
        if not adjacency_dist.has_canonical_format:
            self.fallback_calls += 1
            return self.original_pure(adjacency_dist)
        if adjacency_dist.nnz == 0:
            return self.original_pure(adjacency_dist)
        distances = adjacency_dist.data.astype(np.float64, copy=False)
        positive_distances = distances[distances > 0.0]
        if positive_distances.size > 0:
            sigma = float(np.median(positive_distances))
            if not np.isfinite(sigma) or sigma <= 0.0:
                sigma = float(np.mean(positive_distances))
        else:
            sigma = 1.0
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
        result = sparse.csr_matrix(
            (
                weights,
                adjacency_dist.indices.copy(),
                adjacency_dist.indptr.copy(),
            ),
            shape=adjacency_dist.shape,
            dtype=np.float64,
        )
        result.eliminate_zeros()
        summary = {
            "weight_mode": self.backbone.PURE_HKS_WEIGHT_MODE,
            "sigma_mode": self.backbone.PURE_HKS_SIGMA_MODE,
            "sigma": float(sigma),
            "num_edges": int(result.nnz),
            "weight_min": float(weights.min()) if weights.size else 0.0,
            "weight_max": float(weights.max()) if weights.size else 0.0,
            "weight_mean": float(weights.mean()) if weights.size else 0.0,
        }
        if self.verify:
            reference, reference_summary = self.original_pure(adjacency_dist)
            if not self._same_csr(result, reference) or summary != reference_summary:
                raise AssertionError("Direct pure-HKS CSR differs from COO reference.")
            self.exact_matrix_checks += 1
        return result, summary

    def variation(self, adjacency_dist: Any, c_abs: np.ndarray, gamma: float = 5.0) -> Any:
        from scipy import sparse

        self.va_calls += 1
        e = self.edge_weight
        adjacency_dist = e._validate_adjacency(adjacency_dist)
        c_abs = e._validate_c_abs(c_abs, num_nodes=adjacency_dist.shape[0])
        gamma = float(gamma)
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative.")
        if adjacency_dist.nnz == 0:
            return sparse.csr_matrix(adjacency_dist.shape, dtype=np.float64)
        if not adjacency_dist.has_canonical_format:
            self.fallback_calls += 1
            return self.original_va(adjacency_dist, c_abs, gamma=gamma)

        row = np.repeat(
            np.arange(adjacency_dist.shape[0], dtype=np.int64),
            np.diff(adjacency_dist.indptr),
        )
        col = adjacency_dist.indices.astype(np.int64, copy=False)
        dist = adjacency_dist.data.astype(np.float64, copy=False)
        if np.any(row == col):
            self.fallback_calls += 1
            return self.original_va(adjacency_dist, c_abs, gamma=gamma)
        positive_mask = dist > 0.0
        mean_dist = (
            float(np.mean(dist[positive_mask])) if np.any(positive_mask) else 0.0
        )
        mean_dist_denom = mean_dist + 1e-8
        c_u = c_abs[row]
        c_v = c_abs[col]
        penalty = np.maximum(c_u, c_v)
        weights = np.exp(-(gamma * penalty * dist) / mean_dist_denom)
        result = sparse.csr_matrix(
            (
                weights,
                adjacency_dist.indices.copy(),
                adjacency_dist.indptr.copy(),
            ),
            shape=adjacency_dist.shape,
            dtype=np.float64,
        )
        result.eliminate_zeros()
        if self.verify:
            reference = self.original_va(adjacency_dist, c_abs, gamma=gamma)
            if not self._same_csr(result, reference):
                raise AssertionError("Direct VA-HKS CSR differs from COO reference.")
            self.exact_matrix_checks += 1
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "pure_calls": int(self.pure_calls),
            "variation_aware_calls": int(self.va_calls),
            "verify_original_matrices": bool(self.verify),
            "exact_matrix_checks": int(self.exact_matrix_checks),
            "fallback_calls": int(self.fallback_calls),
            "hks_solver_changed": False,
        }


class TunedHKSEigenSolver:
    """Isolated ARPACK parameter trial for the runtime copy only."""

    def __init__(
        self,
        hks_module: Any,
        *,
        tolerance: float,
        ncv: int | None = None,
        deterministic_v0: bool = False,
        max_eigenvalues: int | None = None,
    ) -> None:
        self.hks_module = hks_module
        self.tolerance = float(tolerance)
        self.ncv = None if ncv is None else int(ncv)
        self.deterministic_v0 = bool(deterministic_v0)
        self.max_eigenvalues = (
            None if max_eigenvalues is None else int(max_eigenvalues)
        )
        self.calls = 0
        self.wall_seconds = 0.0

    def solve(
        self,
        laplacian: Any,
        num_eigenvalues: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.sparse.linalg import eigsh

        started = time.perf_counter()
        matrix = self.hks_module._validate_laplacian(laplacian)
        num_nodes = int(matrix.shape[0])
        requested = int(num_eigenvalues)
        if requested <= 0:
            raise ValueError("num_eigenvalues must be positive.")
        if num_nodes < 2:
            raise ValueError("laplacian must contain at least 2 nodes.")
        if self.max_eigenvalues is not None:
            requested = min(requested, self.max_eigenvalues)
        k = min(requested, num_nodes - 1)
        kwargs: dict[str, Any] = {
            "k": k,
            "which": "SM",
            "tol": self.tolerance,
        }
        if self.ncv is not None:
            kwargs["ncv"] = min(max(self.ncv, k + 2), num_nodes)
        if self.deterministic_v0:
            positions = np.arange(1, num_nodes + 1, dtype=np.float64)
            v0 = np.sin(positions * 0.7548776662466927)
            v0 += np.cos(positions * 0.5698402909980532)
            v0 /= np.linalg.norm(v0)
            kwargs["v0"] = v0
        eigenvalues, eigenvectors = eigsh(matrix, **kwargs)
        order = np.argsort(eigenvalues)
        eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
        eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
        self.calls += 1
        self.wall_seconds += time.perf_counter() - started
        return eigenvalues, eigenvectors

    def summary(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "tolerance": float(self.tolerance),
            "ncv": self.ncv,
            "deterministic_v0": bool(self.deterministic_v0),
            "max_eigenvalues": self.max_eigenvalues,
            "num_eigenvalues_changed": self.max_eigenvalues is not None,
            "solver_changed": False,
            "wall_seconds": float(self.wall_seconds),
        }


@contextmanager
def _apply_runtime_optimization(mode: str) -> Iterator[dict[str, Any]]:
    """Apply one isolated runtime-only optimization and restore all symbols."""

    normalized = str(mode).strip().lower()
    unsorted_radius_mode = normalized in {
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
    }
    nested_verify_mode = normalized in {
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
    }
    audit: dict[str, Any] = {
        "mode": normalized,
        "production_modules_edited": False,
        "mathematical_operations_changed": False,
    }
    if normalized == "none":
        yield audit
        return
    if normalized not in {
        "neighbor_validation_cache",
        "neighbor_and_loop_cache",
        "neighbor_loop_geometry_cache",
        "neighbor_loop_trusted_ac",
        "neighbor_loop_scalar_ac",
        "parallel_radius_scalar_ac",
        "parallel_normals_scalar_ac",
        "bucketed_reductions_scalar_ac",
        "bucketed_reductions_scalar_ac_verify",
        "bucketed_growth_scalar_ac",
        "bucketed_growth_scalar_ac_verify",
        "bucketed_growth_fps_skip_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac_verify",
        "bucketed_growth_scalar_ac_full_verify",
        "numba_csr_growth_scalar_ac",
        "numba_csr_trusted_graph_growth_scalar_ac",
        "r3_hks_tol2e3",
        "r3_hks_k80",
        "r3_hks_ncv256",
        "r4_parallel_fps8_ncv256",
        "r4_parallel_fps16_ncv256",
        "r5_nested_radius_fps8_ncv256",
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
        "parallel_base_views_trusted_ac",
    }:
        raise ValueError(f"Unsupported runtime optimization mode: {mode!r}")

    from vast.boundary import ac_boundary, boundary_expand, graph_closing
    from vast.covert import _frontend

    original = ac_boundary.coerce_neighbor_indices
    cache = (
        TrustedNeighborValidationCache(original)
        if normalized in {
            "numba_csr_trusted_graph_growth_scalar_ac",
            "r3_hks_tol2e3",
            "r3_hks_k80",
            "r3_hks_ncv256",
            "r4_parallel_fps8_ncv256",
            "r4_parallel_fps16_ncv256",
            "r5_nested_radius_fps8_ncv256",
            "r5_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps8_ncv256",
            "r6_unsorted_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps16_ncv256",
            "r6_unsorted_nested_radius_fps16_ncv256_verify",
        }
        else NeighborValidationCache(original)
    )
    patched: list[tuple[Any, str, Any]] = [
        (ac_boundary, "coerce_neighbor_indices", ac_boundary.coerce_neighbor_indices),
        (_frontend, "coerce_neighbor_indices", _frontend.coerce_neighbor_indices),
        (graph_closing, "coerce_neighbor_indices", graph_closing.coerce_neighbor_indices),
        (boundary_expand, "coerce_neighbor_indices", boundary_expand.coerce_neighbor_indices),
    ]
    for owner, attribute, _ in patched:
        setattr(owner, attribute, cache.coerce)
    audit.update(
        {
            "description": (
                "Validate each neighbor-list object once per inference and reuse "
                "the original accepted int64 arrays on later calls."
            ),
            "patched_bindings": [
                f"{owner.__name__}.{attribute}" for owner, attribute, _ in patched
            ],
        }
    )
    if isinstance(cache, TrustedNeighborValidationCache):
        audit["description"] += (
            " Trust full-length neighbor lists constructed inside this pipeline "
            "instead of rescanning every row and index at each first use."
        )
    loop_cache: RepeatedIndexLoopCache | None = None
    if normalized in {
        "neighbor_and_loop_cache",
        "neighbor_loop_geometry_cache",
        "neighbor_loop_trusted_ac",
        "neighbor_loop_scalar_ac",
        "parallel_radius_scalar_ac",
        "parallel_normals_scalar_ac",
        "bucketed_reductions_scalar_ac",
        "bucketed_reductions_scalar_ac_verify",
        "bucketed_growth_scalar_ac",
        "bucketed_growth_scalar_ac_verify",
        "bucketed_growth_fps_skip_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac_verify",
        "bucketed_growth_scalar_ac_full_verify",
        "numba_csr_growth_scalar_ac",
        "numba_csr_trusted_graph_growth_scalar_ac",
        "r3_hks_tol2e3",
        "r3_hks_k80",
        "r3_hks_ncv256",
        "r4_parallel_fps8_ncv256",
        "r4_parallel_fps16_ncv256",
        "r5_nested_radius_fps8_ncv256",
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
        "parallel_base_views_trusted_ac",
    }:
        from vast.covert import _backbone

        if normalized in {
            "numba_csr_growth_scalar_ac",
            "numba_csr_trusted_graph_growth_scalar_ac",
            "r3_hks_tol2e3",
            "r3_hks_k80",
            "r3_hks_ncv256",
            "r4_parallel_fps8_ncv256",
            "r4_parallel_fps16_ncv256",
            "r5_nested_radius_fps8_ncv256",
            "r5_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps8_ncv256",
            "r6_unsorted_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps16_ncv256",
            "r6_unsorted_nested_radius_fps16_ncv256_verify",
        }:
            loop_cache = NumbaCSRIndexLoopCache()
        elif normalized in {
            "bucketed_reductions_scalar_ac",
            "bucketed_reductions_scalar_ac_verify",
            "bucketed_growth_scalar_ac",
            "bucketed_growth_scalar_ac_verify",
            "bucketed_growth_fps_skip_scalar_ac",
            "bucketed_growth_hks_csr_scalar_ac",
            "bucketed_growth_hks_csr_scalar_ac_verify",
            "bucketed_growth_scalar_ac_full_verify",
        }:
            loop_cache = LengthBucketedIndexLoopCache(
                verify_each_reduction=normalized in {
                    "bucketed_reductions_scalar_ac_verify",
                    "bucketed_growth_scalar_ac_full_verify",
                }
            )
        else:
            loop_cache = RepeatedIndexLoopCache()
        loop_patches = [
            (
                _backbone,
                "_knn_spatial_smooth_scores",
                _backbone._knn_spatial_smooth_scores,
            ),
            (
                _backbone,
                "_compute_local_contrast_from_neighbors",
                _backbone._compute_local_contrast_from_neighbors,
            ),
        ]
        patched.extend(loop_patches)
        _backbone._knn_spatial_smooth_scores = loop_cache.knn_smooth_scores
        _backbone._compute_local_contrast_from_neighbors = (
            loop_cache.local_contrast_from_neighbors
        )
        if isinstance(loop_cache, NumbaCSRIndexLoopCache):
            normalization_patch = (
                _backbone,
                "_normalize_neighbor_lists",
                _backbone._normalize_neighbor_lists,
            )
            patched.append(normalization_patch)
            _backbone._normalize_neighbor_lists = lambda raw_lists: raw_lists
            audit["description"] += (
                " Retain cKDTree radius-query rows until one ordered CSR conversion, "
                "then run float64 neighborhood means in a compiled sequential kernel."
            )
            audit["patched_bindings"].append(
                f"{normalization_patch[0].__name__}.{normalization_patch[1]}"
            )
            audit["relaxed_equivalence"] = {
                "enabled": True,
                "neighbor_order_changed": bool(unsorted_radius_mode),
                "accumulator_dtype_changed": False,
                "reduction_implementation_changed": True,
                "trusted_internal_neighbor_validation": isinstance(
                    cache, TrustedNeighborValidationCache
                ),
            }
        elif isinstance(loop_cache, LengthBucketedIndexLoopCache):
            audit["description"] += (
                " Group equal-length KNN/radius neighborhoods into bounded "
                "row blocks; every row retains its original neighbor order and "
                "uses NumPy mean along that contiguous row."
            )
        else:
            audit["description"] += (
                " Reuse identical KNN hood arrays and normalized radius-neighbor "
                "views while retaining the original per-point np.mean calls."
            )
        audit["patched_bindings"].extend(
            f"{owner.__name__}.{attribute}" for owner, attribute, _ in loop_patches
        )
    tangent_cache: TangentBasisConstantCache | None = None
    if normalized in {
        "neighbor_loop_geometry_cache",
        "neighbor_loop_trusted_ac",
        "neighbor_loop_scalar_ac",
        "parallel_radius_scalar_ac",
        "parallel_normals_scalar_ac",
        "bucketed_reductions_scalar_ac",
        "bucketed_reductions_scalar_ac_verify",
        "bucketed_growth_scalar_ac",
        "bucketed_growth_scalar_ac_verify",
        "bucketed_growth_fps_skip_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac_verify",
        "bucketed_growth_scalar_ac_full_verify",
        "numba_csr_growth_scalar_ac",
        "numba_csr_trusted_graph_growth_scalar_ac",
        "r3_hks_tol2e3",
        "r3_hks_k80",
        "r3_hks_ncv256",
        "r4_parallel_fps8_ncv256",
        "r4_parallel_fps16_ncv256",
        "r5_nested_radius_fps8_ncv256",
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
        "parallel_base_views_trusted_ac",
    }:
        if normalized in {
            "neighbor_loop_scalar_ac",
            "parallel_radius_scalar_ac",
            "parallel_normals_scalar_ac",
            "bucketed_reductions_scalar_ac",
            "bucketed_reductions_scalar_ac_verify",
            "bucketed_growth_scalar_ac",
            "bucketed_growth_scalar_ac_verify",
            "bucketed_growth_fps_skip_scalar_ac",
            "bucketed_growth_hks_csr_scalar_ac",
            "bucketed_growth_hks_csr_scalar_ac_verify",
            "bucketed_growth_scalar_ac_full_verify",
            "numba_csr_growth_scalar_ac",
            "numba_csr_trusted_graph_growth_scalar_ac",
            "r3_hks_tol2e3",
            "r3_hks_k80",
            "r3_hks_ncv256",
            "r4_parallel_fps8_ncv256",
            "r4_parallel_fps16_ncv256",
            "r5_nested_radius_fps8_ncv256",
            "r5_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps8_ncv256",
            "r6_unsorted_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps16_ncv256",
            "r6_unsorted_nested_radius_fps16_ncv256_verify",
        }:
            tangent_cache = ScalarCrossTrustedTangentBasisCache(
                reference=ac_boundary._tangent_basis_from_normal,
                verify=(normalized == "bucketed_growth_scalar_ac_full_verify"),
            )
        elif normalized in {"neighbor_loop_trusted_ac", "parallel_base_views_trusted_ac"}:
            tangent_cache = TrustedTangentBasisConstantCache()
        else:
            tangent_cache = TangentBasisConstantCache()
        tangent_patch = (
            ac_boundary,
            "_tangent_basis_from_normal",
            ac_boundary._tangent_basis_from_normal,
        )
        patched.append(tangent_patch)
        ac_boundary._tangent_basis_from_normal = tangent_cache.tangent_basis
        audit["description"] += (
            " Reuse the three exact AC reference-axis constants instead of "
            "allocating an identity matrix for every working point."
        )
        audit["patched_bindings"].append(
            f"{tangent_patch[0].__name__}.{tangent_patch[1]}"
        )
        if isinstance(tangent_cache, TrustedTangentBasisConstantCache):
            audit["description"] += (
                " Skip redundant shape/finiteness guards after AC has already "
                "accepted the normal through normal_valid_mask."
            )
        if isinstance(tangent_cache, ScalarCrossTrustedTangentBasisCache):
            audit["description"] += (
                " Expand the same three-dimensional cross products into their "
                "scalar component formulas, preserving operand order."
            )
    growth_fast_path: ExactGrowthFastPath | None = None
    if normalized in {
        "bucketed_growth_scalar_ac",
        "bucketed_growth_scalar_ac_verify",
        "bucketed_growth_fps_skip_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac_verify",
        "bucketed_growth_scalar_ac_full_verify",
        "numba_csr_growth_scalar_ac",
        "numba_csr_trusted_graph_growth_scalar_ac",
        "r3_hks_tol2e3",
        "r3_hks_k80",
        "r3_hks_ncv256",
        "r4_parallel_fps8_ncv256",
        "r4_parallel_fps16_ncv256",
        "r5_nested_radius_fps8_ncv256",
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
    }:
        original_growth = _frontend.grow_response_descending
        growth_fast_path = ExactGrowthFastPath(
            _frontend,
            original_growth,
            verify=normalized in {
                "bucketed_growth_scalar_ac_verify",
                "bucketed_growth_scalar_ac_full_verify",
            },
        )
        growth_patch = (
            _frontend,
            "grow_response_descending",
            original_growth,
        )
        patched.append(growth_patch)
        _frontend.grow_response_descending = growth_fast_path.grow
        audit["description"] += (
            " Precompute the invariant descending-growth response gates and "
            "inline the same comparisons while retaining both BFS orders."
        )
        audit["patched_bindings"].append(
            f"{growth_patch[0].__name__}.{growth_patch[1]}"
        )
    hks_weight_builders: DirectCanonicalHKSWeightBuilders | None = None
    if normalized in {
        "bucketed_growth_hks_csr_scalar_ac",
        "bucketed_growth_hks_csr_scalar_ac_verify",
    }:
        from vast.covert import _backbone
        from vast.graph import edge_weight

        hks_weight_builders = DirectCanonicalHKSWeightBuilders(
            _backbone,
            edge_weight,
            _backbone._build_gaussian_distance_weights,
            edge_weight.compute_variation_aware_weights,
            verify=(normalized == "bucketed_growth_hks_csr_scalar_ac_verify"),
        )
        pure_weight_patch = (
            _backbone,
            "_build_gaussian_distance_weights",
            _backbone._build_gaussian_distance_weights,
        )
        va_weight_patch = (
            edge_weight,
            "compute_variation_aware_weights",
            edge_weight.compute_variation_aware_weights,
        )
        patched.extend((pure_weight_patch, va_weight_patch))
        _backbone._build_gaussian_distance_weights = hks_weight_builders.pure
        edge_weight.compute_variation_aware_weights = hks_weight_builders.variation
        audit["description"] += (
            " Reuse canonical CSR topology when constructing the two HKS weight "
            "matrices, without changing their data or the ARPACK solver."
        )
        audit["patched_bindings"].extend(
            (
                f"{pure_weight_patch[0].__name__}.{pure_weight_patch[1]}",
                f"{va_weight_patch[0].__name__}.{va_weight_patch[1]}",
            )
        )
    hks_solver_trial: TunedHKSEigenSolver | None = None
    if normalized in {
        "r3_hks_tol2e3",
        "r3_hks_k80",
        "r3_hks_ncv256",
        "r4_parallel_fps8_ncv256",
        "r4_parallel_fps16_ncv256",
        "r5_nested_radius_fps8_ncv256",
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
    }:
        from vast.features import hks as hks_module

        hks_solver_trial = TunedHKSEigenSolver(
            hks_module,
            tolerance=2e-3 if normalized == "r3_hks_tol2e3" else 1e-3,
            ncv=(
                256
                if normalized in {
                    "r3_hks_ncv256",
                    "r4_parallel_fps8_ncv256",
                    "r4_parallel_fps16_ncv256",
                    "r5_nested_radius_fps8_ncv256",
                    "r5_nested_radius_fps8_ncv256_verify",
                    "r6_unsorted_nested_radius_fps8_ncv256",
                    "r6_unsorted_nested_radius_fps8_ncv256_verify",
                    "r6_unsorted_nested_radius_fps16_ncv256",
                    "r6_unsorted_nested_radius_fps16_ncv256_verify",
                }
                else None
            ),
            deterministic_v0=False,
            max_eigenvalues=80 if normalized == "r3_hks_k80" else None,
        )
        hks_solver_patch = (
            hks_module,
            "compute_eigen_decomposition",
            hks_module.compute_eigen_decomposition,
        )
        patched.append(hks_solver_patch)
        hks_module.compute_eigen_decomposition = hks_solver_trial.solve
        if normalized == "r3_hks_tol2e3":
            audit["description"] += (
                " Relax the existing ARPACK residual tolerance from 1e-3 to 2e-3; "
                "the solver, which='SM', k=100, and all HKS equations stay unchanged."
            )
        elif normalized == "r3_hks_k80":
            audit["description"] += (
                " Reduce the HKS eigenspace from k=100 to k=80 while retaining "
                "ARPACK, which='SM', tol=1e-3, adaptive scales, and HKS equations."
            )
        else:
            audit["description"] += (
                " Increase ARPACK's Krylov search subspace from its default 201 "
                "to ncv=256 while retaining k=100, which='SM', tol=1e-3, and "
                "all HKS equations."
            )
        audit["patched_bindings"].append(
            f"{hks_solver_patch[0].__name__}.{hks_solver_patch[1]}"
        )
    fps_trial_enabled = False
    if normalized == "bucketed_growth_fps_skip_scalar_ac":
        if _farthest_point_sample_exact_numba_skip_selected_kernel is None:
            raise RuntimeError("Numba is required for the exact FPS skip-selected trial.")
        from vast.data import preprocess as data_preprocess

        fps_patch = (
            data_preprocess,
            "_farthest_point_sample_exact_numba_kernel",
            data_preprocess._farthest_point_sample_exact_numba_kernel,
        )
        patched.append(fps_patch)
        data_preprocess._farthest_point_sample_exact_numba_kernel = (
            _farthest_point_sample_exact_numba_skip_selected_kernel
        )
        fps_trial_enabled = True
        audit["description"] += (
            " In exact FPS, retain zero minimum distance for selected points and "
            "skip their later distance recomputation; argmax remains sequential."
        )
        audit["patched_bindings"].append(
            f"{fps_patch[0].__name__}.{fps_patch[1]}"
        )
    parallel_fps_enabled = False
    previous_numba_threads: int | None = None
    if normalized in {
        "r4_parallel_fps8_ncv256",
        "r4_parallel_fps16_ncv256",
        "r5_nested_radius_fps8_ncv256",
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
    }:
        if (
            _farthest_point_sample_exact_numba_parallel_kernel is None
            or get_num_threads is None
            or set_num_threads is None
        ):
            raise RuntimeError("Numba parallel support is required for the FPS trial.")
        from vast.data import preprocess as data_preprocess

        previous_numba_threads = int(get_num_threads())
        requested_threads = (
            16
            if normalized in {
                "r4_parallel_fps16_ncv256",
                "r6_unsorted_nested_radius_fps16_ncv256",
                "r6_unsorted_nested_radius_fps16_ncv256_verify",
            }
            else 8
        )
        trial_threads = min(requested_threads, previous_numba_threads)
        set_num_threads(trial_threads)
        fps_patch = (
            data_preprocess,
            "_farthest_point_sample_exact_numba_kernel",
            data_preprocess._farthest_point_sample_exact_numba_kernel,
        )
        patched.append(fps_patch)
        data_preprocess._farthest_point_sample_exact_numba_kernel = (
            _farthest_point_sample_exact_numba_parallel_kernel
        )
        parallel_fps_enabled = True
        audit["description"] += (
            f" Run each exact-FPS pointwise distance update on up to {trial_threads} "
            "Numba threads; keep float64 arithmetic and the serial first-max argmax."
        )
        audit["patched_bindings"].append(
            f"{fps_patch[0].__name__}.{fps_patch[1]}"
        )
    parallel_normal_estimator: ParallelACNormalEstimator | None = None
    if normalized == "parallel_normals_scalar_ac":
        parallel_normal_estimator = ParallelACNormalEstimator(ac_boundary, max_workers=8)
        normal_patch = (
            ac_boundary,
            "estimate_normals_pca",
            ac_boundary.estimate_normals_pca,
        )
        patched.append(normal_patch)
        ac_boundary.estimate_normals_pca = parallel_normal_estimator.estimate
        audit["description"] += (
            " Schedule independent per-point AC PCA normal calculations in "
            "deterministic chunks while preserving each point's NumPy operations."
        )
        audit["patched_bindings"].append(
            f"{normal_patch[0].__name__}.{normal_patch[1]}"
        )
    parallel_radius_builder: ParallelRadiusNeighborCacheBuilder | None = None
    if normalized == "parallel_radius_scalar_ac":
        from vast.covert import _backbone

        parallel_radius_builder = ParallelRadiusNeighborCacheBuilder(
            _backbone, max_workers=3
        )
        radius_patch = (
            _backbone,
            "build_local_contrast_neighbor_cache",
            _backbone.build_local_contrast_neighbor_cache,
        )
        patched.append(radius_patch)
        _backbone.build_local_contrast_neighbor_cache = parallel_radius_builder.build
        audit["description"] += (
            " Query the three independent local-contrast radii concurrently, "
            "using the exact workers=1 query path for each radius."
        )
        audit["patched_bindings"].append(
            f"{radius_patch[0].__name__}.{radius_patch[1]}"
        )
    nested_radius_builder: NestedRadiusNeighborCacheBuilder | None = None
    if normalized in {
        "r5_nested_radius_fps8_ncv256",
        "r5_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps8_ncv256",
        "r6_unsorted_nested_radius_fps8_ncv256_verify",
        "r6_unsorted_nested_radius_fps16_ncv256",
        "r6_unsorted_nested_radius_fps16_ncv256_verify",
    }:
        from vast.covert import _backbone

        nested_radius_builder = NestedRadiusNeighborCacheBuilder(
            _backbone,
            verify=nested_verify_mode,
            return_sorted=False if unsorted_radius_mode else None,
            verify_order=not unsorted_radius_mode,
        )
        nested_radius_patch = (
            _backbone,
            "build_local_contrast_neighbor_cache",
            _backbone.build_local_contrast_neighbor_cache,
        )
        patched.append(nested_radius_patch)
        _backbone.build_local_contrast_neighbor_cache = nested_radius_builder.build
        audit["description"] += (
            " Query only the largest local-contrast radius, then derive each "
            "smaller-radius row by ordered float64 squared-distance filtering. "
            "The maximum-radius query row order and inclusive radius rule are retained."
        )
        if unsorted_radius_mode:
            audit["description"] += (
                " Disable cKDTree's per-row neighbor sorting while keeping exact "
                "neighbor membership; downstream float64 reduction order may change."
            )
        if nested_verify_mode:
            audit["description"] += (
                " Verify every derived row against the corresponding independent "
                "workers=1 cKDTree radius query and fail on any neighbor-set mismatch."
            )
        audit["patched_bindings"].append(
            f"{nested_radius_patch[0].__name__}.{nested_radius_patch[1]}"
        )
    parallel_base_views: ParallelMainlineBaseViews | None = None
    if normalized == "parallel_base_views_trusted_ac":
        from vast.covert import _backbone

        parallel_base_views = ParallelMainlineBaseViews(_backbone, max_workers=3)
        base_views_patch = (
            _backbone,
            "_build_mainline_base_views",
            _backbone._build_mainline_base_views,
        )
        patched.append(base_views_patch)
        _backbone._build_mainline_base_views = parallel_base_views.build
        audit["description"] += (
            " Skip redundant AC normal guards after normal_valid_mask has already "
            "accepted the normal. Run the three independent mainline score-view "
            "chains concurrently without changing any per-view computation."
        )
        audit["patched_bindings"].append(
            f"{base_views_patch[0].__name__}.{base_views_patch[1]}"
        )
    try:
        yield audit
    finally:
        audit["neighbor_validation_cache"] = cache.summary()
        if loop_cache is not None:
            audit["repeated_index_loop_cache"] = loop_cache.summary()
        if tangent_cache is not None:
            audit["tangent_basis_constant_cache"] = tangent_cache.summary()
        if parallel_base_views is not None:
            audit["parallel_mainline_base_views"] = parallel_base_views.summary()
        if parallel_radius_builder is not None:
            audit["parallel_radius_neighbor_cache"] = parallel_radius_builder.summary()
        if nested_radius_builder is not None:
            audit["nested_radius_neighbor_cache"] = nested_radius_builder.summary()
        if parallel_normal_estimator is not None:
            audit["parallel_ac_normal_estimator"] = parallel_normal_estimator.summary()
        if growth_fast_path is not None:
            audit["exact_growth_fast_path"] = growth_fast_path.summary()
        if fps_trial_enabled:
            audit["exact_fps_skip_selected"] = {
                "enabled": True,
                "distance_formula_changed_for_unselected_points": False,
                "argmax_order_changed": False,
                "tie_breaking_changed": False,
            }
        if parallel_fps_enabled:
            audit["parallel_exact_fps"] = {
                "enabled": True,
                "threads": min(requested_threads, int(previous_numba_threads or requested_threads)),
                "distance_dtype_changed": False,
                "argmax_order_changed": False,
                "tie_breaking_changed": False,
            }
        if hks_weight_builders is not None:
            audit["direct_canonical_hks_weight_builders"] = (
                hks_weight_builders.summary()
            )
        if hks_solver_trial is not None:
            audit["tuned_hks_eigen_solver"] = hks_solver_trial.summary()
        for owner, attribute, previous in reversed(patched):
            setattr(owner, attribute, previous)
        if previous_numba_threads is not None and set_num_threads is not None:
            set_num_threads(previous_numba_threads)


def _timed_wrapper(
    collector: RuntimeStageCollector,
    group: str,
    name: str,
    function: Any,
) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with collector.measure(group, name):
            return function(*args, **kwargs)

    return wrapped


@contextmanager
def _instrument_runtime_stages(
    collector: RuntimeStageCollector,
) -> Iterator[dict[str, Any]]:
    """Temporarily time existing call boundaries and restore them afterwards.

    All wrappers forward identical arguments and return the original result
    object.  The production source modules are not edited, and every monkey
    patch is restored in ``finally`` even if inference raises.
    """

    from vast.covert import _backbone, _gtr

    patched: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, attribute: str, replacement: Any) -> None:
        original = getattr(owner, attribute)
        patched.append((owner, attribute, original))
        setattr(owner, attribute, replacement)

    # Backbone substages are sequential and form an interpretable breakdown of
    # the backbone total.  Small array conversions remain as unattributed time.
    backbone_targets = (
        (_backbone.preprocess, "fps_downsample", "01_fps_downsample"),
        (_backbone.preprocess, "isotropic_scale", "02_isotropic_scale"),
        (_backbone, "build_mutual_knn_graph", "03_mutual_knn_graph"),
        (_backbone, "compute_surface_variation", "04_surface_variation"),
        (_backbone, "_compute_pure_hks", "05_pure_hks"),
        (_backbone, "_compute_variation_aware_hks", "06_variation_aware_hks"),
        (_backbone, "_build_mainline_base_views", "07_mainline_base_views"),
    )
    for owner, attribute, stage_name in backbone_targets:
        patch(
            owner,
            attribute,
            _timed_wrapper(
                collector, "backbone", stage_name, getattr(owner, attribute)
            ),
        )

    pipeline_targets = (
        (_backbone, "run_covert_backbone", "01_backbone_total"),
        (_pipeline._frontend, "run_covert_frontend", "02_frontend_total"),
        (_pipeline, "_build_exp21_gtr_adapter", "03_frontend_to_gtr_adapter"),
        (_pipeline, "_build_component_rule_result", "05_scale_consistency_rules"),
        (
            _pipeline,
            "_build_local_normal_u_cleanup_request",
            "06_local_normal_cleanup_request",
        ),
    )
    for owner, attribute, stage_name in pipeline_targets:
        patch(
            owner,
            attribute,
            _timed_wrapper(
                collector, "pipeline", stage_name, getattr(owner, attribute)
            ),
        )

    gtr_profiler = _gtr.RuntimeProfiler(enabled=True)
    preview_profiler = _gtr.RuntimeProfiler(enabled=True)
    state = {"production_gtr_depth": 0}

    original_gtr_core = _gtr._run_covert_gtr_core

    @functools.wraps(original_gtr_core)
    def profiled_gtr_core(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("profiler") is None:
            kwargs["profiler"] = gtr_profiler
        return original_gtr_core(*args, **kwargs)

    patch(_gtr, "_run_covert_gtr_core", profiled_gtr_core)

    original_closing = _gtr._run_covert_gtr_closing

    @functools.wraps(original_closing)
    def profiled_closing(*args: Any, **kwargs: Any) -> Any:
        if state["production_gtr_depth"] > 0:
            return original_closing(*args, **kwargs)
        if kwargs.get("profiler") is None:
            kwargs["profiler"] = preview_profiler
        with collector.measure("pipeline", "04_component_rule_preview_closing"):
            return original_closing(*args, **kwargs)

    patch(_gtr, "_run_covert_gtr_closing", profiled_closing)

    original_run_gtr = _gtr.run_covert_gtr

    @functools.wraps(original_run_gtr)
    def profiled_run_gtr(*args: Any, **kwargs: Any) -> Any:
        state["production_gtr_depth"] += 1
        try:
            with collector.measure("pipeline", "07_gtr_total"):
                return original_run_gtr(*args, **kwargs)
        finally:
            state["production_gtr_depth"] -= 1

    patch(_gtr, "run_covert_gtr", profiled_run_gtr)

    try:
        yield {
            "gtr_profiler": gtr_profiler,
            "preview_profiler": preview_profiler,
        }
    finally:
        for owner, attribute, original in reversed(patched):
            setattr(owner, attribute, original)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _ragged_array_sha256(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(values)).encode("ascii"))
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stage_output_hashes(result: CovertSampleResult) -> dict[str, str]:
    arrays = {
        "fps_indices": result.fps_indices,
        "working_points": result.points,
        "raw_points_aligned": result.raw_points_aligned,
        "pure_hks_features": result.pure_hks_features,
        "va_hks_features": result.va_hks_features,
        "original_sv": result.original_sv,
        "enhanced_sv": result.enhanced_sv,
        "pure_hks_response": result.pure_hks_response,
        "va_hks_response": result.va_hks_response,
        "raw_ac_mask": result.raw_ac_mask,
        "final_ac_suppression_mask": result.final_ac_suppression_mask,
        "pure_hks_root_mask": result.pure_hks_root_mask,
        "pure_hks_suppression_mask": result.pure_hks_suppression_mask,
        "defect_protected_component_mask": result.defect_protected_component_mask,
        "sv_boundary_suppressed": result.sv_boundary_suppressed,
        "va_hks_boundary_suppressed": result.va_hks_boundary_suppressed,
        "semantic_confidence": result.semantic_confidence,
        "seed_confidence": result.seed_confidence,
        "seed_confidence_normalized": result.seed_confidence_normalized,
        "high_confidence_seed_mask": result.high_confidence_seed_mask,
        "q_component_labels": result.q_component_labels,
        "u_component_labels": result.u_component_labels,
        "y_component_labels": result.y_component_labels,
        "raw_z_component_labels": result.raw_z_component_labels,
        "authoritative_z_component_labels": result.authoritative_z_component_labels,
        "w_component_labels": result.w_component_labels,
        "gtr_candidate_labels": result.gtr_candidate_labels,
        "closed_mask": result.closed_mask,
        "restoration_unknown_mask": result.restoration_unknown_mask,
        "restoration_support_mask": result.restoration_support_mask,
        "restored_points": result.restored_points,
        "displacement_magnitude": result.displacement_magnitude,
        "stress_proxy": result.stress_proxy,
        "rejected_mask": result.rejected_mask,
        "final_mask": result.final_mask,
    }
    hashes = {name: _array_sha256(value) for name, value in arrays.items()}
    hashes["neighbor_indices"] = _ragged_array_sha256(result.neighbor_indices)
    hashes["object_scores"] = _canonical_json_sha256(dict(result.object_scores))
    return hashes


def _profile_function_name(filename: str, line: int, function: str) -> str:
    path = Path(filename)
    try:
        shown = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        shown = str(path)
    return f"{shown}:{line}:{function}"


def _cprofile_tables(
    profiler: cProfile.Profile | None,
    *,
    top_n: int,
) -> dict[str, Any] | None:
    if profiler is None:
        return None
    stats = pstats.Stats(profiler)
    rows: list[dict[str, Any]] = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, self_time, cumulative_time, _ = values
        rows.append(
            {
                "function": _profile_function_name(filename, line, function),
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "self_seconds": float(self_time),
                "cumulative_seconds": float(cumulative_time),
            }
        )
    return {
        "note": (
            "cProfile is diagnostic: its instrumentation adds overhead. Use "
            "stage_profile wall time, from a --no-cprofile run, for runtime claims."
        ),
        "top_by_self_time": sorted(
            rows, key=lambda row: float(row["self_seconds"]), reverse=True
        )[:top_n],
        "top_by_cumulative_time": sorted(
            rows, key=lambda row: float(row["cumulative_seconds"]), reverse=True
        )[:top_n],
    }


def _gtr_stage_rows(profiler: Any, denominator: float) -> list[dict[str, Any]]:
    summary = profiler.summary()
    # These are the sequential GTR blocks.  Per-component restoration records
    # are nested under 04_06 and intentionally excluded from this additive table.
    excluded_prefixes = ("04_06_01_",)
    rows: list[dict[str, Any]] = []
    for row in summary["stages_sorted_by_wall_time"]:
        name = str(row["name"])
        if name.startswith(excluded_prefixes):
            continue
        if not name.startswith("04_"):
            continue
        wall = float(row["wall_time_sec"])
        rows.append(
            {
                "name": name,
                "wall_seconds": wall,
                "cpu_seconds": float(row["cpu_time_sec"]),
                "calls": int(row["count"]),
                "wall_percent": float(100.0 * wall / denominator)
                if denominator > 0.0
                else 0.0,
            }
        )
    return sorted(rows, key=lambda row: float(row["wall_seconds"]), reverse=True)


def _append_unattributed(
    rows: list[dict[str, Any]], total_seconds: float, name: str
) -> list[dict[str, Any]]:
    attributed = float(sum(float(row["wall_seconds"]) for row in rows))
    residual = max(float(total_seconds) - attributed, 0.0)
    output = list(rows)
    output.append(
        {
            "name": name,
            "wall_seconds": residual,
            "cpu_seconds": None,
            "calls": 1,
            "wall_percent": float(100.0 * residual / total_seconds)
            if total_seconds > 0.0
            else 0.0,
        }
    )
    return sorted(output, key=lambda row: float(row["wall_seconds"]), reverse=True)


def _build_runtime_profile_payload(
    *,
    result: CovertSampleResult,
    collector: RuntimeStageCollector,
    instrumentation: Mapping[str, Any],
    runner_wall_seconds: float,
    cprofile_result: dict[str, Any] | None,
    query_workers: int,
    optimization_audit: Mapping[str, Any],
) -> dict[str, Any]:
    pipeline_seconds = float(result.timings.pipeline_seconds)
    pipeline_rows = _append_unattributed(
        collector.table("pipeline", pipeline_seconds),
        pipeline_seconds,
        "08_pipeline_unattributed_array_checks",
    )
    backbone_seconds = collector.seconds("pipeline", "01_backbone_total")
    backbone_rows = _append_unattributed(
        collector.table("backbone", backbone_seconds),
        backbone_seconds,
        "08_backbone_unattributed_array_packaging",
    )
    gtr_seconds = collector.seconds("pipeline", "07_gtr_total")
    instrumentation["gtr_profiler"].pipeline_wall_time_sec = gtr_seconds
    preview_seconds = collector.seconds(
        "pipeline", "04_component_rule_preview_closing"
    )
    instrumentation["preview_profiler"].pipeline_wall_time_sec = preview_seconds
    return {
        "schema_version": "covert-runtime-profile-v1",
        "measurement_semantics": {
            "stage_profile": (
                "Lightweight wrappers around the unchanged production call graph; "
                "pipeline/backbone/GTR tables are individually additive aside from "
                "timer resolution and wrapper overhead."
            ),
            "cprofile": (
                "Optional function-level diagnostic with observer overhead; not "
                "used as the authoritative wall-clock measurement."
            ),
            "ground_truth": "Loaded only after prediction; excluded from inference timing.",
            "artifact_writes": "Runtime profile is written after all measured regions.",
        },
        "sample": {
            "sample_id": result.sample_id,
            "pc_path": result.pc_path,
            "gt_path": result.gt_path,
            "raw_point_count": int(result.raw_points_aligned.shape[0]),
            "working_point_count": int(result.points.shape[0]),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "query_workers": int(query_workers),
        },
        "optimization": dict(optimization_audit),
        "identity": {
            "config_hash": result.config_hash,
            "config_source": result.config_source,
            "final_mask_sha256": _array_sha256(result.final_mask),
            "working_points_sha256": _array_sha256(result.points),
            "predicted_positive_points": int(np.sum(result.final_mask)),
            "metrics": dict(result.metrics),
            "stage_output_hashes": _stage_output_hashes(result),
        },
        "wall_clock": {
            "runner_call_seconds": float(runner_wall_seconds),
            **dataclasses.asdict(result.timings),
        },
        "stage_profile": {
            "pipeline": pipeline_rows,
            "backbone": backbone_rows,
            "gtr": _append_unattributed(
                _gtr_stage_rows(instrumentation["gtr_profiler"], gtr_seconds),
                gtr_seconds,
                "04_13_gtr_unattributed_wrapper_and_packaging",
            ),
            "component_rule_preview": _append_unattributed(
                _gtr_stage_rows(
                    instrumentation["preview_profiler"], preview_seconds
                ),
                preview_seconds,
                "04_04_preview_unattributed_wrapper",
            ),
        },
        "cprofile": cprofile_result,
    }


def _format_profile_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Stage | Wall (s) | Share (%) | Calls |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {float(row['wall_seconds']):.6f} | "
            f"{float(row['wall_percent']):.2f} | {int(row['calls'])} |"
        )
    return lines


def _runtime_profile_markdown(payload: Mapping[str, Any]) -> str:
    sample = payload["sample"]
    identity = payload["identity"]
    wall = payload["wall_clock"]
    stages = payload["stage_profile"]
    optimization = payload["optimization"]
    lines = [
        f"# COVERT runtime profile: {sample['sample_id']}",
        "",
        "## Run identity",
        "",
        f"- Point cloud: `{sample['pc_path']}`",
        f"- Ground truth: `{sample['gt_path']}`",
        f"- Raw / working points: {sample['raw_point_count']} / {sample['working_point_count']}",
        f"- Config hash: `{identity['config_hash']}`",
        f"- Runtime optimization: `{optimization['mode']}`",
        f"- Final-mask SHA-256: `{identity['final_mask_sha256']}`",
        f"- Predicted positive points: {identity['predicted_positive_points']}",
        "",
        "## Wall clock",
        "",
        f"- Input load: {float(wall['input_load_seconds']):.6f} s",
        f"- Pipeline: {float(wall['pipeline_seconds']):.6f} s",
        f"- Downsampling (inside pipeline): {float(wall['downsampling_seconds']):.6f} s",
        f"- Algorithm excluding downsampling: {float(wall['algorithm_seconds_excluding_downsampling']):.6f} s",
        f"- GT load / evaluation: {float(wall['gt_load_seconds']):.6f} / {float(wall['evaluation_seconds']):.6f} s",
        f"- Complete runtime entry call: {float(wall['runner_call_seconds']):.6f} s",
        "",
        "## Pipeline breakdown",
        "",
        *_format_profile_table(stages["pipeline"]),
        "",
        "## Backbone breakdown",
        "",
        *_format_profile_table(stages["backbone"]),
        "",
        "## GTR breakdown",
        "",
        *_format_profile_table(stages["gtr"]),
        "",
        "## Interpretation guardrails",
        "",
        "- Percentages in each table use that table's parent stage as 100%.",
        "- The runtime wrappers forward the same objects and do not alter configuration, thresholds, masks, or numerical routines.",
        "- If cProfile was enabled, use it only for hotspot discovery; observer overhead makes that run unsuitable for headline timing.",
        "",
    ]
    return "\n".join(lines)


def _verify_reference_profile(
    payload: Mapping[str, Any], reference_path: Path
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    current_identity = payload["identity"]
    reference_identity = reference["identity"]
    checked_fields = (
        "config_hash",
        "final_mask_sha256",
        "working_points_sha256",
        "predicted_positive_points",
    )
    mismatches = {
        field: {
            "reference": reference_identity.get(field),
            "current": current_identity.get(field),
        }
        for field in checked_fields
        if reference_identity.get(field) != current_identity.get(field)
    }
    reference_stage_hashes = reference_identity.get("stage_output_hashes")
    current_stage_hashes = current_identity.get("stage_output_hashes")
    if reference_stage_hashes is not None:
        if reference_stage_hashes != current_stage_hashes:
            all_stage_names = sorted(
                set(reference_stage_hashes) | set(current_stage_hashes or {})
            )
            mismatches["stage_output_hashes"] = {
                name: {
                    "reference": reference_stage_hashes.get(name),
                    "current": (current_stage_hashes or {}).get(name),
                }
                for name in all_stage_names
                if reference_stage_hashes.get(name)
                != (current_stage_hashes or {}).get(name)
            }
    if mismatches:
        raise AssertionError(
            "Runtime output differs from reference profile: "
            + json.dumps(mismatches, ensure_ascii=False, default=str)
        )
    return {
        "reference_path": str(reference_path),
        "checked_fields": list(checked_fields),
        "stage_hash_count": int(len(reference_stage_hashes or {})),
        "identical": True,
    }


def config_to_dict(config: CovertPipelineConfig) -> dict[str, Any]:
    """Return the canonical JSON-compatible configuration tree."""

    return dataclasses.asdict(config)


def config_sha256(config: CovertPipelineConfig) -> str:
    """Return the deterministic SHA-256 of the canonical config JSON."""

    payload = json.dumps(
        config_to_dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_hash(config: CovertPipelineConfig) -> str:
    """Backward-compatible short name for :func:`config_sha256`."""

    return config_sha256(config)


def config_snapshot_payload(
    config: CovertPipelineConfig,
    *,
    source: str = "runtime.covert_runtime.DEFAULT_CONFIG",
) -> dict[str, Any]:
    """Serialize an actual config object with provenance metadata."""

    serialized_config = json.loads(
        json.dumps(config_to_dict(config), allow_nan=False)
    )
    return {
        "source": str(source),
        "config_hash": config_hash(config),
        "config": serialized_config,
    }


def _dataclass_from_mapping(cls: type[Any], values: Mapping[str, Any]) -> Any:
    hints = get_type_hints(cls)
    known = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} setting(s): {unknown}")
    kwargs: dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        expected = hints.get(item.name)
        if isinstance(expected, type) and dataclasses.is_dataclass(expected):
            if not isinstance(value, Mapping):
                raise ValueError(f"{item.name} must be an object.")
            value = _dataclass_from_mapping(expected, value)
        elif getattr(expected, "__origin__", None) is tuple and isinstance(value, list):
            value = tuple(value)
        kwargs[item.name] = value
    return cls(**kwargs)


def config_from_dict(values: Mapping[str, Any]) -> CovertPipelineConfig:
    return _dataclass_from_mapping(CovertPipelineConfig, values)


def load_config(path: str | Path) -> CovertPipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Configuration JSON root must be an object.")
    config_values = payload.get("config", payload)
    if not isinstance(config_values, Mapping):
        raise ValueError("Configuration snapshot 'config' must be an object.")
    config = config_from_dict(config_values)
    declared_hash = payload.get("config_hash")
    if declared_hash is not None and str(declared_hash) != config_hash(config):
        raise ValueError("Configuration snapshot hash does not match its config object.")
    return config


def _validate_config(config: CovertPipelineConfig) -> None:
    disabled = [
        item.name
        for item in dataclasses.fields(config.switches)
        if not bool(getattr(config.switches, item.name))
    ]
    if disabled:
        raise NotImplementedError(
            "The consolidated production entry point supports only the frozen "
            f"paper mainline; unsupported disabled switch(es): {disabled}"
        )
    if config.preprocess.use_sor:
        raise NotImplementedError("SOR is not part of the frozen paper mainline.")
    if config.new_ac.component_grouping_mode != "gap_repaired":
        raise NotImplementedError("Only gap_repaired New-AC grouping is consolidated.")
    if config.hks.pure_normal_suppression_mode != "legacy_hard":
        raise NotImplementedError("Only Legacy-hard Pure-HKS suppression is consolidated.")
    if config.gtr.local_normal_u_cleanup_mode != "final_mask_only":
        raise NotImplementedError("Only final_mask_only Local Normal-U cleanup is active.")
    if config.sgcr.growth_response_source != "enhanced_sv_q1_q99":
        raise NotImplementedError("Only enhanced-SV Q1-Q99 SGCR support is consolidated.")
    if config.sgcr.growth_mode != "energy_descent":
        raise NotImplementedError("Only energy-descent SGCR growth is consolidated.")
    if config.sgcr.floor_mode != "seed_relative":
        raise NotImplementedError("Only the seed-relative SGCR floor is consolidated.")
    if config.gtr.biharmonic_solver != "spsolve":
        raise NotImplementedError("Only the frozen spsolve biharmonic solver is consolidated.")
    if config.gtr.candidate_classifier_mode != "adaptive_normal_rejection":
        raise NotImplementedError(
            "The frozen COVERT configuration requires adaptive_normal_rejection."
        )
    if not config.gtr.local_normal_reference_enabled:
        raise NotImplementedError(
            "The frozen COVERT configuration requires Local Normal Reference."
        )
    if config.gtr.local_normal_reference_max_hops != 3:
        raise NotImplementedError(
            "The frozen COVERT configuration requires a three-hop Local Normal Reference."
        )




def _binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=bool).reshape(-1)
    gt = np.asarray(target, dtype=bool).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT must have identical working-cloud shapes.")
    tp, fp = int(np.sum(pred & gt)), int(np.sum(pred & ~gt))
    fn, tn = int(np.sum(~pred & gt)), int(np.sum(~pred & ~gt))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "gt_positive_points": int(np.sum(gt)),
        "predicted_positive_points": int(np.sum(pred)),
        "working_point_count": int(pred.size),
    }


def _object_pseudostress_scores(
    gtr_result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect continuous object scores without changing any GTR decision.

    The paper score is the maximum candidate pseudo-stress P90 over candidates
    for which restoration was actually available. P85 and P95 are collected
    from the identical candidate masks and stress field as sensitivity
    readouts. A sample with no restored candidates receives 0.0.
    """

    labels = np.asarray(
        gtr_result["decision_labels_before_seed_consistency"], dtype=np.int32
    ).reshape(-1)
    stress = np.asarray(gtr_result["stress_proxy"], dtype=np.float64).reshape(-1)
    displacement = np.asarray(
        gtr_result["displacement_magnitude"], dtype=np.float64
    ).reshape(-1)
    if labels.shape != stress.shape or labels.shape != displacement.shape:
        raise ValueError(
            "GTR decision labels, displacement, and pseudo-stress must have equal shape."
        )
    if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(displacement)):
        raise ValueError("GTR displacement/pseudo-stress contains non-finite values.")

    restoration_reports = {
        int(report["component_id"]): report
        for report in gtr_result.get("component_restoration_reports", ())
        if isinstance(report, Mapping) and report.get("component_id") is not None
    }
    classifier_result = gtr_result.get("candidate_classifier_result", {})
    classifier_reports = {
        int(report["component_id"]): report
        for report in (
            classifier_result.get("component_reports", ())
            if isinstance(classifier_result, Mapping)
            else ()
        )
        if isinstance(report, Mapping) and report.get("component_id") is not None
    }
    candidate_ids = sorted(int(value) for value in np.unique(labels) if int(value) > 0)
    candidate_readouts: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        report = restoration_reports.get(candidate_id)
        candidate_mask = labels == candidate_id
        restored = bool(
            report is not None
            and not bool(report.get("skipped", False))
            and np.any(candidate_mask)
        )
        if not restored:
            continue
        stress_p85, stress_p90, stress_p95 = np.percentile(
            stress[candidate_mask], OBJECT_PSEUDOSTRESS_PERCENTILES
        )
        displacement_p85, displacement_p90, displacement_p95 = np.percentile(
            displacement[candidate_mask], OBJECT_PSEUDOSTRESS_PERCENTILES
        )
        classifier = classifier_reports.get(candidate_id, {})
        accepted_value = classifier.get("accepted")
        accepted = None if accepted_value is None else bool(accepted_value)
        rejection_reason = classifier.get("rejection_reason")
        normal_like = bool(
            accepted is False
            and (
                bool(classifier.get("adaptive_normal_reject", False))
                or rejection_reason
                in {
                    "relative_normal",
                    "large_shallow",
                    "relative_normal_and_large_shallow",
                    "low_displacement",
                }
            )
        )
        candidate_readouts.append(
            {
                "candidate_id": candidate_id,
                "candidate_point_count": int(np.count_nonzero(candidate_mask)),
                "accepted": accepted,
                "normal_like": normal_like,
                "rejection_reason": rejection_reason,
                "classifier_mode": classifier.get("classifier_mode"),
                "displacement_p85": float(displacement_p85),
                "displacement_p90": float(displacement_p90),
                "displacement_p95": float(displacement_p95),
                "stress_p85": float(stress_p85),
                "stress_p90": float(stress_p90),
                "stress_p95": float(stress_p95),
                "local_reference_valid": bool(
                    classifier.get("local_reference_valid", False)
                ),
                "d_ratio_p95": classifier.get("d_ratio_p95"),
                "s_ratio_p95": classifier.get("s_ratio_p95"),
                "candidate_point_fraction": classifier.get(
                    "candidate_point_fraction"
                ),
                "candidate_d_p95_normalized_by_object_scale": classifier.get(
                    "candidate_d_p95_normalized_by_object_scale"
                ),
            }
        )

    def maximum(field_name: str) -> float:
        return float(max((row[field_name] for row in candidate_readouts), default=0.0))

    p85_score = maximum("stress_p85")
    p90_score = maximum("stress_p90")
    p95_score = maximum("stress_p95")
    scores = {
        "S_obj": p90_score,
        "S_obj_max_candidate_Sp85": p85_score,
        "S_obj_max_candidate_Sp90": p90_score,
        "S_obj_max_candidate_Sp95": p95_score,
        "n_restoration_attempts": int(len(restoration_reports)),
        "n_restored_candidates": int(len(candidate_readouts)),
    }
    return scores, candidate_readouts


def _labels_from_components(components: Sequence[np.ndarray], size: int) -> np.ndarray:
    labels = np.zeros(size, dtype=np.int32)
    for component_id, indices in enumerate(components, start=1):
        labels[np.asarray(indices, dtype=np.int64)] = component_id
    return labels


def _count_positive_components(labels: np.ndarray) -> int:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    return int(np.unique(values[values > 0]).size)


def print_covert_summary(
    result: CovertSampleResult,
    *,
    include_visualization_timing: bool = True,
) -> None:
    """Print a compact summary using only arrays and audits already in the result."""

    suppression_audit = result.audit.get("suppression", {})
    protected_records = suppression_audit.get("defect_veto_component_records", ())
    protected_components = sum(
        bool(record.get("defect_protected", False))
        for record in protected_records
        if isinstance(record, Mapping)
    )
    repair = result.audit.get("q_to_z_repair", {})

    print("[Covert] Mainline summary")
    print(f"[Covert] Working points: {result.points.shape[0]}")
    print(f"[Covert] Raw AC points: {int(np.sum(result.raw_ac_mask))}")
    print(f"[Covert] Final-AC points: {int(np.sum(result.final_ac_suppression_mask))}")
    print(
        "[Covert] Legacy HKS suppression points: "
        f"{int(np.sum(result.pure_hks_suppression_mask))}"
    )
    print(
        "[Covert] Defect-protected components / points: "
        f"{protected_components} / {int(np.sum(result.defect_protected_component_mask))}"
    )
    print(
        "[Covert] High-confidence seed points: "
        f"{int(np.sum(result.high_confidence_seed_mask))}"
    )
    for name, labels in (
        ("Q", result.q_component_labels),
        ("U", result.u_component_labels),
        ("Y", result.y_component_labels),
        ("Raw Z", result.raw_z_component_labels),
        ("Authoritative Z", result.authoritative_z_component_labels),
        ("W", result.w_component_labels),
    ):
        print(
            f"[Covert] {name} component count / points: "
            f"{_count_positive_components(labels)} / {int(np.sum(np.asarray(labels) > 0))}"
        )
    print(
        "[Covert] Q->Z repair required / applied: "
        f"{bool(repair.get('repair_required', False))} / "
        f"{bool(repair.get('repair_applied', False))}"
    )
    print(f"[Covert] GTR candidate points: {int(np.sum(result.gtr_candidate_mask))}")
    print(f"[Covert] Final predicted points: {int(np.sum(result.final_mask))}")
    object_scores = getattr(result, "object_scores", {})
    print(
        "[Covert] S_obj P85 / P90 / P95: "
        f"{float(object_scores.get('S_obj_max_candidate_Sp85', 0.0)):.6f} / "
        f"{float(object_scores.get('S_obj_max_candidate_Sp90', 0.0)):.6f} / "
        f"{float(object_scores.get('S_obj_max_candidate_Sp95', 0.0)):.6f}"
    )
    if result.gt_mask is not None:
        for name in ("tp", "fp", "fn"):
            print(f"[Covert] {name.upper()}: {int(result.metrics.get(name, 0))}")
        for name in ("precision", "recall", "f1"):
            print(f"[Covert] {name}: {float(result.metrics.get(name, 0.0)):.6f}")
    print(f"[Covert] input_load_seconds: {result.timings.input_load_seconds:.6f}")
    print(f"[Covert] downsampling_seconds: {result.timings.downsampling_seconds:.6f}")
    print(
        "[Covert] algorithm_seconds_excluding_downsampling: "
        f"{result.timings.algorithm_seconds_excluding_downsampling:.6f}"
    )
    print(f"[Covert] pipeline_seconds: {result.timings.pipeline_seconds:.6f}")
    print(
        "[Covert] inference_total_seconds: "
        f"{result.timings.inference_total_seconds:.6f}"
    )
    print(f"[Covert] gt_load_seconds: {result.timings.gt_load_seconds:.6f}")
    print(f"[Covert] evaluation_seconds: {result.timings.evaluation_seconds:.6f}")
    if include_visualization_timing:
        print(
            "[Covert] visualization_seconds: "
            f"{result.timings.visualization_seconds:.6f}"
        )


def _load_visualization_dependencies():
    """Import GUI-only dependencies on demand."""

    import matplotlib.pyplot as plt
    import open3d as o3d
    from vast.visualization.open3d_interaction import (
        make_reset_view_callback,
        preserve_camera_while,
        register_view_config_keys,
        resolve_view_config_window_size,
    )

    return (
        plt,
        o3d,
        make_reset_view_callback,
        preserve_camera_while,
        register_view_config_keys,
        resolve_view_config_window_size,
    )


def _visualize(result: CovertSampleResult, stages: Sequence[str]) -> None:
    (
        plt,
        o3d,
        make_reset_view_callback,
        preserve_camera_while,
        register_view_config_keys,
        resolve_view_config_window_size,
    ) = _load_visualization_dependencies()

    available: dict[str, tuple[str, np.ndarray]] = {
        "input": ("input", np.zeros(result.points.shape[0], dtype=np.float64)),
        "sv": ("response", result.original_sv),
        "enhanced-sv": ("response", result.enhanced_sv),
        "pure-hks-response": ("response", result.pure_hks_response),
        "va-hks": ("response", result.va_hks_response),
        "raw-ac": ("mask", result.raw_ac_mask),
        "final-ac": ("mask", result.final_ac_suppression_mask),
        "hks-normal-prior": ("mask", result.pure_hks_suppression_mask),
        "protected": ("mask", result.defect_protected_component_mask),
        "semantic": ("response", result.semantic_confidence),
        "seed": ("response", result.seed_confidence_normalized),
        "q": ("labels", result.q_component_labels),
        "u": ("labels", result.u_component_labels),
        "y": ("labels", result.y_component_labels),
        "raw-z": ("labels", result.raw_z_component_labels),
        "z": ("labels", result.authoritative_z_component_labels),
        "w": ("labels", result.w_component_labels),
        "gtr": ("mask", result.gtr_candidate_mask),
        "final": ("mask", result.final_mask),
        "gt": (
            "mask",
            result.gt_mask
            if result.gt_mask is not None
            else np.zeros(result.points.shape[0], dtype=bool),
        ),
    }

    selected: list[str] = []
    seen: set[str] = set()
    for stage in stages:
        if stage not in available:
            raise ValueError(f"Unknown visualization stage {stage!r}; choose {sorted(available)}")
        if stage in seen:
            continue
        seen.add(stage)
        selected.append(stage)
    if result.gt_mask is None and "gt" in selected:
        print("[View] GT unavailable; GT view will remain unmarked.")
    if not selected:
        print("[View] No available visualization stages; visualization skipped.")
        return

    stage_colors: list[np.ndarray] = []
    for stage in selected:
        kind, values = available[stage]
        values = np.asarray(values).reshape(-1)
        colors = np.tile(np.array([[0.58, 0.58, 0.58]]), (result.points.shape[0], 1))
        if kind == "response":
            finite = np.asarray(values, dtype=np.float64)
            lo, hi = np.percentile(finite, (1.0, 99.0))
            normalized = (
                np.zeros_like(finite)
                if hi <= lo + 1e-12
                else np.clip((finite - lo) / (hi - lo), 0.0, 1.0)
            )
            colors = plt.colormaps["turbo"](normalized)[:, :3]
        elif kind == "labels":
            labels = np.asarray(values, dtype=np.int64)
            positive = labels > 0
            if np.any(positive):
                colors[positive] = plt.colormaps["tab20"](
                    (labels[positive] - 1) % 20
                )[:, :3]
        elif kind == "mask":
            colors[np.asarray(values, dtype=bool)] = plt.colormaps["turbo"](0.95)[:3]
        stage_colors.append(np.asarray(colors, dtype=np.float64))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(result.points)
    cloud.colors = o3d.utility.Vector3dVector(stage_colors[0])
    width, height = resolve_view_config_window_size(1280, 800)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    created = vis.create_window(
        window_name="Covert Mainline Visualization",
        width=width,
        height=height,
    )
    if not created:
        raise RuntimeError("Open3D failed to create the Covert visualization window.")
    try:
        vis.add_geometry(cloud, reset_bounding_box=True)

        def _apply_stage(target_index: int) -> None:
            stage = selected[int(target_index)]
            cloud.colors = o3d.utility.Vector3dVector(stage_colors[int(target_index)])
            vis.update_geometry(cloud)
            vis.poll_events()
            vis.update_renderer()
            key = VISUALIZATION_STAGE_KEY_MAP[stage]
            print(f"[View] {key} -> {VISUALIZATION_STAGE_TITLES[stage]}")

        def _make_callback(target_index: int):
            def _callback(visualizer) -> bool:
                preserve_camera_while(
                    visualizer, lambda: _apply_stage(target_index)
                )
                return False

            return _callback

        for index, stage in enumerate(selected):
            key = VISUALIZATION_STAGE_KEY_MAP[stage]
            callback = _make_callback(index)
            vis.register_key_callback(ord(key), callback)
            vis.register_key_callback(ord(key.lower()), callback)
        vis.register_key_callback(
            ord(";"), make_reset_view_callback("[View] Camera reset")
        )
        register_view_config_keys(vis)

        print("\nVisualization controls:")
        for stage in selected:
            print(
                f"  {VISUALIZATION_STAGE_KEY_MAP[stage]} -> "
                f"{VISUALIZATION_STAGE_TITLES[stage]}"
            )
        print("  ; -> Reset camera")
        print("  [ -> Save window/camera/point-size config")
        print("  Space -> Restore saved window/camera/point-size config")
        _apply_stage(0)
        vis.run()
    finally:
        vis.destroy_window()


def _write_sample_artifacts(result: CovertSampleResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        field_name: np.asarray(getattr(result, field_name))
        for field_name in (
            "points", "raw_points_aligned", "pure_hks_features", "va_hks_features",
            "pure_hks_time_scales", "va_hks_time_scales",
            "original_sv", "enhanced_sv", "pure_hks_response",
            "va_hks_response", "raw_ac_mask", "final_ac_suppression_mask", "pure_hks_root_mask",
            "pure_hks_suppression_mask", "defect_protected_component_mask",
            "sv_boundary_suppressed", "va_hks_boundary_suppressed",
            "semantic_confidence", "seed_confidence", "seed_confidence_normalized",
            "high_confidence_seed_mask",
            "q_component_labels", "u_component_labels", "y_component_labels",
            "raw_z_component_labels", "authoritative_z_component_labels",
            "w_component_labels", "gtr_candidate_labels", "gtr_candidate_mask",
            "closed_mask", "restoration_unknown_mask", "restoration_support_mask",
            "restored_points", "displacement_magnitude", "stress_proxy", "rejected_mask",
            "final_mask",
        )
    }
    arrays["coordinate_scale"] = np.asarray(result.coordinate_scale, dtype=np.float64)
    arrays["sample_id"] = np.asarray(result.sample_id)
    if result.gt_mask is not None:
        arrays["gt_mask"] = np.asarray(result.gt_mask)
    np.savez_compressed(output_dir / "covert_result.npz", **arrays)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "source": result.config_source,
                "config_hash": result.config_hash,
                "config": dict(result.config_snapshot),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "sample_id": result.sample_id,
        "pc_path": result.pc_path,
        "gt_path": result.gt_path,
        "source": result.config_source,
        "config": dict(result.config_snapshot),
        "config_hash": result.config_hash,
        "metrics": dict(result.metrics),
        "object_scores": dict(result.object_scores),
        "timings": dataclasses.asdict(result.timings),
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _run_covert_sample_impl(
    pc_path: str | Path,
    gt_path: str | Path | None = None,
    *,
    config: CovertPipelineConfig = DEFAULT_CONFIG,
    query_workers: int = 1,
    visualize: bool = False,
    visualization_stages: Sequence[str] = ("final",),
    output_dir: str | Path | None = None,
    verbose: bool = False,
    config_source: str | None = None,
    strict_gt: bool = True,
) -> CovertSampleResult:
    """Run the frozen paper detector once; GT is evaluation-only.

    ``strict_gt=False`` is reserved for the interactive visualization CLI so
    an absent or unrelated label file cannot discard an otherwise valid
    inference result. Batch/API evaluation keeps the strict default.
    """

    _validate_config(config)
    resolved_config_hash = config_hash(config)
    resolved_config_source = (
        str(config_source)
        if config_source is not None
        else (
            "covert_sample.DEFAULT_CONFIG"
            if config is DEFAULT_CONFIG
            else "explicit_config_object"
        )
    )
    pc_path = Path(pc_path)
    resolved_gt = Path(gt_path) if gt_path is not None else None
    if verbose:
        print(f"[Covert] Sample: {pc_path.stem}")
        print(f"[Covert] PC path: {pc_path}")
        print(
            "[Covert] GT path: "
            f"{resolved_gt if resolved_gt is not None else 'unavailable'}"
        )
        print(f"[Covert] Config hash: {resolved_config_hash}")
        print(f"[Covert] Query workers: {query_workers}")
        print(f"[Covert] Visualization: {'enabled' if visualize else 'disabled'}")
    if not pc_path.is_file():
        raise FileNotFoundError(f"Point-cloud file not found: {pc_path}")
    if query_workers == 0 or query_workers < -1:
        raise ValueError("query_workers must be -1 or a positive integer.")

    if verbose:
        print("[Covert] Loading point cloud...")
    load_start = time.perf_counter()
    from vast.covert import _point_io

    raw_points = _point_io.load_point_cloud_file(pc_path)
    input_load_seconds = time.perf_counter() - load_start
    if verbose:
        print(f"[Covert] Raw points: {raw_points.shape[0]}")

    veto_thresholds = dataclasses.asdict(config.defect_veto)
    veto_hook = functools.partial(
        build_legacy_hks_defect_veto_override, thresholds=veto_thresholds
    )
    pipeline_start = time.perf_counter()
    pipeline = _pipeline.run_covert_pipeline(
        raw_points,
        config,
        query_workers=query_workers,
        sample_id=pc_path.stem,
        suppression_override_hook=veto_hook,
        verbose=verbose,
    )
    pipeline_seconds = time.perf_counter() - pipeline_start
    inference_total_seconds = input_load_seconds + pipeline_seconds

    frontend = pipeline["exp21_result"]
    base = frontend["base_result"]
    gtr_result = pipeline["gtr_result"]
    object_scores, object_score_candidate_readouts = _object_pseudostress_scores(
        gtr_result
    )
    size = int(np.asarray(frontend["points"]).shape[0])
    audit = frontend["suppression_strategy_audit"]

    gt_mask: np.ndarray | None = None
    metrics: dict[str, Any] = {}
    gt_load_seconds = evaluation_seconds = 0.0
    if resolved_gt is not None:
        gt_start = time.perf_counter()
        from vast.data import io as data_io

        gt_raw, gt_meta = data_io.load_optional_labels_with_meta(
            resolved_gt,
            num_points=raw_points.shape[0],
            reference_points=raw_points,
        )
        gt_load_seconds = time.perf_counter() - gt_start
        if gt_raw is None:
            if strict_gt:
                raise ValueError(
                    f"Ground truth could not be loaded or aligned: {resolved_gt}"
                )
            if verbose:
                print(
                    "[Covert] GT unavailable or unaligned; continuing without "
                    "GT evaluation so visualization remains available."
                )
            resolved_gt = None
        else:
            gt_mask = np.asarray(gt_raw, dtype=np.uint8)
            gt_mask = gt_mask[np.asarray(base["sor_indices"], dtype=np.int64)]
            gt_mask = gt_mask[np.asarray(base["fps_indices"], dtype=np.int64)].astype(bool)
            evaluation_start = time.perf_counter()
            metrics = _binary_metrics(gtr_result["final_defect_mask"], gt_mask)
            metrics["gt_meta"] = gt_meta
            evaluation_seconds = time.perf_counter() - evaluation_start

    u_labels = _labels_from_components(frontend["seed_grown_components"], size)
    downsampling_seconds = float(
        base["summaries"]["preprocess"]["fps_downsample_seconds"]
    )
    algorithm_seconds_excluding_downsampling = float(
        pipeline_seconds - downsampling_seconds
    )
    timing = CovertTiming(
        input_load_seconds=float(input_load_seconds),
        pipeline_seconds=float(pipeline_seconds),
        inference_total_seconds=float(inference_total_seconds),
        gt_load_seconds=float(gt_load_seconds),
        evaluation_seconds=float(evaluation_seconds),
        downsampling_seconds=downsampling_seconds,
        algorithm_seconds_excluding_downsampling=(
            algorithm_seconds_excluding_downsampling
        ),
    )
    result = CovertSampleResult(
        sample_id=pc_path.stem,
        pc_path=str(pc_path),
        gt_path=str(resolved_gt) if resolved_gt is not None else None,
        config_snapshot=config_to_dict(config),
        config_hash=resolved_config_hash,
        config_source=resolved_config_source,
        points=np.asarray(frontend["points"], dtype=np.float64),
        fps_indices=np.asarray(base["fps_indices"], dtype=np.int64),
        raw_points_aligned=np.asarray(base["raw_points_aligned"], dtype=np.float64),
        coordinate_scale=float(base["summaries"]["preprocess"]["scale"]["scale"]),
        neighbor_indices=tuple(np.asarray(v, dtype=np.int64) for v in frontend["neighbor_indices"]),
        pure_hks_features=np.asarray(base["hks_pure"], dtype=np.float64),
        va_hks_features=np.asarray(base["hks_va_gamma_10"], dtype=np.float64),
        pure_hks_time_scales=np.asarray(
            base["summaries"]["pure_hks"]["time_scales"], dtype=np.float64
        ),
        va_hks_time_scales=np.asarray(
            base["summaries"]["variation_aware_hks_gamma_10"]["time_scales"],
            dtype=np.float64,
        ),
        original_sv=np.asarray(base["c_abs_raw_feature"], dtype=np.float64),
        enhanced_sv=np.asarray(base["c_abs_enhanced"], dtype=np.float64),
        pure_hks_response=np.asarray(frontend["pure_hks_response"], dtype=np.float64),
        va_hks_response=np.asarray(frontend["va_hks_response"], dtype=np.float64),
        raw_ac_mask=np.asarray(frontend["raw_ac_mask"], dtype=bool),
        final_ac_suppression_mask=np.asarray(frontend["final_ac_suppression_mask"], dtype=bool),
        pure_hks_root_mask=np.asarray(frontend["pure_hks_suppression_root_mask"], dtype=bool),
        pure_hks_suppression_mask=np.asarray(
            audit["legacy_hks_suppression_mask_after_veto"], dtype=bool
        ),
        defect_protected_component_mask=np.asarray(audit["defect_protected_component_mask"], dtype=bool),
        sv_boundary_suppressed=np.asarray(frontend["sv_boundary_suppressed"], dtype=np.float64),
        va_hks_boundary_suppressed=np.asarray(frontend["va_hks_boundary_suppressed"], dtype=np.float64),
        semantic_confidence=np.asarray(frontend["semantic_confidence"], dtype=np.float64),
        seed_confidence=np.asarray(frontend["seed_confidence"], dtype=np.float64),
        seed_confidence_normalized=np.asarray(frontend["seed_confidence_norm"], dtype=np.float64),
        high_confidence_seed_mask=np.asarray(frontend["high_confidence_seed_mask"], dtype=bool),
        q_component_labels=np.asarray(frontend["seed_size_filtered_component_labels"], dtype=np.int32),
        u_component_labels=u_labels,
        y_component_labels=np.asarray(frontend["seed_label_aware_merged_component_labels"], dtype=np.int32),
        raw_z_component_labels=np.asarray(frontend["raw_seed_confidence_view_z_component_labels"], dtype=np.int32),
        authoritative_z_component_labels=np.asarray(frontend["seed_confidence_component_labels"], dtype=np.int32),
        w_component_labels=np.asarray(frontend["seed_confidence_closed_visualization_component_labels"], dtype=np.int32),
        gtr_candidate_labels=np.asarray(pipeline["component_rule_filtered_candidate_labels"], dtype=np.int32),
        gtr_candidate_mask=np.asarray(
            pipeline["component_rule_filtered_candidate_labels"], dtype=np.int32
        ) > 0,
        closed_mask=np.asarray(gtr_result["closed_mask"], dtype=bool),
        restoration_unknown_mask=np.asarray(
            gtr_result["restoration_unknown_mask"], dtype=bool
        ),
        restoration_support_mask=np.asarray(
            gtr_result["hks_open_boundary_support_inside_closed_mask"], dtype=bool
        ),
        restored_points=np.asarray(gtr_result["restored_points"], dtype=np.float64),
        displacement_magnitude=np.asarray(
            gtr_result["displacement_magnitude"], dtype=np.float64
        ),
        stress_proxy=np.asarray(gtr_result["stress_proxy"], dtype=np.float64),
        rejected_mask=np.asarray(gtr_result["rejected_mask"], dtype=bool),
        final_mask=np.asarray(gtr_result["final_defect_mask"], dtype=bool),
        gt_mask=gt_mask,
        metrics=metrics,
        timings=timing,
        audit={
            "suppression": audit,
            "q_to_z_repair": frontend["q_z_consistency_repair_audit"],
            "scale_consistency": pipeline["component_rule_result"],
            "gtr_summary": gtr_result["summary"],
            "object_screening": {
                "definition": (
                    "S_obj(Pq) = max candidate pseudo-stress Pq over restored "
                    "candidates; 0.0 when none"
                ),
                "canonical_percentile": 90,
                "candidate_readouts": object_score_candidate_readouts,
            },
        },
        object_scores=object_scores,
    )

    if verbose:
        print_covert_summary(result, include_visualization_timing=not visualize)

    visualization_seconds = 0.0
    if visualize:
        visual_start = time.perf_counter()
        _visualize(result, visualization_stages)
        visualization_seconds = time.perf_counter() - visual_start
        result = replace(result, timings=replace(result.timings, visualization_seconds=visualization_seconds))
        if verbose:
            print(f"[Covert] visualization_seconds: {visualization_seconds:.6f}")

    if output_dir is not None:
        write_start = time.perf_counter()
        _write_sample_artifacts(result, Path(output_dir))
        write_seconds = time.perf_counter() - write_start
        result = replace(result, timings=replace(result.timings, artifact_write_seconds=write_seconds))
        # Refresh result.json so it contains the final write-time field.
        summary_path = Path(output_dir) / "result.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["timings"] = dataclasses.asdict(result.timings)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print(f"[Covert] artifact_write_seconds: {write_seconds:.6f}")
    return result


def run_covert_sample(
    pc_path: str | Path,
    gt_path: str | Path | None = None,
    *,
    config: CovertPipelineConfig = DEFAULT_CONFIG,
    query_workers: int = PRODUCTION_QUERY_WORKERS,
    visualize: bool = False,
    visualization_stages: Sequence[str] = ("final",),
    output_dir: str | Path | None = None,
    verbose: bool = False,
    config_source: str | None = None,
) -> CovertSampleResult:
    """Run COVERT on one point cloud with optional aligned ground truth."""

    with _apply_runtime_optimization(PRODUCTION_OPTIMIZATION_MODE):
        return _run_covert_sample_impl(
            pc_path,
            gt_path,
            config=config,
            query_workers=query_workers,
            visualize=visualize,
            visualization_stages=visualization_stages,
            output_dir=output_dir,
            verbose=verbose,
            config_source=config_source,
        )


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text)


def _resolve_pc_path(cli_value: str | Path | None) -> Path:
    resolved = _optional_path(cli_value)
    if resolved is None or not resolved.is_file():
        detail = "" if resolved is None else f" Resolved value: {resolved}"
        raise FileNotFoundError(
            f"No valid point cloud was provided with --pc-path.{detail}"
        )
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pc-path", type=Path, required=True,
        help="Point-cloud file to process.",
    )
    parser.add_argument(
        "--gt-path", type=Path, default=None,
        help="Optional aligned ground-truth file; omit for inference-only use.",
    )
    parser.add_argument(
        "--config-json", type=Path, default=None,
        help="Optional JSON configuration; default uses the frozen COVERT config.",
    )
    parser.add_argument(
        "--query-workers", type=int, default=PRODUCTION_QUERY_WORKERS,
        help="Number of workers used by parallel neighbor queries.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional directory for per-sample detector artifacts.",
    )
    parser.add_argument(
        "--runtime-profile-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cprofile",
        dest="cprofile",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-cprofile",
        dest="cprofile",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(cprofile=False)
    parser.add_argument(
        "--cprofile-top-n",
        type=int,
        default=40,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reference-profile",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--optimization",
        choices=(
            "none",
            "neighbor_validation_cache",
            "neighbor_and_loop_cache",
            "neighbor_loop_geometry_cache",
            "neighbor_loop_trusted_ac",
            "neighbor_loop_scalar_ac",
            "parallel_radius_scalar_ac",
            "parallel_normals_scalar_ac",
            "bucketed_reductions_scalar_ac",
            "bucketed_reductions_scalar_ac_verify",
            "bucketed_growth_scalar_ac",
            "bucketed_growth_scalar_ac_verify",
            "bucketed_growth_fps_skip_scalar_ac",
            "bucketed_growth_hks_csr_scalar_ac",
            "bucketed_growth_hks_csr_scalar_ac_verify",
            "bucketed_growth_scalar_ac_full_verify",
            "numba_csr_growth_scalar_ac",
            "numba_csr_trusted_graph_growth_scalar_ac",
            "r3_hks_tol2e3",
            "r3_hks_k80",
            "r3_hks_ncv256",
            "r4_parallel_fps8_ncv256",
            "r4_parallel_fps16_ncv256",
            "r5_nested_radius_fps8_ncv256",
            "r5_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps8_ncv256",
            "r6_unsorted_nested_radius_fps8_ncv256_verify",
            "r6_unsorted_nested_radius_fps16_ncv256",
            "r6_unsorted_nested_radius_fps16_ncv256_verify",
            "parallel_base_views_trusted_ac",
        ),
        default=PRODUCTION_OPTIMIZATION_MODE,
        help=argparse.SUPPRESS,
    )
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        "--visualize", dest="visualize", action="store_true",
        help="Enable interactive visualization (disabled by default).",
    )
    visualization.add_argument(
        "--no-visualize", "--no-visualization", dest="visualize", action="store_false",
        help="Disable interactive visualization (default).",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", dest="verbose", action="store_true",
        help="Print per-stage progress and timing information.",
    )
    verbosity.add_argument(
        "--no-verbose", dest="verbose", action="store_false",
        help="Disable verbose progress output.",
    )
    parser.set_defaults(visualize=False, verbose=None)
    parser.add_argument(
        "--visualization-stage", action="append", dest="visualization_stages",
        default=None,
        choices=(
            "input", "sv", "enhanced-sv", "pure-hks-response", "va-hks",
            "raw-ac", "final-ac", "hks-normal-prior", "protected", "semantic",
            "seed", "q", "u", "y", "raw-z", "z", "w", "gtr", "final", "gt",
        ),
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pc_path = _resolve_pc_path(args.pc_path)
    gt_path = _optional_path(args.gt_path)
    visualize = bool(args.visualize)
    visualization_stages = (
        tuple(VISUALIZATION_STAGES)
        if args.visualization_stages is None
        else tuple(args.visualization_stages)
    )
    output_dir = _optional_path(
        args.output_dir if args.output_dir is not None else SINGLE_SAMPLE_OUTPUT_DIR
    )
    verbose = VERBOSE if args.verbose is None else bool(args.verbose)
    if args.config_json is None:
        config = DEFAULT_CONFIG
        config_source = "covert_sample.DEFAULT_CONFIG"
    else:
        config = load_config(args.config_json)
        config_source = "explicit_cli_config_json"
    if args.cprofile_top_n <= 0:
        raise ValueError("--cprofile-top-n must be positive.")

    collector = RuntimeStageCollector()
    function_profiler = cProfile.Profile() if bool(args.cprofile) else None
    # Install runtime-only implementations first so the timing wrappers measure
    # the optimized call boundaries as well as the copied production ones.
    with _apply_runtime_optimization(args.optimization) as optimization_audit:
        with _instrument_runtime_stages(collector) as instrumentation:
            run_start = time.perf_counter()
            if function_profiler is not None:
                function_profiler.enable()
            try:
                result = _run_covert_sample_impl(
                    pc_path,
                    gt_path,
                    config=config,
                    query_workers=args.query_workers,
                    visualize=visualize,
                    visualization_stages=visualization_stages,
                    output_dir=output_dir,
                    verbose=verbose,
                    config_source=config_source,
                    strict_gt=not visualize,
                )
            finally:
                if function_profiler is not None:
                    function_profiler.disable()
            runner_wall_seconds = time.perf_counter() - run_start

    profile_payload = _build_runtime_profile_payload(
        result=result,
        collector=collector,
        instrumentation=instrumentation,
        runner_wall_seconds=runner_wall_seconds,
        cprofile_result=_cprofile_tables(
            function_profiler, top_n=int(args.cprofile_top_n)
        ),
        query_workers=int(args.query_workers),
        optimization_audit=optimization_audit,
    )

    if args.reference_profile is not None:
        profile_payload["reference_verification"] = _verify_reference_profile(
            profile_payload, args.reference_profile
        )

    profile_dir_value = args.runtime_profile_dir
    if profile_dir_value is None and (args.cprofile or args.reference_profile):
        profile_dir_value = Path(RUNTIME_PROFILE_OUTPUT_DIR)
    json_path: Path | None = None
    markdown_path: Path | None = None
    if profile_dir_value is not None:
        profile_dir = Path(profile_dir_value)
        profile_dir.mkdir(parents=True, exist_ok=True)
        json_path = profile_dir / "runtime_profile.json"
        markdown_path = profile_dir / "runtime_profile.md"
        json_path.write_text(
            json.dumps(
                profile_payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=str,
            ),
            encoding="utf-8",
        )
        markdown_path.write_text(
            _runtime_profile_markdown(profile_payload), encoding="utf-8"
        )

    summary = {
        "sample_id": result.sample_id,
        "config_hash": result.config_hash,
        "config_source": result.config_source,
        "final_mask_sha256": profile_payload["identity"]["final_mask_sha256"],
        "predicted_positive_points": int(np.sum(result.final_mask)),
        "metrics": dict(result.metrics),
        "timings": dataclasses.asdict(result.timings),
        "cprofile_enabled": bool(args.cprofile),
        "optimization": profile_payload["optimization"],
        "reference_verification": profile_payload.get("reference_verification"),
    }
    if json_path is not None and markdown_path is not None:
        summary["runtime_profile_json"] = str(json_path)
        summary["runtime_profile_markdown"] = str(markdown_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
