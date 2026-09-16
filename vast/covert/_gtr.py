"""Private production runtime mechanically migrated from the frozen reference.

Only symbols reachable from the active consolidated call root are retained.
"""
from __future__ import annotations
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from vast.graph.mutual_knn import build_mutual_knn_graph
from vast.restoration.laplacian_inpaint import build_restoration_adjacency, build_weighted_laplacian, inpaint_global_dirichlet
GTR_NUMERICAL_IMPLEMENTATION_CONSTANTS = {'heatmap_p_low': 1.0, 'heatmap_p_high': 99.0, 'point_displacement_threshold': 0.14, 'point_displacement_adaptive_k': 3, 'point_displacement_adaptive_alpha': 0.25, 'point_displacement_adaptive_min': 0.08, 'point_displacement_adaptive_max': 0.25}
EXP17_PROFILE_INCLUDE_SUBSTAGES = True
EXP17_TOP_LEVEL_STAGE_NAMES = ('03_exp14_tscb_sgcr_total', '04_exp17_gtr_total', '05_metrics', '06_visualization', '07_save_outputs')
EXP17_PARENT_TOTAL_STAGE_NAMES = ('03_exp14_tscb_sgcr_total', '04_exp17_gtr_total', '04_06_gtr_restoration_total', '01_load_and_preprocess', '03_01_exp14_preprocess_or_input_prepare', '03_topology_base_views_targets_suppression')
EXP17_GTR_KEEP_ORIGINAL_MASK = True
EXP17_GTR_BLOCK_OTHER_ORIGINAL_MASKS = True
EXP17_GTR_POST_CLOSING_EXPAND_HOPS = 0
EXP17_GTR_POST_CLOSING_EXPAND_BLOCK_OTHER_ORIGINAL_MASKS = True
EXP17_GTR_POST_CLOSING_EXPAND_OVERLAP_MODE = 'nearest_closed_graph_distance'
EXP17_GTR_USE_POINT_DISPLACEMENT_FILTER = True
EXP17_GTR_POINT_DISPLACEMENT_THRESHOLD = 0.14
EXP17_GTR_POINT_DISPLACEMENT_KEEP_EQUAL = True
EXP17_GTR_POINT_DISPLACEMENT_THRESHOLD_MODE = 'ds_quadrant_remove_lowD_lowS'
EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_K = 3
EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_ALPHA = 0.25
EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_USE_CLAMP = True
EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_MIN = 0.08
EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_MAX = 0.25
EXP17_GTR_BIHARMONIC_REG_EPS = 1e-08
EXP17_GTR_BIHARMONIC_SOLVER = 'spsolve'
EXP17_GTR_BIHARMONIC_FALLBACK_TO_LAPLACIAN = True
EXP17_GTR_USE_HKS_OPEN_BOUNDARY_AS_RESTORATION_SUPPORT = True
EXP17_GTR_HKS_OPEN_BOUNDARY_SUPPORT_USE_EXPANDED = False
EXP17_GTR_EXCLUDE_OPEN_BOUNDARY_SUPPORT_FROM_UNKNOWN = True
HEATMAP_P_LOW = 1.0
HEATMAP_P_HIGH = 99.0
EXP17_GTR_DS_QUADRANT_ENABLE = True
EXP17_GTR_DS_QUADRANT_DISPLACEMENT_NORM_THRESHOLD = 0.5
EXP17_GTR_DS_QUADRANT_STRESS_NORM_THRESHOLD = 0.5
EXP17_GTR_DS_QUADRANT_SCOPE = 'all_points'
EXP17_GTR_DS_QUADRANT_KEEP_EQUAL = True
EXP17_GTR_SEED_CONSISTENCY_VALID_PERCENT = 55.0
EXP17_LOCAL_NORMAL_U_CLEANUP_DEFAULT_MODE = 'pre_component_classification'
EXP17_LOCAL_NORMAL_U_CLEANUP_ALLOWED_MODES = frozenset({'final_mask_only', 'pre_component_classification'})
EXP17_GTR_SEED_CONSISTENCY_REBUILD_SEED_COMPONENTS_IF_NEEDED = True
EXP17_GTR_SEED_CONSISTENCY_REMOVE_MODE = 'candidate_label_aggregate_seed_valid_ratio'
EXP17_GTR_LOCAL_NORMAL_REFERENCE_MAX_HOPS = 3
EXP17_GTR_LOCAL_NORMAL_REFERENCE_EPS = 1e-12

class RuntimeProfiler:
    """Lightweight nested-stage wall/CPU timer for exp17 runtime profiling."""

    def __init__(self, enabled: bool=True):
        self.enabled = bool(enabled)
        self.records: list[dict[str, Any]] = []
        self.stack: list[str] = []
        self.pipeline_wall_time_sec: float | None = None
        self.pipeline_cpu_time_sec: float | None = None

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        self.stack.append(str(name))
        try:
            yield
        finally:
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            self.stack.pop()
            self.records.append({'name': str(name), 'wall_time_sec': float(end_wall - start_wall), 'cpu_time_sec': float(end_cpu - start_cpu), 'meta': {str(k): v for (k, v) in meta.items()}})

    def begin_stage(self, name: str, **meta: Any) -> dict[str, Any] | None:
        """Manual stage start for large blocks that should not be re-indented."""
        if not self.enabled:
            return None
        token = {'name': str(name), 'meta': {str(k): v for (k, v) in meta.items()}, 'start_wall': time.perf_counter(), 'start_cpu': time.process_time()}
        self.stack.append(str(name))
        return token

    def end_stage(self, token: dict[str, Any] | None) -> None:
        if token is None or not self.enabled:
            return
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        if self.stack and self.stack[-1] == token['name']:
            self.stack.pop()
        self.records.append({'name': str(token['name']), 'wall_time_sec': float(end_wall - float(token['start_wall'])), 'cpu_time_sec': float(end_cpu - float(token['start_cpu'])), 'meta': dict(token.get('meta') or {})})

    def summary(self) -> dict[str, Any]:
        total_wall = sum((r['wall_time_sec'] for r in self.records))
        total_cpu = sum((r['cpu_time_sec'] for r in self.records))
        by_name: dict[str, dict[str, Any]] = {}
        for r in self.records:
            name = r['name']
            if name not in by_name:
                by_name[name] = {'name': name, 'wall_time_sec': 0.0, 'cpu_time_sec': 0.0, 'count': 0}
            by_name[name]['wall_time_sec'] += float(r['wall_time_sec'])
            by_name[name]['cpu_time_sec'] += float(r['cpu_time_sec'])
            by_name[name]['count'] += 1
        table = sorted(by_name.values(), key=lambda x: x['wall_time_sec'], reverse=True)
        ratio_base = float(self.pipeline_wall_time_sec) if self.pipeline_wall_time_sec is not None and self.pipeline_wall_time_sec > 0 else float(total_wall)
        for item in table:
            item['wall_time_ratio'] = float(item['wall_time_sec'] / ratio_base) if ratio_base > 0 else 0.0
        top_level_names = set(EXP17_TOP_LEVEL_STAGE_NAMES)
        parent_names = set(EXP17_PARENT_TOTAL_STAGE_NAMES) | top_level_names
        top_level_table = sorted([dict(item) for item in table if item['name'] in top_level_names], key=lambda x: x['wall_time_sec'], reverse=True)
        leaf_table = sorted([dict(item) for item in table if item['name'] not in parent_names and (not str(item['name']).endswith('_total'))], key=lambda x: x['wall_time_sec'], reverse=True)
        return {'total_recorded_wall_time_sec': float(total_wall), 'total_recorded_cpu_time_sec': float(total_cpu), 'pipeline_wall_time_sec': float(self.pipeline_wall_time_sec) if self.pipeline_wall_time_sec is not None else None, 'pipeline_cpu_time_sec': float(self.pipeline_cpu_time_sec) if self.pipeline_cpu_time_sec is not None else None, 'stages_sorted_by_wall_time': table, 'top_level_stages_sorted_by_wall_time': top_level_table, 'leaf_or_substages_sorted_by_wall_time': leaf_table, 'raw_records': self.records}

def _resolve_profiler(profiler: RuntimeProfiler | None) -> RuntimeProfiler:
    if profiler is None:
        return RuntimeProfiler(enabled=False)
    return profiler

def _stage_ctx(profiler: RuntimeProfiler, name: str, *, include: bool=True, **meta: Any):
    if include:
        return profiler.stage(name, **meta)
    return nullcontext()

def _begin_stage(profiler: RuntimeProfiler, name: str, *, include: bool=True, **meta: Any) -> dict[str, Any] | None:
    if not include:
        return None
    return profiler.begin_stage(name, **meta)

def _end_stage(profiler: RuntimeProfiler, token: dict[str, Any] | None) -> None:
    profiler.end_stage(token)

def _percentile_normalize(values: np.ndarray, p_low: float=1.0, p_high: float=99.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values.astype(np.float64, copy=False)
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo + 1e-12), 0.0, 1.0)

def build_covert_gtr_params(config: Any) -> dict[str, Any]:
    """Resolve every active GTR decision setting from CovertPipelineConfig."""
    g = config.gtr
    return {'dilation_hops': int(g.dilation_hops), 'erosion_hops': int(g.erosion_hops), 'keep_original_mask': bool(g.keep_original_mask), 'block_other_original_masks': bool(g.block_other_original_masks), 'closing_mode': str(g.closing_mode), 'closing_overlap_resolution': str(g.closing_overlap_resolution), 'min_closed_component_size': int(g.minimum_closed_component_size), 'post_closing_expand_hops': int(g.post_closing_expand_hops), 'post_closing_expand_block_other_original_masks': True, 'post_closing_expand_overlap_mode': 'nearest_closed_graph_distance', 'restoration_weight': str(g.restoration_weight), 'restoration_mode': str(g.restoration_mode), 'biharmonic_reg_eps': float(g.biharmonic_regularization_epsilon), 'biharmonic_solver': str(g.biharmonic_solver), 'biharmonic_fallback_to_laplacian': bool(g.biharmonic_fallback_to_laplacian), 'restore_per_component': bool(g.restore_per_component), 'min_anchor_points': int(g.minimum_anchor_points), 'anchor_hops': int(g.anchor_hops), 'displacement_threshold': float(g.displacement_threshold), 'use_point_displacement_filter': bool(config.switches.gtr_point_refinement), 'point_displacement_threshold_mode': str(g.point_filter_mode), 'point_displacement_threshold': 0.14, 'point_displacement_adaptive_k': 3, 'point_displacement_adaptive_alpha': 0.25, 'point_displacement_adaptive_use_clamp': True, 'point_displacement_adaptive_min': 0.08, 'point_displacement_adaptive_max': 0.25, 'point_displacement_keep_equal': True, 'point_filter_before_support_restitution': bool(g.point_filter_before_support_restitution), 'use_stress_veto': bool(g.stress_veto_enabled), 'stress_threshold': float(g.stress_threshold), 'use_hks_open_boundary_as_restoration_support': bool(g.use_hks_open_boundary_as_restoration_support), 'hks_open_boundary_support_use_expanded': bool(g.hks_open_boundary_support_use_expanded), 'hks_open_boundary_support_expand_hops': int(g.boundary_support_expand_hops), 'exclude_open_boundary_support_from_unknown': bool(g.exclude_boundary_support_from_unknown), 'exclude_open_boundary_support_from_decision': bool(g.exclude_boundary_support_from_decision), 'do_not_restitute_open_boundary_support': not bool(g.restitute_boundary_support), 'ds_quadrant_enabled': bool(g.ds_quadrant_enabled), 'ds_scope': str(g.ds_scope), 'ds_displacement_normalized_threshold': float(g.ds_displacement_normalized_threshold), 'ds_stress_normalized_threshold': float(g.ds_stress_normalized_threshold), 'ds_keep_equal': bool(g.ds_keep_equal), 'candidate_classifier_mode': str(g.candidate_classifier_mode), 'local_normal_reference_enabled': bool(g.local_normal_reference_enabled), 'local_normal_reference_max_hops': int(g.local_normal_reference_max_hops), 'local_normal_reference_epsilon': float(g.local_normal_reference_epsilon), 'adaptive_relative_normal_max_ratio': float(g.adaptive_relative_normal_max_ratio), 'adaptive_min_large_fraction': float(g.adaptive_min_large_fraction), 'adaptive_max_shallow_displacement_normalized': float(g.adaptive_max_shallow_displacement_normalized), 'param_source': 'covert_config', 'exp06_override_applied': False, 'exp06_override_fields': []}

def _expand_hks_open_boundary_support_mask(base_support_mask: np.ndarray, neighbor_indices: list[np.ndarray], expand_hops: int) -> tuple[np.ndarray, dict[str, Any]]:
    base_support_mask = np.asarray(base_support_mask, dtype=bool).reshape(-1)
    expand_hops = int(expand_hops)
    if expand_hops <= 0 or not np.any(base_support_mask):
        return (base_support_mask.copy(), {'expand_hops': expand_hops, 'base_point_count': int(np.sum(base_support_mask)), 'expanded_point_count': int(np.sum(base_support_mask)), 'added_point_count': 0, 'skipped': True, 'reason': 'zero_expand_hops_or_empty_base_mask'})
    (expanded_mask, _distance) = _graph_dilate_with_distance(base_support_mask, neighbor_indices, max_hops=expand_hops)
    base_point_count = int(np.sum(base_support_mask))
    expanded_point_count = int(np.sum(expanded_mask))
    return (expanded_mask, {'expand_hops': expand_hops, 'base_point_count': base_point_count, 'expanded_point_count': expanded_point_count, 'added_point_count': int(expanded_point_count - base_point_count), 'skipped': False, 'reason': None})

def _extract_labeled_components_from_labels(labels: np.ndarray) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    components: list[dict[str, Any]] = []
    for label_id in sorted((int(x) for x in np.unique(labels) if int(x) > 0)):
        indices = np.flatnonzero(labels == label_id).astype(np.int64, copy=False)
        components.append({'component_id': int(label_id), 'indices': indices, 'size': int(indices.size)})
    return components

def _connected_component_labels_from_mask(mask: np.ndarray, neighbor_indices: list[np.ndarray]) -> np.ndarray:
    """Label graph-connected components inside a binary mask (labels start at 1)."""
    mask_bool = np.asarray(mask, dtype=bool).reshape(-1)
    num_points = int(mask_bool.shape[0])
    labels = np.zeros(num_points, dtype=np.int32)
    next_label = 1
    for seed_index in np.flatnonzero(mask_bool):
        seed_index = int(seed_index)
        if labels[seed_index] != 0:
            continue
        queue: deque[int] = deque([seed_index])
        labels[seed_index] = next_label
        while queue:
            point_index = queue.popleft()
            neighbors = np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1)
            for neighbor_index in neighbors:
                neighbor_index = int(neighbor_index)
                if neighbor_index < 0 or neighbor_index >= num_points:
                    continue
                if not mask_bool[neighbor_index]:
                    continue
                if labels[neighbor_index] != 0:
                    continue
                labels[neighbor_index] = next_label
                queue.append(neighbor_index)
        next_label += 1
    return labels

def _resolve_sgcr_seed_component_labels_for_consistency(tscb_result: dict[str, Any], neighbor_indices: list[np.ndarray], num_points: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve SGCR grow-enabled seed component labels for consistency filtering."""
    num_points = int(num_points)
    empty_labels = np.zeros(num_points, dtype=np.int32)
    label_keys = ('sgcr_grow_seed_component_labels', 'sgcr_grow_seed_labels', 'sgcr_grow_enabled_seed_labels', 'sgcr_seed_component_labels', 'sgcr_seed_labels')
    for key in label_keys:
        if key not in tscb_result or tscb_result[key] is None:
            continue
        labels = np.asarray(tscb_result[key], dtype=np.int32).reshape(-1)
        if labels.shape[0] != num_points:
            continue
        num_seed_components = int(len([x for x in np.unique(labels) if int(x) > 0]))
        return (labels, {'source': key, 'rebuilt_from_mask': False, 'num_seed_components': num_seed_components, 'num_seed_points': int(np.sum(labels > 0)), 'available': True, 'reason': 'ok'})
    if not EXP17_GTR_SEED_CONSISTENCY_REBUILD_SEED_COMPONENTS_IF_NEEDED:
        return (empty_labels, {'source': None, 'rebuilt_from_mask': False, 'num_seed_components': 0, 'num_seed_points': 0, 'available': False, 'reason': 'no_seed_component_labels_or_mask_found'})
    mask_keys = ('sgcr_grow_seed_mask', 'sgcr_grow_enabled_seed_mask', 'sgcr_seed_mask')
    for key in mask_keys:
        if key not in tscb_result or tscb_result[key] is None:
            continue
        mask = np.asarray(tscb_result[key], dtype=bool).reshape(-1)
        if mask.shape[0] != num_points:
            continue
        labels = _connected_component_labels_from_mask(mask, neighbor_indices)
        num_seed_components = int(len([x for x in np.unique(labels) if int(x) > 0]))
        return (labels, {'source': key, 'rebuilt_from_mask': True, 'num_seed_components': num_seed_components, 'num_seed_points': int(np.sum(labels > 0)), 'available': True, 'reason': 'ok'})
    return (empty_labels, {'source': None, 'rebuilt_from_mask': False, 'num_seed_components': 0, 'num_seed_points': 0, 'available': False, 'reason': 'no_seed_component_labels_or_mask_found'})

def _graph_dilate_with_distance(seed_mask: np.ndarray, neighbor_indices: list[np.ndarray], max_hops: int, allowed_mask: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    num_points = seed_mask.shape[0]
    seed_mask = np.asarray(seed_mask, dtype=bool).reshape(-1)
    dilated_mask = seed_mask.copy()
    distance = np.full(num_points, -1, dtype=np.int32)
    distance[seed_mask] = 0
    if allowed_mask is None:
        allowed_mask = np.ones(num_points, dtype=bool)
    else:
        allowed_mask = np.asarray(allowed_mask, dtype=bool).reshape(-1)
    max_hops = int(max_hops)
    if max_hops <= 0:
        return (dilated_mask, distance)
    queue: deque[int] = deque((int(i) for i in np.flatnonzero(seed_mask)))
    while queue:
        point_index = queue.popleft()
        current_distance = int(distance[point_index])
        if current_distance >= max_hops:
            continue
        neighbors = np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1)
        for neighbor_index in neighbors:
            neighbor_index = int(neighbor_index)
            if not allowed_mask[neighbor_index]:
                continue
            if distance[neighbor_index] >= 0:
                continue
            distance[neighbor_index] = current_distance + 1
            dilated_mask[neighbor_index] = True
            queue.append(neighbor_index)
    return (dilated_mask, distance)

def _graph_erode_mask(mask: np.ndarray, neighbor_indices: list[np.ndarray], hops: int) -> np.ndarray:
    eroded = np.asarray(mask, dtype=bool).reshape(-1).copy()
    hops = int(hops)
    if hops <= 0:
        return eroded
    for _ in range(hops):
        if not np.any(eroded):
            break
        remove_indices: list[int] = []
        for point_index in np.flatnonzero(eroded):
            neighbors = np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1)
            if neighbors.size == 0:
                remove_indices.append(int(point_index))
                continue
            if np.any(~eroded[neighbors]):
                remove_indices.append(int(point_index))
        if not remove_indices:
            break
        eroded[np.asarray(remove_indices, dtype=np.int64)] = False
    return eroded

def _close_one_sgcr_component(component_id: int, component_indices: np.ndarray, all_original_candidate_mask: np.ndarray, neighbor_indices: list[np.ndarray], dilation_hops: int, erosion_hops: int, *, block_other_original_masks: bool=EXP17_GTR_BLOCK_OTHER_ORIGINAL_MASKS, remove_other_original_after_closing: bool=True) -> dict[str, Any]:
    num_points = all_original_candidate_mask.shape[0]
    component_mask = np.zeros(num_points, dtype=bool)
    component_mask[np.asarray(component_indices, dtype=np.int64)] = True
    other_original_mask = all_original_candidate_mask & ~component_mask
    allowed_mask = np.ones(num_points, dtype=bool)
    if block_other_original_masks:
        allowed_mask &= ~other_original_mask
    (dilated_mask, dilation_distance) = _graph_dilate_with_distance(component_mask, neighbor_indices, max_hops=dilation_hops, allowed_mask=allowed_mask)
    eroded_mask = _graph_erode_mask(dilated_mask, neighbor_indices, hops=erosion_hops)
    if EXP17_GTR_KEEP_ORIGINAL_MASK:
        closed_mask = eroded_mask | component_mask
    else:
        closed_mask = eroded_mask.copy()
    if remove_other_original_after_closing:
        closed_mask &= ~other_original_mask
    closed_mask |= component_mask
    return {'component_id': int(component_id), 'original_mask': component_mask, 'dilated_mask': dilated_mask, 'eroded_mask': eroded_mask, 'closed_mask_pre_overlap': closed_mask, 'dilation_distance': dilation_distance, 'summary': {'original_size': int(np.sum(component_mask)), 'dilated_size': int(np.sum(dilated_mask)), 'eroded_size': int(np.sum(eroded_mask)), 'closed_size_pre_overlap': int(np.sum(closed_mask)), 'dilation_hops': int(dilation_hops), 'erosion_hops': int(erosion_hops), 'block_other_original_masks': bool(block_other_original_masks), 'remove_other_original_after_closing': bool(remove_other_original_after_closing)}}

def _resolve_sv_values_for_closing(tscb_result: dict[str, Any], num_points: int) -> tuple[np.ndarray, str]:
    """Resolve per-point SV values used for closing overlap priority."""
    candidates: list[tuple[str, Any]] = [('c_abs_enhanced', tscb_result.get('c_abs_enhanced')), ('semantic_sv', tscb_result.get('semantic_sv')), ('sgcr_candidate_score', tscb_result.get('sgcr_candidate_score')), ('suppressed_intersection_min', tscb_result.get('suppressed_intersection_min')), ('c_abs_raw', tscb_result.get('c_abs_raw'))]
    for (source_name, candidate) in candidates:
        if candidate is None:
            continue
        values = np.asarray(candidate, dtype=np.float64).reshape(-1)
        if values.shape[0] != num_points:
            continue
        return (np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), str(source_name))
    return (np.zeros(num_points, dtype=np.float64), 'zeros')

def _compute_closing_component_sv_priority(*, sgcr_candidate_labels: np.ndarray, seed_component_labels: np.ndarray, sv_values: np.ndarray, sv_source: str='unknown') -> dict[str, Any]:
    """Compute per-SGCR-candidate closing priority from contributing seed SV means."""
    sgcr_candidate_labels = np.asarray(sgcr_candidate_labels, dtype=np.int32).reshape(-1)
    seed_component_labels = np.asarray(seed_component_labels, dtype=np.int32).reshape(-1)
    sv_values = np.asarray(sv_values, dtype=np.float64).reshape(-1)
    num_points = int(sgcr_candidate_labels.shape[0])
    if seed_component_labels.shape[0] != num_points or sv_values.shape[0] != num_points:
        raise ValueError('sgcr_candidate_labels, seed_component_labels, and sv_values must share length.')
    candidate_ids = sorted((int(x) for x in np.unique(sgcr_candidate_labels) if int(x) > 0))
    seed_ids = sorted((int(x) for x in np.unique(seed_component_labels) if int(x) > 0))
    seed_sv_means: dict[int, float] = {}
    seed_to_candidate_ids: dict[int, set[int]] = {}
    for seed_component_id in seed_ids:
        seed_indices = np.flatnonzero(seed_component_labels == seed_component_id)
        if seed_indices.size == 0:
            continue
        seed_sv_means[seed_component_id] = float(np.mean(sv_values[seed_indices]))
        overlapped = {int(x) for x in np.unique(sgcr_candidate_labels[seed_indices]) if int(x) > 0}
        seed_to_candidate_ids[seed_component_id] = overlapped
    component_priority: dict[int, float] = {}
    component_priority_source: dict[int, str] = {}
    component_priority_seed_component_id: dict[int, int | None] = {}
    component_priority_reports: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_mask = sgcr_candidate_labels == candidate_id
        num_candidate_points = int(np.sum(candidate_mask))
        contributing_seed_component_ids = sorted((seed_id for (seed_id, overlapped_candidates) in seed_to_candidate_ids.items() if candidate_id in overlapped_candidates))
        if contributing_seed_component_ids:
            best_seed_id = max(contributing_seed_component_ids, key=lambda seed_id: (float(seed_sv_means.get(seed_id, 0.0)), -int(seed_id)))
            priority_score = float(seed_sv_means.get(best_seed_id, 0.0))
            priority_source = 'max_contributing_seed_component_sv_mean'
            priority_seed_component_id: int | None = int(best_seed_id)
        elif num_candidate_points > 0:
            priority_score = float(np.mean(sv_values[candidate_mask]))
            priority_source = 'candidate_component_sv_mean_fallback'
            priority_seed_component_id = None
        else:
            priority_score = 0.0
            priority_source = 'candidate_component_sv_mean_fallback'
            priority_seed_component_id = None
        component_priority[int(candidate_id)] = float(priority_score)
        component_priority_source[int(candidate_id)] = str(priority_source)
        component_priority_seed_component_id[int(candidate_id)] = priority_seed_component_id
        component_priority_reports.append({'component_id': int(candidate_id), 'priority_score': float(priority_score), 'priority_source': str(priority_source), 'priority_seed_component_id': int(priority_seed_component_id) if priority_seed_component_id is not None else None, 'num_candidate_points': int(num_candidate_points), 'num_contributing_seed_components': int(len(contributing_seed_component_ids)), 'contributing_seed_component_ids': [int(x) for x in contributing_seed_component_ids]})
    return {'component_priority': component_priority, 'component_priority_source': component_priority_source, 'component_priority_seed_component_id': component_priority_seed_component_id, 'component_priority_reports': component_priority_reports, 'sv_source': str(sv_source)}

def _resolve_closed_mask_overlaps(closing_results: list[dict[str, Any]], num_points: int, *, component_priority: dict[int, float] | None=None, overlap_resolution: str='nearest_original_graph_distance') -> dict[str, Any]:
    overlap_resolution = str(overlap_resolution).strip().lower()
    owner_labels = np.zeros(num_points, dtype=np.int32)
    for result in closing_results:
        component_id = int(result['component_id'])
        owner_labels[np.asarray(result['original_mask'], dtype=bool)] = component_id
    num_overlap_points = 0
    if overlap_resolution == 'nearest_original_graph_distance':
        for point_index in range(num_points):
            if owner_labels[point_index] > 0:
                continue
            covering: list[tuple[int, int]] = []
            for result in closing_results:
                if not bool(result['closed_mask_pre_overlap'][point_index]):
                    continue
                component_id = int(result['component_id'])
                dist = int(result['dilation_distance'][point_index])
                if dist < 0:
                    dist = 10 ** 9
                covering.append((dist, component_id))
            if not covering:
                continue
            if len(covering) == 1:
                owner_labels[point_index] = covering[0][1]
                continue
            num_overlap_points += 1
            covering.sort(key=lambda item: (item[0], item[1]))
            owner_labels[point_index] = covering[0][1]
        return {'closed_labels': owner_labels, 'closed_mask': owner_labels > 0, 'summary': {'num_components': len(closing_results), 'num_overlap_points': int(num_overlap_points), 'overlap_resolution': 'nearest_original_graph_distance', 'tie_break_rule': 'shorter_dilation_distance_then_smaller_component_id'}}
    if overlap_resolution == 'highest_seed_sv_mean':
        if component_priority is None:
            raise ValueError("component_priority is required when overlap_resolution='highest_seed_sv_mean'")
        for point_index in range(num_points):
            if owner_labels[point_index] > 0:
                continue
            covering_sv: list[tuple[float, int, int]] = []
            for result in closing_results:
                if not bool(result['closed_mask_pre_overlap'][point_index]):
                    continue
                component_id = int(result['component_id'])
                priority = float(component_priority.get(component_id, 0.0))
                dist = int(result['dilation_distance'][point_index])
                if dist < 0:
                    dist = 10 ** 9
                covering_sv.append((-priority, dist, component_id))
            if not covering_sv:
                continue
            if len(covering_sv) == 1:
                owner_labels[point_index] = covering_sv[0][2]
                continue
            num_overlap_points += 1
            covering_sv.sort()
            owner_labels[point_index] = covering_sv[0][2]
        return {'closed_labels': owner_labels, 'closed_mask': owner_labels > 0, 'summary': {'num_components': len(closing_results), 'num_overlap_points': int(num_overlap_points), 'overlap_resolution': 'highest_seed_sv_mean', 'tie_break_rule': 'higher_sv_priority_then_shorter_dilation_distance_then_smaller_component_id', 'component_priority_source': 'max_contributing_seed_component_sv_mean'}}
    raise ValueError(f"Unsupported overlap_resolution={overlap_resolution!r}; expected 'nearest_original_graph_distance' or 'highest_seed_sv_mean'.")

def _post_expand_closed_labels_after_closing(closed_labels_core: np.ndarray, original_candidate_labels: np.ndarray, neighbor_indices: list[np.ndarray], *, expand_hops: int, block_other_original_masks: bool=True, overlap_mode: str='nearest_closed_graph_distance') -> dict[str, Any]:
    """Expand closed component labels after morphology closing.

    Existing closed core labels are never overwritten.
    Expansion only assigns currently-background points.
    If multiple components claim the same point, assign the one with the
    nearest graph distance from its closed core; ties are resolved by smaller
    component_id.
    """
    closed_labels_core = np.asarray(closed_labels_core, dtype=np.int32).reshape(-1)
    original_candidate_labels = np.asarray(original_candidate_labels, dtype=np.int32).reshape(-1)
    num_points = int(closed_labels_core.shape[0])
    expand_hops = int(expand_hops)
    overlap_mode = str(overlap_mode).strip().lower()
    if original_candidate_labels.shape[0] != num_points:
        raise ValueError('original_candidate_labels and closed_labels_core must have the same length.')
    if overlap_mode != 'nearest_closed_graph_distance':
        raise ValueError("EXP17_GTR_POST_CLOSING_EXPAND_OVERLAP_MODE must be 'nearest_closed_graph_distance' for now.")
    component_ids = sorted((int(x) for x in np.unique(closed_labels_core) if int(x) > 0))
    num_components = len(component_ids)
    if expand_hops <= 0 or num_components == 0:
        return {'closed_labels': closed_labels_core.copy(), 'closed_mask': closed_labels_core > 0, 'post_expand_distance': np.full(num_points, -1, dtype=np.int32), 'summary': {'enabled': False, 'expand_hops': expand_hops, 'skipped': True, 'reason': 'zero_expand_hops_or_empty_closed_labels', 'num_core_points': int(np.sum(closed_labels_core > 0)), 'num_expanded_points': int(np.sum(closed_labels_core > 0)), 'num_added_points': 0, 'num_components': int(num_components), 'num_conflict_points': 0, 'overlap_mode': overlap_mode, 'block_other_original_masks': bool(block_other_original_masks), 'component_reports': []}}
    expanded_labels = closed_labels_core.copy()
    owner_distance = np.full(num_points, -1, dtype=np.int32)
    proposal_owner = np.zeros(num_points, dtype=np.int32)
    proposal_distance = np.full(num_points, 10 ** 9, dtype=np.int32)
    num_conflict_points = 0
    for component_id in component_ids:
        seed_mask = closed_labels_core == component_id
        allowed_mask = np.ones(num_points, dtype=bool)
        if block_other_original_masks:
            other_original_candidate_mask = (original_candidate_labels > 0) & (original_candidate_labels != component_id)
            allowed_mask &= ~other_original_candidate_mask
        (dilated_mask, distance) = _graph_dilate_with_distance(seed_mask, neighbor_indices, max_hops=expand_hops, allowed_mask=allowed_mask)
        candidate_mask = dilated_mask & (closed_labels_core == 0) & (distance > 0)
        for point_index in np.flatnonzero(candidate_mask):
            point_index = int(point_index)
            dist = int(distance[point_index])
            old_dist = int(proposal_distance[point_index])
            old_owner = int(proposal_owner[point_index])
            if old_owner > 0:
                num_conflict_points += 1
            if dist < old_dist or (dist == old_dist and component_id < old_owner):
                proposal_distance[point_index] = dist
                proposal_owner[point_index] = component_id
    assigned_mask = proposal_owner > 0
    expanded_labels[assigned_mask] = proposal_owner[assigned_mask]
    owner_distance[closed_labels_core > 0] = 0
    owner_distance[assigned_mask] = proposal_distance[assigned_mask].astype(np.int32, copy=False)
    component_reports: list[dict[str, Any]] = []
    for component_id in component_ids:
        component_reports.append({'component_id': int(component_id), 'core_size': int(np.sum(closed_labels_core == component_id)), 'expanded_size': int(np.sum(expanded_labels == component_id)), 'added_size': int(np.sum((expanded_labels == component_id) & (closed_labels_core == 0)))})
    summary: dict[str, Any] = {'enabled': True, 'expand_hops': expand_hops, 'skipped': False, 'reason': None, 'num_core_points': int(np.sum(closed_labels_core > 0)), 'num_expanded_points': int(np.sum(expanded_labels > 0)), 'num_added_points': int(np.sum((expanded_labels > 0) & (closed_labels_core == 0))), 'num_components': int(num_components), 'num_conflict_points': int(num_conflict_points), 'overlap_mode': overlap_mode, 'block_other_original_masks': bool(block_other_original_masks), 'component_reports': component_reports}
    return {'closed_labels': expanded_labels, 'closed_mask': expanded_labels > 0, 'post_expand_distance': owner_distance, 'summary': summary}

def _build_restoration_laplacian(adjacency_dist: sp.csr_matrix, weight_mode: str) -> sp.csr_matrix:
    adjacency_csr = sp.csr_matrix(adjacency_dist).tocsr()
    if weight_mode.strip().lower() == 'inverse_distance':
        restoration_adj = build_restoration_adjacency(adjacency_csr, weight_mode='inverse_distance')
    else:
        restoration_adj = adjacency_csr.copy()
        restoration_adj.data = np.ones_like(restoration_adj.data, dtype=np.float64)
        restoration_adj.eliminate_zeros()
    return build_weighted_laplacian(restoration_adj)

def _laplacian_restore_points(points: np.ndarray, restoration_laplacian: sp.csr_matrix, unknown_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    unknown_mask = np.asarray(unknown_mask, dtype=bool).reshape(-1)
    if not np.any(unknown_mask):
        return (points.copy(), {'num_unknown_points': 0, 'num_fixed_points': int(points.shape[0]), 'skipped': True, 'reason': 'empty_unknown_mask_after_hks_open_boundary_support'})
    fixed_mask = ~unknown_mask
    (restored_points, solver_status) = inpaint_global_dirichlet(points=points, laplacian=restoration_laplacian, unknown_mask=unknown_mask, fixed_mask=fixed_mask)
    solver_summary: dict[str, Any] = {'num_unknown_points': int(np.sum(unknown_mask)), 'num_fixed_points': int(np.sum(fixed_mask)), **{str(k): v for (k, v) in solver_status.items()}}
    return (restored_points, solver_summary)

def _biharmonic_restore_points(points: np.ndarray, restoration_laplacian: sp.csr_matrix, unknown_mask: np.ndarray, *, reg_eps: float=EXP17_GTR_BIHARMONIC_REG_EPS, solver: str=EXP17_GTR_BIHARMONIC_SOLVER, fallback_to_laplacian: bool=EXP17_GTR_BIHARMONIC_FALLBACK_TO_LAPLACIAN) -> tuple[np.ndarray, dict[str, Any]]:
    """Biharmonic Dirichlet inpainting: minimize ||L X||^2 on unknown points."""
    points = np.asarray(points, dtype=np.float64)
    unknown_mask = np.asarray(unknown_mask, dtype=bool).reshape(-1)
    num_points = int(points.shape[0])
    if not np.any(unknown_mask):
        return (points.copy(), {'method': 'biharmonic_laplacian_dirichlet', 'num_unknown_points': 0, 'num_fixed_points': num_points, 'skipped': True, 'reason': 'empty_unknown_mask'})
    fixed_mask = ~unknown_mask
    if not np.any(fixed_mask):
        return (points.copy(), {'method': 'biharmonic_laplacian_dirichlet', 'num_unknown_points': int(np.sum(unknown_mask)), 'num_fixed_points': 0, 'skipped': True, 'reason': 'empty_fixed_mask', 'fallback_used': False})
    if str(solver).strip().lower() != 'spsolve':
        raise ValueError("Only EXP17_GTR_BIHARMONIC_SOLVER='spsolve' is supported for now.")
    try:
        L = sp.csr_matrix(restoration_laplacian).astype(np.float64).tocsr()
        B = (L.T @ L).tocsr()
        unknown_indices = np.flatnonzero(unknown_mask).astype(np.int64)
        fixed_indices = np.flatnonzero(~unknown_mask).astype(np.int64)
        B_uu = B[unknown_indices, :][:, unknown_indices].tocsr()
        B_uf = B[unknown_indices, :][:, fixed_indices].tocsr()
        if reg_eps > 0:
            B_uu = B_uu + float(reg_eps) * sp.eye(B_uu.shape[0], dtype=np.float64, format='csr')
        rhs = -B_uf @ points[fixed_indices]
        restored_points = points.copy()
        solution = np.zeros((unknown_indices.size, 3), dtype=np.float64)
        for axis in range(3):
            solution[:, axis] = spla.spsolve(B_uu, rhs[:, axis])
        if not np.all(np.isfinite(solution)):
            raise RuntimeError('non-finite biharmonic solution')
        restored_points[unknown_indices] = solution
        summary: dict[str, Any] = {'method': 'biharmonic_laplacian_dirichlet', 'skipped': False, 'failed': False, 'fallback_used': False, 'num_points': num_points, 'num_unknown_points': int(unknown_indices.size), 'num_fixed_points': int(fixed_indices.size), 'laplacian_nnz': int(L.nnz), 'biharmonic_nnz': int(B.nnz), 'system_shape': [int(B_uu.shape[0]), int(B_uu.shape[1])], 'system_nnz': int(B_uu.nnz), 'reg_eps': float(reg_eps), 'solver': str(solver)}
        return (restored_points, summary)
    except Exception as exc:
        if not fallback_to_laplacian:
            raise
        (fallback_points, fallback_summary) = _laplacian_restore_points(points=points, restoration_laplacian=restoration_laplacian, unknown_mask=unknown_mask)
        summary = {'method': 'biharmonic_laplacian_dirichlet', 'failed': True, 'fallback_used': True, 'fallback_method': 'laplacian', 'failure_reason': str(exc), 'fallback_summary': fallback_summary}
        return (fallback_points, summary)

def _restore_points_by_mode(points: np.ndarray, restoration_laplacian: sp.csr_matrix, unknown_mask: np.ndarray, gtr_params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch GTR restoration by EXP17_GTR_RESTORATION_MODE."""
    mode = str(gtr_params['restoration_mode']).strip().lower()
    if mode == 'laplacian':
        (restored_points, summary) = _laplacian_restore_points(points=points, restoration_laplacian=restoration_laplacian, unknown_mask=unknown_mask)
        summary = dict(summary)
        summary['restoration_mode'] = 'laplacian'
        return (restored_points, summary)
    if mode == 'biharmonic':
        (restored_points, summary) = _biharmonic_restore_points(points=points, restoration_laplacian=restoration_laplacian, unknown_mask=unknown_mask, reg_eps=float(gtr_params.get('biharmonic_reg_eps', EXP17_GTR_BIHARMONIC_REG_EPS)), solver=str(gtr_params.get('biharmonic_solver', EXP17_GTR_BIHARMONIC_SOLVER)), fallback_to_laplacian=bool(gtr_params.get('biharmonic_fallback_to_laplacian', EXP17_GTR_BIHARMONIC_FALLBACK_TO_LAPLACIAN)))
        summary = dict(summary)
        summary['restoration_mode'] = 'biharmonic'
        return (restored_points, summary)
    raise ValueError(f"Unsupported EXP17_GTR_RESTORATION_MODE={mode!r}; expected 'laplacian' or 'biharmonic'.")

def _restore_points_per_closed_component(*, points: np.ndarray, restoration_laplacian: sp.csr_matrix, closed_labels: np.ndarray, hks_open_boundary_support_mask: np.ndarray, gtr_params: dict[str, Any], profiler: RuntimeProfiler | None=None) -> dict[str, Any]:
    """Restore each closed component independently on the full graph.

    Other components remain fixed while restoring the current component unknown
    region, so nearby candidates do not form one global hole.
    """
    profiler = _resolve_profiler(profiler)
    include_substages = bool(EXP17_PROFILE_INCLUDE_SUBSTAGES)
    points = np.asarray(points, dtype=np.float64)
    closed_labels = np.asarray(closed_labels, dtype=np.int32).reshape(-1)
    hks_open_boundary_support_mask = np.asarray(hks_open_boundary_support_mask, dtype=bool).reshape(-1)
    num_points = int(points.shape[0])
    if closed_labels.shape[0] != num_points:
        raise ValueError('closed_labels length mismatch with points.')
    if hks_open_boundary_support_mask.shape[0] != num_points:
        raise ValueError('hks_open_boundary_support_mask length mismatch with points.')
    closed_mask = closed_labels > 0
    combined_restored_points = points.copy()
    restoration_unknown_mask = np.zeros(num_points, dtype=bool)
    component_restored_labels = np.zeros(num_points, dtype=np.int32)
    support_inside_closed_mask = closed_mask & hks_open_boundary_support_mask
    support_inside_closed_labels = np.where(support_inside_closed_mask, closed_labels, 0).astype(np.int32, copy=False)
    restoration_mode = str(gtr_params['restoration_mode']).strip().lower()
    component_ids = sorted((int(x) for x in np.unique(closed_labels) if int(x) > 0))
    component_reports: list[dict[str, Any]] = []
    num_restored_components = 0
    num_skipped_components = 0
    for component_id in component_ids:
        component_closed_mask = closed_labels == component_id
        component_support_mask = component_closed_mask & hks_open_boundary_support_mask
        if EXP17_GTR_USE_HKS_OPEN_BOUNDARY_AS_RESTORATION_SUPPORT and EXP17_GTR_EXCLUDE_OPEN_BOUNDARY_SUPPORT_FROM_UNKNOWN:
            component_unknown_mask = component_closed_mask & ~hks_open_boundary_support_mask
        else:
            component_unknown_mask = component_closed_mask.copy()
        num_closed_points = int(np.sum(component_closed_mask))
        num_unknown_points = int(np.sum(component_unknown_mask))
        num_support_points = int(np.sum(component_support_mask))
        if not np.any(component_unknown_mask):
            num_skipped_components += 1
            component_reports.append({'component_id': int(component_id), 'num_closed_points': num_closed_points, 'num_unknown_points': num_unknown_points, 'num_support_points': num_support_points, 'restoration_mode': restoration_mode, 'skipped': True, 'reason': 'empty_component_unknown_mask', 'failed': False, 'fallback_used': False, 'solve_wall_time_sec': 0.0})
            continue
        solve_start = time.perf_counter()
        with _stage_ctx(profiler, '04_06_01_gtr_restore_one_component', include=include_substages, component_id=int(component_id), num_unknown_points=int(num_unknown_points), num_support_points=int(num_support_points), restoration_mode=str(restoration_mode)):
            (restored_points_component, solver_summary_component) = _restore_points_by_mode(points=points, restoration_laplacian=restoration_laplacian, unknown_mask=component_unknown_mask, gtr_params=gtr_params)
        solve_wall_time_sec = float(time.perf_counter() - solve_start)
        combined_restored_points[component_unknown_mask] = restored_points_component[component_unknown_mask]
        restoration_unknown_mask[component_unknown_mask] = True
        component_restored_labels[component_unknown_mask] = int(component_id)
        num_restored_components += 1
        component_reports.append({'component_id': int(component_id), 'num_closed_points': num_closed_points, 'num_unknown_points': num_unknown_points, 'num_support_points': num_support_points, 'restoration_mode': str(solver_summary_component.get('restoration_mode', restoration_mode)), 'skipped': False, 'reason': 'ok', 'failed': bool(solver_summary_component.get('failed', False)), 'fallback_used': bool(solver_summary_component.get('fallback_used', False)), 'solver_summary': dict(solver_summary_component), 'solve_wall_time_sec': solve_wall_time_sec})
    solver_summary = {'method': f'per_component_{restoration_mode}', 'restoration_mode': restoration_mode, 'restore_per_component': True, 'solver_scope': 'full_graph_per_component', 'num_components': int(len(component_ids)), 'num_restored_components': int(num_restored_components), 'num_skipped_components': int(num_skipped_components), 'num_closed_points': int(np.sum(closed_mask)), 'num_restoration_unknown_points': int(np.sum(restoration_unknown_mask)), 'num_support_inside_closed_points': int(np.sum(support_inside_closed_mask)), 'component_reports': component_reports}
    return {'restored_points': combined_restored_points, 'restoration_unknown_mask': restoration_unknown_mask, 'support_inside_closed_mask': support_inside_closed_mask, 'support_inside_closed_labels': support_inside_closed_labels, 'component_restored_labels': component_restored_labels, 'component_reports': component_reports, 'solver_summary': solver_summary}

def _compute_displacement_and_stress(points: np.ndarray, restored_points: np.ndarray, unknown_mask: np.ndarray, neighbor_indices: list[np.ndarray]) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    restored_points = np.asarray(restored_points, dtype=np.float64)
    unknown_mask = np.asarray(unknown_mask, dtype=bool).reshape(-1)
    displacement_magnitude = np.linalg.norm(points - restored_points, axis=1)
    stress_proxy = np.zeros(points.shape[0], dtype=np.float64)
    for point_index in np.flatnonzero(unknown_mask):
        neighbors = np.asarray(neighbor_indices[point_index], dtype=np.int64).reshape(-1)
        if neighbors.size == 0:
            continue
        valid_neighbors = neighbors[unknown_mask[neighbors]]
        if valid_neighbors.size == 0:
            continue
        neighbor_displacements = displacement_magnitude[valid_neighbors]
        stress_proxy[point_index] = float(np.mean(np.abs(displacement_magnitude[point_index] - neighbor_displacements)))
    return {'displacement_magnitude': displacement_magnitude.astype(np.float64, copy=False), 'stress_proxy': stress_proxy.astype(np.float64, copy=False)}

def _compute_componentwise_displacement_and_stress(*, points: np.ndarray, restored_points: np.ndarray, closed_labels: np.ndarray, restoration_unknown_mask: np.ndarray, neighbor_indices: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Compute displacement globally and stress within each restored component only."""
    points = np.asarray(points, dtype=np.float64)
    restored_points = np.asarray(restored_points, dtype=np.float64)
    closed_labels = np.asarray(closed_labels, dtype=np.int32).reshape(-1)
    restoration_unknown_mask = np.asarray(restoration_unknown_mask, dtype=bool).reshape(-1)
    displacement_magnitude = np.linalg.norm(points - restored_points, axis=1).astype(np.float64, copy=False)
    stress_proxy = np.zeros(points.shape[0], dtype=np.float64)
    component_ids = sorted((int(x) for x in np.unique(closed_labels) if int(x) > 0))
    for component_id in component_ids:
        component_unknown_mask = (closed_labels == component_id) & restoration_unknown_mask
        if not np.any(component_unknown_mask):
            continue
        local_result = _compute_displacement_and_stress(points=points, restored_points=restored_points, unknown_mask=component_unknown_mask, neighbor_indices=neighbor_indices)
        stress_proxy[component_unknown_mask] = local_result['stress_proxy'][component_unknown_mask]
    return {'displacement_magnitude': displacement_magnitude, 'stress_proxy': stress_proxy}

def _compute_displacement_stress_quadrants(displacement_magnitude: np.ndarray, stress_proxy: np.ndarray, *, closed_mask: np.ndarray | None=None, unknown_mask: np.ndarray | None=None, scope: str=EXP17_GTR_DS_QUADRANT_SCOPE, displacement_norm_threshold: float=EXP17_GTR_DS_QUADRANT_DISPLACEMENT_NORM_THRESHOLD, stress_norm_threshold: float=EXP17_GTR_DS_QUADRANT_STRESS_NORM_THRESHOLD, keep_equal: bool=EXP17_GTR_DS_QUADRANT_KEEP_EQUAL, enabled: bool=EXP17_GTR_DS_QUADRANT_ENABLE) -> dict[str, Any]:
    """Classify points into four displacement/stress quadrants."""
    displacement_magnitude = np.asarray(displacement_magnitude, dtype=np.float64).reshape(-1)
    stress_proxy = np.asarray(stress_proxy, dtype=np.float64).reshape(-1)
    num_points = int(displacement_magnitude.shape[0])
    if stress_proxy.shape[0] != num_points:
        raise ValueError(f'displacement_magnitude and stress_proxy length mismatch: {num_points} vs {stress_proxy.shape[0]}')
    empty_mask = np.zeros(num_points, dtype=bool)
    empty_labels = np.zeros(num_points, dtype=np.int32)
    empty_norm = np.zeros(num_points, dtype=np.float64)
    if not enabled:
        return {'enabled': False, 'displacement_norm': empty_norm, 'stress_norm': empty_norm.copy(), 'active_scope_mask': empty_mask, 'high_displacement_mask': empty_mask.copy(), 'low_displacement_mask': empty_mask.copy(), 'high_stress_mask': empty_mask.copy(), 'low_stress_mask': empty_mask.copy(), 'high_displacement_high_stress_mask': empty_mask.copy(), 'low_displacement_high_stress_mask': empty_mask.copy(), 'high_displacement_low_stress_mask': empty_mask.copy(), 'low_displacement_low_stress_mask': empty_mask.copy(), 'quadrant_labels': empty_labels, 'summary': {'enabled': False, 'scope': str(scope), 'displacement_norm_threshold': float(displacement_norm_threshold), 'stress_norm_threshold': float(stress_norm_threshold), 'keep_equal': bool(keep_equal), 'num_points': num_points, 'active_scope_point_count': 0, 'high_displacement_high_stress_count': 0, 'low_displacement_high_stress_count': 0, 'high_displacement_low_stress_count': 0, 'low_displacement_low_stress_count': 0, 'high_displacement_high_stress_ratio': 0.0, 'low_displacement_high_stress_ratio': 0.0, 'high_displacement_low_stress_ratio': 0.0, 'low_displacement_low_stress_ratio': 0.0}}
    displacement_norm = _percentile_normalize(displacement_magnitude, p_low=HEATMAP_P_LOW, p_high=HEATMAP_P_HIGH)
    stress_norm = _percentile_normalize(stress_proxy, p_low=HEATMAP_P_LOW, p_high=HEATMAP_P_HIGH)
    scope_key = str(scope).strip().lower()
    if scope_key == 'all_points':
        active_scope_mask = np.ones(num_points, dtype=bool)
    elif scope_key == 'closed_mask':
        if closed_mask is None:
            raise ValueError("closed_mask is required when EXP17_GTR_DS_QUADRANT_SCOPE='closed_mask'")
        active_scope_mask = np.asarray(closed_mask, dtype=bool).reshape(-1)
    elif scope_key == 'unknown_mask':
        if unknown_mask is None:
            raise ValueError("unknown_mask is required when EXP17_GTR_DS_QUADRANT_SCOPE='unknown_mask'")
        active_scope_mask = np.asarray(unknown_mask, dtype=bool).reshape(-1)
    else:
        raise ValueError("EXP17_GTR_DS_QUADRANT_SCOPE must be one of: 'all_points', 'closed_mask', 'unknown_mask'")
    if active_scope_mask.shape[0] != num_points:
        raise ValueError(f'active_scope_mask length mismatch: {active_scope_mask.shape[0]} vs {num_points}')
    if keep_equal:
        high_disp = displacement_norm >= float(displacement_norm_threshold)
        high_stress = stress_norm >= float(stress_norm_threshold)
    else:
        high_disp = displacement_norm > float(displacement_norm_threshold)
        high_stress = stress_norm > float(stress_norm_threshold)
    low_disp = ~high_disp
    low_stress = ~high_stress
    high_disp_high_stress_mask = active_scope_mask & high_disp & high_stress
    low_disp_high_stress_mask = active_scope_mask & low_disp & high_stress
    high_disp_low_stress_mask = active_scope_mask & high_disp & low_stress
    low_disp_low_stress_mask = active_scope_mask & low_disp & low_stress
    quadrant_labels = np.zeros(num_points, dtype=np.int32)
    quadrant_labels[high_disp_high_stress_mask] = 1
    quadrant_labels[low_disp_high_stress_mask] = 2
    quadrant_labels[high_disp_low_stress_mask] = 3
    quadrant_labels[low_disp_low_stress_mask] = 4
    return {'enabled': True, 'displacement_norm': displacement_norm, 'stress_norm': stress_norm, 'active_scope_mask': active_scope_mask, 'high_displacement_mask': active_scope_mask & high_disp, 'low_displacement_mask': active_scope_mask & low_disp, 'high_stress_mask': active_scope_mask & high_stress, 'low_stress_mask': active_scope_mask & low_stress, 'high_displacement_high_stress_mask': high_disp_high_stress_mask, 'low_displacement_high_stress_mask': low_disp_high_stress_mask, 'high_displacement_low_stress_mask': high_disp_low_stress_mask, 'low_displacement_low_stress_mask': low_disp_low_stress_mask, 'quadrant_labels': quadrant_labels, 'summary': {'enabled': True, 'scope': str(scope), 'displacement_norm_threshold': float(displacement_norm_threshold), 'stress_norm_threshold': float(stress_norm_threshold), 'keep_equal': bool(keep_equal), 'num_points': num_points, 'active_scope_point_count': int(np.sum(active_scope_mask)), 'high_displacement_high_stress_count': int(np.sum(high_disp_high_stress_mask)), 'low_displacement_high_stress_count': int(np.sum(low_disp_high_stress_mask)), 'high_displacement_low_stress_count': int(np.sum(high_disp_low_stress_mask)), 'low_displacement_low_stress_count': int(np.sum(low_disp_low_stress_mask)), 'high_displacement_high_stress_ratio': float(np.mean(high_disp_high_stress_mask)), 'low_displacement_high_stress_ratio': float(np.mean(low_disp_high_stress_mask)), 'high_displacement_low_stress_ratio': float(np.mean(high_disp_low_stress_mask)), 'low_displacement_low_stress_ratio': float(np.mean(low_disp_low_stress_mask))}}

def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'p95': 0.0, 'max': 0.0}
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)), 'p90': float(np.percentile(values, 90)), 'p95': float(np.percentile(values, 95)), 'max': float(np.max(values))}


_LOCAL_NORMAL_REFERENCE_STAT_NAMES = ('mean', 'median', 'p75', 'p90', 'p95', 'max')


def build_local_normal_reference_rings(
    *,
    candidate_mask: np.ndarray,
    all_candidate_mask: np.ndarray,
    neighbor_indices: list[np.ndarray],
    max_hops: int = EXP17_GTR_LOCAL_NORMAL_REFERENCE_MAX_HOPS,
) -> dict[str, Any]:
    """Build exterior 1--3 hop controls without crossing candidates."""

    candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    all_candidates = np.asarray(all_candidate_mask, dtype=bool).reshape(-1)
    num_points = int(candidate.size)
    if all_candidates.size != num_points or len(neighbor_indices) != num_points:
        raise ValueError('Local-normal ring inputs must have the same point count.')
    if int(max_hops) != 3:
        raise ValueError('Local-normal reference currently requires exactly 3 hops.')
    other_candidate_mask = all_candidates & ~candidate
    _expanded, hop_distance = _graph_dilate_with_distance(
        candidate,
        neighbor_indices,
        max_hops=3,
        allowed_mask=~other_candidate_mask,
    )
    ring_h1 = (hop_distance == 1) & ~all_candidates
    ring_h2 = (hop_distance == 2) & ~all_candidates
    ring_h3 = (hop_distance == 3) & ~all_candidates
    ring_h1_3 = ring_h1 | ring_h2 | ring_h3
    return {
        'ring_h1_mask': ring_h1,
        'ring_h2_mask': ring_h2,
        'ring_h3_mask': ring_h3,
        'ring_h1_3_mask': ring_h1_3,
        'other_candidate_mask': other_candidate_mask,
        'hop_distance': hop_distance,
        'ring_h1_point_count': int(np.sum(ring_h1)),
        'ring_h2_point_count': int(np.sum(ring_h2)),
        'ring_h3_point_count': int(np.sum(ring_h3)),
        'ring_h1_3_point_count': int(np.sum(ring_h1_3)),
        'reference_ring_reaches_hop3': bool(np.any(ring_h3)),
    }


def _local_normal_reference_distribution(
    values: np.ndarray,
) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {name: None for name in _LOCAL_NORMAL_REFERENCE_STAT_NAMES}
    if not np.all(np.isfinite(values)):
        raise ValueError('invalid_numeric')
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p75': float(np.percentile(values, 75.0)),
        'p90': float(np.percentile(values, 90.0)),
        'p95': float(np.percentile(values, 95.0)),
        'max': float(np.max(values)),
    }


def _safe_local_normal_reference_ratio(
    numerator: float,
    denominator: float,
    *,
    eps: float = EXP17_GTR_LOCAL_NORMAL_REFERENCE_EPS,
) -> float:
    return float(float(numerator) / max(abs(float(denominator)), float(eps)))


def summarize_local_normal_reference(
    *,
    candidate_stat_mask: np.ndarray,
    rings: dict[str, Any],
    candidate_displacement: np.ndarray,
    candidate_stress: np.ndarray,
    control_displacement: np.ndarray,
    control_stress: np.ndarray,
    eps: float = EXP17_GTR_LOCAL_NORMAL_REFERENCE_EPS,
) -> dict[str, Any]:
    candidate_stat = np.asarray(candidate_stat_mask, dtype=bool).reshape(-1)
    candidate_d_all = np.asarray(candidate_displacement, dtype=np.float64).reshape(-1)
    candidate_s_all = np.asarray(candidate_stress, dtype=np.float64).reshape(-1)
    control_d_all = np.asarray(control_displacement, dtype=np.float64).reshape(-1)
    control_s_all = np.asarray(control_stress, dtype=np.float64).reshape(-1)
    num_points = int(candidate_stat.size)
    if any(
        values.size != num_points
        for values in (candidate_d_all, candidate_s_all, control_d_all, control_s_all)
    ):
        raise ValueError('Local-normal statistic arrays must have the same point count.')
    candidate_d = candidate_d_all[candidate_stat]
    candidate_s = candidate_s_all[candidate_stat]
    candidate_d_stats = _local_normal_reference_distribution(candidate_d)
    candidate_s_stats = _local_normal_reference_distribution(candidate_s)
    report: dict[str, Any] = {}
    for statistic in _LOCAL_NORMAL_REFERENCE_STAT_NAMES:
        report[f'candidate_d_{statistic}'] = candidate_d_stats[statistic]
        report[f'candidate_s_{statistic}'] = candidate_s_stats[statistic]
    reference_stats: dict[str, dict[str, dict[str, float | None]]] = {}
    for ring_name, mask_key in (
        ('h1', 'ring_h1_mask'),
        ('h2', 'ring_h2_mask'),
        ('h3', 'ring_h3_mask'),
        ('h1_3', 'ring_h1_3_mask'),
    ):
        ring_mask = np.asarray(rings[mask_key], dtype=bool).reshape(-1)
        if ring_mask.size != num_points:
            raise ValueError(f'{mask_key} length mismatch.')
        if np.any(ring_mask & candidate_stat):
            raise AssertionError(
                'Candidate statistic points must never enter reference statistics.'
            )
        d_stats = _local_normal_reference_distribution(control_d_all[ring_mask])
        s_stats = _local_normal_reference_distribution(control_s_all[ring_mask])
        reference_stats[ring_name] = {'d': d_stats, 's': s_stats}
        report[f'ref_{ring_name}_point_count'] = int(np.sum(ring_mask))
        for statistic in _LOCAL_NORMAL_REFERENCE_STAT_NAMES:
            report[f'ref_{ring_name}_d_{statistic}'] = d_stats[statistic]
            report[f'ref_{ring_name}_s_{statistic}'] = s_stats[statistic]
    combined_d_stats = reference_stats['h1_3']['d']
    combined_s_stats = reference_stats['h1_3']['s']
    for statistic in ('median', 'p90', 'p95'):
        values = (
            candidate_d_stats[statistic],
            candidate_s_stats[statistic],
            combined_d_stats[statistic],
            combined_s_stats[statistic],
        )
        if None in values:
            raise ValueError('empty_reference_ring')
        candidate_d_value, candidate_s_value, reference_d_value, reference_s_value = (
            float(value) for value in values
        )
        report[f'd_ratio_{statistic}'] = _safe_local_normal_reference_ratio(
            candidate_d_value, reference_d_value, eps=eps
        )
        report[f's_ratio_{statistic}'] = _safe_local_normal_reference_ratio(
            candidate_s_value, reference_s_value, eps=eps
        )
        if statistic in {'p90', 'p95'}:
            report[f'd_excess_{statistic}'] = candidate_d_value - reference_d_value
            report[f's_excess_{statistic}'] = candidate_s_value - reference_s_value
    if candidate_d.size == 0:
        raise ValueError('empty_candidate_stat_mask')
    ref_d_p90 = float(combined_d_stats['p90'])
    ref_d_p95 = float(combined_d_stats['p95'])
    ref_s_p90 = float(combined_s_stats['p90'])
    ref_s_p95 = float(combined_s_stats['p95'])
    high_d = candidate_d > ref_d_p90
    high_s = candidate_s > ref_s_p90
    report.update({
        'candidate_fraction_d_above_ref_p90': float(np.mean(high_d)),
        'candidate_fraction_d_above_ref_p95': float(np.mean(candidate_d > ref_d_p95)),
        'candidate_fraction_s_above_ref_p90': float(np.mean(high_s)),
        'candidate_fraction_s_above_ref_p95': float(np.mean(candidate_s > ref_s_p95)),
        'candidate_fraction_d_and_s_above_ref_p90': float(np.mean(high_d & high_s)),
        'candidate_fraction_d_and_s_at_or_below_ref_p90': float(np.mean(~high_d & ~high_s)),
        'ref_highD_highS_fraction': float(np.mean(high_d & high_s)),
        'ref_lowD_highS_fraction': float(np.mean(~high_d & high_s)),
        'ref_lowD_lowS_fraction': float(np.mean(~high_d & ~high_s)),
        'ref_highD_lowS_fraction': float(np.mean(high_d & ~high_s)),
    })
    return report


def run_local_normal_control_restoration(
    *,
    points: np.ndarray,
    restoration_laplacian: sp.csr_matrix,
    neighbor_indices: list[np.ndarray],
    candidate_geometry_mask: np.ndarray,
    candidate_stat_mask: np.ndarray,
    all_candidate_mask: np.ndarray,
    candidate_displacement: np.ndarray,
    candidate_stress: np.ndarray,
    gtr_params: dict[str, Any],
) -> dict[str, Any]:
    points_arr = np.asarray(points, dtype=np.float64)
    candidate_geometry = np.asarray(candidate_geometry_mask, dtype=bool).reshape(-1)
    candidate_stat = np.asarray(candidate_stat_mask, dtype=bool).reshape(-1)
    num_points = int(points_arr.shape[0])
    if (
        points_arr.ndim != 2
        or points_arr.shape[1] != 3
        or candidate_geometry.size != num_points
        or candidate_stat.size != num_points
    ):
        raise ValueError('Invalid points/candidate masks for local-normal control restoration.')
    if np.any(candidate_stat & ~candidate_geometry):
        raise ValueError('candidate_stat_mask must be a subset of candidate_geometry_mask.')
    rings = build_local_normal_reference_rings(
        candidate_mask=candidate_geometry,
        all_candidate_mask=all_candidate_mask,
        neighbor_indices=neighbor_indices,
        max_hops=int(gtr_params['local_normal_reference_max_hops']),
    )
    control_unknown_mask = candidate_geometry | np.asarray(
        rings['ring_h1_3_mask'], dtype=bool
    )
    context_mask = ~control_unknown_mask
    base_report: dict[str, Any] = {
        'candidate_point_count': int(np.sum(candidate_geometry)),
        'candidate_geometry_point_count': int(np.sum(candidate_geometry)),
        'candidate_stat_point_count': int(np.sum(candidate_stat)),
        'ring_h1_point_count': rings['ring_h1_point_count'],
        'ring_h2_point_count': rings['ring_h2_point_count'],
        'ring_h3_point_count': rings['ring_h3_point_count'],
        'ring_h1_3_point_count': rings['ring_h1_3_point_count'],
        'reference_ring_reaches_hop3': rings['reference_ring_reaches_hop3'],
        'control_unknown_point_count': int(np.sum(control_unknown_mask)),
        'context_point_count': int(np.sum(context_mask)),
        'restoration_success': False,
        'restoration_failed': False,
        'restoration_fallback_used': False,
        'runtime_control_restoration': 0.0,
        'solver_report': None,
        'local_normal_reference_valid': False,
        'local_normal_reference_skip_reason': None,
    }
    if not np.any(candidate_stat):
        base_report['local_normal_reference_skip_reason'] = 'empty_candidate_stat_mask'
        return {'report': base_report, 'rings': rings}
    if rings['ring_h1_3_point_count'] == 0:
        base_report['local_normal_reference_skip_reason'] = 'empty_reference_ring'
        return {'report': base_report, 'rings': rings}
    if int(np.sum(context_mask)) < int(gtr_params['min_anchor_points']):
        base_report['local_normal_reference_skip_reason'] = 'insufficient_context'
        return {'report': base_report, 'rings': rings}
    solve_start = time.perf_counter()
    try:
        control_restored_points, solver_report = _restore_points_by_mode(
            points=points_arr,
            restoration_laplacian=restoration_laplacian,
            unknown_mask=control_unknown_mask,
            gtr_params=gtr_params,
        )
    except Exception as exc:
        base_report['runtime_control_restoration'] = float(time.perf_counter() - solve_start)
        base_report['restoration_failed'] = True
        base_report['solver_report'] = {'exception': str(exc)}
        base_report['local_normal_reference_skip_reason'] = 'restoration_failed'
        return {'report': base_report, 'rings': rings}
    base_report['runtime_control_restoration'] = float(time.perf_counter() - solve_start)
    base_report['solver_report'] = dict(solver_report)
    base_report['restoration_failed'] = bool(solver_report.get('failed', False))
    base_report['restoration_fallback_used'] = bool(solver_report.get('fallback_used', False))
    if bool(solver_report.get('skipped', False)):
        base_report['local_normal_reference_skip_reason'] = 'restoration_failed'
        return {'report': base_report, 'rings': rings}
    control_restored_points = np.asarray(control_restored_points, dtype=np.float64)
    if (
        control_restored_points.shape != points_arr.shape
        or not np.all(np.isfinite(control_restored_points))
    ):
        base_report['local_normal_reference_skip_reason'] = 'invalid_numeric'
        return {'report': base_report, 'rings': rings}
    control_fields = _compute_displacement_and_stress(
        points=points_arr,
        restored_points=control_restored_points,
        unknown_mask=control_unknown_mask,
        neighbor_indices=neighbor_indices,
    )
    try:
        statistics = summarize_local_normal_reference(
            candidate_stat_mask=candidate_stat,
            rings=rings,
            candidate_displacement=candidate_displacement,
            candidate_stress=candidate_stress,
            control_displacement=control_fields['displacement_magnitude'],
            control_stress=control_fields['stress_proxy'],
            eps=float(gtr_params['local_normal_reference_epsilon']),
        )
    except (ValueError, AssertionError) as exc:
        reason = str(exc)
        base_report['local_normal_reference_skip_reason'] = (
            reason if reason in {'empty_reference_ring', 'invalid_numeric'} else 'invalid_numeric'
        )
        return {'report': base_report, 'rings': rings}
    base_report.update(statistics)
    base_report['restoration_success'] = True
    base_report['local_normal_reference_valid'] = True
    return {'report': base_report, 'rings': rings}


def build_gtr_local_normal_reference_result(
    *,
    enabled: bool,
    points: np.ndarray,
    restoration_laplacian: sp.csr_matrix,
    neighbor_indices: list[np.ndarray],
    closed_labels: np.ndarray,
    decision_labels_for_component_classification: np.ndarray,
    displacement_magnitude: np.ndarray,
    stress_proxy: np.ndarray,
    gtr_params: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray(closed_labels, dtype=np.int32).reshape(-1)
    decision_labels = np.asarray(
        decision_labels_for_component_classification, dtype=np.int32
    ).reshape(-1)
    if decision_labels.shape != labels.shape:
        raise ValueError('decision_labels_for_component_classification length mismatch.')
    component_ids = sorted(int(value) for value in np.unique(labels) if int(value) > 0)
    if not enabled:
        return {
            'enabled': False,
            'diagnostic_only': True,
            'num_candidates_total': int(len(component_ids)),
            'num_candidates_reference_valid': 0,
            'num_candidates_reference_skipped': 0,
            'candidate_reports': [],
            'reason': 'disabled',
        }
    all_candidate_mask = labels > 0
    candidate_reports: list[dict[str, Any]] = []
    for candidate_id in component_ids:
        candidate_geometry_mask = labels == candidate_id
        candidate_stat_mask = decision_labels == candidate_id
        try:
            control = run_local_normal_control_restoration(
                points=points,
                restoration_laplacian=restoration_laplacian,
                neighbor_indices=neighbor_indices,
                candidate_geometry_mask=candidate_geometry_mask,
                candidate_stat_mask=candidate_stat_mask,
                all_candidate_mask=all_candidate_mask,
                candidate_displacement=displacement_magnitude,
                candidate_stress=stress_proxy,
                gtr_params=gtr_params,
            )
            report = dict(control['report'])
        except (ValueError, RuntimeError, AssertionError) as exc:
            report = {
                'candidate_point_count': int(np.sum(candidate_geometry_mask)),
                'candidate_geometry_point_count': int(np.sum(candidate_geometry_mask)),
                'candidate_stat_point_count': int(np.sum(candidate_stat_mask)),
                'restoration_success': False,
                'local_normal_reference_valid': False,
                'local_normal_reference_skip_reason': 'invalid_input',
                'solver_report': {'exception': str(exc)},
            }
        report['candidate_id'] = int(candidate_id)
        candidate_reports.append(report)
    num_valid = int(sum(
        bool(report.get('local_normal_reference_valid')) for report in candidate_reports
    ))
    return {
        'enabled': True,
        'diagnostic_only': True,
        'num_candidates_total': int(len(component_ids)),
        'num_candidates_reference_valid': num_valid,
        'num_candidates_reference_skipped': int(len(component_ids) - num_valid),
        'candidate_reports': candidate_reports,
        'reason': 'ok',
    }


def _resolve_gtr_object_scale(
    tscb_result: dict[str, Any], *, required: bool
) -> dict[str, Any]:
    summaries = tscb_result.get('summaries') if isinstance(tscb_result, dict) else None
    summaries = summaries if isinstance(summaries, dict) else {}
    preprocess_summary = summaries.get('preprocess')
    preprocess_summary = preprocess_summary if isinstance(preprocess_summary, dict) else {}
    scale_summary = preprocess_summary.get('scale')
    scale_summary = scale_summary if isinstance(scale_summary, dict) else {}
    config_summary = summaries.get('config')
    config_summary = config_summary if isinstance(config_summary, dict) else {}
    for source, raw_value in (
        ('tscb_result.summaries.preprocess.scale.cube_size', scale_summary.get('cube_size')),
        ('tscb_result.summaries.config.cube_size', config_summary.get('cube_size')),
    ):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return {'object_scale': value, 'source': source, 'available': True}
    if required:
        raise ValueError(
            'adaptive_normal_rejection requires authoritative object scale at '
            "tscb_result['summaries']['preprocess']['scale']['cube_size']."
        )
    return {'object_scale': None, 'source': 'unavailable', 'available': False}


def _optional_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _local_reference_reports_by_candidate(
    local_normal_reference_result: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    if not isinstance(local_normal_reference_result, dict):
        return {}
    return {
        int(report['candidate_id']): report
        for report in local_normal_reference_result.get('candidate_reports') or []
        if isinstance(report, dict) and report.get('candidate_id') is not None
    }

def _build_gtr_scores_from_exp14(tscb_result: dict[str, Any], exp06: Any) -> tuple[np.ndarray, str]:
    num_points = int(np.asarray(tscb_result['points']).shape[0])
    candidates: list[tuple[str, np.ndarray | None]] = [('semantic_sv', tscb_result.get('semantic_sv')), ('sgcr_candidate_score', tscb_result.get('sgcr_candidate_score')), ('suppressed_intersection_min', tscb_result.get('suppressed_intersection_min')), ('c_abs_enhanced', tscb_result.get('c_abs_enhanced'))]
    score: np.ndarray | None = None
    source_name = 'zeros'
    for (name, candidate) in candidates:
        if candidate is None:
            continue
        candidate_arr = np.asarray(candidate, dtype=np.float64).reshape(-1)
        if candidate_arr.shape[0] != num_points:
            continue
        score = candidate_arr
        source_name = name
        break
    if score is None:
        score = np.zeros(num_points, dtype=np.float64)
    if hasattr(exp06, 'robust_normalize'):
        score = exp06.robust_normalize(score)
    else:
        score = _percentile_normalize(score)
    return (score.astype(np.float64, copy=False), source_name)

def _resolve_exp14_hks_open_boundary_support_mask(tscb_result: dict[str, Any], num_points: int, *, use_expanded: bool=EXP17_GTR_HKS_OPEN_BOUNDARY_SUPPORT_USE_EXPANDED) -> np.ndarray:
    """Resolve exp14 P-view pure-HKS open-boundary mask as GTR restoration support."""
    key = 'hks_open_boundary_mask_expanded' if use_expanded else 'hks_open_boundary_mask'
    support_mask = tscb_result.get(key)
    if support_mask is None:
        hks_result = tscb_result.get('hks_open_boundary_result')
        if isinstance(hks_result, dict):
            support_mask = hks_result.get('boundary_mask_expanded' if use_expanded else 'boundary_mask')
    if support_mask is None:
        return np.zeros(num_points, dtype=bool)
    support_mask = np.asarray(support_mask, dtype=bool).reshape(-1)
    if support_mask.shape[0] != num_points:
        return np.zeros(num_points, dtype=bool)
    return support_mask

def _classify_gtr_components_legacy(closed_labels: np.ndarray, seed_labels: np.ndarray, gtr_scores: np.ndarray, displacement_magnitude: np.ndarray, stress_proxy: np.ndarray, min_closed_component_size: int, displacement_threshold: float, use_stress_veto: bool, stress_threshold: float) -> dict[str, Any]:
    """Apply only the frozen Legacy-absolute component decision.

    Adaptive-normal-rejection and Local-Normal-Reference diagnostics are not
    part of the paper mainline.  The zero-valued aggregate counters are kept
    solely so downstream result serialization remains schema-compatible.
    """
    closed_labels = np.asarray(closed_labels, dtype=np.int32).reshape(-1)
    seed_labels = np.asarray(seed_labels, dtype=np.int32).reshape(-1)
    gtr_scores = np.asarray(gtr_scores, dtype=np.float64).reshape(-1)
    displacement_magnitude = np.asarray(displacement_magnitude, dtype=np.float64).reshape(-1)
    stress_proxy = np.asarray(stress_proxy, dtype=np.float64).reshape(-1)
    num_working_points = int(closed_labels.size)
    if any((values.size != num_working_points for values in (seed_labels, gtr_scores, displacement_magnitude, stress_proxy))):
        raise ValueError('GTR classifier arrays must share the working-cloud length.')
    final_defect_labels = np.zeros_like(closed_labels)
    rejected_labels = np.zeros_like(closed_labels)
    component_reports: list[dict[str, Any]] = []
    component_ids = sorted((int(value) for value in np.unique(closed_labels) if int(value) > 0))
    for component_id in component_ids:
        seed_mask = seed_labels == component_id
        closed_mask = closed_labels == component_id
        num_seed_points = int(np.sum(seed_mask))
        num_closed_points = int(np.sum(closed_mask))
        disp_stats = _distribution_stats(displacement_magnitude[closed_mask])
        stress_stats = _distribution_stats(stress_proxy[closed_mask])
        score_stats = _distribution_stats(gtr_scores[closed_mask])
        too_small = bool(num_closed_points < min_closed_component_size)
        low_displacement = bool(disp_stats['p90'] < displacement_threshold)
        stress_veto = bool(use_stress_veto and stress_stats['p90'] > stress_threshold)
        accepted = True
        rejection_reason: str | None = None
        if too_small:
            accepted = False
            rejection_reason = 'too_small_closed_component'
        elif low_displacement:
            accepted = False
            rejection_reason = 'low_displacement'
        elif stress_veto:
            accepted = False
            rejection_reason = 'stress_veto'
        component_reports.append({'component_id': component_id, 'classifier_mode': 'legacy_absolute', 'accepted': accepted, 'rejection_reason': rejection_reason, 'too_small': too_small, 'legacy_accepted': accepted, 'legacy_rejection_reason': rejection_reason, 'legacy_low_displacement': low_displacement, 'legacy_stress_veto': stress_veto, 'candidate_point_count': num_closed_points, 'num_seed_points': num_seed_points, 'num_closed_points': num_closed_points, 'displacement_mean': disp_stats['mean'], 'displacement_median': disp_stats['median'], 'displacement_p90': disp_stats['p90'], 'displacement_p95': disp_stats['p95'], 'displacement_max': disp_stats['max'], 'stress_mean': stress_stats['mean'], 'stress_median': stress_stats['median'], 'stress_p90': stress_stats['p90'], 'stress_p95': stress_stats['p95'], 'stress_max': stress_stats['max'], 'score_mean': score_stats['mean'], 'score_p90': score_stats['p90'], 'D_mean': disp_stats['mean'], 'D_p90': disp_stats['p90'], 'D_max': disp_stats['max'], 'stress_p90_report': stress_stats['p90']})
        if accepted:
            final_defect_labels[closed_mask] = component_id
        else:
            rejected_labels[closed_mask] = component_id
    rejection_reasons = [str(report['rejection_reason']) for report in component_reports if not bool(report['accepted'])]
    num_accepted = sum((bool(report['accepted']) for report in component_reports))
    return {'classifier_mode': 'legacy_absolute', 'final_defect_labels': final_defect_labels, 'rejected_labels': rejected_labels, 'final_defect_mask': final_defect_labels > 0, 'rejected_mask': rejected_labels > 0, 'component_reports': component_reports, 'num_accepted_components': int(num_accepted), 'num_rejected_components': int(len(component_reports) - num_accepted), 'num_candidate_too_small_rejected': int(sum((reason == 'too_small_closed_component' for reason in rejection_reasons))), 'num_candidate_relative_normal_rejected': 0, 'num_candidate_large_shallow_rejected': 0, 'num_candidate_both_adaptive_rejected': 0, 'num_candidate_stress_veto_rejected': int(sum((reason == 'stress_veto' for reason in rejection_reasons))), 'num_candidate_legacy_low_displacement_rejected': int(sum((reason == 'low_displacement' for reason in rejection_reasons))), 'num_candidate_accepted': int(num_accepted)}


def _classify_gtr_components(
    closed_labels: np.ndarray,
    seed_labels: np.ndarray,
    gtr_scores: np.ndarray,
    displacement_magnitude: np.ndarray,
    stress_proxy: np.ndarray,
    min_closed_component_size: int,
    displacement_threshold: float,
    use_stress_veto: bool,
    stress_threshold: float,
    *,
    classifier_mode: str,
    local_normal_reference_result: dict[str, Any] | None,
    object_scale: float | None,
    adaptive_relative_normal_max_ratio: float,
    adaptive_min_large_fraction: float,
    adaptive_max_shallow_displacement_normalized: float,
) -> dict[str, Any]:
    """Apply the frozen adaptive-normal-rejection decision."""

    mode = str(classifier_mode).strip().lower()
    if mode not in {'legacy_absolute', 'adaptive_normal_rejection'}:
        raise ValueError(
            "GTR candidate classifier mode must be 'legacy_absolute' or "
            "'adaptive_normal_rejection'."
        )
    reference_enabled = bool(
        isinstance(local_normal_reference_result, dict)
        and local_normal_reference_result.get('enabled')
    )
    if mode == 'adaptive_normal_rejection' and not reference_enabled:
        raise ValueError('adaptive_normal_rejection requires Local Normal Reference.')
    object_scale_value = _optional_finite_float(object_scale)
    if object_scale_value is not None and object_scale_value <= 0.0:
        object_scale_value = None
    if mode == 'adaptive_normal_rejection' and object_scale_value is None:
        raise ValueError(
            'adaptive_normal_rejection requires a finite positive authoritative object_scale.'
        )
    closed_labels = np.asarray(closed_labels, dtype=np.int32).reshape(-1)
    seed_labels = np.asarray(seed_labels, dtype=np.int32).reshape(-1)
    gtr_scores = np.asarray(gtr_scores, dtype=np.float64).reshape(-1)
    displacement_magnitude = np.asarray(displacement_magnitude, dtype=np.float64).reshape(-1)
    stress_proxy = np.asarray(stress_proxy, dtype=np.float64).reshape(-1)
    num_working_points = int(closed_labels.size)
    if any(
        values.size != num_working_points
        for values in (seed_labels, gtr_scores, displacement_magnitude, stress_proxy)
    ):
        raise ValueError('GTR classifier arrays must share the working-cloud length.')
    reference_by_id = _local_reference_reports_by_candidate(
        local_normal_reference_result
    )
    final_defect_labels = np.zeros_like(closed_labels)
    rejected_labels = np.zeros_like(closed_labels)
    component_reports: list[dict[str, Any]] = []
    for component_id in sorted(
        int(value) for value in np.unique(closed_labels) if int(value) > 0
    ):
        seed_mask = seed_labels == component_id
        closed_mask = closed_labels == component_id
        num_seed_points = int(np.sum(seed_mask))
        num_closed_points = int(np.sum(closed_mask))
        disp_stats = _distribution_stats(displacement_magnitude[closed_mask])
        stress_stats = _distribution_stats(stress_proxy[closed_mask])
        score_stats = _distribution_stats(gtr_scores[closed_mask])
        local_reference_report = reference_by_id.get(component_id) or {}
        local_reference_valid = bool(
            local_reference_report.get('local_normal_reference_valid', False)
        )
        d_ratio_p95 = _optional_finite_float(local_reference_report.get('d_ratio_p95'))
        s_ratio_p95 = _optional_finite_float(local_reference_report.get('s_ratio_p95'))
        if not reference_enabled:
            relative_normal_unavailable_reason = 'local_normal_reference_disabled'
        elif not local_reference_valid:
            relative_normal_unavailable_reason = str(
                local_reference_report.get('local_normal_reference_skip_reason')
                or 'invalid_local_reference'
            )
        elif d_ratio_p95 is None or s_ratio_p95 is None:
            relative_normal_unavailable_reason = 'missing_reference_ratio'
        else:
            relative_normal_unavailable_reason = None
        relative_normal_triggered = bool(
            relative_normal_unavailable_reason is None
            and d_ratio_p95 <= adaptive_relative_normal_max_ratio
            and s_ratio_p95 <= adaptive_relative_normal_max_ratio
        )
        candidate_point_fraction = float(num_closed_points / max(num_working_points, 1))
        candidate_d_p95_normalized = (
            float(disp_stats['p95'] / object_scale_value)
            if object_scale_value is not None
            else None
        )
        large_shallow_triggered = bool(
            candidate_point_fraction >= adaptive_min_large_fraction
            and candidate_d_p95_normalized is not None
            and candidate_d_p95_normalized
            <= adaptive_max_shallow_displacement_normalized
        )
        adaptive_normal_reject = bool(
            relative_normal_triggered or large_shallow_triggered
        )
        too_small = bool(num_closed_points < min_closed_component_size)
        legacy_low_displacement = bool(disp_stats['p90'] < displacement_threshold)
        legacy_stress_veto = bool(
            use_stress_veto and stress_stats['p90'] > stress_threshold
        )
        legacy_accepted = True
        legacy_rejection_reason: str | None = None
        if too_small:
            legacy_accepted = False
            legacy_rejection_reason = 'too_small_closed_component'
        elif legacy_low_displacement:
            legacy_accepted = False
            legacy_rejection_reason = 'low_displacement'
        elif legacy_stress_veto:
            legacy_accepted = False
            legacy_rejection_reason = 'stress_veto'
        adaptive_accepted = True
        adaptive_rejection_reason = 'accepted'
        if too_small:
            adaptive_accepted = False
            adaptive_rejection_reason = 'too_small'
        elif relative_normal_triggered and large_shallow_triggered:
            adaptive_accepted = False
            adaptive_rejection_reason = 'relative_normal_and_large_shallow'
        elif relative_normal_triggered:
            adaptive_accepted = False
            adaptive_rejection_reason = 'relative_normal'
        elif large_shallow_triggered:
            adaptive_accepted = False
            adaptive_rejection_reason = 'large_shallow'
        elif legacy_stress_veto:
            adaptive_accepted = False
            adaptive_rejection_reason = 'stress_veto'
        if mode == 'legacy_absolute':
            accepted = legacy_accepted
            rejection_reason = legacy_rejection_reason
        else:
            accepted = adaptive_accepted
            rejection_reason = None if adaptive_accepted else adaptive_rejection_reason
        component_reports.append({
            'component_id': int(component_id),
            'classifier_mode': mode,
            'accepted': bool(accepted),
            'rejection_reason': rejection_reason,
            'too_small': too_small,
            'legacy_accepted': bool(legacy_accepted),
            'legacy_rejection_reason': legacy_rejection_reason,
            'legacy_low_displacement': legacy_low_displacement,
            'legacy_stress_veto': legacy_stress_veto,
            'local_reference_valid': local_reference_valid,
            'd_ratio_p95': d_ratio_p95,
            's_ratio_p95': s_ratio_p95,
            'relative_normal_triggered': relative_normal_triggered,
            'relative_normal_unavailable_reason': relative_normal_unavailable_reason,
            'candidate_point_count': num_closed_points,
            'total_working_point_count': num_working_points,
            'candidate_point_fraction': candidate_point_fraction,
            'candidate_d_p95': disp_stats['p95'],
            'object_scale': object_scale_value,
            'candidate_d_p95_normalized': candidate_d_p95_normalized,
            'candidate_d_p95_normalized_by_object_scale': candidate_d_p95_normalized,
            'normalized_d_p95': candidate_d_p95_normalized,
            'large_shallow_triggered': large_shallow_triggered,
            'adaptive_normal_reject': adaptive_normal_reject,
            'adaptive_accepted': bool(adaptive_accepted),
            'adaptive_rejection_reason': adaptive_rejection_reason,
            'num_seed_points': num_seed_points,
            'num_closed_points': num_closed_points,
            'displacement_mean': disp_stats['mean'],
            'displacement_median': disp_stats['median'],
            'displacement_p90': disp_stats['p90'],
            'displacement_p95': disp_stats['p95'],
            'displacement_max': disp_stats['max'],
            'stress_mean': stress_stats['mean'],
            'stress_median': stress_stats['median'],
            'stress_p90': stress_stats['p90'],
            'stress_p95': stress_stats['p95'],
            'stress_max': stress_stats['max'],
            'score_mean': score_stats['mean'],
            'score_p90': score_stats['p90'],
            'D_mean': disp_stats['mean'],
            'D_p90': disp_stats['p90'],
            'D_max': disp_stats['max'],
            'stress_p90_report': stress_stats['p90'],
        })
        if accepted:
            final_defect_labels[closed_mask] = component_id
        else:
            rejected_labels[closed_mask] = component_id
    rejection_reasons = [
        str(report['rejection_reason'])
        for report in component_reports
        if not bool(report['accepted'])
    ]
    num_accepted = sum(bool(report['accepted']) for report in component_reports)
    return {
        'classifier_mode': mode,
        'final_defect_labels': final_defect_labels,
        'rejected_labels': rejected_labels,
        'final_defect_mask': final_defect_labels > 0,
        'rejected_mask': rejected_labels > 0,
        'component_reports': component_reports,
        'num_accepted_components': int(num_accepted),
        'num_rejected_components': int(len(component_reports) - num_accepted),
        'num_candidate_too_small_rejected': int(sum(
            reason in {'too_small', 'too_small_closed_component'}
            for reason in rejection_reasons
        )),
        'num_candidate_relative_normal_rejected': int(sum(
            reason == 'relative_normal' for reason in rejection_reasons
        )),
        'num_candidate_large_shallow_rejected': int(sum(
            reason == 'large_shallow' for reason in rejection_reasons
        )),
        'num_candidate_both_adaptive_rejected': int(sum(
            reason == 'relative_normal_and_large_shallow'
            for reason in rejection_reasons
        )),
        'num_candidate_stress_veto_rejected': int(sum(
            reason == 'stress_veto' for reason in rejection_reasons
        )),
        'num_candidate_legacy_low_displacement_rejected': int(sum(
            reason == 'low_displacement' for reason in rejection_reasons
        )),
        'num_candidate_accepted': int(num_accepted),
    }

def _prepare_local_normal_u_cleanup(*, num_points: int, enabled: bool=False, mode: str=EXP17_LOCAL_NORMAL_U_CLEANUP_DEFAULT_MODE, cleanup_mask: np.ndarray | None=None, strong_mask: np.ndarray | None=None, weak_mask: np.ndarray | None=None, metadata: dict[str, Any] | None=None) -> dict[str, Any]:
    """Normalize an optional caller-provided Local Normal-U cleanup request.

    This adapter deliberately knows nothing about Q/U/Y or component rules.
    Exp22 owns that mapping and supplies only point masks plus audit metadata.
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in EXP17_LOCAL_NORMAL_U_CLEANUP_ALLOWED_MODES:
        raise ValueError("local_normal_u_cleanup_mode must be one of: 'final_mask_only', 'pre_component_classification'")
    request_metadata = dict(metadata or {})

    def _optional_mask(values: np.ndarray | None, name: str) -> np.ndarray:
        if values is None:
            return np.zeros(num_points, dtype=bool)
        mask = np.asarray(values, dtype=bool).reshape(-1)
        if mask.shape != (num_points,):
            raise ValueError(f'{name} must have shape ({num_points},), got {mask.shape}.')
        return mask.copy()
    strong = _optional_mask(strong_mask, 'local_strong_normal_u_mask')
    weak = _optional_mask(weak_mask, 'local_weak_normal_u_mask')
    if cleanup_mask is None:
        union = strong | weak
    else:
        union = _optional_mask(cleanup_mask, 'local_normal_u_cleanup_mask')
        if np.any((strong | weak) & ~union):
            raise ValueError('local_normal_u_cleanup_mask must contain the supplied strong/weak masks.')
    component_rule_available = bool(request_metadata.get('component_rule_available', True))
    requested_enabled = bool(enabled)
    effective_enabled = bool(requested_enabled and component_rule_available and np.any(union))
    if not requested_enabled:
        reason = 'disabled_by_local_cleanup_switch'
    elif not component_rule_available:
        reason = str(request_metadata.get('reason') or 'component_rule_seed_evidence_unavailable')
    elif not np.any(union):
        reason = str(request_metadata.get('reason') or 'no_local_normal_u_components')
    else:
        reason = 'enabled'
    strong_components = list(request_metadata.get('strong_normal_u_components') or [])
    weak_components = list(request_metadata.get('weak_normal_u_components') or [])
    all_components = list(request_metadata.get('normal_u_components') or strong_components + weak_components)
    return {'overall': {'enabled': requested_enabled, 'effective_enabled': effective_enabled, 'mode': mode_key, 'component_rule_available': component_rule_available, 'reason': reason, 'num_strong_normal_u_components': int(len(strong_components)), 'num_weak_normal_u_components': int(len(weak_components)), 'num_total_normal_u_components': int(len(all_components)), 'requested_strong_normal_u_point_count': int(np.sum(strong)), 'requested_weak_normal_u_point_count': int(np.sum(weak)), 'requested_union_u_point_count': int(np.sum(union)), 'applied_cleanup_point_count': 0, 'num_gtr_candidates_touched': 0, 'num_accepted_to_rejected': 0, 'num_rejected_to_accepted': 0, 'num_accepted_unchanged': 0, 'num_rejected_unchanged': 0, 'num_candidates_emptied_by_cleanup': 0, 'final_removed_point_count': 0}, 'strong_normal_q_ids': [int(item['q_id']) for item in strong_components if 'q_id' in item], 'weak_normal_q_ids': [int(item['q_id']) for item in weak_components if 'q_id' in item], 'normal_q_ids': [int(item['q_id']) for item in all_components if 'q_id' in item], 'normal_y_ids': sorted({int(item['y_id']) for item in all_components if 'y_id' in item}), 'strong_normal_u_components': strong_components, 'weak_normal_u_components': weak_components, 'normal_u_components': all_components, 'local_strong_normal_u_mask': strong, 'local_weak_normal_u_mask': weak, 'local_normal_u_cleanup_mask': union, 'local_normal_u_cleanup_applied_mask': np.zeros(num_points, dtype=bool), 'local_normal_u_cleanup_final_removed_mask': np.zeros(num_points, dtype=bool), 'candidate_reports': []}

def _build_local_normal_u_candidate_audit(*, decision_labels_before: np.ndarray, decision_labels_after: np.ndarray, strong_mask: np.ndarray, weak_mask: np.ndarray, cleanup_mask: np.ndarray, displacement_magnitude: np.ndarray, stress_proxy: np.ndarray, gtr_scores: np.ndarray, classification_before: dict[str, Any], classification_after: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build per-candidate before/after diagnostics from fixed D/S fields."""
    before_labels = np.asarray(decision_labels_before, dtype=np.int32).reshape(-1)
    after_labels = np.asarray(decision_labels_after, dtype=np.int32).reshape(-1)
    strong = np.asarray(strong_mask, dtype=bool).reshape(-1)
    weak = np.asarray(weak_mask, dtype=bool).reshape(-1)
    cleanup = np.asarray(cleanup_mask, dtype=bool).reshape(-1)
    displacement = np.asarray(displacement_magnitude, dtype=np.float64).reshape(-1)
    stress = np.asarray(stress_proxy, dtype=np.float64).reshape(-1)
    scores = np.asarray(gtr_scores, dtype=np.float64).reshape(-1)
    lengths = {before_labels.size, after_labels.size, strong.size, weak.size, cleanup.size, displacement.size, stress.size, scores.size}
    if len(lengths) != 1:
        raise ValueError('Local Normal-U candidate audit arrays must have equal length.')
    before_reports = {int(report['component_id']): report for report in classification_before.get('component_reports', [])}
    after_reports = {int(report['component_id']): report for report in classification_after.get('component_reports', [])}
    touched_ids = sorted((int(value) for value in np.unique(before_labels[cleanup & (before_labels > 0)]) if int(value) > 0))
    reports: list[dict[str, Any]] = []
    transitions = {'num_accepted_to_rejected': 0, 'num_rejected_to_accepted': 0, 'num_accepted_unchanged': 0, 'num_rejected_unchanged': 0, 'num_candidates_emptied_by_cleanup': 0}
    for candidate_id in touched_ids:
        before_mask = before_labels == candidate_id
        after_mask = after_labels == candidate_id
        removed_mask = before_mask & cleanup
        before_report = before_reports.get(candidate_id, {})
        after_report = after_reports.get(candidate_id, {})
        accepted_before = bool(before_report.get('accepted', False))
        emptied = bool(np.any(before_mask) and (not np.any(after_mask)))
        accepted_after = bool(after_report.get('accepted', False)) if not emptied else False
        if accepted_before and (not accepted_after):
            transitions['num_accepted_to_rejected'] += 1
        elif not accepted_before and accepted_after:
            transitions['num_rejected_to_accepted'] += 1
        elif accepted_before:
            transitions['num_accepted_unchanged'] += 1
        else:
            transitions['num_rejected_unchanged'] += 1
        if emptied:
            transitions['num_candidates_emptied_by_cleanup'] += 1

        def _stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
            return _distribution_stats(values[mask])
        before_disp = _stats(displacement, before_mask)
        after_disp = _stats(displacement, after_mask)
        before_stress = _stats(stress, before_mask)
        after_stress = _stats(stress, after_mask)
        before_score = _stats(scores, before_mask)
        after_score = _stats(scores, after_mask)
        reports.append({'candidate_id': candidate_id, 'y_id': candidate_id, 'original_decision_point_count': int(np.sum(before_mask)), 'removed_local_u_point_count': int(np.sum(removed_mask)), 'retained_point_count': int(np.sum(after_mask)), 'strong_normal_u_removed_point_count': int(np.sum(before_mask & strong)), 'weak_normal_u_removed_point_count': int(np.sum(before_mask & weak)), 'displacement_mean_before': before_disp['mean'], 'displacement_p90_before': before_disp['p90'], 'displacement_p95_before': before_disp['p95'], 'stress_mean_before': before_stress['mean'], 'stress_p90_before': before_stress['p90'], 'stress_p95_before': before_stress['p95'], 'score_mean_before': before_score['mean'], 'score_p90_before': before_score['p90'], 'displacement_mean_after': after_disp['mean'], 'displacement_p90_after': after_disp['p90'], 'displacement_p95_after': after_disp['p95'], 'stress_mean_after': after_stress['mean'], 'stress_p90_after': after_stress['p90'], 'stress_p95_after': after_stress['p95'], 'score_mean_after': after_score['mean'], 'score_p90_after': after_score['p90'], 'accepted_before_cleanup': accepted_before, 'rejection_reason_before_cleanup': before_report.get('rejection_reason'), 'accepted_after_cleanup': accepted_after, 'rejection_reason_after_cleanup': 'emptied_by_local_u_cleanup' if emptied else after_report.get('rejection_reason')})
    return (reports, transitions)

def _estimate_point_spacing_by_knn(points: np.ndarray, *, k: int=EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_K) -> dict[str, Any]:
    """Estimate local point-cloud sampling spacing via k nearest neighbors."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    num_points = int(points.shape[0])
    if num_points <= 1:
        zeros = np.zeros(num_points, dtype=np.float64)
        return {'local_spacing': zeros, 'median_spacing': 0.0, 'mean_spacing': 0.0, 'p25_spacing': 0.0, 'p75_spacing': 0.0, 'p90_spacing': 0.0, 'k': 0, 'k_requested': int(k), 'num_points': num_points}
    k_eff = int(max(1, min(int(k), num_points - 1)))
    tree = cKDTree(points)
    (distances, _indices) = tree.query(points, k=k_eff + 1)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim == 1:
        distances = distances.reshape(num_points, -1)
    neighbor_distances = np.nan_to_num(distances[:, 1:], nan=0.0, posinf=0.0, neginf=0.0)
    local_spacing = np.mean(neighbor_distances, axis=1)
    local_spacing = np.nan_to_num(local_spacing, nan=0.0, posinf=0.0, neginf=0.0)
    return {'local_spacing': local_spacing, 'median_spacing': float(np.median(local_spacing)), 'mean_spacing': float(np.mean(local_spacing)), 'p25_spacing': float(np.percentile(local_spacing, 25)), 'p75_spacing': float(np.percentile(local_spacing, 75)), 'p90_spacing': float(np.percentile(local_spacing, 90)), 'k': int(k_eff), 'k_requested': int(k), 'num_points': num_points}

def _resolve_point_displacement_threshold(points: np.ndarray, *, mode: str=EXP17_GTR_POINT_DISPLACEMENT_THRESHOLD_MODE, fixed_threshold: float=EXP17_GTR_POINT_DISPLACEMENT_THRESHOLD, adaptive_k: int=EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_K, adaptive_alpha: float=EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_ALPHA, adaptive_use_clamp: bool=EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_USE_CLAMP, adaptive_min: float=EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_MIN, adaptive_max: float=EXP17_GTR_POINT_DISPLACEMENT_ADAPTIVE_MAX) -> tuple[float, dict[str, Any]]:
    """Resolve fixed, sample-adaptive, or D/S-quadrant point-level filter mode."""
    mode_key = str(mode).strip().lower()
    if mode_key == 'fixed':
        threshold = float(fixed_threshold)
        summary = {'mode': 'fixed', 'threshold': threshold, 'fixed_threshold': float(fixed_threshold), 'adaptive_enabled': False, 'uses_displacement_threshold': True, 'uses_ds_quadrant_filter': False}
        return (float(threshold), summary)
    if mode_key == 'adaptive_nn_spacing':
        spacing_result = _estimate_point_spacing_by_knn(points, k=adaptive_k)
        raw_threshold = float(adaptive_alpha) * float(spacing_result['median_spacing'])
        if adaptive_use_clamp:
            threshold = float(np.clip(raw_threshold, adaptive_min, adaptive_max))
        else:
            threshold = float(raw_threshold)
        summary = {'mode': 'adaptive_nn_spacing', 'threshold': float(threshold), 'raw_threshold': float(raw_threshold), 'adaptive_enabled': True, 'adaptive_k': int(adaptive_k), 'adaptive_k_requested': int(spacing_result.get('k_requested', adaptive_k)), 'adaptive_k_effective': int(spacing_result['k']), 'adaptive_alpha': float(adaptive_alpha), 'adaptive_use_clamp': bool(adaptive_use_clamp), 'adaptive_min': float(adaptive_min), 'adaptive_max': float(adaptive_max), 'spacing_median': float(spacing_result['median_spacing']), 'spacing_mean': float(spacing_result['mean_spacing']), 'spacing_p25': float(spacing_result['p25_spacing']), 'spacing_p75': float(spacing_result['p75_spacing']), 'spacing_p90': float(spacing_result['p90_spacing']), 'spacing_num_points': int(spacing_result['num_points']), 'uses_displacement_threshold': True, 'uses_ds_quadrant_filter': False}
        return (float(threshold), summary)
    if mode_key == 'ds_quadrant_remove_lowd_lows':
        summary = {'mode': 'ds_quadrant_remove_lowD_lowS', 'threshold': None, 'raw_threshold': None, 'adaptive_enabled': False, 'uses_displacement_threshold': False, 'uses_ds_quadrant_filter': True, 'removed_quadrant': 'low_displacement_low_stress', 'removed_quadrant_label': 4, 'kept_quadrants': ['high_displacement_high_stress', 'low_displacement_high_stress', 'high_displacement_low_stress'], 'kept_quadrant_labels': [1, 2, 3]}
        return (float('nan'), summary)
    raise ValueError("EXP17_GTR_POINT_DISPLACEMENT_THRESHOLD_MODE must be one of: 'fixed', 'adaptive_nn_spacing', 'ds_quadrant_remove_lowD_lowS'")

def _apply_ds_quadrant_filter_to_accepted_labels(*, final_defect_labels: np.ndarray, rejected_labels: np.ndarray, quadrant_labels: np.ndarray, low_displacement_low_stress_mask: np.ndarray, enabled: bool=EXP17_GTR_USE_POINT_DISPLACEMENT_FILTER, protected_keep_mask: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove View-6 lowD-lowS points inside accepted components.

    Keeps View 3 / View 4 / View 5. Matches the existing displacement-threshold
    point filter style: removed points are cleared from final_defect_labels and
    are not moved into rejected_labels.
    """
    final_defect_labels = np.asarray(final_defect_labels, dtype=np.int32).reshape(-1)
    rejected_labels = np.asarray(rejected_labels, dtype=np.int32).reshape(-1)
    quadrant_labels = np.asarray(quadrant_labels, dtype=np.int32).reshape(-1)
    low_displacement_low_stress_mask = np.asarray(low_displacement_low_stress_mask, dtype=bool).reshape(-1)
    num_points = int(final_defect_labels.shape[0])
    if rejected_labels.shape[0] != num_points or quadrant_labels.shape[0] != num_points or low_displacement_low_stress_mask.shape[0] != num_points:
        raise ValueError('final_defect_labels / rejected_labels / quadrant_labels / low_displacement_low_stress_mask length mismatch.')
    if protected_keep_mask is None:
        protected_keep_mask_bool = np.zeros(num_points, dtype=bool)
    else:
        protected_keep_mask_bool = np.asarray(protected_keep_mask, dtype=bool).reshape(-1)
        if protected_keep_mask_bool.shape[0] != num_points:
            raise ValueError(f'protected_keep_mask length mismatch: {protected_keep_mask_bool.shape[0]} vs {num_points}')
    accepted_mask = final_defect_labels > 0
    num_input_accepted_points = int(np.sum(accepted_mask))
    if not enabled:
        return (final_defect_labels.copy(), rejected_labels.copy(), {'enabled': False, 'mode': 'ds_quadrant_remove_lowD_lowS', 'rule': 'keep View3 highD-highS, View4 lowD-highS, View5 highD-lowS; remove View6 lowD-lowS', 'removed_quadrant_label': 4, 'threshold': None, 'num_input_accepted_points': num_input_accepted_points, 'num_removed_lowD_lowS_points': 0, 'num_removed_points': 0, 'num_kept_points': num_input_accepted_points, 'num_protected_points': int(np.sum(protected_keep_mask_bool & accepted_mask)), 'removed_ratio_in_accepted': 0.0, 'uses_ds_quadrant_filter': True, 'uses_displacement_threshold': False})
    remove_mask = accepted_mask & low_displacement_low_stress_mask
    remove_mask &= ~protected_keep_mask_bool
    filtered_final_defect_labels = final_defect_labels.copy()
    filtered_final_defect_labels[remove_mask] = 0
    filtered_rejected_labels = rejected_labels.copy()
    removed_by_ds_labels = final_defect_labels.copy()
    removed_by_ds_labels[~remove_mask] = 0
    num_removed = int(np.sum(remove_mask))
    num_kept = int(np.sum(filtered_final_defect_labels > 0))
    summary: dict[str, Any] = {'enabled': True, 'mode': 'ds_quadrant_remove_lowD_lowS', 'rule': 'keep View3 highD-highS, View4 lowD-highS, View5 highD-lowS; remove View6 lowD-lowS', 'removed_quadrant_label': 4, 'threshold': None, 'num_input_accepted_points': num_input_accepted_points, 'num_removed_lowD_lowS_points': num_removed, 'num_removed_points': num_removed, 'num_kept_points': num_kept, 'num_protected_points': int(np.sum(protected_keep_mask_bool & accepted_mask)), 'removed_ratio_in_accepted': float(num_removed / max(num_input_accepted_points, 1)), 'uses_ds_quadrant_filter': True, 'uses_displacement_threshold': False, 'removed_by_ds_labels_nonzero_count': int(np.sum(removed_by_ds_labels > 0)), 'num_removed_by_quadrant_label_4': int(np.sum(remove_mask & (quadrant_labels == 4)))}
    return (filtered_final_defect_labels, filtered_rejected_labels, summary)

def _apply_point_displacement_filter_to_accepted_labels(final_defect_labels: np.ndarray, rejected_labels: np.ndarray, displacement_magnitude: np.ndarray, *, threshold: float, keep_equal: bool=True, enabled: bool=True, protected_keep_mask: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Filter low-displacement points inside accepted components.

    This does not decide whether a component is accepted/rejected.
    It only refines point-level output after component acceptance.

    Points marked by protected_keep_mask (e.g. open-boundary support overlap)
    are kept even when their displacement is below threshold.
    """
    final_defect_labels = np.asarray(final_defect_labels, dtype=np.int32).reshape(-1)
    rejected_labels = np.asarray(rejected_labels, dtype=np.int32).reshape(-1)
    displacement_magnitude = np.asarray(displacement_magnitude, dtype=np.float64).reshape(-1)
    if protected_keep_mask is None:
        protected_keep_mask_bool = np.zeros(final_defect_labels.shape[0], dtype=bool)
    else:
        protected_keep_mask_bool = np.asarray(protected_keep_mask, dtype=bool).reshape(-1)
        if protected_keep_mask_bool.shape[0] != final_defect_labels.shape[0]:
            raise ValueError(f'protected_keep_mask length mismatch: {protected_keep_mask_bool.shape[0]} vs {final_defect_labels.shape[0]}')
    if not enabled:
        return (final_defect_labels.copy(), rejected_labels.copy(), {'enabled': False, 'threshold': float(threshold), 'keep_equal': bool(keep_equal), 'num_removed_points': 0, 'num_kept_points': int(np.sum(final_defect_labels > 0)), 'num_protected_keep_points': int(np.sum(protected_keep_mask_bool & (final_defect_labels > 0))), 'component_reports': []})
    filtered_final_labels = final_defect_labels.copy()
    filtered_rejected_labels = rejected_labels.copy()
    num_input_final_points = int(np.sum(final_defect_labels > 0))
    num_removed_points = 0
    component_filter_reports: list[dict[str, Any]] = []
    component_ids = sorted((int(x) for x in np.unique(final_defect_labels) if int(x) > 0))
    for component_id in component_ids:
        component_mask = final_defect_labels == component_id
        component_indices = np.flatnonzero(component_mask).astype(np.int64, copy=False)
        values = displacement_magnitude[component_mask]
        local_protected = protected_keep_mask_bool[component_indices]
        if keep_equal:
            displacement_condition = values >= threshold
        else:
            displacement_condition = values > threshold
        keep_mask_local = local_protected | displacement_condition
        remove_indices = component_indices[~keep_mask_local]
        if remove_indices.size > 0:
            filtered_final_labels[remove_indices] = 0
            num_removed_points += int(remove_indices.size)
        num_input_points = int(component_indices.size)
        num_kept_points = int(np.sum(keep_mask_local))
        value_stats = _distribution_stats(values)
        component_filter_reports.append({'component_id': int(component_id), 'num_input_points': num_input_points, 'num_kept_points': num_kept_points, 'num_removed_points': int(num_input_points - num_kept_points), 'num_protected_keep_points': int(np.sum(local_protected)), 'min_displacement': float(np.min(values)) if values.size > 0 else 0.0, 'mean_displacement': value_stats['mean'], 'p90_displacement': value_stats['p90'], 'max_displacement': value_stats['max']})
    summary: dict[str, Any] = {'enabled': True, 'threshold': float(threshold), 'keep_equal': bool(keep_equal), 'num_input_final_points': num_input_final_points, 'num_kept_points': int(np.sum(filtered_final_labels > 0)), 'num_removed_points': int(num_removed_points), 'num_protected_keep_points': int(np.sum(protected_keep_mask_bool & (filtered_final_labels > 0))), 'component_reports': component_filter_reports}
    return (filtered_final_labels, filtered_rejected_labels, summary)

def _run_covert_gtr_closing(points: np.ndarray, neighbor_indices: list[np.ndarray], sgcr_candidate_labels: np.ndarray, tscb_result: dict[str, Any], exp06: Any, profiler: RuntimeProfiler | None=None, *, gtr_params_override: dict[str, Any] | None=None) -> dict[str, Any]:
    """Run the exact GTR morphology path without restoration or decisions.

    This helper owns the production closing implementation used by the full
    GTR pipeline.  Its ``closed_labels_core`` is the candidate state after
    per-component morphology and overlap ownership, before post-closing
    expansion and before biharmonic restoration.
    """
    profiler = _resolve_profiler(profiler)
    include_substages = bool(EXP17_PROFILE_INCLUDE_SUBSTAGES)
    _tok = _begin_stage(profiler, '04_01_gtr_resolve_params', include=include_substages)
    gtr_params = dict(gtr_params_override)
    dilation_hops = int(gtr_params['dilation_hops'])
    erosion_hops = int(gtr_params['erosion_hops'])
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_02_gtr_resolve_candidates', include=include_substages)
    candidate_labels = np.asarray(sgcr_candidate_labels, dtype=np.int32).reshape(-1)
    num_points = int(points.shape[0])
    if candidate_labels.shape != (num_points,):
        raise ValueError(f'sgcr_candidate_labels must have one entry per point; got {candidate_labels.shape} for {num_points} points.')
    components = _extract_labeled_components_from_labels(candidate_labels)
    all_original_candidate_mask = candidate_labels > 0
    closing_mode = str(gtr_params['closing_mode']).strip().lower()
    closing_overlap_resolution = str(gtr_params['closing_overlap_resolution']).strip().lower()
    seed_component_labels_for_closing: np.ndarray | None = None
    seed_label_summary_for_closing: dict[str, Any] | None = None
    priority_result: dict[str, Any] = {'component_priority': {}, 'component_priority_source': {}, 'component_priority_seed_component_id': {}, 'component_priority_reports': [], 'sv_source': 'n/a'}
    sv_source_for_closing = 'n/a'
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_03_gtr_closing', include=include_substages)
    if closing_mode == 'independent_no_block_sv_priority':
        (seed_component_labels_for_closing, seed_label_summary_for_closing) = _resolve_sgcr_seed_component_labels_for_consistency(tscb_result=tscb_result, neighbor_indices=neighbor_indices, num_points=num_points)
        (sv_values_for_closing, sv_source_for_closing) = _resolve_sv_values_for_closing(tscb_result, num_points)
        priority_result = _compute_closing_component_sv_priority(sgcr_candidate_labels=candidate_labels, seed_component_labels=seed_component_labels_for_closing, sv_values=sv_values_for_closing, sv_source=sv_source_for_closing)
        closing_reports: list[dict[str, Any]] = []
        for component in components:
            closing_reports.append(_close_one_sgcr_component(component_id=int(component['component_id']), component_indices=component['indices'], all_original_candidate_mask=all_original_candidate_mask, neighbor_indices=neighbor_indices, dilation_hops=dilation_hops, erosion_hops=erosion_hops, block_other_original_masks=False, remove_other_original_after_closing=False))
        overlap_result = _resolve_closed_mask_overlaps(closing_reports, num_points, component_priority=priority_result['component_priority'], overlap_resolution='highest_seed_sv_mean')
    elif closing_mode == 'legacy_block_other_nearest_distance':
        closing_reports = []
        for component in components:
            closing_reports.append(_close_one_sgcr_component(component_id=int(component['component_id']), component_indices=component['indices'], all_original_candidate_mask=all_original_candidate_mask, neighbor_indices=neighbor_indices, dilation_hops=dilation_hops, erosion_hops=erosion_hops, block_other_original_masks=True, remove_other_original_after_closing=True))
        overlap_result = _resolve_closed_mask_overlaps(closing_reports, num_points, overlap_resolution='nearest_original_graph_distance')
        closing_overlap_resolution = 'nearest_original_graph_distance'
    else:
        raise ValueError(f"Unsupported EXP17_GTR_CLOSING_MODE={closing_mode!r}; expected 'independent_no_block_sv_priority' or 'legacy_block_other_nearest_distance'.")
    closed_labels_core = np.asarray(overlap_result['closed_labels'], dtype=np.int32).reshape(-1)
    closed_mask_core = closed_labels_core > 0
    closing_summary = {'mode': str(closing_mode), 'overlap_resolution': str(overlap_result['summary'].get('overlap_resolution', closing_overlap_resolution)), 'sv_source': str(sv_source_for_closing), 'num_components': int(overlap_result['summary'].get('num_components', len(components))), 'num_overlap_points': int(overlap_result['summary'].get('num_overlap_points', 0)), 'tie_break_rule': str(overlap_result['summary'].get('tie_break_rule', 'higher_sv_priority_then_shorter_dilation_distance_then_smaller_component_id' if closing_mode == 'independent_no_block_sv_priority' else 'shorter_dilation_distance_then_smaller_component_id')), **{key: value for (key, value) in overlap_result['summary'].items() if key not in {'mode', 'overlap_resolution', 'sv_source', 'num_components', 'num_overlap_points', 'tie_break_rule'}}}
    post_expand_result = _post_expand_closed_labels_after_closing(closed_labels_core=closed_labels_core, original_candidate_labels=candidate_labels, neighbor_indices=neighbor_indices, expand_hops=int(gtr_params.get('post_closing_expand_hops', EXP17_GTR_POST_CLOSING_EXPAND_HOPS)), block_other_original_masks=bool(gtr_params.get('post_closing_expand_block_other_original_masks', EXP17_GTR_POST_CLOSING_EXPAND_BLOCK_OTHER_ORIGINAL_MASKS)), overlap_mode=str(gtr_params.get('post_closing_expand_overlap_mode', EXP17_GTR_POST_CLOSING_EXPAND_OVERLAP_MODE)))
    closed_labels = np.asarray(post_expand_result['closed_labels'], dtype=np.int32).reshape(-1)
    post_closing_expand_distance = np.asarray(post_expand_result['post_expand_distance'], dtype=np.int32).reshape(-1)
    post_closing_expand_summary = post_expand_result['summary']
    _end_stage(profiler, _tok)
    return {'seed_mask': all_original_candidate_mask, 'seed_labels': candidate_labels, 'closed_mask_core': closed_mask_core, 'closed_labels_core': closed_labels_core, 'closed_mask': closed_labels > 0, 'closed_labels': closed_labels, 'post_closing_expand_distance': post_closing_expand_distance, 'post_closing_expand_summary': post_closing_expand_summary, 'component_closing_reports': closing_reports, 'closing_mode': str(closing_mode), 'closing_overlap_resolution': str(closing_summary.get('overlap_resolution', closing_overlap_resolution)), 'closing_component_priority_reports': priority_result['component_priority_reports'], 'closing_sv_source': str(sv_source_for_closing), 'closing_overlap_summary': overlap_result['summary'], 'closing_summary': closing_summary, 'resolved_gtr_params': gtr_params, 'components': components, 'seed_component_labels_for_closing': seed_component_labels_for_closing, 'seed_label_summary_for_closing': seed_label_summary_for_closing, 'summary': {'method': 'exp17_gtr_closing_only_from_sgcr', 'restoration_executed': False, 'num_sgcr_components': int(len(components)), 'num_closed_core_components': int(len(_extract_labeled_components_from_labels(closed_labels_core))), 'num_closed_core_points': int(np.sum(closed_mask_core)), 'num_closed_points_after_post_expand': int(np.sum(closed_labels > 0)), 'resolved_gtr_params': gtr_params, 'closing': closing_summary, 'post_closing_expand': post_closing_expand_summary}}

def _run_covert_gtr_core(points: np.ndarray, adjacency_dist: sp.csr_matrix, neighbor_indices: list[np.ndarray], sgcr_candidate_labels: np.ndarray, tscb_result: dict[str, Any], exp06: Any, profiler: RuntimeProfiler | None=None, *, local_normal_u_cleanup_mask: np.ndarray | None=None, local_strong_normal_u_mask: np.ndarray | None=None, local_weak_normal_u_mask: np.ndarray | None=None, local_normal_u_cleanup_enabled: bool=False, local_normal_u_cleanup_mode: str=EXP17_LOCAL_NORMAL_U_CLEANUP_DEFAULT_MODE, local_normal_u_cleanup_metadata: dict[str, Any] | None=None, gtr_seed_consistency_enabled: bool | None=None, local_normal_reference_enabled: bool | None=None, gtr_candidate_classifier_mode: str | None=None, gtr_params_override: dict[str, Any] | None=None, production_mainline: bool=False) -> dict[str, Any]:
    profiler = _resolve_profiler(profiler)
    include_substages = bool(EXP17_PROFILE_INCLUDE_SUBSTAGES)
    seed_consistency_requested = False
    closing_result = _run_covert_gtr_closing(points=points, neighbor_indices=neighbor_indices, sgcr_candidate_labels=sgcr_candidate_labels, tscb_result=tscb_result, exp06=exp06, profiler=profiler, gtr_params_override=gtr_params_override)
    gtr_params = closing_result['resolved_gtr_params']
    candidate_classifier_mode = str(gtr_params['candidate_classifier_mode']).strip().lower()
    local_normal_reference_requested = bool(gtr_params['local_normal_reference_enabled'])
    if candidate_classifier_mode == 'adaptive_normal_rejection' and not local_normal_reference_requested:
        raise ValueError('adaptive_normal_rejection requires Local Normal Reference.')
    object_scale_result = _resolve_gtr_object_scale(
        tscb_result,
        required=(candidate_classifier_mode == 'adaptive_normal_rejection'),
    )
    object_scale = object_scale_result['object_scale']
    dilation_hops = int(gtr_params['dilation_hops'])
    erosion_hops = int(gtr_params['erosion_hops'])
    min_closed_component_size = int(gtr_params['min_closed_component_size'])
    restoration_weight = str(gtr_params['restoration_weight'])
    displacement_threshold = float(gtr_params['displacement_threshold'])
    use_stress_veto = bool(gtr_params['use_stress_veto'])
    stress_threshold = float(gtr_params['stress_threshold'])
    hks_support_expand_hops = int(gtr_params['hks_open_boundary_support_expand_hops'])
    components = closing_result['components']
    all_original_candidate_mask = closing_result['seed_mask']
    num_points = int(points.shape[0])
    closed_labels_core = closing_result['closed_labels_core']
    closed_mask_core = closing_result['closed_mask_core']
    closed_labels = closing_result['closed_labels']
    closed_mask = closing_result['closed_mask']
    post_closing_expand_distance = closing_result['post_closing_expand_distance']
    post_closing_expand_summary = closing_result['post_closing_expand_summary']
    closing_reports = closing_result['component_closing_reports']
    closing_summary = closing_result['closing_summary']
    sv_source_for_closing = closing_result['closing_sv_source']
    overlap_summary = closing_result['closing_overlap_summary']
    closing_priority_reports = closing_result['closing_component_priority_reports']
    seed_component_labels_for_closing = closing_result['seed_component_labels_for_closing']
    seed_label_summary_for_closing = closing_result['seed_label_summary_for_closing']
    _tok = _begin_stage(profiler, '04_04_gtr_hks_open_boundary_support', include=include_substages)
    hks_open_boundary_support_mask_base = np.zeros(points.shape[0], dtype=bool)
    hks_open_boundary_support_expand_summary: dict[str, Any] = {'expand_hops': hks_support_expand_hops, 'base_point_count': 0, 'expanded_point_count': 0, 'added_point_count': 0, 'skipped': True, 'reason': 'hks_open_boundary_support_disabled'}
    if bool(gtr_params['use_hks_open_boundary_as_restoration_support']):
        hks_open_boundary_support_mask_base = _resolve_exp14_hks_open_boundary_support_mask(tscb_result=tscb_result, num_points=points.shape[0], use_expanded=bool(gtr_params['hks_open_boundary_support_use_expanded']))
        (hks_open_boundary_support_mask, hks_open_boundary_support_expand_summary) = _expand_hks_open_boundary_support_mask(hks_open_boundary_support_mask_base, neighbor_indices, expand_hops=hks_support_expand_hops)
    else:
        hks_open_boundary_support_mask = hks_open_boundary_support_mask_base.copy()
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_05_gtr_laplacian_build', include=include_substages)
    restoration_laplacian = _build_restoration_laplacian(adjacency_dist, restoration_weight)
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_06_gtr_restoration_total', include=include_substages)
    if bool(gtr_params['restore_per_component']):
        restoration_result = _restore_points_per_closed_component(points=points, restoration_laplacian=restoration_laplacian, closed_labels=closed_labels, hks_open_boundary_support_mask=hks_open_boundary_support_mask, gtr_params=gtr_params, profiler=profiler if include_substages else None)
        restored_points = restoration_result['restored_points']
        restoration_unknown_mask = restoration_result['restoration_unknown_mask']
        unknown_mask = restoration_unknown_mask
        support_inside_closed_mask = restoration_result['support_inside_closed_mask']
        support_inside_closed_labels = restoration_result['support_inside_closed_labels']
        solver_summary = restoration_result['solver_summary']
        component_restoration_reports = restoration_result['component_reports']
        component_restored_labels = restoration_result['component_restored_labels']
    else:
        support_inside_closed_mask = closed_mask & hks_open_boundary_support_mask
        support_inside_closed_labels = np.where(support_inside_closed_mask, closed_labels, 0).astype(np.int32, copy=False)
        if bool(gtr_params['use_hks_open_boundary_as_restoration_support']) and bool(gtr_params['exclude_open_boundary_support_from_unknown']):
            restoration_unknown_mask = closed_mask & ~hks_open_boundary_support_mask
        else:
            restoration_unknown_mask = closed_mask.copy()
        unknown_mask = restoration_unknown_mask
        (restored_points, solver_summary) = _restore_points_by_mode(points=points, restoration_laplacian=restoration_laplacian, unknown_mask=unknown_mask, gtr_params=gtr_params)
        solver_summary = dict(solver_summary)
        solver_summary['restore_per_component'] = False
        solver_summary['solver_scope'] = 'full_graph_global_union_unknown'
        component_restoration_reports = []
        component_restored_labels = np.where(restoration_unknown_mask, closed_labels, 0).astype(np.int32, copy=False)
        pass
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_07_gtr_displacement_stress', include=include_substages)
    if bool(gtr_params['restore_per_component']):
        displacement_stress = _compute_componentwise_displacement_and_stress(points=points, restored_points=restored_points, closed_labels=closed_labels, restoration_unknown_mask=restoration_unknown_mask, neighbor_indices=neighbor_indices)
    else:
        displacement_stress = _compute_displacement_and_stress(points=points, restored_points=restored_points, unknown_mask=unknown_mask, neighbor_indices=neighbor_indices)
    displacement_magnitude = displacement_stress['displacement_magnitude']
    stress_proxy = displacement_stress['stress_proxy']
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_08_gtr_ds_quadrant', include=include_substages)
    ds_quadrant_result = _compute_displacement_stress_quadrants(displacement_magnitude=displacement_magnitude, stress_proxy=stress_proxy, closed_mask=closed_mask, unknown_mask=unknown_mask, scope=str(gtr_params['ds_scope']), displacement_norm_threshold=float(gtr_params['ds_displacement_normalized_threshold']), stress_norm_threshold=float(gtr_params['ds_stress_normalized_threshold']), keep_equal=bool(gtr_params['ds_keep_equal']), enabled=bool(gtr_params['ds_quadrant_enabled']))
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_09_gtr_seed_consistency', include=include_substages)
    if seed_component_labels_for_closing is not None and seed_label_summary_for_closing is not None:
        seed_component_labels_for_consistency = seed_component_labels_for_closing
        seed_label_summary = seed_label_summary_for_closing
    else:
        (seed_component_labels_for_consistency, seed_label_summary) = _resolve_sgcr_seed_component_labels_for_consistency(tscb_result=tscb_result, neighbor_indices=neighbor_indices, num_points=points.shape[0])
    seed_consistency_enabled = bool(seed_consistency_requested and seed_label_summary.get('available', False))
    decision_labels_before_seed_consistency = closed_labels.copy()
    if bool(gtr_params['do_not_restitute_open_boundary_support']):
        decision_labels_before_seed_consistency[support_inside_closed_mask] = 0
    valid_support = np.asarray(ds_quadrant_result['high_displacement_high_stress_mask'] | ds_quadrant_result['low_displacement_high_stress_mask'], dtype=bool)
    empty_mask = np.zeros(num_points, dtype=bool)
    seed_consistency_result = {'enabled': False, 'filtered_candidate_labels': decision_labels_before_seed_consistency.copy(), 'removed_candidate_mask': empty_mask, 'removed_candidate_labels': np.zeros(num_points, dtype=np.int32), 'kept_candidate_labels': decision_labels_before_seed_consistency.copy(), 'removed_candidate_ids': [], 'valid_seed_support_mask': valid_support, 'seed_component_labels': seed_component_labels_for_consistency, 'component_reports': [], 'seed_component_reports': [], 'candidate_reports': [], 'summary': {'enabled': False, 'valid_percent': float(EXP17_GTR_SEED_CONSISTENCY_VALID_PERCENT), 'remove_mode': str(EXP17_GTR_SEED_CONSISTENCY_REMOVE_MODE), 'num_removed_candidate_components': 0, 'num_removed_candidate_points': 0}}
    seed_gtr_consistency_removed_labels = np.where(seed_consistency_result['removed_candidate_mask'], decision_labels_before_seed_consistency, 0).astype(np.int32, copy=False)
    (gtr_scores, gtr_score_source) = _build_gtr_scores_from_exp14(tscb_result, exp06)
    decision_labels = decision_labels_before_seed_consistency.copy()
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_09_01_local_normal_u_cleanup', include=bool(include_substages and local_normal_u_cleanup_enabled))
    local_normal_u_cleanup_result = _prepare_local_normal_u_cleanup(num_points=num_points, enabled=local_normal_u_cleanup_enabled, mode=local_normal_u_cleanup_mode, cleanup_mask=local_normal_u_cleanup_mask, strong_mask=local_strong_normal_u_mask, weak_mask=local_weak_normal_u_mask, metadata=local_normal_u_cleanup_metadata)
    cleanup_overall = local_normal_u_cleanup_result['overall']
    cleanup_mode = str(cleanup_overall['mode'])
    cleanup_effective = bool(cleanup_overall['effective_enabled'])
    cleanup_union_mask = np.asarray(local_normal_u_cleanup_result['local_normal_u_cleanup_mask'], dtype=bool)
    decision_labels_before_local_u_cleanup = decision_labels.copy()
    decision_labels_after_local_u_cleanup = decision_labels.copy()
    decision_applied_cleanup_mask = cleanup_union_mask & (decision_labels_before_local_u_cleanup > 0)
    if cleanup_effective and cleanup_mode == 'pre_component_classification':
        decision_labels_after_local_u_cleanup[decision_applied_cleanup_mask] = 0
    local_normal_u_cleanup_result['decision_labels_before_local_u_cleanup'] = decision_labels_before_local_u_cleanup
    local_normal_u_cleanup_result['decision_labels_after_local_u_cleanup'] = decision_labels_after_local_u_cleanup
    decision_labels_for_component_classification = decision_labels_after_local_u_cleanup if cleanup_effective and cleanup_mode == 'pre_component_classification' else decision_labels_before_local_u_cleanup
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_09_02_gtr_local_normal_reference', include=bool(include_substages and local_normal_reference_requested))
    local_normal_reference_result = build_gtr_local_normal_reference_result(
        enabled=local_normal_reference_requested,
        points=points,
        restoration_laplacian=restoration_laplacian,
        neighbor_indices=neighbor_indices,
        closed_labels=closed_labels,
        decision_labels_for_component_classification=decision_labels_for_component_classification,
        displacement_magnitude=displacement_magnitude,
        stress_proxy=stress_proxy,
        gtr_params=gtr_params,
    )
    local_normal_reference_result['consumed_by_candidate_classifier'] = bool(
        candidate_classifier_mode == 'adaptive_normal_rejection'
    )
    local_normal_reference_result['diagnostic_only'] = bool(
        candidate_classifier_mode == 'legacy_absolute'
    )
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_10_gtr_component_classification', include=include_substages)
    classifier_kwargs = {
        'seed_labels': sgcr_candidate_labels,
        'gtr_scores': gtr_scores,
        'displacement_magnitude': displacement_magnitude,
        'stress_proxy': stress_proxy,
        'min_closed_component_size': min_closed_component_size,
        'displacement_threshold': displacement_threshold,
        'use_stress_veto': use_stress_veto,
        'stress_threshold': stress_threshold,
        'classifier_mode': candidate_classifier_mode,
        'local_normal_reference_result': local_normal_reference_result,
        'object_scale': object_scale,
        'adaptive_relative_normal_max_ratio': float(gtr_params['adaptive_relative_normal_max_ratio']),
        'adaptive_min_large_fraction': float(gtr_params['adaptive_min_large_fraction']),
        'adaptive_max_shallow_displacement_normalized': float(gtr_params['adaptive_max_shallow_displacement_normalized']),
    }
    classification_before_local_u_cleanup = _classify_gtr_components(
        closed_labels=decision_labels_before_local_u_cleanup,
        **classifier_kwargs,
    )
    if cleanup_effective and cleanup_mode == 'pre_component_classification':
        classification_after_local_u_cleanup = _classify_gtr_components(
            closed_labels=decision_labels_after_local_u_cleanup,
            **classifier_kwargs,
        )
        classification_result = classification_after_local_u_cleanup
    else:
        classification_after_local_u_cleanup = classification_before_local_u_cleanup
        classification_result = classification_before_local_u_cleanup
    (candidate_cleanup_reports, cleanup_transitions) = _build_local_normal_u_candidate_audit(decision_labels_before=decision_labels_before_local_u_cleanup, decision_labels_after=decision_labels_after_local_u_cleanup, strong_mask=local_normal_u_cleanup_result['local_strong_normal_u_mask'], weak_mask=local_normal_u_cleanup_result['local_weak_normal_u_mask'], cleanup_mask=cleanup_union_mask if cleanup_effective else np.zeros(num_points, dtype=bool), displacement_magnitude=displacement_magnitude, stress_proxy=stress_proxy, gtr_scores=gtr_scores, classification_before=classification_before_local_u_cleanup, classification_after=classification_after_local_u_cleanup)
    local_normal_u_cleanup_result['candidate_reports'] = candidate_cleanup_reports
    local_normal_u_cleanup_result['classification_reports_before_cleanup'] = list(classification_before_local_u_cleanup.get('component_reports', []))
    local_normal_u_cleanup_result['classification_reports_after_cleanup'] = list(classification_after_local_u_cleanup.get('component_reports', []))
    cleanup_overall['num_gtr_candidates_touched'] = int(len(candidate_cleanup_reports))
    cleanup_overall.update(cleanup_transitions)
    if cleanup_effective and cleanup_mode == 'pre_component_classification':
        local_normal_u_cleanup_result['local_normal_u_cleanup_applied_mask'] = decision_applied_cleanup_mask
        cleanup_overall['applied_cleanup_point_count'] = int(np.sum(decision_applied_cleanup_mask))
    final_defect_labels_after_component_decision = np.asarray(classification_result['final_defect_labels'], dtype=np.int32).reshape(-1)
    rejected_labels_after_component_decision = np.asarray(classification_result['rejected_labels'], dtype=np.int32).reshape(-1)
    component_reports = classification_result['component_reports']
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_11_gtr_point_refinement', include=include_substages)
    if bool(gtr_params['do_not_restitute_open_boundary_support']):
        protected_keep_mask_for_point_filter = np.zeros_like(support_inside_closed_mask, dtype=bool)
    else:
        protected_keep_mask_for_point_filter = support_inside_closed_mask
    point_threshold_mode = str(gtr_params['point_displacement_threshold_mode']).strip().lower()
    (resolved_point_displacement_threshold, point_threshold_summary) = _resolve_point_displacement_threshold(points, mode=str(gtr_params['point_displacement_threshold_mode']), fixed_threshold=float(gtr_params['point_displacement_threshold']), adaptive_k=int(gtr_params['point_displacement_adaptive_k']), adaptive_alpha=float(gtr_params['point_displacement_adaptive_alpha']), adaptive_use_clamp=bool(gtr_params['point_displacement_adaptive_use_clamp']), adaptive_min=float(gtr_params['point_displacement_adaptive_min']), adaptive_max=float(gtr_params['point_displacement_adaptive_max']))
    point_filter_enabled = bool(gtr_params.get('use_point_displacement_filter', EXP17_GTR_USE_POINT_DISPLACEMENT_FILTER))
    if point_threshold_mode == 'ds_quadrant_remove_lowd_lows':
        (final_defect_labels_before_support_restitution, rejected_labels_before_support_restitution, point_filter_summary) = _apply_ds_quadrant_filter_to_accepted_labels(final_defect_labels=final_defect_labels_after_component_decision, rejected_labels=rejected_labels_after_component_decision, quadrant_labels=ds_quadrant_result['quadrant_labels'], low_displacement_low_stress_mask=ds_quadrant_result['low_displacement_low_stress_mask'], enabled=point_filter_enabled, protected_keep_mask=protected_keep_mask_for_point_filter)
    else:
        (final_defect_labels_before_support_restitution, rejected_labels_before_support_restitution, point_filter_summary) = _apply_point_displacement_filter_to_accepted_labels(final_defect_labels=final_defect_labels_after_component_decision, rejected_labels=rejected_labels_after_component_decision, displacement_magnitude=displacement_magnitude, threshold=float(resolved_point_displacement_threshold), keep_equal=bool(gtr_params.get('point_displacement_keep_equal', EXP17_GTR_POINT_DISPLACEMENT_KEEP_EQUAL)), enabled=point_filter_enabled, protected_keep_mask=protected_keep_mask_for_point_filter)
    point_filter_summary = dict(point_filter_summary)
    point_filter_summary['threshold_resolution'] = point_threshold_summary
    point_filter_summary['point_displacement_threshold_mode'] = point_threshold_mode
    if point_threshold_mode == 'ds_quadrant_remove_lowd_lows':
        point_filter_summary['threshold'] = None
        point_filter_summary['uses_ds_quadrant_filter'] = True
        point_filter_summary['uses_displacement_threshold'] = False
        resolved_point_displacement_threshold_for_result: float | None = None
    else:
        point_filter_summary['threshold'] = float(resolved_point_displacement_threshold)
        point_filter_summary['uses_ds_quadrant_filter'] = False
        point_filter_summary['uses_displacement_threshold'] = True
        resolved_point_displacement_threshold_for_result = float(resolved_point_displacement_threshold)
    final_defect_labels_without_local_u_cleanup = final_defect_labels_before_support_restitution.copy()
    if cleanup_effective and cleanup_mode == 'pre_component_classification':
        before_component_final = np.asarray(classification_before_local_u_cleanup['final_defect_labels'], dtype=np.int32).reshape(-1)
        before_component_rejected = np.asarray(classification_before_local_u_cleanup['rejected_labels'], dtype=np.int32).reshape(-1)
        if point_threshold_mode == 'ds_quadrant_remove_lowd_lows':
            (final_defect_labels_without_local_u_cleanup, _, _) = _apply_ds_quadrant_filter_to_accepted_labels(final_defect_labels=before_component_final, rejected_labels=before_component_rejected, quadrant_labels=ds_quadrant_result['quadrant_labels'], low_displacement_low_stress_mask=ds_quadrant_result['low_displacement_low_stress_mask'], enabled=point_filter_enabled, protected_keep_mask=protected_keep_mask_for_point_filter)
        else:
            (final_defect_labels_without_local_u_cleanup, _, _) = _apply_point_displacement_filter_to_accepted_labels(final_defect_labels=before_component_final, rejected_labels=before_component_rejected, displacement_magnitude=displacement_magnitude, threshold=float(resolved_point_displacement_threshold), keep_equal=bool(gtr_params.get('point_displacement_keep_equal', EXP17_GTR_POINT_DISPLACEMENT_KEEP_EQUAL)), enabled=point_filter_enabled, protected_keep_mask=protected_keep_mask_for_point_filter)
    if cleanup_effective and cleanup_mode == 'final_mask_only':
        final_cleanup_applied_mask = cleanup_union_mask & (final_defect_labels_before_support_restitution > 0)
        final_defect_labels_before_support_restitution = final_defect_labels_before_support_restitution.copy()
        final_defect_labels_before_support_restitution[final_cleanup_applied_mask] = 0
        local_normal_u_cleanup_result['local_normal_u_cleanup_applied_mask'] = final_cleanup_applied_mask
        cleanup_overall['applied_cleanup_point_count'] = int(np.sum(final_cleanup_applied_mask))
    final_cleanup_removed_mask = (final_defect_labels_without_local_u_cleanup > 0) & (final_defect_labels_before_support_restitution == 0)
    local_normal_u_cleanup_result['local_normal_u_cleanup_final_removed_mask'] = final_cleanup_removed_mask
    cleanup_overall['final_removed_point_count'] = int(np.sum(final_cleanup_removed_mask))
    final_defect_labels = final_defect_labels_before_support_restitution.copy()
    rejected_labels = rejected_labels_before_support_restitution.copy()
    support_restitution_summary = {'enabled': bool(gtr_params['use_hks_open_boundary_as_restoration_support']), 'do_not_restitute_open_boundary_support': bool(gtr_params['do_not_restitute_open_boundary_support']), 'support_used_for_restoration': bool(gtr_params['use_hks_open_boundary_as_restoration_support']), 'support_excluded_from_restoration_unknown': bool(gtr_params['exclude_open_boundary_support_from_unknown']), 'restituted_before_decision': bool(not gtr_params['do_not_restitute_open_boundary_support']), 'post_decision_restitution_applied': False, 'num_support_inside_closed_points': int(np.sum(support_inside_closed_mask)), 'num_support_inside_decision_points_before_seed_filter': int(np.sum(support_inside_closed_mask & (decision_labels_before_seed_consistency > 0))), 'num_support_inside_decision_points_after_seed_filter': int(np.sum(support_inside_closed_mask & (decision_labels_before_local_u_cleanup > 0))), 'reason': 'support overlap points are used as restoration support only and not returned to decision' if gtr_params['do_not_restitute_open_boundary_support'] else 'support overlap labels are restored before seed consistency and component classification'}
    final_defect_mask = final_defect_labels > 0
    rejected_mask = rejected_labels > 0
    _end_stage(profiler, _tok)
    _tok = _begin_stage(profiler, '04_12_gtr_result_packaging', include=include_substages)
    num_closed_components = len(_extract_labeled_components_from_labels(closed_labels))
    raw_support_point_count = int(np.sum(hks_open_boundary_support_mask_base))
    expanded_support_point_count = int(np.sum(hks_open_boundary_support_mask))
    _end_stage(profiler, _tok)
    return {'seed_mask': all_original_candidate_mask, 'seed_labels': np.asarray(sgcr_candidate_labels, dtype=np.int32).reshape(-1), 'closed_mask_core': closed_mask_core, 'closed_labels_core': closed_labels_core, 'closed_mask': closed_labels > 0, 'closed_labels': closed_labels, 'post_closing_expand_distance': post_closing_expand_distance, 'post_closing_expand_summary': post_closing_expand_summary, 'unknown_mask': unknown_mask, 'hks_open_boundary_support_mask_base': hks_open_boundary_support_mask_base, 'hks_open_boundary_support_mask_raw': hks_open_boundary_support_mask_base, 'hks_open_boundary_support_mask': hks_open_boundary_support_mask, 'hks_open_boundary_support_inside_closed_mask': support_inside_closed_mask, 'hks_open_boundary_support_inside_closed_labels': support_inside_closed_labels, 'restoration_unknown_mask': restoration_unknown_mask, 'component_restoration_reports': component_restoration_reports, 'component_restored_labels': component_restored_labels, 'decision_labels_before_seed_consistency': decision_labels_before_seed_consistency, 'decision_mask_before_seed_consistency': decision_labels_before_seed_consistency > 0, 'decision_labels_after_seed_consistency': decision_labels, 'decision_mask_after_seed_consistency': decision_labels > 0, 'decision_labels_before_local_u_cleanup': decision_labels_before_local_u_cleanup, 'decision_labels_after_local_u_cleanup': decision_labels_after_local_u_cleanup, 'decision_labels': decision_labels_for_component_classification, 'decision_mask': decision_labels_for_component_classification > 0, 'support_restitution_before_decision_mask': support_inside_closed_mask, 'support_restitution_before_decision_labels': support_inside_closed_labels, 'restored_points': restored_points, 'displacement_magnitude': displacement_magnitude, 'stress_proxy': stress_proxy, 'displacement_stress_quadrant_result': ds_quadrant_result, 'displacement_stress_quadrant_labels': ds_quadrant_result['quadrant_labels'], 'ds_high_displacement_high_stress_mask': ds_quadrant_result['high_displacement_high_stress_mask'], 'ds_low_displacement_high_stress_mask': ds_quadrant_result['low_displacement_high_stress_mask'], 'ds_high_displacement_low_stress_mask': ds_quadrant_result['high_displacement_low_stress_mask'], 'ds_low_displacement_low_stress_mask': ds_quadrant_result['low_displacement_low_stress_mask'], 'seed_gtr_consistency_result': seed_consistency_result, 'seed_gtr_consistency_removed_mask': seed_consistency_result['removed_candidate_mask'], 'seed_gtr_consistency_removed_labels': seed_gtr_consistency_removed_labels, 'seed_gtr_consistency_filtered_candidate_labels': seed_consistency_result['filtered_candidate_labels'], 'seed_gtr_consistency_valid_support_mask': seed_consistency_result['valid_seed_support_mask'], 'gtr_seed_consistency_enabled': bool(seed_consistency_requested), 'gtr_candidate_classifier_mode': candidate_classifier_mode, 'candidate_classifier_result': classification_result, 'local_normal_reference_result': local_normal_reference_result, 'gtr_scores': gtr_scores, 'gtr_score_source': gtr_score_source, 'local_normal_u_cleanup_result': local_normal_u_cleanup_result, 'local_strong_normal_u_mask': local_normal_u_cleanup_result['local_strong_normal_u_mask'], 'local_weak_normal_u_mask': local_normal_u_cleanup_result['local_weak_normal_u_mask'], 'local_normal_u_cleanup_mask': local_normal_u_cleanup_result['local_normal_u_cleanup_mask'], 'local_normal_u_cleanup_applied_mask': local_normal_u_cleanup_result['local_normal_u_cleanup_applied_mask'], 'local_normal_u_cleanup_final_removed_mask': local_normal_u_cleanup_result['local_normal_u_cleanup_final_removed_mask'], 'final_defect_labels_without_local_u_cleanup': final_defect_labels_without_local_u_cleanup, 'final_defect_mask': final_defect_mask, 'final_defect_labels': final_defect_labels, 'final_defect_labels_after_component_decision': final_defect_labels_after_component_decision, 'final_defect_labels_before_support_restitution': final_defect_labels_before_support_restitution, 'rejected_mask': rejected_mask, 'rejected_labels': rejected_labels, 'rejected_labels_after_component_decision': rejected_labels_after_component_decision, 'rejected_labels_before_support_restitution': rejected_labels_before_support_restitution, 'point_displacement_filter_summary': point_filter_summary, 'resolved_point_displacement_threshold': resolved_point_displacement_threshold_for_result, 'point_displacement_threshold_summary': point_threshold_summary, 'support_restitution_summary': support_restitution_summary, 'component_reports': component_reports, 'component_closing_reports': closing_reports, 'closing_mode': str(gtr_params['closing_mode']), 'closing_overlap_resolution': str(closing_summary.get('overlap_resolution', gtr_params['closing_overlap_resolution'])), 'closing_component_priority_reports': closing_priority_reports, 'closing_sv_source': str(sv_source_for_closing), 'closing_overlap_summary': overlap_summary, 'summary': {'method': 'exp17_exp14_sgcr_per_component_gtr', 'resolved_gtr_param_source': gtr_params['param_source'], 'exp06_override_applied': gtr_params['exp06_override_applied'], 'exp06_override_fields': list(gtr_params['exp06_override_fields']), 'resolved_gtr_params': gtr_params, 'gtr_seed_consistency_enabled': bool(seed_consistency_requested), 'gtr_seed_consistency_effective': bool(seed_consistency_enabled), 'gtr_candidate_classifier_mode': candidate_classifier_mode, 'gtr_local_normal_reference_enabled': bool(local_normal_reference_requested), 'restoration_mode': str(gtr_params['restoration_mode']), 'num_sgcr_components': len(components), 'num_closed_components': int(num_closed_components), 'num_closed_core_points': int(np.sum(closed_mask_core)), 'num_closed_points_after_post_expand': int(np.sum(closed_mask)), 'num_accepted_components': int(classification_result['num_accepted_components']), 'num_rejected_components': int(classification_result['num_rejected_components']), 'num_candidate_too_small_rejected': int(classification_result['num_candidate_too_small_rejected']), 'num_candidate_relative_normal_rejected': int(classification_result['num_candidate_relative_normal_rejected']), 'num_candidate_large_shallow_rejected': int(classification_result['num_candidate_large_shallow_rejected']), 'num_candidate_both_adaptive_rejected': int(classification_result['num_candidate_both_adaptive_rejected']), 'num_candidate_stress_veto_rejected': int(classification_result['num_candidate_stress_veto_rejected']), 'num_candidate_accepted': int(classification_result['num_candidate_accepted']), 'num_final_defect_points': int(np.sum(final_defect_mask)), 'num_final_defect_points_after_component_decision': int(np.sum(final_defect_labels_after_component_decision > 0)), 'num_final_defect_points_after_point_filter_before_support_restitution': int(np.sum(final_defect_labels_before_support_restitution > 0)), 'num_final_defect_points_after_support_restitution': int(np.sum(final_defect_mask)), 'gtr_score_source': gtr_score_source, 'point_displacement_filter': point_filter_summary, 'point_displacement_threshold': point_threshold_summary, 'support_restitution': support_restitution_summary, 'closing': closing_summary, 'post_closing_expand': post_closing_expand_summary, 'component_closing': [report['summary'] for report in closing_reports], 'closing_component_priority_reports': closing_priority_reports, 'restoration': {**solver_summary, 'restore_per_component': bool(gtr_params['restore_per_component']), 'solver_scope': str(solver_summary.get('solver_scope', 'full_graph_per_component' if gtr_params['restore_per_component'] else 'full_graph_global_union_unknown')), 'num_restored_components': int(solver_summary.get('num_restored_components', len([report for report in component_restoration_reports if not report.get('skipped', False)]))), 'num_skipped_components': int(solver_summary.get('num_skipped_components', len([report for report in component_restoration_reports if report.get('skipped', False)]))), 'component_reports': component_restoration_reports, 'num_closed_points': int(np.sum(closed_mask)), 'num_closed_core_points': int(np.sum(closed_mask_core)), 'num_closed_points_after_post_expand': int(np.sum(closed_mask)), 'num_post_closing_added_points': int(np.sum((closed_labels > 0) & (closed_labels_core == 0))), 'num_restoration_unknown_points': int(np.sum(restoration_unknown_mask)), 'raw_support_point_count': raw_support_point_count, 'expanded_support_point_count': expanded_support_point_count, 'num_hks_open_boundary_support_points': expanded_support_point_count, 'num_hks_open_boundary_support_inside_closed_points': int(np.sum(support_inside_closed_mask))}, 'hks_open_boundary_restoration_support': {'enabled': bool(gtr_params['use_hks_open_boundary_as_restoration_support']), 'use_expanded': bool(gtr_params['hks_open_boundary_support_use_expanded']), 'exp17_support_expand_hops': int(hks_support_expand_hops), 'exclude_from_restoration_unknown': bool(gtr_params['exclude_open_boundary_support_from_unknown']), 'exclude_from_unknown': bool(gtr_params['exclude_open_boundary_support_from_unknown']), 'exclude_from_decision': bool(gtr_params['exclude_open_boundary_support_from_decision']), 'do_not_restitute_open_boundary_support': bool(gtr_params['do_not_restitute_open_boundary_support']), 'restituted_before_decision': bool(not gtr_params['do_not_restitute_open_boundary_support']), 'post_decision_restitution_applied': False, 'raw_support_point_count': raw_support_point_count, 'base_support_point_count': raw_support_point_count, 'expanded_support_point_count': expanded_support_point_count, 'support_point_count': expanded_support_point_count, 'support_inside_closed_point_count': int(np.sum(support_inside_closed_mask)), 'support_inside_closed_labeled_point_count': int(np.sum(support_inside_closed_labels > 0)), 'support_inside_closed_component_ids': [int(x) for x in np.unique(support_inside_closed_labels) if int(x) > 0], 'closed_point_count': int(np.sum(closed_mask)), 'restoration_unknown_point_count': int(np.sum(restoration_unknown_mask)), 'decision_point_count': int(np.sum(decision_labels > 0)), 'source': 'hks_open_boundary_mask_expanded' if gtr_params['hks_open_boundary_support_use_expanded'] else 'hks_open_boundary_mask', 'exp17_support_expand': hks_open_boundary_support_expand_summary}, 'displacement_threshold': displacement_threshold, 'use_stress_veto': use_stress_veto, 'stress_threshold': stress_threshold, 'stress_definition': 'neighbor_displacement_gradient_proxy', 'displacement_stress_quadrants': ds_quadrant_result['summary'], 'seed_gtr_consistency_filter': {'production_enabled': bool(seed_consistency_requested), 'effective_enabled': bool(seed_consistency_enabled), 'seed_label_resolution': seed_label_summary, 'component_reports': seed_consistency_result['component_reports'], 'seed_component_reports': seed_consistency_result.get('seed_component_reports', []), 'candidate_reports': seed_consistency_result.get('candidate_reports', []), **seed_consistency_result['summary']}, 'local_normal_reference': {'enabled': bool(local_normal_reference_result['enabled']), 'diagnostic_only': bool(local_normal_reference_result['diagnostic_only']), 'consumed_by_candidate_classifier': bool(local_normal_reference_result.get('consumed_by_candidate_classifier', False)), 'num_candidates_total': int(local_normal_reference_result['num_candidates_total']), 'num_candidates_reference_valid': int(local_normal_reference_result['num_candidates_reference_valid']), 'num_candidates_reference_skipped': int(local_normal_reference_result['num_candidates_reference_skipped']), 'reason': local_normal_reference_result.get('reason')}, 'candidate_classifier': {'mode': candidate_classifier_mode, 'legacy_displacement_threshold': displacement_threshold, 'use_stress_veto': use_stress_veto, 'stress_threshold': stress_threshold, 'relative_normal_max_ratio': None, 'min_large_fraction': None, 'max_shallow_d_norm': None, 'object_scale': object_scale, 'object_scale_source': object_scale_result['source'], 'local_normal_reference_required': False, 'num_candidate_too_small_rejected': int(classification_result['num_candidate_too_small_rejected']), 'num_candidate_relative_normal_rejected': int(classification_result['num_candidate_relative_normal_rejected']), 'num_candidate_large_shallow_rejected': int(classification_result['num_candidate_large_shallow_rejected']), 'num_candidate_both_adaptive_rejected': int(classification_result['num_candidate_both_adaptive_rejected']), 'num_candidate_stress_veto_rejected': int(classification_result['num_candidate_stress_veto_rejected']), 'num_candidate_accepted': int(classification_result['num_candidate_accepted'])}, 'local_normal_u_cleanup': {**local_normal_u_cleanup_result['overall'], 'strong_normal_q_ids': local_normal_u_cleanup_result['strong_normal_q_ids'], 'weak_normal_q_ids': local_normal_u_cleanup_result['weak_normal_q_ids'], 'normal_y_ids': local_normal_u_cleanup_result['normal_y_ids'], 'strong_normal_u_components': local_normal_u_cleanup_result['strong_normal_u_components'], 'weak_normal_u_components': local_normal_u_cleanup_result['weak_normal_u_components'], 'candidate_reports': candidate_cleanup_reports}, 'dilation_hops': dilation_hops, 'erosion_hops': erosion_hops, 'component_reports': component_reports}}

def run_covert_gtr(*, points: np.ndarray, adjacency_dist: sp.csr_matrix, neighbor_indices: list[np.ndarray], candidate_labels: np.ndarray, frontend_adapter: dict[str, Any], config: Any, score_compat: Any, local_normal_u_cleanup_request: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen production GTR route with explicit semantic config."""
    params = build_covert_gtr_params(config)
    cleanup = local_normal_u_cleanup_request
    result = _run_covert_gtr_core(points=points, adjacency_dist=sp.csr_matrix(adjacency_dist).tocsr(), neighbor_indices=neighbor_indices, sgcr_candidate_labels=np.asarray(candidate_labels, dtype=np.int32), tscb_result=frontend_adapter, exp06=score_compat, local_normal_u_cleanup_mask=cleanup['local_normal_u_cleanup_mask'], local_strong_normal_u_mask=cleanup['local_strong_normal_u_mask'], local_weak_normal_u_mask=cleanup['local_weak_normal_u_mask'], local_normal_u_cleanup_enabled=bool(cleanup['enabled']), local_normal_u_cleanup_mode=str(cleanup['mode']), local_normal_u_cleanup_metadata=cleanup, gtr_seed_consistency_enabled=False, gtr_candidate_classifier_mode=str(params['candidate_classifier_mode']), local_normal_reference_enabled=bool(params['local_normal_reference_enabled']), gtr_params_override=params, production_mainline=True)
    result['summary']['candidate_classifier'].update({
        'relative_normal_max_ratio': float(params['adaptive_relative_normal_max_ratio']),
        'min_large_fraction': float(params['adaptive_min_large_fraction']),
        'max_shallow_d_norm': float(params['adaptive_max_shallow_displacement_normalized']),
        'local_normal_reference_required': bool(
            params['candidate_classifier_mode'] == 'adaptive_normal_rejection'
        ),
    })
    return result
