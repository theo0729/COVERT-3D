"""Private production runtime mechanically migrated from the frozen reference.

Only symbols reachable from the active consolidated call root are retained.
"""
from __future__ import annotations
import argparse
import csv
import heapq
import json
import sys
from collections.abc import Sequence
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from vast.boundary.ac_boundary import coerce_neighbor_indices, extract_ac_boundary_seed
from vast.boundary.boundary_expand import expand_boundary_mask
from vast.boundary.graph_closing import close_gaps
from vast.covert import _backbone
BACKBONE_FPS_MODE = 'exact_numba'
BACKBONE_FPS_PREFILTER_NUM_POINTS = 10000
BACKBONE_FPS_SEED = 0
BACKBONE_LOCAL_CONTRAST_QUERY_WORKERS = -1
LOCAL_RESPONSE_GRAPH_QUERY_WORKERS = -1
AC_ANGLE_THRESHOLD_DEG = 110.0
AC_MIN_NEIGHBORS = 8
AC_CLOSING_DILATION_HOPS = 0
AC_CLOSING_EROSION_HOPS = 0
AC_EXPAND_HOPS = 1
EXP21_AC_COMPONENT_DEBUG_ENABLE = True
MAX_AC_COMPONENT_GAP_AUDIT_HOPS = 5
EXP21_AC_CONFIG_OVERRIDE_SOURCE: str | None = None
COMPONENT_AWARE_SUPPRESSION_ENABLE = True
AC_COMPONENT_AWARE_SUPPRESSION_ENABLE: bool | None = True
PURE_HKS_COMPONENT_AWARE_SUPPRESSION_ENABLE: bool | None = None
PURE_HKS_NORMAL_SUPPRESSION_MODE: str | None = 'component_neutral'
PURE_HKS_NORMAL_SUPPRESSION_MODES = frozenset({'legacy_hard', 'component_neutral', 'component_soft', 'disabled'})
PURE_HKS_SOFT_ATTENUATION_FACTOR = 0.5
AC_COMPONENT_MIN_RELATIVE_TO_LARGEST = 0.5
AC_COMPONENT_GROUPING_MODE = 'gap_repaired'
AC_COMPONENT_GROUPING_MODES = frozenset({'raw', 'gap_repaired'})
AC_COMPONENT_GROUP_MAX_GAP_EDGES = 2
PURE_HKS_NORMAL_MIN_POINT_FRACTION = 0.02
PURE_HKS_NORMAL_MIN_RELATIVE_TO_LARGEST = 0.5
SUPPRESSION_COMBINATION_MODE_SPECS = {'legacy_ac__legacy_hks': (False, 'legacy_hard'), 'new_ac__legacy_hks': (True, 'legacy_hard'), 'legacy_ac__new_hks': (False, 'component_neutral'), 'new_ac__new_hks': (True, 'component_neutral'), 'legacy_ac__soft_hks': (False, 'component_soft'), 'new_ac__soft_hks': (True, 'component_soft'), 'legacy_ac__disabled_hks': (False, 'disabled'), 'new_ac__disabled_hks': (True, 'disabled')}
PURE_NONZERO_GROWTH_PERCENTILE = 10.0
PURE_DISPLAY_LOWER_PERCENTILE = 1.0
PURE_DISPLAY_UPPER_PERCENTILE = 99.0
PURE_STRUCTURE_ROOT_MIN_DISPLAY_PERCENT = 60.0
PURE_DISPLAY_LEVEL_HOP_BUDGETS = (1, 2, 3, 4, 6)
PURE_RANK_CUTS = (20.0, 40.0, 60.0, 80.0)
PURE_RESPONSE_EPS = 1e-12
BOUNDARY_RESPONSE_KEEP_WEIGHT = 0.01
VA_HKS_USE_MASK_ROOT_GROWTH = True
VA_HKS_NONZERO_GROWTH_PERCENTILE = 10.0
VA_HKS_DISPLAY_LOWER_PERCENTILE = 1.0
VA_HKS_DISPLAY_UPPER_PERCENTILE = 99.0
VA_HKS_DISPLAY_LEVEL_HOP_BUDGETS = (1, 2, 3, 4, 6)
SUPPRESSION_STRATEGY_LEGACY = 'legacy'
SV_FUSION_MODE = 'mask_sv'
SEMANTIC_GATE_NONZERO_PERCENTILE = 50.0
SEMANTIC_GATE_REPLACEMENT_VALUE = 0.0
SEMANTIC_CONFIDENCE_EXPONENT = 2.0
ORIGINAL_SV_GATE_PERCENTILE = 50.0
ORIGINAL_SV_GATE_REPLACEMENT_VALUE = 0.0
HIGH_CONFIDENCE_SEED_THRESHOLD = 0.5
SEED_CONFIDENCE_DISPLAY_LOWER_PERCENTILE = 1.0
SEED_CONFIDENCE_DISPLAY_UPPER_PERCENTILE = 99.0
SEED_CONFIDENCE_P80_COMPONENT_EXTRACTION_ENABLE = True
SEED_CONFIDENCE_COMPONENT_PERCENTILE = 97.0
SEED_CONFIDENCE_COMPONENT_CLOSING_DILATION_HOPS = 8
SEED_CONFIDENCE_COMPONENT_CLOSING_EROSION_HOPS = 8
SEED_CONFIDENCE_COMPONENT_CLOSING_KEEP_ORIGINAL = True
VIEW_Q_TO_VIEW_Z_CONSISTENCY_REPAIR_ENABLE = False
HIGH_CONFIDENCE_SEED_REMOVE_ON_PURE_HKS_GROWN_OPEN_BOUNDARY = True
SEED_COMPONENT_BOUNDARY_FILTER_ENABLE = True
SEED_COMPONENT_BOUNDARY_DILATION_HOPS = 2
SEED_COMPONENT_BOUNDARY_HIT_RATIO_THRESHOLD = 0.5
SEED_COMPONENT_MIN_SIZE = 2
SEED_COMPONENT_RED_SV_QUANTILE = 0.98
SEED_COMPONENT_LOCAL_GRAPH_ENABLE = True
SEED_COMPONENT_LOCAL_K = 8
SEED_COMPONENT_LOCAL_SCALE_K = 6
SEED_COMPONENT_LOCAL_RADIUS_FACTOR = 1.6
SGCR_LABEL_AWARE_MERGE_ENABLE = True
SGCR_LABEL_AWARE_MERGE_LABEL_PERCENTILE = 80.0
SGCR_LABEL_AWARE_MERGE_MIN_ENERGY_RATIO = 0.7
RESPONSE_QUANTILES = (0.0, 10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 90.0, 95.0, 99.0, 100.0)

def _as_vector(values: np.ndarray, *, name: str, length: int, dtype: Any) -> np.ndarray:
    """Return a one-dimensional array and validate its expected length."""
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f'{name} must have shape ({length},), got {array.shape}.')
    return array

def _as_bool_mask(values: np.ndarray, *, name: str, length: int) -> np.ndarray:
    """Return a validated boolean point mask."""
    return _as_vector(values, name=name, length=length, dtype=bool)

def apply_boundary_mask_attenuation(response: np.ndarray, boundary_mask: np.ndarray, keep_weight: float) -> np.ndarray:
    """Multiply only boundary-mask response values by one validated weight."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    boundary_mask = np.asarray(boundary_mask, dtype=bool).reshape(-1)
    if boundary_mask.shape != response.shape:
        raise ValueError(f'boundary_mask must have the same shape as response; got {boundary_mask.shape} and {response.shape}.')
    if not np.isfinite(response).all():
        raise ValueError('response must contain only finite values.')
    keep_weight = float(keep_weight)
    if not np.isfinite(keep_weight) or not 0.0 <= keep_weight <= 1.0:
        raise ValueError('keep_weight must be finite and lie in [0, 1].')
    suppressed = response.copy()
    suppressed[boundary_mask] *= keep_weight
    return suppressed

def normalize_response_minmax_01(response: np.ndarray) -> np.ndarray:
    """Normalize one finite response vector to [0, 1] using its exact min/max."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    if response.size == 0:
        raise ValueError('response must contain at least one value.')
    if not np.isfinite(response).all():
        raise ValueError('response must contain only finite values.')
    response_min = float(np.min(response))
    response_max = float(np.max(response))
    if response_max <= response_min:
        return np.zeros_like(response, dtype=np.float64)
    return np.clip((response - response_min) / (response_max - response_min + 1e-12), 0.0, 1.0).astype(np.float64, copy=False)

def normalize_response_q1_q99_with_nonzero_upper_fallback(response: np.ndarray, *, p_low: float, p_high: float) -> tuple[np.ndarray, float, float, float, bool]:
    """Q-percentile normalize, replacing a zero upper bound by min positive.

    The normal path deliberately matches exp14's Q1-Q99 normalization.  On the
    fallback path, the effective upper bound is the smallest positive score;
    therefore every positive score clips to 1 while exact zeros remain 0.
    """
    values = np.asarray(response, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError('response must contain at least one value.')
    if not np.isfinite(values).all():
        raise ValueError('response must contain only finite values.')
    lower_percentile = float(p_low)
    upper_percentile = float(p_high)
    if not np.isfinite(lower_percentile) or not np.isfinite(upper_percentile) or (not 0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError('p_low and p_high must satisfy 0 <= p_low < p_high <= 100.')
    q_low = float(np.percentile(values, lower_percentile))
    raw_q_high = float(np.percentile(values, upper_percentile))
    effective_q_high = raw_q_high
    fallback_applied = False
    if raw_q_high == 0.0:
        positive_values = values[values > 0.0]
        if positive_values.size > 0:
            effective_q_high = float(np.min(positive_values))
            fallback_applied = True
    if effective_q_high <= q_low:
        normalized = np.zeros(values.shape, dtype=np.float64)
    elif fallback_applied:
        normalized = np.clip((values - q_low) / (effective_q_high - q_low), 0.0, 1.0).astype(np.float64, copy=False)
    else:
        normalized = np.clip((values - q_low) / (effective_q_high - q_low + 1e-12), 0.0, 1.0).astype(np.float64, copy=False)
    return (normalized, q_low, raw_q_high, effective_q_high, fallback_applied)

def extract_seed_confidence_percentile_components(seed_confidence: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, *, enabled: bool, percentile: float=80.0) -> dict[str, Any]:
    """Extract strict all-point-percentile components from raw View-E scores.

    Spatial contact is defined by an edge in Exp21's existing global mutual-KNN
    graph.  The function is diagnostic-only: callers decide whether the result
    participates in any later pipeline stage.
    """
    scores = np.asarray(seed_confidence, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError('seed_confidence must contain at least one value.')
    if not np.isfinite(scores).all():
        raise ValueError('seed_confidence must contain only finite values.')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be True or False.')
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError('percentile must be finite and lie in [0, 100].')
    num_points = int(scores.size)
    neighbor_list = coerce_neighbor_indices(neighbor_indices, num_points)
    valid_mask = np.zeros(num_points, dtype=bool)
    component_labels = np.zeros(num_points, dtype=np.int32)
    components: list[np.ndarray] = []
    threshold: float | None = None
    if bool(enabled):
        threshold = float(np.percentile(scores, percentile))
        valid_mask = np.asarray(scores > threshold, dtype=bool)
        visited = np.zeros(num_points, dtype=bool)
        for start_index_raw in np.flatnonzero(valid_mask):
            start_index = int(start_index_raw)
            if visited[start_index]:
                continue
            queue: deque[int] = deque([start_index])
            visited[start_index] = True
            component: list[int] = []
            while queue:
                point_index = queue.popleft()
                component.append(point_index)
                for neighbor_index_raw in neighbor_list[point_index]:
                    neighbor_index = int(neighbor_index_raw)
                    if valid_mask[neighbor_index] and (not visited[neighbor_index]):
                        visited[neighbor_index] = True
                        queue.append(neighbor_index)
            components.append(np.asarray(component, dtype=np.int64))
        for (component_id, component) in enumerate(components, start=1):
            component_labels[component] = int(component_id)
    component_sizes = [int(component.size) for component in components]
    summary = {'enabled': bool(enabled), 'pipeline_effect': 'diagnostic_only_no_downstream_change', 'score_source': 'raw_seed_confidence_shown_in_view_E', 'threshold_source': 'all_point_seed_confidence_percentile', 'percentile': percentile, 'threshold': threshold, 'comparison_for_keep': 'seed_confidence > threshold', 'spatial_contact_graph': 'exp14_global_mutual_knn_neighbor_indices', 'valid_point_count': int(np.sum(valid_mask)), 'valid_point_ratio': float(np.mean(valid_mask)), 'component_count': int(len(components)), 'component_size_min': min(component_sizes) if component_sizes else 0, 'component_size_max': max(component_sizes) if component_sizes else 0, 'component_sizes': component_sizes}
    return {'enabled': bool(enabled), 'percentile': percentile, 'threshold': threshold, 'valid_mask': valid_mask, 'components': components, 'component_labels': component_labels, 'summary': summary}

def build_marker_consistent_view_z(raw_view_z: dict[str, Any], view_q_components: list[np.ndarray], retained_q_seed_mask: np.ndarray, global_neighbor_indices: list[np.ndarray] | np.ndarray, q_component_neighbor_indices: list[np.ndarray] | np.ndarray, *, repair_enable: bool=False) -> dict[str, Any]:
    """Optionally repair View-Z support/edges to preserve retained Q markers.

    Raw View-Z is never changed.  Only violation samples add missing retained-Q
    points and already-authoritative, within-Q-component graph edges before a
    new View-Z componentization.  The strict Q-to-Z invariant remains active.
    """
    if not isinstance(repair_enable, (bool, np.bool_)):
        raise ValueError('repair_enable must be True or False.')
    raw_valid = np.asarray(raw_view_z['valid_mask'], dtype=bool).reshape(-1)
    raw_labels = np.asarray(raw_view_z['component_labels'], dtype=np.int32).reshape(-1)
    num_points = int(raw_valid.size)
    if raw_labels.shape != raw_valid.shape:
        raise ValueError('Raw View-Z mask and labels must have identical shapes.')
    if not np.array_equal(raw_labels > 0, raw_valid):
        raise ValueError('Raw View-Z labels must partition raw valid mask exactly.')
    q_mask = _as_bool_mask(retained_q_seed_mask, name='retained_q_seed_mask', length=num_points)
    global_neighbors = coerce_neighbor_indices(global_neighbor_indices, num_points)
    q_neighbors = coerce_neighbor_indices(q_component_neighbor_indices, num_points)
    q_labels = np.zeros(num_points, dtype=np.int32)
    normalized_q_components: list[np.ndarray] = []
    for (q_id, component_raw) in enumerate(view_q_components, start=1):
        component = np.asarray(component_raw, dtype=np.int64).reshape(-1)
        if component.size == 0:
            raise ValueError('View-Q components must not be empty.')
        if np.any((component < 0) | (component >= num_points)):
            raise ValueError('View-Q component contains an invalid point index.')
        if np.unique(component).size != component.size:
            raise ValueError('View-Q component contains duplicate point indices.')
        if np.any(q_labels[component] > 0):
            raise ValueError('View-Q components must be point-disjoint.')
        q_labels[component] = int(q_id)
        normalized_q_components.append(component)
    if not np.array_equal(q_labels > 0, q_mask):
        raise AssertionError('retained_q_seed_mask must equal the union of authoritative View-Q components.')
    offending: list[dict[str, Any]] = []
    missing_total = 0
    multi_count = 0
    max_raw_ids = 0
    for (q_id, component) in enumerate(normalized_q_components, start=1):
        component_labels = raw_labels[component]
        positive_ids = np.unique(component_labels[component_labels > 0])
        missing_count = int(np.sum(component_labels == 0))
        multi = bool(positive_ids.size > 1)
        missing = bool(missing_count > 0)
        missing_total += missing_count
        multi_count += int(multi)
        max_raw_ids = max(max_raw_ids, int(positive_ids.size))
        if multi or missing:
            offending.append({'q_id': int(q_id), 'q_size': int(component.size), 'raw_positive_z_ids': [int(value) for value in positive_ids], 'raw_positive_z_count': int(positive_ids.size), 'raw_missing_q_point_count': missing_count, 'multi_z_violation': multi, 'missing_z_violation': missing, 'old_mapping_exists': bool(positive_ids.size == 1 and (not missing)), 'repaired_z_id': None, 'repaired_z_size': None, 'repaired_w_size': None, 'repair_added_q_point_count_for_component': 0})
    repair_required = bool(offending)
    stage_enabled = bool(raw_view_z.get('enabled', True))
    repair_applied = bool(repair_enable and stage_enabled and repair_required)
    added_q_support = np.zeros(num_points, dtype=bool)
    augmentation_count = 0
    authoritative = raw_view_z
    if repair_applied:
        repaired_valid = raw_valid | q_mask
        added_q_support = repaired_valid & ~raw_valid
        if np.any(added_q_support & ~q_mask):
            raise AssertionError('View-Z repair added a non-Q support point.')
        augmented_sets = [set((int(value) for value in row)) for row in global_neighbors]
        original_sets = [set(row) for row in augmented_sets]
        added_edges: set[tuple[int, int]] = set()
        for point in np.flatnonzero(q_mask):
            left = int(point)
            q_id = int(q_labels[left])
            for neighbor_raw in q_neighbors[left]:
                right = int(neighbor_raw)
                if right == left or int(q_labels[right]) != q_id:
                    continue
                edge = (min(left, right), max(left, right))
                if right not in original_sets[left] or left not in original_sets[right]:
                    added_edges.add(edge)
                augmented_sets[left].add(right)
                augmented_sets[right].add(left)
        augmentation_count = len(added_edges)
        augmented_neighbors = [np.asarray(sorted(row), dtype=np.int64) for row in augmented_sets]
        component_state = label_mask_components(repaired_valid, augmented_neighbors)
        authoritative_labels = np.asarray(component_state['component_labels'], dtype=np.int32)
        authoritative = {'enabled': raw_view_z['enabled'], 'percentile': raw_view_z['percentile'], 'threshold': raw_view_z['threshold'], 'valid_mask': repaired_valid, 'components': component_state['components'], 'component_labels': authoritative_labels, 'summary': {**dict(raw_view_z['summary']), 'component_stage': 'marker_consistent_authoritative_view_z', 'raw_definition_unchanged': True, 'repair_support_addition': 'missing_retained_View_Q_points_only', 'repair_edge_addition': 'existing_within_same_View_Q_component_edges_only', 'valid_point_count': int(np.sum(repaired_valid)), 'component_count': int(len(component_state['components']))}}
        sizes = np.asarray([component.size for component in authoritative['components']], dtype=np.int64)
        offending_by_q = {int(row['q_id']): row for row in offending}
        for (q_id, component) in enumerate(normalized_q_components, start=1):
            containing = np.unique(authoritative_labels[component])
            if containing.size != 1 or int(containing[0]) <= 0:
                raise AssertionError('Marker-consistent View-Z repair failed: every View-Q component must map to exactly one positive authoritative View-Z component.')
            row = offending_by_q.get(q_id)
            if row is not None:
                z_id = int(containing[0])
                row['repaired_z_id'] = z_id
                row['repaired_z_size'] = int(sizes[z_id - 1])
                row['repair_added_q_point_count_for_component'] = int(np.sum(added_q_support[component]))
    authoritative_valid = np.asarray(authoritative['valid_mask'], dtype=bool)
    authoritative_labels = np.asarray(authoritative['component_labels'], dtype=np.int32)
    all_q_positive = bool(np.all(authoritative_labels[q_mask] > 0))
    exactly_one = True
    post_violation_count = 0
    for component in normalized_q_components:
        containing = np.unique(authoritative_labels[component])
        valid = bool(containing.size == 1 and int(containing[0]) > 0)
        exactly_one = exactly_one and valid
        post_violation_count += int(not valid)
    if repair_applied and (not all_q_positive or not exactly_one):
        raise AssertionError('Marker-consistent View-Z remains inconsistent after repair.')
    if repair_applied and (not np.array_equal(authoritative_valid & ~raw_valid, added_q_support)):
        raise AssertionError('Authoritative View-Z support delta is inconsistent.')
    audit = {'enabled': bool(repair_enable), 'view_z_stage_enabled': stage_enabled, 'repair_required': repair_required, 'repair_applied': repair_applied, 'raw_z_component_count': int(len(raw_view_z['components'])), 'raw_z_point_count': int(np.sum(raw_valid)), 'q_component_count': int(len(normalized_q_components)), 'retained_q_point_count': int(np.sum(q_mask)), 'q_components_with_multiple_raw_z': int(multi_count), 'q_components_with_missing_raw_z_points': int(sum((bool(row['missing_z_violation']) for row in offending))), 'missing_q_point_count_total': int(missing_total), 'multi_z_q_component_count': int(multi_count), 'maximum_raw_z_ids_per_q': int(max_raw_ids), 'added_q_support_point_count': int(np.sum(added_q_support)), 'added_q_support_fraction_of_q': float(np.sum(added_q_support) / np.sum(q_mask)) if np.any(q_mask) else 0.0, 'q_graph_edge_augmentation_count': int(augmentation_count), 'authoritative_z_component_count': int(len(authoritative['components'])), 'authoritative_z_point_count': int(np.sum(authoritative_valid)), 'z_component_count_delta': int(len(authoritative['components']) - len(raw_view_z['components'])), 'z_point_count_delta': int(np.sum(authoritative_valid) - np.sum(raw_valid)), 'all_q_points_in_positive_z': all_q_positive, 'every_q_maps_exactly_one_z': exactly_one, 'post_repair_violation_count': int(post_violation_count), 'raw_active_z_count': int(np.unique(raw_labels[q_mask][raw_labels[q_mask] > 0]).size), 'authoritative_active_z_count': int(np.unique(authoritative_labels[q_mask][authoritative_labels[q_mask] > 0]).size), 'raw_w_morphology_group_count': None, 'authoritative_w_morphology_group_count': None, 'offending_q_components': offending, 'gt_used': False}
    return {'raw_view_z': raw_view_z, 'authoritative_view_z': authoritative, 'added_q_support_mask': added_q_support, 'q_component_labels': q_labels, 'audit': audit}

def group_seed_confidence_components_for_closing_by_view_y(spatial_components: list[np.ndarray], spatial_component_labels: np.ndarray, valid_mask: np.ndarray, view_y_component_labels: np.ndarray, seed_mask: np.ndarray, *, enabled: bool) -> dict[str, Any]:
    """Build temporary closing groups from seed-containing View-Z components.

    View-Z labels and component identities are never changed.  Seedless View-Z
    components receive morphology-group ID 0 and are omitted before View-W
    morphology closing.
    """
    spatial_component_labels = np.asarray(spatial_component_labels, dtype=np.int32).reshape(-1)
    num_points = int(spatial_component_labels.size)
    valid_mask = _as_bool_mask(valid_mask, name='valid_mask', length=num_points)
    view_y_component_labels = _as_vector(view_y_component_labels, name='view_y_component_labels', length=num_points, dtype=np.int32)
    seed_mask = _as_bool_mask(seed_mask, name='seed_mask', length=num_points)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be True or False.')
    num_spatial_components = int(len(spatial_components))
    if np.any((spatial_component_labels < 0) | (spatial_component_labels > num_spatial_components)):
        raise ValueError('spatial_component_labels contain an invalid component ID.')
    if not np.array_equal(spatial_component_labels > 0, valid_mask):
        raise ValueError('spatial component labels must partition valid_mask exactly.')
    if bool(enabled) and np.any(spatial_component_labels[seed_mask] <= 0):
        raise AssertionError('Every retained View-Q seed point must belong to a positive View-Z component before seedless View-Z components are removed.')
    active_spatial_component_ids = np.unique(spatial_component_labels[seed_mask]) if bool(enabled) else np.zeros(0, dtype=np.int32)
    active_spatial_component_ids = np.asarray(active_spatial_component_ids[active_spatial_component_ids > 0], dtype=np.int32)
    active_spatial_component_id_set = {int(value) for value in active_spatial_component_ids}
    removed_spatial_component_ids = np.asarray([component_id for component_id in range(1, num_spatial_components + 1) if component_id not in active_spatial_component_id_set], dtype=np.int32)
    closing_input_mask = np.isin(spatial_component_labels, active_spatial_component_ids)
    parent = np.arange(num_spatial_components + 1, dtype=np.int32)

    def _find(component_id: int) -> int:
        component_id = int(component_id)
        while int(parent[component_id]) != component_id:
            parent[component_id] = parent[int(parent[component_id])]
            component_id = int(parent[component_id])
        return component_id

    def _union(component_a: int, component_b: int) -> None:
        root_a = _find(component_a)
        root_b = _find(component_b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b
    y_component_reports: list[dict[str, Any]] = []
    if bool(enabled):
        for view_y_component_id_raw in np.unique(view_y_component_labels):
            view_y_component_id = int(view_y_component_id_raw)
            if view_y_component_id <= 0:
                continue
            spatial_z_ids = np.unique(spatial_component_labels[(view_y_component_labels == view_y_component_id) & closing_input_mask])
            spatial_z_ids = spatial_z_ids[spatial_z_ids > 0]
            if spatial_z_ids.size > 1:
                anchor = int(spatial_z_ids[0])
                for other_id in spatial_z_ids[1:]:
                    _union(anchor, int(other_id))
            y_component_reports.append({'view_y_component_id': view_y_component_id, 'intersecting_spatial_view_z_component_ids': [int(value) for value in spatial_z_ids], 'causes_morphology_grouping': bool(spatial_z_ids.size > 1)})
    groups_by_root: dict[int, list[int]] = {}
    for spatial_component_id_raw in active_spatial_component_ids:
        spatial_component_id = int(spatial_component_id_raw)
        root = _find(spatial_component_id)
        groups_by_root.setdefault(root, []).append(spatial_component_id)
    ordered_groups = sorted(groups_by_root.values(), key=lambda component_ids: min(component_ids))
    merged_components: list[np.ndarray] = []
    merged_component_labels = np.zeros(num_points, dtype=np.int32)
    spatial_to_merged_component_id = np.zeros(num_spatial_components + 1, dtype=np.int32)
    merged_component_reports: list[dict[str, Any]] = []
    for (merged_component_id, spatial_component_ids) in enumerate(ordered_groups, start=1):
        merged_indices = np.unique(np.concatenate([np.asarray(spatial_components[spatial_component_id - 1], dtype=np.int64).reshape(-1) for spatial_component_id in spatial_component_ids])).astype(np.int64, copy=False)
        merged_components.append(merged_indices)
        merged_component_labels[merged_indices] = int(merged_component_id)
        spatial_to_merged_component_id[spatial_component_ids] = int(merged_component_id)
        intersecting_y_ids = np.unique(view_y_component_labels[merged_indices])
        intersecting_y_ids = intersecting_y_ids[intersecting_y_ids > 0]
        merged_component_reports.append({'morphology_group_id': int(merged_component_id), 'spatial_view_z_component_ids': [int(value) for value in spatial_component_ids], 'intersecting_view_y_component_ids': [int(value) for value in intersecting_y_ids], 'component_size': int(merged_indices.size), 'contains_multiple_spatial_components': bool(len(spatial_component_ids) > 1)})
    if not np.array_equal(merged_component_labels > 0, closing_input_mask):
        raise AssertionError('View-Y morphology grouping changed the seed-filtered closing mask.')
    return {'enabled': bool(enabled), 'closing_input_mask': closing_input_mask, 'active_spatial_view_z_component_ids': active_spatial_component_ids, 'removed_seedless_spatial_view_z_component_ids': removed_spatial_component_ids, 'morphology_group_components': merged_components, 'morphology_group_labels': merged_component_labels, 'spatial_to_morphology_group_id': spatial_to_merged_component_id, 'y_component_reports': y_component_reports, 'morphology_group_reports': merged_component_reports, 'summary': {'enabled': bool(enabled), 'rule': 'remove_seedless_View_Z_components_then_temporarily_group_the_remaining_components_for_closing_when_they_intersect_the_same_positive_View_Y_component', 'view_z_component_identity_preserved': True, 'grouping_applies_only_during_morphology_closing': True, 'view_z_display_and_component_labels_unchanged': True, 'seed_filter_applied_before_morphology_closing': True, 'only_seed_containing_view_z_components_closed': True, 'seedless_view_z_components_are_ordinary_background': True, 'transitive_grouping_used': True, 'spatial_contact_required_after_view_y_match': False, 'num_view_z_components': num_spatial_components, 'num_seed_containing_view_z_components': int(active_spatial_component_ids.size), 'num_seedless_view_z_components_removed_before_closing': int(removed_spatial_component_ids.size), 'active_view_z_component_ids': active_spatial_component_ids.tolist(), 'removed_seedless_view_z_component_ids': removed_spatial_component_ids.tolist(), 'num_morphology_groups': int(len(merged_components)), 'num_view_y_components_causing_grouping': int(sum((report['causes_morphology_grouping'] for report in y_component_reports))), 'spatial_to_morphology_group_id': spatial_to_merged_component_id[1:].tolist(), 'view_y_component_reports': y_component_reports, 'morphology_group_reports': merged_component_reports}}

def close_seed_confidence_components_independently(components: list[np.ndarray], neighbor_indices: list[np.ndarray] | np.ndarray, *, num_points: int, enabled: bool, dilation_hops: int=8, erosion_hops: int=8, keep_original: bool=True) -> dict[str, Any]:
    """Close each component alone and measure its maximum growth multiple.

    For each call to graph closing, the foreground mask contains exactly one
    component.  Points belonging to every other component are therefore
    indistinguishable from normal background points, matching GTR's independent
    no-block closing mode.
    """
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be True or False.')
    if not isinstance(keep_original, (bool, np.bool_)):
        raise ValueError('keep_original must be True or False.')
    if isinstance(num_points, (bool, np.bool_)) or not isinstance(num_points, (int, np.integer)):
        raise ValueError('num_points must be an integer.')
    num_points = int(num_points)
    if num_points < 1:
        raise ValueError('num_points must be at least 1.')
    dilation_hops = int(dilation_hops)
    erosion_hops = int(erosion_hops)
    if dilation_hops < 0 or erosion_hops < 0:
        raise ValueError('dilation_hops and erosion_hops must be non-negative.')
    neighbor_list = coerce_neighbor_indices(neighbor_indices, num_points)
    closed_components: list[np.ndarray] = []
    component_reports: list[dict[str, Any]] = []
    closed_claim_count = np.zeros(num_points, dtype=np.int32)
    closed_max_growth_multiple_by_point = np.zeros(num_points, dtype=np.float64)
    if bool(enabled):
        for (component_id, component_raw) in enumerate(components, start=1):
            component = np.asarray(component_raw, dtype=np.int64).reshape(-1)
            if component.size == 0:
                raise ValueError('components must not contain an empty component.')
            if np.any((component < 0) | (component >= num_points)):
                raise ValueError('components contain an out-of-range point index.')
            component_mask = np.zeros(num_points, dtype=bool)
            component_mask[component] = True
            (closed_mask, closing_summary) = close_gaps(component_mask, neighbor_list, dilation_hops=dilation_hops, erosion_hops=erosion_hops)
            if bool(keep_original):
                closed_mask |= component_mask
            closed_indices = np.flatnonzero(closed_mask).astype(np.int64, copy=False)
            original_size = int(component.size)
            closed_size = int(closed_indices.size)
            max_growth_multiple = float(closed_size / original_size)
            closed_components.append(closed_indices)
            closed_claim_count[closed_indices] += 1
            closed_max_growth_multiple_by_point[closed_indices] = np.maximum(closed_max_growth_multiple_by_point[closed_indices], max_growth_multiple)
            component_reports.append({'component_id': int(component_id), 'original_size': original_size, 'dilated_size': int(closing_summary['num_boundary_after_dilation']), 'eroded_size': int(closing_summary['num_boundary_after_closing']), 'closed_size': closed_size, 'max_growth_multiple': max_growth_multiple, 'formula': 'closed_size / original_size'})
    closed_union_mask = closed_claim_count > 0
    closed_overlap_mask = closed_claim_count > 1
    closed_visualization_component_labels = np.zeros(num_points, dtype=np.int32)
    for (component_id, closed_component) in enumerate(closed_components, start=1):
        unassigned = closed_visualization_component_labels[closed_component] == 0
        closed_visualization_component_labels[closed_component[unassigned]] = int(component_id)
    for (component_id, original_component) in enumerate(components, start=1):
        closed_visualization_component_labels[np.asarray(original_component, dtype=np.int64).reshape(-1)] = int(component_id)
    growth_multiples = np.asarray([report['max_growth_multiple'] for report in component_reports], dtype=np.float64)
    summary = {'enabled': bool(enabled), 'pipeline_effect': 'diagnostic_only_no_downstream_change', 'method': 'per_component_graph_dilation_then_erosion', 'independent_component_closing': True, 'simultaneous_union_closing_used': False, 'other_components_during_each_close': 'ordinary_background_points', 'other_component_blocking_used': False, 'dilation_hops': dilation_hops, 'erosion_hops': erosion_hops, 'keep_original_component': bool(keep_original), 'num_components': int(len(closed_components)), 'closed_union_point_count_for_diagnostics': int(np.sum(closed_union_mask)), 'independent_proposal_overlap_point_count': int(np.sum(closed_overlap_mask)), 'visualization_overlap_resolution': 'smaller_component_id_then_restore_original_Z_component_ids', 'max_growth_multiple_min': float(np.min(growth_multiples)) if growth_multiples.size else 0.0, 'max_growth_multiple_mean': float(np.mean(growth_multiples)) if growth_multiples.size else 0.0, 'max_growth_multiple_max': float(np.max(growth_multiples)) if growth_multiples.size else 0.0, 'component_reports': component_reports}
    return {'enabled': bool(enabled), 'closed_components': closed_components, 'component_reports': component_reports, 'max_growth_multiples': growth_multiples, 'closed_union_mask': closed_union_mask, 'closed_claim_count': closed_claim_count, 'closed_overlap_mask': closed_overlap_mask, 'closed_visualization_component_labels': closed_visualization_component_labels, 'closed_max_growth_multiple_by_point': closed_max_growth_multiple_by_point, 'summary': summary}

def close_seed_confidence_view_z_morphology_groups(view_z_components: list[np.ndarray], view_z_component_labels: np.ndarray, morphology_group_components: list[np.ndarray], spatial_to_morphology_group_id: np.ndarray, morphology_group_reports: list[dict[str, Any]], neighbor_indices: list[np.ndarray] | np.ndarray, *, num_points: int, enabled: bool, dilation_hops: int=8, erosion_hops: int=8, keep_original: bool=True) -> dict[str, Any]:
    """Close seed-filtered View-Y groups while preserving spatial View-Z IDs."""
    num_points = int(num_points)
    view_z_component_labels = _as_vector(view_z_component_labels, name='view_z_component_labels', length=num_points, dtype=np.int32)
    neighbor_list = coerce_neighbor_indices(neighbor_indices, num_points)
    spatial_to_group = np.asarray(spatial_to_morphology_group_id, dtype=np.int32).reshape(-1)
    num_view_z_components = int(len(view_z_components))
    num_morphology_groups = int(len(morphology_group_components))
    if spatial_to_group.size != num_view_z_components + 1:
        raise ValueError('spatial_to_morphology_group_id must include ID 0 plus one entry for every spatial View-Z component.')
    if spatial_to_group.size and int(spatial_to_group[0]) != 0:
        raise ValueError('spatial_to_morphology_group_id[0] must be 0.')
    if np.any((spatial_to_group[1:] < 0) | (spatial_to_group[1:] > num_morphology_groups)):
        raise ValueError('A View-Z component has an invalid morphology-group ID.')
    active_view_z_component_ids = np.flatnonzero(spatial_to_group > 0)
    active_view_z_component_ids = active_view_z_component_ids[active_view_z_component_ids > 0].astype(np.int32, copy=False)
    removed_view_z_component_ids = np.flatnonzero(spatial_to_group == 0)
    removed_view_z_component_ids = removed_view_z_component_ids[removed_view_z_component_ids > 0].astype(np.int32, copy=False)
    if len(morphology_group_reports) != num_morphology_groups:
        raise ValueError('morphology_group_reports must align with morphology groups.')
    reported_view_z_component_ids = sorted((int(component_id) for report in morphology_group_reports for component_id in report['spatial_view_z_component_ids']))
    if reported_view_z_component_ids != active_view_z_component_ids.tolist():
        raise ValueError('Morphology-group reports must cover each seed-containing View-Z component exactly once and no seedless View-Z components.')
    group_closing = close_seed_confidence_components_independently(morphology_group_components, neighbor_list, num_points=num_points, enabled=enabled, dilation_hops=dilation_hops, erosion_hops=erosion_hops, keep_original=keep_original)
    view_z_component_reports: list[dict[str, Any]] = []
    if bool(enabled):
        for view_z_component_id_raw in active_view_z_component_ids:
            view_z_component_id = int(view_z_component_id_raw)
            component_raw = view_z_components[view_z_component_id - 1]
            component = np.asarray(component_raw, dtype=np.int64).reshape(-1)
            morphology_group_id = int(spatial_to_group[view_z_component_id])
            group_report = group_closing['component_reports'][morphology_group_id - 1]
            group_members = morphology_group_reports[morphology_group_id - 1]['spatial_view_z_component_ids']
            max_growth_multiple = float(int(group_report['closed_size']) / int(component.size))
            view_z_component_reports.append({'component_id': int(view_z_component_id), 'original_size': int(component.size), 'morphology_group_id': morphology_group_id, 'morphology_group_view_z_component_ids': [int(value) for value in group_members], 'morphology_group_original_size': int(group_report['original_size']), 'morphology_group_dilated_size': int(group_report['dilated_size']), 'morphology_group_eroded_size': int(group_report['eroded_size']), 'morphology_group_closed_size': int(group_report['closed_size']), 'max_growth_multiple': max_growth_multiple, 'formula': 'morphology_group_closed_size / original_View_Z_component_size'})
    closed_visualization_component_labels = np.asarray(group_closing['closed_visualization_component_labels'], dtype=np.int32).copy()
    view_z_growth_multiples = np.asarray([report['max_growth_multiple'] for report in view_z_component_reports], dtype=np.float64)
    view_y_group_union_closing_used = bool(any((len(report['spatial_view_z_component_ids']) > 1 for report in morphology_group_reports)))
    summary = dict(group_closing['summary'])
    summary.update({'method': 'independent_view_y_morphology_group_closing', 'view_z_component_identity_preserved': True, 'same_view_y_components_closed_as_one_group': True, 'seed_filter_applied_before_morphology_closing': True, 'only_seed_containing_view_z_components_closed': True, 'seedless_view_z_components_are_ordinary_background': True, 'one_view_w_component_per_morphology_group': True, 'view_w_split_back_to_view_z_components': False, 'view_y_group_union_closing_used': view_y_group_union_closing_used, 'independent_component_closing': not view_y_group_union_closing_used, 'independent_morphology_group_closing': True, 'simultaneous_union_closing_used': view_y_group_union_closing_used, 'simultaneous_all_component_union_closing_used': False, 'num_components': int(active_view_z_component_ids.size), 'num_total_view_z_components': num_view_z_components, 'num_seed_containing_view_z_components': int(active_view_z_component_ids.size), 'num_seedless_view_z_components_removed_before_closing': int(removed_view_z_component_ids.size), 'active_view_z_component_ids': active_view_z_component_ids.tolist(), 'removed_seedless_view_z_component_ids': removed_view_z_component_ids.tolist(), 'num_morphology_groups': num_morphology_groups, 'num_view_w_components': num_morphology_groups, 'other_components_during_each_close': 'components_outside_the_active_morphology_group_are_ordinary_background_points', 'visualization_overlap_resolution': 'one_View_W_component_per_morphology_group_with_no_split_back_to_constituent_View_Z_IDs', 'max_growth_multiple_min': float(np.min(view_z_growth_multiples)) if view_z_growth_multiples.size else 0.0, 'max_growth_multiple_mean': float(np.mean(view_z_growth_multiples)) if view_z_growth_multiples.size else 0.0, 'max_growth_multiple_max': float(np.max(view_z_growth_multiples)) if view_z_growth_multiples.size else 0.0, 'component_reports': view_z_component_reports, 'morphology_group_reports': group_closing['component_reports']})
    return {'enabled': bool(enabled), 'active_view_z_component_ids': active_view_z_component_ids, 'removed_seedless_view_z_component_ids': removed_view_z_component_ids, 'closed_components': group_closing['closed_components'], 'component_reports': view_z_component_reports, 'morphology_group_closing_reports': group_closing['component_reports'], 'max_growth_multiples': view_z_growth_multiples, 'closed_union_mask': group_closing['closed_union_mask'], 'closed_claim_count': group_closing['closed_claim_count'], 'closed_overlap_mask': group_closing['closed_overlap_mask'], 'closed_visualization_component_labels': closed_visualization_component_labels, 'closed_max_growth_multiple_by_point': group_closing['closed_max_growth_multiple_by_point'], 'summary': summary}

def map_view_q_seed_components_to_view_z_components(view_q_components: list[np.ndarray], view_z_component_labels: np.ndarray, view_z_closing_reports: list[dict[str, Any]], *, num_points: int, enabled: bool) -> dict[str, Any]:
    """Annotate every View-Q seed component from its containing View-Z component."""
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be True or False.')
    num_points = int(num_points)
    view_z_component_labels = _as_vector(view_z_component_labels, name='view_z_component_labels', length=num_points, dtype=np.int32)
    closing_report_by_z_id = {int(report['component_id']): report for report in view_z_closing_reports}
    mapped_z_component_ids: list[int] = []
    max_growth_multiples: list[float] = []
    mapping_reports: list[dict[str, Any]] = []
    max_growth_multiple_by_point = np.zeros(num_points, dtype=np.float64)
    if bool(enabled):
        for (view_q_component_id, view_q_component_raw) in enumerate(view_q_components, start=1):
            view_q_component = np.asarray(view_q_component_raw, dtype=np.int64).reshape(-1)
            if view_q_component.size == 0:
                raise ValueError('View-Q components must not be empty.')
            containing_z_ids = np.unique(view_z_component_labels[view_q_component])
            if containing_z_ids.size != 1 or int(containing_z_ids[0]) <= 0:
                raise AssertionError(f'Every View-Q seed component must be a subset of exactly one positive View-Z component; View-Q component {view_q_component_id} maps to {containing_z_ids.tolist()}.')
            view_z_component_id = int(containing_z_ids[0])
            closing_report = closing_report_by_z_id.get(view_z_component_id)
            if closing_report is None:
                raise AssertionError(f'Missing independent-closing report for View-Z component {view_z_component_id}.')
            max_growth_multiple = float(closing_report['max_growth_multiple'])
            mapped_z_component_ids.append(view_z_component_id)
            max_growth_multiples.append(max_growth_multiple)
            max_growth_multiple_by_point[view_q_component] = max_growth_multiple
            mapping_reports.append({'view_q_seed_component_id': int(view_q_component_id), 'view_q_seed_component_size': int(view_q_component.size), 'view_z_component_id': view_z_component_id, 'view_z_component_original_size': int(closing_report['original_size']), 'view_q_is_subset_of_view_z': True, 'view_z_max_growth_multiple': max_growth_multiple, 'assigned_max_growth_multiple': max_growth_multiple})
    mapped_z_ids_array = np.asarray(mapped_z_component_ids, dtype=np.int32)
    growth_multiples_array = np.asarray(max_growth_multiples, dtype=np.float64)
    unique_z_ids = np.unique(mapped_z_ids_array)
    return {'enabled': bool(enabled), 'mapped_view_z_component_ids': mapped_z_ids_array, 'max_growth_multiples': growth_multiples_array, 'max_growth_multiple_by_point': max_growth_multiple_by_point, 'mapping_reports': mapping_reports, 'summary': {'enabled': bool(enabled), 'pipeline_effect': 'diagnostic_annotation_only', 'mapping': 'each_View_Q_seed_component_to_containing_View_Z_component', 'required_relation': 'View_Q_component_is_subset_of_View_Z_component', 'all_view_q_components_have_exactly_one_view_z_parent': bool(len(mapping_reports) == len(view_q_components) if bool(enabled) else True), 'num_view_q_components': int(len(mapping_reports)), 'num_referenced_view_z_components': int(unique_z_ids.size), 'view_z_component_reuse_count': int(len(mapped_z_component_ids) - unique_z_ids.size), 'mapping_reports': mapping_reports}}

def merge_grown_components_sequential_label_aware(grown_sources: list[dict[str, Any]], neighbor_indices: list[np.ndarray], raw_enhanced_sv: np.ndarray, *, enabled: bool, label_percentile: float, min_energy_ratio: float) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Build directed grown-component links using source-upstream P-labels.

    Every candidate arrow points from the higher-ranked component to the lower-
    ranked component.  Its source side uses the largest label among the source
    and all currently connected upstream ancestors; its target side always uses
    the target component's own label.  Accepted arrows are collapsed into weakly
    connected point components only after all decisions finish.
    """
    raw_enhanced_sv = np.asarray(raw_enhanced_sv, dtype=np.float64).reshape(-1)
    num_points = int(raw_enhanced_sv.size)
    if num_points == 0:
        raise ValueError('raw_enhanced_sv must contain at least one value.')
    if not np.isfinite(raw_enhanced_sv).all():
        raise ValueError('raw_enhanced_sv must contain only finite values.')
    if len(neighbor_indices) != num_points:
        raise ValueError('neighbor_indices must contain one row per point.')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be True or False.')
    label_percentile = float(label_percentile)
    min_energy_ratio = float(min_energy_ratio)
    if not np.isfinite(label_percentile) or not 0.0 <= label_percentile <= 100.0:
        raise ValueError('label_percentile must be finite and lie in [0, 100].')
    if not np.isfinite(min_energy_ratio) or not 0.0 <= min_energy_ratio <= 1.0:
        raise ValueError('min_energy_ratio must be finite and lie in [0, 1].')
    num_sources = int(len(grown_sources))
    source_point_sets: list[np.ndarray] = []
    source_component_ids: list[int] = []
    energy_labels = np.zeros(num_sources, dtype=np.float64)
    point_sources: list[list[int]] = [[] for _ in range(num_points)]
    for (source_index, source) in enumerate(grown_sources):
        grown_indices = np.unique(np.asarray(source['grown_indices'], dtype=np.int64).reshape(-1))
        if grown_indices.size == 0:
            raise ValueError('Every grown source must contain at least one point.')
        if np.any((grown_indices < 0) | (grown_indices >= num_points)):
            raise ValueError('A grown source contains an out-of-range point index.')
        source_component_id = int(source['source_seed_component_id'])
        source_point_sets.append(grown_indices)
        source_component_ids.append(source_component_id)
        energy_labels[source_index] = float(np.percentile(raw_enhanced_sv[grown_indices], label_percentile))
        for point_index in grown_indices:
            point_sources[int(point_index)].append(source_index)
    processing_order = sorted(range(num_sources), key=lambda source_index: (-float(energy_labels[source_index]), int(source_component_ids[source_index])))
    rank_by_source = {source_index: rank for (rank, source_index) in enumerate(processing_order, start=1)}
    adjacency: list[set[int]] = [set() for _ in range(num_sources)]
    for (source_index, grown_indices) in enumerate(source_point_sets):
        for point_index in grown_indices:
            neighbors = np.asarray(neighbor_indices[int(point_index)], dtype=np.int64).reshape(-1)
            if np.any((neighbors < 0) | (neighbors >= num_points)):
                raise ValueError('neighbor_indices contains an out-of-range index.')
            for neighbor_index in neighbors:
                for other_source in point_sources[int(neighbor_index)]:
                    if other_source == source_index:
                        continue
                    adjacency[source_index].add(other_source)
                    adjacency[other_source].add(source_index)
    adjacent_pairs = {(min(source_index, other_source), max(source_index, other_source)) for (source_index, neighbors) in enumerate(adjacency) for other_source in neighbors if source_index != other_source}
    tested_pairs: set[tuple[int, int]] = set()
    accepted_connections: list[dict[str, Any]] = []
    accepted_directed_edges: list[tuple[int, int]] = []
    merge_decisions: list[dict[str, Any]] = []
    blocked_adjacency_count = 0
    upstream_max_source = list(range(num_sources))
    if enabled:
        for source_index in processing_order:
            ordered_neighbors = sorted(adjacency[source_index], key=lambda other_source: rank_by_source[other_source])
            for other_source in ordered_neighbors:
                if rank_by_source[other_source] <= rank_by_source[source_index]:
                    continue
                pair = (min(source_index, other_source), max(source_index, other_source))
                if pair in tested_pairs:
                    continue
                tested_pairs.add(pair)
                source_upstream = int(upstream_max_source[source_index])
                target_existing_upstream = int(upstream_max_source[other_source])
                source_label = float(energy_labels[source_index])
                source_upstream_label = float(energy_labels[source_upstream])
                target_label = float(energy_labels[other_source])
                required_minimum = source_upstream_label * min_energy_ratio
                accepted = bool(target_label >= required_minimum)
                decision = {'from_ranked_component_id': int(rank_by_source[source_index]), 'to_ranked_component_id': int(rank_by_source[other_source]), 'from_energy_label': source_label, 'from_upstream_max_ranked_component_id': int(rank_by_source[source_upstream]), 'from_upstream_max_energy_label': source_upstream_label, 'to_energy_label': target_label, 'to_existing_upstream_max_ranked_component_id': int(rank_by_source[target_existing_upstream]), 'to_existing_upstream_max_energy_label': float(energy_labels[target_existing_upstream]), 'required_target_label': float(required_minimum), 'accepted': accepted}
                merge_decisions.append(decision)
                if accepted:
                    accepted_connections.append(dict(decision))
                    accepted_directed_edges.append((source_index, other_source))
                    if rank_by_source[source_upstream] < rank_by_source[target_existing_upstream]:
                        upstream_max_source[other_source] = source_upstream
                else:
                    blocked_adjacency_count += 1
    parent = list(range(num_sources))

    def _find_root(source_index: int) -> int:
        while parent[source_index] != source_index:
            parent[source_index] = parent[parent[source_index]]
            source_index = parent[source_index]
        return source_index

    def _union(source_a: int, source_b: int) -> None:
        root_a = _find_root(source_a)
        root_b = _find_root(source_b)
        if root_a == root_b:
            return
        if rank_by_source[root_a] <= rank_by_source[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b
    for (source_index, target_index) in accepted_directed_edges:
        _union(source_index, target_index)
    groups: dict[int, list[int]] = {}
    for source_index in range(num_sources):
        root = _find_root(source_index)
        groups.setdefault(root, []).append(source_index)
    ordered_roots = sorted(groups, key=lambda root: min((rank_by_source[index] for index in groups[root])))
    group_point_masks = {root: np.zeros(num_points, dtype=bool) for root in ordered_roots}
    overlap_point_count = 0
    overlap_points_resolved = 0
    candidate_points = np.asarray([bool(sources) for sources in point_sources], dtype=bool)
    for point_index in np.flatnonzero(candidate_points):
        point_index = int(point_index)
        covering_sources = point_sources[point_index]
        covering_roots = {_find_root(source) for source in covering_sources}
        if len(covering_sources) > 1:
            overlap_point_count += 1
        if len(covering_roots) == 1:
            owner_root = next(iter(covering_roots))
        else:
            overlap_points_resolved += 1
            owner_source = min(covering_sources, key=lambda source: (abs(float(energy_labels[source]) - float(raw_enhanced_sv[point_index])), rank_by_source[source]))
            owner_root = _find_root(owner_source)
        group_point_masks[owner_root][point_index] = True
    merged_components: list[np.ndarray] = []
    merged_group_metadata: list[dict[str, Any]] = []
    for root in ordered_roots:
        merged_indices = np.flatnonzero(group_point_masks[root]).astype(np.int64, copy=False)
        if merged_indices.size == 0:
            continue
        member_sources = sorted(groups[root], key=lambda source: rank_by_source[source])
        merged_components.append(merged_indices)
        merged_group_metadata.append({'merged_component_id': int(len(merged_components)), 'ranked_component_ids': [int(rank_by_source[source]) for source in member_sources], 'source_seed_component_ids': [int(source_component_ids[source]) for source in member_sources], 'grown_energy_labels': [float(energy_labels[source]) for source in member_sources], 'group_max_energy_label': float(max((energy_labels[source] for source in member_sources))), 'directed_connections': [{'from_ranked_component_id': int(rank_by_source[source]), 'to_ranked_component_id': int(rank_by_source[target])} for (source, target) in accepted_directed_edges if source in member_sources and target in member_sources]})
    source_grown_energy_labels = {int(source_component_ids[source]): float(energy_labels[source]) for source in range(num_sources)}
    source_ranked_component_ids = {int(source_component_ids[source]): int(rank_by_source[source]) for source in range(num_sources)}
    summary = {'enabled': bool(enabled), 'implementation': 'exp21_sequential_directed_upstream_ratio_merge', 'label_source': 'raw_unnormalized_enhanced_sv_over_grown_component', 'label_statistic': f'p{label_percentile:g}', 'label_percentile': float(label_percentile), 'min_energy_ratio': float(min_energy_ratio), 'merge_comparison': 'target_own_label >= source_upstream_max_label * min_energy_ratio', 'adjacency_rule': 'grown_component_one_hop_mutual_knn_contact', 'processing_order': 'descending_grown_energy_label_directed_arrows', 'connection_direction': 'higher_ranked_component -> lower_ranked_component', 'final_merge_rule': 'weakly_connected_components_of_accepted_arrows', 'num_grown_sources': int(num_sources), 'num_adjacent_source_pairs': int(len(adjacent_pairs)), 'num_tested_source_pairs': int(len(tested_pairs)), 'num_label_merge_edges': int(len(accepted_connections)), 'num_directed_connections': int(len(accepted_directed_edges)), 'num_blocked_label_adjacencies': int(blocked_adjacency_count), 'num_skipped_adjacencies_merge_disabled': int(0 if enabled else len(adjacent_pairs)), 'num_connected_groups': int(len(groups)), 'num_merged_groups_before_size_filter': int(len(merged_components)), 'overlap_point_count': int(overlap_point_count), 'overlap_points_resolved': int(overlap_points_resolved), 'source_grown_energy_labels': source_grown_energy_labels, 'source_ranked_component_ids': source_ranked_component_ids, 'accepted_merge_connections': accepted_connections, 'merge_decisions': merge_decisions, 'merged_source_groups': merged_group_metadata}
    return (merged_components, summary)

def build_seed_component_local_neighbor_indices(points: np.ndarray, seed_mask: np.ndarray, *, local_k: int, local_scale_k: int, local_radius_factor: float) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Build a sparse seed-only graph from local all-cloud neighbor queries.

    The KD-tree contains every working point, but queries are issued only for
    seed points.  An undirected seed edge survives when both endpoints rank one
    another within ``local_k`` and its length does not exceed the geometric mean
    of the endpoints' local sampling scales times ``local_radius_factor``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'points must have shape (N, 3), got {points.shape}.')
    if not np.isfinite(points).all():
        raise ValueError('points must contain only finite values.')
    num_points = int(points.shape[0])
    seed_mask = _as_bool_mask(seed_mask, name='seed_mask', length=num_points)
    if isinstance(local_k, (bool, np.bool_)) or not isinstance(local_k, (int, np.integer)):
        raise ValueError('local_k must be an integer.')
    if isinstance(local_scale_k, (bool, np.bool_)) or not isinstance(local_scale_k, (int, np.integer)):
        raise ValueError('local_scale_k must be an integer.')
    local_k = int(local_k)
    local_scale_k = int(local_scale_k)
    local_radius_factor = float(local_radius_factor)
    if local_k < 1:
        raise ValueError('local_k must be at least 1.')
    if local_scale_k < 1:
        raise ValueError('local_scale_k must be at least 1.')
    if not np.isfinite(local_radius_factor) or local_radius_factor <= 0.0:
        raise ValueError('local_radius_factor must be finite and positive.')
    seed_indices = np.flatnonzero(seed_mask).astype(np.int64, copy=False)
    neighbor_sets = [set() for _ in range(num_points)]
    local_scales = np.zeros(num_points, dtype=np.float64)
    query_neighbor_count = min(max(local_k, local_scale_k) + 1, num_points)
    if seed_indices.size > 0 and query_neighbor_count > 0:
        tree = cKDTree(points)
        try:
            (query_distances, query_indices) = tree.query(points[seed_indices], k=query_neighbor_count, workers=int(LOCAL_RESPONSE_GRAPH_QUERY_WORKERS))
        except TypeError:
            (query_distances, query_indices) = tree.query(points[seed_indices], k=query_neighbor_count)
        query_distances = np.asarray(query_distances, dtype=np.float64)
        query_indices = np.asarray(query_indices, dtype=np.int64)
        if query_distances.ndim == 1:
            query_distances = query_distances.reshape(seed_indices.size, -1)
            query_indices = query_indices.reshape(seed_indices.size, -1)
        candidate_indices: dict[int, np.ndarray] = {}
        candidate_sets: dict[int, set[int]] = {}
        for (row_index, seed_index_raw) in enumerate(seed_indices):
            seed_index = int(seed_index_raw)
            row_distances = query_distances[row_index]
            row_indices = query_indices[row_index]
            valid = np.isfinite(row_distances) & (row_indices >= 0) & (row_indices < num_points) & (row_indices != seed_index)
            row_distances = row_distances[valid]
            row_indices = row_indices[valid]
            local_candidates = row_indices[:local_k].astype(np.int64, copy=False)
            candidate_indices[seed_index] = local_candidates
            candidate_sets[seed_index] = {int(index) for index in local_candidates}
            positive_distances = row_distances[row_distances > 0.0]
            scale_distances = positive_distances[:local_scale_k]
            if scale_distances.size > 0:
                local_scales[seed_index] = float(np.median(scale_distances))
        directed_seed_candidate_count = 0
        mutual_seed_edge_count = 0
        removed_by_radius_count = 0
        kept_edge_count = 0
        for seed_index_raw in seed_indices:
            seed_index = int(seed_index_raw)
            for neighbor_index_raw in candidate_indices[seed_index]:
                neighbor_index = int(neighbor_index_raw)
                if not seed_mask[neighbor_index]:
                    continue
                directed_seed_candidate_count += 1
                if neighbor_index <= seed_index:
                    continue
                if seed_index not in candidate_sets.get(neighbor_index, set()):
                    continue
                mutual_seed_edge_count += 1
                edge_distance = float(np.linalg.norm(points[seed_index] - points[neighbor_index]))
                pair_scale = float(np.sqrt(local_scales[seed_index] * local_scales[neighbor_index]))
                edge_limit = float(local_radius_factor * pair_scale)
                if pair_scale <= 0.0 or edge_distance > edge_limit:
                    removed_by_radius_count += 1
                    continue
                neighbor_sets[seed_index].add(neighbor_index)
                neighbor_sets[neighbor_index].add(seed_index)
                kept_edge_count += 1
    else:
        directed_seed_candidate_count = 0
        mutual_seed_edge_count = 0
        removed_by_radius_count = 0
        kept_edge_count = 0
    neighbor_indices = [np.asarray(sorted(neighbors), dtype=np.int64) for neighbors in neighbor_sets]
    positive_seed_scales = local_scales[seed_mask & (local_scales > 0.0)]
    summary = {'enabled': True, 'mode': 'seed_query_local_mutual_knn_with_local_radius', 'all_point_graph_rebuilt': False, 'query_source': 'seed points queried against a KD-tree of all points', 'num_points': num_points, 'num_seed_points': int(seed_indices.size), 'local_k': local_k, 'local_scale_k': local_scale_k, 'local_radius_factor': local_radius_factor, 'query_neighbor_count_including_self_slot': int(query_neighbor_count), 'local_scale_stat': 'median of nearest positive all-cloud distances', 'edge_limit': 'distance(i,j) <= factor * sqrt(local_scale_i * local_scale_j)', 'mutual_rule': 'j in local_k(i) and i in local_k(j)', 'num_directed_seed_candidate_edges': int(directed_seed_candidate_count), 'num_mutual_seed_edges_before_radius': int(mutual_seed_edge_count), 'num_seed_edges_removed_by_radius': int(removed_by_radius_count), 'num_kept_undirected_seed_edges': int(kept_edge_count), 'num_seed_points_with_positive_local_scale': int(positive_seed_scales.size), 'local_scale_min': float(np.min(positive_seed_scales)) if positive_seed_scales.size else 0.0, 'local_scale_median': float(np.median(positive_seed_scales)) if positive_seed_scales.size else 0.0, 'local_scale_max': float(np.max(positive_seed_scales)) if positive_seed_scales.size else 0.0}
    return (neighbor_indices, summary)

def filter_seed_components_by_size_and_view_c_sv(components: list[np.ndarray], view_c_sv_display_norm: np.ndarray, *, min_size: int, red_sv_quantile: float) -> dict[str, Any]:
    """Conditionally remove small seed components whose points are all red."""
    view_c_sv_display_norm = np.asarray(view_c_sv_display_norm, dtype=np.float64).reshape(-1)
    if view_c_sv_display_norm.size == 0:
        raise ValueError('view_c_sv_display_norm must contain at least one value.')
    if not np.isfinite(view_c_sv_display_norm).all():
        raise ValueError('view_c_sv_display_norm must contain only finite values.')
    if np.any((view_c_sv_display_norm < 0.0) | (view_c_sv_display_norm > 1.0)):
        raise ValueError('view_c_sv_display_norm values must lie in [0, 1].')
    if isinstance(min_size, (bool, np.bool_)) or not isinstance(min_size, (int, np.integer)):
        raise ValueError('min_size must be an integer.')
    min_size = int(min_size)
    if min_size < 1:
        raise ValueError('min_size must be at least 1.')
    red_sv_quantile = float(red_sv_quantile)
    if not np.isfinite(red_sv_quantile) or not 0.0 <= red_sv_quantile <= 1.0:
        raise ValueError('red_sv_quantile must be finite and lie in [0, 1].')
    red_sv_threshold = float(np.quantile(view_c_sv_display_norm, red_sv_quantile))
    kept_components: list[np.ndarray] = []
    removed_components: list[np.ndarray] = []
    component_reports: list[dict[str, Any]] = []
    for (component_id, component_value) in enumerate(components, start=1):
        component = np.asarray(component_value, dtype=np.int64).reshape(-1)
        if component.size == 0:
            raise ValueError('seed components must not be empty.')
        if np.any(component < 0) or np.any(component >= view_c_sv_display_norm.size):
            raise ValueError('seed component contains an out-of-range point index.')
        component_sv = view_c_sv_display_norm[component]
        below_min_size = bool(component.size < min_size)
        all_points_red = bool(np.all(component_sv >= red_sv_threshold))
        removed = bool(below_min_size and all_points_red)
        component_reports.append({'input_component_id': int(component_id), 'component_size': int(component.size), 'below_min_size': below_min_size, 'view_c_sv_display_norm_min': float(np.min(component_sv)), 'view_c_sv_display_norm_mean': float(np.mean(component_sv)), 'view_c_sv_display_norm_max': float(np.max(component_sv)), 'all_points_red': all_points_red, 'removed': removed, 'decision': 'remove_small_all_red' if removed else 'keep_small_contains_non_red' if below_min_size else 'keep_meets_min_size'})
        if removed:
            removed_components.append(component)
        else:
            kept_components.append(component)
    return {'kept_components': kept_components, 'removed_components': removed_components, 'component_reports': component_reports, 'view_c_sv_display_norm': view_c_sv_display_norm, 'red_sv_quantile': red_sv_quantile, 'red_sv_threshold': red_sv_threshold, 'small_component_count': int(sum((report['below_min_size'] for report in component_reports))), 'small_non_red_kept_component_count': int(sum((report['below_min_size'] and (not report['all_points_red']) for report in component_reports))), 'small_all_red_removed_component_count': int(len(removed_components))}

def extract_production_q_seed_components(seed_mask: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, *, points: np.ndarray | None=None, exp14_module: Any, seed_min_size: int, view_c_sv_display_norm: np.ndarray, red_sv_quantile: float, seed_component_local_graph_enable: bool=False, seed_component_local_k: int=8, seed_component_local_scale_k: int=6, seed_component_local_radius_factor: float=1.6, boundary_filter_mask: np.ndarray, boundary_filter_enable: bool, boundary_dilation_hops: int, boundary_hit_ratio_threshold: float) -> dict[str, Any]:
    """Run the authoritative View-Q extraction/cleaning prefix only.

    This is the exact component partition, AC-boundary component filter, and
    conditional small/all-red component filter used by
    :func:`run_size_only_seed_component_growth`.  It deliberately stops before
    SGCR (View U), merging, Q-to-Z, or GTR so frontend-only experiments can
    inspect production Q without executing any downstream stage.
    """
    seed_mask = np.asarray(seed_mask, dtype=bool).reshape(-1)
    num_points = int(seed_mask.size)
    neighbors = coerce_neighbor_indices(neighbor_indices, num_points)
    seed_min_size = int(seed_min_size)
    if seed_min_size < 1:
        raise ValueError('seed_min_size must be at least 1.')
    view_c_sv_display_norm = _as_vector(view_c_sv_display_norm, name='view_c_sv_display_norm', length=num_points, dtype=np.float64)
    if not np.isfinite(view_c_sv_display_norm).all() or np.any((view_c_sv_display_norm < 0.0) | (view_c_sv_display_norm > 1.0)):
        raise ValueError('view_c_sv_display_norm must contain finite values in [0, 1].')
    red_sv_quantile = float(red_sv_quantile)
    if not np.isfinite(red_sv_quantile) or not 0.0 <= red_sv_quantile <= 1.0:
        raise ValueError('red_sv_quantile must be finite and lie in [0, 1].')
    boundary_filter_mask = _as_bool_mask(boundary_filter_mask, name='boundary_filter_mask', length=num_points)
    if not isinstance(boundary_filter_enable, (bool, np.bool_)):
        raise ValueError('boundary_filter_enable must be True or False.')
    boundary_dilation_hops = int(boundary_dilation_hops)
    if boundary_dilation_hops < 0:
        raise ValueError('boundary_dilation_hops must be non-negative.')
    boundary_hit_ratio_threshold = float(boundary_hit_ratio_threshold)
    if not np.isfinite(boundary_hit_ratio_threshold) or not 0.0 <= boundary_hit_ratio_threshold <= 1.0:
        raise ValueError('boundary_hit_ratio_threshold must be finite and lie in [0, 1].')
    if not isinstance(seed_component_local_graph_enable, (bool, np.bool_)):
        raise ValueError('seed_component_local_graph_enable must be True or False.')
    seed_component_local_graph_enable = bool(seed_component_local_graph_enable)
    for (name, value) in (('seed_component_local_k', seed_component_local_k), ('seed_component_local_scale_k', seed_component_local_scale_k)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer.')
        if int(value) < 1:
            raise ValueError(f'{name} must be at least 1.')
    seed_component_local_k = int(seed_component_local_k)
    seed_component_local_scale_k = int(seed_component_local_scale_k)
    seed_component_local_radius_factor = float(seed_component_local_radius_factor)
    if not np.isfinite(seed_component_local_radius_factor) or seed_component_local_radius_factor <= 0.0:
        raise ValueError('seed_component_local_radius_factor must be finite and positive.')
    if seed_component_local_graph_enable:
        if points is None:
            raise ValueError('points are required when the seed-component local graph is enabled.')
        (component_neighbor_indices, component_graph_summary) = build_seed_component_local_neighbor_indices(points, seed_mask, local_k=seed_component_local_k, local_scale_k=seed_component_local_scale_k, local_radius_factor=seed_component_local_radius_factor)
    else:
        component_neighbor_indices = neighbors
        directed_seed_edges = int(sum((np.sum(seed_mask[np.asarray(row, dtype=np.int64)]) for (point_index, row) in enumerate(neighbors) if seed_mask[point_index])))
        component_graph_summary = {'enabled': False, 'mode': 'original_global_mutual_knn', 'all_point_graph_rebuilt': False, 'query_source': 'reuse exp14 global mutual-KNN neighbor_indices', 'num_points': num_points, 'num_seed_points': int(np.sum(seed_mask)), 'local_k': seed_component_local_k, 'local_scale_k': seed_component_local_scale_k, 'local_radius_factor': seed_component_local_radius_factor, 'num_directed_seed_candidate_edges': directed_seed_edges, 'num_kept_undirected_seed_edges': int(directed_seed_edges // 2)}
    raw_components = exp14_module._connected_components_from_mask(seed_mask, component_neighbor_indices)
    raw_component_labels = exp14_module._components_to_label_array(raw_components, num_points)
    boundary_kept_components: list[np.ndarray] = []
    boundary_removed_components: list[np.ndarray] = []
    boundary_component_reports: list[dict[str, Any]] = []
    for (component_id, component_raw) in enumerate(raw_components, start=1):
        component = np.asarray(component_raw, dtype=np.int64).reshape(-1)
        boundary_hit_count = int(np.sum(boundary_filter_mask[component]))
        boundary_hit_ratio = float(boundary_hit_count / component.size)
        would_remove_under_current_rule = bool(boundary_hit_ratio > boundary_hit_ratio_threshold)
        removed_as_boundary_noise = bool(boundary_filter_enable and would_remove_under_current_rule)
        boundary_component_reports.append({'raw_component_id': int(component_id), 'component_size': int(component.size), 'boundary_hit_count': boundary_hit_count, 'boundary_hit_ratio': boundary_hit_ratio, 'would_remove_under_current_rule': would_remove_under_current_rule, 'actually_removed': removed_as_boundary_noise, 'removed_as_boundary_noise': removed_as_boundary_noise})
        if removed_as_boundary_noise:
            boundary_removed_components.append(component)
        else:
            boundary_kept_components.append(component)
    boundary_kept_component_labels = exp14_module._components_to_label_array(boundary_kept_components, num_points)
    boundary_removed_component_labels = exp14_module._components_to_label_array(boundary_removed_components, num_points)
    boundary_kept_seed_mask = np.asarray(boundary_kept_component_labels > 0, dtype=bool)
    boundary_removed_seed_mask = np.asarray(boundary_removed_component_labels > 0, dtype=bool)
    conditional_filter_result = filter_seed_components_by_size_and_view_c_sv(boundary_kept_components, view_c_sv_display_norm, min_size=seed_min_size, red_sv_quantile=red_sv_quantile)
    kept_components = conditional_filter_result['kept_components']
    removed_components = conditional_filter_result['removed_components']
    kept_component_labels = exp14_module._components_to_label_array(kept_components, num_points)
    removed_component_labels = exp14_module._components_to_label_array(removed_components, num_points)
    kept_seed_mask = np.asarray(kept_component_labels > 0, dtype=bool)
    removed_seed_mask = np.asarray(removed_component_labels > 0, dtype=bool)
    return {'frontend_only': True, 'calls_u': False, 'calls_q_to_z': False, 'calls_gtr': False, 'component_neighbor_indices': component_neighbor_indices, 'component_graph_summary': component_graph_summary, 'raw_components': raw_components, 'raw_component_labels': raw_component_labels, 'boundary_kept_components': boundary_kept_components, 'boundary_removed_components': boundary_removed_components, 'boundary_component_reports': boundary_component_reports, 'boundary_kept_component_labels': boundary_kept_component_labels, 'boundary_removed_component_labels': boundary_removed_component_labels, 'boundary_kept_seed_mask': boundary_kept_seed_mask, 'boundary_removed_seed_mask': boundary_removed_seed_mask, 'conditional_filter_result': conditional_filter_result, 'kept_components': kept_components, 'removed_components': removed_components, 'kept_component_labels': kept_component_labels, 'removed_component_labels': removed_component_labels, 'kept_seed_mask': kept_seed_mask, 'removed_seed_mask': removed_seed_mask}

def run_size_only_seed_component_growth(seed_mask: np.ndarray, c_abs_enhanced: np.ndarray, neighbor_indices: list[np.ndarray], *, points: np.ndarray | None=None, exp14_module: Any, seed_min_size: int, view_c_sv_display_norm: np.ndarray, red_sv_quantile: float, seed_component_local_graph_enable: bool=False, seed_component_local_k: int=8, seed_component_local_scale_k: int=6, seed_component_local_radius_factor: float=1.6, boundary_filter_mask: np.ndarray, boundary_filter_enable: bool, boundary_dilation_hops: int, boundary_hit_ratio_threshold: float, merge_enable: bool, merge_label_percentile: float, merge_min_energy_ratio: float, sgcr_config: Any | None=None) -> dict[str, Any]:
    """Split/filter seeds, reuse SGCR growth, then run Exp21's grown merge.

    The historical function name is retained for callers, but the component
    filter is no longer size-only: a below-minimum component is removed only
    when all of its points are red/high-SV in the normalized View C response.
    """
    seed_mask = np.asarray(seed_mask, dtype=bool).reshape(-1)
    num_points = int(seed_mask.shape[0])
    c_abs_enhanced = _as_vector(c_abs_enhanced, name='c_abs_enhanced', length=num_points, dtype=np.float64)
    if not np.isfinite(c_abs_enhanced).all():
        raise ValueError('c_abs_enhanced must contain only finite values.')
    if len(neighbor_indices) != num_points:
        raise ValueError('neighbor_indices must contain one row per point.')
    seed_min_size = int(seed_min_size)
    if seed_min_size < 1:
        raise ValueError('seed_min_size must be at least 1.')
    view_c_sv_display_norm = _as_vector(view_c_sv_display_norm, name='view_c_sv_display_norm', length=num_points, dtype=np.float64)
    if not np.isfinite(view_c_sv_display_norm).all() or np.any((view_c_sv_display_norm < 0.0) | (view_c_sv_display_norm > 1.0)):
        raise ValueError('view_c_sv_display_norm must contain finite values in [0, 1].')
    red_sv_quantile = float(red_sv_quantile)
    if not np.isfinite(red_sv_quantile) or not 0.0 <= red_sv_quantile <= 1.0:
        raise ValueError('red_sv_quantile must be finite and lie in [0, 1].')
    boundary_filter_mask = _as_bool_mask(boundary_filter_mask, name='boundary_filter_mask', length=num_points)
    if not isinstance(boundary_filter_enable, (bool, np.bool_)):
        raise ValueError('boundary_filter_enable must be True or False.')
    boundary_dilation_hops = int(boundary_dilation_hops)
    if boundary_dilation_hops < 0:
        raise ValueError('boundary_dilation_hops must be non-negative.')
    boundary_hit_ratio_threshold = float(boundary_hit_ratio_threshold)
    if not np.isfinite(boundary_hit_ratio_threshold) or not 0.0 <= boundary_hit_ratio_threshold <= 1.0:
        raise ValueError('boundary_hit_ratio_threshold must be finite and lie in [0, 1].')
    if not isinstance(seed_component_local_graph_enable, (bool, np.bool_)):
        raise ValueError('seed_component_local_graph_enable must be True or False.')
    seed_component_local_graph_enable = bool(seed_component_local_graph_enable)
    if isinstance(seed_component_local_k, (bool, np.bool_)) or not isinstance(seed_component_local_k, (int, np.integer)):
        raise ValueError('seed_component_local_k must be an integer.')
    if isinstance(seed_component_local_scale_k, (bool, np.bool_)) or not isinstance(seed_component_local_scale_k, (int, np.integer)):
        raise ValueError('seed_component_local_scale_k must be an integer.')
    seed_component_local_k = int(seed_component_local_k)
    seed_component_local_scale_k = int(seed_component_local_scale_k)
    seed_component_local_radius_factor = float(seed_component_local_radius_factor)
    if seed_component_local_k < 1:
        raise ValueError('seed_component_local_k must be at least 1.')
    if seed_component_local_scale_k < 1:
        raise ValueError('seed_component_local_scale_k must be at least 1.')
    if not np.isfinite(seed_component_local_radius_factor) or seed_component_local_radius_factor <= 0.0:
        raise ValueError('seed_component_local_radius_factor must be finite and positive.')
    q_result = extract_production_q_seed_components(seed_mask, neighbor_indices, points=points, exp14_module=exp14_module, seed_min_size=seed_min_size, view_c_sv_display_norm=view_c_sv_display_norm, red_sv_quantile=red_sv_quantile, seed_component_local_graph_enable=seed_component_local_graph_enable, seed_component_local_k=seed_component_local_k, seed_component_local_scale_k=seed_component_local_scale_k, seed_component_local_radius_factor=seed_component_local_radius_factor, boundary_filter_mask=boundary_filter_mask, boundary_filter_enable=boundary_filter_enable, boundary_dilation_hops=boundary_dilation_hops, boundary_hit_ratio_threshold=boundary_hit_ratio_threshold)
    component_neighbor_indices = q_result['component_neighbor_indices']
    component_graph_summary = q_result['component_graph_summary']
    raw_components = q_result['raw_components']
    raw_component_labels = q_result['raw_component_labels']
    boundary_kept_components = q_result['boundary_kept_components']
    boundary_removed_components = q_result['boundary_removed_components']
    boundary_component_reports = q_result['boundary_component_reports']
    boundary_kept_component_labels = q_result['boundary_kept_component_labels']
    boundary_removed_component_labels = q_result['boundary_removed_component_labels']
    boundary_kept_seed_mask = q_result['boundary_kept_seed_mask']
    boundary_removed_seed_mask = q_result['boundary_removed_seed_mask']
    conditional_filter_result = q_result['conditional_filter_result']
    kept_components = q_result['kept_components']
    removed_components = q_result['removed_components']
    kept_component_labels = q_result['kept_component_labels']
    removed_component_labels = q_result['removed_component_labels']
    kept_seed_mask = q_result['kept_seed_mask']
    removed_seed_mask = q_result['removed_seed_mask']
    growth_support_norm = np.asarray(exp14_module._percentile_minmax_normalize(c_abs_enhanced), dtype=np.float64)
    max_component_ratio = float(sgcr_config.maximum_component_ratio)
    minimum_growth_size = int(sgcr_config.minimum_growth_component_size)
    max_component_size = max(int(num_points * max_component_ratio), minimum_growth_size)
    grow_reports: list[dict[str, Any]] = []
    grown_components: list[np.ndarray] = []
    grown_sources: list[dict[str, Any]] = []
    grown_union_mask = np.zeros(num_points, dtype=bool)
    for (source_component_id, seed_component) in enumerate(kept_components, start=1):
        seed_component = np.asarray(seed_component, dtype=np.int64).reshape(-1)
        grow_report = exp14_module._grow_one_sgcr_component(seed_indices=seed_component, sv_norm=growth_support_norm, neighbor_indices=neighbor_indices, max_grow_hops=int(sgcr_config.max_grow_hops), local_bg_percentile=0.0, seed_level_percentile=float(sgcr_config.seed_level_percentile), alpha_candidates=(), support_min_norm=float(sgcr_config.absolute_support_floor), max_component_size=max_component_size, growth_mode=str(sgcr_config.growth_mode), seed_relative_grow_ratio=float(sgcr_config.seed_relative_grow_ratio), seed_relative_use_support_min_norm=bool(sgcr_config.use_absolute_support_floor), seed_relative_keep_equal=bool(sgcr_config.keep_equal_to_floor), energy_descent_require_lower_than_current=bool(sgcr_config.require_strict_descent), energy_descent_eps=float(sgcr_config.descent_epsilon), energy_descent_allow_plateau=bool(sgcr_config.allow_plateau), energy_descent_floor_mode=str(sgcr_config.floor_mode), energy_descent_use_priority_queue=bool(sgcr_config.use_priority_queue), compute_inactive_local_background=False)
        grown_indices = np.asarray(grow_report['indices'], dtype=np.int64).reshape(-1)
        grow_report['source_seed_component_id'] = int(source_component_id)
        grow_report['seed_indices'] = seed_component.copy()
        grow_reports.append(grow_report)
        grown_components.append(grown_indices)
        grown_sources.append({'source_seed_component_id': int(source_component_id), 'seed_indices': seed_component.copy(), 'grown_indices': grown_indices, 'grow_report': grow_report})
        grown_union_mask[grown_indices] = True
    grown_component_labels = exp14_module._components_to_label_array(grown_components, num_points)
    (label_aware_merged_components, label_aware_merge_summary) = merge_grown_components_sequential_label_aware(grown_sources=grown_sources, neighbor_indices=neighbor_indices, raw_enhanced_sv=c_abs_enhanced, enabled=merge_enable, label_percentile=merge_label_percentile, min_energy_ratio=merge_min_energy_ratio)
    for source in grown_sources:
        source_component_id = int(source['source_seed_component_id'])
        grown_energy_label = float(label_aware_merge_summary['source_grown_energy_labels'][source_component_id])
        ranked_component_id = int(label_aware_merge_summary['source_ranked_component_ids'][source_component_id])
        source['grown_energy_label'] = grown_energy_label
        source['ranked_component_id'] = ranked_component_id
        source['grow_report']['grown_energy_label'] = grown_energy_label
        source['grow_report']['ranked_component_id'] = ranked_component_id
    label_aware_merged_component_labels = exp14_module._components_to_label_array(label_aware_merged_components, num_points)
    label_aware_merged_mask = np.asarray(label_aware_merged_component_labels > 0, dtype=bool)
    raw_sizes = [int(np.asarray(component).size) for component in raw_components]
    boundary_kept_sizes = [int(np.asarray(component).size) for component in boundary_kept_components]
    boundary_removed_sizes = [int(np.asarray(component).size) for component in boundary_removed_components]
    kept_sizes = [int(np.asarray(component).size) for component in kept_components]
    grown_sizes = [int(np.asarray(component).size) for component in grown_components]
    merged_sizes = [int(np.asarray(component).size) for component in label_aware_merged_components]
    grow_report_summaries = [{'source_seed_component_id': int(report['source_seed_component_id']), 'seed_size': int(report['seed_size']), 'grown_size': int(report['size']), 'added_size': int(report['size'] - report['seed_size']), 'growth_mode': str(report['growth_mode']), 'support_threshold': float(report['support_threshold']), 'seed_self_energy': float(report['seed_self_energy']), 'grown_energy_label': float(report['grown_energy_label']), 'ranked_component_id': int(report['ranked_component_id']), 'oversized': bool(report['oversized'])} for report in grow_reports]
    summary = {'method': 'boundary_and_conditional_small_high_sv_filtered_seed_components_then_original_sgcr_growth_and_exp21_sequential_directed_label_aware_merge', 'component_graph': component_graph_summary, 'boundary_component_filter': {'enabled': bool(boundary_filter_enable), 'mask_source': 'pure_hks_grown_open_boundary_mask expanded by additional hops', 'includes_pure_structure_roots': False, 'additional_dilation_hops': int(boundary_dilation_hops), 'hit_ratio_threshold': float(boundary_hit_ratio_threshold), 'remove_comparison': 'boundary_hit_ratio > threshold', 'research_purpose': 'prevent small open-boundary response residue from being amplified by proposal growth and morphology', 'expanded_boundary_mask_point_count': int(np.sum(boundary_filter_mask)), 'input_component_count': int(len(raw_components)), 'input_seed_point_count': int(np.sum(seed_mask)), 'kept_component_count': int(len(boundary_kept_components)), 'kept_seed_point_count': int(np.sum(boundary_kept_seed_mask)), 'kept_component_size_min': min(boundary_kept_sizes) if boundary_kept_sizes else 0, 'kept_component_size_max': max(boundary_kept_sizes) if boundary_kept_sizes else 0, 'removed_component_count': int(len(boundary_removed_components)), 'removed_seed_point_count': int(np.sum(boundary_removed_seed_mask)), 'would_remove_component_count': int(sum((bool(report['would_remove_under_current_rule']) for report in boundary_component_reports))), 'would_remove_seed_point_count': int(sum((int(report['component_size']) for report in boundary_component_reports if report['would_remove_under_current_rule']))), 'removed_component_size_min': min(boundary_removed_sizes) if boundary_removed_sizes else 0, 'removed_component_size_max': max(boundary_removed_sizes) if boundary_removed_sizes else 0, 'component_reports': boundary_component_reports}, 'seed_component_min_size': seed_min_size, 'original_exp14_seed_min_size': seed_min_size, 'matches_original_exp14_seed_min_size': True, 'small_component_filter': {'view': 'C', 'sv_normalization': 'all_point_p1_p99_clip_minmax_01', 'red_sv_quantile': float(conditional_filter_result['red_sv_quantile']), 'red_sv_threshold': float(conditional_filter_result['red_sv_threshold']), 'remove_comparison': 'component_size < seed_component_min_size AND all(component_view_c_sv_display_norm >= red_sv_threshold)', 'component_reports': conditional_filter_result['component_reports'], 'small_component_count': int(conditional_filter_result['small_component_count']), 'small_non_red_kept_component_count': int(conditional_filter_result['small_non_red_kept_component_count']), 'small_all_red_removed_component_count': int(conditional_filter_result['small_all_red_removed_component_count'])}, 'size_filter_comparison': 'deprecated name: remove only when component_size < seed_component_min_size AND all View-C P1-P99 normalized SV >= all-point Q(red_sv_quantile)', 'mean_score_pruning_applied': False, 'topology_neighbor_filter_applied': False, 'topology_guard_applied': False, 'growth_support': "base_result['c_abs_enhanced']", 'growth_support_normalization': 'q1_q99_clip_minmax_01', 'growth_mode': str(sgcr_config.growth_mode), 'max_grow_hops': int(sgcr_config.max_grow_hops), 'seed_relative_grow_ratio': float(sgcr_config.seed_relative_grow_ratio), 'support_min_norm': float(sgcr_config.absolute_support_floor), 'max_component_size': int(max_component_size), 'raw_seed_point_count': int(np.sum(seed_mask)), 'raw_component_count': int(len(raw_components)), 'raw_component_size_min': min(raw_sizes) if raw_sizes else 0, 'raw_component_size_max': max(raw_sizes) if raw_sizes else 0, 'boundary_filter_kept_seed_point_count': int(np.sum(boundary_kept_seed_mask)), 'boundary_filter_kept_component_count': int(len(boundary_kept_components)), 'boundary_filter_removed_seed_point_count': int(np.sum(boundary_removed_seed_mask)), 'boundary_filter_removed_component_count': int(len(boundary_removed_components)), 'kept_seed_point_count': int(np.sum(kept_seed_mask)), 'kept_component_count': int(len(kept_components)), 'kept_component_size_min': min(kept_sizes) if kept_sizes else 0, 'kept_component_size_max': max(kept_sizes) if kept_sizes else 0, 'removed_seed_point_count': int(np.sum(removed_seed_mask)), 'removed_component_count': int(len(removed_components)), 'grown_union_point_count': int(np.sum(grown_union_mask)), 'grown_union_added_point_count': int(np.sum(grown_union_mask & ~kept_seed_mask)), 'grown_source_count': int(len(grown_components)), 'grown_source_size_min': min(grown_sizes) if grown_sizes else 0, 'grown_source_size_max': max(grown_sizes) if grown_sizes else 0, 'label_aware_merge_applied': bool(merge_enable), 'label_aware_merged_component_count': int(len(label_aware_merged_components)), 'label_aware_merged_point_count': int(np.sum(label_aware_merged_mask)), 'label_aware_merged_component_size_min': min(merged_sizes) if merged_sizes else 0, 'label_aware_merged_component_size_max': max(merged_sizes) if merged_sizes else 0, 'label_aware_merge': label_aware_merge_summary, 'boundary_seed_attachment_applied': False, 'growth_ratio_filter_applied': False, 'minimum_candidate_filter_applied': False, 'grow_reports': grow_report_summaries}
    return {'component_neighbor_indices': component_neighbor_indices, 'component_graph_summary': component_graph_summary, 'raw_components': raw_components, 'raw_component_labels': raw_component_labels, 'boundary_filter_mask': boundary_filter_mask, 'boundary_kept_components': boundary_kept_components, 'boundary_kept_component_labels': boundary_kept_component_labels, 'boundary_kept_seed_mask': boundary_kept_seed_mask, 'boundary_removed_components': boundary_removed_components, 'boundary_removed_component_labels': boundary_removed_component_labels, 'boundary_removed_seed_mask': boundary_removed_seed_mask, 'boundary_component_reports': boundary_component_reports, 'kept_components': kept_components, 'kept_component_labels': kept_component_labels, 'kept_seed_mask': kept_seed_mask, 'removed_components': removed_components, 'removed_component_labels': removed_component_labels, 'removed_seed_mask': removed_seed_mask, 'view_c_sv_display_norm': conditional_filter_result['view_c_sv_display_norm'], 'red_sv_quantile': conditional_filter_result['red_sv_quantile'], 'red_sv_threshold': conditional_filter_result['red_sv_threshold'], 'conditional_filter_component_reports': conditional_filter_result['component_reports'], 'growth_support_norm': growth_support_norm, 'grow_reports': grow_reports, 'grown_sources': grown_sources, 'grown_components': grown_components, 'grown_component_labels': grown_component_labels, 'grown_union_mask': grown_union_mask, 'label_aware_merged_components': label_aware_merged_components, 'label_aware_merged_component_labels': label_aware_merged_component_labels, 'label_aware_merged_mask': label_aware_merged_mask, 'label_aware_merge_summary': label_aware_merge_summary, 'summary': summary}

def compute_empirical_rank_percentile(response: np.ndarray) -> np.ndarray:
    """Compute average-tie empirical rank percentiles in the closed range [0, 100]."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    if response.size == 0:
        raise ValueError('response must contain at least one value.')
    if not np.isfinite(response).all():
        raise ValueError('response must contain only finite values.')
    if response.size == 1:
        return np.zeros(1, dtype=np.float64)
    ranks = np.asarray(rankdata(response, method='average'), dtype=np.float64)
    percentiles = 100.0 * (ranks - 1.0) / float(response.size - 1)
    return np.clip(percentiles, 0.0, 100.0).astype(np.float64, copy=False)

def compute_nonzero_response_percentile_threshold(response: np.ndarray, percentile: float=PURE_NONZERO_GROWTH_PERCENTILE) -> float:
    """Return a percentile threshold computed from strictly nonzero responses."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    percentile = float(percentile)
    if response.size == 0:
        raise ValueError('response must contain at least one value.')
    if not np.isfinite(response).all():
        raise ValueError('response must contain only finite values.')
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError('percentile must be finite and lie in [0, 100].')
    nonzero_response = response[response != 0.0]
    if nonzero_response.size == 0:
        raise ValueError('Cannot compute a nonzero-response percentile: every response is zero.')
    return float(np.percentile(nonzero_response, percentile))

def compute_display_normalized_response(response: np.ndarray, lower_percentile: float=PURE_DISPLAY_LOWER_PERCENTILE, upper_percentile: float=PURE_DISPLAY_UPPER_PERCENTILE, eps: float=PURE_RESPONSE_EPS) -> tuple[np.ndarray, float, float]:
    """Apply the Q1-Q99 clipped normalization used by the View 1 heatmap."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    if response.size == 0:
        raise ValueError('response must contain at least one value.')
    if not np.isfinite(response).all():
        raise ValueError('response must contain only finite values.')
    lower_percentile = float(lower_percentile)
    upper_percentile = float(upper_percentile)
    eps = float(eps)
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError('Display percentiles must satisfy 0 <= lower < upper <= 100.')
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError('eps must be finite and non-negative.')
    q_low = float(np.percentile(response, lower_percentile))
    q_high = float(np.percentile(response, upper_percentile))
    if q_high <= q_low + eps:
        display_norm = np.zeros_like(response, dtype=np.float64)
    else:
        display_norm = np.clip((response - q_low) / (q_high - q_low + eps), 0.0, 1.0)
    return (display_norm.astype(np.float64, copy=False), q_low, q_high)

def compute_reference_display_norm(response: np.ndarray, q_low: float, q_high: float, eps: float=1e-12) -> np.ndarray:
    """Normalize display colors against a caller-supplied fixed value range."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    q_low = float(q_low)
    q_high = float(q_high)
    eps = float(eps)
    if response.size == 0:
        raise ValueError('response must contain at least one value.')
    if not np.isfinite(response).all():
        raise ValueError('response must contain only finite values.')
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high < q_low:
        raise ValueError('q_low and q_high must be finite with q_low <= q_high.')
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError('eps must be finite and positive.')
    return np.clip((response - q_low) / (q_high - q_low + eps), 0.0, 1.0).astype(np.float64, copy=False)

def classify_display_level(display_norm: np.ndarray) -> np.ndarray:
    """Classify Q1-Q99 display values into five closed-lower display levels."""
    display_norm = np.asarray(display_norm, dtype=np.float64).reshape(-1)
    if not np.isfinite(display_norm).all():
        raise ValueError('display_norm must contain only finite values.')
    if np.any((display_norm < 0.0) | (display_norm > 1.0)):
        raise ValueError('display_norm values must lie in [0, 1].')
    levels = np.searchsorted(np.asarray((0.2, 0.4, 0.6, 0.8), dtype=np.float64), display_norm, side='right')
    return np.asarray(levels, dtype=np.int32)

def build_pure_structure_root_mask(display_norm: np.ndarray, min_display_percent: float=PURE_STRUCTURE_ROOT_MIN_DISPLAY_PERCENT) -> np.ndarray:
    """Select additional growth roots from the configurable View 2 color tail."""
    display_norm = np.asarray(display_norm, dtype=np.float64).reshape(-1)
    min_display_percent = float(min_display_percent)
    if not np.isfinite(display_norm).all():
        raise ValueError('display_norm must contain only finite values.')
    if np.any((display_norm < 0.0) | (display_norm > 1.0)):
        raise ValueError('display_norm values must lie in [0, 1].')
    if not np.isfinite(min_display_percent) or not 0.0 <= min_display_percent <= 100.0:
        raise ValueError('min_display_percent must be finite and lie in [0, 100].')
    return np.asarray(display_norm >= min_display_percent / 100.0, dtype=bool)

def label_mask_components(mask: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray) -> dict[str, Any]:
    """Label connected components of one point mask on the production graph.

    Labels are deterministic (first unseen point order), positive inside the
    mask, and zero outside it.  This helper intentionally consumes no sample
    metadata or ground truth.
    """
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    num_points = int(mask.size)
    neighbors = coerce_neighbor_indices(neighbor_indices, num_points)
    labels = np.zeros(num_points, dtype=np.int32)
    components: list[np.ndarray] = []
    for start_value in np.flatnonzero(mask):
        start = int(start_value)
        if labels[start] != 0:
            continue
        component_id = len(components) + 1
        queue: deque[int] = deque([start])
        labels[start] = component_id
        indices: list[int] = []
        while queue:
            current = queue.popleft()
            indices.append(current)
            for neighbor_value in neighbors[current]:
                neighbor = int(neighbor_value)
                if mask[neighbor] and labels[neighbor] == 0:
                    labels[neighbor] = component_id
                    queue.append(neighbor)
        components.append(np.asarray(indices, dtype=np.int64))
    if not np.array_equal(labels > 0, mask):
        raise AssertionError('Connected-component labels do not reproduce mask.')
    return {'component_labels': labels, 'components': components, 'component_sizes': np.asarray([component.size for component in components], dtype=np.int64)}

def _relative_size_is_eligible(point_count: int, largest_point_count: int, min_relative_to_largest: float) -> bool:
    """Evaluate a relative-size gate, using exact integer math at one half."""
    point_count = int(point_count)
    largest_point_count = int(largest_point_count)
    threshold = float(min_relative_to_largest)
    if point_count < 0 or largest_point_count < 0:
        raise ValueError('Component sizes must be non-negative.')
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError('min_relative_to_largest must be finite and in [0, 1].')
    if largest_point_count == 0:
        return False
    if threshold == 0.5:
        return bool(2 * point_count >= largest_point_count)
    return bool(point_count >= threshold * largest_point_count)

def build_component_suppression_eligibility(candidate_mask: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, *, min_relative_to_largest: float, min_point_fraction: float=0.0) -> dict[str, Any]:
    """Build legacy/all and size-qualified component eligibility masks."""
    candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    num_points = int(candidate_mask.size)
    min_point_fraction = float(min_point_fraction)
    if not np.isfinite(min_point_fraction) or not 0.0 <= min_point_fraction <= 1.0:
        raise ValueError('min_point_fraction must be finite and in [0, 1].')
    labeled = label_mask_components(candidate_mask, neighbor_indices)
    sizes = np.asarray(labeled['component_sizes'], dtype=np.int64)
    largest = int(np.max(sizes)) if sizes.size else 0
    total_candidate_points = int(np.sum(sizes))
    eligible_mask = np.zeros(num_points, dtype=bool)
    records: list[dict[str, Any]] = []
    for (component_id, component) in enumerate(labeled['components'], start=1):
        point_count = int(component.size)
        point_fraction = point_count / num_points if num_points else 0.0
        relative = point_count / largest if largest else 0.0
        relative_pass = _relative_size_is_eligible(point_count, largest, min_relative_to_largest)
        fraction_pass = bool(point_fraction >= min_point_fraction)
        eligible = bool(relative_pass and fraction_pass)
        if eligible:
            eligible_mask[component] = True
        records.append({'component_id': int(component_id), 'indices': component, 'point_count': point_count, 'point_fraction_of_cloud': float(point_fraction), 'fraction_of_all_candidate_points': float(point_count / total_candidate_points) if total_candidate_points else 0.0, 'relative_to_largest': float(relative), 'eligible_legacy': True, 'eligible_new': eligible, 'point_fraction_pass': fraction_pass, 'relative_to_largest_pass': relative_pass})
    return {**labeled, 'candidate_mask': candidate_mask.copy(), 'legacy_eligible_mask': candidate_mask.copy(), 'new_eligible_mask': eligible_mask, 'largest_component_size': largest, 'total_candidate_points': total_candidate_points, 'min_point_fraction': min_point_fraction, 'min_relative_to_largest': float(min_relative_to_largest), 'component_records': records}

def resolve_ac_component_grouping_mode(mode: str) -> str:
    """Validate and normalize the AC macro-boundary grouping mode."""
    resolved = str(mode).strip().lower()
    if resolved not in AC_COMPONENT_GROUPING_MODES:
        allowed = ', '.join(sorted(AC_COMPONENT_GROUPING_MODES))
        raise ValueError(f'AC component grouping mode must be one of: {allowed}.')
    return resolved

def _validate_ac_component_group_max_gap_edges(max_gap_edges: int) -> int:
    """Validate the depth bound used only for component identity repair."""
    if isinstance(max_gap_edges, (bool, np.bool_)):
        raise ValueError('max_gap_edges must be a non-negative integer.')
    resolved = int(max_gap_edges)
    if resolved < 0:
        raise ValueError('max_gap_edges must be a non-negative integer.')
    return resolved

def _find_ac_component_gap_edges(component_labels: np.ndarray, components: Sequence[np.ndarray], neighbor_indices: list[np.ndarray] | np.ndarray, *, max_gap_edges: int, points: np.ndarray | None=None) -> list[dict[str, Any]]:
    """Find raw-component links with a depth-limited multi-source BFS.

    The traversal runs on the complete authoritative working graph and may
    cross non-AC points.  Only raw AC endpoints are reported; traversed bridge
    points are deliberately not returned as mask support.
    """
    labels = np.asarray(component_labels, dtype=np.int32).reshape(-1)
    num_points = int(labels.size)
    neighbors = coerce_neighbor_indices(neighbor_indices, num_points)
    max_gap_edges = _validate_ac_component_group_max_gap_edges(max_gap_edges)
    if points is not None:
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.shape != (num_points, 3):
            raise ValueError(f'points must have shape ({num_points}, 3).')
        if not np.isfinite(point_array).all():
            raise ValueError('points must contain only finite coordinates.')
    else:
        point_array = None
    num_components = len(components)
    if num_components <= 1 or max_gap_edges == 0:
        return []
    best_by_pair: dict[tuple[int, int], tuple[int, int, int]] = {}
    for (source_component_id, component_values) in enumerate(components, start=1):
        component = np.sort(np.asarray(component_values, dtype=np.int64).reshape(-1))
        distances = np.full(num_points, -1, dtype=np.int32)
        source_endpoints = np.full(num_points, -1, dtype=np.int64)
        queue: list[tuple[int, int, int]] = []
        for point_value in component:
            point = int(point_value)
            distances[point] = 0
            source_endpoints[point] = point
            heapq.heappush(queue, (0, point, point))
        while queue:
            (distance, source_endpoint, point) = heapq.heappop(queue)
            if int(distances[point]) != distance or int(source_endpoints[point]) != source_endpoint:
                continue
            target_component_id = int(labels[point])
            if target_component_id > 0 and target_component_id != source_component_id:
                component_id_a = min(source_component_id, target_component_id)
                component_id_b = max(source_component_id, target_component_id)
                if source_component_id == component_id_a:
                    nearest_point_a = source_endpoint
                    nearest_point_b = point
                else:
                    nearest_point_a = point
                    nearest_point_b = source_endpoint
                pair = (component_id_a, component_id_b)
                candidate = (int(distance), int(nearest_point_a), int(nearest_point_b))
                previous = best_by_pair.get(pair)
                if previous is None or candidate < previous:
                    best_by_pair[pair] = candidate
            if distance >= max_gap_edges:
                continue
            next_distance = distance + 1
            for neighbor_value in neighbors[point]:
                neighbor = int(neighbor_value)
                current = (int(distances[neighbor]), int(source_endpoints[neighbor]))
                candidate_state = (next_distance, source_endpoint)
                if current[0] < 0 or candidate_state < current:
                    distances[neighbor] = next_distance
                    source_endpoints[neighbor] = source_endpoint
                    heapq.heappush(queue, (next_distance, source_endpoint, neighbor))
    reports: list[dict[str, Any]] = []
    for ((component_id_a, component_id_b), (gap_edges, nearest_point_a, nearest_point_b)) in sorted(best_by_pair.items()):
        if gap_edges > max_gap_edges:
            continue
        euclidean_distance = None
        if point_array is not None:
            euclidean_distance = float(np.linalg.norm(point_array[nearest_point_a] - point_array[nearest_point_b]))
        reports.append({'component_id_a': int(component_id_a), 'component_id_b': int(component_id_b), 'min_graph_gap_edges': int(gap_edges), 'nearest_point_a': int(nearest_point_a), 'nearest_point_b': int(nearest_point_b), 'nearest_pair_euclidean_distance': euclidean_distance})
    return reports

def build_gap_repaired_ac_component_grouping(raw_ac_mask: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, *, min_relative_to_largest: float=AC_COMPONENT_MIN_RELATIVE_TO_LARGEST, max_gap_edges: int=AC_COMPONENT_GROUP_MAX_GAP_EDGES, points: np.ndarray | None=None, raw_eligibility: dict[str, Any] | None=None) -> dict[str, Any]:
    """Build macro AC groups without adding bridge points to AC support."""
    raw_ac = np.asarray(raw_ac_mask, dtype=bool).reshape(-1)
    num_points = int(raw_ac.size)
    neighbors = coerce_neighbor_indices(neighbor_indices, num_points)
    max_gap_edges = _validate_ac_component_group_max_gap_edges(max_gap_edges)
    if raw_eligibility is None:
        raw_eligibility = build_component_suppression_eligibility(raw_ac, neighbors, min_relative_to_largest=min_relative_to_largest)
    raw_candidate = _as_bool_mask(raw_eligibility['candidate_mask'], name='raw_eligibility.candidate_mask', length=num_points)
    if not np.array_equal(raw_candidate, raw_ac):
        raise ValueError('raw_eligibility candidate_mask must equal raw_ac_mask.')
    raw_labels = _as_vector(raw_eligibility['component_labels'], name='raw_ac_component_labels', length=num_points, dtype=np.int32)
    raw_components = [np.asarray(component, dtype=np.int64).reshape(-1) for component in raw_eligibility['components']]
    if not np.array_equal(raw_labels > 0, raw_ac):
        raise AssertionError('Raw AC component labels must reproduce raw AC mask.')
    edge_reports = _find_ac_component_gap_edges(raw_labels, raw_components, neighbors, max_gap_edges=max_gap_edges, points=points)
    num_raw_components = len(raw_components)
    component_adjacency: list[set[int]] = [set() for _ in range(num_raw_components + 1)]
    for edge in edge_reports:
        component_id_a = int(edge['component_id_a'])
        component_id_b = int(edge['component_id_b'])
        component_adjacency[component_id_a].add(component_id_b)
        component_adjacency[component_id_b].add(component_id_a)
    component_groups: list[list[int]] = []
    visited: set[int] = set()
    for start_component_id in range(1, num_raw_components + 1):
        if start_component_id in visited:
            continue
        queue: deque[int] = deque([start_component_id])
        visited.add(start_component_id)
        group: list[int] = []
        while queue:
            component_id = queue.popleft()
            group.append(component_id)
            for neighbor_component_id in sorted(component_adjacency[component_id]):
                if neighbor_component_id not in visited:
                    visited.add(neighbor_component_id)
                    queue.append(neighbor_component_id)
        component_groups.append(sorted(group))
    raw_component_to_group = np.zeros(num_raw_components + 1, dtype=np.int32)
    repaired_group_to_raw_components: dict[int, list[int]] = {}
    for (group_id, component_ids) in enumerate(component_groups, start=1):
        repaired_group_to_raw_components[group_id] = list(component_ids)
        raw_component_to_group[component_ids] = group_id
    repaired_group_labels = raw_component_to_group[raw_labels]
    if not np.array_equal(repaired_group_labels > 0, raw_ac):
        raise AssertionError('Repaired group labels must cover exactly the original raw AC points.')
    raw_sizes = np.asarray([component.size for component in raw_components], dtype=np.int64)
    group_sizes = np.asarray([int(sum((raw_sizes[component_id - 1] for component_id in ids))) for ids in component_groups], dtype=np.int64)
    largest_group_size = int(np.max(group_sizes)) if group_sizes.size else 0
    eligible_group_ids: list[int] = []
    group_reports: list[dict[str, Any]] = []
    point_array = None if points is None else np.asarray(points, dtype=np.float64)
    if point_array is not None and point_array.shape != (num_points, 3):
        raise ValueError(f'points must have shape ({num_points}, 3).')
    for (group_id, component_ids) in enumerate(component_groups, start=1):
        group_size = int(group_sizes[group_id - 1])
        eligible = _relative_size_is_eligible(group_size, largest_group_size, min_relative_to_largest)
        if eligible:
            eligible_group_ids.append(group_id)
        inside_edges = [edge for edge in edge_reports if int(raw_component_to_group[int(edge['component_id_a'])]) == group_id and int(raw_component_to_group[int(edge['component_id_b'])]) == group_id]
        component_sizes = [int(raw_sizes[component_id - 1]) for component_id in component_ids]
        component_centroids: list[dict[str, Any]] = []
        if point_array is not None:
            for component_id in component_ids:
                indices = raw_components[component_id - 1]
                component_centroids.append({'component_id': int(component_id), 'centroid_xyz': np.mean(point_array[indices], axis=0).astype(float).tolist()})
        group_reports.append({'group_id': int(group_id), 'raw_component_ids': [int(value) for value in component_ids], 'num_raw_components': len(component_ids), 'group_point_count': group_size, 'relative_to_largest_group': float(group_size / largest_group_size) if largest_group_size else 0.0, 'eligible': bool(eligible), 'largest_raw_component_size': max(component_sizes, default=0), 'smallest_raw_component_size': min(component_sizes, default=0), 'raw_component_sizes': component_sizes, 'number_of_component_graph_edges_inside_group': len(inside_edges), 'max_pairwise_link_gap_used': max((int(edge['min_graph_gap_edges']) for edge in inside_edges), default=0), 'raw_component_centroids_xyz': component_centroids, 'component_graph_edges': [dict(edge) for edge in inside_edges]})
    eligible_group_array = np.asarray(eligible_group_ids, dtype=np.int32)
    repaired_eligible_core = np.isin(repaired_group_labels, eligible_group_array) & raw_ac
    raw_eligible_core = _as_bool_mask(raw_eligibility['new_eligible_mask'], name='raw_eligibility.new_eligible_mask', length=num_points)
    rescued = np.asarray(repaired_eligible_core & ~raw_eligible_core, dtype=bool)
    removed = np.asarray(raw_ac & ~repaired_eligible_core, dtype=bool)
    if np.any(repaired_eligible_core & ~raw_ac):
        raise AssertionError('Gap repair must never add bridge points to AC core.')
    raw_component_reports: list[dict[str, Any]] = []
    for record in raw_eligibility['component_records']:
        component_id = int(record['component_id'])
        indices = raw_components[component_id - 1]
        centroid = np.mean(point_array[indices], axis=0).astype(float).tolist() if point_array is not None else None
        raw_component_reports.append({'component_id': component_id, 'point_count': int(record['point_count']), 'relative_to_largest': float(record['relative_to_largest']), 'raw_eligible': bool(record['eligible_new']), 'repaired_group_id': int(raw_component_to_group[component_id]), 'selected_by_repaired_group': bool(np.all(repaired_eligible_core[indices])), 'rescued_by_grouping': bool(np.any(rescued[indices])), 'centroid_xyz': centroid})
    return {'raw_ac_mask': raw_ac.copy(), 'raw_ac_component_labels': raw_labels.copy(), 'raw_ac_component_reports': raw_component_reports, 'repaired_ac_group_labels': np.asarray(repaired_group_labels, dtype=np.int32), 'repaired_ac_group_reports': group_reports, 'raw_component_to_repaired_group': raw_component_to_group, 'repaired_group_to_raw_components': repaired_group_to_raw_components, 'ac_grouping_repair_edge_reports': edge_reports, 'raw_ac_eligible_core_mask': raw_eligible_core.copy(), 'repaired_ac_eligible_core_mask': repaired_eligible_core, 'ac_removed_by_group_eligibility_mask': removed, 'ac_fragments_rescued_by_grouping_mask': rescued, 'eligible_repaired_group_ids': eligible_group_array, 'largest_repaired_group_size': largest_group_size, 'num_raw_components': num_raw_components, 'num_repaired_groups': len(component_groups), 'component_group_max_gap_edges': max_gap_edges, 'component_min_relative_to_largest': float(min_relative_to_largest), 'graph_scope': 'complete_exp14_working_mutual_knn_graph', 'hop_definition': 'number_of_graph_edges', 'paths_may_traverse_non_ac_points': True, 'bridge_points_are_mask_support': False}

def build_resolved_ac_config(*, ac_component_aware_enabled: bool, component_grouping_mode: str | None=None, component_group_max_gap_edges: int | None=None) -> dict[str, Any]:
    """Capture the resolved AC values and best-available provenance."""
    resolved_grouping_mode = resolve_ac_component_grouping_mode(AC_COMPONENT_GROUPING_MODE if component_grouping_mode is None else component_grouping_mode)
    resolved_group_max_gap_edges = _validate_ac_component_group_max_gap_edges(AC_COMPONENT_GROUP_MAX_GAP_EDGES if component_group_max_gap_edges is None else component_group_max_gap_edges)
    invocation_source = str(EXP21_AC_CONFIG_OVERRIDE_SOURCE) if EXP21_AC_CONFIG_OVERRIDE_SOURCE else 'exp21 direct'

    def module_source(value: Any, default: Any) -> str:
        if value == default:
            return 'exp21 default'
        if EXP21_AC_CONFIG_OVERRIDE_SOURCE:
            return str(EXP21_AC_CONFIG_OVERRIDE_SOURCE)
        return 'explicit module override'
    component_source = str(EXP21_AC_CONFIG_OVERRIDE_SOURCE or 'explicit AC override') if AC_COMPONENT_AWARE_SUPPRESSION_ENABLE is not None else 'master fallback'
    return {'angle_threshold_deg': float(AC_ANGLE_THRESHOLD_DEG), 'min_neighbors': int(AC_MIN_NEIGHBORS), 'closing_dilation_hops': int(AC_CLOSING_DILATION_HOPS), 'closing_erosion_hops': int(AC_CLOSING_EROSION_HOPS), 'expand_hops': int(AC_EXPAND_HOPS), 'component_aware_enabled': bool(ac_component_aware_enabled), 'component_grouping_mode': resolved_grouping_mode, 'component_group_max_gap_edges': resolved_group_max_gap_edges, 'component_min_relative_to_largest': float(AC_COMPONENT_MIN_RELATIVE_TO_LARGEST), 'query_workers': int(BACKBONE_LOCAL_CONTRAST_QUERY_WORKERS), 'local_response_graph_query_workers': int(LOCAL_RESPONSE_GRAPH_QUERY_WORKERS), 'invocation_source': invocation_source, 'sources': {'angle_threshold_deg': module_source(AC_ANGLE_THRESHOLD_DEG, 110.0), 'min_neighbors': module_source(AC_MIN_NEIGHBORS, 8), 'closing_dilation_hops': module_source(AC_CLOSING_DILATION_HOPS, 0), 'closing_erosion_hops': module_source(AC_CLOSING_EROSION_HOPS, 0), 'expand_hops': module_source(AC_EXPAND_HOPS, 1), 'component_aware_enabled': component_source, 'component_grouping_mode': module_source(resolved_grouping_mode, 'raw'), 'component_group_max_gap_edges': module_source(resolved_group_max_gap_edges, 2), 'component_min_relative_to_largest': module_source(AC_COMPONENT_MIN_RELATIVE_TO_LARGEST, 0.5), 'query_workers': module_source(BACKBONE_LOCAL_CONTRAST_QUERY_WORKERS, -1), 'local_response_graph_query_workers': module_source(LOCAL_RESPONSE_GRAPH_QUERY_WORKERS, -1)}}

def _grow_from_roots(*, response: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, display_level: np.ndarray, root_mask: np.ndarray, level_hop_budgets: Sequence[int], normal_threshold: float, blocked_mask: np.ndarray | None=None) -> dict[str, Any]:
    """Apply the unchanged legacy root-budget and descending-growth helpers."""
    root_hop_budget = build_root_hop_budget(display_level, root_mask, level_hop_budgets=level_hop_budgets)
    growth_result = grow_pure_hks_descending(response=response, neighbor_indices=neighbor_indices, root_mask=root_mask, root_hop_budget=root_hop_budget, normal_threshold=normal_threshold, eps=PURE_RESPONSE_EPS, blocked_mask=blocked_mask)
    return {'root_mask': np.asarray(root_mask, dtype=bool).copy(), 'root_hop_budget': root_hop_budget, 'growth_result': growth_result, 'suppression_mask': np.asarray(growth_result['suppression_region_mask'], dtype=bool)}

def resolve_pure_hks_normal_suppression_mode(pure_hks_normal_suppression_mode: str | None, *, pure_hks_component_aware_suppression_enable: bool) -> str:
    """Resolve the authoritative mode with the historical bool as fallback."""
    if pure_hks_normal_suppression_mode is None:
        return 'component_neutral' if bool(pure_hks_component_aware_suppression_enable) else 'legacy_hard'
    normalized = str(pure_hks_normal_suppression_mode).strip().lower()
    if normalized not in PURE_HKS_NORMAL_SUPPRESSION_MODES:
        choices = ', '.join(sorted(PURE_HKS_NORMAL_SUPPRESSION_MODES))
        raise ValueError('pure_hks_normal_suppression_mode must be one of: ' + choices)
    return normalized

def resolve_pure_hks_soft_attenuation_factor(pure_hks_soft_attenuation_factor: float | None) -> float:
    """Resolve one experiment alpha while preserving the 0.5 default."""
    value = PURE_HKS_SOFT_ATTENUATION_FACTOR if pure_hks_soft_attenuation_factor is None else pure_hks_soft_attenuation_factor
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('pure_hks_soft_attenuation_factor must be a finite scalar in [0, 1].')
    factor = float(value)
    if not np.isfinite(factor) or not 0.0 <= factor <= 1.0:
        raise ValueError('pure_hks_soft_attenuation_factor must be finite and in [0, 1].')
    return factor

def resolve_component_aware_suppression_flags(component_aware_suppression_enable: bool=False, *, ac_component_aware_suppression_enable: bool | None=None, pure_hks_component_aware_suppression_enable: bool | None=None, pure_hks_normal_suppression_mode: str | None=None, pure_hks_soft_attenuation_factor: float | None=None) -> dict[str, Any]:
    """Resolve AC and authoritative Pure-HKS mode with old fallbacks."""
    if not isinstance(component_aware_suppression_enable, (bool, np.bool_)):
        raise ValueError('component_aware_suppression_enable must be boolean.')
    for (name, value) in (('ac_component_aware_suppression_enable', ac_component_aware_suppression_enable), ('pure_hks_component_aware_suppression_enable', pure_hks_component_aware_suppression_enable)):
        if value is not None and (not isinstance(value, (bool, np.bool_))):
            raise ValueError(f'{name} must be True, False, or None.')
    master = bool(component_aware_suppression_enable)
    ac_enabled = master if ac_component_aware_suppression_enable is None else bool(ac_component_aware_suppression_enable)
    pure_bool_fallback = master if pure_hks_component_aware_suppression_enable is None else bool(pure_hks_component_aware_suppression_enable)
    pure_mode = resolve_pure_hks_normal_suppression_mode(pure_hks_normal_suppression_mode, pure_hks_component_aware_suppression_enable=pure_bool_fallback)
    soft_factor = resolve_pure_hks_soft_attenuation_factor(pure_hks_soft_attenuation_factor)
    pure_enabled = pure_mode in {'component_neutral', 'component_soft'}
    combination_name = next((name for (name, spec) in SUPPRESSION_COMBINATION_MODE_SPECS.items() if spec == (ac_enabled, pure_mode)))
    compatibility_mode = 'legacy' if not ac_enabled and (not pure_enabled) else 'new' if ac_enabled and pure_enabled else combination_name
    return {'component_aware_suppression_enable': master, 'ac_component_aware_suppression_enabled': ac_enabled, 'pure_hks_component_aware_suppression_enabled': pure_enabled, 'pure_hks_normal_suppression_mode': pure_mode, 'pure_hks_soft_attenuation_factor': soft_factor, 'suppression_combination_name': combination_name, 'compatibility_mode': compatibility_mode}

def _build_ac_suppression_branch(*, ac_core_mask: np.ndarray, response: np.ndarray, display_level: np.ndarray, neighbor_indices: list[np.ndarray], normal_threshold: float, closing_dilation_hops: int, closing_erosion_hops: int, expand_hops: int, level_hop_budgets: Sequence[int]) -> dict[str, Any]:
    """Run the frozen closing/expand/Pure-HKS-growth AC branch once."""
    ac_core = np.asarray(ac_core_mask, dtype=bool).reshape(-1)
    (ac_closed, closing_summary) = close_gaps(ac_core, neighbor_indices, dilation_hops=closing_dilation_hops, erosion_hops=closing_erosion_hops)
    ac_expanded = expand_boundary_mask(ac_closed, neighbor_indices, hops=expand_hops)
    ac_growth = _grow_from_roots(response=response, neighbor_indices=neighbor_indices, display_level=display_level, root_mask=ac_expanded, level_hop_budgets=level_hop_budgets, normal_threshold=normal_threshold)
    return {'ac_core_mask': ac_core.copy(), 'ac_closed_mask': np.asarray(ac_closed, dtype=bool), 'ac_expanded_mask': np.asarray(ac_expanded, dtype=bool), 'ac_closing_summary': closing_summary, 'ac_growth_result': ac_growth['growth_result'], 'ac_root_hop_budget': ac_growth['root_hop_budget'], 'ac_suppression_mask': ac_growth['suppression_mask']}

def _build_pure_hks_suppression_branch(*, pure_hks_root_mask: np.ndarray, response: np.ndarray, display_level: np.ndarray, neighbor_indices: list[np.ndarray], normal_threshold: float, level_hop_budgets: Sequence[int]) -> dict[str, Any]:
    """Run the frozen Pure-HKS normal-structure branch once."""
    pure_root = np.asarray(pure_hks_root_mask, dtype=bool).reshape(-1)
    pure_growth = _grow_from_roots(response=response, neighbor_indices=neighbor_indices, display_level=display_level, root_mask=pure_root, level_hop_budgets=level_hop_budgets, normal_threshold=normal_threshold)
    return {'pure_hks_root_mask': pure_root.copy(), 'pure_hks_growth_result': pure_growth['growth_result'], 'pure_hks_suppression_mask': pure_growth['suppression_mask']}

def _build_suppression_combination_preview(*, combination_name: str, ac_mode: str, pure_hks_mode: str, ac_branch: dict[str, Any], pure_hks_branch: dict[str, Any], response: np.ndarray, display_level: np.ndarray, neighbor_indices: list[np.ndarray], normal_threshold: float, level_hop_budgets: Sequence[int]) -> dict[str, Any]:
    """Combine independently selected branch roots through exact total growth."""
    total_roots = np.asarray(ac_branch['ac_expanded_mask'] | pure_hks_branch['pure_hks_root_mask'], dtype=bool)
    if pure_hks_mode != 'disabled':
        total_growth = _grow_from_roots(response=response, neighbor_indices=neighbor_indices, display_level=display_level, root_mask=total_roots, level_hop_budgets=level_hop_budgets, normal_threshold=normal_threshold)
    else:
        total_growth = {'root_hop_budget': np.asarray(ac_branch['ac_root_hop_budget'], dtype=np.int32).copy(), 'growth_result': ac_branch['ac_growth_result'], 'suppression_mask': np.asarray(ac_branch['ac_suppression_mask'], dtype=bool).copy()}
    return {'suppression_combination_name': combination_name, 'ac_mode': ac_mode, 'pure_hks_mode': pure_hks_mode, 'ac_component_aware_suppression_enabled': ac_mode == 'new', 'pure_hks_component_aware_suppression_enabled': pure_hks_mode == 'new', 'ac_core_mask': np.asarray(ac_branch['ac_core_mask'], dtype=bool).copy(), 'ac_closed_mask': np.asarray(ac_branch['ac_closed_mask'], dtype=bool).copy(), 'ac_expanded_mask': np.asarray(ac_branch['ac_expanded_mask'], dtype=bool).copy(), 'ac_closing_summary': ac_branch['ac_closing_summary'], 'pure_hks_root_mask': np.asarray(pure_hks_branch['pure_hks_root_mask'], dtype=bool).copy(), 'growth_root_mask': total_roots, 'root_hop_budget': total_growth['root_hop_budget'], 'growth_result': total_growth['growth_result'], 'ac_growth_result': ac_branch['ac_growth_result'], 'pure_hks_growth_result': pure_hks_branch['pure_hks_growth_result'], 'ac_suppression_mask': np.asarray(ac_branch['ac_suppression_mask'], dtype=bool).copy(), 'pure_hks_suppression_mask': np.asarray(pure_hks_branch['pure_hks_suppression_mask'], dtype=bool).copy(), 'total_suppression_mask': total_growth['suppression_mask']}

def build_mainline_new_ac_legacy_hks(*, raw_ac_mask: np.ndarray, pure_hks_normal_candidate_mask: np.ndarray, pure_hks_response: np.ndarray, pure_hks_display_level: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, pure_hks_normal_threshold: float, points: np.ndarray, config: Any) -> dict[str, Any]:
    """Build only Frozen New-AC plus Legacy-hard Pure-HKS suppression."""
    (ac, hks) = (config.new_ac, config.hks)
    raw_ac = np.asarray(raw_ac_mask, dtype=bool).reshape(-1)
    candidate = np.asarray(pure_hks_normal_candidate_mask, dtype=bool).reshape(-1)
    response = np.asarray(pure_hks_response, dtype=np.float64).reshape(-1)
    display_level = np.asarray(pure_hks_display_level, dtype=np.int32).reshape(-1)
    neighbors = coerce_neighbor_indices(neighbor_indices, response.size)
    raw_eligibility = build_component_suppression_eligibility(raw_ac, neighbors, min_relative_to_largest=float(ac.component_min_relative_to_largest))
    repaired = build_gap_repaired_ac_component_grouping(raw_ac, neighbors, min_relative_to_largest=float(ac.component_min_relative_to_largest), max_gap_edges=int(ac.component_group_max_gap_edges), points=points, raw_eligibility=raw_eligibility)
    selected_core = np.asarray(repaired['repaired_ac_eligible_core_mask'], dtype=bool).copy()
    ac_eligibility = dict(raw_eligibility)
    ac_eligibility.update({'component_grouping_mode': 'gap_repaired', 'component_group_max_gap_edges': int(ac.component_group_max_gap_edges), 'raw_new_eligible_mask': np.asarray(raw_eligibility['new_eligible_mask'], dtype=bool).copy(), 'new_eligible_mask': selected_core, 'selected_new_eligible_mask': selected_core, **repaired})
    pure_eligibility = build_component_suppression_eligibility(candidate, neighbors, min_relative_to_largest=0.5, min_point_fraction=0.02)
    ac_branch = _build_ac_suppression_branch(ac_core_mask=selected_core, response=response, display_level=display_level, neighbor_indices=neighbors, normal_threshold=float(pure_hks_normal_threshold), closing_dilation_hops=int(ac.closing_dilation_hops), closing_erosion_hops=int(ac.closing_erosion_hops), expand_hops=int(ac.expand_hops), level_hop_budgets=tuple((int(value) for value in hks.pure_display_level_hop_budgets)))
    pure_branch = _build_pure_hks_suppression_branch(pure_hks_root_mask=np.asarray(pure_eligibility['legacy_eligible_mask'], dtype=bool), response=response, display_level=display_level, neighbor_indices=neighbors, normal_threshold=float(pure_hks_normal_threshold), level_hop_budgets=tuple((int(value) for value in hks.pure_display_level_hop_budgets)))
    selected = _build_suppression_combination_preview(combination_name='new_ac__legacy_hks', ac_mode='new', pure_hks_mode='legacy', ac_branch=ac_branch, pure_hks_branch=pure_branch, response=response, display_level=display_level, neighbor_indices=neighbors, normal_threshold=float(pure_hks_normal_threshold), level_hop_budgets=tuple((int(value) for value in hks.pure_display_level_hop_budgets)))
    zeros = np.zeros(response.size, dtype=bool)
    ac_factor = np.ones(response.size, dtype=np.float64)
    ac_factor[selected['ac_suppression_mask']] = 0.0
    combined_factor = np.ones(response.size, dtype=np.float64)
    combined_factor[selected['total_suppression_mask']] = 0.0
    selected.update({'ac_component_aware_suppression_enabled': True, 'pure_hks_component_aware_suppression_enabled': False, 'pure_hks_normal_suppression_mode': 'legacy_hard', 'pure_hks_soft_attenuation_factor': 0.5, 'pure_hks_hard_component_mask': candidate.copy(), 'pure_hks_soft_component_mask': zeros.copy(), 'pure_hks_neutral_component_mask': zeros.copy(), 'pure_hks_attenuation_factor': combined_factor.copy(), 'ac_attenuation_factor': ac_factor, 'combined_attenuation_factor': combined_factor})
    return {'raw_evidence_shared': True, 'ac_eligibility': ac_eligibility, 'pure_hks_eligibility': pure_eligibility, 'new_ac__legacy_hks': selected}

def assign_rank_levels(rank_percentile: np.ndarray) -> np.ndarray:
    """Map empirical ranks to five diagnostic levels; these do not drive growth."""
    rank_percentile = np.asarray(rank_percentile, dtype=np.float64).reshape(-1)
    if not np.isfinite(rank_percentile).all():
        raise ValueError('rank_percentile must contain only finite values.')
    levels = np.searchsorted(np.asarray(PURE_RANK_CUTS, dtype=np.float64), rank_percentile, side='right')
    return np.asarray(levels, dtype=np.int32)

def build_root_hop_budget(display_level: np.ndarray, root_mask: np.ndarray, level_hop_budgets: Sequence[int]=PURE_DISPLAY_LEVEL_HOP_BUDGETS) -> np.ndarray:
    """Map display levels through a configurable five-value root-hop table."""
    display_level = np.asarray(display_level, dtype=np.int32).reshape(-1)
    root_mask = _as_bool_mask(root_mask, name='root_mask', length=display_level.shape[0])
    if np.any((display_level < 0) | (display_level > 4)):
        raise ValueError('display_level values must lie in 0..4.')
    hop_table = np.asarray(level_hop_budgets)
    if hop_table.shape != (5,):
        raise ValueError('level_hop_budgets must contain exactly five values.')
    if not np.issubdtype(hop_table.dtype, np.integer):
        raise ValueError('level_hop_budgets must contain integers.')
    hop_table = hop_table.astype(np.int32, copy=False)
    if np.any(hop_table < 0):
        raise ValueError('level_hop_budgets must contain non-negative values.')
    root_hop_budget = np.full(display_level.shape[0], -1, dtype=np.int32)
    root_hop_budget[root_mask] = hop_table[display_level[root_mask]]
    return root_hop_budget

def _is_legal_growth_edge(current_index: int, next_index: int, response: np.ndarray, normal_threshold: float, eps: float) -> bool:
    """Return whether one edge satisfies descending and adaptive lower-bound gates."""
    return bool(response[next_index] <= response[current_index] + eps and response[next_index] >= normal_threshold - eps)

def grow_response_descending(response: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, root_mask: np.ndarray, root_hop_budget: np.ndarray, normal_threshold: float, eps: float=PURE_RESPONSE_EPS, blocked_mask: np.ndarray | None=None) -> dict[str, np.ndarray]:
    """Grow on any response with multi-source root-specific remaining-hop budgets.

    ``best_remaining`` drives suppression membership.  A small second state-space
    traversal over ``(point, remaining_hops)`` computes the true minimum legal
    growth depth without changing the suppression mask.  Optional blocked points
    are never entered, so they also cannot act as propagation intermediates.
    """
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    num_points = int(response.shape[0])
    if num_points == 0:
        raise ValueError('response must contain at least one point.')
    if not np.isfinite(response).all():
        raise ValueError('response must contain only finite values.')
    neighbor_list = coerce_neighbor_indices(neighbor_indices, num_points)
    root_mask = _as_bool_mask(root_mask, name='root_mask', length=num_points)
    blocked = np.zeros(num_points, dtype=bool) if blocked_mask is None else _as_bool_mask(blocked_mask, name='blocked_mask', length=num_points)
    root_hop_budget = _as_vector(root_hop_budget, name='root_hop_budget', length=num_points, dtype=np.int32)
    normal_threshold = float(normal_threshold)
    eps = float(eps)
    if not np.isfinite(normal_threshold):
        raise ValueError('normal_threshold must be finite.')
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError('eps must be finite and non-negative.')
    if np.any(root_hop_budget[~root_mask] != -1):
        raise ValueError('Non-root hop budgets must equal -1.')
    if np.any(root_hop_budget[root_mask] < 0):
        raise ValueError('Root hop budgets must be non-negative.')
    if np.any(root_mask & blocked):
        raise ValueError('root_mask and blocked_mask must be disjoint.')
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
        for next_value in neighbor_list[current_index]:
            next_index = int(next_value)
            if blocked[next_index]:
                continue
            if not _is_legal_growth_edge(current_index, next_index, response, normal_threshold, eps):
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
    state_min_depth = np.full((num_points, max_budget + 1), unseen_depth, dtype=np.int32)
    min_growth_depth = np.full(num_points, -1, dtype=np.int32)
    depth_queue: deque[tuple[int, int]] = deque()
    for root_index in root_indices:
        budget = int(root_hop_budget[root_index])
        state_min_depth[root_index, budget] = 0
        min_growth_depth[root_index] = 0
        if budget > 0:
            depth_queue.append((int(root_index), budget))
    while depth_queue:
        (current_index, remaining) = depth_queue.popleft()
        current_depth = int(state_min_depth[current_index, remaining])
        if remaining <= 0:
            continue
        new_remaining = remaining - 1
        for next_value in neighbor_list[current_index]:
            next_index = int(next_value)
            if blocked[next_index]:
                continue
            if not _is_legal_growth_edge(current_index, next_index, response, normal_threshold, eps):
                continue
            new_depth = current_depth + 1
            if new_depth >= int(state_min_depth[next_index, new_remaining]):
                continue
            state_min_depth[next_index, new_remaining] = new_depth
            if min_growth_depth[next_index] < 0 or new_depth < int(min_growth_depth[next_index]):
                min_growth_depth[next_index] = new_depth
            if new_remaining > 0:
                depth_queue.append((next_index, new_remaining))
    if not np.array_equal(min_growth_depth >= 0, suppression_region_mask):
        raise AssertionError('Depth-state reachability disagrees with best_remaining growth.')
    reached_by_parent = np.flatnonzero(growth_parent >= 0)
    for child_index in reached_by_parent:
        parent_index = int(growth_parent[child_index])
        if not np.any(neighbor_list[parent_index] == child_index):
            raise AssertionError('Recorded growth parent edge is not present in the shared graph.')
        if not _is_legal_growth_edge(parent_index, int(child_index), response, normal_threshold, eps):
            raise AssertionError('Recorded growth parent edge violates a response constraint.')
        if int(best_remaining[child_index]) != int(growth_parent_remaining[child_index]) - 1:
            raise AssertionError('Recorded growth parent edge violates remaining-hop accounting.')
    return {'best_remaining': best_remaining, 'min_growth_depth': min_growth_depth, 'grown_only_mask': grown_only_mask.astype(bool, copy=False), 'suppression_region_mask': suppression_region_mask.astype(bool, copy=False), 'growth_parent': growth_parent, 'growth_parent_remaining': growth_parent_remaining}

def grow_pure_hks_descending(response: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray, root_mask: np.ndarray, root_hop_budget: np.ndarray, normal_threshold: float, eps: float=PURE_RESPONSE_EPS, blocked_mask: np.ndarray | None=None) -> dict[str, np.ndarray]:
    """Backward-compatible name for the generic descending-growth primitive."""
    return grow_response_descending(response=response, neighbor_indices=neighbor_indices, root_mask=root_mask, root_hop_budget=root_hop_budget, normal_threshold=normal_threshold, eps=eps, blocked_mask=blocked_mask)

def build_mainline_confidence(*, sv_response: np.ndarray, va_hks_response: np.ndarray, original_sv: np.ndarray, suppression: dict[str, Any], neighbor_indices: list[np.ndarray] | np.ndarray, config: Any) -> dict[str, Any]:
    """Compute only the active mask-SV/minimum/Legacy-VA confidence branch."""
    (hks, conf) = (config.hks, config.confidence)
    sv = np.asarray(sv_response, dtype=np.float64).reshape(-1)
    va = np.asarray(va_hks_response, dtype=np.float64).reshape(-1)
    original = np.asarray(original_sv, dtype=np.float64).reshape(-1)
    total_mask = np.asarray(suppression['total_suppression_mask'], dtype=bool)
    ac_mask = np.asarray(suppression['ac_suppression_mask'], dtype=bool)
    if not sv.shape == va.shape == original.shape == total_mask.shape == ac_mask.shape:
        raise ValueError('Mainline confidence arrays must have identical shapes.')
    keep_weight = float(hks.boundary_keep_weight)
    sv_suppressed = apply_boundary_mask_attenuation(sv, total_mask, keep_weight)
    va_threshold = compute_nonzero_response_percentile_threshold(va, percentile=float(hks.va_nonzero_growth_percentile))
    (va_display_norm, _va_q01, _va_q99) = compute_display_normalized_response(va, lower_percentile=float(hks.va_display_lower_percentile), upper_percentile=float(hks.va_display_upper_percentile), eps=PURE_RESPONSE_EPS)
    va_display_level = classify_display_level(va_display_norm)
    va_root_budget = build_root_hop_budget(va_display_level, total_mask, level_hop_budgets=tuple((int(value) for value in hks.va_display_level_hop_budgets)))
    va_growth = grow_pure_hks_descending(response=va, neighbor_indices=neighbor_indices, root_mask=total_mask, root_hop_budget=va_root_budget, normal_threshold=va_threshold, eps=PURE_RESPONSE_EPS)
    va_growth_mask = np.asarray(va_growth['suppression_region_mask'], dtype=bool)
    va_suppressed = apply_boundary_mask_attenuation(va, va_growth_mask, keep_weight)
    sv_norm = normalize_response_minmax_01(sv_suppressed)
    va_norm = normalize_response_minmax_01(va_suppressed)
    active_fusion = np.minimum(sv_norm, va_norm).astype(np.float64, copy=False)
    semantic_threshold = compute_nonzero_response_percentile_threshold(active_fusion, percentile=float(conf.semantic_gate_nonzero_percentile))
    semantic_pass = np.asarray(active_fusion >= semantic_threshold, dtype=bool)
    semantic_gated = np.where(semantic_pass, active_fusion, float(conf.semantic_gate_replacement_value)).astype(np.float64, copy=False)
    semantic_powered = np.power(semantic_gated, float(conf.semantic_confidence_exponent)).astype(np.float64, copy=False)
    original_threshold = float(np.percentile(original, float(conf.original_sv_gate_percentile)))
    original_pass = np.asarray(original >= original_threshold, dtype=bool)
    original_gated = np.where(original_pass, original, float(conf.original_sv_gate_replacement_value)).astype(np.float64, copy=False)
    seed = (original_gated * semantic_powered).astype(np.float64, copy=False)
    (seed_norm, q01, raw_q99, effective_q99, fallback) = normalize_response_q1_q99_with_nonzero_upper_fallback(seed, p_low=float(conf.seed_display_lower_percentile), p_high=float(conf.seed_display_upper_percentile))
    seed_before_ac = np.asarray(seed_norm >= float(conf.high_confidence_seed_threshold), dtype=bool)
    removed_by_ac = np.asarray(seed_before_ac & ac_mask, dtype=bool)
    seed_mask = np.asarray(seed_before_ac & ~removed_by_ac, dtype=bool)
    sv_factor = np.ones(sv.size, dtype=np.float64)
    sv_factor[total_mask] = keep_weight
    va_factor = np.ones(sv.size, dtype=np.float64)
    va_factor[va_growth_mask] = keep_weight
    zeros = np.zeros(sv.size, dtype=bool)
    return {'sv_branch_suppression_mask': total_mask.copy(), 'va_hks_branch_suppression_mask': total_mask.copy(), 'va_hks_growth_blocked_mask': zeros.copy(), 'va_hks_mask_root_growth_allowed': True, 'va_hks_mask_root_growth_effective': True, 'pure_hks_soft_component_mask': zeros.copy(), 'pure_hks_soft_attenuation_factor': 0.5, 'sv_effective_attenuation_factor': sv_factor, 'va_hks_direct_effective_attenuation_factor': va_factor, 'va_hks_growth_effective_attenuation_factor': va_factor, 'va_hks_effective_attenuation_factor': va_factor, 'sv_suppressed': sv_suppressed, 'sv_suppressed_minmax': sv_norm, 'sv_response_minmax': sv_norm, 'va_hks_direct_suppressed': va_suppressed, 'va_hks_growth_root_hop_budget': va_root_budget, 'va_hks_growth_result': va_growth, 'va_hks_growth_suppression_mask': va_growth_mask, 'va_hks_growth_suppressed': va_suppressed, 'va_hks_active_suppression_mode': 'mask_root_growth', 'va_hks_active_suppression_mask': va_growth_mask, 'va_hks_suppressed': va_suppressed, 'va_hks_suppressed_minmax': va_norm, 'fusion_both_masked_min': active_fusion, 'fusion_va_hks_only_masked_min': active_fusion, 'active_fusion_summary_key': 'both_masked', 'active_fusion_min': active_fusion, 'semantic_confidence': active_fusion, 'semantic_gate_threshold': semantic_threshold, 'semantic_gate_pass_mask': semantic_pass, 'semantic_confidence_gated': semantic_gated, 'semantic_confidence_powered': semantic_powered, 'original_sv_gate_threshold': original_threshold, 'original_sv_gate_pass_mask': original_pass, 'original_sv_gated': original_gated, 'seed_confidence': seed, 'seed_confidence_norm': seed_norm, 'seed_confidence_normalization_q01': q01, 'seed_confidence_normalization_raw_q99': raw_q99, 'seed_confidence_normalization_effective_q99': effective_q99, 'seed_confidence_normalization_fallback_applied': fallback, 'high_confidence_seed_mask_before_boundary_filter': seed_before_ac, 'high_confidence_seed_boundary_removed_mask': removed_by_ac, 'high_confidence_seed_mask': seed_mask, 'va_hks_growth_threshold': va_threshold, 'va_hks_growth_display_norm': va_display_norm, 'va_hks_growth_display_level': va_display_level}

def _summarize_response(response: np.ndarray, normal_threshold: float) -> dict[str, Any]:
    """Summarize raw Pure-HKS response values and requested quantiles."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    nonzero_response = response[response != 0.0]
    quantile_values = np.percentile(response, RESPONSE_QUANTILES)
    return {'growth_threshold_source': 'nonzero_response_percentile', 'nonzero_growth_percentile': float(PURE_NONZERO_GROWTH_PERCENTILE), 'normal_threshold': float(normal_threshold), 'nonzero_response_count': int(nonzero_response.size), 'nonzero_response_ratio': float(np.mean(response != 0.0)), 'response_min': float(np.min(response)), 'response_max': float(np.max(response)), 'response_mean': float(np.mean(response)), 'response_zero_ratio': float(np.mean(response == 0.0)), 'quantiles': {f'Q{int(quantile):d}': float(value) for (quantile, value) in zip(RESPONSE_QUANTILES, quantile_values, strict=True)}}

def _summarize_score_vector(values: np.ndarray) -> dict[str, float]:
    """Return compact finite score statistics for one debug fusion array."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('values must be a non-empty finite vector.')
    return {'min': float(np.min(values)), 'max': float(np.max(values)), 'mean': float(np.mean(values)), 'q50': float(np.percentile(values, 50.0)), 'q90': float(np.percentile(values, 90.0)), 'q99': float(np.percentile(values, 99.0))}

def _summarize_boundary_branch_attenuation(response: np.ndarray, suppressed: np.ndarray, boundary_mask: np.ndarray, keep_weight: float) -> dict[str, Any]:
    """Summarize one raw/suppressed branch without normalizing either array."""
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    suppressed = np.asarray(suppressed, dtype=np.float64).reshape(-1)
    boundary_mask = np.asarray(boundary_mask, dtype=bool).reshape(-1)
    if response.shape != suppressed.shape or response.shape != boundary_mask.shape:
        raise ValueError('response, suppressed, and boundary_mask shapes must match.')
    if not np.isfinite(response).all() or not np.isfinite(suppressed).all():
        raise ValueError('Branch responses must contain only finite values.')
    outside_mask = ~boundary_mask
    nonzero_mask = boundary_mask & (np.abs(response) > PURE_RESPONSE_EPS)

    def _mean_or_zero(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else 0.0
    mask_ratio_values = suppressed[nonzero_mask] / response[nonzero_mask]
    outside_difference = np.abs(suppressed[outside_mask] - response[outside_mask])
    return {'raw_min': float(np.min(response)), 'raw_max': float(np.max(response)), 'raw_mean': float(np.mean(response)), 'suppressed_min': float(np.min(suppressed)), 'suppressed_max': float(np.max(suppressed)), 'suppressed_mean': float(np.mean(suppressed)), 'mask_raw_mean': _mean_or_zero(response[boundary_mask]), 'mask_suppressed_mean': _mean_or_zero(suppressed[boundary_mask]), 'outside_raw_mean': _mean_or_zero(response[outside_mask]), 'outside_suppressed_mean': _mean_or_zero(suppressed[outside_mask]), 'nonzero_mask_count': int(np.sum(nonzero_mask)), 'nonzero_mask_suppressed_over_raw_mean': _mean_or_zero(mask_ratio_values), 'outside_max_abs_difference': float(np.max(outside_difference)) if outside_difference.size else 0.0, 'mask_attenuation_valid': bool(np.allclose(suppressed[boundary_mask], float(keep_weight) * response[boundary_mask], rtol=1e-12, atol=1e-15)), 'outside_unchanged': bool(np.allclose(suppressed[outside_mask], response[outside_mask], rtol=1e-12, atol=1e-15))}

def _count_levels(levels: np.ndarray, active_mask: np.ndarray | None=None) -> dict[str, int]:
    """Count level 0..4 values globally or within an optional point mask."""
    levels = np.asarray(levels, dtype=np.int32).reshape(-1)
    if active_mask is None:
        selected = levels
    else:
        mask = _as_bool_mask(active_mask, name='active_mask', length=levels.shape[0])
        selected = levels[mask]
    if np.any((selected < 0) | (selected > 4)):
        raise ValueError('Selected levels must lie in 0..4.')
    return {f'level{level}': int(np.sum(selected == level)) for level in range(5)}

def _root_has_legal_first_step(root_index: int, response: np.ndarray, neighbor_indices: list[np.ndarray], normal_threshold: float) -> bool:
    """Return whether a positive-budget root has at least one legal first edge."""
    for next_value in neighbor_indices[root_index]:
        if _is_legal_growth_edge(root_index, int(next_value), response, normal_threshold, PURE_RESPONSE_EPS):
            return True
    return False

def _summarize_root_budgets(response: np.ndarray, neighbor_indices: list[np.ndarray], root_mask: np.ndarray, root_hop_budget: np.ndarray, display_level: np.ndarray, level_hop_budgets: Sequence[int], normal_threshold: float) -> dict[str, Any]:
    """Build root diagnostics around the nonzero-response percentile gate."""
    hop_table = np.asarray(level_hop_budgets, dtype=np.int32)
    if hop_table.shape != (5,):
        raise ValueError('level_hop_budgets must contain exactly five values.')
    summary: dict[str, Any] = {}
    for level in range(5):
        hop_budget = int(hop_table[level])
        level_mask = root_mask & (display_level == level)
        root_indices = np.flatnonzero(level_mask)
        if np.any(root_hop_budget[root_indices] != hop_budget):
            raise AssertionError('Root budget does not match its display-level hop table.')
        above_or_equal = response[root_indices] >= normal_threshold
        legal_first_step_count = 0
        if hop_budget > 0:
            legal_first_step_count = sum((_root_has_legal_first_step(int(root_index), response, neighbor_indices, normal_threshold) for root_index in root_indices))
        summary[str(level)] = {'display_level': int(level), 'hop_budget': hop_budget, 'root_count': int(root_indices.size), 'root_count_response_ge_growth_threshold': int(np.sum(above_or_equal)), 'root_count_response_lt_growth_threshold': int(np.sum(~above_or_equal)), 'root_with_legal_first_step_count': int(legal_first_step_count)}
    return summary

def _apply_experimental_suppression_override(hook: Any, *, context: dict[str, Any], base_selected_suppression: dict[str, Any], num_points: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Validate an experiment-owned suppression preview returned by ``hook``.

    This seam is deliberately mask-only and receives no GT, category, or sample
    identity.  Production callers omit it, so all historical mode resolution
    and masks remain untouched.
    """
    if not callable(hook):
        raise TypeError('suppression_override_hook must be callable or None.')
    payload = hook(context)
    if not isinstance(payload, dict):
        raise TypeError('suppression override hook must return a dictionary.')
    name = str(payload.get('suppression_combination_name', '')).strip()
    preview = payload.get('selected_suppression')
    if not name or not isinstance(preview, dict):
        raise ValueError('suppression override requires a non-empty name and selected_suppression dictionary.')
    required_masks = ('ac_core_mask', 'ac_closed_mask', 'ac_expanded_mask', 'pure_hks_root_mask', 'growth_root_mask', 'ac_suppression_mask', 'pure_hks_suppression_mask', 'total_suppression_mask', 'sv_suppression_mask', 'va_hks_suppression_mask', 'pure_hks_hard_component_mask', 'pure_hks_soft_component_mask', 'pure_hks_neutral_component_mask')
    normalized = dict(preview)
    for key in required_masks:
        normalized[key] = _as_bool_mask(normalized.get(key), name=key, length=num_points).copy()
    va_growth_blocked_value = normalized.get('va_hks_growth_blocked_mask')
    if va_growth_blocked_value is not None:
        normalized['va_hks_growth_blocked_mask'] = _as_bool_mask(va_growth_blocked_value, name='va_hks_growth_blocked_mask', length=num_points).copy()
    for key in ('root_hop_budget', 'pure_hks_attenuation_factor'):
        values = np.asarray(normalized.get(key)).reshape(-1)
        if values.shape != (num_points,):
            raise ValueError(f"override '{key}' must have shape ({num_points},).")
        normalized[key] = values.copy()
    for key in ('ac_closing_summary', 'growth_result', 'ac_growth_result', 'pure_hks_growth_result'):
        if key not in normalized:
            raise ValueError(f"suppression override is missing '{key}'.")
    for key in ('ac_core_mask', 'ac_closed_mask', 'ac_expanded_mask', 'ac_suppression_mask'):
        if not np.array_equal(normalized[key], base_selected_suppression[key]):
            raise AssertionError(f"experimental override changed frozen AC field '{key}'.")
    ac_mask = normalized['ac_suppression_mask']
    hks_mask = normalized['pure_hks_suppression_mask']
    allow_pure_hks_ac_overlap = bool(normalized.get('allow_pure_hks_ac_overlap', False))
    if np.any(hks_mask & ac_mask) and (not allow_pure_hks_ac_overlap):
        raise AssertionError('experimental HKS mask must exclude Final-AC points.')
    if not np.array_equal(normalized['total_suppression_mask'], ac_mask | hks_mask):
        raise AssertionError('experimental total suppression must equal FinalAC OR HKS.')
    for key in ('sv_suppression_mask', 'va_hks_suppression_mask'):
        if np.any(ac_mask & ~normalized[key]):
            raise AssertionError(f"experimental branch mask '{key}' dropped AC suppression.")
    va_growth_blocked = normalized.get('va_hks_growth_blocked_mask')
    if va_growth_blocked is not None and np.any(normalized['va_hks_suppression_mask'] & va_growth_blocked):
        raise AssertionError('experimental VA roots must be disjoint from its growth barrier.')
    normalized['suppression_combination_name'] = name
    normalized['experimental_override_active'] = True
    normalized['allow_pure_hks_ac_overlap'] = allow_pure_hks_ac_overlap
    normalized['va_hks_allow_mask_root_growth'] = bool(normalized.get('va_hks_allow_mask_root_growth', False))
    audit = payload.get('audit') or {}
    if not isinstance(audit, dict):
        raise TypeError('suppression override audit must be a dictionary.')
    return (name, normalized, audit)

def _run_covert_frontend_core(pc_path: str | Path, gt_path: str | Path | None=None, *, suppression_strategy: str=SUPPRESSION_STRATEGY_LEGACY, suppression_override_hook: Any | None=None, q_to_z_consistency_repair_enable: bool=VIEW_Q_TO_VIEW_Z_CONSISTENCY_REPAIR_ENABLE, seed_component_boundary_filter_enable: bool | None=None, base_result_override: dict[str, Any] | None=None, config: Any | None=None, production_mainline: bool=False, verbose: bool=False) -> dict[str, Any]:
    """Run Pure-HKS growth plus both selectable VA-HKS suppression modes."""
    pc_path = Path(pc_path)
    (ac, hks) = (config.new_ac, config.hks)
    (conf, seed) = (config.confidence, config.seed_graph)
    (marker, sgcr) = (config.marker, config.sgcr)
    COMPONENT_AWARE_SUPPRESSION_ENABLE = True
    AC_COMPONENT_AWARE_SUPPRESSION_ENABLE = True
    PURE_HKS_COMPONENT_AWARE_SUPPRESSION_ENABLE = False
    PURE_HKS_NORMAL_SUPPRESSION_MODE = 'legacy_hard'
    PURE_HKS_SOFT_ATTENUATION_FACTOR = 0.5
    AC_ANGLE_THRESHOLD_DEG = float(ac.angle_threshold_deg)
    AC_MIN_NEIGHBORS = int(ac.min_neighbors)
    AC_CLOSING_DILATION_HOPS = int(ac.closing_dilation_hops)
    AC_CLOSING_EROSION_HOPS = int(ac.closing_erosion_hops)
    AC_EXPAND_HOPS = int(ac.expand_hops)
    AC_COMPONENT_MIN_RELATIVE_TO_LARGEST = float(ac.component_min_relative_to_largest)
    AC_COMPONENT_GROUPING_MODE = str(ac.component_grouping_mode)
    AC_COMPONENT_GROUP_MAX_GAP_EDGES = int(ac.component_group_max_gap_edges)
    PURE_NONZERO_GROWTH_PERCENTILE = float(hks.pure_nonzero_growth_percentile)
    PURE_DISPLAY_LOWER_PERCENTILE = float(hks.pure_display_lower_percentile)
    PURE_DISPLAY_UPPER_PERCENTILE = float(hks.pure_display_upper_percentile)
    PURE_STRUCTURE_ROOT_MIN_DISPLAY_PERCENT = float(hks.pure_structure_root_min_display_percent)
    PURE_DISPLAY_LEVEL_HOP_BUDGETS = tuple(hks.pure_display_level_hop_budgets)
    BOUNDARY_RESPONSE_KEEP_WEIGHT = float(hks.boundary_keep_weight)
    VA_HKS_USE_MASK_ROOT_GROWTH = bool(hks.va_mask_root_growth)
    VA_HKS_NONZERO_GROWTH_PERCENTILE = float(hks.va_nonzero_growth_percentile)
    VA_HKS_DISPLAY_LOWER_PERCENTILE = float(hks.va_display_lower_percentile)
    VA_HKS_DISPLAY_UPPER_PERCENTILE = float(hks.va_display_upper_percentile)
    VA_HKS_DISPLAY_LEVEL_HOP_BUDGETS = tuple(hks.va_display_level_hop_budgets)
    SV_FUSION_MODE = str(conf.sv_fusion_mode)
    SEMANTIC_GATE_NONZERO_PERCENTILE = float(conf.semantic_gate_nonzero_percentile)
    SEMANTIC_GATE_REPLACEMENT_VALUE = float(conf.semantic_gate_replacement_value)
    SEMANTIC_CONFIDENCE_EXPONENT = float(conf.semantic_confidence_exponent)
    ORIGINAL_SV_GATE_PERCENTILE = float(conf.original_sv_gate_percentile)
    ORIGINAL_SV_GATE_REPLACEMENT_VALUE = float(conf.original_sv_gate_replacement_value)
    HIGH_CONFIDENCE_SEED_THRESHOLD = float(conf.high_confidence_seed_threshold)
    SEED_CONFIDENCE_DISPLAY_LOWER_PERCENTILE = float(conf.seed_display_lower_percentile)
    SEED_CONFIDENCE_DISPLAY_UPPER_PERCENTILE = float(conf.seed_display_upper_percentile)
    HIGH_CONFIDENCE_SEED_REMOVE_ON_PURE_HKS_GROWN_OPEN_BOUNDARY = bool(config.switches.point_boundary_seed_removal)
    SEED_CONFIDENCE_P80_COMPONENT_EXTRACTION_ENABLE = True
    SEED_CONFIDENCE_COMPONENT_PERCENTILE = float(marker.raw_z_percentile)
    SEED_CONFIDENCE_COMPONENT_CLOSING_DILATION_HOPS = int(marker.w_dilation_hops)
    SEED_CONFIDENCE_COMPONENT_CLOSING_EROSION_HOPS = int(marker.w_erosion_hops)
    SEED_CONFIDENCE_COMPONENT_CLOSING_KEEP_ORIGINAL = bool(marker.w_keep_original)
    SEED_COMPONENT_LOCAL_GRAPH_ENABLE = bool(config.switches.seed_local_graph)
    SEED_COMPONENT_LOCAL_K = int(seed.local_k)
    SEED_COMPONENT_LOCAL_SCALE_K = int(seed.local_scale_k)
    SEED_COMPONENT_LOCAL_RADIUS_FACTOR = float(seed.local_radius_factor)
    SEED_COMPONENT_MIN_SIZE = int(seed.min_component_size)
    SEED_COMPONENT_RED_SV_QUANTILE = float(seed.red_sv_quantile)
    SGCR_LABEL_AWARE_MERGE_ENABLE = bool(config.switches.label_aware_merge)
    SGCR_LABEL_AWARE_MERGE_LABEL_PERCENTILE = float(sgcr.label_percentile)
    SGCR_LABEL_AWARE_MERGE_MIN_ENERGY_RATIO = float(sgcr.minimum_energy_ratio)
    EXP21_AC_COMPONENT_DEBUG_ENABLE = False
    MAX_AC_COMPONENT_GAP_AUDIT_HOPS = 5
    BACKBONE_LOCAL_CONTRAST_QUERY_WORKERS = 1
    SEED_COMPONENT_BOUNDARY_FILTER_ENABLE = False
    resolved_gt_path = Path(gt_path) if gt_path is not None else None
    if resolved_gt_path is not None and (not resolved_gt_path.is_file()):
        raise FileNotFoundError(f'Ground-truth file not found: {resolved_gt_path}')
    suppression_strategy = SUPPRESSION_STRATEGY_LEGACY
    if not isinstance(q_to_z_consistency_repair_enable, (bool, np.bool_)):
        raise ValueError('q_to_z_consistency_repair_enable must be True or False.')
    q_to_z_consistency_repair_enable = bool(q_to_z_consistency_repair_enable)
    if suppression_strategy != SUPPRESSION_STRATEGY_LEGACY and suppression_override_hook is not None:
        raise ValueError('suppression_override_hook cannot be combined with an explicit built-in suppression_strategy.')
    keep_weight = float(BOUNDARY_RESPONSE_KEEP_WEIGHT)
    if not np.isfinite(keep_weight) or not 0.0 <= keep_weight <= 1.0:
        raise ValueError('BOUNDARY_RESPONSE_KEEP_WEIGHT must be finite and lie in [0, 1].')
    resolved_suppression = resolve_component_aware_suppression_flags(COMPONENT_AWARE_SUPPRESSION_ENABLE, ac_component_aware_suppression_enable=AC_COMPONENT_AWARE_SUPPRESSION_ENABLE, pure_hks_component_aware_suppression_enable=PURE_HKS_COMPONENT_AWARE_SUPPRESSION_ENABLE, pure_hks_normal_suppression_mode=PURE_HKS_NORMAL_SUPPRESSION_MODE, pure_hks_soft_attenuation_factor=PURE_HKS_SOFT_ATTENUATION_FACTOR)
    component_aware_suppression_enable = bool(resolved_suppression['component_aware_suppression_enable'])
    ac_component_aware_suppression_enabled = bool(resolved_suppression['ac_component_aware_suppression_enabled'])
    pure_hks_component_aware_suppression_enabled = bool(resolved_suppression['pure_hks_component_aware_suppression_enabled'])
    pure_hks_normal_suppression_mode = str(resolved_suppression['pure_hks_normal_suppression_mode'])
    pure_hks_soft_attenuation_factor = float(resolved_suppression['pure_hks_soft_attenuation_factor'])
    ac_component_grouping_mode = resolve_ac_component_grouping_mode(AC_COMPONENT_GROUPING_MODE)
    ac_component_group_max_gap_edges = _validate_ac_component_group_max_gap_edges(AC_COMPONENT_GROUP_MAX_GAP_EDGES)
    if verbose:
        print(f'[Covert] Resolved AC Grouping: mode={ac_component_grouping_mode}, max_gap_edges={ac_component_group_max_gap_edges}, relative_size_threshold={float(AC_COMPONENT_MIN_RELATIVE_TO_LARGEST):g}')
    if not isinstance(VA_HKS_USE_MASK_ROOT_GROWTH, (bool, np.bool_)):
        raise ValueError('VA_HKS_USE_MASK_ROOT_GROWTH must be True or False.')
    if not isinstance(HIGH_CONFIDENCE_SEED_REMOVE_ON_PURE_HKS_GROWN_OPEN_BOUNDARY, (bool, np.bool_)):
        raise ValueError('HIGH_CONFIDENCE_SEED_REMOVE_ON_PURE_HKS_GROWN_OPEN_BOUNDARY must be True or False.')
    sv_fusion_mode = str(SV_FUSION_MODE).strip().lower()
    if sv_fusion_mode not in {'raw_sv', 'mask_sv'}:
        raise ValueError("SV_FUSION_MODE must be either 'raw_sv' or 'mask_sv'.")
    semantic_gate_percentile = float(SEMANTIC_GATE_NONZERO_PERCENTILE)
    semantic_gate_replacement_value = float(SEMANTIC_GATE_REPLACEMENT_VALUE)
    if not np.isfinite(semantic_gate_percentile) or not 0.0 <= semantic_gate_percentile <= 100.0:
        raise ValueError('SEMANTIC_GATE_NONZERO_PERCENTILE must be finite and lie in [0, 100].')
    if not np.isfinite(semantic_gate_replacement_value) or not 0.0 <= semantic_gate_replacement_value <= 1.0:
        raise ValueError('SEMANTIC_GATE_REPLACEMENT_VALUE must be finite and lie in [0, 1].')
    semantic_confidence_exponent = float(SEMANTIC_CONFIDENCE_EXPONENT)
    if not np.isfinite(semantic_confidence_exponent) or semantic_confidence_exponent <= 0.0:
        raise ValueError('SEMANTIC_CONFIDENCE_EXPONENT must be finite and greater than zero.')
    original_sv_gate_percentile = float(ORIGINAL_SV_GATE_PERCENTILE)
    original_sv_gate_replacement_value = float(ORIGINAL_SV_GATE_REPLACEMENT_VALUE)
    if not np.isfinite(original_sv_gate_percentile) or not 0.0 <= original_sv_gate_percentile <= 100.0:
        raise ValueError('ORIGINAL_SV_GATE_PERCENTILE must be finite and lie in [0, 100].')
    if not np.isfinite(original_sv_gate_replacement_value) or original_sv_gate_replacement_value < 0.0:
        raise ValueError('ORIGINAL_SV_GATE_REPLACEMENT_VALUE must be finite and non-negative.')
    high_confidence_seed_threshold = float(HIGH_CONFIDENCE_SEED_THRESHOLD)
    if not np.isfinite(high_confidence_seed_threshold) or not 0.0 <= high_confidence_seed_threshold <= 1.0:
        raise ValueError('HIGH_CONFIDENCE_SEED_THRESHOLD must be finite and lie in [0, 1].')
    if not isinstance(SEED_CONFIDENCE_P80_COMPONENT_EXTRACTION_ENABLE, (bool, np.bool_)):
        raise ValueError('SEED_CONFIDENCE_P80_COMPONENT_EXTRACTION_ENABLE must be True or False.')
    seed_confidence_p80_component_extraction_enable = bool(SEED_CONFIDENCE_P80_COMPONENT_EXTRACTION_ENABLE)
    seed_confidence_component_percentile = float(SEED_CONFIDENCE_COMPONENT_PERCENTILE)
    if not np.isfinite(seed_confidence_component_percentile) or not 0.0 <= seed_confidence_component_percentile <= 100.0:
        raise ValueError('SEED_CONFIDENCE_COMPONENT_PERCENTILE must be finite and lie in [0, 100].')
    for (parameter_name, parameter_value) in (('SEED_CONFIDENCE_COMPONENT_CLOSING_DILATION_HOPS', SEED_CONFIDENCE_COMPONENT_CLOSING_DILATION_HOPS), ('SEED_CONFIDENCE_COMPONENT_CLOSING_EROSION_HOPS', SEED_CONFIDENCE_COMPONENT_CLOSING_EROSION_HOPS)):
        if isinstance(parameter_value, (bool, np.bool_)) or not isinstance(parameter_value, (int, np.integer)):
            raise ValueError(f'{parameter_name} must be an integer.')
        if int(parameter_value) < 0:
            raise ValueError(f'{parameter_name} must be non-negative.')
    seed_confidence_component_closing_dilation_hops = int(SEED_CONFIDENCE_COMPONENT_CLOSING_DILATION_HOPS)
    seed_confidence_component_closing_erosion_hops = int(SEED_CONFIDENCE_COMPONENT_CLOSING_EROSION_HOPS)
    if not isinstance(SEED_CONFIDENCE_COMPONENT_CLOSING_KEEP_ORIGINAL, (bool, np.bool_)):
        raise ValueError('SEED_CONFIDENCE_COMPONENT_CLOSING_KEEP_ORIGINAL must be True or False.')
    seed_confidence_component_closing_keep_original = bool(SEED_CONFIDENCE_COMPONENT_CLOSING_KEEP_ORIGINAL)
    seed_component_min_size = int(SEED_COMPONENT_MIN_SIZE)
    if seed_component_min_size < 1:
        raise ValueError('SEED_COMPONENT_MIN_SIZE must be at least 1.')
    seed_component_red_sv_quantile = float(SEED_COMPONENT_RED_SV_QUANTILE)
    if not np.isfinite(seed_component_red_sv_quantile) or not 0.0 <= seed_component_red_sv_quantile <= 1.0:
        raise ValueError('SEED_COMPONENT_RED_SV_QUANTILE must be finite and lie in [0, 1].')
    if not isinstance(SEED_COMPONENT_LOCAL_GRAPH_ENABLE, (bool, np.bool_)):
        raise ValueError('SEED_COMPONENT_LOCAL_GRAPH_ENABLE must be True or False.')
    seed_component_local_graph_enable = bool(SEED_COMPONENT_LOCAL_GRAPH_ENABLE)
    if isinstance(SEED_COMPONENT_LOCAL_K, (bool, np.bool_)) or not isinstance(SEED_COMPONENT_LOCAL_K, (int, np.integer)):
        raise ValueError('SEED_COMPONENT_LOCAL_K must be an integer.')
    seed_component_local_k = int(SEED_COMPONENT_LOCAL_K)
    if seed_component_local_k < 1:
        raise ValueError('SEED_COMPONENT_LOCAL_K must be at least 1.')
    if isinstance(SEED_COMPONENT_LOCAL_SCALE_K, (bool, np.bool_)) or not isinstance(SEED_COMPONENT_LOCAL_SCALE_K, (int, np.integer)):
        raise ValueError('SEED_COMPONENT_LOCAL_SCALE_K must be an integer.')
    seed_component_local_scale_k = int(SEED_COMPONENT_LOCAL_SCALE_K)
    if seed_component_local_scale_k < 1:
        raise ValueError('SEED_COMPONENT_LOCAL_SCALE_K must be at least 1.')
    seed_component_local_radius_factor = float(SEED_COMPONENT_LOCAL_RADIUS_FACTOR)
    if not np.isfinite(seed_component_local_radius_factor) or seed_component_local_radius_factor <= 0.0:
        raise ValueError('SEED_COMPONENT_LOCAL_RADIUS_FACTOR must be finite and positive.')
    seed_component_boundary_filter_override_applied = bool(seed_component_boundary_filter_enable is not None)
    if seed_component_boundary_filter_enable is None:
        if not isinstance(SEED_COMPONENT_BOUNDARY_FILTER_ENABLE, (bool, np.bool_)):
            raise ValueError('SEED_COMPONENT_BOUNDARY_FILTER_ENABLE must be True or False.')
        seed_component_boundary_filter_enable = bool(SEED_COMPONENT_BOUNDARY_FILTER_ENABLE)
    else:
        if not isinstance(seed_component_boundary_filter_enable, (bool, np.bool_)):
            raise ValueError('seed_component_boundary_filter_enable must be True, False, or None.')
        seed_component_boundary_filter_enable = bool(seed_component_boundary_filter_enable)
    if isinstance(SEED_COMPONENT_BOUNDARY_DILATION_HOPS, (bool, np.bool_)) or not isinstance(SEED_COMPONENT_BOUNDARY_DILATION_HOPS, (int, np.integer)):
        raise ValueError('SEED_COMPONENT_BOUNDARY_DILATION_HOPS must be an integer.')
    seed_component_boundary_dilation_hops = int(SEED_COMPONENT_BOUNDARY_DILATION_HOPS)
    if seed_component_boundary_dilation_hops < 0:
        raise ValueError('SEED_COMPONENT_BOUNDARY_DILATION_HOPS must be non-negative.')
    seed_component_boundary_hit_ratio_threshold = float(SEED_COMPONENT_BOUNDARY_HIT_RATIO_THRESHOLD)
    if not np.isfinite(seed_component_boundary_hit_ratio_threshold) or not 0.0 <= seed_component_boundary_hit_ratio_threshold <= 1.0:
        raise ValueError('SEED_COMPONENT_BOUNDARY_HIT_RATIO_THRESHOLD must be finite and lie in [0, 1].')
    if not isinstance(SGCR_LABEL_AWARE_MERGE_ENABLE, (bool, np.bool_)):
        raise ValueError('SGCR_LABEL_AWARE_MERGE_ENABLE must be True or False.')
    merge_enable = bool(SGCR_LABEL_AWARE_MERGE_ENABLE)
    merge_label_percentile = float(SGCR_LABEL_AWARE_MERGE_LABEL_PERCENTILE)
    if not np.isfinite(merge_label_percentile) or not 0.0 <= merge_label_percentile <= 100.0:
        raise ValueError('SGCR_LABEL_AWARE_MERGE_LABEL_PERCENTILE must be finite and lie in [0, 100].')
    merge_min_energy_ratio = float(SGCR_LABEL_AWARE_MERGE_MIN_ENERGY_RATIO)
    if not np.isfinite(merge_min_energy_ratio) or not 0.0 <= merge_min_energy_ratio <= 1.0:
        raise ValueError('SGCR_LABEL_AWARE_MERGE_MIN_ENERGY_RATIO must be finite and lie in [0, 1].')
    exp14 = _backbone
    base_result = base_result_override
    points = np.asarray(base_result['points'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Backbone points must have shape (N, 3), got {points.shape}.')
    num_points = int(points.shape[0])
    neighbor_indices = coerce_neighbor_indices(base_result['neighbor_indices'], num_points)
    pure_hks_response = _as_vector(base_result['base_views']['pure_hks_only']['score_local'], name='pure_hks_response', length=num_points, dtype=np.float64)
    if not np.isfinite(pure_hks_response).all():
        raise ValueError('Pure-HKS score_local contains non-finite values.')
    sv_response = np.asarray(base_result['base_views']['c_abs_only']['score_local'], dtype=np.float64).reshape(-1)
    va_hks_response = np.asarray(base_result['base_views']['variation_aware_hks_only']['score_local'], dtype=np.float64).reshape(-1)
    if sv_response.shape != (num_points,):
        raise ValueError(f'SV score_local must have shape ({num_points},), got {sv_response.shape}.')
    if va_hks_response.shape != (num_points,):
        raise ValueError(f'VA-HKS score_local must have shape ({num_points},), got {va_hks_response.shape}.')
    if not np.isfinite(sv_response).all():
        raise ValueError('SV score_local contains non-finite values.')
    if not np.isfinite(va_hks_response).all():
        raise ValueError('VA-HKS score_local contains non-finite values.')
    original_sv = _as_vector(base_result['c_abs_raw_feature'], name='original_sv', length=num_points, dtype=np.float64)
    if not np.isfinite(original_sv).all():
        raise ValueError('Original non-enhanced SV contains non-finite values.')
    original_sv_display_q01 = float(np.percentile(original_sv, 1.0))
    original_sv_display_q99 = float(np.percentile(original_sv, 99.0))
    original_sv_view_c_display_norm = compute_reference_display_norm(original_sv, original_sv_display_q01, original_sv_display_q99)
    gt_mask_value = base_result.get('gt_mask')
    gt_mask = None
    if gt_mask_value is not None:
        gt_mask = _as_bool_mask(gt_mask_value, name='gt_mask', length=num_points)
    ac_result = extract_ac_boundary_seed(points, neighbor_indices, angle_threshold=np.deg2rad(AC_ANGLE_THRESHOLD_DEG), min_neighbors=AC_MIN_NEIGHBORS, low_neighbor_policy='boundary')
    ac_seed_mask = _as_bool_mask(ac_result['boundary_mask'], name='ac_seed_mask', length=num_points)
    normal_threshold = compute_nonzero_response_percentile_threshold(pure_hks_response, percentile=PURE_NONZERO_GROWTH_PERCENTILE)
    rank_percentile = compute_empirical_rank_percentile(pure_hks_response)
    (display_norm, display_q01, display_q99) = compute_display_normalized_response(pure_hks_response, lower_percentile=PURE_DISPLAY_LOWER_PERCENTILE, upper_percentile=PURE_DISPLAY_UPPER_PERCENTILE, eps=PURE_RESPONSE_EPS)
    display_level = classify_display_level(display_norm)
    pure_structure_root_mask = build_pure_structure_root_mask(display_norm, min_display_percent=PURE_STRUCTURE_ROOT_MIN_DISPLAY_PERCENT)
    level_hop_budgets = tuple((int(value) for value in PURE_DISPLAY_LEVEL_HOP_BUDGETS))
    suppression_previews = build_mainline_new_ac_legacy_hks(raw_ac_mask=ac_seed_mask, pure_hks_normal_candidate_mask=pure_structure_root_mask, pure_hks_response=pure_hks_response, pure_hks_display_level=display_level, neighbor_indices=neighbor_indices, pure_hks_normal_threshold=normal_threshold, points=points, config=config)
    selected_suppression = suppression_previews['new_ac__legacy_hks']
    suppression_mode = 'new_ac_legacy_hks_baseline'
    built_in_override_hook = None
    effective_suppression_override_hook = suppression_override_hook if built_in_override_hook is None else built_in_override_hook
    suppression_strategy_audit: dict[str, Any] = {}
    suppression_strategy_override_active = effective_suppression_override_hook is not None
    if suppression_strategy_override_active:
        hook_base_preview = {key: value.copy() if isinstance(value, np.ndarray) else value for (key, value) in selected_suppression.items()}
        hook_context = {'pure_hks_response': pure_hks_response.copy(), 'pure_hks_display_level': display_level.copy(), 'pure_hks_normal_candidate_mask': pure_structure_root_mask.copy(), 'sv_response': sv_response.copy(), 'va_hks_response': va_hks_response.copy(), 'pure_hks_normal_component_labels': np.asarray(suppression_previews['pure_hks_eligibility']['component_labels'], dtype=np.int32).copy(), 'original_sv': original_sv.copy(), 'neighbor_indices': [item.copy() for item in neighbor_indices], 'display_level_hop_budgets': np.asarray(level_hop_budgets, dtype=np.int32), 'pure_hks_normal_threshold': float(normal_threshold), 'legacy_hks_suppression_mask': np.asarray(suppression_previews['new_ac__legacy_hks']['pure_hks_suppression_mask'], dtype=bool).copy(), 'final_ac_suppression_mask': np.asarray(selected_suppression['ac_suppression_mask'], dtype=bool).copy(), 'base_selected_suppression': hook_base_preview}
        (override_name, selected_suppression, suppression_strategy_audit) = _apply_experimental_suppression_override(effective_suppression_override_hook, context=hook_context, base_selected_suppression=hook_base_preview, num_points=num_points)
        suppression_previews[override_name] = selected_suppression
        suppression_mode = 'experimental'
    for alias in ('legacy', 'new', 'legacy_ac__legacy_hks', 'new_ac__new_hks', 'legacy_ac__new_hks', 'new_ac__soft_hks', 'legacy_ac__soft_hks', 'new_ac__disabled_hks', 'legacy_ac__disabled_hks'):
        suppression_previews[alias] = selected_suppression
    suppression_combination_name = str(selected_suppression['suppression_combination_name'])
    ac_closed_mask = _as_bool_mask(selected_suppression['ac_closed_mask'], name='ac_closed_mask', length=num_points)
    ac_expanded_mask = _as_bool_mask(selected_suppression['ac_expanded_mask'], name='ac_expanded_mask', length=num_points)
    ac_eligible_core_mask = _as_bool_mask(selected_suppression['ac_core_mask'], name='ac_eligible_core_mask', length=num_points)
    raw_ac_component_labels = _as_vector(suppression_previews['ac_eligibility']['component_labels'], name='raw_ac_component_labels', length=num_points, dtype=np.int32)
    repaired_ac_group_labels = _as_vector(suppression_previews['ac_eligibility']['repaired_ac_group_labels'], name='repaired_ac_group_labels', length=num_points, dtype=np.int32)
    raw_ac_eligible_core_mask = _as_bool_mask(suppression_previews['ac_eligibility']['raw_ac_eligible_core_mask'], name='raw_ac_eligible_core_mask', length=num_points)
    repaired_ac_eligible_core_mask = _as_bool_mask(suppression_previews['ac_eligibility']['repaired_ac_eligible_core_mask'], name='repaired_ac_eligible_core_mask', length=num_points)
    ac_fragments_rescued_by_grouping_mask = _as_bool_mask(suppression_previews['ac_eligibility']['ac_fragments_rescued_by_grouping_mask'], name='ac_fragments_rescued_by_grouping_mask', length=num_points)
    ac_removed_by_eligibility_mask = np.asarray(ac_seed_mask & ~ac_eligible_core_mask, dtype=bool)
    ac_removed_by_group_eligibility_mask = ac_removed_by_eligibility_mask.copy()
    if np.any(ac_fragments_rescued_by_grouping_mask & ~ac_seed_mask):
        raise AssertionError('Gap-repaired rescue mask must remain inside raw AC.')
    closing_is_identity_configuration = bool(int(AC_CLOSING_DILATION_HOPS) == 0 and int(AC_CLOSING_EROSION_HOPS) == 0)
    closing_identity_check = bool(np.array_equal(ac_closed_mask, ac_eligible_core_mask))
    if closing_is_identity_configuration and (not closing_identity_check):
        raise AssertionError('AC graph closing with zero dilation/erosion hops must equal the eligible core exactly.')
    resolved_ac_config = build_resolved_ac_config(ac_component_aware_enabled=ac_component_aware_suppression_enabled, component_grouping_mode=ac_component_grouping_mode, component_group_max_gap_edges=ac_component_group_max_gap_edges)
    raw_ac_component_reports = suppression_previews['ac_eligibility']['raw_ac_component_reports']
    repaired_ac_group_reports = suppression_previews['ac_eligibility']['repaired_ac_group_reports']
    raw_component_to_repaired_group = np.asarray(suppression_previews['ac_eligibility']['raw_component_to_repaired_group'], dtype=np.int32)
    repaired_group_to_raw_components = suppression_previews['ac_eligibility']['repaired_group_to_raw_components']
    ac_grouping_repair_edge_reports = suppression_previews['ac_eligibility']['ac_grouping_repair_edge_reports']
    if not isinstance(EXP21_AC_COMPONENT_DEBUG_ENABLE, (bool, np.bool_)):
        raise ValueError('EXP21_AC_COMPONENT_DEBUG_ENABLE must be True or False.')
    ac_component_reports = []
    ac_component_connectivity_audit = {'enabled': False, 'raw_ac_mask': ac_seed_mask.copy(), 'raw_ac_component_labels': raw_ac_component_labels.copy(), 'ac_eligible_core_mask': ac_eligible_core_mask.copy(), 'ac_removed_by_eligibility_mask': ac_removed_by_eligibility_mask.copy(), 'largest_ac_component_id': 0, 'eligible_ac_component_ids': np.empty(0, dtype=np.int32), 'component_reports': ac_component_reports, 'max_gap_audit_hops': int(MAX_AC_COMPONENT_GAP_AUDIT_HOPS), 'query_workers': int(BACKBONE_LOCAL_CONTRAST_QUERY_WORKERS), 'graph_scope': 'complete_exp14_working_mutual_knn_graph', 'component_graph_scope': 'raw_ac_induced_subgraph', 'hop_definition': 'number_of_graph_edges', 'paths_may_traverse_non_ac_points': True, 'summary': {'enabled': False, 'reason': 'EXP21_AC_COMPONENT_DEBUG_ENABLE=False', 'raw_ac_point_count': int(np.sum(ac_seed_mask)), 'num_components': len(suppression_previews['ac_eligibility']['component_records']), 'eligible_core_point_count': int(np.sum(ac_eligible_core_mask)), 'removed_by_eligibility_point_count': int(np.sum(ac_removed_by_eligibility_mask)), 'max_gap_audit_hops': int(MAX_AC_COMPONENT_GAP_AUDIT_HOPS), 'graph_scope': 'complete_exp14_working_mutual_knn_graph', 'component_graph_scope': 'raw_ac_induced_subgraph', 'hop_definition': 'number_of_graph_edges', 'paths_may_traverse_non_ac_points': True, 'component_reports': []}}
    closing_summary = selected_suppression['ac_closing_summary']
    pure_hks_suppression_root_mask = _as_bool_mask(selected_suppression['pure_hks_root_mask'], name='pure_hks_suppression_root_mask', length=num_points)
    growth_root_mask = _as_bool_mask(selected_suppression['growth_root_mask'], name='growth_root_mask', length=num_points)
    root_hop_budget = np.asarray(selected_suppression['root_hop_budget'], dtype=np.int32)
    growth_result = selected_suppression['growth_result']
    pure_hks_open_boundary_growth_result = selected_suppression['ac_growth_result']
    pure_hks_open_boundary_root_hop_budget = build_root_hop_budget(display_level, ac_expanded_mask, level_hop_budgets=level_hop_budgets)
    pure_hks_grown_open_boundary_mask = np.asarray(pure_hks_open_boundary_growth_result['suppression_region_mask'], dtype=bool)
    seed_component_boundary_filter_mask = expand_boundary_mask(pure_hks_grown_open_boundary_mask, neighbor_indices, hops=seed_component_boundary_dilation_hops)
    seed_component_boundary_filter_mask = _as_bool_mask(seed_component_boundary_filter_mask, name='seed_component_boundary_filter_mask', length=num_points)
    sample_id = str(base_result['summaries']['load']['sample_id'])
    response_summary = _summarize_response(pure_hks_response, normal_threshold)
    display_level_counts = _count_levels(display_level)
    root_display_level_counts = _count_levels(display_level, growth_root_mask)
    rank_diagnostic_levels = assign_rank_levels(rank_percentile)
    rank_diagnostic_root_counts = _count_levels(rank_diagnostic_levels, growth_root_mask)
    root_budget_summary = _summarize_root_budgets(pure_hks_response, neighbor_indices, growth_root_mask, root_hop_budget, display_level, level_hop_budgets, normal_threshold)
    grown_only_mask = growth_result['grown_only_mask']
    suppression_region_mask = growth_result['suppression_region_mask']
    min_growth_depth = growth_result['min_growth_depth']
    boundary_mask = suppression_region_mask
    va_hks_growth_threshold = compute_nonzero_response_percentile_threshold(va_hks_response, percentile=VA_HKS_NONZERO_GROWTH_PERCENTILE)
    (va_hks_growth_display_norm, va_hks_display_q01, va_hks_display_q99) = compute_display_normalized_response(va_hks_response, lower_percentile=VA_HKS_DISPLAY_LOWER_PERCENTILE, upper_percentile=VA_HKS_DISPLAY_UPPER_PERCENTILE, eps=PURE_RESPONSE_EPS)
    va_hks_growth_display_level = classify_display_level(va_hks_growth_display_norm)
    va_hks_level_hop_budgets = tuple((int(value) for value in VA_HKS_DISPLAY_LEVEL_HOP_BUDGETS))
    va_hks_use_mask_root_growth = bool(VA_HKS_USE_MASK_ROOT_GROWTH)
    confidence_previews: dict[str, dict[str, Any]] = {}
    selected_confidence = build_mainline_confidence(sv_response=sv_response, va_hks_response=va_hks_response, original_sv=original_sv, suppression=selected_suppression, neighbor_indices=neighbor_indices, config=config)
    for alias in (suppression_combination_name, 'legacy', 'new', 'new_ac__soft_hks', 'legacy_ac__legacy_hks'):
        confidence_previews[alias] = selected_confidence
    sv_boundary_suppressed = selected_confidence['sv_suppressed']
    va_hks_direct_mask_suppressed = selected_confidence['va_hks_direct_suppressed']
    va_hks_growth_root_hop_budget = selected_confidence['va_hks_growth_root_hop_budget']
    va_hks_growth_result = selected_confidence['va_hks_growth_result']
    va_hks_growth_grown_only_mask = va_hks_growth_result['grown_only_mask']
    va_hks_growth_suppression_mask = selected_confidence['va_hks_growth_suppression_mask']
    va_hks_growth_mask_suppressed = selected_confidence['va_hks_growth_suppressed']
    va_hks_active_suppression_mode = selected_confidence['va_hks_active_suppression_mode']
    va_hks_active_suppression_mask = selected_confidence['va_hks_active_suppression_mask']
    va_hks_boundary_suppressed = selected_confidence['va_hks_suppressed']
    sv_boundary_suppressed_minmax = selected_confidence['sv_suppressed_minmax']
    va_hks_boundary_suppressed_minmax = selected_confidence['va_hks_suppressed_minmax']
    fusion_both_masked_min = selected_confidence['fusion_both_masked_min']
    sv_response_minmax = selected_confidence['sv_response_minmax']
    fusion_va_hks_only_masked_min = selected_confidence['fusion_va_hks_only_masked_min']
    active_fusion_min = selected_confidence['active_fusion_min']
    active_fusion_summary_key = selected_confidence['active_fusion_summary_key']
    semantic_confidence = selected_confidence['semantic_confidence']
    semantic_gate_threshold = selected_confidence['semantic_gate_threshold']
    semantic_gate_pass_mask = selected_confidence['semantic_gate_pass_mask']
    semantic_confidence_gated = selected_confidence['semantic_confidence_gated']
    semantic_confidence_powered = selected_confidence['semantic_confidence_powered']
    original_sv_gate_threshold = selected_confidence['original_sv_gate_threshold']
    original_sv_gate_pass_mask = selected_confidence['original_sv_gate_pass_mask']
    original_sv_gated = selected_confidence['original_sv_gated']
    seed_confidence = selected_confidence['seed_confidence']
    seed_confidence_norm = selected_confidence['seed_confidence_norm']
    seed_confidence_normalization_q01 = selected_confidence['seed_confidence_normalization_q01']
    seed_confidence_normalization_raw_q99 = selected_confidence['seed_confidence_normalization_raw_q99']
    seed_confidence_normalization_effective_q99 = selected_confidence['seed_confidence_normalization_effective_q99']
    seed_confidence_normalization_fallback_applied = selected_confidence['seed_confidence_normalization_fallback_applied']
    seed_confidence_display_q01 = seed_confidence_normalization_q01
    seed_confidence_display_q99 = seed_confidence_normalization_effective_q99
    raw_seed_confidence_spatial_component_result = extract_seed_confidence_percentile_components(seed_confidence, neighbor_indices, enabled=seed_confidence_p80_component_extraction_enable, percentile=seed_confidence_component_percentile)
    high_confidence_seed_boundary_filter_enabled = bool(HIGH_CONFIDENCE_SEED_REMOVE_ON_PURE_HKS_GROWN_OPEN_BOUNDARY)
    high_confidence_seed_mask_before_boundary_filter = selected_confidence['high_confidence_seed_mask_before_boundary_filter']
    high_confidence_seed_boundary_removed_mask = selected_confidence['high_confidence_seed_boundary_removed_mask']
    high_confidence_seed_mask = selected_confidence['high_confidence_seed_mask']
    seed_component_growth_result = run_size_only_seed_component_growth(high_confidence_seed_mask, base_result['c_abs_enhanced'], neighbor_indices, points=points, exp14_module=exp14, seed_min_size=seed_component_min_size, view_c_sv_display_norm=original_sv_view_c_display_norm, red_sv_quantile=seed_component_red_sv_quantile, seed_component_local_graph_enable=seed_component_local_graph_enable, seed_component_local_k=seed_component_local_k, seed_component_local_scale_k=seed_component_local_scale_k, seed_component_local_radius_factor=seed_component_local_radius_factor, boundary_filter_mask=seed_component_boundary_filter_mask, boundary_filter_enable=seed_component_boundary_filter_enable, boundary_dilation_hops=seed_component_boundary_dilation_hops, boundary_hit_ratio_threshold=seed_component_boundary_hit_ratio_threshold, merge_enable=merge_enable, merge_label_percentile=merge_label_percentile, merge_min_energy_ratio=merge_min_energy_ratio, sgcr_config=config.sgcr)
    seed_component_labels = seed_component_growth_result['raw_component_labels']
    seed_component_neighbor_indices = seed_component_growth_result['component_neighbor_indices']
    seed_component_graph_summary = seed_component_growth_result['component_graph_summary']
    seed_boundary_filtered_components = seed_component_growth_result['boundary_kept_components']
    seed_boundary_filtered_component_labels = seed_component_growth_result['boundary_kept_component_labels']
    seed_boundary_filtered_mask = seed_component_growth_result['boundary_kept_seed_mask']
    seed_boundary_removed_components = seed_component_growth_result['boundary_removed_components']
    seed_boundary_removed_component_labels = seed_component_growth_result['boundary_removed_component_labels']
    seed_boundary_removed_mask = seed_component_growth_result['boundary_removed_seed_mask']
    seed_size_filtered_component_labels = seed_component_growth_result['kept_component_labels']
    seed_size_filtered_components = seed_component_growth_result['kept_components']
    seed_size_filtered_mask = seed_component_growth_result['kept_seed_mask']
    seed_size_removed_component_labels = seed_component_growth_result['removed_component_labels']
    seed_size_removed_mask = seed_component_growth_result['removed_seed_mask']
    seed_growth_support_norm = seed_component_growth_result['growth_support_norm']
    seed_grown_component_labels = seed_component_growth_result['grown_component_labels']
    seed_grown_mask = seed_component_growth_result['grown_union_mask']
    seed_label_aware_merged_components = seed_component_growth_result['label_aware_merged_components']
    seed_label_aware_merged_component_labels = seed_component_growth_result['label_aware_merged_component_labels']
    seed_label_aware_merged_mask = seed_component_growth_result['label_aware_merged_mask']
    seed_label_aware_merge_summary = seed_component_growth_result['label_aware_merge_summary']
    q_z_consistency_result = build_marker_consistent_view_z(raw_seed_confidence_spatial_component_result, seed_size_filtered_components, seed_size_filtered_mask, neighbor_indices, seed_component_neighbor_indices, repair_enable=q_to_z_consistency_repair_enable)
    seed_confidence_spatial_component_result = q_z_consistency_result['authoritative_view_z']
    seed_confidence_view_y_morphology_grouping_result = group_seed_confidence_components_for_closing_by_view_y(seed_confidence_spatial_component_result['components'], seed_confidence_spatial_component_result['component_labels'], seed_confidence_spatial_component_result['valid_mask'], seed_label_aware_merged_component_labels, seed_size_filtered_mask, enabled=seed_confidence_p80_component_extraction_enable)
    seed_confidence_percentile_component_summary = dict(seed_confidence_spatial_component_result['summary'])
    seed_confidence_percentile_component_summary.update({'component_stage': 'spatial_view_z_identity_preserved', 'view_y_morphology_grouping': seed_confidence_view_y_morphology_grouping_result['summary']})
    seed_confidence_percentile_component_result = {'enabled': seed_confidence_spatial_component_result['enabled'], 'percentile': seed_confidence_spatial_component_result['percentile'], 'threshold': seed_confidence_spatial_component_result['threshold'], 'valid_mask': seed_confidence_spatial_component_result['valid_mask'], 'components': seed_confidence_spatial_component_result['components'], 'component_labels': seed_confidence_spatial_component_result['component_labels'], 'summary': seed_confidence_percentile_component_summary}
    seed_confidence_component_closing_result = close_seed_confidence_view_z_morphology_groups(seed_confidence_percentile_component_result['components'], seed_confidence_percentile_component_result['component_labels'], seed_confidence_view_y_morphology_grouping_result['morphology_group_components'], seed_confidence_view_y_morphology_grouping_result['spatial_to_morphology_group_id'], seed_confidence_view_y_morphology_grouping_result['morphology_group_reports'], neighbor_indices, num_points=num_points, enabled=seed_confidence_p80_component_extraction_enable, dilation_hops=seed_confidence_component_closing_dilation_hops, erosion_hops=seed_confidence_component_closing_erosion_hops, keep_original=seed_confidence_component_closing_keep_original)
    view_q_to_view_z_mapping_result = map_view_q_seed_components_to_view_z_components(seed_size_filtered_components, seed_confidence_percentile_component_result['component_labels'], seed_confidence_component_closing_result['component_reports'], num_points=num_points, enabled=seed_confidence_p80_component_extraction_enable)
    q_z_consistency_repair_audit = dict(q_z_consistency_result['audit'])
    q_z_consistency_repair_audit['authoritative_active_z_count'] = int(seed_confidence_view_y_morphology_grouping_result['active_spatial_view_z_component_ids'].size)
    q_z_consistency_repair_audit['authoritative_w_morphology_group_count'] = int(len(seed_confidence_view_y_morphology_grouping_result['morphology_group_components']))
    if not q_z_consistency_repair_audit['repair_applied']:
        q_z_consistency_repair_audit['raw_w_morphology_group_count'] = int(len(seed_confidence_view_y_morphology_grouping_result['morphology_group_components']))
    elif q_z_consistency_repair_audit['missing_q_point_count_total'] == 0:
        raw_grouping_audit = group_seed_confidence_components_for_closing_by_view_y(raw_seed_confidence_spatial_component_result['components'], raw_seed_confidence_spatial_component_result['component_labels'], raw_seed_confidence_spatial_component_result['valid_mask'], seed_label_aware_merged_component_labels, seed_size_filtered_mask, enabled=seed_confidence_p80_component_extraction_enable)
        q_z_consistency_repair_audit['raw_w_morphology_group_count'] = int(len(raw_grouping_audit['morphology_group_components']))
    closing_by_z_id = {int(report['component_id']): report for report in seed_confidence_component_closing_result['component_reports']}
    for report in q_z_consistency_repair_audit['offending_q_components']:
        repaired_z_id = report.get('repaired_z_id')
        if repaired_z_id is None:
            continue
        closing_report = closing_by_z_id.get(int(repaired_z_id))
        if closing_report is not None:
            report['repaired_w_size'] = int(closing_report['morphology_group_closed_size'])
    sv_display_q01 = float(np.percentile(sv_response, 1.0))
    sv_display_q99 = float(np.percentile(sv_response, 99.0))
    sv_suppressed_display_q01 = float(np.percentile(sv_boundary_suppressed, 1.0))
    sv_suppressed_display_q99 = float(np.percentile(sv_boundary_suppressed, 99.0))
    va_hks_direct_suppressed_display_q01 = float(np.percentile(va_hks_direct_mask_suppressed, 1.0))
    va_hks_direct_suppressed_display_q99 = float(np.percentile(va_hks_direct_mask_suppressed, 99.0))
    va_hks_growth_suppressed_display_q01 = float(np.percentile(va_hks_growth_mask_suppressed, 1.0))
    va_hks_growth_suppressed_display_q99 = float(np.percentile(va_hks_growth_mask_suppressed, 99.0))
    va_hks_suppressed_display_q01 = va_hks_growth_suppressed_display_q01 if va_hks_use_mask_root_growth else va_hks_direct_suppressed_display_q01
    va_hks_suppressed_display_q99 = va_hks_growth_suppressed_display_q99 if va_hks_use_mask_root_growth else va_hks_direct_suppressed_display_q99
    sv_suppression_summary = _summarize_boundary_branch_attenuation(sv_response, sv_boundary_suppressed, boundary_mask, keep_weight)
    va_hks_suppression_summary = _summarize_boundary_branch_attenuation(va_hks_response, va_hks_boundary_suppressed, va_hks_active_suppression_mask, keep_weight)
    va_hks_direct_suppression_summary = _summarize_boundary_branch_attenuation(va_hks_response, va_hks_direct_mask_suppressed, boundary_mask, keep_weight)
    va_hks_growth_suppression_summary = _summarize_boundary_branch_attenuation(va_hks_response, va_hks_growth_mask_suppressed, va_hks_growth_suppression_mask, keep_weight)
    summaries: dict[str, Any] = {'sample': {'sample_id': sample_id, 'pc_path': str(pc_path), 'gt_path': str(resolved_gt_path) if resolved_gt_path is not None else None, 'num_working_points': num_points}, 'backbone': {'source': 'exp14.run_topology_suppression_debug_pipeline', 'pure_hks_response_source': "base_result['base_views']['pure_hks_only']['score_local']", 'sv_response_source': "base_result['base_views']['c_abs_only']['score_local']", 'va_hks_response_source': "base_result['base_views']['variation_aware_hks_only']['score_local']", 'fps_mode': BACKBONE_FPS_MODE, 'fps_prefilter_num_points': int(BACKBONE_FPS_PREFILTER_NUM_POINTS), 'fps_seed': int(BACKBONE_FPS_SEED), 'working_graph': base_result['summaries'].get('graph', {})}, 'resolved_ac_config': resolved_ac_config, 'ac_component_grouping_mode': ac_component_grouping_mode, 'ac_component_group_max_gap_edges': ac_component_group_max_gap_edges, 'ac_component_connectivity_audit': ac_component_connectivity_audit['summary'], 'ac_macro_boundary_grouping': {'mode': ac_component_grouping_mode, 'component_group_max_gap_edges': ac_component_group_max_gap_edges, 'component_min_relative_to_largest': float(AC_COMPONENT_MIN_RELATIVE_TO_LARGEST), 'graph_scope': 'complete_exp14_working_mutual_knn_graph', 'hop_definition': 'number_of_graph_edges', 'paths_may_traverse_non_ac_points': True, 'bridge_points_are_mask_support': False, 'num_raw_components': len(raw_ac_component_reports), 'num_repaired_groups': len(repaired_ac_group_reports), 'raw_eligible_core_point_count': int(np.sum(raw_ac_eligible_core_mask)), 'repaired_eligible_core_point_count': int(np.sum(repaired_ac_eligible_core_mask)), 'selected_eligible_core_point_count': int(np.sum(ac_eligible_core_mask)), 'rescued_fragment_point_count': int(np.sum(ac_fragments_rescued_by_grouping_mask)), 'raw_component_to_repaired_group': raw_component_to_repaired_group.astype(int).tolist(), 'repaired_group_to_raw_components': {str(group_id): [int(value) for value in component_ids] for (group_id, component_ids) in repaired_group_to_raw_components.items()}, 'raw_component_reports': raw_ac_component_reports, 'repaired_group_reports': repaired_ac_group_reports, 'component_graph_edge_reports': ac_grouping_repair_edge_reports}, 'ac_stage_lineage': {'stage_a': 'raw_ac_mask', 'stage_b': 'raw_ac_component_labels', 'stage_c': 'repaired_ac_group_labels', 'stage_d': 'ac_eligible_core_mask (raw AC points only)', 'stage_e': 'ac_closed_mask', 'stage_f': 'ac_expanded_mask', 'stage_g': 'ac_response_growth_mask/final_ac_suppression_mask', 'grouping_repair_is_not_morphological_closing': True, 'grouping_bridge_points_are_excluded_from_core': True, 'eligibility_is_evaluated_before_closing': True, 'closing_zero_hops_requires_core_identity': closing_is_identity_configuration, 'closing_zero_hops_core_identity_verified': closing_identity_check}, 'component_aware_suppression': {'deprecated_master_fallback': True, 'enabled': component_aware_suppression_enable, 'selected_mode': suppression_mode, 'strategy': suppression_strategy, 'ac_component_aware_suppression_enabled': ac_component_aware_suppression_enabled, 'pure_hks_component_aware_suppression_enabled': pure_hks_component_aware_suppression_enabled, 'pure_hks_normal_suppression_mode': pure_hks_normal_suppression_mode, 'pure_hks_soft_attenuation_factor': float(pure_hks_soft_attenuation_factor), 'suppression_combination_name': suppression_combination_name, 'experimental_override_active': bool(suppression_strategy_override_active), 'experimental_override_summary': suppression_strategy_audit.get('summary', {}), 'decision_inputs': 'component_sizes, working_cloud_size, raw_masks_and_responses_only', 'uses_gt': False, 'uses_category_or_sample_name': False, 'ac_min_relative_to_largest': float(AC_COMPONENT_MIN_RELATIVE_TO_LARGEST), 'ac_component_grouping_mode': ac_component_grouping_mode, 'ac_component_group_max_gap_edges': ac_component_group_max_gap_edges, 'pure_hks_normal_min_point_fraction': float(PURE_HKS_NORMAL_MIN_POINT_FRACTION), 'pure_hks_normal_min_relative_to_largest': float(PURE_HKS_NORMAL_MIN_RELATIVE_TO_LARGEST), 'num_ac_components': len(suppression_previews['ac_eligibility']['component_records']), 'num_raw_ac_components': len(suppression_previews['ac_eligibility']['component_records']), 'largest_ac_component_size': int(suppression_previews['ac_eligibility']['largest_component_size']), 'num_ac_components_new_eligible': sum((bool(np.any(ac_eligible_core_mask[raw_ac_component_labels == component_id])) for component_id in range(1, len(raw_ac_component_reports) + 1))), 'num_new_ac_eligible_components': sum((bool(np.any(ac_eligible_core_mask[raw_ac_component_labels == component_id])) for component_id in range(1, len(raw_ac_component_reports) + 1))), 'num_repaired_ac_groups': len(repaired_ac_group_reports), 'num_repaired_ac_groups_eligible': sum((bool(report['eligible']) for report in repaired_ac_group_reports)), 'largest_repaired_ac_group_size': int(suppression_previews['ac_eligibility']['largest_repaired_group_size']), 'ac_fragments_rescued_point_count': int(np.sum(ac_fragments_rescued_by_grouping_mask)), 'new_ac_eligible_point_count': int(np.sum(suppression_previews['ac_eligibility']['new_eligible_mask'])), 'ac_core_point_count': int(np.sum(selected_suppression['ac_core_mask'])), 'ac_closed_point_count': int(np.sum(selected_suppression['ac_closed_mask'])), 'ac_expanded_point_count': int(np.sum(selected_suppression['ac_expanded_mask'])), 'ac_response_growth_point_count': int(np.sum(selected_suppression['ac_suppression_mask'] & ~selected_suppression['ac_expanded_mask'])), 'final_ac_suppression_point_count': int(np.sum(selected_suppression['ac_suppression_mask'])), 'num_pure_hks_normal_components': len(suppression_previews['pure_hks_eligibility']['component_records']), 'num_pure_hks_normal_components_new_eligible': sum((bool(record['eligible_new']) for record in suppression_previews['pure_hks_eligibility']['component_records'])), 'num_hks_components_total': len(suppression_previews['pure_hks_eligibility']['component_records']), 'num_hks_hard_components': sum((pure_hks_normal_suppression_mode == 'legacy_hard' or (pure_hks_normal_suppression_mode in {'component_neutral', 'component_soft'} and bool(record['eligible_new'])) for record in suppression_previews['pure_hks_eligibility']['component_records'])), 'num_hks_soft_components': sum((pure_hks_normal_suppression_mode == 'component_soft' and (not bool(record['eligible_new'])) for record in suppression_previews['pure_hks_eligibility']['component_records'])), 'num_hks_neutral_components': sum((pure_hks_normal_suppression_mode == 'component_neutral' and (not bool(record['eligible_new'])) for record in suppression_previews['pure_hks_eligibility']['component_records'])), 'pure_hks_hard_point_count': int(np.sum(selected_suppression['pure_hks_hard_component_mask'])), 'pure_hks_soft_point_count': int(np.sum(selected_suppression['pure_hks_soft_component_mask'])), 'pure_hks_neutral_point_count': int(np.sum(selected_suppression['pure_hks_neutral_component_mask'])), 'legacy_total_suppression_count': int(np.sum(suppression_previews['legacy']['total_suppression_mask'])), 'new_total_suppression_count': int(np.sum(suppression_previews['new']['total_suppression_mask']))}, 'ac': {'angle_threshold_degrees': float(AC_ANGLE_THRESHOLD_DEG), 'min_neighbors': int(AC_MIN_NEIGHBORS), 'closing_dilation_hops': int(AC_CLOSING_DILATION_HOPS), 'closing_erosion_hops': int(AC_CLOSING_EROSION_HOPS), 'expand_hops': int(AC_EXPAND_HOPS), 'raw_seed_count': int(np.sum(ac_seed_mask)), 'raw_seed_ratio': float(np.mean(ac_seed_mask)), 'eligible_core_count': int(np.sum(ac_eligible_core_mask)), 'removed_by_eligibility_count': int(np.sum(ac_removed_by_eligibility_mask)), 'closed_count': int(np.sum(ac_closed_mask)), 'closed_ratio': float(np.mean(ac_closed_mask)), 'expanded_root_count': int(np.sum(ac_expanded_mask)), 'expanded_root_ratio': float(np.mean(ac_expanded_mask)), 'ac_extraction_summary': ac_result['summary'], 'closing_summary': closing_summary, 'closing_zero_hops_core_identity_required': closing_is_identity_configuration, 'closing_zero_hops_core_identity_verified': closing_identity_check}, 'pure_structure_roots': {'source': 'pure_hks_display_norm', 'selection': 'display_norm >= min_display_percent / 100', 'min_display_percent': float(PURE_STRUCTURE_ROOT_MIN_DISPLAY_PERCENT), 'min_display_norm': float(PURE_STRUCTURE_ROOT_MIN_DISPLAY_PERCENT / 100.0), 'selected_count': int(np.sum(pure_structure_root_mask)), 'selected_ratio': float(np.mean(pure_structure_root_mask)), 'eligible_selected_count': int(np.sum(pure_hks_suppression_root_mask)), 'eligible_selected_ratio': float(np.mean(pure_hks_suppression_root_mask)), 'overlap_with_ac_expanded_count': int(np.sum(pure_structure_root_mask & ac_expanded_mask)), 'added_non_ac_count': int(np.sum(pure_structure_root_mask & ~ac_expanded_mask)), 'combined_growth_root_count': int(np.sum(growth_root_mask)), 'combined_growth_root_ratio': float(np.mean(growth_root_mask))}, 'pure_hks_grown_open_boundary': {'source_roots': 'ac_expanded_mask', 'growth_response': 'pure_hks_response', 'growth_rule': 'same descending response and hop budgets as combined growth', 'includes_pure_structure_roots': False, 'root_count': int(np.sum(ac_expanded_mask)), 'grown_only_count': int(np.sum(pure_hks_open_boundary_growth_result['grown_only_mask'])), 'mask_point_count': int(np.sum(pure_hks_grown_open_boundary_mask)), 'mask_point_ratio': float(np.mean(pure_hks_grown_open_boundary_mask))}, 'pure_hks': response_summary, 'display_q01': float(display_q01), 'display_q99': float(display_q99), 'sv_suppressed_display_q01': sv_suppressed_display_q01, 'sv_suppressed_display_q99': sv_suppressed_display_q99, 'va_hks_suppressed_display_q01': va_hks_suppressed_display_q01, 'va_hks_suppressed_display_q99': va_hks_suppressed_display_q99, 'va_hks_direct_suppressed_display_q01': va_hks_direct_suppressed_display_q01, 'va_hks_direct_suppressed_display_q99': va_hks_direct_suppressed_display_q99, 'va_hks_growth_suppressed_display_q01': va_hks_growth_suppressed_display_q01, 'va_hks_growth_suppressed_display_q99': va_hks_growth_suppressed_display_q99, 'display_level_counts': display_level_counts, 'root_display_level_counts': root_display_level_counts, 'root_hop_budget_source': 'display_level_from_q1_q99_normalized_response_via_configurable_hop_table', 'display_level_hop_budgets': list(level_hop_budgets), 'display': {'display_q01': float(display_q01), 'display_q99': float(display_q99), 'display_level_counts': display_level_counts, 'root_display_level_counts': root_display_level_counts, 'root_hop_budget_source': 'display_level_from_q1_q99_normalized_response_via_configurable_hop_table', 'display_level_hop_budgets': list(level_hop_budgets)}, 'empirical_rank_diagnostics': {'used_for_root_hop_budget': False, 'rank_cuts': [float(value) for value in PURE_RANK_CUTS], 'rank_based_root_level_counts': rank_diagnostic_root_counts}, 'root_budget': root_budget_summary, 'growth': {'grown_only_count': int(np.sum(grown_only_mask)), 'grown_only_ratio': float(np.mean(grown_only_mask)), 'final_suppression_count': int(np.sum(suppression_region_mask)), 'final_suppression_ratio': float(np.mean(suppression_region_mask)), 'suppression_contains_all_expanded_roots': bool(np.all(suppression_region_mask[ac_expanded_mask])), 'suppression_contains_all_pure_structure_roots': bool(np.all(suppression_region_mask[pure_hks_suppression_root_mask])), 'suppression_contains_all_combined_growth_roots': bool(np.all(suppression_region_mask[growth_root_mask]))}, 'va_hks_suppression_modes': {'use_mask_root_growth_for_pipeline': va_hks_use_mask_root_growth, 'active_mode': va_hks_active_suppression_mode, 'root_source': 'suppression_region_mask', 'root_count': int(np.sum(boundary_mask)), 'nonzero_growth_percentile': float(VA_HKS_NONZERO_GROWTH_PERCENTILE), 'growth_threshold': va_hks_growth_threshold, 'display_q01': float(va_hks_display_q01), 'display_q99': float(va_hks_display_q99), 'display_level_hop_budgets': list(va_hks_level_hop_budgets), 'root_display_level_counts': _count_levels(va_hks_growth_display_level, boundary_mask), 'direct_false_mode': {'suppression_mask_source': 'suppression_region_mask', 'suppression_mask_count': int(np.sum(boundary_mask)), 'suppression_mask_ratio': float(np.mean(boundary_mask)), 'attenuation': va_hks_direct_suppression_summary}, 'growth_true_mode': {'suppression_mask_source': 'va_hks_growth_suppression_mask', 'grown_only_count': int(np.sum(va_hks_growth_grown_only_mask)), 'grown_only_ratio': float(np.mean(va_hks_growth_grown_only_mask)), 'suppression_mask_count': int(np.sum(va_hks_growth_suppression_mask)), 'suppression_mask_ratio': float(np.mean(va_hks_growth_suppression_mask)), 'suppression_contains_all_roots': bool(np.all(va_hks_growth_suppression_mask[boundary_mask])), 'depth_counts': {f'depth_{depth}_count': int(np.sum(va_hks_growth_result['min_growth_depth'] == depth)) for depth in range(max(va_hks_level_hop_budgets, default=0) + 1)}, 'attenuation': va_hks_growth_suppression_summary}}, 'boundary_branch_suppression': {'sv_mask_source': 'suppression_region_mask', 'va_hks_mask_source': va_hks_active_suppression_mode, 'keep_weight': keep_weight, 'mask_count': int(np.sum(boundary_mask)), 'mask_ratio': float(np.mean(boundary_mask)), 'va_hks_active_mask_count': int(np.sum(va_hks_active_suppression_mask)), 'va_hks_active_mask_ratio': float(np.mean(va_hks_active_suppression_mask)), 'display_reference_ranges': {'sv': {'q01': sv_display_q01, 'q99': sv_display_q99, 'source': 'raw_sv_response'}, 'sv_suppressed_self_stretch': {'q01': sv_suppressed_display_q01, 'q99': sv_suppressed_display_q99, 'source': 'sv_boundary_suppressed'}, 'va_hks': {'q01': va_hks_display_q01, 'q99': va_hks_display_q99, 'source': 'raw_va_hks_response'}, 'va_hks_suppressed_self_stretch': {'q01': va_hks_suppressed_display_q01, 'q99': va_hks_suppressed_display_q99, 'source': 'active_va_hks_boundary_suppressed'}, 'va_hks_direct_suppressed_self_stretch': {'q01': va_hks_direct_suppressed_display_q01, 'q99': va_hks_direct_suppressed_display_q99, 'source': 'va_hks_direct_mask_suppressed'}, 'va_hks_growth_suppressed_self_stretch': {'q01': va_hks_growth_suppressed_display_q01, 'q99': va_hks_growth_suppressed_display_q99, 'source': 'va_hks_growth_mask_suppressed'}}, 'sv': sv_suppression_summary, 'va_hks': va_hks_suppression_summary, 'va_hks_direct_false_mode': va_hks_direct_suppression_summary, 'va_hks_growth_true_mode': va_hks_growth_suppression_summary}, 'minimum_fusion_debug': {'normalization': 'independent_exact_minmax_01', 'fusion': 'pointwise_minimum', 'sv_fusion_mode': sv_fusion_mode, 'both_masked': {'sv_source': 'sv_boundary_suppressed', 'va_hks_source': f'va_hks_boundary_suppressed[{va_hks_active_suppression_mode}]', 'sv_normalized': _summarize_score_vector(sv_boundary_suppressed_minmax), 'va_hks_normalized': _summarize_score_vector(va_hks_boundary_suppressed_minmax), 'fusion_result': _summarize_score_vector(fusion_both_masked_min)}, 'va_hks_only_masked': {'sv_source': 'sv_response', 'va_hks_source': f'va_hks_boundary_suppressed[{va_hks_active_suppression_mode}]', 'sv_normalized': _summarize_score_vector(sv_response_minmax), 'va_hks_normalized': _summarize_score_vector(va_hks_boundary_suppressed_minmax), 'fusion_result': _summarize_score_vector(fusion_va_hks_only_masked_min)}, 'active_fusion_key': active_fusion_summary_key, 'active_fusion_result': _summarize_score_vector(active_fusion_min), 'downstream_seed_source': active_fusion_summary_key, 'semantic_sv_multiplication_applied': True, 'seed_generation_applied': True}, 'semantic_gate': {'input_source': f'active_fusion_min[{sv_fusion_mode}]', 'threshold_source': 'nonzero_semantic_confidence_percentile', 'nonzero_percentile': semantic_gate_percentile, 'threshold': semantic_gate_threshold, 'comparison_for_keep': 'semantic_confidence >= threshold', 'replacement_value_below_threshold': semantic_gate_replacement_value, 'threshold_below_replacement_value': bool(semantic_gate_threshold < semantic_gate_replacement_value), 'pass_count': int(np.sum(semantic_gate_pass_mask)), 'pass_ratio': float(np.mean(semantic_gate_pass_mask)), 'replaced_count': int(np.sum(~semantic_gate_pass_mask)), 'replaced_ratio': float(np.mean(~semantic_gate_pass_mask)), 'replacement_raised_count': int(np.sum(semantic_confidence_gated > semantic_confidence)), 'replacement_lowered_count': int(np.sum(semantic_confidence_gated < semantic_confidence)), 'output': _summarize_score_vector(semantic_confidence_gated)}, 'original_sv_gate': {'input_source': "base_result['c_abs_raw_feature']", 'threshold_source': 'all_point_original_sv_percentile', 'percentile': original_sv_gate_percentile, 'threshold': original_sv_gate_threshold, 'comparison_for_keep': 'original_sv >= threshold', 'replacement_value_below_threshold': original_sv_gate_replacement_value, 'pass_count': int(np.sum(original_sv_gate_pass_mask)), 'pass_ratio': float(np.mean(original_sv_gate_pass_mask)), 'zeroed_count': int(np.sum(~original_sv_gate_pass_mask)), 'zeroed_ratio': float(np.mean(~original_sv_gate_pass_mask)), 'output': _summarize_score_vector(original_sv_gated)}, 'high_confidence_seed': {'pipeline_source': f'minimum_fusion_debug.{active_fusion_summary_key}', 'semantic_confidence_source': 'semantic_confidence_powered', 'original_sv_source': 'original_sv_gated', 'formula': 'seed_confidence = original_sv_gated * (semantic_confidence_gated ** semantic_confidence_exponent)', 'semantic_confidence_exponent': semantic_confidence_exponent, 'original_sv_gated': _summarize_score_vector(original_sv_gated), 'semantic_confidence_gated': _summarize_score_vector(semantic_confidence_gated), 'semantic_confidence_powered': _summarize_score_vector(semantic_confidence_powered), 'seed_confidence_raw': _summarize_score_vector(seed_confidence), 'normalization': 'all_point_q1_q99_with_nonzero_q99_fallback', 'normalization_q01': seed_confidence_normalization_q01, 'normalization_raw_q99': seed_confidence_normalization_raw_q99, 'normalization_effective_q99': seed_confidence_normalization_effective_q99, 'normalization_fallback_applied': seed_confidence_normalization_fallback_applied, 'normalization_fallback_rule': 'if raw all-point Q99 == 0 and positive scores exist, effective Q99 = minimum strictly positive score', 'display_q01': seed_confidence_display_q01, 'display_effective_q99': seed_confidence_display_q99, 'threshold_mode': 'fixed_normalized_threshold', 'seed_threshold': high_confidence_seed_threshold, 'seed_comparison': 'seed_confidence_norm >= seed_threshold', 'seed_point_count_before_boundary_filter': int(np.sum(high_confidence_seed_mask_before_boundary_filter)), 'pure_hks_grown_open_boundary_filter': {'enabled': high_confidence_seed_boundary_filter_enabled, 'mask_source': 'pure_hks_grown_open_boundary_mask', 'mask_root_source': 'ac_expanded_mask_only', 'includes_pure_structure_roots': False, 'operation': 'delete overlapping seed points before connected components', 'removed_seed_point_count': int(np.sum(high_confidence_seed_boundary_removed_mask)), 'remaining_seed_point_count': int(np.sum(high_confidence_seed_mask))}, 'seed_point_count': int(np.sum(high_confidence_seed_mask)), 'seed_point_ratio': float(np.mean(high_confidence_seed_mask)), 'connected_component_filtering_applied': True, 'component_boundary_filter_applied': bool(seed_component_boundary_filter_enable), 'component_boundary_filter_override_applied': bool(seed_component_boundary_filter_override_applied), 'component_mean_pruning_applied': False, 'sgcr_growth_applied': True, 'sgcr_label_aware_merge_applied': bool(merge_enable), 'sgcr_boundary_seed_attachment_applied': False, 'sgcr_growth_ratio_filter_applied': False, 'sgcr_minimum_candidate_filter_applied': False, 'sgcr_merge_and_post_filters_applied': False}, 'seed_confidence_percentile_components': seed_confidence_percentile_component_result['summary'], 'seed_confidence_spatial_components': seed_confidence_spatial_component_result['summary'], 'q_z_consistency_repair': q_z_consistency_repair_audit, 'seed_confidence_view_y_morphology_grouping': seed_confidence_view_y_morphology_grouping_result['summary'], 'seed_confidence_component_closing': seed_confidence_component_closing_result['summary'], 'view_q_to_view_z_component_mapping': view_q_to_view_z_mapping_result['summary'], 'seed_component_growth': seed_component_growth_result['summary'], 'depth': {f'depth_{depth}_count': int(np.sum(min_growth_depth == depth)) for depth in range(max(level_hop_budgets, default=0) + 1)}}
    result: dict[str, Any] = {'base_result': base_result, 'exp14_module': exp14, 'sample_id': sample_id, 'points': points, 'gt_mask': gt_mask, 'neighbor_indices': neighbor_indices, 'component_aware_suppression_enable': component_aware_suppression_enable, 'ac_component_aware_suppression_enabled': ac_component_aware_suppression_enabled, 'pure_hks_component_aware_suppression_enabled': pure_hks_component_aware_suppression_enabled, 'pure_hks_normal_suppression_mode': pure_hks_normal_suppression_mode, 'pure_hks_soft_attenuation_factor': float(pure_hks_soft_attenuation_factor), 'suppression_mode': suppression_mode, 'suppression_strategy': suppression_strategy, 'suppression_combination_name': suppression_combination_name, 'suppression_strategy_override_active': bool(suppression_strategy_override_active), 'suppression_strategy_audit': suppression_strategy_audit, 'resolved_ac_config': resolved_ac_config, 'ac_component_grouping_mode': ac_component_grouping_mode, 'ac_component_group_max_gap_edges': ac_component_group_max_gap_edges, 'ac_component_debug_enabled': bool(EXP21_AC_COMPONENT_DEBUG_ENABLE), 'max_ac_component_gap_audit_hops': int(MAX_AC_COMPONENT_GAP_AUDIT_HOPS), 'ac_component_min_relative_to_largest': float(AC_COMPONENT_MIN_RELATIVE_TO_LARGEST), 'pure_hks_normal_min_point_fraction': float(PURE_HKS_NORMAL_MIN_POINT_FRACTION), 'pure_hks_normal_min_relative_to_largest': float(PURE_HKS_NORMAL_MIN_RELATIVE_TO_LARGEST), 'suppression_previews': suppression_previews, 'confidence_previews': confidence_previews, 'pure_hks_response': pure_hks_response, 'pure_hks_rank_percentile': rank_percentile, 'pure_hks_display_norm': display_norm, 'pure_hks_display_level': display_level, 'display_level_hop_budgets': np.asarray(level_hop_budgets, dtype=np.int32), 'pure_hks_normal_threshold': normal_threshold, 'pure_hks_nonzero_growth_percentile': float(PURE_NONZERO_GROWTH_PERCENTILE), 'ac_seed_mask': ac_seed_mask, 'raw_ac_mask': ac_seed_mask, 'raw_ac_component_labels': raw_ac_component_labels, 'raw_ac_component_reports': raw_ac_component_reports, 'repaired_ac_group_labels': repaired_ac_group_labels, 'repaired_ac_group_reports': repaired_ac_group_reports, 'raw_component_to_repaired_group': raw_component_to_repaired_group, 'repaired_group_to_raw_components': repaired_group_to_raw_components, 'ac_grouping_repair_edge_reports': ac_grouping_repair_edge_reports, 'raw_ac_eligible_core_mask': raw_ac_eligible_core_mask, 'repaired_ac_eligible_core_mask': repaired_ac_eligible_core_mask, 'ac_eligible_core_mask': ac_eligible_core_mask, 'ac_removed_by_eligibility_mask': ac_removed_by_eligibility_mask, 'ac_removed_by_group_eligibility_mask': ac_removed_by_group_eligibility_mask, 'ac_fragments_rescued_by_grouping_mask': ac_fragments_rescued_by_grouping_mask, 'ac_closed_mask': ac_closed_mask, 'ac_expanded_mask': ac_expanded_mask, 'ac_response_growth_mask': np.asarray(selected_suppression['ac_suppression_mask'] & ~selected_suppression['ac_expanded_mask'], dtype=bool), 'final_ac_suppression_mask': np.asarray(selected_suppression['ac_suppression_mask'], dtype=bool), 'ac_component_connectivity_audit': ac_component_connectivity_audit, 'ac_component_reports': ac_component_reports, 'ac_closing_zero_hops_core_identity_verified': closing_identity_check, 'ac_component_labels': suppression_previews['ac_eligibility']['component_labels'], 'ac_component_records': suppression_previews['ac_eligibility']['component_records'], 'legacy_ac_eligible_core_mask': suppression_previews['legacy']['ac_core_mask'], 'new_ac_eligible_core_mask': suppression_previews['new']['ac_core_mask'], 'legacy_ac_closed_mask': suppression_previews['legacy']['ac_closed_mask'], 'new_ac_closed_mask': suppression_previews['new']['ac_closed_mask'], 'legacy_ac_expanded_mask': suppression_previews['legacy']['ac_expanded_mask'], 'new_ac_expanded_mask': suppression_previews['new']['ac_expanded_mask'], 'pure_structure_root_mask': pure_structure_root_mask, 'pure_hks_normal_candidate_mask': pure_structure_root_mask, 'pure_hks_normal_component_labels': suppression_previews['pure_hks_eligibility']['component_labels'], 'pure_hks_normal_component_records': suppression_previews['pure_hks_eligibility']['component_records'], 'pure_hks_component_eligible': suppression_previews['pure_hks_eligibility']['new_eligible_mask'], 'pure_hks_hard_component_mask': selected_suppression['pure_hks_hard_component_mask'], 'pure_hks_soft_component_mask': selected_suppression['pure_hks_soft_component_mask'], 'pure_hks_neutral_component_mask': selected_suppression['pure_hks_neutral_component_mask'], 'pure_hks_attenuation_factor_selected': selected_suppression['pure_hks_attenuation_factor'], 'ac_attenuation_factor_selected': selected_suppression['ac_attenuation_factor'], 'combined_attenuation_factor_selected': selected_suppression['combined_attenuation_factor'], 'pure_hks_suppression_root_mask': pure_hks_suppression_root_mask, 'legacy_pure_hks_eligible_root_mask': suppression_previews['legacy']['pure_hks_root_mask'], 'new_pure_hks_eligible_root_mask': suppression_previews['new']['pure_hks_root_mask'], 'growth_root_mask': growth_root_mask, 'pure_structure_root_min_display_percent': float(PURE_STRUCTURE_ROOT_MIN_DISPLAY_PERCENT), 'root_hop_budget': root_hop_budget, 'grown_only_mask': grown_only_mask, 'suppression_region_mask': suppression_region_mask, 'legacy_ac_suppression_mask': suppression_previews['legacy']['ac_suppression_mask'], 'new_ac_suppression_mask': suppression_previews['new']['ac_suppression_mask'], 'legacy_pure_hks_normal_suppression_mask': suppression_previews['legacy']['pure_hks_suppression_mask'], 'new_pure_hks_normal_suppression_mask': suppression_previews['new']['pure_hks_suppression_mask'], 'legacy_total_suppression_mask': suppression_previews['legacy']['total_suppression_mask'], 'new_total_suppression_mask': suppression_previews['new']['total_suppression_mask'], 'pure_hks_open_boundary_root_hop_budget': pure_hks_open_boundary_root_hop_budget, 'pure_hks_open_boundary_grown_only_mask': np.asarray(pure_hks_open_boundary_growth_result['grown_only_mask'], dtype=bool), 'pure_hks_grown_open_boundary_mask': pure_hks_grown_open_boundary_mask, 'boundary_response_keep_weight': keep_weight, 'sv_response': sv_response, 'sv_legacy_preview': confidence_previews['legacy']['sv_suppressed'], 'sv_new_preview': confidence_previews['new']['sv_suppressed'], 'sv_soft_preview': confidence_previews['new_ac__soft_hks']['sv_suppressed'], 'sv_boundary_suppressed': sv_boundary_suppressed, 'sv_branch_suppression_mask': selected_confidence['sv_branch_suppression_mask'], 'sv_effective_attenuation_factor': selected_confidence['sv_effective_attenuation_factor'], 'sv_response_minmax': sv_response_minmax, 'sv_boundary_suppressed_minmax': sv_boundary_suppressed_minmax, 'va_hks_response': va_hks_response, 'va_hks_legacy_preview': confidence_previews['legacy']['va_hks_suppressed'], 'va_hks_new_preview': confidence_previews['new']['va_hks_suppressed'], 'va_hks_soft_preview': confidence_previews['new_ac__soft_hks']['va_hks_suppressed'], 'va_hks_use_mask_root_growth': va_hks_use_mask_root_growth, 'va_hks_active_suppression_mode': va_hks_active_suppression_mode, 'va_hks_active_suppression_mask': va_hks_active_suppression_mask, 'va_hks_branch_suppression_mask': selected_confidence['va_hks_branch_suppression_mask'], 'va_hks_growth_blocked_mask': selected_confidence['va_hks_growth_blocked_mask'], 'va_hks_mask_root_growth_allowed': selected_confidence['va_hks_mask_root_growth_allowed'], 'va_hks_mask_root_growth_effective': selected_confidence['va_hks_mask_root_growth_effective'], 'va_hks_direct_mask_suppressed': va_hks_direct_mask_suppressed, 'va_hks_nonzero_growth_percentile': float(VA_HKS_NONZERO_GROWTH_PERCENTILE), 'va_hks_display_lower_percentile': float(VA_HKS_DISPLAY_LOWER_PERCENTILE), 'va_hks_display_upper_percentile': float(VA_HKS_DISPLAY_UPPER_PERCENTILE), 'va_hks_growth_threshold': va_hks_growth_threshold, 'va_hks_growth_display_norm': va_hks_growth_display_norm, 'va_hks_growth_display_level': va_hks_growth_display_level, 'va_hks_display_level_hop_budgets': np.asarray(va_hks_level_hop_budgets, dtype=np.int32), 'va_hks_growth_root_hop_budget': va_hks_growth_root_hop_budget, 'va_hks_growth_grown_only_mask': va_hks_growth_grown_only_mask, 'va_hks_growth_suppression_mask': va_hks_growth_suppression_mask, 'va_hks_growth_mask_suppressed': va_hks_growth_mask_suppressed, 'va_hks_growth_min_depth': va_hks_growth_result['min_growth_depth'], 'va_hks_growth_best_remaining': va_hks_growth_result['best_remaining'], 'va_hks_growth_parent': va_hks_growth_result['growth_parent'], 'va_hks_growth_parent_remaining': va_hks_growth_result['growth_parent_remaining'], 'va_hks_boundary_suppressed': va_hks_boundary_suppressed, 'va_hks_effective_attenuation_factor': selected_confidence['va_hks_effective_attenuation_factor'], 'va_hks_boundary_suppressed_minmax': va_hks_boundary_suppressed_minmax, 'fusion_both_masked_min': fusion_both_masked_min, 'fusion_va_hks_only_masked_min': fusion_va_hks_only_masked_min, 'sv_fusion_mode': sv_fusion_mode, 'active_fusion_summary_key': active_fusion_summary_key, 'active_fusion_min': active_fusion_min, 'original_sv': original_sv, 'original_sv_display_q01': original_sv_display_q01, 'original_sv_display_q99': original_sv_display_q99, 'original_sv_gate_percentile': original_sv_gate_percentile, 'original_sv_gate_threshold': original_sv_gate_threshold, 'original_sv_gate_replacement_value': original_sv_gate_replacement_value, 'original_sv_gate_pass_mask': original_sv_gate_pass_mask, 'original_sv_gated': original_sv_gated, 'semantic_confidence': semantic_confidence, 'semantic_confidence_legacy_preview': confidence_previews['legacy']['semantic_confidence'], 'semantic_confidence_new_preview': confidence_previews['new']['semantic_confidence'], 'semantic_confidence_soft_preview': confidence_previews['new_ac__soft_hks']['semantic_confidence'], 'semantic_gate_nonzero_percentile': semantic_gate_percentile, 'semantic_gate_threshold': semantic_gate_threshold, 'semantic_gate_replacement_value': semantic_gate_replacement_value, 'semantic_gate_pass_mask': semantic_gate_pass_mask, 'semantic_confidence_gated': semantic_confidence_gated, 'semantic_confidence_exponent': semantic_confidence_exponent, 'semantic_confidence_powered': semantic_confidence_powered, 'seed_confidence': seed_confidence, 'seed_confidence_norm': seed_confidence_norm, 'seed_confidence_legacy_preview': confidence_previews['legacy']['seed_confidence'], 'seed_confidence_new_preview': confidence_previews['new']['seed_confidence'], 'seed_confidence_soft_preview': confidence_previews['new_ac__soft_hks']['seed_confidence'], 'seed_confidence_norm_legacy_preview': confidence_previews['legacy']['seed_confidence_norm'], 'seed_confidence_norm_new_preview': confidence_previews['new']['seed_confidence_norm'], 'seed_confidence_norm_soft_preview': confidence_previews['new_ac__soft_hks']['seed_confidence_norm'], 'legacy_seed_preview_mask': confidence_previews['legacy']['high_confidence_seed_mask'], 'new_seed_preview_mask': confidence_previews['new']['high_confidence_seed_mask'], 'soft_seed_preview_mask': confidence_previews['new_ac__soft_hks']['high_confidence_seed_mask'], 'seed_confidence_normalization_q01': seed_confidence_normalization_q01, 'seed_confidence_normalization_raw_q99': seed_confidence_normalization_raw_q99, 'seed_confidence_normalization_effective_q99': seed_confidence_normalization_effective_q99, 'seed_confidence_normalization_fallback_applied': seed_confidence_normalization_fallback_applied, 'seed_confidence_display_lower_percentile': float(SEED_CONFIDENCE_DISPLAY_LOWER_PERCENTILE), 'seed_confidence_display_upper_percentile': float(SEED_CONFIDENCE_DISPLAY_UPPER_PERCENTILE), 'seed_confidence_display_q01': seed_confidence_display_q01, 'seed_confidence_display_q99': seed_confidence_display_q99, 'seed_confidence_p80_component_extraction_enable': seed_confidence_p80_component_extraction_enable, 'seed_confidence_component_percentile': seed_confidence_component_percentile, 'seed_confidence_component_threshold': seed_confidence_percentile_component_result['threshold'], 'seed_confidence_component_valid_mask': seed_confidence_percentile_component_result['valid_mask'], 'raw_seed_confidence_view_z_valid_mask': raw_seed_confidence_spatial_component_result['valid_mask'], 'raw_seed_confidence_view_z_components': raw_seed_confidence_spatial_component_result['components'], 'raw_seed_confidence_view_z_component_labels': raw_seed_confidence_spatial_component_result['component_labels'], 'q_z_consistency_repair_enable': q_to_z_consistency_repair_enable, 'q_z_consistency_repair_audit': q_z_consistency_repair_audit, 'q_z_consistency_added_q_support_mask': q_z_consistency_result['added_q_support_mask'], 'seed_confidence_components': seed_confidence_percentile_component_result['components'], 'seed_confidence_component_labels': seed_confidence_percentile_component_result['component_labels'], 'seed_confidence_spatial_components': seed_confidence_spatial_component_result['components'], 'seed_confidence_spatial_component_labels': seed_confidence_spatial_component_result['component_labels'], 'seed_confidence_view_z_closing_input_mask': seed_confidence_view_y_morphology_grouping_result['closing_input_mask'], 'seed_confidence_seed_containing_view_z_component_ids': seed_confidence_view_y_morphology_grouping_result['active_spatial_view_z_component_ids'], 'seed_confidence_seedless_view_z_component_ids': seed_confidence_view_y_morphology_grouping_result['removed_seedless_spatial_view_z_component_ids'], 'seed_confidence_spatial_to_morphology_group_id': seed_confidence_view_y_morphology_grouping_result['spatial_to_morphology_group_id'], 'seed_confidence_view_y_component_grouping_reports': seed_confidence_view_y_morphology_grouping_result['y_component_reports'], 'seed_confidence_morphology_group_reports': seed_confidence_view_y_morphology_grouping_result['morphology_group_reports'], 'seed_confidence_component_closing_dilation_hops': seed_confidence_component_closing_dilation_hops, 'seed_confidence_component_closing_erosion_hops': seed_confidence_component_closing_erosion_hops, 'seed_confidence_component_closing_keep_original': seed_confidence_component_closing_keep_original, 'seed_confidence_closed_components': seed_confidence_component_closing_result['closed_components'], 'seed_confidence_component_closing_reports': seed_confidence_component_closing_result['component_reports'], 'seed_confidence_morphology_group_closing_reports': seed_confidence_component_closing_result['morphology_group_closing_reports'], 'seed_confidence_component_max_growth_multiples': seed_confidence_component_closing_result['max_growth_multiples'], 'seed_confidence_closed_union_mask': seed_confidence_component_closing_result['closed_union_mask'], 'seed_confidence_closed_claim_count': seed_confidence_component_closing_result['closed_claim_count'], 'seed_confidence_closed_overlap_mask': seed_confidence_component_closing_result['closed_overlap_mask'], 'seed_confidence_closed_visualization_component_labels': seed_confidence_component_closing_result['closed_visualization_component_labels'], 'seed_confidence_closed_max_growth_multiple_by_point': seed_confidence_component_closing_result['closed_max_growth_multiple_by_point'], 'high_confidence_seed_threshold': high_confidence_seed_threshold, 'high_confidence_seed_boundary_filter_enabled': high_confidence_seed_boundary_filter_enabled, 'high_confidence_seed_mask_before_boundary_filter': high_confidence_seed_mask_before_boundary_filter, 'high_confidence_seed_boundary_removed_mask': high_confidence_seed_boundary_removed_mask, 'high_confidence_seed_mask': high_confidence_seed_mask, 'seed_component_boundary_filter_enable': seed_component_boundary_filter_enable, 'seed_component_boundary_filter_override_applied': seed_component_boundary_filter_override_applied, 'seed_component_boundary_dilation_hops': seed_component_boundary_dilation_hops, 'seed_component_boundary_hit_ratio_threshold': seed_component_boundary_hit_ratio_threshold, 'seed_component_boundary_filter_mask': seed_component_boundary_filter_mask, 'seed_component_local_graph_enable': seed_component_local_graph_enable, 'seed_component_local_k': seed_component_local_k, 'seed_component_local_scale_k': seed_component_local_scale_k, 'seed_component_local_radius_factor': seed_component_local_radius_factor, 'seed_component_neighbor_indices': seed_component_neighbor_indices, 'seed_component_graph_summary': seed_component_graph_summary, 'seed_component_boundary_filter_reports': seed_component_growth_result['boundary_component_reports'], 'seed_component_min_size': seed_component_min_size, 'seed_component_red_sv_quantile': seed_component_growth_result['red_sv_quantile'], 'seed_component_red_sv_threshold': seed_component_growth_result['red_sv_threshold'], 'seed_component_view_c_sv_display_norm': seed_component_growth_result['view_c_sv_display_norm'], 'seed_component_size_sv_filter_reports': seed_component_growth_result['conditional_filter_component_reports'], 'seed_components': seed_component_growth_result['raw_components'], 'seed_component_labels': seed_component_labels, 'seed_boundary_filtered_components': seed_boundary_filtered_components, 'seed_boundary_filtered_component_labels': seed_boundary_filtered_component_labels, 'seed_boundary_filtered_mask': seed_boundary_filtered_mask, 'seed_boundary_removed_components': seed_boundary_removed_components, 'seed_boundary_removed_component_labels': seed_boundary_removed_component_labels, 'seed_boundary_removed_mask': seed_boundary_removed_mask, 'seed_size_filtered_components': seed_component_growth_result['kept_components'], 'seed_size_filtered_component_labels': seed_size_filtered_component_labels, 'seed_size_filtered_mask': seed_size_filtered_mask, 'seed_size_filtered_component_to_seed_confidence_component_ids': view_q_to_view_z_mapping_result['mapped_view_z_component_ids'], 'seed_size_filtered_component_max_growth_multiples': view_q_to_view_z_mapping_result['max_growth_multiples'], 'seed_size_filtered_component_max_growth_multiple_by_point': view_q_to_view_z_mapping_result['max_growth_multiple_by_point'], 'seed_size_filtered_component_mapping_reports': view_q_to_view_z_mapping_result['mapping_reports'], 'seed_size_removed_components': seed_component_growth_result['removed_components'], 'seed_size_removed_component_labels': seed_size_removed_component_labels, 'seed_size_removed_mask': seed_size_removed_mask, 'seed_growth_support_norm': seed_growth_support_norm, 'seed_grow_reports': seed_component_growth_result['grow_reports'], 'seed_grown_components': seed_component_growth_result['grown_components'], 'seed_grown_component_labels': seed_grown_component_labels, 'seed_grown_mask': seed_grown_mask, 'seed_label_aware_merged_components': seed_label_aware_merged_components, 'seed_label_aware_merged_component_labels': seed_label_aware_merged_component_labels, 'seed_label_aware_merged_mask': seed_label_aware_merged_mask, 'seed_label_aware_merge_summary': seed_label_aware_merge_summary, 'sv_display_q01': sv_display_q01, 'sv_display_q99': sv_display_q99, 'va_hks_display_q01': va_hks_display_q01, 'va_hks_display_q99': va_hks_display_q99, 'sv_suppressed_display_q01': sv_suppressed_display_q01, 'sv_suppressed_display_q99': sv_suppressed_display_q99, 'va_hks_suppressed_display_q01': va_hks_suppressed_display_q01, 'va_hks_suppressed_display_q99': va_hks_suppressed_display_q99, 'va_hks_direct_suppressed_display_q01': va_hks_direct_suppressed_display_q01, 'va_hks_direct_suppressed_display_q99': va_hks_direct_suppressed_display_q99, 'va_hks_growth_suppressed_display_q01': va_hks_growth_suppressed_display_q01, 'va_hks_growth_suppressed_display_q99': va_hks_growth_suppressed_display_q99, 'min_growth_depth': min_growth_depth, 'best_remaining': growth_result['best_remaining'], 'growth_parent': growth_result['growth_parent'], 'growth_parent_remaining': growth_result['growth_parent_remaining'], 'summaries': summaries}
    summaries['invariants'] = {'array_shapes_match': all((np.asarray(result[name]).shape == (num_points,) for name in ('final_ac_suppression_mask', 'semantic_confidence', 'seed_confidence_norm', 'seed_label_aware_merged_component_labels'))), 'protected_disjoint_from_final_ac': not bool(np.any(np.asarray(result['suppression_strategy_audit'].get('defect_protected_component_mask', np.zeros(num_points, dtype=bool)), dtype=bool) & np.asarray(result['final_ac_suppression_mask'], dtype=bool)))}
    return result

def run_covert_frontend(base_result: dict[str, Any], config: Any, *, suppression_override_hook: Any | None=None, verbose: bool=False) -> dict[str, Any]:
    """Run the single frozen New-AC/Legacy-HKS frontend route."""
    sample_id = str(base_result['summaries']['load']['sample_id'])
    result = _run_covert_frontend_core(Path(f'{sample_id}.npy'), suppression_strategy=SUPPRESSION_STRATEGY_LEGACY, suppression_override_hook=suppression_override_hook, q_to_z_consistency_repair_enable=bool(config.switches.q_to_z_repair), seed_component_boundary_filter_enable=False, base_result_override=base_result, config=config, production_mainline=True, verbose=verbose)
    result.pop('suppression_previews', None)
    result.pop('confidence_previews', None)
    return result

def classify_pure_hks_components_by_final_ac(pure_hks_candidate_mask: np.ndarray, final_ac_suppression_mask: np.ndarray, neighbor_indices: list[np.ndarray] | np.ndarray) -> dict[str, Any]:
    """Split original H>=0.6 components by overlap with authoritative Final-AC.

    A candidate component is boundary-classified when *any* of its points
    overlaps ``final_ac_suppression_mask``.  The complete original component is
    then excluded from the normal root pool.  This classification never expands
    or replaces the authoritative Final-AC boundary itself.
    """
    candidate = np.asarray(pure_hks_candidate_mask, dtype=bool).reshape(-1)
    final_ac = _as_bool_mask(final_ac_suppression_mask, name='final_ac_suppression_mask', length=candidate.size)
    component_state = label_mask_components(candidate, neighbor_indices)
    component_labels = np.asarray(component_state['component_labels'], dtype=np.int32)
    boundary_mask = np.zeros(candidate.size, dtype=bool)
    normal_mask = np.zeros(candidate.size, dtype=bool)
    semantic_labels = np.zeros(candidate.size, dtype=np.int8)
    records: list[dict[str, Any]] = []
    for (component_id, component_value) in enumerate(component_state['components'], start=1):
        component = np.asarray(component_value, dtype=np.int64).reshape(-1)
        overlaps_final_ac = bool(np.any(final_ac[component]))
        component_class = 'boundary_hks_component' if overlaps_final_ac else 'normal_hks_component'
        target_mask = boundary_mask if overlaps_final_ac else normal_mask
        target_mask[component] = True
        semantic_labels[component] = 1 if overlaps_final_ac else 2
        records.append({'component_id': int(component_id), 'size': int(component.size), 'overlaps_final_ac_suppression_mask': overlaps_final_ac, 'component_class': component_class})
    if np.any(boundary_mask & normal_mask):
        raise AssertionError('Pure-HKS semantic component classes must be disjoint.')
    if not np.array_equal(boundary_mask | normal_mask, candidate):
        raise AssertionError('Pure-HKS semantic classes must partition H>=0.6 candidates.')
    return {'pure_hks_candidate_component_labels': component_labels, 'pure_hks_semantic_component_labels': semantic_labels, 'pure_hks_boundary_component_mask': boundary_mask, 'pure_hks_normal_component_mask': normal_mask, 'components': component_state['components'], 'component_records': records}

def _restrict_growth_result(growth_result: dict[str, Any], suppression_mask: np.ndarray, root_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Restrict an audit growth state after applying semantic boundary priority."""
    suppression = np.asarray(suppression_mask, dtype=bool).reshape(-1)
    roots = _as_bool_mask(root_mask, name='root_mask', length=suppression.size)
    restricted = {key: np.asarray(growth_result[key]).copy() for key in ('best_remaining', 'min_growth_depth', 'grown_only_mask', 'suppression_region_mask', 'growth_parent', 'growth_parent_remaining')}
    restricted['best_remaining'][~suppression] = -1
    restricted['min_growth_depth'][~suppression] = -1
    restricted['growth_parent'][~suppression] = -1
    restricted['growth_parent_remaining'][~suppression] = -1
    parent = restricted['growth_parent']
    has_parent = parent >= 0
    invalid_parent = has_parent.copy()
    invalid_parent[has_parent] = ~suppression[parent[has_parent]]
    parent[invalid_parent] = -1
    restricted['growth_parent_remaining'][invalid_parent] = -1
    restricted['suppression_region_mask'] = suppression.copy()
    restricted['grown_only_mask'] = suppression & ~roots
    return restricted

def _merge_partitioned_growth_results(*, ac_growth_result: dict[str, Any], normal_growth_result: dict[str, Any], final_ac_suppression_mask: np.ndarray, normal_structure_support_mask: np.ndarray, root_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Create one legacy-shaped audit state from disjoint AC/normal semantics."""
    final_ac = np.asarray(final_ac_suppression_mask, dtype=bool).reshape(-1)
    normal_support = _as_bool_mask(normal_structure_support_mask, name='normal_structure_support_mask', length=final_ac.size)
    roots = _as_bool_mask(root_mask, name='root_mask', length=final_ac.size)
    total = final_ac | normal_support
    merged: dict[str, np.ndarray] = {}
    for (key, dtype) in (('best_remaining', np.int32), ('min_growth_depth', np.int32), ('growth_parent', np.int64), ('growth_parent_remaining', np.int32)):
        merged[key] = np.full(final_ac.size, -1, dtype=dtype)
        ac_values = np.asarray(ac_growth_result[key], dtype=dtype).reshape(-1)
        normal_values = np.asarray(normal_growth_result[key], dtype=dtype).reshape(-1)
        merged[key][final_ac] = ac_values[final_ac]
        merged[key][normal_support] = normal_values[normal_support]
    merged['min_growth_depth'][roots] = 0
    merged['suppression_region_mask'] = total
    merged['grown_only_mask'] = total & ~roots
    return merged
