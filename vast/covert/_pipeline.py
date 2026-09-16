"""Private production runtime mechanically migrated from the frozen reference.

Only symbols reachable from the active consolidated call root are retained.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from vast.covert import _backbone, _frontend, _gtr, _score_compat
from vast.scoring.component_filter import SEED_STATE_STRONG_NORMAL, SEED_STATE_WEAK_NORMAL, build_component_filter_result
import vast.scoring.component_filter as _component_filter
GTR_AC_OPEN_BOUNDARY_SUPPORT_EXPAND_HOPS = 1
GTR_RESTITUTE_OPEN_BOUNDARY_SUPPORT_AFTER_DECISION = False
GTR_OPEN_BOUNDARY_RESTITUTION_MODE = 'after_decision' if GTR_RESTITUTE_OPEN_BOUNDARY_SUPPORT_AFTER_DECISION else 'disabled'
EXP22_LOCAL_NORMAL_U_CLEANUP_ENABLE = True
EXP22_LOCAL_NORMAL_U_CLEANUP_MODE = 'final_mask_only'
EXP22_LOCAL_NORMAL_U_CLEANUP_ALLOWED_MODES = frozenset({'final_mask_only', 'pre_component_classification'})

def _as_vector(values: Any, *, name: str, num_points: int, dtype: Any) -> np.ndarray:
    vector = np.asarray(values, dtype=dtype).reshape(-1)
    if vector.shape != (num_points,):
        raise ValueError(f'{name} must have shape ({num_points},), got {vector.shape}.')
    return vector

def _component_records_from_labels(labels: np.ndarray) -> list[dict[str, Any]]:
    """Convert a point-label array to the component records expected by exp17 views."""
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    records: list[dict[str, Any]] = []
    for component_id in sorted((int(label) for label in np.unique(labels) if int(label) > 0)):
        indices = np.flatnonzero(labels == component_id).astype(np.int64, copy=False)
        records.append({'component_id': component_id, 'indices': indices, 'size': int(indices.size)})
    return records

def _positive_component_ids(labels: np.ndarray) -> list[int]:
    """Return sorted positive IDs from a component-label vector."""
    return sorted((int(value) for value in np.unique(labels) if int(value) > 0))

def _component_sizes(labels: np.ndarray) -> dict[int, int]:
    """Count points for every positive component ID."""
    return {component_id: int(np.sum(labels == component_id)) for component_id in _positive_component_ids(labels)}

def _build_component_rule_inputs(exp21_result: dict[str, Any], preview_closed_labels_core: np.ndarray) -> list[dict[str, int]]:
    """Adapt Exp21/preview outputs to the Exp25-validated Q lineage contract."""
    num_points = int(np.asarray(exp21_result['points']).shape[0])
    q_labels = _as_vector(exp21_result['seed_size_filtered_component_labels'], name='seed_size_filtered_component_labels', num_points=num_points, dtype=np.int32)
    y_labels = _as_vector(exp21_result['seed_label_aware_merged_component_labels'], name='seed_label_aware_merged_component_labels', num_points=num_points, dtype=np.int32)
    z_labels = _as_vector(exp21_result['seed_confidence_component_labels'], name='seed_confidence_component_labels', num_points=num_points, dtype=np.int32)
    preview_labels = _as_vector(preview_closed_labels_core, name='component_rule_preview_closed_labels_core', num_points=num_points, dtype=np.int32)
    for (name, labels) in (('Q', q_labels), ('Y', y_labels), ('Z', z_labels), ('GTR preview', preview_labels)):
        if np.any(labels < 0):
            raise ValueError(f'{name} component labels contain negative IDs.')
    q_sizes = _component_sizes(q_labels)
    y_sizes = _component_sizes(y_labels)
    z_sizes = _component_sizes(z_labels)
    preview_sizes = _component_sizes(preview_labels)
    u_components = [np.asarray(component, dtype=np.int64).reshape(-1) for component in exp21_result['seed_grown_components']]
    expected_q_ids = list(range(1, len(u_components) + 1))
    if sorted(q_sizes) != expected_q_ids:
        raise AssertionError('View-U authoritative IDs are Q IDs, but Q labels and U proposal ordering do not agree.')
    grow_reports = {int(report['source_seed_component_id']): report for report in exp21_result['seed_grow_reports']}
    if set(grow_reports) != set(q_sizes):
        raise AssertionError('View-Q IDs and independent grow-report IDs disagree.')
    q_to_y: dict[int, int] = {}
    for merged_group in exp21_result['seed_label_aware_merge_summary']['merged_source_groups']:
        y_id = int(merged_group['merged_component_id'])
        for q_id_raw in merged_group['source_seed_component_ids']:
            q_id = int(q_id_raw)
            if q_id in q_to_y:
                raise AssertionError(f'Q{q_id} appears in multiple View-Y groups.')
            q_to_y[q_id] = y_id
    if set(q_to_y) != set(q_sizes):
        raise AssertionError('View-Y source groups do not cover every View-Q seed.')
    if set(q_to_y.values()) != set(y_sizes):
        raise AssertionError('View-Y labels and source-group IDs disagree.')
    q_to_z_reports = {int(report['view_q_seed_component_id']): report for report in exp21_result['seed_size_filtered_component_mapping_reports']}
    if set(q_to_z_reports) != set(q_sizes):
        raise AssertionError('View-Q IDs and authoritative Q-to-Z reports disagree.')
    z_closing_reports = {int(report['component_id']): report for report in exp21_result['seed_confidence_component_closing_reports']}
    w_proposal_reports = {int(index): report for (index, report) in enumerate(exp21_result['seed_confidence_morphology_group_closing_reports'], start=1)}
    inputs: list[dict[str, int]] = []
    for q_id in sorted(q_sizes):
        q_mask = q_labels == q_id
        q_indices = np.flatnonzero(q_mask)
        u_component = u_components[q_id - 1]
        if np.any((u_component < 0) | (u_component >= num_points)):
            raise AssertionError(f'View-U proposal for Q{q_id} has invalid indices.')
        if not np.all(np.isin(q_indices, u_component)):
            raise AssertionError(f'View-Q seed Q{q_id} is not contained in View U.')
        u_size = int(u_component.size)
        if int(grow_reports[q_id]['size']) != u_size:
            raise AssertionError(f'Q{q_id} grow report size disagrees with View U.')
        y_id = int(q_to_y[q_id])
        if not np.all(preview_labels[y_labels == y_id] == y_id):
            raise AssertionError(f'Preview GTR candidate Y{y_id} does not retain every original Y point.')
        if y_id not in preview_sizes:
            raise AssertionError(f'Preview GTR candidate Y{y_id} is missing.')
        q_to_z = q_to_z_reports[q_id]
        z_id = int(q_to_z['view_z_component_id'])
        containing_z_ids = sorted((int(value) for value in np.unique(z_labels[q_mask])))
        if containing_z_ids != [z_id]:
            raise AssertionError(f'Q{q_id}->Z labels {containing_z_ids} disagree with report Z{z_id}.')
        z_report = z_closing_reports.get(z_id)
        if z_report is None:
            raise AssertionError(f'Z{z_id} has no authoritative closing report.')
        z_size = int(z_sizes[z_id])
        if int(z_report['original_size']) != z_size:
            raise AssertionError(f'Z{z_id} report size disagrees with labels.')
        w_id = int(z_report['morphology_group_id'])
        w_size = int(z_report['morphology_group_closed_size'])
        w_proposal = w_proposal_reports.get(w_id)
        if w_proposal is None or int(w_proposal['closed_size']) != w_size:
            raise AssertionError(f'Z{z_id} authoritative W size disagrees with proposal W{w_id}.')
        inputs.append({'q_id': int(q_id), 'y_id': y_id, 'q_size': int(q_sizes[q_id]), 'u_size': u_size, 'y_size': int(y_sizes[y_id]), 'z_size': z_size, 'w_size': w_size, 'gtr_size': int(preview_sizes[y_id]), 'view_z_component_id': z_id, 'view_w_component_id': w_id})
    return inputs

def _build_component_rule_result(exp21_result: dict[str, Any], *, enabled: bool, preview_closed_labels_core: np.ndarray | None, thresholds: dict[str, int] | None=None) -> dict[str, Any]:
    """Build the auditable rule result without mutating Exp21's View-Y labels."""
    original_labels = np.asarray(exp21_result['seed_label_aware_merged_component_labels'], dtype=np.int32).reshape(-1)
    original_y_ids = _positive_component_ids(original_labels)
    original_point_count = int(np.sum(original_labels > 0))
    switch_source = "exp21_result['seed_confidence_p80_component_extraction_enable']"
    if not enabled:
        filtered_labels = original_labels.copy()
        rule_result: dict[str, Any] = {'enabled': False, 'switch_source': switch_source, 'classification_performed': False, 'seed_reports': [], 'y_reports': [], 'normal_y_ids': [], 'defect_y_ids': [], 'unknown_y_ids': original_y_ids}
    else:
        if preview_closed_labels_core is None:
            raise ValueError('Enabled component filtering requires preview closing labels.')
        rule_inputs = _build_component_rule_inputs(exp21_result, preview_closed_labels_core)
        rule_result = build_component_filter_result(rule_inputs, enabled=True, thresholds=thresholds)
        rule_result['switch_source'] = switch_source
        rule_result['classification_performed'] = True
        classified_y_ids = sorted(rule_result['normal_y_ids'] + rule_result['defect_y_ids'] + rule_result['unknown_y_ids'])
        if classified_y_ids != original_y_ids:
            raise AssertionError('Component-rule Y reports do not partition all original View-Y IDs.')
        filtered_labels = np.where(np.isin(original_labels, rule_result['normal_y_ids']), 0, original_labels).astype(np.int32, copy=False)
    retained_point_count = int(np.sum(filtered_labels > 0))
    overall = {'enabled': bool(enabled), 'switch_source': switch_source, 'number_of_original_y_components': int(len(original_y_ids)), 'number_of_normal_y': int(len(rule_result['normal_y_ids'])), 'number_of_defect_y': int(len(rule_result['defect_y_ids'])), 'number_of_unknown_y': int(len(rule_result['unknown_y_ids'])), 'number_of_y_removed_before_gtr': int(len(rule_result['normal_y_ids'])), 'original_candidate_point_count': original_point_count, 'retained_candidate_point_count': retained_point_count, 'removed_candidate_point_count': int(original_point_count - retained_point_count)}
    rule_result['overall'] = overall
    rule_result['filtered_candidate_labels'] = filtered_labels
    rule_result['filtered_candidate_mask'] = filtered_labels > 0
    return rule_result

def _build_local_normal_u_cleanup_request(exp21_result: dict[str, Any], component_rule_result: dict[str, Any], *, enabled: bool=EXP22_LOCAL_NORMAL_U_CLEANUP_ENABLE, mode: str=EXP22_LOCAL_NORMAL_U_CLEANUP_MODE) -> dict[str, Any]:
    """Map retained-Y strong/weak Normal seed evidence to authoritative U cores."""
    mode_key = str(mode).strip().lower()
    if mode_key not in EXP22_LOCAL_NORMAL_U_CLEANUP_ALLOWED_MODES:
        raise ValueError("EXP22_LOCAL_NORMAL_U_CLEANUP_MODE must be one of: 'final_mask_only', 'pre_component_classification'")
    num_points = int(np.asarray(exp21_result['points']).shape[0])
    strong_mask = np.zeros(num_points, dtype=bool)
    weak_mask = np.zeros(num_points, dtype=bool)
    seed_reports = component_rule_result.get('seed_reports')
    component_rule_available = bool(component_rule_result.get('enabled') and component_rule_result.get('classification_performed') and isinstance(seed_reports, list) and (len(seed_reports) > 0))
    retained_y_ids = {int(value) for key in ('defect_y_ids', 'unknown_y_ids') for value in component_rule_result.get(key) or []}
    normal_y_ids = {int(value) for value in component_rule_result.get('normal_y_ids') or []}
    u_components = [np.asarray(component, dtype=np.int64).reshape(-1) for component in exp21_result.get('seed_grown_components') or []]
    strong_components: list[dict[str, Any]] = []
    weak_components: list[dict[str, Any]] = []
    skipped_reports: list[dict[str, Any]] = []
    if component_rule_available:
        for report in seed_reports:
            q_id = int(report['q_id'])
            y_id = int(report['y_id'])
            seed_state = str(report.get('seed_state') or 'unknown')
            if seed_state not in {SEED_STATE_STRONG_NORMAL, SEED_STATE_WEAK_NORMAL}:
                skipped_reports.append({'q_id': q_id, 'y_id': y_id, 'seed_state': seed_state, 'reason': 'seed_state_not_normal'})
                continue
            if y_id in normal_y_ids:
                skipped_reports.append({'q_id': q_id, 'y_id': y_id, 'seed_state': seed_state, 'reason': 'whole_y_already_removed_as_normal'})
                continue
            if y_id not in retained_y_ids:
                skipped_reports.append({'q_id': q_id, 'y_id': y_id, 'seed_state': seed_state, 'reason': 'y_not_retained_as_defect_or_unknown'})
                continue
            if q_id <= 0 or q_id > len(u_components):
                raise AssertionError(f'Local Normal-U Q{q_id} has no authoritative U component.')
            u_indices = u_components[q_id - 1]
            if np.any((u_indices < 0) | (u_indices >= num_points)):
                raise AssertionError(f'Local Normal-U component U{q_id} contains invalid point indices.')
            unique_u_indices = np.unique(u_indices)
            component = {'q_id': q_id, 'u_id': q_id, 'y_id': y_id, 'seed_state': seed_state, 'u_size': int(unique_u_indices.size)}
            if seed_state == SEED_STATE_STRONG_NORMAL:
                strong_mask[unique_u_indices] = True
                strong_components.append(component)
            else:
                weak_mask[unique_u_indices] = True
                weak_components.append(component)
    union_mask = strong_mask | weak_mask
    requested_enabled = bool(enabled)
    if not requested_enabled:
        reason = 'disabled_by_exp22_switch'
    elif not component_rule_available:
        reason = 'component_rule_seed_evidence_unavailable'
    elif not np.any(union_mask):
        reason = 'no_local_normal_u_components_in_retained_y'
    else:
        reason = 'enabled'
    normal_components = strong_components + weak_components
    effective_enabled = bool(requested_enabled and component_rule_available and np.any(union_mask))
    return {'enabled': requested_enabled, 'effective_enabled': effective_enabled, 'mode': mode_key, 'component_rule_available': component_rule_available, 'reason': reason, 'local_strong_normal_u_mask': strong_mask, 'local_weak_normal_u_mask': weak_mask, 'local_normal_u_cleanup_mask': union_mask, 'strong_normal_u_components': strong_components, 'weak_normal_u_components': weak_components, 'normal_u_components': normal_components, 'strong_normal_q_ids': [int(item['q_id']) for item in strong_components], 'weak_normal_q_ids': [int(item['q_id']) for item in weak_components], 'normal_q_ids': [int(item['q_id']) for item in normal_components], 'normal_y_ids': sorted({int(item['y_id']) for item in normal_components}), 'skipped_seed_reports': skipped_reports, 'overall': {'enabled': requested_enabled, 'effective_enabled': effective_enabled, 'mode': mode_key, 'component_rule_available': component_rule_available, 'reason': reason, 'num_strong_normal_u_components': int(len(strong_components)), 'num_weak_normal_u_components': int(len(weak_components)), 'num_total_normal_u_components': int(len(normal_components)), 'requested_strong_normal_u_point_count': int(np.sum(strong_mask)), 'requested_weak_normal_u_point_count': int(np.sum(weak_mask)), 'requested_union_u_point_count': int(np.sum(union_mask))}}

def _build_exp21_gtr_adapter(exp21_result: dict[str, Any]) -> dict[str, Any]:
    """Expose exp21 outputs through the field contract consumed by exp17 GTR.

    This is intentionally a shallow copy of exp21's exp14 backbone result.  It
    retains the graph, aligned points, base views, and geometry metadata while
    replacing every SGCR/GTR-facing field with its exp21 counterpart.
    """
    points = np.asarray(exp21_result['points'], dtype=np.float64)
    num_points = int(points.shape[0])
    high_confidence_seed_mask = _as_vector(exp21_result['high_confidence_seed_mask'], name='high_confidence_seed_mask', num_points=num_points, dtype=bool)
    seed_component_labels = _as_vector(exp21_result['seed_size_filtered_component_labels'], name='seed_size_filtered_component_labels', num_points=num_points, dtype=np.int32)
    seed_component_mask = seed_component_labels > 0
    candidate_labels = _as_vector(exp21_result['seed_label_aware_merged_component_labels'], name='seed_label_aware_merged_component_labels', num_points=num_points, dtype=np.int32)
    candidate_mask = candidate_labels > 0
    semantic_confidence = _as_vector(exp21_result['semantic_confidence'], name='semantic_confidence', num_points=num_points, dtype=np.float64)
    seed_confidence = _as_vector(exp21_result['seed_confidence'], name='seed_confidence', num_points=num_points, dtype=np.float64)
    seed_confidence_norm = _as_vector(exp21_result['seed_confidence_norm'], name='seed_confidence_norm', num_points=num_points, dtype=np.float64)
    ac_open_boundary_mask = _as_vector(exp21_result['ac_closed_mask'], name='ac_closed_mask', num_points=num_points, dtype=bool)
    base_result = exp21_result['base_result']
    adapter = dict(base_result)
    adapter_summaries = dict(base_result.get('summaries', {}))
    exp21_growth_summary = exp21_result['summaries']['seed_component_growth']
    candidate_components = _component_records_from_labels(candidate_labels)
    candidate_summary = {'method': 'exp21_label_aware_merged_grown_components', 'num_components': int(len(candidate_components)), 'num_candidate_components': int(len(candidate_components)), 'candidate_point_count': int(np.sum(candidate_mask)), 'candidate_point_ratio': float(np.mean(candidate_mask)), 'seed_component_count': int(len([label for label in np.unique(seed_component_labels) if label > 0])), 'seed_point_count': int(np.sum(seed_component_mask)), 'seed_threshold': exp21_result.get('high_confidence_seed_threshold'), 'seed_min_size': exp21_result.get('seed_component_min_size'), 'max_grow_hops': exp21_growth_summary.get('max_grow_hops'), 'growth_mode': exp21_growth_summary.get('growth_mode'), 'source': "exp21_result['seed_label_aware_merged_component_labels']", 'label_aware_merge': exp21_result['seed_label_aware_merge_summary'], 'exp21_seed_component_growth': exp21_growth_summary}
    sgcr_result = {'components': candidate_components, 'candidate_mask': candidate_mask, 'candidate_labels': candidate_labels, 'summary': candidate_summary}
    adapter.update({'points': points, 'suppressed_intersection_min': semantic_confidence, 'semantic_sv': seed_confidence, 'semantic_sv_norm': seed_confidence_norm, 'sgcr_candidate_score': seed_confidence_norm, 'sgcr_seed_mask': high_confidence_seed_mask, 'sgcr_seed_component_labels': seed_component_labels, 'sgcr_grow_seed_mask': seed_component_mask, 'sgcr_grow_seed_component_labels': seed_component_labels, 'sgcr_candidate_mask': candidate_mask, 'sgcr_candidate_labels': candidate_labels, 'sgcr_result': sgcr_result, 'hks_open_boundary_mask': ac_open_boundary_mask, 'hks_open_boundary_mask_expanded': ac_open_boundary_mask, 'hks_open_boundary_result': {'boundary_mask': ac_open_boundary_mask, 'boundary_mask_expanded': ac_open_boundary_mask, 'summary': {'method': 'exp21_ac_closed_boundary_adapter', 'source': "exp21_result['ac_closed_mask']", 'pre_gtr_expand': True, 'point_count': int(np.sum(ac_open_boundary_mask)), 'point_ratio': float(np.mean(ac_open_boundary_mask))}}})
    adapter_summaries['sgcr_summary'] = candidate_summary
    adapter_summaries['exp22_upstream_adapter'] = {'upstream': 'exp21_ac_pure_hks_boundary_growth_debug', 'candidate_labels': 'seed_label_aware_merged_component_labels', 'seed_consistency_labels': 'seed_size_filtered_component_labels', 'gtr_score': 'seed_confidence', 'boundary_support': 'ac_closed_mask', 'boundary_support_stage': 'after_ac_closing_before_ac_expansion', 'gtr_boundary_support_expand_hops': int(GTR_AC_OPEN_BOUNDARY_SUPPORT_EXPAND_HOPS), 'restitute_boundary_support_after_decision': bool(GTR_RESTITUTE_OPEN_BOUNDARY_SUPPORT_AFTER_DECISION), 'boundary_support_restitution_mode': GTR_OPEN_BOUNDARY_RESTITUTION_MODE, 'legacy_exp14_sgcr_post_filters_rerun': False}
    adapter['summaries'] = adapter_summaries
    return adapter

def run_covert_pipeline(raw_points: np.ndarray, config: Any, *, query_workers: int=1, sample_id: str='sample', suppression_override_hook: Any | None=None, verbose: bool=False) -> dict[str, Any]:
    """Run the one reachable Paper mainline without module-global mutation."""
    base_result = _backbone.run_covert_backbone(raw_points, config, query_workers=query_workers, sample_id=sample_id)
    frontend = _frontend.run_covert_frontend(base_result, config, suppression_override_hook=suppression_override_hook, verbose=verbose)
    adapter = _build_exp21_gtr_adapter(frontend)
    points = np.asarray(frontend['points'], dtype=np.float64)
    neighbor_indices = adapter['neighbor_indices']
    adjacency_dist = adapter['adjacency_dist']
    original_labels = np.asarray(frontend['seed_label_aware_merged_component_labels'], dtype=np.int32).reshape(-1)
    component_rule_enabled = bool(frontend['seed_confidence_p80_component_extraction_enable'])
    preview = None
    preview_labels = None
    if component_rule_enabled:
        preview = _gtr._run_covert_gtr_closing(points=points, neighbor_indices=neighbor_indices, sgcr_candidate_labels=original_labels, tscb_result=adapter, exp06=_score_compat, gtr_params_override=_gtr.build_covert_gtr_params(config))
        preview_labels = np.asarray(preview['closed_labels_core'], dtype=np.int32)
    component_rule = _build_component_rule_result(frontend, enabled=component_rule_enabled, preview_closed_labels_core=preview_labels, thresholds={field: int(getattr(config.scale_consistency, field)) for field in ('small_defect_max_multiple', 'large_defect_min_multiple', 'late_gtr_over_y_min_multiple', 'late_gtr_over_y_min_numerator', 'late_gtr_over_y_min_denominator', 'late_w_over_y_max_numerator', 'late_w_over_y_max_denominator', 'trusted_z_over_q_min_multiple', 'trusted_gtr_over_w_min_multiple', 'trusted_y_over_u_max_numerator', 'trusted_y_over_u_max_denominator', 'trusted_gtr_over_u_min_multiple', 'weak_y_over_u_min_multiple', 'normal_mass_majority_numerator', 'normal_mass_majority_denominator')})
    filtered_labels = np.asarray(component_rule['filtered_candidate_labels'], dtype=np.int32)
    cleanup = _build_local_normal_u_cleanup_request(frontend, component_rule, enabled=bool(config.switches.final_local_normal_u_cleanup), mode=str(config.gtr.local_normal_u_cleanup_mode))
    gtr_result = _gtr.run_covert_gtr(points=points, adjacency_dist=adjacency_dist, neighbor_indices=neighbor_indices, candidate_labels=filtered_labels, frontend_adapter=adapter, config=config, score_compat=_score_compat, local_normal_u_cleanup_request=cleanup)
    if np.asarray(gtr_result['final_defect_mask']).shape != (points.shape[0],):
        raise AssertionError('GTR final mask shape disagrees with the working cloud.')
    return {'sample_id': str(sample_id), 'points': points, 'exp21_result': frontend, 'tscb_result': adapter, 'sgcr_candidate_mask': original_labels > 0, 'sgcr_candidate_labels': original_labels, 'component_rule_result': component_rule, 'component_rule_preview_result': preview, 'component_rule_preview_closed_labels_core': preview_labels, 'component_rule_filtered_candidate_labels': filtered_labels, 'component_rule_filtered_candidate_mask': filtered_labels > 0, 'component_rule_normal_y_ids': list(component_rule['normal_y_ids']), 'component_rule_defect_y_ids': list(component_rule['defect_y_ids']), 'component_rule_unknown_y_ids': list(component_rule['unknown_y_ids']), 'local_normal_u_cleanup_request': cleanup, 'gtr_result': gtr_result, 'invariants': {'final_mask_shape_valid': True, 'filtered_candidate_subset': bool(np.all((filtered_labels > 0) <= (original_labels > 0)))}}
