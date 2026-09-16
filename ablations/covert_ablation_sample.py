"""Isolated single-sample runner for main and supplementary COVERT ablations."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations.ablation_modes import (  # noqa: E402
    SUPPORTED_ABLATION_MODES,
    get_ablation_spec,
)
from ablations.local_covert import (  # noqa: E402
    build_sv_only_defect_veto_override,
    run_ablation_pipeline,
)
from covert_sample import (  # noqa: E402
    DEFAULT_CONFIG,
    CovertPipelineConfig,
    CovertTiming,
    _binary_metrics,
    _labels_from_components,
    _validate_config,
    config_hash,
    config_to_dict,
    load_config,
    run_covert_sample as run_production_sample,
)
from vast.covert._veto import build_legacy_hks_defect_veto_override  # noqa: E402


ABLATION_MODE = "full"


@dataclass(frozen=True)
class AblationSampleResult:
    sample_id: str
    pc_path: str
    gt_path: str | None
    ablation_mode: str
    production_config_hash: str
    config_snapshot: Mapping[str, Any]
    points: np.ndarray
    neighbor_indices: tuple[np.ndarray, ...]
    final_mask: np.ndarray
    gt_mask: np.ndarray | None
    metrics: Mapping[str, Any]
    timings: CovertTiming
    gtr_candidate_mask: np.ndarray
    audit: Mapping[str, Any]
    ablation_metadata: Mapping[str, Any]


def _metrics_with_iou(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    if {"tp", "fp", "fn"} <= result.keys():
        denominator = int(result["tp"]) + int(result["fp"]) + int(result["fn"])
        result["iou"] = float(int(result["tp"]) / denominator) if denominator else 0.0
    return result


def _wrap_production_result(result: Any, mode: str) -> AblationSampleResult:
    diagnostics = {"production_delegate": True}
    spec = get_ablation_spec(mode)
    return AblationSampleResult(
        sample_id=result.sample_id,
        pc_path=result.pc_path,
        gt_path=result.gt_path,
        ablation_mode=mode,
        production_config_hash=result.config_hash,
        config_snapshot=dict(result.config_snapshot),
        points=np.asarray(result.points, dtype=np.float64),
        neighbor_indices=tuple(np.asarray(v, dtype=np.int64) for v in result.neighbor_indices),
        final_mask=np.asarray(result.final_mask, dtype=bool),
        gt_mask=(None if result.gt_mask is None else np.asarray(result.gt_mask, dtype=bool)),
        metrics=_metrics_with_iou(result.metrics),
        timings=result.timings,
        gtr_candidate_mask=np.asarray(result.gtr_candidate_mask, dtype=bool),
        audit=result.audit,
        ablation_metadata={
            "ablation_mode": mode,
            "ablation_spec": spec.to_dict(),
            "production_config_hash": result.config_hash,
            "semantic_overrides_only": True,
            "diagnostics": diagnostics,
        },
    )


def _write_artifacts(result: AblationSampleResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": result.sample_id,
        "pc_path": result.pc_path,
        "gt_path": result.gt_path,
        "ablation_mode": result.ablation_mode,
        "production_config_hash": result.production_config_hash,
        "working_point_count": int(result.points.shape[0]),
        "predicted_positive_points": int(np.sum(result.final_mask)),
        "metrics": dict(result.metrics),
        "timings": dataclasses.asdict(result.timings),
        "ablation_metadata": dict(result.ablation_metadata),
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "final_mask.npz", final_mask=result.final_mask.astype(np.uint8))


def run_ablation_sample(
    pc_path: str | Path,
    gt_path: str | Path | None = None,
    *,
    ablation_mode: str = ABLATION_MODE,
    config: CovertPipelineConfig = DEFAULT_CONFIG,
    query_workers: int = 1,
    absolute_threshold: float | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> AblationSampleResult:
    """Run one ablation; GT is loaded only after inference completes."""

    mode = get_ablation_spec(ablation_mode).mode
    if mode == "fixed_absolute_threshold_verification":
        if absolute_threshold is None:
            raise ValueError(
                "fixed_absolute_threshold_verification requires "
                "absolute_threshold."
            )
        if not np.isfinite(float(absolute_threshold)) or float(absolute_threshold) < 0:
            raise ValueError("absolute_threshold must be finite and non-negative.")
    elif absolute_threshold is not None:
        raise ValueError(
            "absolute_threshold is valid only for "
            "fixed_absolute_threshold_verification."
        )
    _validate_config(config)
    base_hash = config_hash(config)
    pc_path = Path(pc_path)
    resolved_gt = Path(gt_path) if gt_path is not None else None
    if not pc_path.is_file():
        raise FileNotFoundError(f"Point-cloud file not found: {pc_path}")
    if query_workers == 0 or query_workers < -1:
        raise ValueError("query_workers must be -1 or a positive integer.")

    if mode == "full":
        production = run_production_sample(
            pc_path,
            resolved_gt,
            config=config,
            query_workers=query_workers,
            visualize=False,
            output_dir=None,
            verbose=verbose,
        )
        result = _wrap_production_result(production, mode)
        if output_dir is not None:
            _write_artifacts(result, Path(output_dir))
        return result

    load_start = time.perf_counter()
    from vast.covert import _point_io

    raw_points = _point_io.load_point_cloud_file(pc_path)
    input_load_seconds = time.perf_counter() - load_start
    veto_builder = (
        build_sv_only_defect_veto_override
        if mode == "wo_va_hks"
        else build_legacy_hks_defect_veto_override
    )
    veto_hook = functools.partial(
        veto_builder,
        thresholds=dataclasses.asdict(config.defect_veto),
    )
    pipeline_start = time.perf_counter()
    pipeline = run_ablation_pipeline(
        raw_points,
        config,
        ablation_mode=mode,
        query_workers=query_workers,
        sample_id=pc_path.stem,
        suppression_override_hook=veto_hook,
        absolute_threshold=absolute_threshold,
        verbose=verbose,
    )
    pipeline_seconds = time.perf_counter() - pipeline_start
    frontend = pipeline["exp21_result"]
    base = frontend["base_result"]
    gtr_result = pipeline["gtr_result"]
    points = np.asarray(frontend["points"], dtype=np.float64)
    final_mask = np.asarray(gtr_result["final_defect_mask"], dtype=bool).reshape(-1)

    gt_mask: np.ndarray | None = None
    metrics: dict[str, Any] = {}
    gt_load_seconds = 0.0
    evaluation_seconds = 0.0
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
            raise ValueError(f"Ground truth could not be loaded or aligned: {resolved_gt}")
        gt_mask = np.asarray(gt_raw, dtype=np.uint8)
        gt_mask = gt_mask[np.asarray(base["sor_indices"], dtype=np.int64)]
        gt_mask = gt_mask[np.asarray(base["fps_indices"], dtype=np.int64)].astype(bool)
        evaluation_start = time.perf_counter()
        metrics = _metrics_with_iou(_binary_metrics(final_mask, gt_mask))
        metrics["gt_meta"] = gt_meta
        evaluation_seconds = time.perf_counter() - evaluation_start

    timings = CovertTiming(
        input_load_seconds=float(input_load_seconds),
        pipeline_seconds=float(pipeline_seconds),
        inference_total_seconds=float(input_load_seconds + pipeline_seconds),
        gt_load_seconds=float(gt_load_seconds),
        evaluation_seconds=float(evaluation_seconds),
    )
    spec = get_ablation_spec(mode)
    diagnostics = dict(pipeline["ablation_diagnostics"])
    audit: dict[str, Any] = {
        "suppression": frontend["suppression_strategy_audit"],
        "q_to_z_repair": frontend["q_z_consistency_repair_audit"],
        "scale_consistency": pipeline["component_rule_result"],
        "gtr_summary": gtr_result["summary"],
        "ablation_diagnostics": diagnostics,
    }
    if mode == "fixed_absolute_threshold_verification":
        audit["fixed_absolute_replay"] = {
            "baseline_threshold": float(absolute_threshold),
            "final_defect_labels": np.asarray(
                gtr_result["final_defect_labels"], dtype=np.int32
            ).copy(),
            "component_reports": [
                dict(report)
                for report in gtr_result["candidate_classifier_result"][
                    "component_reports"
                ]
            ],
        }
    result = AblationSampleResult(
        sample_id=pc_path.stem,
        pc_path=str(pc_path),
        gt_path=str(resolved_gt) if resolved_gt is not None else None,
        ablation_mode=mode,
        production_config_hash=base_hash,
        config_snapshot=config_to_dict(config),
        points=points,
        neighbor_indices=tuple(
            np.asarray(v, dtype=np.int64) for v in frontend["neighbor_indices"]
        ),
        final_mask=final_mask,
        gt_mask=gt_mask,
        metrics=metrics,
        timings=timings,
        gtr_candidate_mask=np.asarray(
            pipeline["component_rule_filtered_candidate_mask"], dtype=bool
        ),
        audit=audit,
        ablation_metadata={
            "ablation_mode": mode,
            "ablation_spec": spec.to_dict(),
            "production_config_hash": base_hash,
            "semantic_overrides_only": True,
            "diagnostics": diagnostics,
        },
    )
    if output_dir is not None:
        _write_artifacts(result, Path(output_dir))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pc_path", type=Path)
    parser.add_argument("--gt-path", type=Path)
    parser.add_argument(
        "--ablation-mode", choices=SUPPORTED_ABLATION_MODES, default=ABLATION_MODE
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--query-workers", type=int, default=1)
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        help=(
            "Required only for fixed_absolute_threshold_verification; rejects "
            "a candidate when D_p90 is strictly below this value."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = DEFAULT_CONFIG if args.config is None else load_config(args.config)
    result = run_ablation_sample(
        args.pc_path,
        args.gt_path,
        ablation_mode=args.ablation_mode,
        config=config,
        query_workers=args.query_workers,
        absolute_threshold=args.absolute_threshold,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    summary = {
        "sample": result.sample_id,
        "ablation_mode": result.ablation_mode,
        "working_point_count": int(result.points.shape[0]),
        "predicted_positive_points": int(np.sum(result.final_mask)),
        "metrics": dict(result.metrics),
        "diagnostics": dict(result.ablation_metadata["diagnostics"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
