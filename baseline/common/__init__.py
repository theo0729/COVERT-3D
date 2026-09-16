"""Shared, method-independent utilities for controlled baselines."""

from .benchmark import (
    BaselineResult,
    PreprocessConfig,
    collect_environment,
    build_good_metrics,
    compute_binary_metrics,
    discover_defect_samples,
    discover_good_samples,
    export_benchmark_cache,
    list_cache_files,
    load_cache_file,
    run_good_evaluation,
    stable_config_hash,
    write_csv,
    write_csv_atomic,
)

__all__ = [
    "BaselineResult",
    "PreprocessConfig",
    "collect_environment",
    "build_good_metrics",
    "compute_binary_metrics",
    "discover_defect_samples",
    "discover_good_samples",
    "export_benchmark_cache",
    "list_cache_files",
    "load_cache_file",
    "run_good_evaluation",
    "stable_config_hash",
    "write_csv",
    "write_csv_atomic",
]
