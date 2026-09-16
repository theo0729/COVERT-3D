"""CLI for freezing the common 10k-point eight-category benchmark cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.common.benchmark import PreprocessConfig, export_benchmark_cache
from experiments.benchmark_categories import BENCHMARK_CATEGORIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/real3d_ad"),
    )
    parser.add_argument(
        "--mvtec-root",
        type=Path,
        default=Path("data/mvtec_3d_ad_converted"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "baseline" / "benchmark_cache",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(BENCHMARK_CATEGORIES),
        choices=BENCHMARK_CATEGORIES,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export at most N GT samples per category (smoke tests only).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = PreprocessConfig(
        num_working_points=10_000,
        fps_mode="exact_numba",
        fps_prefilter_num_points=10_000,
        fps_seed=0,
        cube_size=64.0,
        use_sor_before_fps=False,
    )
    rows = export_benchmark_cache(
        dataset_root=args.dataset_root,
        mvtec_root=args.mvtec_root,
        cache_dir=args.cache_dir,
        categories=args.categories,
        preprocess_config=config,
        limit_per_category=args.limit,
        force=args.force,
        continue_on_error=args.continue_on_error,
    )
    summary = {
        "cache_dir": str(args.cache_dir.resolve()),
        "num_rows": len(rows),
        "num_successful": sum(row["status"] != "error" for row in rows),
        "num_failed": sum(row["status"] == "error" for row in rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
