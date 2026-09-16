"""Shared cache, validation, metric, and provenance utilities.

The cache exporter deliberately reproduces only the frozen COVERT
data-preparation prefix. It does not execute any COVERT feature, candidate, or
verification module. All baselines therefore receive the same normalized FPS
points and the same aligned binary ground truth without paying the cost of the
full COVERT pipeline on every cache export.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from experiments.benchmark_categories import is_mvtec_category
from vast.data import io as data_io
from vast.data import preprocess


POINT_SUFFIXES = (".pcd", ".ply", ".npy", ".txt", ".xyz", ".csv")


@dataclass(frozen=True)
class PreprocessConfig:
    """Frozen working-cloud configuration shared by every baseline."""

    num_working_points: int = 10_000
    fps_mode: str = "exact_numba"
    fps_prefilter_num_points: int = 10_000
    fps_seed: int = 0
    cube_size: float = 64.0
    use_sor_before_fps: bool = False

    def validate(self) -> None:
        if self.num_working_points <= 0:
            raise ValueError("num_working_points must be positive.")
        if self.fps_prefilter_num_points < self.num_working_points:
            raise ValueError(
                "fps_prefilter_num_points must be at least num_working_points."
            )
        if self.fps_seed < 0:
            raise ValueError("fps_seed must be non-negative.")
        if not np.isfinite(self.cube_size) or self.cube_size <= 0.0:
            raise ValueError("cube_size must be finite and positive.")
        if self.use_sor_before_fps:
            raise ValueError(
                "The controlled cache must not run SOR before FPS; SOR is a baseline."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineResult:
    """Common output contract for a baseline method."""

    pred_mask: np.ndarray
    runtime_sec: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def stable_config_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return a deterministic short SHA-256 hash for a resolved config."""

    if length <= 0:
        raise ValueError("length must be positive.")
    digest = hashlib.sha256(_canonical_json(dict(config)).encode("utf-8")).hexdigest()
    return digest[:length]


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _as_scalar_text(value: np.ndarray | Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar string field, got shape {array.shape}.")
    return str(array.reshape(()).item())


def discover_defect_samples(
    dataset_root: str | Path,
    categories: Sequence[str],
    limit_per_category: int | None = None,
    *,
    mvtec_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Pair GT and point clouds from separate Real3D/MVTec roots."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Real3D-AD root not found: {root}")
    resolved_mvtec_root = (
        Path(mvtec_root).expanduser().resolve()
        if mvtec_root is not None
        else (root / "MvTec").resolve()
    )
    if limit_per_category is not None and limit_per_category <= 0:
        raise ValueError("limit_per_category must be positive when provided.")

    samples: list[dict[str, Any]] = []
    for raw_category in categories:
        category = str(raw_category).strip().lower()
        is_mvtec = is_mvtec_category(category)
        category_root = (resolved_mvtec_root if is_mvtec else root) / category
        gt_dir = category_root / "gt"
        test_dir = category_root / "test"
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"GT directory not found: {gt_dir}")
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        gt_files = sorted(
            (
                path for path in gt_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".npy"}
            ),
            key=lambda path: path.name.lower(),
        )
        if limit_per_category is not None:
            gt_files = gt_files[:limit_per_category]
        for gt_path in gt_files:
            pc_path = next(
                (
                    test_dir / f"{gt_path.stem}{suffix}"
                    for suffix in POINT_SUFFIXES
                    if (test_dir / f"{gt_path.stem}{suffix}").is_file()
                ),
                None,
            )
            if pc_path is None:
                raise FileNotFoundError(
                    f"Matching test point cloud not found for {gt_path}: {test_dir}"
                )
            samples.append(
                {
                    "dataset": "MVTec3D" if is_mvtec else "Real3D-AD",
                    "category": category,
                    "sample_id": gt_path.stem,
                    "pc_path": pc_path.resolve(),
                    "gt_path": gt_path.resolve(),
                    "sample_kind": "defective",
                }
            )
    return samples


def _is_native_good_name(path: Path) -> bool:
    return "good" in path.stem.strip().lower().split("_")


def _point_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in POINT_SUFFIXES
        ),
        key=lambda path: (path.name.lower(), str(path)),
    )


def _manifest_good_names(category_root: Path) -> set[str]:
    manifest_path = category_root / "conversion_manifest.json"
    if not manifest_path.is_file():
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid records list in {manifest_path}")
    result: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if str(record.get("defect_type", "")).strip().lower() != "good":
            continue
        output_pcd = str(record.get("output_pcd", "")).strip()
        if output_pcd:
            result.add(Path(output_pcd).name.lower())
    return result


def discover_good_samples(
    dataset_root: str | Path,
    mvtec_root: str | Path,
    categories: Sequence[str],
    limit_per_category: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover mutually exclusive good samples in deterministic Main8 order.

    Real3D supports both its documented sibling ``good`` directory and the
    local converted layout where native ``*_good*`` files live in ``test``.
    MVTec uses the local conversion manifest plus its native ``*_good`` name.
    In both datasets, any point cloud with a same-stem GT file is rejected.
    """

    real_root = Path(dataset_root).expanduser().resolve()
    mvtec = Path(mvtec_root).expanduser().resolve()
    if limit_per_category is not None and limit_per_category <= 0:
        raise ValueError("limit_per_category must be positive when provided.")
    category_order = {
        str(category).strip().lower(): index
        for index, category in enumerate(categories)
    }
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_category in categories:
        category = str(raw_category).strip().lower()
        is_mvtec = is_mvtec_category(category)
        dataset = "MVTec3D" if is_mvtec else "Real3D-AD"
        category_root = (mvtec if is_mvtec else real_root) / category
        if not category_root.is_dir():
            failures.append({
                "dataset": dataset,
                "category": category,
                "sample_id": "",
                "status": "discovery_error",
                "failure_type": "MissingCategoryDirectory",
                "failure_message": str(category_root),
            })
            continue
        test_dir = category_root / "test"
        gt_dir = category_root / "gt"
        candidates: dict[str, Path] = {
            str(path.resolve()).lower(): path
            for path in _point_files(category_root / "good")
        }
        test_files = _point_files(test_dir)
        if is_mvtec:
            try:
                manifest_names = _manifest_good_names(category_root)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                failures.append({
                    "dataset": dataset,
                    "category": category,
                    "sample_id": "",
                    "status": "discovery_error",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                })
                manifest_names = set()
            for path in test_files:
                if path.name.lower() in manifest_names or _is_native_good_name(path):
                    candidates[str(path.resolve()).lower()] = path
        else:
            for path in test_files:
                if _is_native_good_name(path):
                    candidates[str(path.resolve()).lower()] = path

        accepted = 0
        for path in sorted(candidates.values(), key=lambda item: item.name.lower()):
            matching_gt = next(
                (
                    gt_dir / f"{path.stem}{suffix}"
                    for suffix in (".txt", ".csv", ".npy")
                    if (gt_dir / f"{path.stem}{suffix}").is_file()
                ),
                None,
            )
            if matching_gt is not None:
                failures.append({
                    "dataset": dataset,
                    "category": category,
                    "sample_id": path.stem,
                    "status": "discovery_error",
                    "failure_type": "GoodSampleHasGroundTruth",
                    "failure_message": str(matching_gt),
                })
                continue
            identity = (dataset, category, path.stem)
            if identity in seen:
                failures.append({
                    "dataset": dataset,
                    "category": category,
                    "sample_id": path.stem,
                    "status": "discovery_error",
                    "failure_type": "DuplicateGoodSample",
                    "failure_message": str(path),
                })
                continue
            seen.add(identity)
            accepted += 1
            samples.append({
                "dataset": dataset,
                "category": category,
                "sample_id": path.stem,
                "pc_path": path.resolve(),
                "gt_path": None,
                "sample_kind": "good",
            })
            if limit_per_category is not None and accepted >= limit_per_category:
                break
        if accepted == 0:
            failures.append({
                "dataset": dataset,
                "category": category,
                "sample_id": "",
                "status": "discovery_error",
                "failure_type": "NoGoodSamples",
                "failure_message": str(category_root),
            })

    samples.sort(key=lambda sample: (
        category_order[str(sample["category"])],
        str(sample["sample_id"]),
        str(sample["pc_path"]),
    ))
    return samples, failures


def _atomic_savez_compressed(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp.npz",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _export_one_cache_sample(
    sample: Mapping[str, Any],
    cache_dir: Path,
    preprocess_config: PreprocessConfig,
    preprocess_config_hash: str,
) -> dict[str, Any]:
    category = str(sample["category"])
    sample_id = str(sample["sample_id"])
    pc_path = Path(sample["pc_path"])
    gt_path = Path(sample["gt_path"])

    raw_points = np.asarray(data_io.load_point_cloud_file(pc_path), dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] != 3:
        raise ValueError(f"Invalid raw point shape for {sample_id}: {raw_points.shape}")
    if not np.isfinite(raw_points).all():
        raise ValueError(f"Raw point cloud contains NaN/Inf: {pc_path}")

    gt_mask_raw, gt_meta = data_io.load_optional_labels_with_meta(
        gt_path,
        num_points=raw_points.shape[0],
        reference_points=raw_points,
    )
    if gt_mask_raw is None:
        raise RuntimeError(f"Unable to align GT with raw points: {gt_path}")
    gt_mask_raw = np.asarray(gt_mask_raw, dtype=bool).reshape(-1)
    if gt_mask_raw.shape[0] != raw_points.shape[0]:
        raise ValueError(
            f"Raw GT length mismatch for {sample_id}: "
            f"{gt_mask_raw.shape[0]} vs {raw_points.shape[0]}"
        )

    sampled_points, fps_indices, fps_summary = preprocess.fps_downsample(
        raw_points,
        num_samples=preprocess_config.num_working_points,
        # FPS is deliberately label-free.  GT is mapped only after the point
        # indices have been frozen, which makes the no-oracle data path
        # explicit and auditable.
        gt_mask=None,
        fps_mode=preprocess_config.fps_mode,
        fps_prefilter_num_points=preprocess_config.fps_prefilter_num_points,
        seed=preprocess_config.fps_seed,
        return_fps_summary=True,
    )
    fps_indices = np.asarray(fps_indices, dtype=np.int64).reshape(-1)
    gt_mask = gt_mask_raw[fps_indices]
    points, d_max, center, scale = preprocess.isotropic_scale(
        sampled_points,
        cube_size=preprocess_config.cube_size,
    )
    points = np.asarray(points, dtype=np.float64)

    expected_points_shape = (preprocess_config.num_working_points, 3)
    expected_mask_shape = (preprocess_config.num_working_points,)
    if points.shape != expected_points_shape:
        raise ValueError(
            f"Unexpected working point shape for {sample_id}: "
            f"{points.shape}, expected {expected_points_shape}"
        )
    if gt_mask.shape != expected_mask_shape:
        raise ValueError(
            f"Unexpected working GT shape for {sample_id}: "
            f"{gt_mask.shape}, expected {expected_mask_shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"Working point cloud contains NaN/Inf: {sample_id}")
    if np.unique(fps_indices).shape[0] != fps_indices.shape[0]:
        raise ValueError(f"FPS returned duplicate indices for {sample_id}.")

    cache_path = cache_dir / category / f"{sample_id}.npz"
    _atomic_savez_compressed(
        cache_path,
        points=points,
        gt_mask=gt_mask.astype(np.uint8),
        fps_selected_indices=fps_indices,
        sample_id=np.asarray(sample_id),
        category=np.asarray(category),
        pc_path=np.asarray(str(pc_path)),
        gt_path=np.asarray(str(gt_path)),
        fps_selected_index_hash=np.asarray(
            str(fps_summary.get("selected_index_hash", ""))
        ),
        fps_mode=np.asarray(str(fps_summary.get("fps_mode", ""))),
        fps_backend=np.asarray(str(fps_summary.get("backend", ""))),
        fps_seed=np.asarray(preprocess_config.fps_seed, dtype=np.int64),
        fps_start_index=np.asarray(
            int(fps_summary.get("fps_start_index", 0)), dtype=np.int64
        ),
        num_raw_points=np.asarray(raw_points.shape[0], dtype=np.int64),
        num_working_points=np.asarray(points.shape[0], dtype=np.int64),
        normalization_center=np.asarray(center, dtype=np.float64),
        normalization_scale=np.asarray(scale, dtype=np.float64),
        normalization_d_max=np.asarray(d_max, dtype=np.float64),
        preprocess_config_hash=np.asarray(preprocess_config_hash),
        preprocess_config_json=np.asarray(
            _canonical_json(preprocess_config.to_dict())
        ),
        fps_summary_json=np.asarray(_canonical_json(fps_summary)),
        gt_meta_json=np.asarray(_canonical_json(gt_meta or {})),
    )

    return {
        "category": category,
        "sample_id": sample_id,
        "pc_path": str(pc_path),
        "gt_path": str(gt_path),
        "cache_path": str(cache_path.resolve()),
        "status": "exported",
        "error": "",
        "num_raw_points": int(raw_points.shape[0]),
        "num_working_points": int(points.shape[0]),
        "num_gt_positive": int(np.count_nonzero(gt_mask)),
        "gt_positive_ratio": float(np.mean(gt_mask)),
        "fps_mode": str(fps_summary.get("fps_mode", "")),
        "fps_backend": str(fps_summary.get("backend", "")),
        "fps_seed": int(preprocess_config.fps_seed),
        "fps_selected_index_hash": str(
            fps_summary.get("selected_index_hash", "")
        ),
        "preprocess_config_hash": preprocess_config_hash,
        "cache_sha256": _file_sha256(cache_path),
    }


CACHE_MANIFEST_FIELDS = [
    "category",
    "sample_id",
    "pc_path",
    "gt_path",
    "cache_path",
    "status",
    "error",
    "num_raw_points",
    "num_working_points",
    "num_gt_positive",
    "gt_positive_ratio",
    "fps_mode",
    "fps_backend",
    "fps_seed",
    "fps_selected_index_hash",
    "preprocess_config_hash",
    "cache_sha256",
]


def export_benchmark_cache(
    dataset_root: str | Path,
    cache_dir: str | Path,
    categories: Sequence[str],
    mvtec_root: str | Path | None = None,
    preprocess_config: PreprocessConfig | None = None,
    limit_per_category: int | None = None,
    force: bool = False,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Export the deterministic shared working-cloud cache and manifest."""

    config = preprocess_config or PreprocessConfig()
    config.validate()
    config_hash = stable_config_hash(config.to_dict())
    output_root = Path(cache_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples = discover_defect_samples(
        dataset_root=dataset_root,
        categories=categories,
        limit_per_category=limit_per_category,
        mvtec_root=mvtec_root,
    )

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        category = str(sample["category"])
        sample_id = str(sample["sample_id"])
        cache_path = output_root / category / f"{sample_id}.npz"
        print(
            f"[cache {index:03d}/{len(samples):03d}] "
            f"{category}/{sample_id}",
            flush=True,
        )
        try:
            if cache_path.exists() and not force:
                loaded = load_cache_file(
                    cache_path,
                    expected_num_points=config.num_working_points,
                    expected_preprocess_config_hash=config_hash,
                )
                rows.append(
                    {
                        "category": loaded["category"],
                        "sample_id": loaded["sample_id"],
                        "pc_path": loaded["pc_path"],
                        "gt_path": loaded["gt_path"],
                        "cache_path": str(cache_path.resolve()),
                        "status": "cached",
                        "error": "",
                        "num_raw_points": loaded["num_raw_points"],
                        "num_working_points": int(loaded["points"].shape[0]),
                        "num_gt_positive": int(np.count_nonzero(loaded["gt_mask"])),
                        "gt_positive_ratio": float(np.mean(loaded["gt_mask"])),
                        "fps_mode": loaded["fps_mode"],
                        "fps_backend": loaded["fps_backend"],
                        "fps_seed": loaded["fps_seed"],
                        "fps_selected_index_hash": loaded[
                            "fps_selected_index_hash"
                        ],
                        "preprocess_config_hash": config_hash,
                        "cache_sha256": _file_sha256(cache_path),
                    }
                )
            else:
                rows.append(
                    _export_one_cache_sample(
                        sample=sample,
                        cache_dir=output_root,
                        preprocess_config=config,
                        preprocess_config_hash=config_hash,
                    )
                )
        except Exception as exc:
            row = {
                "category": category,
                "sample_id": sample_id,
                "pc_path": str(sample["pc_path"]),
                "gt_path": str(sample["gt_path"]),
                "cache_path": str(cache_path.resolve()),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "num_raw_points": "",
                "num_working_points": "",
                "num_gt_positive": "",
                "gt_positive_ratio": "",
                "fps_mode": config.fps_mode,
                "fps_backend": "",
                "fps_seed": config.fps_seed,
                "fps_selected_index_hash": "",
                "preprocess_config_hash": config_hash,
                "cache_sha256": "",
            }
            rows.append(row)
            write_csv(output_root / "cache_manifest.csv", rows, CACHE_MANIFEST_FIELDS)
            if not continue_on_error:
                raise

        write_csv(output_root / "cache_manifest.csv", rows, CACHE_MANIFEST_FIELDS)

    config_payload = {
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "mvtec_root": (
            str(Path(mvtec_root).expanduser().resolve())
            if mvtec_root is not None else None
        ),
        "cache_dir": str(output_root),
        "categories": [str(category).lower() for category in categories],
        "limit_per_category": limit_per_category,
        "preprocess": config.to_dict(),
        "preprocess_config_hash": config_hash,
        "num_discovered_samples": len(samples),
        "num_successful_samples": sum(row["status"] != "error" for row in rows),
        "num_failed_samples": sum(row["status"] == "error" for row in rows),
    }
    (output_root / "cache_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "environment.json").write_text(
        json.dumps(collect_environment(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def load_cache_file(
    path: str | Path,
    expected_num_points: int | None = 10_000,
    expected_preprocess_config_hash: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate one shared benchmark cache file."""

    cache_path = Path(path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        required = {
            "points",
            "gt_mask",
            "fps_selected_indices",
            "sample_id",
            "category",
            "pc_path",
            "gt_path",
            "fps_selected_index_hash",
            "preprocess_config_hash",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Cache {cache_path} is missing fields: {missing}")

        points = np.asarray(data["points"], dtype=np.float64)
        gt_mask = np.asarray(data["gt_mask"], dtype=bool).reshape(-1)
        fps_selected_indices = np.asarray(
            data["fps_selected_indices"], dtype=np.int64
        ).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Invalid cached point shape: {points.shape}")
        if gt_mask.shape != (points.shape[0],):
            raise ValueError(
                f"Cached GT length mismatch: {gt_mask.shape} vs {points.shape[0]}"
            )
        if fps_selected_indices.shape != (points.shape[0],):
            raise ValueError(
                "Cached FPS-index length mismatch: "
                f"{fps_selected_indices.shape} vs {points.shape[0]}"
            )
        if np.unique(fps_selected_indices).size != fps_selected_indices.size:
            raise ValueError(f"Cached FPS indices contain duplicates: {cache_path}")
        if expected_num_points is not None and points.shape[0] != expected_num_points:
            raise ValueError(
                f"Cache has {points.shape[0]} points; expected {expected_num_points}."
            )
        if not np.isfinite(points).all():
            raise ValueError(f"Cached points contain NaN/Inf: {cache_path}")

        config_hash = _as_scalar_text(data["preprocess_config_hash"])
        if (
            expected_preprocess_config_hash is not None
            and config_hash != expected_preprocess_config_hash
        ):
            raise ValueError(
                f"Cache config hash mismatch for {cache_path}: "
                f"{config_hash} != {expected_preprocess_config_hash}"
            )

        return {
            "points": points,
            "gt_mask": gt_mask,
            "fps_selected_indices": fps_selected_indices,
            "sample_id": _as_scalar_text(data["sample_id"]),
            "category": _as_scalar_text(data["category"]),
            "pc_path": _as_scalar_text(data["pc_path"]),
            "gt_path": _as_scalar_text(data["gt_path"]),
            "fps_selected_index_hash": _as_scalar_text(
                data["fps_selected_index_hash"]
            ),
            "preprocess_config_hash": config_hash,
            "fps_mode": _as_scalar_text(data["fps_mode"])
            if "fps_mode" in data.files
            else "",
            "fps_backend": _as_scalar_text(data["fps_backend"])
            if "fps_backend" in data.files
            else "",
            "fps_seed": int(np.asarray(data["fps_seed"]).reshape(()).item())
            if "fps_seed" in data.files
            else 0,
            "num_raw_points": int(
                np.asarray(data["num_raw_points"]).reshape(()).item()
            )
            if "num_raw_points" in data.files
            else int(points.shape[0]),
            "normalization_center": np.asarray(
                data["normalization_center"], dtype=np.float64
            ).reshape(3)
            if "normalization_center" in data.files
            else None,
            "normalization_scale": float(
                np.asarray(data["normalization_scale"]).reshape(()).item()
            )
            if "normalization_scale" in data.files
            else None,
            "cache_path": str(cache_path),
        }


def list_cache_files(
    cache_dir: str | Path,
    categories: Sequence[str],
) -> list[Path]:
    root = Path(cache_dir).expanduser().resolve()
    files: list[Path] = []
    for raw_category in categories:
        category = str(raw_category).strip().lower()
        category_dir = root / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Cache category directory not found: {category_dir}")
        files.extend(sorted(category_dir.glob("*.npz"), key=lambda path: path.name.lower()))
    return files


def compute_binary_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute the exact binary metrics used by the COVERT benchmark."""

    pred = np.asarray(pred_mask, dtype=bool).reshape(-1)
    gt = np.asarray(gt_mask, dtype=bool).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"pred_mask and gt_mask mismatch: {pred.shape} vs {gt.shape}")

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(np.count_nonzero(~pred & ~gt))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_csv_atomic(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write a complete CSV in the caller process and replace it atomically."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8-sig",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    try:
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_good_metrics(
    rows: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    *,
    method: str,
    discovery_failures: Sequence[Mapping[str, Any]] = (),
    status: str = "completed",
) -> dict[str, Any]:
    """Aggregate only all-normal FPR values; no P/R/F1/IoU are emitted."""

    normalized = [dict(row) for row in rows]
    by_category: dict[str, dict[str, Any]] = {}
    pooled: list[float] = []
    category_means: list[float] = []
    for raw_category in categories:
        category = str(raw_category).strip().lower()
        selected = [row for row in normalized if row.get("category") == category]
        successful = [row for row in selected if row.get("status") == "success"]
        values = [float(row["fpr"]) for row in successful]
        pooled.extend(values)
        mean_fpr = float(np.mean(values)) if values else None
        p95_fpr = float(np.percentile(values, 95.0)) if values else None
        if mean_fpr is not None:
            category_means.append(mean_fpr)
        by_category[category] = {
            "sample_count": len(selected),
            "success_count": len(successful),
            "failure_count": len(selected) - len(successful),
            "mean_fpr": mean_fpr,
            "p95_fpr": p95_fpr,
        }
    successful = [row for row in normalized if row.get("status") == "success"]
    return {
        "status": status,
        "method": method,
        "metric_definition": "predicted_defect_points / num_working_points",
        "gt_policy": (
            "No GT is passed to preprocessing or inference; the all-normal "
            "truth is applied only by the FPR evaluator."
        ),
        "samples": normalized,
        "categories": by_category,
        "overall": {
            "category_count": len(categories),
            "valid_category_count": len(category_means),
            "sample_count": len(normalized),
            "success_count": len(successful),
            "failure_count": len(normalized) - len(successful),
            "class_balanced_mean_fpr": (
                float(np.mean(category_means)) if category_means else None
            ),
            "pooled_p95_fpr": (
                float(np.percentile(pooled, 95.0)) if pooled else None
            ),
        },
        "discovery_failures": [dict(item) for item in discovery_failures],
    }


def run_good_evaluation(
    *,
    method: str,
    inference: Callable[[np.ndarray], BaselineResult] | None = None,
    inference_parameters: Mapping[str, Any] | None = None,
    dataset_root: str | Path,
    mvtec_root: str | Path,
    categories: Sequence[str],
    output_dir: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    max_samples_per_category: int | None = None,
    sample_workers: int = 1,
    inner_thread_limit: int | None = 1,
    numba_threads: int | None = None,
) -> dict[str, Any]:
    """Run a method's frozen, label-free inference on the good cohort.

    The helper intentionally writes only ``good_metrics.json`` into the method
    result directory.  It never constructs or passes a GT mask to FPS,
    normalization, or ``inference``.
    """

    from baseline.common.parallel import (
        execute_good_sample,
        iter_process_results,
        limited_inner_threads,
        limited_numba_threads,
        validate_execution_settings,
    )

    workers, inner = validate_execution_settings(sample_workers, inner_thread_limit)
    numba_limit = None if numba_threads is None else int(numba_threads)
    if numba_limit is not None and numba_limit <= 0:
        raise ValueError("numba_threads must be positive when configured.")
    if inference_parameters is None and inference is None:
        raise ValueError("Good evaluation requires inference_parameters or inference.")
    if inference_parameters is None and workers > 1:
        raise ValueError(
            "Process-based good evaluation requires serializable inference_parameters; "
            "a Python callable is supported only with sample_workers=1."
        )
    config = preprocess_config or PreprocessConfig()
    config.validate()
    samples, discovery_failures = discover_good_samples(
        dataset_root=dataset_root,
        mvtec_root=mvtec_root,
        categories=categories,
        limit_per_category=max_samples_per_category,
    )
    result_path = Path(output_dir).expanduser().resolve() / "good_metrics.json"
    rows: list[dict[str, Any]] = []

    def publish(status: str) -> dict[str, Any]:
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                str(row.get("dataset", "")),
                str(row.get("category", "")),
                str(row.get("sample_id", "")),
            ),
        )
        payload = build_good_metrics(
            ordered_rows,
            categories,
            method=method,
            discovery_failures=discovery_failures,
            status=status,
        )
        execution = {
            "configured_sample_workers": workers,
            "effective_sample_workers": min(workers, len(samples)) if samples else 0,
            "inner_thread_limit": inner,
            "parallel_unit": "independent_sample",
        }
        if numba_limit is not None:
            execution["numba_threads"] = numba_limit
        payload["execution"] = execution
        _atomic_write_json(result_path, payload)
        return payload

    publish("running")
    if inference_parameters is not None:
        tasks = [
            {
                "sample": dict(sample),
                "method": method,
                "parameters": dict(inference_parameters),
                "preprocess_config": config.to_dict(),
                "inner_thread_limit": inner,
                "numba_threads": numba_limit,
            }
            for sample in samples
        ]
        completed = iter_process_results(
            tasks, execute_good_sample, sample_workers=workers
        )
        for index, (task, payload, process_error) in enumerate(completed, start=1):
            sample = task["sample"]
            row = dict(payload) if payload is not None else {
                "dataset": sample["dataset"],
                "category": sample["category"],
                "sample_id": sample["sample_id"],
                "status": "failure",
                "num_points": None,
                "fp": None,
                "tn": None,
                "fpr": None,
                "inference_total_seconds": None,
                "failure_type": type(process_error).__name__,
                "failure_message": str(process_error),
            }
            rows.append(row)
            publish("running")
            print(
                f"[{method} good {index:03d}/{len(samples):03d}] "
                f"{row['category']}/{row['sample_id']} {row['status']}",
                flush=True,
            )
    else:
        assert inference is not None
        for index, sample in enumerate(samples, start=1):
            try:
                if sample.get("sample_kind") != "good" or sample.get("gt_path") is not None:
                    raise ValueError("Good evaluation received a non-good SampleSpec.")
                with limited_inner_threads(inner):
                    raw_points = np.asarray(
                        data_io.load_point_cloud_file(Path(sample["pc_path"])),
                        dtype=np.float64,
                    )
                    sampled_points, fps_indices, _ = preprocess.fps_downsample(
                        raw_points,
                        num_samples=config.num_working_points,
                        gt_mask=None,
                        fps_mode=config.fps_mode,
                        fps_prefilter_num_points=config.fps_prefilter_num_points,
                        seed=config.fps_seed,
                        return_fps_summary=True,
                    )
                    points, _, _, _ = preprocess.isotropic_scale(
                        sampled_points, cube_size=config.cube_size
                    )
                    points = np.asarray(points, dtype=np.float64)
                    fps_indices = np.asarray(fps_indices, dtype=np.int64).reshape(-1)
                    expected_shape = (config.num_working_points, 3)
                    if points.shape != expected_shape:
                        raise RuntimeError(
                            f"Good working cloud shape {points.shape}; expected {expected_shape}."
                        )
                    if np.unique(fps_indices).size != config.num_working_points:
                        raise RuntimeError("Good-sample FPS returned duplicate indices.")
                    with limited_numba_threads(numba_limit):
                        result = inference(points)
                pred_mask = np.asarray(result.pred_mask, dtype=bool).reshape(-1)
                if pred_mask.shape != (config.num_working_points,):
                    raise RuntimeError(
                        f"{method} output shape {pred_mask.shape}; expected "
                        f"{(config.num_working_points,)}."
                    )
                fp = int(np.count_nonzero(pred_mask))
                num_points = int(pred_mask.size)
                row = {
                    "dataset": sample["dataset"], "category": sample["category"],
                    "sample_id": sample["sample_id"], "status": "success",
                    "num_points": num_points, "fp": fp, "tn": num_points - fp,
                    "fpr": float(fp / num_points),
                    "inference_total_seconds": float(result.runtime_sec),
                    "failure_type": "", "failure_message": "",
                }
            except BaseException as exc:
                row = {
                    "dataset": sample["dataset"], "category": sample["category"],
                    "sample_id": sample["sample_id"], "status": "failure",
                    "num_points": None, "fp": None, "tn": None, "fpr": None,
                    "inference_total_seconds": None,
                    "failure_type": type(exc).__name__, "failure_message": str(exc),
                }
            rows.append(row)
            publish("running")
            print(
                f"[{method} good {index:03d}/{len(samples):03d}] "
                f"{row['category']}/{row['sample_id']} {row['status']}",
                flush=True,
            )
    return publish("completed")


def collect_environment() -> dict[str, Any]:
    """Collect enough runtime provenance to reproduce a baseline run."""

    package_versions: dict[str, str] = {"numpy": np.__version__}
    try:
        import scipy

        package_versions["scipy"] = scipy.__version__
    except Exception:
        package_versions["scipy"] = "unavailable"
    try:
        import open3d

        package_versions["open3d"] = open3d.__version__
    except Exception:
        package_versions["open3d"] = "unavailable"
    try:
        import sklearn

        package_versions["scikit-learn"] = sklearn.__version__
    except Exception:
        package_versions["scikit-learn"] = "unavailable"

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": package_versions,
    }
