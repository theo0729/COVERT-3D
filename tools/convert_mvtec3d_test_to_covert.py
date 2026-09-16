#!/usr/bin/env python3
"""Convert an MVTec 3D-AD test split to COVERT/Real3D-AD files.

The input is an MVTec category directory (for example ``.../potato``) whose
test split contains ``<defect>/xyz/<id>.tiff`` and matching
``<defect>/gt/<id>.png`` files.  The output layout is::

    <output>/test/<id>_<defect>.pcd
    <output>/gt/<id>_<defect>.txt   # defective samples only

PCD files contain binary little-endian float32 XYZ data.  GT text files have
one ``x y z label`` row per PCD point and binary labels, matching the local
Real3D-AD-PCD convention used by COVERT.  By default, the converter also fits
the support plane from the organized image border and removes it before the
PCD and aligned GT are written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import tifffile
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency 'tifffile'. Install it with: pip install tifffile"
    ) from exc

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency 'Pillow'. Install it with: pip install Pillow"
    ) from exc

try:
    import open3d as o3d
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    o3d = None


SCRIPT_VERSION = "mvtec3d-test-to-covert-v2"
TIFF_SUFFIXES = {".tif", ".tiff"}
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_PLANE_DISTANCE_THRESHOLD = 0.005
DEFAULT_PLANE_BORDER_WIDTH = 10
DEFAULT_PLANE_RANSAC_N = 50
DEFAULT_PLANE_RANSAC_ITERATIONS = 1_000
DEFAULT_PLANE_RANSAC_SEED = 0
MIN_PLANE_BORDER_INLIER_RATIO = 0.8


@dataclass(frozen=True)
class SampleSpec:
    """A matched MVTec XYZ raster and pixel GT mask."""

    defect_type: str
    source_id: str
    xyz_path: Path
    mask_path: Path

    @property
    def output_stem(self) -> str:
        return f"{self.source_id}_{self.defect_type}"

    @property
    def is_good(self) -> bool:
        return self.defect_type.casefold() == "good"


@dataclass(frozen=True)
class ConversionRecord:
    """Auditable per-sample conversion statistics."""

    defect_type: str
    source_id: str
    source_xyz: str
    source_mask: str
    output_pcd: str
    output_gt: str | None
    source_height: int
    source_width: int
    source_points: int
    depth_valid_points: int
    valid_points: int
    invalid_points: int
    support_plane_points_removed: int
    plane_border_points: int
    plane_border_inliers: int
    plane_border_inlier_ratio: float | None
    plane_model: tuple[float, float, float, float] | None
    source_positive_pixels: int
    positive_points_after_depth_filter: int
    written_positive_points: int
    positive_pixels_without_valid_xyz: int
    positive_points_removed_with_support_plane: int
    coordinate_scale: float


def _resolve_test_root(source: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    if source.name.casefold() == "test" and source.is_dir():
        return source.parent, source
    test_root = source / "test"
    if test_root.is_dir():
        return source, test_root
    raise FileNotFoundError(
        f"Expected an MVTec category directory containing 'test', or the test "
        f"directory itself: {source}"
    )


def _validate_name(token: str, description: str) -> None:
    if not SAFE_NAME_PATTERN.fullmatch(token):
        raise ValueError(
            f"Unsafe {description} {token!r}; expected letters, digits, '.', '_' or '-'."
        )


def discover_samples(
    source: Path,
    selected_classes: Iterable[str] | None = None,
    limit_per_class: int | None = None,
) -> tuple[Path, Path, list[SampleSpec]]:
    """Discover and strictly pair all requested XYZ TIFF and GT PNG files."""

    category_root, test_root = _resolve_test_root(source)
    available_dirs = sorted(
        (path for path in test_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    available_by_key = {path.name.casefold(): path for path in available_dirs}
    if len(available_by_key) != len(available_dirs):
        raise ValueError(f"Case-insensitive duplicate class directories under {test_root}")

    if selected_classes:
        requested = []
        seen = set()
        for value in selected_classes:
            key = value.casefold()
            if key not in available_by_key:
                raise ValueError(
                    f"Unknown class {value!r}. Available classes: "
                    f"{', '.join(path.name for path in available_dirs)}"
                )
            if key not in seen:
                requested.append(available_by_key[key])
                seen.add(key)
        class_dirs = sorted(requested, key=lambda path: path.name.casefold())
    else:
        class_dirs = available_dirs

    if not class_dirs:
        raise ValueError(f"No test classes found under {test_root}")
    if limit_per_class is not None and limit_per_class <= 0:
        raise ValueError("limit_per_class must be positive")

    samples: list[SampleSpec] = []
    output_stems: set[str] = set()
    for class_dir in class_dirs:
        defect_type = class_dir.name
        _validate_name(defect_type, "class name")
        xyz_dir = class_dir / "xyz"
        gt_dir = class_dir / "gt"
        if not xyz_dir.is_dir():
            raise FileNotFoundError(f"Missing XYZ directory: {xyz_dir}")
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Missing GT directory: {gt_dir}")

        xyz_files = sorted(
            (
                path
                for path in xyz_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in TIFF_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
        if limit_per_class is not None:
            xyz_files = xyz_files[:limit_per_class]
        if not xyz_files:
            raise ValueError(f"No .tif/.tiff files found in {xyz_dir}")

        seen_source_ids: set[str] = set()
        for xyz_path in xyz_files:
            source_id = xyz_path.stem
            _validate_name(source_id, "sample id")
            source_key = source_id.casefold()
            if source_key in seen_source_ids:
                raise ValueError(f"Duplicate sample id {source_id!r} in {xyz_dir}")
            seen_source_ids.add(source_key)

            mask_path = gt_dir / f"{source_id}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing matching GT mask for {xyz_path}: {mask_path}"
                )
            sample = SampleSpec(defect_type, source_id, xyz_path, mask_path)
            output_key = sample.output_stem.casefold()
            if output_key in output_stems:
                raise ValueError(f"Output filename collision: {sample.output_stem}")
            output_stems.add(output_key)
            samples.append(sample)

    return category_root, test_root, samples


def _load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim == 2:
        mask = raw > 0
    elif raw.ndim == 3 and raw.shape[2] in (1, 3, 4):
        # Be tolerant of losslessly RGB-encoded masks; alpha is not a label.
        mask = np.any(raw[..., : min(raw.shape[2], 3)] > 0, axis=2)
    else:
        raise ValueError(f"Unsupported GT mask shape {raw.shape}: {path}")
    if mask.shape != expected_shape:
        raise ValueError(
            f"XYZ/GT spatial shape mismatch for {path}: "
            f"expected {expected_shape}, got {mask.shape}"
        )
    return np.asarray(mask, dtype=bool)


def identify_support_plane(
    organized: np.ndarray,
    depth_valid: np.ndarray,
    *,
    distance_threshold: float = DEFAULT_PLANE_DISTANCE_THRESHOLD,
    border_width: int = DEFAULT_PLANE_BORDER_WIDTH,
    ransac_n: int = DEFAULT_PLANE_RANSAC_N,
    ransac_iterations: int = DEFAULT_PLANE_RANSAC_ITERATIONS,
    ransac_seed: int = DEFAULT_PLANE_RANSAC_SEED,
) -> tuple[np.ndarray, dict[str, object]]:
    """Identify support-plane pixels using Real3D-style border RANSAC."""

    if o3d is None:
        raise RuntimeError(
            "Support-plane removal requires Open3D. Activate the covert "
            "environment or install it with: pip install open3d"
        )
    if not math.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("plane distance threshold must be positive and finite")
    height, width = organized.shape[:2]
    if border_width <= 0 or 2 * border_width >= min(height, width):
        raise ValueError(
            f"plane border width must be positive and less than half the image size; "
            f"got {border_width} for {height}x{width}"
        )
    if ransac_n < 3:
        raise ValueError("plane RANSAC requires at least 3 points")
    if ransac_iterations <= 0:
        raise ValueError("plane RANSAC iterations must be positive")

    border = np.zeros((height, width), dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border_valid = border.reshape(-1) & depth_valid
    flat_points = organized.reshape(-1, 3)
    border_points = np.ascontiguousarray(flat_points[border_valid], dtype=np.float64)
    if border_points.shape[0] < ransac_n:
        raise ValueError(
            f"Only {border_points.shape[0]} valid border points are available, "
            f"fewer than plane RANSAC n={ransac_n}"
        )

    # Open3D's global RANSAC RNG is seeded so repeated conversions are stable.
    if hasattr(o3d.utility, "random") and hasattr(o3d.utility.random, "seed"):
        o3d.utility.random.seed(int(ransac_seed))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(border_points))
    model, border_inliers = cloud.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=int(ransac_n),
        num_iterations=int(ransac_iterations),
    )
    model_array = np.asarray(model, dtype=np.float64)
    normal_norm = float(np.linalg.norm(model_array[:3]))
    if model_array.shape != (4,) or not np.isfinite(model_array).all() or normal_norm <= 0:
        raise RuntimeError(f"Open3D returned an invalid support plane: {model}")

    inlier_ratio = float(len(border_inliers) / border_points.shape[0])
    if inlier_ratio < MIN_PLANE_BORDER_INLIER_RATIO:
        raise RuntimeError(
            f"Support-plane fit is unreliable: only {inlier_ratio:.3f} of valid "
            f"border points are inliers (required >= {MIN_PLANE_BORDER_INLIER_RATIO:.3f})"
        )

    valid_indices = np.flatnonzero(depth_valid)
    valid_points = np.asarray(flat_points[valid_indices], dtype=np.float64)
    distances = np.abs(valid_points @ model_array[:3] + model_array[3]) / normal_norm
    plane_at_valid = distances <= distance_threshold
    plane_mask = np.zeros(flat_points.shape[0], dtype=bool)
    plane_mask[valid_indices[plane_at_valid]] = True
    if np.count_nonzero(depth_valid & ~plane_mask) == 0:
        raise RuntimeError("Support-plane removal would delete every valid point")

    stats: dict[str, object] = {
        "support_plane_points_removed": int(np.count_nonzero(plane_mask)),
        "plane_border_points": int(border_points.shape[0]),
        "plane_border_inliers": int(len(border_inliers)),
        "plane_border_inlier_ratio": inlier_ratio,
        "plane_model": tuple(float(value) for value in model_array),
    }
    return plane_mask, stats


def load_mvtec_sample(
    sample: SampleSpec,
    coordinate_scale: float = 1.0,
    *,
    remove_support_plane: bool = True,
    plane_distance_threshold: float = DEFAULT_PLANE_DISTANCE_THRESHOLD,
    plane_border_width: int = DEFAULT_PLANE_BORDER_WIDTH,
    plane_ransac_n: int = DEFAULT_PLANE_RANSAC_N,
    plane_ransac_iterations: int = DEFAULT_PLANE_RANSAC_ITERATIONS,
    plane_ransac_seed: int = DEFAULT_PLANE_RANSAC_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load one sample, remove missing depth and optionally its support plane."""

    organized = np.asarray(tifffile.imread(sample.xyz_path))
    if organized.ndim != 3 or organized.shape[2] != 3:
        raise ValueError(
            f"Expected an H x W x 3 XYZ TIFF, got {organized.shape}: {sample.xyz_path}"
        )
    if not np.issubdtype(organized.dtype, np.number):
        raise ValueError(f"XYZ TIFF is not numeric: {sample.xyz_path}")

    height, width = int(organized.shape[0]), int(organized.shape[1])
    pixel_mask = _load_mask(sample.mask_path, (height, width)).reshape(-1)
    flat_points = organized.reshape(-1, 3)

    # MVTec 3D-AD uses (0, 0, 0), hence z == 0, for missing depth.  Requiring
    # finite, positive z also rejects malformed measurements without relying on
    # x/y being non-zero (x == 0 or y == 0 can be a legitimate coordinate).
    finite = np.isfinite(flat_points).all(axis=1)
    depth_valid = finite & (flat_points[:, 2] > 0)
    source_positive = int(np.count_nonzero(pixel_mask))
    positive_after_depth = int(np.count_nonzero(pixel_mask[depth_valid]))
    if sample.is_good and source_positive:
        raise ValueError(
            f"Normal class 'good' has {source_positive} positive GT pixels: {sample.mask_path}"
        )

    plane_stats: dict[str, object] = {
        "support_plane_points_removed": 0,
        "plane_border_points": 0,
        "plane_border_inliers": 0,
        "plane_border_inlier_ratio": None,
        "plane_model": None,
    }
    if remove_support_plane:
        plane_mask, plane_stats = identify_support_plane(
            organized,
            depth_valid,
            distance_threshold=plane_distance_threshold,
            border_width=plane_border_width,
            ransac_n=plane_ransac_n,
            ransac_iterations=plane_ransac_iterations,
            ransac_seed=plane_ransac_seed,
        )
        valid = depth_valid & ~plane_mask
    else:
        valid = depth_valid

    points = np.ascontiguousarray(flat_points[valid], dtype=np.float32)
    if points.shape[0] == 0:
        raise ValueError(f"No valid XYZ points remain after filtering: {sample.xyz_path}")
    if coordinate_scale != 1.0:
        points *= np.float32(coordinate_scale)
    if not np.isfinite(points).all():
        raise ValueError(
            f"coordinate_scale={coordinate_scale} overflowed XYZ values: {sample.xyz_path}"
        )

    labels = np.ascontiguousarray(pixel_mask[valid].astype(np.uint8))
    valid_positive = int(np.count_nonzero(labels))

    stats = {
        "source_height": height,
        "source_width": width,
        "source_points": int(flat_points.shape[0]),
        "depth_valid_points": int(np.count_nonzero(depth_valid)),
        "valid_points": int(points.shape[0]),
        "invalid_points": int(flat_points.shape[0] - np.count_nonzero(depth_valid)),
        "source_positive_pixels": source_positive,
        "positive_points_after_depth_filter": positive_after_depth,
        "written_positive_points": valid_positive,
        "positive_pixels_without_valid_xyz": source_positive - positive_after_depth,
        "positive_points_removed_with_support_plane": positive_after_depth - valid_positive,
        **plane_stats,
    }
    return points, labels, stats


def _atomic_target(target: Path) -> tuple[object, Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    return handle, Path(handle.name)


def write_binary_xyz_pcd(path: Path, points: np.ndarray) -> None:
    """Write an Open3D-compatible PCD v0.7 binary XYZ file atomically."""

    array = np.ascontiguousarray(points, dtype="<f4")
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"Expected a non-empty N x 3 point array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Point array contains NaN or infinity")

    count = int(array.shape[0])
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {count}\n"
        "DATA binary\n"
    ).encode("ascii")

    handle, temporary = _atomic_target(path)
    try:
        with handle:
            handle.write(header)
            array.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_xyz_label_gt(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    """Write Real3D-AD-style ``x y z label`` text atomically."""

    point_array = np.asarray(points, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"Expected N x 3 points, got {point_array.shape}")
    if label_array.shape[0] != point_array.shape[0]:
        raise ValueError("GT label count does not match point count")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("GT labels must be binary 0/1")

    table = np.column_stack((point_array, label_array))
    handle, temporary = _atomic_target(path)
    try:
        with handle:
            np.savetxt(handle, table, fmt="%.8f %.8f %.8f %.6f")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_binary_xyz_pcd(path: Path) -> np.ndarray:
    """Strictly read back the PCD subset emitted by this converter."""

    header: dict[str, str] = {}
    with path.open("rb") as handle:
        for _ in range(64):
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"PCD header ended before DATA: {path}")
            line = raw_line.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=1)
            header[fields[0].upper()] = fields[1] if len(fields) == 2 else ""
            if fields[0].upper() == "DATA":
                payload = handle.read()
                break
        else:
            raise ValueError(f"PCD header exceeds 64 lines: {path}")

    expected = {
        "VERSION": "0.7",
        "FIELDS": "x y z",
        "SIZE": "4 4 4",
        "TYPE": "F F F",
        "COUNT": "1 1 1",
        "HEIGHT": "1",
        "DATA": "binary",
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise ValueError(f"Unexpected PCD {key}={header.get(key)!r}: {path}")
    try:
        point_count = int(header["POINTS"])
        width = int(header["WIDTH"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid PCD WIDTH/POINTS: {path}") from exc
    if point_count <= 0 or width != point_count:
        raise ValueError(f"Inconsistent PCD WIDTH/POINTS: {path}")
    expected_bytes = point_count * 3 * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PCD payload has {len(payload)} bytes; expected {expected_bytes}: {path}"
        )
    points = np.frombuffer(payload, dtype="<f4").reshape(point_count, 3).copy()
    if not np.isfinite(points).all():
        raise ValueError(f"PCD payload contains NaN/Inf: {path}")
    return points


def _write_json_atomic(path: Path, payload: object) -> None:
    handle, temporary = _atomic_target(path)
    try:
        with handle:
            encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def convert_dataset(
    source: Path,
    output: Path,
    *,
    selected_classes: Sequence[str] | None = None,
    limit_per_class: int | None = None,
    coordinate_scale: float = 1.0,
    remove_support_plane: bool = True,
    plane_distance_threshold: float = DEFAULT_PLANE_DISTANCE_THRESHOLD,
    plane_border_width: int = DEFAULT_PLANE_BORDER_WIDTH,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Convert selected samples and return the manifest payload."""

    if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be a positive finite number")
    category_root, test_root, samples = discover_samples(
        source, selected_classes=selected_classes, limit_per_class=limit_per_class
    )
    output_root = output.expanduser().resolve()
    if output_root in {category_root, test_root}:
        raise ValueError("Output must not overwrite the MVTec source/category test directory")

    output_test = output_root / "test"
    output_gt = output_root / "gt"
    manifest_path = output_root / "conversion_manifest.json"
    intended = [output_test / f"{sample.output_stem}.pcd" for sample in samples]
    intended.extend(
        output_gt / f"{sample.output_stem}.txt" for sample in samples if not sample.is_good
    )
    intended.append(manifest_path)
    existing = [path for path in intended if path.exists()]
    if existing and not overwrite:
        preview = "\n".join(f"  {path}" for path in existing[:10])
        suffix = "\n  ..." if len(existing) > 10 else ""
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output file(s). "
            f"Use --overwrite to replace them:\n{preview}{suffix}"
        )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "converter_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_category_root": str(category_root),
        "source_test_root": str(test_root),
        "output_root": str(output_root),
        "pcd_format": "PCD v0.7, binary little-endian float32, fields x y z",
        "gt_format": "space-delimited x y z binary_label, defective samples only",
        "valid_point_rule": "finite XYZ and z > 0",
        "coordinate_scale": float(coordinate_scale),
        "support_plane_removal": {
            "enabled": bool(remove_support_plane),
            "method": "Open3D RANSAC fitted to organized-image border points",
            "distance_threshold_in_source_units": float(plane_distance_threshold),
            "border_width_pixels": int(plane_border_width),
            "ransac_n": DEFAULT_PLANE_RANSAC_N,
            "ransac_iterations": DEFAULT_PLANE_RANSAC_ITERATIONS,
            "ransac_seed": DEFAULT_PLANE_RANSAC_SEED,
            "minimum_border_inlier_ratio": MIN_PLANE_BORDER_INLIER_RATIO,
        },
        "selected_classes": sorted({sample.defect_type for sample in samples}),
        "limit_per_class": limit_per_class,
        "dry_run": bool(dry_run),
        "sample_count": len(samples),
        "records": [],
    }
    if dry_run:
        manifest["planned_outputs"] = [str(path) for path in intended[:-1]]
        return manifest

    records: list[dict[str, object]] = []
    for index, sample in enumerate(samples, start=1):
        points, labels, stats = load_mvtec_sample(
            sample,
            coordinate_scale,
            remove_support_plane=remove_support_plane,
            plane_distance_threshold=plane_distance_threshold,
            plane_border_width=plane_border_width,
        )
        pcd_path = output_test / f"{sample.output_stem}.pcd"
        gt_path = None if sample.is_good else output_gt / f"{sample.output_stem}.txt"
        write_binary_xyz_pcd(pcd_path, points)
        # A lightweight read-back verifies header, point count and payload bytes.
        verified_points = read_binary_xyz_pcd(pcd_path)
        if not np.array_equal(verified_points, points):
            raise RuntimeError(f"PCD read-back mismatch: {pcd_path}")
        if gt_path is not None:
            write_xyz_label_gt(gt_path, points, labels)

        record = ConversionRecord(
            defect_type=sample.defect_type,
            source_id=sample.source_id,
            source_xyz=str(sample.xyz_path),
            source_mask=str(sample.mask_path),
            output_pcd=str(pcd_path),
            output_gt=str(gt_path) if gt_path is not None else None,
            coordinate_scale=float(coordinate_scale),
            **stats,
        )
        records.append(asdict(record))
        print(
            f"[{index:>3}/{len(samples)}] {sample.output_stem}: "
            f"{stats['valid_points']} points, "
            f"{stats['written_positive_points']} positive, "
            f"{stats['support_plane_points_removed']} plane points removed"
        )

    manifest["records"] = records
    manifest["total_valid_points"] = sum(int(row["valid_points"]) for row in records)
    manifest["total_positive_points"] = sum(
        int(row["written_positive_points"]) for row in records
    )
    manifest["total_positive_pixels_without_valid_xyz"] = sum(
        int(row["positive_pixels_without_valid_xyz"]) for row in records
    )
    manifest["total_support_plane_points_removed"] = sum(
        int(row["support_plane_points_removed"]) for row in records
    )
    manifest["total_positive_points_removed_with_support_plane"] = sum(
        int(row["positive_points_removed_with_support_plane"]) for row in records
    )
    _write_json_atomic(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an MVTec 3D-AD category test split to binary XYZ PCD files "
            "and Real3D-AD-style x y z label GT files for COVERT."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="MVTec category root (for example .../potato) or its test directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination root; test/ and gt/ are created below it.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Optional test classes to convert (for example good cut combined).",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="Convert at most this many samples from each selected class (smoke tests).",
    )
    parser.add_argument(
        "--coordinate-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply XYZ by this value before writing (default: 1.0). "
            "COVERT normalizes each cloud, so no unit conversion is normally needed."
        ),
    )
    parser.add_argument(
        "--keep-support-plane",
        action="store_true",
        help="Disable the default Real3D-style support-plane removal.",
    )
    parser.add_argument(
        "--plane-distance-threshold",
        type=float,
        default=DEFAULT_PLANE_DISTANCE_THRESHOLD,
        help=(
            "RANSAC support-plane distance threshold in source coordinate units "
            f"(default: {DEFAULT_PLANE_DISTANCE_THRESHOLD}, i.e. 5 mm for MVTec)."
        ),
    )
    parser.add_argument(
        "--plane-border-width",
        type=int,
        default=DEFAULT_PLANE_BORDER_WIDTH,
        help=(
            "Width in pixels of the organized-image border used to fit the plane "
            f"(default: {DEFAULT_PLANE_BORDER_WIDTH})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace colliding output files. Unrelated files are never removed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovery and print planned files without writing output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = convert_dataset(
            args.source,
            args.output,
            selected_classes=args.classes,
            limit_per_class=args.limit_per_class,
            coordinate_scale=args.coordinate_scale,
            remove_support_plane=not args.keep_support_plane,
            plane_distance_threshold=args.plane_distance_threshold,
            plane_border_width=args.plane_border_width,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(
            f"Done: {manifest['sample_count']} samples -> {manifest['output_root']}"
        )
        print(
            f"Manifest: {Path(str(manifest['output_root'])) / 'conversion_manifest.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
