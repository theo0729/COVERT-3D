"""Point cloud and ground-truth label I/O utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

o3d = None


def _require_open3d():
    """Load the optional point-cloud backend only for an operation that needs it."""

    global o3d
    if o3d is None:
        try:
            import open3d as loaded_open3d
        except ModuleNotFoundError as exc:  # pragma: no cover - local dependency.
            raise RuntimeError("This operation requires Open3D.") from exc
        o3d = loaded_open3d
    return o3d

logger = logging.getLogger(__name__)

POINT_CLOUD_SUFFIXES = {".npy", ".ply", ".pcd", ".txt", ".xyz", ".csv"}
LABEL_SUFFIXES = {".npy", ".txt", ".csv"}


def load_point_cloud_file(path: str | Path) -> np.ndarray:
    """Load a point cloud from a supported file and return XYZ coordinates."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Point-cloud file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in POINT_CLOUD_SUFFIXES:
        raise ValueError(
            f"Unsupported point-cloud file extension '{suffix}'. "
            f"Supported: {sorted(POINT_CLOUD_SUFFIXES)}"
        )

    if suffix == ".npy":
        data = np.load(input_path)
        return _extract_xyz(data, source=input_path)

    if suffix in {".ply", ".pcd"}:
        backend = _require_open3d()
        point_cloud = backend.io.read_point_cloud(str(input_path))
        points = np.asarray(point_cloud.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Open3D returned invalid points for file: {input_path}")
        if points.shape[0] == 0:
            raise ValueError(f"Point cloud is empty: {input_path}")
        return points

    data = _load_text_array(input_path)
    return _extract_xyz(data, source=input_path)


def load_optional_labels(
    path: str | Path | None,
    num_points: int | None = None,
    reference_points: np.ndarray | None = None,
    coordinate_tolerance: float | None = None,
    gt_coordinate_decimals: int = 6,
    use_hash_match: bool = True,
) -> np.ndarray | None:
    """Load optional labels or index-list GT when it can be aligned safely."""
    labels, _ = load_optional_labels_with_meta(
        path,
        num_points=num_points,
        reference_points=reference_points,
        coordinate_tolerance=coordinate_tolerance,
        gt_coordinate_decimals=gt_coordinate_decimals,
        use_hash_match=use_hash_match,
    )
    return labels


def load_optional_labels_with_meta(
    path: str | Path | None,
    num_points: int | None = None,
    reference_points: np.ndarray | None = None,
    coordinate_tolerance: float | None = None,
    gt_coordinate_decimals: int = 6,
    use_hash_match: bool = True,
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    """Load labels and return coordinate-GT matching metadata when available."""
    if path is None:
        return None, None

    label_path = Path(path)
    if not label_path.exists() or not label_path.is_file():
        return None, None

    suffix = label_path.suffix.lower()
    if suffix not in LABEL_SUFFIXES:
        return None, None

    try:
        data = np.load(label_path) if suffix == ".npy" else _load_text_array(label_path)
    except ValueError as exc:
        logger.warning("Failed to load label file %s: %s", label_path, exc)
        return None, None

    labels, meta = _interpret_label_array_with_meta(
        data,
        num_points=num_points,
        source=label_path,
        reference_points=reference_points,
        coordinate_tolerance=coordinate_tolerance,
        gt_coordinate_decimals=gt_coordinate_decimals,
        use_hash_match=use_hash_match,
    )
    if labels is None:
        return None, meta
    return labels.astype(np.uint8), meta


def _interpret_label_array(
    data: np.ndarray,
    num_points: int | None,
    source: Path,
    reference_points: np.ndarray | None = None,
    coordinate_tolerance: float | None = None,
    gt_coordinate_decimals: int = 6,
    use_hash_match: bool = True,
) -> np.ndarray | None:
    """Interpret label arrays and return only labels."""
    labels, _ = _interpret_label_array_with_meta(
        data,
        num_points=num_points,
        source=source,
        reference_points=reference_points,
        coordinate_tolerance=coordinate_tolerance,
        gt_coordinate_decimals=gt_coordinate_decimals,
        use_hash_match=use_hash_match,
    )
    return labels


def _interpret_label_array_with_meta(
    data: np.ndarray,
    num_points: int | None,
    source: Path,
    reference_points: np.ndarray | None = None,
    coordinate_tolerance: float | None = None,
    gt_coordinate_decimals: int = 6,
    use_hash_match: bool = True,
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    """Interpret label arrays while avoiding likely coordinate GT misuse."""
    array = np.asarray(data)
    if array.size == 0:
        logger.warning("Label file is empty and will be ignored: %s", source)
        return None, None

    try:
        values = array.astype(np.float64)
    except (TypeError, ValueError):
        logger.warning("Label file is non-numeric and will be ignored: %s", source)
        return None, None
    if not np.isfinite(values).all():
        logger.warning("Label file contains NaN/Inf and will be ignored: %s", source)
        return None, None

    if values.ndim == 0:
        values = values.reshape(1)

    if values.ndim == 1:
        return _interpret_label_vector(values, num_points=num_points, source=source), None

    if values.ndim != 2:
        logger.warning(
            "Unsupported label array shape %s; ignore: %s",
            values.shape,
            source,
        )
        return None, None

    if values.shape[1] == 1:
        return _interpret_label_vector(values[:, 0], num_points=num_points, source=source), None

    if values.shape[0] == 1 and values.shape[1] != 3:
        return _interpret_label_vector(values.reshape(-1), num_points=num_points, source=source), None

    if values.shape[1] >= 4 and reference_points is not None and _looks_like_label_column(values[:, -1]):
        return _coordinate_label_gt_to_mask_with_meta(
            values=values,
            reference_points=reference_points,
            num_points=num_points,
            tolerance=coordinate_tolerance,
            source=source,
            hash_decimals=gt_coordinate_decimals,
            use_hash_match=use_hash_match,
        )

    if values.shape[1] >= 4 and _looks_like_xyz_label_array(values):
        if reference_points is None:
            logger.warning(
                "Label file looks like x y z label coordinates, "
                "but reference_points is not available; ignore: %s",
                source,
            )
            return None, _coordinate_label_base_meta(values, method="coordinate_label_missing_reference")
        return _coordinate_label_gt_to_mask_with_meta(
            values=values,
            reference_points=reference_points,
            num_points=num_points,
            tolerance=coordinate_tolerance,
            source=source,
            hash_decimals=gt_coordinate_decimals,
            use_hash_match=use_hash_match,
        )

    if values.shape[1] >= 3 and reference_points is not None:
        return _coordinate_gt_to_mask_with_meta(
            gt_points=values[:, :3],
            reference_points=reference_points,
            num_points=num_points,
            tolerance=coordinate_tolerance,
            source=source,
            hash_decimals=gt_coordinate_decimals,
            use_hash_match=use_hash_match,
        )

    if values.shape[1] >= 3:
        logger.warning(
            "Label file looks like defect coordinates (shape %s), "
            "but reference_points is not available; ignore: %s",
            values.shape,
            source,
        )
        return None, None

    logger.warning(
        "Label file shape %s cannot be aligned; ignore: %s",
        values.shape,
        source,
    )
    return None, None


def _interpret_label_vector(
    values: np.ndarray,
    num_points: int | None,
    source: Path,
) -> np.ndarray | None:
    """Interpret a one-dimensional label vector or anomaly-index list."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return None

    if num_points is None:
        return (vector > 0).astype(np.uint8)

    if vector.size == num_points:
        return (vector > 0).astype(np.uint8)

    if _is_integer_like(vector):
        indices = vector.astype(np.int64)
        if indices.size == 0:
            return np.zeros(num_points, dtype=np.uint8)

        mask = np.zeros(num_points, dtype=np.uint8)
        if indices.min(initial=0) >= 0 and indices.max(initial=-1) < num_points:
            mask[np.unique(indices)] = 1
            return mask
        if indices.min(initial=1) >= 1 and indices.max(initial=0) <= num_points:
            mask[np.unique(indices - 1)] = 1
            return mask

    logger.warning(
        "Label length %d does not match point count %d; ignore: %s",
        vector.size,
        num_points,
        source,
    )
    return None


def _is_integer_like(values: np.ndarray) -> bool:
    """Return True when all values represent integer indices."""
    return bool(np.all(np.equal(values, np.rint(values))))


def _looks_like_xyz_label_array(values: np.ndarray) -> bool:
    """Return True for table-like x y z label arrays."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4 or values.shape[0] == 0:
        return False
    label_values = values[:, -1]
    return _looks_like_label_column(label_values)


def _looks_like_label_column(values: np.ndarray) -> bool:
    """Heuristically detect a finite non-negative label column."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return False
    if np.any(values < 0):
        return False

    rounded = np.rint(values)
    integer_like = bool(np.all(np.isclose(values, rounded, atol=1e-8, rtol=0.0)))
    unique_values = np.unique(rounded if integer_like else values)
    if unique_values.size <= 2 and np.all(np.isin(unique_values, [0.0, 1.0])):
        return True
    if integer_like and unique_values.size <= max(16, int(np.sqrt(values.size))):
        return True
    if unique_values.size <= 8 and np.max(values) <= 255:
        return True
    return bool(unique_values.size <= max(8, values.size // 4))


def _looks_like_coordinate_columns(values: np.ndarray) -> bool:
    """Heuristically detect coordinate-like first three columns."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or values.shape[0] == 0:
        return False
    coordinates = values[:, :3]
    if not np.isfinite(coordinates).all():
        return False
    unique_counts = [np.unique(coordinates[:, axis]).size for axis in range(3)]
    if max(unique_counts) < min(10, coordinates.shape[0]):
        return False
    return bool(np.any(np.ptp(coordinates, axis=0) > 0.0))


def _coordinate_label_base_meta(values: np.ndarray, method: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    positive_count = int(np.sum(values[:, -1] > 0)) if values.ndim == 2 and values.shape[1] else 0
    return {
        "method": method,
        "format": "xyz_label",
        "num_label_rows": int(values.shape[0]) if values.ndim >= 1 else 0,
        "num_positive_label_rows": positive_count,
        "num_gt_input": positive_count,
        "num_gt_matched": 0,
        "match_ratio": 0.0,
        "tolerance": None,
    }


def _coordinate_label_gt_to_mask_with_meta(
    values: np.ndarray,
    reference_points: np.ndarray,
    num_points: int | None,
    tolerance: float | None,
    source: Path,
    hash_decimals: int = 6,
    use_hash_match: bool = True,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Interpret x y z label GT and align positive-label coordinates."""
    values = np.asarray(values, dtype=np.float64)
    positive_mask = values[:, -1] > 0
    positive_count = int(np.sum(positive_mask))
    if positive_count == 0:
        reference_points = np.asarray(reference_points)
        mask_length = reference_points.shape[0] if reference_points.ndim >= 1 else int(num_points or 0)
        meta = _coordinate_label_base_meta(values, method="coordinate_label_empty_positive")
        meta["match_ratio"] = 1.0
        meta["tolerance"] = 0.0
        return np.zeros(mask_length, dtype=np.uint8), meta

    mask, meta = _coordinate_gt_to_mask_with_meta(
        gt_points=values[positive_mask, :3],
        reference_points=reference_points,
        num_points=num_points,
        tolerance=tolerance,
        source=source,
        hash_decimals=hash_decimals,
        use_hash_match=use_hash_match,
    )
    enriched_meta = {
        **meta,
        "method": f"coordinate_label_{meta.get('method', 'unknown')}",
        "format": "xyz_label",
        "num_label_rows": int(values.shape[0]),
        "num_positive_label_rows": positive_count,
        "num_gt_input": positive_count,
    }
    return mask, enriched_meta


def _hash_decimal_candidates(start_decimals: int) -> list[int]:
    """Return hash decimal candidates from precise to coarser values."""
    start = max(0, int(start_decimals))
    stop = max(0, min(start, 4))
    return list(range(start, stop - 1, -1))


def _coordinate_hash_match(
    gt_points: np.ndarray,
    reference_points: np.ndarray,
    decimals: int = 6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fast exact-ish coordinate matching after decimal rounding."""
    rounded_ref = np.round(reference_points, decimals=decimals)
    rounded_gt = np.round(gt_points, decimals=decimals)
    ref_lookup: dict[tuple[float, float, float], int] = {}
    for idx, point in enumerate(rounded_ref):
        ref_lookup.setdefault((float(point[0]), float(point[1]), float(point[2])), idx)

    matched_indices: list[int] = []
    for point in rounded_gt:
        ref_idx = ref_lookup.get((float(point[0]), float(point[1]), float(point[2])))
        if ref_idx is not None:
            matched_indices.append(ref_idx)

    mask = np.zeros(reference_points.shape[0], dtype=np.uint8)
    if matched_indices:
        mask[np.unique(np.asarray(matched_indices, dtype=np.int64))] = 1

    num_matched = len(matched_indices)
    num_gt = int(gt_points.shape[0])
    meta: dict[str, object] = {
        "method": "hash",
        "num_gt_input": num_gt,
        "num_gt_matched": int(num_matched),
        "match_ratio": float(num_matched / num_gt) if num_gt else 0.0,
        "tolerance": 0.0,
        "decimals": int(decimals),
    }
    return mask, meta


def _coordinate_gt_to_mask(
    gt_points: np.ndarray,
    reference_points: np.ndarray,
    num_points: int | None,
    tolerance: float | None,
    source: Path,
    hash_decimals: int = 6,
    use_hash_match: bool = True,
) -> np.ndarray | None:
    """Map defect coordinates to nearest reference points."""
    mask, _ = _coordinate_gt_to_mask_with_meta(
        gt_points=gt_points,
        reference_points=reference_points,
        num_points=num_points,
        tolerance=tolerance,
        source=source,
        hash_decimals=hash_decimals,
        use_hash_match=use_hash_match,
    )
    return mask


def _coordinate_gt_to_mask_with_meta(
    gt_points: np.ndarray,
    reference_points: np.ndarray,
    num_points: int | None,
    tolerance: float | None,
    source: Path,
    hash_decimals: int = 6,
    use_hash_match: bool = True,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Map defect coordinates to nearest reference points and report match stats."""
    gt_points = np.asarray(gt_points, dtype=np.float64)
    reference_points = np.asarray(reference_points, dtype=np.float64)
    base_meta: dict[str, object] = {
        "method": "none",
        "num_gt_input": int(gt_points.shape[0]) if gt_points.ndim >= 1 else 0,
        "num_gt_matched": 0,
        "match_ratio": 0.0,
        "tolerance": None if tolerance is None else float(tolerance),
    }
    if reference_points.ndim != 2 or reference_points.shape[1] != 3:
        logger.warning("reference_points must have shape (N, 3); ignore GT: %s", source)
        return None, base_meta
    if gt_points.ndim != 2 or gt_points.shape[1] != 3 or gt_points.shape[0] == 0:
        logger.warning("GT coordinates must have shape (M, 3); ignore: %s", source)
        return None, base_meta
    if num_points is not None and reference_points.shape[0] != num_points:
        logger.warning(
            "Reference point count %d does not match num_points %d; ignore GT: %s",
            reference_points.shape[0],
            num_points,
            source,
        )
        return None, base_meta
    if not np.isfinite(gt_points).all() or not np.isfinite(reference_points).all():
        logger.warning("GT/reference coordinates contain NaN/Inf; ignore: %s", source)
        return None, base_meta

    if use_hash_match:
        for decimals in _hash_decimal_candidates(hash_decimals):
            hash_mask, hash_meta = _coordinate_hash_match(
                gt_points,
                reference_points,
                decimals=decimals,
            )
            if int(hash_meta["num_gt_matched"]) == int(hash_meta["num_gt_input"]):
                return hash_mask, hash_meta

    effective_tolerance = float(tolerance) if tolerance is not None else _estimate_coordinate_tolerance(reference_points)
    distances, indices, method = _nearest_reference_points(gt_points, reference_points)
    matched = distances <= effective_tolerance
    num_unmatched = int(np.sum(~matched))
    if num_unmatched:
        logger.warning(
            "%d/%d GT coordinates exceed tolerance %g: %s",
            num_unmatched,
            gt_points.shape[0],
            effective_tolerance,
            source,
        )
    if not np.any(matched):
        logger.warning("No GT coordinates matched the point cloud; ignore: %s", source)
        meta = {
            "method": method,
            "num_gt_input": int(gt_points.shape[0]),
            "num_gt_matched": 0,
            "match_ratio": 0.0,
            "tolerance": effective_tolerance,
        }
        return None, meta

    mask = np.zeros(reference_points.shape[0], dtype=np.uint8)
    mask[np.unique(indices[matched])] = 1
    num_matched = int(np.sum(matched))
    meta = {
        "method": method,
        "num_gt_input": int(gt_points.shape[0]),
        "num_gt_matched": num_matched,
        "match_ratio": float(num_matched / gt_points.shape[0]),
        "tolerance": effective_tolerance,
        "distance_mean": float(np.mean(distances[matched])) if np.any(matched) else None,
        "distance_max": float(np.max(distances[matched])) if np.any(matched) else None,
    }
    return mask, meta


def _nearest_reference_points(
    query_points: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Find nearest reference point using scipy, Open3D, then NumPy."""
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ModuleNotFoundError:
        open3d_result = _nearest_reference_points_open3d(query_points, reference_points)
        if open3d_result is not None:
            distances, indices = open3d_result
            return distances, indices, "open3d_kdtree"
        distances, indices = _nearest_reference_points_numpy(query_points, reference_points)
        return distances, indices, "numpy_chunked"

    tree = cKDTree(reference_points)
    distances, indices = tree.query(query_points, k=1)
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64), "scipy_ckdtree"


def _nearest_reference_points_open3d(
    query_points: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find nearest neighbors with Open3D KDTreeFlann when available."""
    if o3d is None:
        return None

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(reference_points.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    distances = np.empty(query_points.shape[0], dtype=np.float64)
    indices = np.empty(query_points.shape[0], dtype=np.int64)
    for i, point in enumerate(query_points):
        count, idx, dist_sq = tree.search_knn_vector_3d(point.astype(np.float64), 1)
        if count <= 0:
            distances[i] = np.inf
            indices[i] = -1
            continue
        indices[i] = int(idx[0])
        distances[i] = float(np.sqrt(dist_sq[0]))
    return distances, indices


def _nearest_reference_points_numpy(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked NumPy nearest-neighbor fallback for environments without scipy."""
    distances = np.empty(query_points.shape[0], dtype=np.float64)
    indices = np.empty(query_points.shape[0], dtype=np.int64)
    for start in range(0, query_points.shape[0], chunk_size):
        end = min(start + chunk_size, query_points.shape[0])
        diff = query_points[start:end, None, :] - reference_points[None, :, :]
        dist_sq = np.einsum("mnj,mnj->mn", diff, diff, optimize=True)
        local_indices = np.argmin(dist_sq, axis=1)
        indices[start:end] = local_indices
        distances[start:end] = np.sqrt(dist_sq[np.arange(end - start), local_indices])
    return distances, indices


def _estimate_coordinate_tolerance(reference_points: np.ndarray) -> float:
    """Estimate matching tolerance from median nearest-neighbor spacing."""
    if reference_points.shape[0] < 2:
        return 1e-8
    sample = reference_points
    if reference_points.shape[0] > 5000:
        rng = np.random.default_rng(0)
        sample_indices = rng.choice(reference_points.shape[0], size=5000, replace=False)
        sample = reference_points[sample_indices]

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ModuleNotFoundError:
        open3d_distances = _second_nearest_reference_points_open3d(sample, reference_points)
        if open3d_distances is not None:
            distances = open3d_distances
        else:
            distances, _ = _second_nearest_reference_points_numpy(sample)
    else:
        tree = cKDTree(reference_points)
        distances, _ = tree.query(sample, k=2)
        distances = np.asarray(distances, dtype=np.float64)[:, 1]

    finite_positive = distances[np.isfinite(distances) & (distances > 0)]
    if finite_positive.size == 0:
        return 1e-8
    return max(float(np.median(finite_positive) * 3.0), 1e-8)


def _second_nearest_reference_points_numpy(
    points: np.ndarray,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute second-nearest distances within a sample using chunked NumPy."""
    distances = np.empty(points.shape[0], dtype=np.float64)
    indices = np.empty(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        diff = points[start:end, None, :] - points[None, :, :]
        dist_sq = np.einsum("mnj,mnj->mn", diff, diff, optimize=True)
        local_indices = np.argpartition(dist_sq, kth=1, axis=1)[:, 1]
        indices[start:end] = local_indices
        distances[start:end] = np.sqrt(dist_sq[np.arange(end - start), local_indices])
    return distances, indices


def _second_nearest_reference_points_open3d(
    sample_points: np.ndarray,
    reference_points: np.ndarray,
) -> np.ndarray | None:
    """Estimate nearest-neighbor spacing with Open3D KDTreeFlann."""
    if o3d is None:
        return None

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(reference_points.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(point_cloud)
    distances = np.empty(sample_points.shape[0], dtype=np.float64)
    for i, point in enumerate(sample_points):
        count, _, dist_sq = tree.search_knn_vector_3d(point.astype(np.float64), 2)
        if count >= 2:
            distances[i] = float(np.sqrt(dist_sq[1]))
        else:
            distances[i] = np.inf
    return distances


def _load_text_array(path: Path) -> np.ndarray:
    """Load numeric arrays from text-like files."""
    delimiter = "," if path.suffix.lower() == ".csv" else None
    try:
        data = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - depends on file contents.
        raise ValueError(f"Failed to load text data from: {path}") from exc

    if data.size == 0:
        raise ValueError(f"Text data is empty: {path}")
    if data.ndim == 0:
        data = data.reshape(1, 1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _extract_xyz(data: np.ndarray, source: Path) -> np.ndarray:
    """Extract XYZ columns from a loaded array."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Point cloud must be a 2D array: {source}")
    if array.shape[1] < 3:
        raise ValueError(f"Point cloud must contain at least 3 columns: {source}")
    points = array[:, :3]
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {source}")
    return points
