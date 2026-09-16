"""Visualization-free point-cloud loading for the consolidated pipeline."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np


_PCD_TYPES = {
    ("F", 4): "f4",
    ("F", 8): "f8",
    ("I", 1): "i1",
    ("I", 2): "i2",
    ("I", 4): "i4",
    ("I", 8): "i8",
    ("U", 1): "u1",
    ("U", 2): "u2",
    ("U", 4): "u4",
    ("U", 8): "u8",
}

_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _validated_xyz(values: np.ndarray, *, source: Path) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape (N, 3): {source}")
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud is empty: {source}")
    if not np.isfinite(points).all():
        raise ValueError(f"Point cloud contains NaN/Inf: {source}")
    return points


def _read_ascii_matrix(handle: BinaryIO, *, rows: int | None = None) -> np.ndarray:
    values: list[np.ndarray] = []
    while rows is None or len(values) < rows:
        line = handle.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue
        values.append(np.fromstring(stripped.decode("ascii"), sep=" ", dtype=np.float64))
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    width = values[0].size
    if any(row.size != width for row in values):
        raise ValueError("Inconsistent ASCII point-record width.")
    return np.vstack(values)


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Decode one LZF stream with strict input, reference, and size checks."""

    if expected_size < 0:
        raise ValueError("LZF expected size must be non-negative.")
    source = memoryview(data)
    output = bytearray()
    cursor = 0
    while cursor < len(source):
        control = int(source[cursor])
        cursor += 1
        if control < 32:
            literal_length = control + 1
            literal_end = cursor + literal_length
            if literal_end > len(source):
                raise ValueError("Truncated LZF literal run.")
            if len(output) + literal_length > expected_size:
                raise ValueError("LZF output exceeds the declared uncompressed size.")
            output.extend(source[cursor:literal_end])
            cursor = literal_end
            continue

        match_length = control >> 5
        if match_length == 7:
            if cursor >= len(source):
                raise ValueError("Truncated LZF extended match length.")
            match_length += int(source[cursor])
            cursor += 1
        if cursor >= len(source):
            raise ValueError("Truncated LZF back reference.")
        reference_offset = ((control & 0x1F) << 8) + int(source[cursor]) + 1
        cursor += 1
        reference = len(output) - reference_offset
        if reference < 0:
            raise ValueError("Invalid LZF back reference.")
        match_length += 2
        if len(output) + match_length > expected_size:
            raise ValueError("LZF output exceeds the declared uncompressed size.")
        # LZF permits overlapping matches, so copy forward one byte at a time.
        for _ in range(match_length):
            output.append(output[reference])
            reference += 1

    if len(output) != expected_size:
        raise ValueError(
            "LZF decompressed size does not match the declared uncompressed size: "
            f"expected {expected_size}, got {len(output)}."
        )
    return bytes(output)


def _load_pcd(path: Path) -> np.ndarray:
    header: dict[str, list[str]] = {}
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA record: {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break

        fields = header.get("FIELDS") or header.get("FIELD")
        sizes = [int(value) for value in header.get("SIZE", [])]
        types = [value.upper() for value in header.get("TYPE", [])]
        if not fields or len(fields) != len(sizes) or len(fields) != len(types):
            raise ValueError(f"Incomplete PCD field schema: {path}")
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        if len(counts) != len(fields):
            raise ValueError(f"Invalid PCD COUNT schema: {path}")
        if any(size <= 0 for size in sizes) or any(count <= 0 for count in counts):
            raise ValueError(f"PCD SIZE and COUNT values must be positive: {path}")
        point_count = int(
            (header.get("POINTS") or [
                str(int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0]))
            ])[0]
        )
        if point_count <= 0:
            raise ValueError(f"PCD POINTS must be positive: {path}")
        encoding = header["DATA"][0].lower()

        offsets: dict[str, int] = {}
        cursor = 0
        for name, count in zip(fields, counts, strict=True):
            offsets[name.lower()] = cursor
            cursor += count
        if not {"x", "y", "z"}.issubset(offsets):
            raise ValueError(f"PCD has no complete XYZ fields: {path}")
        field_index = {name.lower(): index for index, name in enumerate(fields)}
        if any(counts[field_index[name]] != 1 for name in ("x", "y", "z")):
            raise ValueError(f"Vector-valued PCD coordinate fields are unsupported: {path}")

        if encoding == "ascii":
            matrix = _read_ascii_matrix(handle, rows=point_count)
            if matrix.shape != (point_count, cursor):
                raise ValueError(f"PCD ASCII data size does not match header: {path}")
            xyz = matrix[:, [offsets["x"], offsets["y"], offsets["z"]]]
            return _validated_xyz(xyz, source=path)
        if encoding not in {"binary", "binary_compressed"}:
            raise ValueError(
                "Unsupported PCD DATA encoding "
                f"{encoding!r}; expected ascii, binary, or binary_compressed: {path}"
            )

        scalar_dtypes: list[np.dtype] = []
        for name, size, kind, count in zip(fields, sizes, types, counts, strict=True):
            code = _PCD_TYPES.get((kind, size))
            if code is None:
                raise ValueError(f"Unsupported PCD scalar type {kind}{size}: {path}")
            scalar_dtypes.append(np.dtype("<" + code))

        if encoding == "binary_compressed":
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated PCD binary_compressed size prefix: {path}")
            compressed_size, uncompressed_size = struct.unpack("<II", prefix)
            payload = handle.read()
            if len(payload) != compressed_size:
                raise ValueError(
                    "PCD binary_compressed payload length does not match compressed_size: "
                    f"expected {compressed_size}, got {len(payload)}: {path}"
                )
            field_bytes = [
                point_count * size * count
                for size, count in zip(sizes, counts, strict=True)
            ]
            expected_uncompressed_size = sum(field_bytes)
            if uncompressed_size != expected_uncompressed_size:
                raise ValueError(
                    "PCD binary_compressed uncompressed_size does not match field schema: "
                    f"expected {expected_uncompressed_size}, got {uncompressed_size}: {path}"
                )
            decompressed = _lzf_decompress(payload, uncompressed_size)
            block_offsets: list[int] = []
            block_cursor = 0
            for length in field_bytes:
                block_offsets.append(block_cursor)
                block_cursor += length
            xyz = np.column_stack(
                tuple(
                    np.frombuffer(
                        decompressed,
                        dtype=scalar_dtypes[field_index[name]],
                        count=point_count,
                        offset=block_offsets[field_index[name]],
                    )
                    for name in ("x", "y", "z")
                )
            )
            return _validated_xyz(xyz, source=path)

        dtype_fields: list[tuple[object, ...]] = []
        for name, count, scalar_dtype in zip(
            fields, counts, scalar_dtypes, strict=True
        ):
            dtype_fields.append(
                (name, scalar_dtype)
                if count == 1
                else (name, scalar_dtype, (count,))
            )
        records = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=point_count)
        if records.size != point_count:
            raise ValueError(f"PCD binary data size does not match header: {path}")
        xyz = np.column_stack(
            tuple(np.asarray(records[name]).reshape(point_count) for name in ("x", "y", "z"))
        )
        return _validated_xyz(xyz, source=path)


def _load_ply(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Invalid PLY signature: {path}")
        encoding = ""
        vertex_count: int | None = None
        in_vertices = False
        properties: list[tuple[str, str]] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            values = line.decode("ascii").strip().split()
            if not values or values[0] in {"comment", "obj_info"}:
                continue
            if values[0] == "format":
                encoding = values[1]
            elif values[0] == "element":
                in_vertices = values[1] == "vertex"
                if in_vertices:
                    vertex_count = int(values[2])
            elif values[0] == "property" and in_vertices:
                if values[1] == "list":
                    raise ValueError(f"List property inside PLY vertex element: {path}")
                properties.append((values[2], values[1]))
            elif values[0] == "end_header":
                break
        if vertex_count is None or not {"x", "y", "z"}.issubset(
            name for name, _kind in properties
        ):
            raise ValueError(f"PLY has no complete vertex XYZ schema: {path}")
        names = [name for name, _kind in properties]
        xyz_columns = [names.index(name) for name in ("x", "y", "z")]
        if encoding == "ascii":
            matrix = _read_ascii_matrix(handle, rows=vertex_count)
            if matrix.shape != (vertex_count, len(properties)):
                raise ValueError(f"PLY ASCII data size does not match header: {path}")
            return _validated_xyz(matrix[:, xyz_columns], source=path)
        if encoding not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY encoding {encoding!r}: {path}")
        endian = "<" if encoding == "binary_little_endian" else ">"
        try:
            dtype = np.dtype(
                [(name, np.dtype(endian + _PLY_TYPES[kind])) for name, kind in properties]
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY scalar type {exc.args[0]!r}: {path}") from exc
        records = np.fromfile(handle, dtype=dtype, count=vertex_count)
        if records.size != vertex_count:
            raise ValueError(f"PLY binary data size does not match header: {path}")
        xyz = np.column_stack(tuple(records[name] for name in ("x", "y", "z")))
        return _validated_xyz(xyz, source=path)


def load_point_cloud_file(path: str | Path) -> np.ndarray:
    """Load XYZ without importing visualization libraries."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Point-cloud file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError(f"NumPy point cloud must have at least three columns: {source}")
        return _validated_xyz(values[:, :3], source=source)
    if suffix == ".pcd":
        return _load_pcd(source)
    if suffix == ".ply":
        return _load_ply(source)
    if suffix in {".txt", ".xyz", ".csv"}:
        matrix = np.genfromtxt(
            source,
            delimiter="," if suffix == ".csv" else None,
            dtype=np.float64,
        )
        matrix = np.atleast_2d(matrix)
        if matrix.shape[1] < 3:
            raise ValueError(f"Text point cloud must have at least three columns: {source}")
        return _validated_xyz(matrix[:, :3], source=source)
    raise ValueError(f"Unsupported point-cloud extension {suffix!r}: {source}")


__all__ = ["load_point_cloud_file"]
