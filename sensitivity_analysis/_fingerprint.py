"""Deterministic source fingerprint for the production COVERT execution path.

The dependency set is discovered from ``covert_sample.py`` by recursively
parsing local imports.  Only production packages that can affect inference or
evaluation are eligible; visualization, tests, baselines, ablations and the
sensitivity framework itself are deliberately excluded.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FINGERPRINT_SCHEMA = 2

_ENTRYPOINT_FILES = ("covert_sample.py",)
_ALLOWED_VAST_PACKAGES = frozenset(
    {
        "boundary",
        "covert",
        "data",
        "evaluation",
        "features",
        "graph",
        "restoration",
        "scoring",
    }
)

# These files were called out by the final audit because the v1 fingerprint
# omitted them.  Recursive discovery must find every one or fail loudly.
REQUIRED_IMPLEMENTATION_FILES = (
    "vast/data/io.py",
    "vast/data/preprocess.py",
    "vast/evaluation/scoring.py",
    "vast/features/hks.py",
    "vast/features/surface_variation.py",
    "vast/graph/edge_weight.py",
    "vast/graph/laplacian.py",
    "vast/graph/mutual_knn.py",
)


def _canonical_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Production dependency escaped project root: {path}") from exc


def _allowed_module(module: str) -> bool:
    if module == "covert_sample":
        return True
    parts = module.split(".")
    return bool(
        parts
        and parts[0] == "vast"
        and (len(parts) == 1 or parts[1] in _ALLOWED_VAST_PACKAGES)
    )


def _module_target(module: str) -> Path | None:
    """Resolve an eligible local module to its source file, if it exists."""

    if not _allowed_module(module):
        return None
    parts = module.split(".")
    module_file = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
    package_file = PROJECT_ROOT.joinpath(*parts, "__init__.py")
    if module_file.is_file():
        return module_file.resolve()
    if package_file.is_file():
        return package_file.resolve()
    return None


def _package_initializers(module: str) -> Iterable[Path]:
    """Yield package initializers executed while importing ``module``."""

    parts = module.split(".")
    for length in range(1, len(parts)):
        candidate = PROJECT_ROOT.joinpath(*parts[:length], "__init__.py")
        if candidate.is_file():
            yield candidate.resolve()


def _files_for_module(module: str) -> tuple[Path, ...]:
    target = _module_target(module)
    if target is None:
        return ()
    return tuple((*_package_initializers(module), target))


def _module_context(path: Path) -> tuple[str, str]:
    relative = path.resolve().relative_to(PROJECT_ROOT)
    if relative.name == "__init__.py":
        module = ".".join(relative.parent.parts)
        return module, module
    module = ".".join(relative.with_suffix("").parts)
    package = module.rpartition(".")[0]
    return module, package


def _imported_modules(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    _, current_package = _module_context(path)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if not current_package:
                raise RuntimeError(f"Relative import outside a package in {path}")
            relative_name = "." * node.level + (node.module or "")
            base = importlib.util.resolve_name(relative_name, current_package)
        else:
            base = node.module or ""
        if not base:
            continue
        modules.add(base)
        for alias in node.names:
            if alias.name != "*":
                modules.add(f"{base}.{alias.name}")
    return modules


def collect_production_implementation_files() -> tuple[Path, ...]:
    """Return the stable, duplicate-free production dependency source list."""

    pending = [(PROJECT_ROOT / relative).resolve() for relative in _ENTRYPOINT_FILES]
    missing_entrypoints = [path for path in pending if not path.is_file()]
    if missing_entrypoints:
        raise FileNotFoundError(
            f"Missing production fingerprint entrypoint(s): {missing_entrypoints}"
        )

    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in discovered:
            continue
        discovered.add(path)
        for module in _imported_modules(path):
            for dependency in _files_for_module(module):
                if dependency not in discovered:
                    pending.append(dependency)

    ordered = tuple(sorted(discovered, key=_canonical_relative))
    identities = {_canonical_relative(path) for path in ordered}
    missing_required = sorted(set(REQUIRED_IMPLEMENTATION_FILES) - identities)
    if missing_required:
        raise RuntimeError(
            "Production dependency discovery omitted required files: "
            f"{missing_required}"
        )
    if len(identities) != len(ordered):
        raise AssertionError("Duplicate canonical paths entered implementation fingerprint")
    return ordered


def implementation_files() -> tuple[str, ...]:
    return tuple(
        _canonical_relative(path) for path in collect_production_implementation_files()
    )


def implementation_sha256() -> str:
    """Hash schema, canonical relative path and raw bytes in stable order."""

    digest = hashlib.sha256()
    digest.update(f"schema:{IMPLEMENTATION_FINGERPRINT_SCHEMA}".encode("ascii"))
    digest.update(b"\0")
    for path in collect_production_implementation_files():
        identity = _canonical_relative(path)
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def production_file_sha256() -> dict[str, str]:
    return {
        _canonical_relative(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in collect_production_implementation_files()
    }


def implementation_metadata() -> dict[str, object]:
    files = implementation_files()
    return {
        "implementation_fingerprint_schema": IMPLEMENTATION_FINGERPRINT_SCHEMA,
        "implementation_file_count": len(files),
        "implementation_files": list(files),
        "implementation_hash": implementation_sha256(),
    }


__all__ = [
    "IMPLEMENTATION_FINGERPRINT_SCHEMA",
    "REQUIRED_IMPLEMENTATION_FILES",
    "collect_production_implementation_files",
    "implementation_files",
    "implementation_metadata",
    "implementation_sha256",
    "production_file_sha256",
]
