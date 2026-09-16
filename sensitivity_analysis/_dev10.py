"""Frozen Dev10 physical-group handling for sensitivity aggregation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "experiments"
    / "dev10_physical_groups.csv"
)
FALLBACK_GROUPS = frozenset(
    {
        "fish/275_bulge",
        "fish/309_sink",
        "fish/310_bulge",
        "fish/319_bulge",
        "fish/328_bulges",
        "diamond/427_bulge",
        "diamond/470_bulge",
        "diamond/365_bulge",
        "diamond/370_bulge",
        "diamond/535_sink",
    }
)


def physical_base_sample_id(sample_id: str) -> str:
    value = str(sample_id).strip()
    return value[:-4] if value.lower().endswith("_cut") else value


def physical_group_key(category: str, sample_id: str) -> str:
    return f"{str(category).strip()}/{physical_base_sample_id(sample_id)}"


def load_dev10_groups(manifest_path: Path = DEFAULT_MANIFEST) -> tuple[frozenset[str], dict[str, Any]]:
    """Load the public frozen Dev10 manifest, falling back to the same fixed list."""

    path = Path(manifest_path)
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        groups = frozenset(
            str(row.get("development_group_key") or "").strip()
            for row in rows
            if str(row.get("development_group_key") or "").strip()
        )
        if groups != FALLBACK_GROUPS:
            raise RuntimeError(
                "The frozen development_manifest.csv does not match the declared Dev10 groups: "
                f"missing={sorted(FALLBACK_GROUPS - groups)}, "
                f"unexpected={sorted(groups - FALLBACK_GROUPS)}"
            )
        metadata = {
            "source": "public_frozen_dev10_manifest",
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "physical_group_count": len(groups),
        }
        return groups, metadata

    metadata = {
        "source": "explicit_fallback",
        "path": str(path.resolve()),
        "sha256": None,
        "physical_group_count": len(FALLBACK_GROUPS),
    }
    return FALLBACK_GROUPS, metadata


def is_dev10(category: str, sample_id: str, groups: frozenset[str]) -> bool:
    return physical_group_key(category, sample_id) in groups


__all__ = [
    "DEFAULT_MANIFEST",
    "FALLBACK_GROUPS",
    "is_dev10",
    "load_dev10_groups",
    "physical_base_sample_id",
    "physical_group_key",
]
