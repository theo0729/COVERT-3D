"""General file I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to YAML."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to pretty JSON."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
