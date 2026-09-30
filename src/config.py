"""Project configuration loading and local path overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and merge an optional sibling ``local.yaml`` override."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")

    local_path = path.with_name("local.yaml")
    if local_path.is_file() and local_path.resolve() != path.resolve():
        with local_path.open("r", encoding="utf-8") as stream:
            override = yaml.safe_load(stream) or {}
        if not isinstance(override, dict):
            raise ValueError(f"Local configuration must contain a YAML mapping: {local_path}")
        config = _merge_mapping(config, override)
    return config


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve absolute paths as-is and relative paths from the project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(project_root) / path
