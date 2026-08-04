from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_AUTO_VALUES = {None, "", "auto", "none", "null"}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _checkpoint_run_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        return path.parent if path.name in {"checkpoints", "final_model"} else path
    if path.parent.name in {"checkpoints", "final_model"}:
        return path.parent.parent
    return path.parent


def _read_include_state(config_path: Path) -> bool | None:
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    try:
        value = config["datasets"]["vla_data"]["include_state"]
    except (KeyError, TypeError):
        return None
    return _parse_bool(value)


def resolve_include_state(value: Any, checkpoint_path: str | Path | None) -> bool:
    """Resolve state input from an override or checkpoint-side configuration."""

    normalized = value.strip().lower() if isinstance(value, str) else value
    if normalized not in _AUTO_VALUES:
        return _parse_bool(value)
    if checkpoint_path in (None, "", "null", "None"):
        return False

    run_dir = _checkpoint_run_dir(checkpoint_path)
    for name in ("config.yaml", "config.full.yaml"):
        include_state = _read_include_state(run_dir / name)
        if include_state is not None:
            return include_state
    return False
