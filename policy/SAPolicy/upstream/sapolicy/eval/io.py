"""Persist eval rollout metrics to disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if "video" in key:
            out[key] = str(value) if value is not None else None
        else:
            out[key] = _to_jsonable(value)
    return out


def save_eval_metrics(
    output_dir: str | Path,
    metrics: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "metrics.json"
    payload: dict[str, Any] = {"metrics": serialize_metrics(metrics)}
    if meta:
        payload["meta"] = _to_jsonable(meta)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out_path


def resolve_eval_run_dir(cfg, eval_cfg) -> str:
    """Canonical eval run directory under the experiment output tree."""
    explicit = eval_cfg.get("output_dir", None)
    if explicit is not None and str(explicit).strip():
        return os.path.expandvars(os.path.expanduser(str(explicit)))

    run_name = str(eval_cfg.get("run_name", "default"))
    return os.path.join(cfg.output_dir, "eval", run_name)
