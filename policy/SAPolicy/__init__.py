"""Thin XPolicyLab entry for the local SAPolicy tree at ``~/sa``.

All model loading, DiT sampling, crop/resize, and TCP geometry live in
``~/sa/SpatialAlignPolicy``. This package only adapts that server to the
XPolicyLab ``Model`` interface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

DEFAULT_SAPOLICY_ROOT = Path.home() / "sa" / "SpatialAlignPolicy"


def ensure_sapolicy_on_path(root: str | Path | None = None) -> Path:
    resolved = Path(root or DEFAULT_SAPOLICY_ROOT).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"SAPolicy code not found at {resolved}. Expected ~/sa/SpatialAlignPolicy."
        )
    value = str(resolved)
    if value not in sys.path:
        sys.path.insert(0, value)
    return resolved


def get_model(usr_args: dict[str, Any]):
    """Same entry as ``sapolicy.eval.robotwin.deploy_policy.get_model``."""
    ensure_sapolicy_on_path(usr_args.get("sapolicy_root") or usr_args.get("workspace"))
    from sapolicy.eval.robotwin.deploy_policy import get_model as load_sapolicy

    return load_sapolicy(usr_args)
