"""XPolicyLab entry for SAPolicy's vendored inference source."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

DEFAULT_SAPOLICY_ROOT = Path(__file__).resolve().parent / "upstream"


def ensure_sapolicy_on_path(root: str | Path | None = None) -> Path:
    resolved = Path(root or DEFAULT_SAPOLICY_ROOT).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"SAPolicy code not found at {resolved}. Install the vendored source."
        )
    loaded = sys.modules.get("sapolicy")
    if loaded is not None:
        locations = list(getattr(loaded, "__path__", []))
        if not locations or any(not Path(p).resolve().is_relative_to(resolved) for p in locations):
            raise RuntimeError("A different SAPolicy source is already imported; use a fresh model process")
    value = str(resolved)
    if value not in sys.path:
        sys.path.insert(0, value)
    return resolved


def get_model(usr_args: dict[str, Any]):
    """Use the maintained inference wrapper with the selected model source."""
    ensure_sapolicy_on_path(usr_args.get("sapolicy_root") or usr_args.get("workspace"))
    from .upstream.sapolicy.eval.robotwin.sa_policy_server import SAPolicyRoboTwinModel

    return SAPolicyRoboTwinModel(
        cfg_file=usr_args['sapolicy_cfg'],
        resolved_cfg=usr_args.get('resolved_cfg'),
        ckpt_path=usr_args['ckpt_path'],
        workspace=usr_args.get('workspace'),
        n_action_steps=int(usr_args.get('n_action_steps', 8)),
        device=usr_args.get('device', 'cuda'),
        use_ema=bool(usr_args.get('use_ema', True)),
        normalizer_path=usr_args.get('normalizer_path'),
        tcp_forward_offset_m=usr_args.get('tcp_forward_offset_m'),
        strict_checkpoint=True,
    )
