"""Checkpoint normalization and loading helpers for vision backbones."""

from sapolicy.logger import Log


def _prepare_state_dict(state_dict):
    """Normalize common checkpoint wrappers into a plain state dict."""
    if not isinstance(state_dict, dict):
        return state_dict

    for key in ("model", "state_dict", "teacher", "student"):
        value = state_dict.get(key, None)
        if isinstance(value, dict):
            return _prepare_state_dict(value)

    return state_dict


def _summarize_keys(keys, max_items=5):
    """Create a compact summary for missing or unexpected state-dict keys."""
    if not keys:
        return ""
    if len(keys) <= max_items:
        return ", ".join(keys)
    shown = ", ".join(keys[:max_items])
    return f"{shown}, ... ({len(keys)} total)"


def _load_pretrained_weights(
    module,
    state_dict,
    allow_partial=False,
    log_prefix="",
    fallback_to_partial=True,
):
    """Load module weights, optionally skipping unmatched tensors."""
    module_state = module.state_dict()
    cleaned_state = {}
    skipped_for_shape = 0
    skipped_missing = 0

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        if allow_partial:
            target_param = module_state.get(key, None)
            if target_param is None:
                print(f"Missing key: {key}")
                skipped_missing += 1
                continue
            if target_param.shape != value.shape:
                print(f"Shape mismatch: {key} {target_param.shape} {value.shape}")
                skipped_for_shape += 1
                continue

        cleaned_state[key] = value

    try:
        missing, unexpected = module.load_state_dict(
            cleaned_state, strict=not allow_partial
        )
    except RuntimeError as err:
        if allow_partial:
            raise
        if not fallback_to_partial:
            raise RuntimeError(
                f"{log_prefix}Strict checkpoint load failed; the checkpoint "
                "does not match the requested backbone architecture."
            ) from err
        err_msg = str(err).split("\n")[0]
        Log.warn(
            f"{log_prefix}Strict checkpoint load failed ({err_msg}). "
            "Retrying with partial matches."
        )
        return _load_pretrained_weights(
            module,
            state_dict,
            allow_partial=True,
            log_prefix=log_prefix,
            fallback_to_partial=fallback_to_partial,
        )

    if allow_partial:
        if skipped_missing or skipped_for_shape:
            Log.warn(
                f"{log_prefix}Skipped {skipped_missing} unmatched keys and "
                f"{skipped_for_shape} shape-mismatched tensors when loading weights."
            )
        if missing:
            Log.warn(f"{log_prefix}Missing keys after load: {_summarize_keys(missing)}")
        if unexpected:
            Log.warn(
                f"{log_prefix}Unexpected keys after load: {_summarize_keys(unexpected)}"
            )
    elif missing or unexpected:
        Log.warn(
            f"{log_prefix}Missing keys: {_summarize_keys(missing)}, "
            f"unexpected keys: {_summarize_keys(unexpected)}"
        )
