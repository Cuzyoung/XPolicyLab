"""Train the vendored SAPolicy with a resolved native-ABC training configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from XPolicyLab.policy.SAPolicy import ensure_sapolicy_on_path
from XPolicyLab.policy.SAPolicy.data import load_training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check", action="store_true", help="Validate inputs without loading weights or training"
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_training_config(args.config, args.overrides)
    # Never delete a previous experiment as an implicit consequence of training.
    cfg.confirm_delete_previous_dir = False
    cfg.preserve_output_dir = True
    cfg.model.clear_output_dir = False
    output = Path(str(cfg.output_dir)).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not cfg.get("resume_training", False):
        raise FileExistsError(
            f"Training output is nonempty: {output}; select a new output_dir "
            "or explicit resume_training=true"
        )
    if cfg.get("logger") is not None:
        raise ValueError("This entry requires logger: null; no external logging is configured")
    if not cfg.get("callbacks", {}).get("model_checkpoint"):
        raise ValueError("Training requires an explicit callbacks.model_checkpoint policy")
    if cfg.get("resume_training", False):
        resume = Path(
            str(
                cfg.get("resume_checkpoint")
                or Path(str(cfg.callbacks.model_checkpoint.dirpath)) / "last.ckpt"
            )
        )
        if not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {resume}")
        cfg.resume_checkpoint = str(resume.resolve())
    cfg.setdefault("logger", None)
    for path in (cfg.model.pipeline.load_pretrain_backbone, cfg.get("warm_start_ckpt")):
        if path and not Path(str(path)).is_file():
            raise FileNotFoundError(str(path))
    ensure_sapolicy_on_path()
    if args.check:
        print(f"Training configuration valid: {output}")
        return
    from sapolicy.entrys.train_net import train_net

    train_net(cfg)


if __name__ == "__main__":
    main()
