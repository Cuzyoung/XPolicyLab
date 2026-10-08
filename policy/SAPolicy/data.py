"""Validate and prepare the native ABC episode format without re-encoding RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from XPolicyLab.policy.SAPolicy import ensure_sapolicy_on_path


def load_training_config(path: str, overrides=()):
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    for split in ("train_dataset", "val_dataset"):
        node = config.data.get(split)
        if node is None:
            continue
        for dataset in node.dataset_opts:
            if dataset.get("_target_") != "sapolicy.dataset.abc_episode.AbcEpisodeDataset":
                raise ValueError("This entry supports native ABC episodes only")
            for key in ("data_root", "mjcf_path"):
                value = dataset.get(key)
                if not value or not Path(str(value)).exists():
                    raise FileNotFoundError(f"Set an existing absolute {key} in {split}: {value}")
            normalizer = dataset.get("normalizer_load_path")
            if normalizer and not Path(str(normalizer)).is_file():
                raise FileNotFoundError(f"Configured normalizer is missing: {normalizer}")
            if (
                dataset.get("normalize_actions", True)
                and not normalizer
                and not dataset.get("normalizer_save_path")
            ):
                raise ValueError(
                    "Provide normalizer_load_path, or an explicit normalizer_save_path "
                    "to fit training statistics"
                )
    return config


def prepare(config) -> dict:
    import hydra

    ensure_sapolicy_on_path()
    datasets = []
    for node in config.data.train_dataset.dataset_opts:
        # The matched dataset computes FK and fits/loads its own normalizer.
        # Reading real samples also verifies video decoding and camera aliases.
        dataset = hydra.utils.instantiate(node)
        for index in sorted({0, len(dataset) - 1}):
            sample = dataset[index]
            action = np.asarray(sample["action"])
            if action.shape != (
                int(config.action_sequence_length),
                int(config.model.pipeline.action_cfg.action_dim),
            ):
                raise ValueError(f"Unexpected action shape: {action.shape}")
            if not np.isfinite(action).all():
                raise ValueError("Dataset contains non-finite actions")
        datasets.append(
            {
                "name": str(node.get("dataset_name", "ABC")),
                "root": str(node.data_root),
                "samples": len(dataset),
                "cameras": list(dataset.camera_names),
            }
        )
    return {
        "format": "abc_native",
        "datasets": datasets,
        "action_source": "command_joints_fk",
        "extra_tcp_offset_m": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", required=True, help="Validation manifest; native episodes stay in place"
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    result = prepare(load_training_config(args.config, args.overrides))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
