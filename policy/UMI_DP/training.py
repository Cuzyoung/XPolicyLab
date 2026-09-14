"""Run the vendored LeRobot training workspace using the standard run directory."""

import json
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path="upstream/diffusion_policy/config")
def main(cfg):
    info = json.loads((Path(os.environ["UMI_DP_DATASET_DIR"]) / "meta/info.json").read_text())
    with __import__("omegaconf").open_dict(cfg):
        cfg.source_fps = float(info["fps"])
    OmegaConf.resolve(cfg)
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
