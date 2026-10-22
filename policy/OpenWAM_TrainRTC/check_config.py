"""Compose and validate a training config without loading models or data."""
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent / "OpenWAM"
sys.path.insert(0, str(ROOT))
from openwam.model.continuation import ContinuationConfig

with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
    cfg = compose(config_name="train", overrides=sys.argv[1:])
value = OmegaConf.select(cfg, "model.architecture.continuation", default={})
c = ContinuationConfig.parse(value)
arch = cfg.model.architecture
if c.method != "none" and (arch.framework != "dual_system" or arch.variant not in ("joint_self_attn", "joint_cross_attn")):
    raise ValueError("Continuation requires dual_system joint_self_attn/joint_cross_attn")
if c.method == "legato" and cfg.model.action_backbone.shift_action != 1.0:
    raise ValueError("Legato requires shift_action=1.0")
print("source:", ROOT)
print(OmegaConf.to_yaml(cfg, resolve=True))
