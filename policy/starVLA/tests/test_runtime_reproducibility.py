from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


POLICY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = POLICY_DIR / "source_starvla"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

try:
    from XPolicyLab.policy.starVLA.runtime_config import resolve_include_state
except ImportError:
    resolve_include_state = None


class RuntimeRegistryTest(unittest.TestCase):
    def test_public_and_legacy_mixtures_share_h50_q99_schema(self):
        from starVLA.dataloader.gr00t_lerobot.registry import (
            DATASET_NAMED_MIXTURES,
            ROBOT_TYPE_CONFIG_MAP,
        )

        public_type = DATASET_NAMED_MIXTURES["robodojo_arx_x5_h50_q99"][0][2]
        legacy_type = DATASET_NAMED_MIXTURES["robodojo_v21_all_h50_q99"][0][2]

        self.assertEqual(public_type, "robodojo_arx_x5_h50_q99")
        self.assertEqual(legacy_type, public_type)
        config = ROBOT_TYPE_CONFIG_MAP[public_type]
        self.assertEqual(config.action_indices, list(range(50)))
        self.assertEqual(sum(config.action_key_dims.values()), 14)
        self.assertEqual(sum(config.state_key_dims.values()), 14)


class IncludeStateResolutionTest(unittest.TestCase):
    def test_model_import_resolves_runtime_config_with_repo_pythonpath(self):
        module = importlib.import_module("XPolicyLab.policy.starVLA.model")
        self.assertTrue(hasattr(module, "Model"))

    def setUp(self):
        self.assertTrue(
            callable(resolve_include_state),
            "runtime_config.resolve_include_state is not implemented",
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)
        self.checkpoint = self.run_dir / "checkpoints" / "model.pt"
        self.checkpoint.parent.mkdir()
        self.checkpoint.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_yaml(self, name: str, include_state: bool | None):
        config = {"datasets": {"vla_data": {}}}
        if include_state is not None:
            config["datasets"]["vla_data"]["include_state"] = include_state
        (self.run_dir / name).write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_explicit_value_overrides_checkpoint_config(self):
        self._write_yaml("config.yaml", True)
        self.assertFalse(resolve_include_state("false", self.checkpoint))
        self.assertTrue(resolve_include_state("true", self.checkpoint))

    def test_auto_prefers_config_yaml_then_full_config(self):
        self._write_yaml("config.yaml", False)
        self._write_yaml("config.full.yaml", True)
        self.assertFalse(resolve_include_state("auto", self.checkpoint))

        (self.run_dir / "config.yaml").unlink()
        self.assertTrue(resolve_include_state("auto", self.checkpoint))

    def test_auto_defaults_to_false_without_checkpoint_setting(self):
        self._write_yaml("config.yaml", None)
        self.assertFalse(resolve_include_state("auto", self.checkpoint))


if __name__ == "__main__":
    unittest.main()
