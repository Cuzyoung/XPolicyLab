"""Planner-only tests using stdlib fixtures; never start a training subprocess."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[2] / "training/launch.py"
SPEC = importlib.util.spec_from_file_location("training_launch", FILE)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class TrainingPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.xpl = self.workspace / "XPolicyLab"
        self.policy = self.xpl / "policy/ACT"
        self.policy.mkdir(parents=True)
        (self.policy / "train.sh").write_text("exit 99\n")
        (self.policy / "process_data.sh").write_text("exit 99\n")
        self.config = {"policy": "ACT", "params": {"run": "demo", "python": sys.executable}}
        self.patches = [patch.object(launch, "XPL", self.xpl), patch.object(launch, "WORKSPACE", self.workspace)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_plan_does_not_run_processes_or_create_outputs(self):
        with patch.object(launch.subprocess, "Popen", side_effect=AssertionError("must not execute")):
            job = launch.plan(self.config, {})
        self.assertEqual(job["train"]["argv"][-6:], ["RoboDojo_real", "demo", "yam_dual", "joint", "0", "0"])
        self.assertEqual(len(job["prepare"]), 1)
        self.assertFalse(Path(job["output"]).exists())
        self.assertFalse(Path(job["log_dir"]).exists())

    def test_spaces_and_shell_metacharacters_are_literal_args(self):
        config = copy.deepcopy(self.config)
        config["train"] = {"args": ["${instruction}"]}
        text = "pick a cup; $(touch /tmp/not-executed)"
        job = launch.plan(config, {"instruction": text})
        self.assertEqual(job["train"]["argv"][-1], text)

    def test_no_preparation_requires_explicit_artifacts(self):
        (self.policy / "process_data.sh").unlink()
        with self.assertRaisesRegex(ValueError, "prepared"):
            launch.plan(self.config, {})
        self.config["prepared"] = [str(self.workspace / "dataset/meta/info.json")]
        self.assertEqual(launch.plan(self.config, {})["prepare"], [])

    def test_global_batch_and_gpu_validation(self):
        for override in ({"gpus": "0,0"}, {"gpus": "0,1", "batch": "1", "batch_scope": "global"}):
            with self.assertRaises(ValueError):
                launch.plan(self.config, override)

    def test_failed_prepare_never_starts_training(self):
        job = launch.plan(self.config, {})
        with patch.object(launch, "run_stage", side_effect=RuntimeError("prepare failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                launch.execute(job, "all")
            self.assertEqual(run.call_count, 1)
        self.assertFalse((Path(job["output"]).parent / ".RoboDojo_real-demo-yam_dual-joint-0.launch-lock").exists())

    def test_nonempty_output_rejected_before_preparation(self):
        job = launch.plan(self.config, {})
        output = Path(job["output"])
        output.mkdir(parents=True)
        (output / "checkpoint").write_text("preserve")
        with patch.object(launch, "run_stage", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                launch.execute(job, "all")


if __name__ == "__main__":
    unittest.main()
