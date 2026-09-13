"""Plan or run data preparation and native training, without importing models."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

XPL = Path(__file__).resolve().parents[1]
WORKSPACE = XPL.parent
NONSTANDARD = {"EventVLA", "Hy_Embodied_05_VLA"}
SOURCE_MARKERS = {
    "Pi_05": "openpi/scripts/train.py",
    "Xiaomi_Robotics_1": "xiaomi_robotics_1/xr1/tools/train.py",
    "LingBot_VLA2": "lingbot_vla_v2/tasks/vla/train_lingbotvla.py",
    "GR00T_N17": "gr00t_n17/gr00t/experiment/launch_finetune.py",
    "OpenWAM": "OpenWAM/scripts/train.py",
    "MolmoACT2": "molmoact2/lerobot/pyproject.toml",
}


def inventory():
    """Presence describes an entry point, not GPU/embodiment validation."""
    result = []
    for folder in sorted((XPL / "policy").iterdir()):
        if not folder.is_dir():
            continue
        train = folder / "train.sh"
        stub = folder.name == "demo_policy"
        result.append({
            "policy": folder.name,
            "training": "template" if stub else "native_entry" if train.is_file() else "no_entry",
            "prepare": "process_data.sh" if (folder / "process_data.sh").is_file() else "prepared_data_required",
            "arguments": "explicit" if folder.name in NONSTANDARD else "xpolicy_six",
            "source": ("present" if (folder / SOURCE_MARKERS[folder.name]).is_file() else "missing")
                      if folder.name in SOURCE_MARKERS else "not_audited",
        })
    return result


def expand(value, variables):
    if isinstance(value, str):
        # Only named placeholders are interpreted; JSON/Hydra braces remain literal.
        def replace(match):
            key = match[1]
            if key not in variables or variables[key] is None:
                raise ValueError(f"Missing parameter: {key} (supply --set {key}=...)")
            return str(variables[key])
        return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", replace, value)
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def plan(config, overrides):
    unknown = set(config) - {"policy", "params", "env", "prepare", "train", "requires", "prepared", "output", "log_dir"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    policy_name = config.get("policy")
    available = {item["policy"]: item for item in inventory()}
    if policy_name not in available or available[policy_name]["training"] != "native_entry":
        raise ValueError(f"No implemented native training entry for {policy_name!r}")
    policy = XPL / "policy" / policy_name
    if available[policy_name]["source"] == "missing":
        raise ValueError(f"{policy_name} training source is missing: {SOURCE_MARKERS[policy_name]}")
    variables = {"bench": "RoboDojo_real", "robot": "yam_dual", "action": "joint",
                 "seed": "0", "gpus": "0", "python": sys.executable,
                 **config.get("params", {}), **overrides}
    variables.update(workspace=str(WORKSPACE), xpolicy=str(XPL), policy=str(policy))
    # Resolve dependent params such as dataset_root -> dataset without evaluating shell code.
    for _ in range(len(variables) + 1):
        resolved = expand(variables, variables)
        if resolved == variables:
            break
        variables = resolved
    if any(isinstance(value, str) and "${" in value for value in variables.values()):
        raise ValueError("Cyclic or unresolved parameter references")
    run = variables.get("run")
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run):
        raise ValueError("run is required and must be a simple directory name")
    gpu_ids = str(variables["gpus"]).split(",")
    if any(not part.isdigit() for part in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpus must be unique comma-separated GPU indices")
    if int(variables["seed"]) < 0:
        raise ValueError("seed must be nonnegative")
    for key in ("batch", "steps", "save_steps"):
        if key in variables and int(variables[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if variables.get("batch_scope") == "global" and int(variables["batch"]) % len(gpu_ids):
        raise ValueError("Global batch must be divisible by the number of GPUs")
    interpreter = shutil.which(str(variables["python"]))
    if interpreter is None:
        raise ValueError(f"Existing interpreter not found: {variables['python']}")
    # Do not resolve the venv symlink to its system executable.
    interpreter = str(Path(interpreter).absolute())
    variables.update(python=interpreter, python_bin=str(Path(interpreter).parent),
                     python_env=str(Path(interpreter).parent.parent), gpu_count=str(len(gpu_ids)))
    settings = expand(config, variables)
    common = [str(variables[key]) for key in ("bench", "run", "robot", "action", "seed", "gpus")]
    train = settings.get("train", {})
    if policy_name in NONSTANDARD and "args" not in train:
        raise ValueError(f"{policy_name} needs explicit train.args; see its policy README")
    train = {"entry": "train.sh", "args": common, **train}
    prepare = settings.get("prepare")
    if prepare is None:
        if (policy / "process_data.sh").is_file():
            prepare = [{"entry": "process_data.sh", "args": common[:4]}]
        else:
            prepare = []
    if not isinstance(prepare, list):
        raise ValueError("prepare must be a list of stages")
    if not prepare and not settings.get("prepared"):
        raise ValueError("No preparation stage: declare prepared artifact paths explicitly")
    def stage(spec):
        if not isinstance(spec, dict) or set(spec) - {"entry", "args", "cwd", "requires", "produces"}:
            raise ValueError("Stage keys: entry, args, cwd, requires, produces")
        entry = Path(spec["entry"])
        if not entry.is_absolute():
            entry = policy / entry
        if not entry.is_file():
            raise ValueError(f"Stage entry missing: {entry}")
        arguments = spec.get("args", [])
        if not isinstance(arguments, list) or not all(isinstance(arg, str) for arg in arguments):
            raise ValueError("Stage args must be a string array, never a shell command")
        command = (["bash", str(entry)] if entry.suffix == ".sh"
                   else [interpreter, str(entry)] if entry.suffix == ".py" else None)
        if command is None:
            raise ValueError("Stage entry must be a .sh or .py file")
        cwd = Path(spec.get("cwd", str(policy)))
        if not cwd.is_absolute():
            cwd = policy / cwd
        if not cwd.is_dir():
            raise ValueError(f"Stage working directory missing: {cwd}")
        return {"argv": command + arguments, "cwd": str(cwd),
                "requires": spec.get("requires", []), "produces": spec.get("produces", [])}
    env = settings.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
        raise ValueError("env must map names to strings")
    output = settings.get("output", str(policy / "checkpoints" / "-".join(common[:5])))
    return {"policy": policy_name, "run": run, "python": interpreter,
            "prepare": [stage(spec) for spec in prepare], "train": stage(train),
            "requires": [interpreter, *settings.get("requires", [])],
            "prepared": settings.get("prepared", []), "output": output,
            "log_dir": settings.get("log_dir", str(WORKSPACE / "data/training" / policy_name / run)),
            "env": {"CUDA_VISIBLE_DEVICES": str(variables["gpus"]),
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MAX_JOBS": "1", **env}}


def require_paths(paths):
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"Artifact paths must be absolute: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")


def run_stage(spec, env, log):
    require_paths(spec["requires"])
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(spec['argv'])}\n")
        stream.flush()
        child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait()
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise
    if code:
        raise RuntimeError(f"Stage failed with exit code {code}; see {log}")
    require_paths(spec["produces"])


def execute(job, phase):
    require_paths(job["requires"])
    output = Path(job["output"])
    if not output.is_absolute():
        raise ValueError("output must be an absolute path")
    # Native scripts may use --overwrite: guard before any expensive stage.
    if phase != "prepare" and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing nonempty training output: {output}; choose a new run")
    logs = Path(job["log_dir"])
    if not logs.is_absolute():
        raise ValueError("log_dir must be absolute")
    logs.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".launch-lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir()  # Fail rather than run concurrently against the same output.
    env = dict(os.environ, **job["env"])
    env["PATH"] = str(Path(job["python"]).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE), str(XPL), str(WORKSPACE / "src"), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    try:
        if phase != "train":
            for index, spec in enumerate(job["prepare"]):
                print(f"Preparing stage {index + 1}; log: {logs / f'prepare-{index + 1}.log'}", flush=True)
                run_stage(spec, env, logs / f"prepare-{index + 1}.log")
        require_paths(job["prepared"])
        if phase != "prepare":
            print(f"Training; log: {logs / 'train.log'}", flush=True)
            run_stage(job["train"], env, logs / "train.log")
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--recipe")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--phase", choices=["all", "prepare", "train"], default="all")
    parser.add_argument("--execute", action="store_true", help="Opt in to data processing and GPU training")
    args = parser.parse_args()
    if args.list:
        from XPolicyLab.training.recipes import RECIPES
        print(json.dumps({"recipes": sorted(RECIPES), "policies": inventory()}, indent=2))
        return
    if args.recipe:
        from XPolicyLab.training.recipes import recipe
        config = recipe(args.recipe)
    elif args.config:
        config = json.loads(args.config.read_text())
    else:
        parser.error("Choose --recipe, --config or --list")
    overrides = {}
    for assignment in args.set:
        key, separator, value = assignment.partition("=")
        if not separator:
            parser.error("--set requires KEY=VALUE")
        overrides[key] = value
    job = plan(config, overrides)
    # Never persist or print environment values: credentials belong in env.
    display = {**job, "env": sorted(job["env"]), "phase": args.phase, "execute": args.execute}
    print(json.dumps(display, indent=2), flush=True)
    if args.execute:
        execute(job, args.phase)


if __name__ == "__main__":
    main()
