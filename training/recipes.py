"""YAM recipes compose existing converters and native policy entry points."""

import copy

RECIPES = {
    "pi05-yam": "Pi_05", "pi05-yam-joint-ee": "Pi_05",
    "xr1-yam": "Xiaomi_Robotics_1", "lingbot-vla2-yam": "LingBot_VLA2",
    "gr00t-n17-yam": "GR00T_N17", "openwam-yam": "OpenWAM",
}


def step(entry, args, **kwargs):
    return {"entry": entry, "args": args, **kwargs}


def recipe(name):
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe {name!r}; choices: {sorted(RECIPES)}")
    job = {
        "policy": RECIPES[name],
        "params": {"data_root": "${workspace}/data/training/datasets/" + name,
                   "dataset": "${data_root}/${run}",
                   "output": "${workspace}/data/training/checkpoints/" + name + "/${run}",
                   "steps": "3000", "save_steps": "500", "batch": "1", "workers": "0"},
        "requires": ["${pretrained}"],
        "output": "${output}",
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "disabled", "WANDB_DISABLED": "true"},
        "prepare": [],
    }
    env, prepare = job["env"], job["prepare"]
    if name in {"pi05-yam", "pi05-yam-joint-ee", "lingbot-vla2-yam", "gr00t-n17-yam"}:
        args = ["${source}", "--repo-id", "${run}", "--output-root", "${dataset}",
                "--streaming-encoding", "--encoder-threads", "1"]
        if name == "pi05-yam-joint-ee":
            args.append("--include-ee-pose")
        prepare.append(step("${workspace}/scripts/datasets/convert_yam_to_lerobot.py", args,
                            requires=["${source}"],
                            produces=["${dataset}/meta/info.json"]))
    if name.startswith("pi05"):
        job["params"]["batch_scope"] = "global"
        config_name = "pi05_yam_joint_ee" if name.endswith("joint-ee") else "pi05_yam"
        job["params"]["assets"] = "${workspace}/data/training/assets/${run}"
        env.update(OPENPI_PYTHON="${python}", UV_PROJECT_ENVIRONMENT="${python_env}",
                   PYTHONPATH="${policy}/openpi/src:${policy}/openpi/packages/openpi-client/src",
                   OPENPI_TRAIN_CONFIG_NAME=config_name, OPENPI_LEROBOT_REPO_ID="${run}",
                   HF_LEROBOT_HOME="${data_root}", OPENPI_BASE_PARAMS="${pretrained}",
                   OPENPI_ASSETS_BASE_DIR="${assets}", OPENPI_CHECKPOINT_DIR="${output}",
                   OPENPI_NUM_TRAIN_STEPS="${steps}", OPENPI_SAVE_INTERVAL="${save_steps}",
                   OPENPI_BATCH_SIZE="${batch}", OPENPI_NUM_WORKERS="${workers}",
                   OPENPI_FSDP_DEVICES="${gpu_count}", OPENPI_RESUME="0")
        prepare.append(step("openpi/scripts/compute_norm_stats.py",
                            ["--config-name", config_name, "--num-workers", "${workers}"],
                            cwd="openpi"))
        job["prepared"] = ["${dataset}/meta/info.json", "${assets}/" + config_name + "/${run}/norm_stats.json"]
    elif name == "xr1-yam":
        job["params"].update(action="ee")
        prepare.append(step("prepare_yam.py", ["--source", "${source}", "--dataset", "${dataset}",
                                               "--name", "${run}", "--instruction", "${instruction}",
                                               "--batch", "${batch}"], requires=["${source}"]))
        env.update(OUTPUT_DIR="${dataset}", DATA_CONFIG_NAME="${run}",
                   PRETRAINED_PATH="${pretrained}", RUN_ROOT="${output}",
                   MAX_STEPS="${steps}", SAVE_INTERVAL="${save_steps}",
                   XR1_QWEN_VL_CONFIG_SOURCE="${processor}", XR1_LOGGER="csv")
        job["requires"].append("${processor}/tokenizer.json")
        job["prepared"] = ["${dataset}/manifest.json", "${dataset}/norm_stats.json",
                           "${policy}/xiaomi_robotics_1/xr1/configs/data/${run}.yaml"]
    elif name == "lingbot-vla2-yam":
        env.update(LINGBOT_VLA2_ENV_DIR="${python_env}", LINGBOT_VLA2_DATASET_PATH="${dataset}",
                   PYTHONPATH="${policy}/lingbot_vla_v2",
                   LINGBOT_VLA2_NORM_STATS_PATH="${dataset}/norm_stats.json",
                   LINGBOT_VLA2_MODEL_PATH="${pretrained}", LINGBOT_VLA2_TOKENIZER_PATH="${processor}",
                   LINGBOT_VLA2_CHECKPOINT_DIR="${output}", LINGBOT_VLA2_MAX_STEPS="${steps}",
                   LINGBOT_VLA2_SAVE_STEPS="${save_steps}", LINGBOT_VLA2_TRAIN_WORKERS="${workers}",
                   LINGBOT_VLA2_MICRO_BATCH_SIZE="${batch}", LINGBOT_VLA2_ENABLE_RESUME="false")
        prepare.append(step("lingbot_vla_v2/train.sh", ["scripts/compute_norm_stats.py",
            "configs/vla/norm_compute/post_data.yaml", "--data.data_name", "yam_dual_absolute",
            "--data.robot_name", "yam_dual_absolute", "--data.train_path", "${dataset}",
            "--data.robot_config_root", "${policy}/robot_configs", "--data.norm_path", "${dataset}/norm_stats.json",
            "--data.num_workers", "${workers}", "--train.chunk_size", "50",
            "--train.micro_batch_size", "${batch}", "--train.output_dir", "${dataset}/norm-compute"],
            cwd="lingbot_vla_v2"))
        job["requires"].append("${processor}/tokenizer.json")
        job["prepared"] = ["${dataset}/meta/info.json", "${dataset}/norm_stats.json"]
    elif name == "gr00t-n17-yam":
        job["params"]["batch_scope"] = "global"
        env.update(GR00T_ENV_DIR="${python_env}", GR00T_CONVERT_ENV_DIR="${converter_env}",
                   GR00T_ALLOW_INSTALL="0", GR00T_LEROBOT_HOME="${data_root}", GR00T_SRC_DATASET="${run}",
                   GR00T_BASE_MODEL="${pretrained}", GR00T_COSMOS_MODEL="${cosmos}",
                   GR00T_CHECKPOINT_DIR="${output}", MAX_STEPS="${steps}", SAVE_STEPS="${save_steps}",
                   GLOBAL_BATCH_SIZE="${batch}", NUM_GPUS="${gpu_count}",
                   DATALOADER_NUM_WORKERS="${workers}", USE_WANDB="0")
        job["requires"].extend(["${cosmos}/config.json", "${converter_env}/bin/python",
                                "${converter_env}/bin/ffmpeg"])
        prepare.append(step("process_data.sh", ["${bench}", "${run}", "${robot}", "${action}"]))
        job["prepared"] = ["${data_root}/${bench}-${run}-${robot}-${action}/meta/stats.json",
                           "${data_root}/${bench}-${run}-${robot}-${action}/meta/modality.json"]
    elif name == "openwam-yam":
        job["params"].update(action="ee")
        env.update(OPENWAM_PYTHON="${python}", OPENWAM_DATASET_DIR="${dataset}",
                   OPENWAM_FINETUNE_CKPT_PATH="${pretrained}", OPENWAM_CHECKPOINT_DIR="${output}")
        prepare.extend([
            step("${workspace}/scripts/datasets/prepare_openwam_yam_dataset.py",
                 ["--episodes", "${source}", "--output", "${dataset}", "--task", "${run}",
                  "--instruction", "${instruction}", "--frequency", "30"], requires=["${source}"]),
            step("process_data.sh", ["${bench}", "${run}", "${robot}", "${action}"]),
        ])
        job["prepared"] = ["${dataset}/meta/openwam_yam_manifest.json",
                           "${dataset}/meta/robodojo_real_yam_dual_normalization_stats.npy"]
        job["train"] = {"args": ["${bench}", "${run}", "${robot}", "${action}", "${seed}", "${gpus}",
                                  "training.max_steps=${steps}", "training.num_epochs=null",
                                  "training.save_steps=${save_steps}", "training.batch_size=${batch}",
                                  "training.dataset_num_workers=${workers}", "project.wandb.project=null"]}
    return copy.deepcopy(job)
