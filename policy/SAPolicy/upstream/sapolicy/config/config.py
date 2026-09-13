from typing import List

from numpy import isin
from sapolicy.config import yacs
from sapolicy.config.yacs import CfgNode as CN
from sapolicy.logger import Log
import argparse
import os
from ast import literal_eval
from os.path import join
import yaml


def define_basic_cfg():
    cfg = CN()
    cfg.print_cfg = False
    cfg.seed = 42
    cfg.resume_training = False
    # When True, after fit(ckpt_path=...) realign LR scheduler to the *new*
    # max_epochs/num_training_steps at the resumed global_step (dreamer4-style).
    cfg.resume_reset_scheduler = False
    cfg.confirm_delete_previous_dir = False
    return cfg


def parse_cfg(cfg, args):
    cfg.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    cfg.entry = args.entry
    cfg.exp_name = cfg.exp_name.replace(
        "FILENAME", os.path.basename(args.cfg_file).split(".")[0]
    )
    cfg.exp_name = cfg.exp_name.replace(
        "GITBRANCH", os.popen("git describe --all").readline().strip()[6:]
    )
    cfg.exp_name = cfg.exp_name.replace(
        "GITCOMMIT", os.popen("git describe --tags --always").readline().strip()
    )
    cfg.exp_name = cfg.exp_name.replace(
        "TODAY", os.popen("date +%Y-%m-%d").readline().strip()
    )

    if cfg.local_rank == 0:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open("/tmp/HTCODE_TIMESTAMP.txt", "w") as f:
            f.write(timestamp)
    else:
        import time

        while not os.path.exists("/tmp/HTCODE_TIMESTAMP.txt"):
            time.sleep(1)
        with open("/tmp/HTCODE_TIMESTAMP.txt", "r") as f:
            timestamp = f.read().strip()

    cfg.exp_name = cfg.exp_name.replace("DEBUG", timestamp)
    return cfg


def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def process_cfg_name(cfg, global_cfg):
    """
    处理cfg中的${}引用
    """

    def is_ref_value(value):
        return isinstance(value, str) and "${" in value

    def process_value(value):
        if is_ref_value(value):
            splits_head = value.split("${")
            for idx, split_head in enumerate(splits_head[1:]):
                ref_key = split_head.split("}")[0]
                if ref_key.startswith("env:"):
                    env_key = ref_key[4:]
                    if ":" in env_key:
                        default_val = env_key.split(":")[1]
                        env_key = env_key.split(":")[0]
                    else:
                        default_val = None
                    value = value.replace(
                        "${" + ref_key + "}", os.getenv(env_key, default_val)
                    )
                elif is_ref_value(global_cfg[ref_key]):
                    global_cfg[ref_key] = process_cfg_name(
                        global_cfg[ref_key], global_cfg
                    )
                    value = value.replace(
                        "${" + ref_key + "}", str(global_cfg[ref_key])
                    )
                else:
                    value = value.replace(
                        "${" + ref_key + "}", str(global_cfg[ref_key])
                    )
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            value = value.split(",")
            value[0] = value[0][1:]
            value[-1] = value[-1][:-1]
            # 去除掉value的空格
            value = [v.strip() for v in value]
            # 如果可以是int， 则转为int
            # 如果可以是float， 则转为float
            # 否则保持str
            value = [
                (
                    int(v)
                    if v.isdigit()
                    else float(v) if v.replace(".", "", 1).isdigit() else v
                )
                for v in value
            ]
            # return value
        # if '{HT_ENV:' in value:
        #     splits = value.split('{HT_ENV:')
        #     for split in splits:
        #         ref_key = split.split('}')[0]
        #         if ':' in ref_key:
        #             ref_key, default_val = ref_key.split(':')
        #             env_val = os.getenv(ref_key, default_val)
        #         else:
        #             env_val = os.getenv(ref_key)
        #         value = value.replace('{HT_ENV:' + ref_key + '}', env_val)
        return value

    if isinstance(cfg, int) or isinstance(cfg, float):
        return cfg

    if isinstance(cfg, str):
        return process_value(cfg)

    if isinstance(cfg, list) or isinstance(cfg, List):
        return [process_cfg_name(item, global_cfg) for item in cfg]

    if isinstance(cfg, CN) or isinstance(cfg, dict):
        for key, value in cfg.items():
            cfg[key] = process_cfg_name(value, global_cfg)
        return cfg

    raise ValueError(f"Unsupported type: {type(cfg)}")
    # for key, value in cfg.items():
    #     if isinstance(value, CN) or isinstance(value, dict):
    #         cfg[key] = process_cfg_name(value, global_cfg)
    #     elif isinstance(value, list) or isinstance(value, List):
    #         cfg[key] = [process_value(item, global_cfg) for item in value]
    #     elif is_ref_value(value):
    #         cfg[key] = process_value(value, global_cfg)
    # return cfg


def process_cfg_file(cfg):
    """
    处理cfg中的CFG:, HT_LIST:, HT_ENV:引用
    """

    def process_value(value):
        if isinstance(value, str) and value.startswith("CFG:"):
            return merge_cfg(value.split(":")[-1], CN())
        else:
            return value

    if isinstance(cfg, int) or isinstance(cfg, float):
        return cfg

    if isinstance(cfg, str):
        return process_value(cfg)

    if isinstance(cfg, list) or isinstance(cfg, List):
        return [process_cfg_file(item) for item in cfg]

    if isinstance(cfg, CN) or isinstance(cfg, dict):
        for key, value in cfg.items():
            cfg[key] = process_cfg_file(value)
        return cfg

    raise ValueError(f"Unsupported type: {type(cfg)}")


def merge_cfg(cfg_file, cfg, loaded_cfg_files=None):
    # 递归往前调用，同时记录顺序到 loaded_cfg_files
    # 先检查value里面有没有CFG:, 将file load为字典赋值给对应的key
    # 先检查value里面有没有HT_LIST:, 将file load为list赋值给对应key
    # 检查value里面有没有HT_ENV:TESTING:default_val, 将env读进来，赋值给对应的key
    # 每次会优先检测parent_cfg
    # 然后检查configs
    try:
        with open(cfg_file, "r") as f:
            current_cfg = yacs.load_cfg(f)
    except Exception as e:
        raise ValueError(f"Error loading {cfg_file}: {e}")

    # 如果一个文件循环调用自己，就需要停止并且报错
    if "parent_cfg" in current_cfg.keys():
        if current_cfg.parent_cfg == cfg_file:
            raise ValueError(f"Circular reference detected in {cfg_file}")
        cfg = merge_cfg(current_cfg.parent_cfg, cfg, loaded_cfg_files)
    if "include" in current_cfg.keys():
        if cfg_file in current_cfg.include:
            raise ValueError(f"Circular reference detected in {cfg_file}")
        for cfg_file_ in current_cfg.include:
            cfg = merge_cfg(cfg_file_, cfg, loaded_cfg_files)

    current_cfg = process_cfg_file(current_cfg)

    try:
        cfg.merge_from_other_cfg(current_cfg)
    except:
        raise ValueError(
            f"Error merging {cfg_file} with {loaded_cfg_files}: \n"
            + f"\n----cfg----\n{cfg}"
            + f"\n----current_cfg----\n{current_cfg}"
        )
    if loaded_cfg_files is not None:
        loaded_cfg_files.append(cfg_file)
    return cfg


def merge_from_opts(cfg, opts) -> CN:
    for opt in opts:
        if opt in ("", "-"):
            continue
        if "=" not in opt:
            raise ValueError(
                f"Invalid config override {opt!r}: expected key=value "
                f"(e.g. confirm_delete_previous_dir=true). "
                f"To merge another yaml use --include path/to.yaml, not a bare path."
            )
        key, value = opt.split("=", 1)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        cfg[key] = value
        process_cfg_file(cfg)
    return cfg


def make_cfg(args):
    if args.cfg_file[:8] == "configs/":  # 默认的cfg文件在conf
        cfg = define_basic_cfg()
        loaded_cfg_files = []
        cfg_ = merge_cfg(args.cfg_file, cfg, loaded_cfg_files)
        if args.include is not None:
            for config in args.include:
                merge_cfg(config, cfg_, loaded_cfg_files)
        cfg_ = merge_from_opts(cfg_, args.opts)
        cfg_ = parse_cfg(cfg_, args)  # 处理一些自定义pattern
        cfg_ = process_cfg_file(cfg_)  # 处理config file
        cfg_ = process_cfg_name(cfg_, cfg_)  # 处理环境变量
    else:
        with open(args.cfg_file, "r") as f:
            cfg_ = yacs.load_cfg(f)
        cfg_ = merge_from_opts(cfg_, args.opts)
        cfg_ = parse_cfg(cfg_, args)
        cfg_ = process_cfg_file(cfg_)
        cfg_ = process_cfg_name(cfg_, cfg_)
    return cfg_


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument("--entry", type=str, default="train_net")
parser.add_argument("--include", type=str, action="append", default=[])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = make_cfg(args)
