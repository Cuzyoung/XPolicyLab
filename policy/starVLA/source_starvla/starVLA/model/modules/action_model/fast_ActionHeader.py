# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action → language model-readable sequence" converter (Fast_Action_Tokenizer).
    Its core objective is to convert continuous/discrete raw robot actions (raw_actions) into
    pseudo-natural language token strings like <robot_action_12><robot_action_3><robot_action_87> ...
    This facilitates direct integration into multimodal large models (VLM/LLM) dialogue templates,
    leveraging their language modeling capabilities for action prediction.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch.nn as nn
from transformers import AutoProcessor, PreTrainedTokenizerFast


def _load_fast_processor(pretrained_path: str = "physical-intelligence/fast"):
    """Load the published FAST class and tokenizer through one explicit path.

    Constructing the processor directly works across Transformers versions and
    preserves asset/configuration errors instead of retrying after any exception.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(pretrained_path).expanduser()
    if not local_dir.is_dir():
        local_dir = Path(snapshot_download(pretrained_path))
    spec = importlib.util.spec_from_file_location(
        "processing_action_tokenizer",
        os.path.join(local_dir, "processing_action_tokenizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    UniversalActionProcessor = mod.UniversalActionProcessor

    bpe_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(local_dir, "tokenizer.json"),
        clean_up_tokenization_spaces=False,
    )

    with open(os.path.join(local_dir, "processor_config.json"), "r") as f:
        cfg = json.load(f)

    processor = UniversalActionProcessor(
        bpe_tokenizer=bpe_tokenizer,
        scale=cfg["scale"],
        vocab_size=cfg["vocab_size"],
        min_token=cfg["min_token"],
        action_dim=cfg.get("action_dim"),
        time_horizon=cfg.get("time_horizon"),
    )
    return processor


class Fast_Action_Tokenizer(nn.Module):
    """FAST processor with strict validation of generated action coefficients."""

    def __init__(self, fast_tokenizer_name):
        super().__init__()

        self.fast_tokenizer = _load_fast_processor(fast_tokenizer_name)

    def decode_action_tokens(self, tokens):
        """Reject malformed generations before the upstream decoder's zero fallback."""
        if not tokens:
            raise ValueError("FAST returned an empty action batch")
        expected = self.fast_tokenizer.time_horizon * self.fast_tokenizer.action_dim
        for index, sequence in enumerate(tokens):
            if not sequence:
                raise ValueError(f"FAST sample {index} generated no action tokens")
            decoded = self.fast_tokenizer.bpe_tokenizer.decode(sequence)
            if len(decoded) != expected:
                raise ValueError(f"FAST sample {index} has {len(decoded)} decoded coefficients; expected {expected}")
        return self.fast_tokenizer.decode(tokens)

    def encoder_action2fastoken(self, raw_actions):
        # x: (batch_size, chunck, dim)
        batch_actions = np.stack(raw_actions, axis=0)  # (B, T, D)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)

        return batch_fast_tokens  # List[str]

    def decoder_action(self, generated_ids):
        # api https://huggingface.co/physical-intelligence/fast
        # return: (batch_size, chunck, dim)
        pred_actions = self.fast_tokenizer.decode([generated_ids - self._ACTION_TOKEN_MIN])
        return pred_actions

    def fit_tokenizer_on_datasets(
        self,
        action_dataset,
        datasets_path="<your_local_path>",
    ):
        # If datasets_path exists, load directly
        if os.path.exists(datasets_path):
            self.fast_tokenizer = AutoProcessor.from_pretrained(datasets_path, trust_remote_code=True)
            return
        else:
            # If not found, Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """Build the FAST processor from the deployment binding or checkpoint config."""
    action_cfg = config.framework.action_model if config is not None else {}
    tokenizer_path = os.environ.get("STARVLA_FAST_TOKENIZER") or action_cfg.get("fast_tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Set STARVLA_FAST_TOKENIZER or framework.action_model.fast_tokenizer_path")
    action_model = Fast_Action_Tokenizer(fast_tokenizer_name=tokenizer_path)

    return action_model


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        start_debugpy_once()

    fast_tokenizer_name = "physical-intelligence/fast"
    fast_tokenizer = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)
    raw_actions = [np.random.randn(16, 7), np.random.randn(16, 7)]

    tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_name, trust_remote_code=True)

    action_data = np.random.rand(2, 16, 7)
    tokens = tokenizer(action_data)
    decoded_actions = tokenizer.decode(tokens)

    # self func test
    vlm_tokens = fast_tokenizer.encoder_action2fastoken(raw_actions)
    print(vlm_tokens)
    pred_actions = fast_tokenizer.decoder_action(np.array([12, 3, 45, 87]))
    print(pred_actions)
