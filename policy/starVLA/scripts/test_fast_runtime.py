"""Keep invalid FAST generations from masquerading as successful zero actions."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "source_starvla"))
    return import_module("starVLA.model.modules.action_model.fast_ActionHeader")


@pytest.mark.parametrize("tokens,length", [([], 56), ([None], 56), ([[]], 56), ([[1]], 55)])
def test_invalid_tokens_never_reach_zero_fallback(module, tokens, length):
    def fallback(_):
        pytest.fail("invalid tokens reached the upstream zero fallback")

    holder = SimpleNamespace(
        fast_tokenizer=SimpleNamespace(
            time_horizon=8,
            action_dim=7,
            bpe_tokenizer=SimpleNamespace(decode=lambda _: "a" * length),
            decode=fallback,
        )
    )
    with pytest.raises(ValueError, match="FAST"):
        module.Fast_Action_Tokenizer.decode_action_tokens(holder, tokens)


def test_valid_tokens_keep_upstream_decoding(module):
    expected = np.arange(56).reshape(1, 8, 7)
    holder = SimpleNamespace(
        fast_tokenizer=SimpleNamespace(
            time_horizon=8,
            action_dim=7,
            bpe_tokenizer=SimpleNamespace(decode=lambda _: "a" * 56),
            decode=lambda _: expected,
        )
    )
    assert module.Fast_Action_Tokenizer.decode_action_tokens(holder, [[1, 2]]) is expected


def test_tokenizer_asset_path_is_configurable(module, monkeypatch):
    monkeypatch.setattr(
        module, "Fast_Action_Tokenizer", lambda fast_tokenizer_name: fast_tokenizer_name
    )
    config = SimpleNamespace(
        framework=SimpleNamespace(action_model={"fast_tokenizer_path": "/assets/from-config"})
    )
    monkeypatch.delenv("STARVLA_FAST_TOKENIZER", raising=False)
    assert module.get_action_model(config) == "/assets/from-config"
    monkeypatch.setenv("STARVLA_FAST_TOKENIZER", "/assets/from-local-binding")
    assert module.get_action_model(config) == "/assets/from-local-binding"


def test_missing_tokenizer_path_is_not_replaced_with_a_developer_path(module, monkeypatch):
    monkeypatch.delenv("STARVLA_FAST_TOKENIZER", raising=False)
    with pytest.raises(ValueError, match="STARVLA_FAST_TOKENIZER"):
        module.get_action_model(SimpleNamespace(framework=SimpleNamespace(action_model={})))


@pytest.mark.parametrize("do_sample", [False, True])
def test_generation_honors_requested_sampling(module, monkeypatch, do_sample):
    framework = import_module("starVLA.model.framework.VLM4A.QwenFast")
    monkeypatch.setattr(framework, "to_pil_preserve", lambda images: images)
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return "generated"

    holder = SimpleNamespace(
        qwen_vl_interface=SimpleNamespace(
            build_qwenvl_inputs=lambda **_: {},
            model=SimpleNamespace(generate=generate),
        ),
        action_model=SimpleNamespace(decode_action_tokens=lambda _: np.ones((1, 8, 7))),
        _extract_action_token_ids=lambda _: [[1]],
        _decode_action_tokens=lambda value: value,
    )
    framework.Qwenvl_Fast.predict_action(
        holder,
        examples=[{"image": [], "lang": "test"}],
        do_sample=do_sample,
    )
    assert captured["do_sample"] is do_sample
