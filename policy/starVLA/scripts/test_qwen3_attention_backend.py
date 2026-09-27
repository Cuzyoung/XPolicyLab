"""Regression coverage for the checkpoint-selected Qwen3 attention backend."""

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("backend", ["flash_attention_2", "sdpa"])
def test_checkpoint_attention_backend_reaches_model_loader(monkeypatch, backend):
    source = Path(__file__).resolve().parents[1] / "source_starvla"
    monkeypatch.syspath_prepend(str(source))
    module = import_module("starVLA.model.modules.vlm.QWen3")
    captured = {}

    def load_model(model_id, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=2048),
            )
        )

    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    monkeypatch.setattr(module.Qwen3VLForConditionalGeneration, "from_pretrained", load_model)
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda _: SimpleNamespace(
            tokenizer=SimpleNamespace(padding_side=None),
        ),
    )
    config = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {
                    "base_vlm": "test-qwen3",
                    "attn_implementation": backend,
                }
            }
        }
    )
    module._QWen3_VL_Interface(config=config)
    assert captured["attn_implementation"] == backend
