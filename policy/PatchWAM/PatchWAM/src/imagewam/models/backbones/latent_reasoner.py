"""LatentReasoner: Qwen3 as an in-graph reasoning generator for FLUX.2 conditioning.

Instead of using Qwen3 as a frozen prompt encoder (precomputed cache), this module
keeps Qwen3 (frozen base + LoRA) inside the training graph and produces, per sample:

  prompt_hidden [B, L, d_out]  -- layers (9, 18, 27) concat over the templated prompt,
                                  identical format to the FLUX.2 qwen3 cache, so the
                                  pretrained ``txt_in`` sees a familiar distribution;
  reason_hidden [B, K, d_out]  -- K extra "reasoning" representations appended after
                                  the prompt, produced by one of two modes:

    mode="thought":     K learnable thought embeddings are appended to the prompt and
                        everything runs in a single forward pass. Fully differentiable;
                        sequential compute depth is bounded by network depth.
    mode="latent_loop": Coconut-style continuous latent reasoning. After the prompt
                        forward (KV cache kept), the loop runs K steps; each step feeds
                        the previous step's last-layer hidden state back as the next
                        input embedding through a feedback projector. No token is ever
                        sampled, so the whole trajectory stays differentiable and the
                        step count is fixed by construction.

The feedback projector rescales hidden states (RMS >> embedding RMS for a frozen
model) back to embedding magnitude before an identity-initialized linear map; without
it the LoRA adapters would see inputs far outside the embedding distribution.

Trainable parameters: LoRA A/B on the Qwen linear suffixes, the thought / begin-of-
thought embeddings, and the feedback projector. The Qwen base stays frozen; checkpoints
store only the trainable subset (``trainable_state_dict``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class FeedbackProjector(nn.Module):
    """Map a last-layer hidden state back into input-embedding space.

    RMS-normalize (gain initialized to the embedding-table RMS so the output starts
    at embedding magnitude), then an identity-initialized linear map.
    """

    def __init__(self, hidden_size: int, embed_rms: float):
        super().__init__()
        self.gain = nn.Parameter(torch.full((hidden_size,), float(embed_rms)))
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        x = x * self.gain.float()
        return self.proj(x.to(self.proj.weight.dtype))


class LatentReasoner(nn.Module):
    def __init__(
        self,
        qwen_path: str,
        d_out: int = 7680,
        mode: str = "latent_loop",
        num_latent: int = 16,
        qwen_layers: Sequence[int] = (9, 18, 27),
        system_prompt: Optional[str] = None,
        max_prompt_len: int = 128,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_suffixes: Sequence[str] = DEFAULT_LORA_SUFFIXES,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if mode not in {"thought", "latent_loop"}:
            raise ValueError(f"Unsupported LatentReasoner mode: {mode!r}")
        self.mode = mode
        self.num_latent = int(num_latent)
        self.qwen_layers = tuple(int(l) for l in qwen_layers)
        self.d_out = int(d_out)
        self.system_prompt = system_prompt
        self.max_prompt_len = int(max_prompt_len)
        self.dim = self.d_out

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(qwen_path)
        self.tok.padding_side = "left"
        lm = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch_dtype)
        # Only the decoder stack is needed; drop lm_head to save memory.
        self.qwen = lm.model
        self.qwen.requires_grad_(False)

        hidden_size = int(self.qwen.config.hidden_size)
        if self.d_out != hidden_size * len(self.qwen_layers):
            raise ValueError(
                f"d_out={self.d_out} must equal hidden_size*len(qwen_layers)="
                f"{hidden_size * len(self.qwen_layers)}"
            )

        embed_weight = self.qwen.get_input_embeddings().weight
        embed_rms = float(embed_weight.detach().float().pow(2).mean().sqrt())
        embed_std = float(embed_weight.detach().float().std())

        if self.mode == "thought":
            self.thought_embeddings = nn.Parameter(
                torch.randn(self.num_latent, hidden_size) * embed_std
            )
        else:
            self.bot_embedding = nn.Parameter(torch.randn(1, hidden_size) * embed_std)
            self.feedback = FeedbackProjector(hidden_size, embed_rms=embed_rms)

        self.lora_rank = int(lora_rank)
        if self.lora_rank > 0:
            from .lora import apply_lora_to_linear_suffixes

            result = apply_lora_to_linear_suffixes(
                self.qwen,
                target_suffixes=tuple(lora_target_suffixes),
                rank=self.lora_rank,
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
            )
            if result.replaced == 0:
                raise ValueError(
                    f"LoRA matched no Linear layers for suffixes={tuple(lora_target_suffixes)}"
                )
        self.to(torch_dtype)
        self.set_trainable()
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "LatentReasoner(mode=%s, num_latent=%d, lora_rank=%d): %.1fM trainable params",
            self.mode,
            self.num_latent,
            self.lora_rank,
            n_trainable / 1e6,
        )

    def set_trainable(self) -> None:
        """Freeze the Qwen base; enable LoRA + thought/BoT embeddings + projector."""
        self.requires_grad_(False)
        for name, param in self.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                param.requires_grad = True
        if self.mode == "thought":
            self.thought_embeddings.requires_grad = True
        else:
            self.bot_embedding.requires_grad = True
            self.feedback.requires_grad_(True)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if k in trainable_names
        }

    def _tokenize(self, prompts: Sequence[str], device: torch.device):
        texts = []
        for prompt in prompts:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": str(self.system_prompt)})
            messages.append({"role": "user", "content": str(prompt)})
            try:
                text = self.tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_prompt_len,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        return (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)

    def _concat_layers(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([hidden_states[l] for l in self.qwen_layers], dim=-1)

    def _forward_thought(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        K = self.num_latent
        prompt_embeds = self.qwen.get_input_embeddings()(input_ids)
        thought = self.thought_embeddings.to(prompt_embeds.dtype)
        thought = thought.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prompt_embeds, thought], dim=1)
        full_mask = torch.cat(
            [attention_mask, attention_mask.new_ones(B, K)], dim=1
        )
        out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            position_ids=self._position_ids(full_mask),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = self._concat_layers(out.hidden_states)
        prompt_hidden, reason_hidden = hidden[:, :-K], hidden[:, -K:]
        return prompt_hidden, reason_hidden

    def _forward_latent_loop(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        prompt_out = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            output_hidden_states=True,
            use_cache=True,
        )
        prompt_hidden = self._concat_layers(prompt_out.hidden_states)
        past = prompt_out.past_key_values
        # Left padding => every sample's last physical position is a real token;
        # the next RoPE position for sample i is its real token count.
        next_pos = attention_mask.long().sum(dim=-1, keepdim=True)
        step_mask = attention_mask
        embeds = self.bot_embedding.to(prompt_hidden.dtype).unsqueeze(0).expand(B, 1, -1)
        reason_steps = []
        for step in range(self.num_latent):
            step_mask = torch.cat([step_mask, step_mask.new_ones(B, 1)], dim=1)
            out = self.qwen(
                inputs_embeds=embeds,
                attention_mask=step_mask,
                position_ids=next_pos + step,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
            past = out.past_key_values
            reason_steps.append(self._concat_layers(out.hidden_states)[:, -1])
            if step + 1 < self.num_latent:
                embeds = self.feedback(out.hidden_states[-1][:, -1]).unsqueeze(1)
        reason_hidden = torch.stack(reason_steps, dim=1)
        return prompt_hidden, reason_hidden

    def forward(self, prompts: Sequence[str] | str, device=None, dtype=None):
        """Encode prompts + reasoning; returns (text_hidden [B,L+K,d_out], mask [B,L+K])."""
        if isinstance(prompts, str):
            prompts = [prompts]
        target_device = device if device is not None else next(self.parameters()).device
        input_ids, attention_mask = self._tokenize(prompts, next(self.parameters()).device)
        if self.mode == "thought":
            prompt_hidden, reason_hidden = self._forward_thought(input_ids, attention_mask)
        else:
            prompt_hidden, reason_hidden = self._forward_latent_loop(input_ids, attention_mask)
        text_hidden = torch.cat([prompt_hidden, reason_hidden], dim=1)
        mask = torch.cat(
            [
                attention_mask.bool(),
                attention_mask.new_ones(
                    attention_mask.shape[0], reason_hidden.shape[1]
                ).bool(),
            ],
            dim=1,
        )
        if dtype is not None:
            text_hidden = text_hidden.to(dtype)
        return text_hidden.to(target_device), mask.to(target_device)
