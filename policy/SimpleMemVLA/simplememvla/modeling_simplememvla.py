import torch
from torch import nn
from transformers import PreTrainedModel

from .configuration_simplememvla import SimpleMemVLAConfig
from .dit_action_head import DiTActionHead


class SimpleMemVLAForActionPrediction(PreTrainedModel):

    config_class = SimpleMemVLAConfig
    base_model_prefix = "simplememvla"

    def __init__(self, config: SimpleMemVLAConfig, backbone: nn.Module):
        super().__init__(config)
        self.backbone = backbone
        if config.action_dim is None or config.action_horizon is None:
            raise ValueError("SimpleMemVLAConfig.action_dim and action_horizon must be set")
        if config.use_proprio and config.state_dim is None:
            raise ValueError("SimpleMemVLAConfig.state_dim must be set when use_proprio=True")
        config.hidden_size = self.backbone.config.text_config.hidden_size
        config.backbone_config = self.backbone.config.to_dict()
        rope_parameters = self.backbone.config.text_config.rope_parameters
        if config.dit_rope_theta is None:
            config.dit_rope_theta = rope_parameters.get("rope_theta")
        if config.dit_mrope_section is None:
            config.dit_mrope_section = rope_parameters.get("mrope_section")
        if config.dit_partial_rotary_factor is None:
            config.dit_partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        self.action_head = DiTActionHead(
            vlm_hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            hidden_size=config.dit_hidden_size,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            rope_theta=config.dit_rope_theta,
            mrope_section=config.dit_mrope_section,
            partial_rotary_factor=config.dit_partial_rotary_factor,
            use_proprio=config.use_proprio,
            state_dim=config.state_dim,
        )
        for module in self.modules():
            module._is_hf_initialized = True
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_kwargs: dict,
    ) -> torch.Tensor:
        position_ids, _ = self.backbone.model.get_rope_index(
            input_ids,
            image_grid_thw=mm_kwargs.get("image_grid_thw"),
            video_grid_thw=mm_kwargs.get("video_grid_thw"),
            attention_mask=attention_mask,
            mm_token_type_ids=mm_kwargs.get("mm_token_type_ids"),
        )
        return position_ids

    def _gather_condition(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        span: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        span = span.to(device=hidden_states.device, dtype=torch.bool)
        lengths = span.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        if max_len == 0:
            raise ValueError("sub-task condition span is empty")

        batch_size, _, hdim = hidden_states.shape
        device = hidden_states.device
        embed = self.get_input_embeddings()
        cond_hidden = hidden_states.new_zeros(batch_size, max_len, hdim)
        cond_token = hidden_states.new_zeros(batch_size, max_len, hdim)
        cond_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=device)
        cond_pos = position_ids.new_zeros(3, batch_size, max_len)
        for b in range(batch_size):
            idx = span[b].nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            if n == 0:
                continue
            cond_hidden[b, :n] = hidden_states[b, idx]
            cond_token[b, :n] = embed(input_ids[b, idx])
            cond_mask[b, :n] = 1
            cond_pos[:, b, :n] = position_ids[:, b, idx]
        return cond_hidden, cond_token, cond_mask, cond_pos

    @torch.no_grad()
    def predict_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        condition_span: torch.Tensor,
        num_steps: int = 10,
        temperature: float = 1.0,
        state: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        position_ids = self._position_ids(input_ids, attention_mask, kwargs)
        mm_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        hidden_states = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **mm_kwargs,
        ).last_hidden_state
        cond_hidden, cond_token, cond_mask, cond_pos = self._gather_condition(
            hidden_states=hidden_states,
            input_ids=input_ids,
            span=condition_span,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return self.action_head.sample(
            cond_hidden,
            condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask,
            condition_token_embeds=cond_token,
            num_steps=num_steps,
            temperature=temperature,
            state=state,
        )

    @torch.no_grad()
    def generate_subtask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
        **gen_kwargs,
    ) -> torch.LongTensor:
        gen_inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if eos_token_id is not None:
            gen_inputs["eos_token_id"] = eos_token_id
        gen_inputs.update(gen_kwargs)
        return self.backbone.generate(**gen_inputs)
