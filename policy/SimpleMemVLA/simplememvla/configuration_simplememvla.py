from transformers import PretrainedConfig


class SimpleMemVLAConfig(PretrainedConfig):

    model_type = "simplememvla"

    def __init__(
        self,
        hidden_size: int | None = None,
        action_dim: int | None = None,
        action_horizon: int | None = None,
        use_proprio: bool = True,
        state_dim: int | None = None,
        dit_hidden_size: int = 2048,
        dit_depth: int = 16,
        dit_num_heads: int = 16,
        dit_mlp_ratio: float = 3.5,
        dit_rope_theta: float | None = None,
        dit_mrope_section: list[int] | tuple[int, int, int] | None = None,
        dit_partial_rotary_factor: float | None = None,
        robot_tag: str | None = None,
        control_frequency_hz: int | None = None,
        history_video_sec: float = 60.0,
        history_video_fps: float = 2.0,
        native_video_fps: float | None = None,
        variable_history: bool = False,
        image_keys: list[str] | None = None,
        history_image_keys: list[str] | None = None,
        backbone_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_proprio = use_proprio
        self.state_dim = state_dim
        self.dit_hidden_size = dit_hidden_size
        self.dit_depth = dit_depth
        self.dit_num_heads = dit_num_heads
        self.dit_mlp_ratio = dit_mlp_ratio
        self.dit_rope_theta = dit_rope_theta
        self.dit_mrope_section = (
            list(dit_mrope_section) if dit_mrope_section is not None else None
        )
        self.dit_partial_rotary_factor = dit_partial_rotary_factor
        self.robot_tag = robot_tag
        self.control_frequency_hz = control_frequency_hz
        self.history_video_sec = history_video_sec
        self.history_video_fps = history_video_fps
        self.native_video_fps = native_video_fps
        self.variable_history = variable_history
        self.image_keys = list(image_keys) if image_keys is not None else None
        self.history_image_keys = (
            list(history_image_keys) if history_image_keys is not None else None
        )
        self.backbone_config = backbone_config
