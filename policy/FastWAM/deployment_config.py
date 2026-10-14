"""Use checkpoint training settings when constructing a FastWAM policy."""

from pathlib import Path

from omegaconf import OmegaConf


def use_training_config(sim_config, training_config_path):
    """Keep evaluation options, replacing model/data with the saved contract."""
    saved = OmegaConf.to_container(
        OmegaConf.load(Path(training_config_path).expanduser()), resolve=True
    )
    model = saved["model"]
    # This training harness only audits initialization around the native factory.
    if model["_target_"] == "model_factory.create_fastwam":
        model["_target_"] = "fastwam.runtime.create_fastwam"
    if model["_target_"] != "fastwam.runtime.create_fastwam":
        raise ValueError("Saved config must use the native FastWAM factory.")
    # Keep the verified inference path eager; an enabled training compilation
    # switch needs a separately validated deployment recipe.
    if model.get("compile_training_denoise", False):
        raise ValueError("Inference deployment expects compile_training_denoise=false.")
    for key in ("train_shift", "infer_shift", "num_train_timesteps"):
        if key not in model["action_scheduler"]:
            raise ValueError(f"Saved action scheduler is missing {key}.")
    processor = saved["data"]["train"]["processor"]
    action_dim = int(processor["action_output_dim"])
    state_dim = int(processor["proprio_output_dim"])
    if int(model["action_dit_config"]["action_dim"]) != action_dim:
        raise ValueError("Saved model and processor action dimensions disagree.")
    if int(model["proprio_dim"]) != state_dim:
        raise ValueError("Saved model and processor state dimensions disagree.")
    # The full finetuned checkpoint supplies these parameters. Match the native
    # sim recipe: do not reopen a training-only ActionDiT initializer on QZ.
    model["action_dit_pretrained_path"] = None
    model["skip_dit_load_from_pretrain"] = True
    model["load_text_encoder"] = True
    # Preserve unrelated lazy Hydra evaluation paths; they may require a live
    # run context that is deliberately absent during offline configuration.
    result = OmegaConf.create(OmegaConf.to_container(sim_config, resolve=False))
    result.model = model
    result.data = saved["data"]
    if "mixed_precision" in saved:
        result.mixed_precision = saved["mixed_precision"]
    return result
