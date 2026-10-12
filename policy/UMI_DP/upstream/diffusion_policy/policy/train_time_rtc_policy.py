"""CalibWrist trained-prefix policy; the legacy UMI sampler stays unchanged.

Constructor, prefix conditioning and masked loss follow the recorded CalibWrist
source in UPSTREAM.md. Reuse its original conditional_sample and normalizer API.
"""


import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
)
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.model.diffusion.train_time_rtc_unet import (
    TrainTimeRtcConditionalUnet1D,
)
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.model.vision.timm_obs_encoder import (
    TimmObsEncoder,
)
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.policy.base_image_policy import (
    BaseImagePolicy,
)
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.policy.diffusion_unet_timm_policy import (
    DiffusionUnetTimmPolicy,
)


class TrainTimeRtcDiffusionUnetTimmPolicy(DiffusionUnetTimmPolicy):
    """Separate trained network construction; no RtcGuidance wrapper."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: TimmObsEncoder,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        input_pertub=0.1,
        inpaint_fixed_action_prefix=False,
        train_diffusion_n_samples=1,
        rtc=None,
        # parameters passed to step
        **kwargs,
    ):
        BaseImagePolicy.__init__(self)

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta["action"]["horizon"]
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        rtc = rtc or {}
        self.rtc_enabled = bool(rtc.get("enabled", False))
        self.rtc_simulated_delay = int(rtc.get("simulated_delay", 0))
        if self.rtc_enabled and not 1 < self.rtc_simulated_delay <= action_horizon:
            raise ValueError("RTC simulated_delay must be in [2, action_horizon]")
        delay_weights = rtc.get("delay_weights")
        self.rtc_delay_weights = (
            None if delay_weights is None else tuple(float(w) for w in delay_weights)
        )
        if self.rtc_delay_weights is not None and (
            len(self.rtc_delay_weights) != self.rtc_simulated_delay
            or any(not np.isfinite(w) or w < 0 for w in self.rtc_delay_weights)
            or sum(self.rtc_delay_weights) <= 0
        ):
            raise ValueError(
                "RTC delay_weights must be finite, nonnegative, and match simulated_delay"
            )

        model = TrainTimeRtcConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=1 if self.rtc_enabled else None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon  # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        if self.rtc_enabled and not inpaint_fixed_action_prefix:
            raise ValueError("RTC requires inpaint_fixed_action_prefix=True")
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def predict_action(
        self, obs_dict: dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(
            size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            if (
                fixed_action_prefix.ndim != 3
                or fixed_action_prefix.shape[0] != B
                or fixed_action_prefix.shape[2] != self.action_dim
                or not 0 <= n_fixed_steps < self.action_horizon
            ):
                raise ValueError("fixed_action_prefix must have shape (B, prefix<T, action_dim)")
            cond_data[:, :n_fixed_steps] = self.normalizer["action"].normalize(fixed_action_prefix)
            cond_mask[:, :n_fixed_steps] = True

        local_cond = cond_mask[:, :, :1].to(cond_data.dtype) if self.rtc_enabled else None

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        result = {"action": action_pred, "action_pred": action_pred}
        return result

    def compute_loss(self, batch, return_statistics=False):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(
                global_cond, repeats=self.train_diffusion_n_samples, dim=0
            )
            nactions = torch.repeat_interleave(
                nactions, repeats=self.train_diffusion_n_samples, dim=0
            )

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device
        )

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],),
            device=trajectory.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise_new, timesteps)

        local_cond = None
        loss_mask = None
        if self.rtc_enabled:
            # Prefix actions are clean; the suffix retains the sampled diffusion timestep.
            if self.rtc_delay_weights is None:
                weights = torch.arange(
                    self.rtc_simulated_delay - 1,
                    -1,
                    -1,
                    device=trajectory.device,
                    dtype=torch.float32,
                ).softmax(0)
            else:
                weights = torch.tensor(
                    self.rtc_delay_weights, device=trajectory.device, dtype=torch.float32
                )
            delays = torch.multinomial(weights, nactions.shape[0], replacement=True)
            prefix_mask = (
                torch.arange(self.action_horizon, device=trajectory.device)[None] < delays[:, None]
            )
            noisy_trajectory = torch.where(prefix_mask[:, :, None], trajectory, noisy_trajectory)
            local_cond = prefix_mask[:, :, None].to(trajectory.dtype)
            loss_mask = (~prefix_mask)[:, :, None]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        if loss_mask is None:
            loss_sum = loss.sum()
            loss_count = torch.tensor(loss.numel(), device=loss.device)
        else:
            loss_sum = (loss * loss_mask).sum()
            loss_count = loss_mask.sum() * self.action_dim

        loss = loss_sum / loss_count
        if return_statistics:
            return loss, loss_sum, loss_count
        return loss

    def compute_loss_statistics(self, batch):
        return self.compute_loss(batch, return_statistics=True)
