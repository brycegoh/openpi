"""Server-scoped Real-Time Chunking (RTC) guidance for async inference.

Provides a minimal interface for wrapping a flow-model denoising step with
prefix-guided inpainting. Compatible with PI0/PI05 PyTorch sampling loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class AsyncRTCConfig:
    """Configuration for async RTC guidance.

    Attributes:
        enabled: Whether RTC guidance is enabled.
        prefix_attention_schedule: Schedule for prefix attention weights (zeros|ones|linear|exp).
        max_guidance_weight: Maximum guidance weight. If None, uses num_flow_matching_steps.
        sigma_d: Prior variance for the guidance weight formula. Lower values give stronger
            guidance and smoother transitions.
        full_trajectory_alignment: If True, skip gradient computation and use error directly.
    """

    enabled: bool = False
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float | None = None
    sigma_d: float = 1.0
    full_trajectory_alignment: bool = False


class AsyncRTCProcessor:
    """RTC-style prefix guidance wrapper around an existing denoiser.

    Wraps a single denoising step with guidance from a previously-executed action prefix,
    blending frozen/soft/fresh regions via weighted inpainting.
    """

    def __init__(self, cfg: AsyncRTCConfig, *, postprocess: Callable[[Tensor], Tensor] | None = None):
        self.cfg = cfg
        self._postprocess = postprocess

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int | None,
        time: float | Tensor,
        original_denoise_step_partial: Callable[[Tensor], Tensor],
        overlap_end: int | None = None,
        num_flow_matching_steps: int | None = None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        Args:
            x_t: Current noisy sample tensor (B, T, A) or (T, A).
            prev_chunk_left_over: Previous chunk for inpainting guidance.
            inference_delay: Latency in action steps (d).
            time: Current denoising timestep (1 = noise, 0 = clean).
            original_denoise_step_partial: Computes base velocity given x_t.
            overlap_end: Where soft masking region ends.
            num_flow_matching_steps: Used as max_guidance_weight when not explicitly set.

        Returns:
            Guided velocity tensor.
        """
        if not self.cfg.enabled or prev_chunk_left_over is None or inference_delay is None:
            return original_denoise_step_partial(x_t)

        tau = 1 - time

        x_t_local = x_t.clone().detach()

        squeezed = False
        if x_t_local.ndim < 3:
            x_t_local = x_t_local.unsqueeze(0)
            squeezed = True

        prev = prev_chunk_left_over
        if prev.ndim < 3:
            prev = prev.unsqueeze(0)

        batch_size, chunk_t, chunk_a = x_t_local.shape
        prev_a = prev.shape[2]

        if overlap_end is None:
            overlap_end = chunk_t - inference_delay

        T_prefix = prev.shape[1]
        overlap_end = min(overlap_end, T_prefix)

        target_a = prev_a if self._postprocess is not None else chunk_a

        if prev.shape[1] < chunk_t:
            padded = torch.zeros(batch_size, chunk_t, target_a, device=x_t_local.device, dtype=x_t_local.dtype)
            padded[:, : prev.shape[1], :] = prev.to(device=x_t_local.device, dtype=x_t_local.dtype)
            prev = padded
        else:
            prev = prev[:, :chunk_t, :target_a].to(device=x_t_local.device, dtype=x_t_local.dtype)

        weights_1d = self._get_prefix_weights(inference_delay, overlap_end, chunk_t).to(x_t_local.device)
        weights = weights_1d.unsqueeze(0).unsqueeze(-1)

        # Enable gradients for the correction term without building a backward graph
        # through the denoiser/model parameters.
        with torch.enable_grad():
            with torch.no_grad():
                v_t = original_denoise_step_partial(x_t_local)

            x_t_local.requires_grad_(True)

            time_tensor = torch.as_tensor(time, device=x_t_local.device, dtype=x_t_local.dtype)
            x1_t = x_t_local - time_tensor * v_t.detach()

            x1_t_for_loss = x1_t
            if self._postprocess is not None:
                x1_t_for_loss = self._postprocess(x1_t_for_loss)

            err = (prev - x1_t_for_loss) * weights

            if self.cfg.full_trajectory_alignment:
                correction = err
            else:
                correction = torch.autograd.grad(x1_t_for_loss, x_t_local, err.detach(), retain_graph=False)[0]

        max_gw = self.cfg.max_guidance_weight
        if max_gw is None:
            max_gw = float(num_flow_matching_steps) if num_flow_matching_steps is not None else 10.0
        max_guidance_weight = torch.as_tensor(max_gw, device=x_t_local.device)

        tau_tensor = torch.as_tensor(tau, device=x_t_local.device, dtype=x_t_local.dtype)
        squared_one_minus_tau = (1 - tau_tensor) ** 2

        prior_variance = torch.as_tensor(self.cfg.sigma_d ** 2, device=x_t_local.device, dtype=x_t_local.dtype)
        inv_r2 = (squared_one_minus_tau + tau_tensor ** 2 * prior_variance) / (
            squared_one_minus_tau * prior_variance
        )

        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction
        if squeezed:
            result = result.squeeze(0)
        return result

    def _get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = int(min(start, end))
        end = int(end)
        total = int(total)
        schedule = (self.cfg.prefix_attention_schedule or "linear").lower()

        if schedule == "zeros":
            weights = torch.zeros(total)
            weights[: min(start, total)] = 1.0
            return weights
        if schedule == "ones":
            weights = torch.ones(total)
            weights[max(end, 0):] = 0.0
            return weights

        lin = self._linweights(start, end, total)
        if schedule == "exp":
            lin = lin * torch.expm1(lin).div(math.e - 1)

        weights = self._add_trailing_zeros(lin, total, end)
        weights = self._add_leading_ones(weights, start, total)
        return weights

    @staticmethod
    def _linweights(start: int, end: int, total: int) -> Tensor:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start
        if end <= start or linspace_steps <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    @staticmethod
    def _add_trailing_zeros(weights: Tensor, total: int, end: int) -> Tensor:
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        return torch.cat([weights, torch.zeros(zeros_len)])

    @staticmethod
    def _add_leading_ones(weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        return torch.cat([torch.ones(ones_len), weights])
