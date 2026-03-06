"""Training-Time Action Conditioning (TTAC) for efficient Real-Time Chunking.

Reference: https://arxiv.org/abs/2512.05964

At training time, TTAC simulates inference delay by:
1. Sampling a delay d (number of prefix actions from the start of the chunk)
2. Setting prefix action timesteps to 0.0 (fully denoised / ground truth)
3. Masking loss to only train on postfix (non-prefix) actions

At inference time, prefix positions in x_t are replaced with previously
committed actions and their timesteps are set to 0.0.

OpenPI flow matching convention:
  x_t = t * noise + (1 - t) * actions
  t=0 → clean actions, t=1 → pure noise
  Denoising proceeds from t=1 to t=0.
"""

import dataclasses
import enum
from collections.abc import Sequence

import torch


class TTACDelayDistribution(str, enum.Enum):
    UNIFORM = "UNIFORM"
    EXP = "EXP"


@dataclasses.dataclass
class TTACConfig:
    """Configuration for Training-Time Action Conditioning."""

    enabled: bool = False
    min_delay: int = 0
    max_delay: int = 6
    delay_distribution: TTACDelayDistribution = TTACDelayDistribution.UNIFORM
    exp_decay: float = 1.0

    def __post_init__(self):
        if self.min_delay < 0:
            raise ValueError(f"min_delay must be >= 0, got {self.min_delay}")
        if self.max_delay < self.min_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= min_delay ({self.min_delay})"
            )
        if self.exp_decay <= 0:
            raise ValueError(f"exp_decay must be positive, got {self.exp_decay}")


def sample_ttac_delay(
    config: TTACConfig,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample a delay per batch element.

    Returns:
        Long tensor of shape (batch_size,) with values in [min_delay, max_delay].
    """
    if config.max_delay == config.min_delay:
        return torch.full((batch_size,), config.min_delay, device=device, dtype=torch.long)

    if config.delay_distribution == TTACDelayDistribution.UNIFORM:
        return torch.randint(
            config.min_delay, config.max_delay + 1, (batch_size,), device=device, dtype=torch.long
        )

    if config.delay_distribution == TTACDelayDistribution.EXP:
        delay_values = torch.arange(
            config.min_delay, config.max_delay + 1, device=device, dtype=torch.long
        )
        weights = torch.exp(-config.exp_decay * delay_values.to(dtype=torch.float32))
        probs = weights / weights.sum()
        samples = torch.multinomial(probs, batch_size, replacement=True)
        return delay_values[samples]

    raise ValueError(f"Unknown delay distribution: {config.delay_distribution}")


def apply_ttac_training(
    time: torch.Tensor,
    delay: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create per-token timesteps and postfix mask for TTAC training.

    Prefix positions (indices < delay) get t=0.0 (ground truth / clean).
    Postfix positions (indices >= delay) keep the original sampled time.

    Args:
        time: Sampled timestep per batch element, shape (B,).
        delay: Delay per batch element, shape (B,).
        seq_len: Total action sequence length (action_horizon).

    Returns:
        time_tokens: Per-token timesteps, shape (B, seq_len).
        postfix_mask: Boolean mask, True for postfix positions, shape (B, seq_len).
    """
    device = time.device
    delay = torch.clamp(delay, max=seq_len)
    prefix_mask = torch.arange(seq_len, device=device)[None, :] < delay[:, None]
    time_tokens = time[:, None].expand(-1, seq_len).clone()
    time_tokens = time_tokens.masked_fill(prefix_mask, 0.0)
    postfix_mask = ~prefix_mask
    return time_tokens, postfix_mask


def masked_mean(
    losses: torch.Tensor,
    mask: torch.Tensor | None,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """Compute masked mean over losses, restricting to postfix positions.

    Matches the kinetix reference: normalizes by the number of valid *positions*
    (not elements), so for (B, T, D) losses with (B, T) mask, the denominator
    counts valid (b, t) positions rather than valid (b, t, d) elements.

    Args:
        losses: Loss tensor, shape (B, T, D) or (B, T).
        mask: Boolean mask, shape (B, T). True for positions to include.
            If None, computes unmasked mean over reduce_dims (or all dims).
        reduce_dims: Dimensions to reduce over. If None, reduces all dims.

    Returns:
        Mean of masked losses, reduced over the specified dimensions.
    """
    if mask is None:
        if reduce_dims is None:
            return losses.mean()
        return losses.mean(dim=reduce_dims)

    # Keep mask as (B, T) or (B, T, 1) — do NOT expand to (B, T, D) so the
    # denominator counts positions, matching the kinetix reference:
    #   loss_mask = ~mask[:, :, None]   # (B, T, 1)
    #   sum(loss * loss_mask) / (sum(loss_mask) + eps)
    if losses.ndim == 3 and mask.ndim == 2:
        mask_for_loss = mask.unsqueeze(-1)  # (B, T, 1), broadcasts to (B, T, D)
    else:
        mask_for_loss = mask

    mask_f = mask_for_loss.float()
    masked_losses = losses * mask_f

    if reduce_dims is None:
        return masked_losses.sum() / mask_f.sum().clamp(min=1e-8)

    return masked_losses.sum(dim=reduce_dims) / mask_f.sum(dim=reduce_dims).clamp(min=1e-8)


def apply_ttac_inference(
    x_t: torch.Tensor,
    time: float,
    inference_delay: int,
    prev_chunk_leftover: torch.Tensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply TTAC at inference time.

    Replaces prefix positions in x_t with previous chunk's leftover actions
    and creates per-token timesteps with 0.0 for prefix positions.

    Args:
        x_t: Current noisy actions, shape (B, T, D).
        time: Current denoising timestep (scalar float).
        inference_delay: Number of prefix positions to condition on.
        prev_chunk_leftover: Previous chunk's remaining actions, shape (B, >=delay, D).
        chunk_size: Total action chunk size.

    Returns:
        x_t_conditioned: Conditioned noisy actions, shape (B, T, D).
        time_tokens: Per-token timesteps, shape (B, T).
    """
    batch_size = x_t.shape[0]
    device = x_t.device

    if prev_chunk_leftover is None or inference_delay <= 0:
        time_tokens = torch.full(
            (batch_size, chunk_size), time, dtype=torch.float32, device=device
        )
        return x_t, time_tokens

    x_t_conditioned = x_t.clone()
    d = min(inference_delay, prev_chunk_leftover.shape[1], chunk_size)
    x_t_conditioned[:, :d] = prev_chunk_leftover[:, :d]

    time_tokens = torch.full(
        (batch_size, chunk_size), time, dtype=torch.float32, device=device
    )
    time_tokens[:, :d] = 0.0

    return x_t_conditioned, time_tokens
