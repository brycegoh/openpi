"""PyTorch implementation of Real-Time Chunking (RTC) processor.

RTC improves real-time inference by treating chunk generation as an inpainting problem,
strategically handling overlapping timesteps between action chunks using prefix attention
weights and VJP-based correction during the denoising process.

Reference implementation:
    https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
Sign convention verified against:
    https://github.com/huggingface/lerobot/issues/2511
"""

import enum
import dataclasses
import math

import torch


class RTCAttentionSchedule(str, enum.Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclasses.dataclass
class RTCConfig:
    enabled: bool = True
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 5.0
    execution_horizon: int = 10


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: RTCAttentionSchedule,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute prefix attention weights for RTC inpainting.

    Matches the kinetix reference:
    https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L40

    With start=2, end=6, total=10, the LINEAR schedule output will be:
        1  1  4/5 3/5 2/5 1/5 0  0  0  0
            ^              ^
            start           end

    `start` (inclusive) is where the chunk starts being allowed to change.
    `end` (exclusive) is where the chunk stops paying attention to the prefix.
    `end` takes precedence: if end < start, start is pushed down to end.
    """
    start = min(start, end)
    indices = torch.arange(total, device=device, dtype=torch.float32)

    if schedule == RTCAttentionSchedule.ONES:
        w = torch.ones(total, device=device)
    elif schedule == RTCAttentionSchedule.ZEROS:
        w = (indices < start).float()
    elif schedule in (RTCAttentionSchedule.LINEAR, RTCAttentionSchedule.EXP):
        # Matches kinetix: clip((start - 1 - i) / (end - start + 1) + 1, 0, 1)
        # which simplifies to: clip((end - i) / (end - start + 1), 0, 1)
        denom = max(end - start + 1, 1)
        w = torch.clamp((end - indices) / denom, 0, 1)
        if schedule == RTCAttentionSchedule.EXP:
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return torch.where(indices >= end, torch.zeros_like(w), w)
