import enum
import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np

class RTCAttentionSchedule(str, enum.Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"

class RTCTracker:
    """Tracks RTC parameters across denoising steps for debugging and visualization."""

    def __init__(self, enabled: bool = False):
        """Initialize RTCTracker.

        Args:
            enabled: Whether to track RTC data. If False, all tracking operations are no-ops.
        """
        self.enabled = enabled
        self._tracking_history = None

    def reset(self):
        """Reset tracker for a new inference run."""
        self._tracking_history = None

    def set_tracking_history(self, tracking_history: dict):
        """Set tracking history from scan output.

        Args:
            tracking_history: Dictionary of JAX arrays from jax.lax.scan output
        """
        if not self.enabled:
            return

        # Convert JAX arrays to numpy for compatibility with visualization code
        self._tracking_history = {
            key: np.array(val) if val is not None else None
            for key, val in tracking_history.items()
        }

    def get_tracking_history(self) -> dict | None:
        """Get tracking history in a format compatible with visualization code.

        Returns:
            Dictionary with numpy arrays of tracked parameters, or None if tracking disabled.
        """
        if not self.enabled:
            return None
        return self._tracking_history

@dataclasses.dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = True
    debug: bool = False  # Enable debugging and detailed tracking of RTC parameters

    # Core RTC settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 5.0
    execution_horizon: int = 10


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(
        self,
        rtc_config: RTCConfig,
        verbose: bool = False,
        visualize_gradients: bool = False,
        viz_output_dir: str = ".",
    ):
        """Initialize RTC processor.

        Args:
            rtc_config (RTCConfig): Configuration holding RTC parameters used by
                the processor, including for example:
                - execution_horizon: number of timesteps used to build prefix weights
                - prefix_attention_schedule: strategy for prefix weights
                  (ZEROS, ONES, LINEAR, EXP)
                - max_guidance_weight: upper bound applied to the guidance scale
            verbose (bool): Enable verbose debug logging.
            visualize_gradients (bool): Enable gradient visualization using torchviz.
            viz_output_dir (str): Directory to save gradient visualizations.
        """
        self.rtc_config = rtc_config
        self.tracker = RTCTracker(enabled=rtc_config.debug if rtc_config else False)

    def rtc_enabled(self) -> bool:
        return self.rtc_config is not None and self.rtc_config.enabled

    def get_prefix_weights(self, start: int, end: int, total: int, schedule: RTCAttentionSchedule) -> jax.Array:
        """With start=2, end=6, total=10, the output will be:
        1  1  4/5 3/5 2/5 1/5 0  0  0  0
            ^              ^
            start           end
        `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
        paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
        entire prefix is attended to.

        `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
        if `end` is 0, then the entire prefix will always be ignored.
        """
        logging.info("=" * 80)
        logging.info(f"get_prefix_weights:")
        logging.info(f"start: {start}, end: {end}, total: {total}, schedule: {schedule}")
        logging.info("=" * 80)
        start = jnp.minimum(start, end)
        if schedule == RTCAttentionSchedule.ONES:
            w = jnp.ones(total)
        elif schedule == RTCAttentionSchedule.ZEROS:
            w = (jnp.arange(total) < start).astype(jnp.float32)
        elif schedule == RTCAttentionSchedule.LINEAR or schedule == RTCAttentionSchedule.EXP:
            # For positions < start: weight = 1
            # For positions >= start and < end: linearly decrease from 1 to 0
            # For positions >= end: weight = 0
            w = jnp.where(
                jnp.arange(total) < start,
                1.0,
                jnp.clip((end - jnp.arange(total)) / (end - start), 0, 1)
            )
            if schedule == RTCAttentionSchedule.EXP:
                w = w * jnp.expm1(w) / (jnp.e - 1)
        return jnp.where(jnp.arange(total) >= end, 0, w)