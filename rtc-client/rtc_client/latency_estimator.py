"""Jacobson-Karels latency estimator for RTC inference delay.

Estimates round-trip inference latency and converts it to action steps
for use as the `inference_delay` (hard mask region) in RTC prefix weights.

Ported from the DRTC implementation:
    https://github.com/jackvial/drtc
"""

import math


class JKLatencyEstimator:
    """Jacobson-Karels style latency estimator with exponential smoothing.

    Maintains a smoothed mean and deviation estimate of round-trip latency,
    combining them to produce a conservative estimate that adapts to variance.

    The estimator takes wall-clock seconds as input and converts to action steps
    at the output via ``ceil(estimate_seconds * control_hz / interpolation_factor)``,
    clamped to ``action_horizon // 2`` per the RTC constraint (d <= H/2).

    When an ``InterpolatedActionWrapper`` stretches each raw action into multiple
    control ticks, ``interpolation_factor`` corrects the conversion so that
    ``estimate_steps`` reflects raw queue actions consumed, not control ticks.

    Args:
        control_hz: Control loop frequency for quantizing to action steps.
        action_horizon: Prediction horizon H (number of actions per chunk).
        interpolation_factor: Ratio of control ticks per raw action (>= 1.0).
        alpha: Smoothing factor for mean (default 0.125 per RFC 6298).
        beta: Smoothing factor for deviation (default 0.25 per RFC 6298).
        k: Scaling factor for deviation in estimate.
    """

    def __init__(
        self,
        control_hz: float,
        action_horizon: int,
        interpolation_factor: float = 1.0,
        alpha: float = 0.125,
        beta: float = 0.25,
        k: float = 1.5,
    ):
        self._control_hz = control_hz
        self._action_horizon = action_horizon
        self._interpolation_factor = interpolation_factor
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self._smoothed_rtt: float = 0.0
        self._rtt_deviation: float = 0.0
        self._initialized: bool = False

    def update(self, measured_rtt_seconds: float) -> None:
        """Update the latency estimate with a new RTT measurement in seconds."""
        if not self._initialized:
            self._smoothed_rtt = measured_rtt_seconds
            self._rtt_deviation = 0.0
            self._initialized = True
            return

        error = measured_rtt_seconds - self._smoothed_rtt
        self._smoothed_rtt = (1 - self.alpha) * self._smoothed_rtt + self.alpha * measured_rtt_seconds
        self._rtt_deviation = (1 - self.beta) * self._rtt_deviation + self.beta * abs(error)

    @property
    def estimate_seconds(self) -> float:
        """Smoothed latency estimate in seconds: l_hat = l_bar + K * sigma."""
        if not self._initialized:
            return 0.0
        return self._smoothed_rtt + self.k * self._rtt_deviation

    @property
    def estimate_steps(self) -> int:
        """Latency estimate quantized to action steps, clamped to action_horizon // 2."""
        if not self._initialized:
            return 0
        raw = max(1, math.ceil(self.estimate_seconds * self._control_hz / self._interpolation_factor))
        d_max = max(1, self._action_horizon // 2)
        return min(raw, d_max)

    def reset(self) -> None:
        """Reset the estimator state."""
        self._smoothed_rtt = 0.0
        self._rtt_deviation = 0.0
        self._initialized = False
