"""Latency estimation classes for async inference.

Provides abstract base class and implementations for estimating round-trip latency
in the DRTC algorithm.
"""

import math
from abc import ABC, abstractmethod
from collections import deque


class LatencyEstimatorBase(ABC):
    """Abstract base class for latency estimators.

    The estimate_steps property enforces the RTC constraint: d <= H/2
    where d is the inference delay and H is the prediction horizon (action_chunk_size).
    With s = d (maximum soft masking), the constraint d <= H - s becomes d <= H/2.
    """

    def __init__(
        self,
        fps: float,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        self._fps = fps
        self._action_chunk_size = action_chunk_size
        self._s_min = s_min

    @property
    def fps(self) -> float:
        return self._fps

    @abstractmethod
    def update(self, measured_rtt: float) -> None:
        """Update the latency estimate with a new RTT measurement."""
        ...

    @property
    @abstractmethod
    def estimate_seconds(self) -> float:
        """Get the latency estimate in seconds."""
        ...

    @property
    def estimate_steps(self) -> int:
        """Get the latency estimate quantized to action steps.

        Upper-bounded by H/2 per RTC constraint.
        """
        raw = max(1, math.ceil(self.estimate_seconds * self._fps))
        if self._action_chunk_size is not None:
            d_max = self._action_chunk_size // 2
            return min(raw, max(1, d_max))
        return raw

    @abstractmethod
    def reset(self) -> None:
        """Reset the estimator state."""
        ...


class JKLatencyEstimator(LatencyEstimatorBase):
    """Jacobson-Karels style latency estimator with exponential smoothing.

    Maintains a smoothed mean and deviation estimate of round-trip latency,
    combining them to produce a conservative estimate that adapts to variance.
    """

    def __init__(
        self,
        fps: float,
        alpha: float = 0.125,
        beta: float = 0.25,
        k: float = 1.0,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        super().__init__(fps, action_chunk_size, s_min=s_min)
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.smoothed_rtt: float = 0.0
        self.rtt_deviation: float = 0.0
        self._initialized: bool = False

    def update(self, measured_rtt: float) -> None:
        if not self._initialized:
            self.smoothed_rtt = measured_rtt
            self.rtt_deviation = 0
            self._initialized = True
            return

        error = measured_rtt - self.smoothed_rtt
        self.smoothed_rtt = (1 - self.alpha) * self.smoothed_rtt + self.alpha * measured_rtt
        self.rtt_deviation = (1 - self.beta) * self.rtt_deviation + self.beta * abs(error)

    @property
    def estimate_seconds(self) -> float:
        if not self._initialized:
            return self._s_min / self._fps
        return self.smoothed_rtt + self.k * self.rtt_deviation

    def reset(self) -> None:
        self.smoothed_rtt = 0.0
        self.rtt_deviation = 0.0
        self._initialized = False


class MaxLast10Estimator(LatencyEstimatorBase):
    """Conservative latency estimator using max of last 10 measurements."""

    def __init__(
        self,
        fps: float,
        window_size: int = 10,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        super().__init__(fps, action_chunk_size, s_min=s_min)
        self._window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, measured_rtt: float) -> None:
        self._buffer.append(measured_rtt)

    @property
    def estimate_seconds(self) -> float:
        if not self._buffer:
            return self._s_min / self._fps
        return max(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()


class FixedLatencyEstimator(LatencyEstimatorBase):
    """Fixed latency estimator for baseline comparisons."""

    def __init__(
        self,
        fps: float,
        fixed_latency_s: float = 0.1,
        action_chunk_size: int | None = None,
        s_min: int = 1,
    ):
        super().__init__(fps, action_chunk_size, s_min=s_min)
        self._fixed_latency_s = fixed_latency_s

    def update(self, measured_rtt: float) -> None:
        pass

    @property
    def estimate_seconds(self) -> float:
        return self._fixed_latency_s

    def reset(self) -> None:
        pass


def make_latency_estimator(
    kind: str,
    fps: float,
    alpha: float = 0.125,
    beta: float = 0.25,
    k: float = 1.0,
    fixed_latency_s: float = 0.1,
    action_chunk_size: int | None = None,
    s_min: int = 1,
) -> LatencyEstimatorBase:
    """Factory function to create a latency estimator.

    Args:
        kind: "jk" | "max_last_10" | "fixed"
        fps: Control loop frequency.
        alpha: JK smoothing factor for mean.
        beta: JK smoothing factor for deviation.
        k: JK scaling factor for deviation.
        fixed_latency_s: Fixed latency in seconds (only for kind="fixed").
        action_chunk_size: Prediction horizon H for upper bound clamping.
        s_min: Minimum execution horizon in steps.
    """
    if kind == "jk":
        return JKLatencyEstimator(
            fps=fps, alpha=alpha, beta=beta, k=k,
            action_chunk_size=action_chunk_size, s_min=s_min,
        )
    elif kind == "max_last_10":
        return MaxLast10Estimator(
            fps=fps, action_chunk_size=action_chunk_size, s_min=s_min,
        )
    elif kind == "fixed":
        return FixedLatencyEstimator(
            fps=fps, fixed_latency_s=fixed_latency_s,
            action_chunk_size=action_chunk_size, s_min=s_min,
        )
    else:
        raise ValueError(f"Unknown latency estimator type: {kind}. Use 'jk', 'max_last_10', or 'fixed'.")
