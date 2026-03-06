"""Real-Time Chunking (RTC) inference manager for continuous action execution.

This module provides an async inference client that decouples action execution
from model inference. It uses a threshold-based mechanism to trigger new
inference calls before the current action queue is exhausted, enabling either:
  - Reduced idle time when inference is slower than action execution
  - Continuous actions when inference is faster than action execution

Usage:
    manager = RTCInferenceManager(
        policy=WebsocketClientPolicy(host, port),
        get_observation_fn=robot.get_observation,
        action_horizon=50,
    )
    manager.start()
    while running:
        action = manager.get_action()
        if action is not None:
            robot.execute(action)
        time.sleep(1.0 / control_hz)
    manager.stop()
"""

import logging
import time
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

from rtc_client import image_tools as _image_tools
from rtc_client.base_policy import BasePolicy
from rtc_client.latency_estimator import JKLatencyEstimator

logger = logging.getLogger(__name__)


class RTCActionQueue:
    """Thread-safe action queue that stores both raw and processed actions.

    Raw actions (pre-output-transform) are kept for RTC inpainting feedback
    to the server. Processed actions are used for robot execution.
    """

    def __init__(self):
        self._processed_actions: Optional[np.ndarray] = None
        self._raw_actions: Optional[np.ndarray] = None
        self._index: int = 0
        self._lock = threading.Lock()

    def push(
        self,
        processed_actions: np.ndarray,
        raw_actions: Optional[np.ndarray] = None,
    ) -> None:
        """Replace queue contents with a new action chunk.

        For RTC mode, skips actions that were already executed during inference
        by only keeping actions from the current index forward in the old queue,
        then replacing with the new chunk (offset by the number already consumed).

        Args:
            processed_actions: Post-processed actions for execution, shape (H, D).
            raw_actions: Raw model outputs for RTC feedback, shape (H, D).
        """
        with self._lock:
            self._processed_actions = processed_actions
            self._raw_actions = raw_actions
            self._index = 0

    def push_rtc(
        self,
        processed_actions: np.ndarray,
        raw_actions: Optional[np.ndarray] = None,
        actions_consumed_during_inference: int = 0,
    ) -> None:
        """Push new RTC-corrected actions, accounting for actions executed during inference.

        The new chunk from the model already accounts for the overlap via inpainting.
        We skip the first `actions_consumed_during_inference` actions since those
        timesteps have already been executed.
        """
        with self._lock:
            skip = min(actions_consumed_during_inference, len(processed_actions) - 1)
            self._processed_actions = processed_actions[skip:]
            self._raw_actions = raw_actions[skip:] if raw_actions is not None else None
            self._index = 0

    def get(self) -> Optional[np.ndarray]:
        """Get the next action to execute. Returns None if queue is empty."""
        with self._lock:
            if self._processed_actions is None or self._index >= len(self._processed_actions):
                return None
            action = self._processed_actions[self._index]
            self._index += 1
            return action

    def remaining(self) -> int:
        """Number of actions remaining in the queue."""
        with self._lock:
            if self._processed_actions is None:
                return 0
            return max(0, len(self._processed_actions) - self._index)

    def total(self) -> int:
        """Total number of actions in the current chunk."""
        with self._lock:
            if self._processed_actions is None:
                return 0
            return len(self._processed_actions)

    def get_raw_leftover(self) -> Optional[np.ndarray]:
        """Get remaining raw (pre-transform) actions for RTC feedback."""
        with self._lock:
            if self._raw_actions is None:
                return None
            remaining = self._raw_actions[self._index:]
            if len(remaining) == 0:
                return None
            return remaining.copy()

    def get_index(self) -> int:
        with self._lock:
            return self._index

    def empty(self) -> bool:
        return self.remaining() <= 0


class RTCInferenceManager:
    """Manages async inference with threshold-based pre-fetching for continuous action execution.

    The manager runs an inference thread that monitors the action queue. When the
    number of remaining actions drops below a configurable threshold (percentage of
    action horizon), a new inference call is triggered with the current observation.

    For RTC-enabled inference, the leftover raw actions from the previous chunk are
    sent to the server so it can use inpainting to produce temporally consistent actions.

    All non-image observation fields returned by ``get_observation_fn`` are
    forwarded verbatim to the server.  Callers can inject server-side overrides
    (e.g. ``model_action_horizon``) directly into the observation dict without
    needing a dedicated manager API.

    Args:
        policy: A policy client (e.g., WebsocketClientPolicy) that implements infer().
        get_observation_fn: Callable that returns the current observation dict.
        action_horizon: The number of actions per inference chunk.
        refill_threshold: Fraction of action_horizon below which to trigger inference (0-1).
            Default 0.25 means inference starts when 25% of actions remain.
        rtc_enabled: Whether to use RTC inpainting (sends prev actions to server).
        rtc_config: RTC configuration dict sent to server. Keys:
            - enabled (bool): Enable RTC on server side.
            - execution_horizon (int): Number of overlap timesteps for inpainting.
            - prefix_attention_schedule (str): Weight schedule ("LINEAR", "EXP", etc.).
            - max_guidance_weight (float | None): Clipping parameter (beta) for guidance
              weight. None = auto (num_steps). Original RTC paper uses 5.0.
            - sigma_d (float): Prior data std dev for guidance. Original RTC paper
              assumes 1.0. Soare optimization uses < 1 (e.g. 0.2) for smoother transitions.
    """

    def __init__(
        self,
        policy: BasePolicy,
        get_observation_fn: Callable[[], Dict],
        action_horizon: int,
        refill_threshold: float = 0.25,
        rtc_enabled: bool = True,
        rtc_config: Optional[Dict] = None,
        control_hz: float = 50.0,
        interpolation_factor: float = 1.0,
    ):
        self._policy = policy
        self._get_observation = get_observation_fn
        self._action_horizon = action_horizon
        self._refill_threshold = refill_threshold
        self._rtc_enabled = rtc_enabled
        self._rtc_config = rtc_config or {
            "enabled": True,
            "execution_horizon": 10,
            "prefix_attention_schedule": "LINEAR",
            # Soare optimization defaults: beta=num_steps (None=auto), sigma_d=0.2
            # Set max_guidance_weight=5.0 and sigma_d=1.0 for original RTC paper behavior.
            "max_guidance_weight": None,
            "sigma_d": 0.2,
        }

        self._action_queue = RTCActionQueue()
        self._shutdown_event = threading.Event()
        self._first_inference_done = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._cold_start_time = 0.0
        self._latency_estimator = JKLatencyEstimator(
            control_hz=control_hz,
            action_horizon=action_horizon,
            interpolation_factor=interpolation_factor,
        )

    @property
    def action_queue(self) -> RTCActionQueue:
        return self._action_queue

    def start(self, wait_for_first_action: bool = True) -> None:
        """Start the background inference thread.

        Args:
            wait_for_first_action: If True, block until the first inference completes
                so the action queue is populated before returning.
        """
        if self._inference_thread is not None and self._inference_thread.is_alive():
            logger.warning("Inference thread already running")
            return

        self._shutdown_event.clear()
        self._first_inference_done.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="RTCInference",
        )
        self._inference_thread.start()
        logger.info(
            "Started RTC inference manager (horizon=%d, threshold=%.0f%%, rtc=%s)",
            self._action_horizon,
            self._refill_threshold * 100,
            self._rtc_enabled,
        )

        if wait_for_first_action:
            logger.info("Waiting for first inference to complete...")
            self._first_inference_done.wait()
            logger.info(
                "First inference (cold start) complete in %.1fms, action queue has %d actions",
                self._cold_start_time * 1000,
                self._action_queue.remaining(),
            )

    def stop(self) -> None:
        """Stop the inference thread and wait for it to finish."""
        self._shutdown_event.set()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=10.0)
            self._inference_thread = None

        self._first_inference_done.clear()

        if self._cold_start_time > 0:
            logger.info("Cold start inference: %.1fms (excluded from stats)", self._cold_start_time * 1000)
        if self._inference_count > 0:
            avg_ms = (self._total_inference_time / self._inference_count) * 1000
            logger.info(
                "Stopped RTC inference manager. %d inferences (excl. cold start), avg %.1fms",
                self._inference_count,
                avg_ms,
            )

    def get_action(self) -> Optional[np.ndarray]:
        """Get the next action from the queue. Returns None if no action available."""
        return self._action_queue.get()

    @staticmethod
    def _is_image_array(arr: np.ndarray) -> bool:
        if arr.ndim != 3:
            return False
        h, w, c = arr.shape
        if c in (1, 3, 4) and h > 4 and w > 4:
            return True  # HWC
        if h in (1, 3, 4) and w > 4 and c > 4:
            return True  # CHW
        return False

    @staticmethod
    def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
        """Transpose CHW to HWC if needed. resize_with_pad expects HWC."""
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            return np.transpose(arr, (1, 2, 0))
        return arr

    def _preprocess_images(self, obs: dict) -> dict:
        """Resize all detected image arrays in *obs* to 224x224 and convert to uint8."""
        out: dict = {}
        for key, value in obs.items():
            if isinstance(value, dict):
                out[key] = self._preprocess_images(value)
            elif isinstance(value, np.ndarray) and self._is_image_array(value):
                img = self._ensure_hwc(value)
                img = _image_tools.convert_to_uint8(img)
                img = _image_tools.resize_with_pad(img, 224, 224)
                out[key] = img
            else:
                out[key] = value
        return out

    def _inference_loop(self) -> None:
        """Background loop that triggers inference when action queue runs low."""
        threshold_count = max(1, int(self._action_horizon * self._refill_threshold))
        logger.info("Inference loop started, threshold=%d actions", threshold_count)

        while not self._shutdown_event.is_set():
            remaining = self._action_queue.remaining()

            if remaining > threshold_count:
                time.sleep(0.005)
                continue

            try:
                success = self._run_inference()
                if success and not self._first_inference_done.is_set():
                    self._first_inference_done.set()
            except Exception:
                logger.exception("Error in inference loop")
                time.sleep(0.1)

    def _run_inference(self) -> bool:
        """Execute one inference call and update the action queue.

        Returns True if actions were successfully received and enqueued.
        """
        index_before = self._action_queue.get_index()
        start_time = time.monotonic()

        obs = self._get_observation()
        obs = self._preprocess_images(obs)

        if self._rtc_enabled:
            prev_raw = self._action_queue.get_raw_leftover()
            if prev_raw is not None:
                obs["_rtc_prev_actions"] = prev_raw
            obs["_rtc_config"] = {
                **self._rtc_config,
                "inference_delay": self._latency_estimator.estimate_steps,
            }

        result = self._policy.infer(obs)

        elapsed = time.monotonic() - start_time
        is_cold_start = not self._first_inference_done.is_set()
        if is_cold_start:
            self._cold_start_time = elapsed
        else:
            self._inference_count += 1
            self._total_inference_time += elapsed
            self._latency_estimator.update(elapsed)

        actions = result.get("actions")
        raw_actions = result.get("raw_actions")

        if actions is None:
            logger.error("No actions returned from policy")
            return False

        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        if raw_actions is not None and raw_actions.ndim == 1:
            raw_actions = raw_actions[np.newaxis, :]

        index_after = self._action_queue.get_index()
        consumed = max(0, index_after - index_before)

        if self._rtc_enabled and raw_actions is not None:
            self._action_queue.push_rtc(
                processed_actions=actions,
                raw_actions=raw_actions,
                actions_consumed_during_inference=consumed,
            )
        else:
            self._action_queue.push(
                processed_actions=actions,
                raw_actions=raw_actions,
            )

        label = "cold_start" if is_cold_start else f"#{self._inference_count}"
        logger.debug(
            "Inference %s: %.1fms, consumed_during=%d, new_queue_size=%d",
            label,
            elapsed * 1000,
            consumed,
            self._action_queue.remaining(),
        )
        return True


class InterpolatedActionWrapper:
    """Wrapper that linearly interpolates between model actions to lengthen execution time.

    Sits between the consumer and RTCInferenceManager.get_action(). Does not modify
    the action queue—raw_actions and leftover chunks for RTC inpainting stay unchanged.

    With factor=1.7, each model action yields ~1.7 control steps on average, giving
    inference ~70% more time per chunk.

    Args:
        manager: The RTCInferenceManager to wrap.
        factor: Steps per model action (>= 1.0). Default 1.7.
    """

    def __init__(
        self,
        manager: RTCInferenceManager,
        factor: float = 1.7,
    ):
        if factor < 1.0:
            raise ValueError("factor must be >= 1.0")
        self._manager = manager
        self._factor = factor
        manager._latency_estimator._interpolation_factor = factor
        self._buffer: List[np.ndarray] = []
        self._prev: Optional[np.ndarray] = None
        self._accumulator: float = 0.0

    def get_action(self) -> Optional[np.ndarray]:
        if self._buffer:
            return self._buffer.pop(0)

        next_action = self._manager.get_action()
        if next_action is None:
            return None

        if self._prev is None:
            self._prev = next_action.copy()
            return next_action

        self._accumulator += self._factor
        steps = max(1, int(self._accumulator))
        self._accumulator -= steps

        for i in range(1, steps):
            t = i / steps
            self._buffer.append(
                self._prev + t * (next_action - self._prev)
            )
        self._buffer.append(next_action.copy())
        self._prev = next_action.copy()

        return self._buffer.pop(0)

    @property
    def action_queue(self) -> RTCActionQueue:
        """Passthrough to underlying manager's action queue."""
        return self._manager.action_queue

    def start(
        self,
        wait_for_first_action: bool = True,
    ) -> None:
        """Delegate to manager.start()."""
        self._manager.start(
            wait_for_first_action=wait_for_first_action,
        )

    def stop(self) -> None:
        """Delegate to manager.stop()."""
        self._manager.stop()
