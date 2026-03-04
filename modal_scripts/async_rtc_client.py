"""Asynchronous RTC inference client for overlapping inference with action execution.

This module provides ``RTCAsyncInferenceClient``, a client that runs inference
in a background thread while the main thread executes actions at the robot's
control frequency.  It handles both cases:

* **Inference faster than execution** -- action chunks queue up so there is
  never a gap.
* **Inference slower than execution** -- the client sends the next observation
  to the server *before* the current chunk runs out, minimising the window
  where the robot has no fresh actions.

When ``interpolate_actions=True`` and the chunk cannot fill the inference
window at the requested ``control_freq``, the client resamples the action
waypoints along an **ease-out** curve so that:

1. The robot covers most of the planned trajectory early, then
2. decelerates smoothly toward the end of the window.

The deceleration means the robot is nearly stationary right when the next
observation is captured, reducing motion blur and giving the model a sharper
image to plan from.

Usage (see also ``examples/async_rtc_example.py``)::

    client = RTCAsyncInferenceClient(
        server_url="https://…modal.run",
        model_config={...},
        control_freq=30.0,
        interpolate_actions=True,   # enable interpolation + dampening
        damping_power=2.0,          # quadratic ease-out
    )

    obs = robot.get_observation()
    first_action = client.warmup(obs)
    robot.execute(first_action)

    while True:
        obs = robot.get_observation()
        action = client.step(obs)
        if action is not None:
            robot.execute(action)
        time.sleep(1.0 / control_freq)

    client.close()
"""

from __future__ import annotations

import collections
import functools
import logging
import math
import threading
import time
from typing import Any, Callable

import msgpack
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inline msgpack-numpy helpers (same wire format as openpi_client.msgpack_numpy
# so the server can unpack them transparently).
# ---------------------------------------------------------------------------

def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def interpolate_with_damping(
    actions: np.ndarray,
    target_count: int,
    power: float = 2.0,
) -> list[np.ndarray]:
    """Resample *actions* into *target_count* steps with ease-out dampening.

    The ease-out curve ``s(t) = 1 - (1 - t)^power`` maps uniform time
    ``t in [0, 1]`` to non-uniform progress ``s`` through the original
    action sequence.  Early timesteps advance quickly through the waypoints
    (robot moves fast), while late timesteps barely advance (robot
    decelerates).  At ``t = 1`` the velocity is zero -- the robot is
    stationary, giving the camera a blur-free frame for the next
    observation.

    Args:
        actions: Array of shape ``(N, action_dim)`` -- the raw waypoints.
        target_count: Number of output actions to produce (typically
            ``ceil(est_latency * control_freq)``).
        power: Exponent of the ease-out curve.  ``1`` is linear (no
            dampening), ``2`` is quadratic, ``3`` is cubic, etc.  Higher
            values produce sharper deceleration.

    Returns:
        List of ``target_count`` arrays, each of shape ``(action_dim,)``.
    """
    n = len(actions)
    if n == 0:
        return []
    if n == 1:
        return [actions[0].copy() for _ in range(target_count)]

    result: list[np.ndarray] = []
    last_idx = n - 1

    for i in range(target_count):
        t = i / max(target_count - 1, 1)

        # Ease-out: fast start, decelerating to zero velocity at t=1
        s = 1.0 - (1.0 - t) ** power

        idx_f = s * last_idx
        lo = int(idx_f)
        hi = min(lo + 1, last_idx)
        frac = idx_f - lo

        interpolated = actions[lo] * (1.0 - frac) + actions[hi] * frac
        result.append(interpolated)

    return result


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

class RTCAsyncInferenceClient:
    """Overlaps server inference with robot action execution.

    Architecture::

        Main thread (control loop)            Background thread (inference)
        ──────────────────────────            ─────────────────────────────
        step(obs)                             (blocked, waiting for trigger)
          ├─ store latest obs
          ├─ if buffer low → trigger  ───────→  read latest obs
          ├─ check for completed result       │  POST /infer to server
          │    → add actions to buffer        │  receive action chunk
          └─ return buffer.popleft()          │  (interpolate if needed)
                                              └─ add to buffer, signal done

    The background thread keeps at most *one* inference in flight.  It is
    triggered when the action buffer drops below a threshold computed from
    the running average of measured inference latency.
    """

    def __init__(
        self,
        server_url: str,
        model_config: dict[str, str],
        control_freq: float = 10.0,
        action_horizon: int | None = None,
        latency_buffer_multiplier: float = 1.5,
        max_buffer_size: int = 300,
        http_timeout: float = 120.0,
        default_latency: float = 1.0,
        infer_fn: Callable[[dict], dict] | None = None,
        interpolate_actions: bool = False,
        damping_power: float = 2.0,
    ):
        """
        Args:
            server_url: Base URL of the inference server (``https://…modal.run``).
            model_config: Dict merged into every request so the server knows
                which model to load.  Expected keys: ``hf_repo_id``,
                ``folder_path``, ``config_name``, and optionally
                ``dataset_repo_id``, ``stats_json_path``.
            control_freq: Robot control frequency in Hz.
            action_horizon: Max actions to keep from each chunk (after latency
                skip).  ``None`` means use all remaining actions.
            latency_buffer_multiplier: Safety factor applied to the latency
                estimate when deciding the trigger threshold.
            max_buffer_size: Hard cap on action buffer length.
            http_timeout: Timeout in seconds for HTTP requests.
            default_latency: Assumed latency (seconds) before any measurement.
            infer_fn: Optional override for the inference call.  Signature:
                ``(obs_with_config: dict) -> result_dict``.  Useful for
                testing or local mock servers.
            interpolate_actions: When ``True`` and inference is slower than
                chunk execution, resample the action waypoints with ease-out
                dampening to fill the full inference window at
                ``control_freq``.  The robot decelerates toward the end so
                the camera captures a sharp frame for the next observation.
                When ``False`` (default) actions are delivered at
                ``control_freq`` and ``step()`` returns ``None`` when the
                buffer is empty.
            damping_power: Exponent for the ease-out curve used when
                ``interpolate_actions`` is ``True``.  ``1.0`` = linear
                (no dampening), ``2.0`` = quadratic, ``3.0`` = cubic.
                Higher values produce sharper deceleration at the end.
        """
        self._server_url = server_url.rstrip("/")
        self._model_config = dict(model_config)
        self._control_freq = control_freq
        self._action_horizon = action_horizon
        self._latency_buf_mult = latency_buffer_multiplier
        self._http_timeout = http_timeout
        self._default_latency = default_latency
        self._infer_fn = infer_fn
        self._interpolate = interpolate_actions
        self._damping_power = damping_power

        # Thread-safe action buffer (individual timestep arrays)
        self._action_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=max_buffer_size
        )
        self._buffer_lock = threading.Lock()

        # Latest observation, written by main thread, read by inference thread.
        self._latest_obs: dict | None = None
        self._obs_lock = threading.Lock()

        # Inference coordination
        self._inference_pending = False
        self._inference_lock = threading.Lock()
        self._trigger_event = threading.Event()
        self._first_done = threading.Event()

        # Latency tracking
        self._latencies: collections.deque[float] = collections.deque(maxlen=20)
        self._last_latency: float = 0.0

        # Lifecycle
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="RTCAsyncInference"
        )
        self._thread.start()

        # Counters
        self._actions_returned = 0
        self._chunks_received = 0
        self._actions_skipped = 0
        self._empty_count = 0
        self._interpolated_chunks = 0

        # Snapshot of _actions_returned when the current inference started,
        # used to compute *actual* actions consumed (not an estimate).
        self._actions_returned_at_infer_start: int = 0

        # If the most recent chunk was interpolated, store its parameters so
        # we can map consumed interpolated actions back to original-trajectory
        # progress when computing the stale-action skip for the NEXT chunk.
        # Format: (original_waypoints, target_count, power) or None.
        self._last_interp_params: tuple[int, int, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, obs: dict) -> np.ndarray | None:
        """Update observation, maybe trigger inference, return next action.

        Call this once per control-loop iteration.  Returns ``None`` when the
        buffer is empty (the robot should hold its last position or take a
        safe default action).
        """
        with self._obs_lock:
            self._latest_obs = obs

        self._maybe_trigger()

        with self._buffer_lock:
            if self._action_buffer:
                self._actions_returned += 1
                return self._action_buffer.popleft()

        self._empty_count += 1
        return None

    def warmup(self, obs: dict, timeout: float = 120.0) -> np.ndarray:
        """Block until the first action chunk arrives.

        Call this *before* entering the control loop so the first ``step()``
        has actions immediately.

        Raises:
            TimeoutError: if the first inference does not finish in *timeout* s.
            RuntimeError: if inference finished but produced no actions.
        """
        with self._obs_lock:
            self._latest_obs = obs
        self._trigger_event.set()

        if not self._first_done.wait(timeout=timeout):
            raise TimeoutError(
                f"First inference did not complete within {timeout}s"
            )

        with self._buffer_lock:
            if self._action_buffer:
                self._actions_returned += 1
                return self._action_buffer.popleft()

        raise RuntimeError("First inference completed but no actions produced")

    def actions_remaining(self) -> int:
        with self._buffer_lock:
            return len(self._action_buffer)

    def is_inferring(self) -> bool:
        with self._inference_lock:
            return self._inference_pending

    def get_stats(self) -> dict[str, Any]:
        avg_lat = self._avg_latency()
        return {
            "actions_returned": self._actions_returned,
            "chunks_received": self._chunks_received,
            "actions_skipped": self._actions_skipped,
            "interpolated_chunks": self._interpolated_chunks,
            "empty_buffer_hits": self._empty_count,
            "avg_latency_ms": avg_lat * 1000,
            "last_latency_ms": self._last_latency * 1000,
            "buffer_size": self.actions_remaining(),
        }

    def close(self) -> None:
        self._shutdown.set()
        self._trigger_event.set()
        self._thread.join(timeout=5.0)

    # Context-manager support
    def __enter__(self) -> RTCAsyncInferenceClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_trigger(self) -> None:
        with self._inference_lock:
            if self._inference_pending:
                return

        remaining = self.actions_remaining()

        if self._interpolate:
            # Trigger LATE: capture the observation while the robot is in the
            # dampened tail of the trajectory (nearly stationary → sharp
            # camera frame).  This means the next chunk is based on a fresh
            # observation instead of one that is ``latency`` seconds stale.
            # The trade-off is an idle gap roughly equal to the inference
            # latency, which is acceptable -- stale actions are worse than
            # no actions.
            threshold = 1
        else:
            # Without interpolation there is no dampening guarantee, so
            # trigger early enough that the next chunk can (ideally) arrive
            # before the buffer drains.
            threshold = int(
                self._avg_latency() * self._control_freq * self._latency_buf_mult
            ) + 1

        if remaining <= threshold:
            self._trigger_event.set()

    def _avg_latency(self) -> float:
        if not self._latencies:
            return self._default_latency
        return sum(self._latencies) / len(self._latencies)

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        logger.info("[RTCAsync] Inference thread started")
        while not self._shutdown.is_set():
            triggered = self._trigger_event.wait(timeout=0.1)
            if self._shutdown.is_set():
                break
            if not triggered:
                continue
            self._trigger_event.clear()

            with self._obs_lock:
                obs = self._latest_obs
            if obs is None:
                continue

            with self._inference_lock:
                self._inference_pending = True
            self._actions_returned_at_infer_start = self._actions_returned

            try:
                t0 = time.monotonic()
                result = self._do_infer(obs)
                latency = time.monotonic() - t0

                self._last_latency = latency
                self._latencies.append(latency)
                self._chunks_received += 1

                actual_consumed = (
                    self._actions_returned
                    - self._actions_returned_at_infer_start
                )
                self._ingest_chunk(result, latency, actual_consumed)

                logger.info(
                    "[RTCAsync] Chunk #%d  latency=%.0fms  buffer=%d",
                    self._chunks_received,
                    latency * 1000,
                    self.actions_remaining(),
                )

                if not self._first_done.is_set():
                    self._first_done.set()

            except Exception:
                logger.exception("[RTCAsync] Inference error")
            finally:
                with self._inference_lock:
                    self._inference_pending = False

        logger.info("[RTCAsync] Inference thread stopped")

    def _do_infer(self, obs: dict) -> dict:
        obs_with_config = {**obs, **self._model_config}

        if self._infer_fn is not None:
            return self._infer_fn(obs_with_config)

        return self._infer_http(obs_with_config)

    def _infer_http(self, payload: dict) -> dict:
        import requests as _requests

        data = packb(payload)
        resp = _requests.post(
            f"{self._server_url}/infer",
            data=data,
            headers={"Content-Type": "application/x-msgpack"},
            timeout=self._http_timeout,
        )
        resp.raise_for_status()
        return unpackb(resp.content)

    def _ingest_chunk(
        self, result: dict, latency: float, actual_consumed: int
    ) -> None:
        actions = result.get("actions")
        if actions is None:
            logger.warning("[RTCAsync] Server result missing 'actions' key")
            return

        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        chunk_size = actions.shape[0]

        # ``actual_consumed`` counts buffer pops during inference.  When the
        # previous chunk was interpolated (N waypoints → M steps with ease-
        # out), those M consumed steps do NOT correspond to M original
        # actions of trajectory progress -- the dampening curve means the
        # robot covered fewer equivalent waypoints.  We invert the ease-out
        # to recover the true trajectory progress.
        if self._last_interp_params is not None and actual_consumed > 0:
            orig_n, target_m, power = self._last_interp_params
            t = min(actual_consumed / max(target_m, 1), 1.0)
            s = 1.0 - (1.0 - t) ** power
            effective_consumed = int(s * orig_n)
            logger.debug(
                "[RTCAsync] Inverse map: %d interpolated consumed → "
                "%d equivalent original (N=%d M=%d p=%.1f)",
                actual_consumed, effective_consumed, orig_n, target_m, power,
            )
        else:
            effective_consumed = actual_consumed

        start = min(effective_consumed, chunk_size - 1)

        end = chunk_size
        if self._action_horizon is not None:
            end = min(start + self._action_horizon, chunk_size)

        self._actions_skipped += start
        usable = end - start

        # --- Interpolation with dampening --------------------------------
        # If enabled and the usable actions can't fill the estimated
        # inference window at control_freq, resample them with an ease-out
        # curve so the robot decelerates toward the end.
        if self._interpolate and usable >= 1:
            natural_duration = usable / self._control_freq
            est_latency = self._avg_latency()

            if natural_duration < est_latency:
                target_count = math.ceil(est_latency * self._control_freq)
                resampled = interpolate_with_damping(
                    actions[start:end],
                    target_count,
                    power=self._damping_power,
                )
                self._interpolated_chunks += 1
                self._last_interp_params = (
                    usable, target_count, self._damping_power,
                )

                logger.debug(
                    "[RTCAsync] Interpolated: %d waypoints → %d actions  "
                    "(power=%.1f, est_lat=%.0fms)",
                    usable,
                    target_count,
                    self._damping_power,
                    est_latency * 1000,
                )

                with self._buffer_lock:
                    for a in resampled:
                        self._action_buffer.append(a)
                return

        # --- Default: add raw actions ------------------------------------
        self._last_interp_params = None
        logger.debug(
            "[RTCAsync] Chunk ingest: chunk_size=%d  actual_consumed=%d  "
            "start=%d  usable=%d",
            chunk_size,
            actual_consumed,
            start,
            usable,
        )

        with self._buffer_lock:
            for i in range(start, end):
                self._action_buffer.append(actions[i])
