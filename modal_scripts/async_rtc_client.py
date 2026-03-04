"""Asynchronous RTC inference client for overlapping inference with action execution.

This module provides ``RTCAsyncInferenceClient``, a client that runs model
inference in a background thread while the main thread executes actions at
the robot's control frequency.

Design philosophy
-----------------
**Fresh observations always beat smooth but stale actions.**

The client uses a fixed, low trigger threshold (default 1 remaining action)
to decide when to request new inference.  This means the observation sent to
the server is captured as late as possible -- right when the current action
chunk is nearly exhausted -- giving the model the most up-to-date view of
the world.  If inference takes longer than the remaining actions, the robot
idles until the result arrives.  That idle gap is preferable to executing
actions that were planned from an observation that is ``latency`` seconds
old.

When ``interpolate_actions=True`` the client also resamples the action
waypoints with an ease-out curve so the robot **decelerates** toward the end
of each chunk.  By the time the observation is captured the robot is nearly
stationary, giving the camera a sharp, motion-blur-free frame.

Inference trigger strategy
--------------------------
``_maybe_trigger()`` fires when ``actions_remaining <= trigger_threshold``
(default 1).  Only one inference can be in flight at a time.

* **Inference faster than execution** -- the trigger fires near the end of
  each chunk.  The result usually arrives before the buffer fully drains
  (zero or near-zero idle gap).
* **Inference slower than execution** -- the trigger fires at the same late
  point.  The buffer drains and the robot idles for roughly
  ``latency - chunk_duration`` seconds, then resumes with actions derived
  from a fresh observation.

Usage (see also ``examples/async_rtc_example.py``)::

    client = RTCAsyncInferenceClient(
        server_url="https://…modal.run",
        model_config={...},
        control_freq=30.0,
        interpolate_actions=True,
        damping_power=2.0,
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
          │    (fixed threshold)              │  POST /infer to server
          ├─ pop action from buffer           │  receive action chunk
          └─ return action (or None)          │  (interpolate if needed)
                                              └─ add to buffer, signal done

    The background thread keeps at most *one* inference in flight.  It is
    triggered when the action buffer drops to ``trigger_threshold`` (default
    1).  This late trigger ensures that the observation sent to the server
    is as fresh as possible.  An idle gap after the buffer drains is
    preferred over executing actions derived from a stale observation.
    """

    def __init__(
        self,
        server_url: str,
        model_config: dict[str, str],
        control_freq: float = 10.0,
        action_horizon: int | None = None,
        trigger_threshold: int = 1,
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
            action_horizon: Max actions to keep from each chunk (after stale-
                action skip).  ``None`` means use all remaining actions.
            trigger_threshold: Request new inference when the buffer has this
                many actions (or fewer) remaining.  A low value (default 1)
                captures the freshest possible observation.  Set higher only
                if you need to overlap inference with execution and can
                tolerate older observations.
            max_buffer_size: Hard cap on action buffer length.
            http_timeout: Timeout in seconds for HTTP requests.
            default_latency: Assumed latency (seconds) before any measurement
                is available.  Used by the interpolation logic to size the
                first resampled chunk.
            infer_fn: Optional override for the inference call.  Signature:
                ``(obs_with_config: dict) -> result_dict``.  Useful for
                testing or local mock servers.
            interpolate_actions: When ``True`` and inference is slower than
                chunk execution, resample the action waypoints with ease-out
                dampening so the robot decelerates toward the end of each
                chunk.  The nearly-stationary tail gives the camera a sharp
                frame right before the next observation is captured.
            damping_power: Exponent for the ease-out curve used when
                ``interpolate_actions`` is ``True``.  ``1.0`` = linear
                (no dampening), ``2.0`` = quadratic, ``3.0`` = cubic.
                Higher values produce sharper deceleration at the end.
        """
        self._server_url = server_url.rstrip("/")
        self._model_config = dict(model_config)
        self._control_freq = control_freq
        self._action_horizon = action_horizon
        self._trigger_threshold = trigger_threshold
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

        # Latency tracking (used by interpolation sizing, not by trigger)
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
        """Request inference when the buffer is nearly exhausted.

        Uses a fixed threshold so the observation is always captured as late
        as possible (freshest state).  Any resulting idle gap is preferred
        over actions derived from a stale observation.
        """
        with self._inference_lock:
            if self._inference_pending:
                return

        if self.actions_remaining() <= self._trigger_threshold:
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
        # previous chunk was interpolated (N waypoints -> M steps with ease-
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
                "[RTCAsync] Inverse map: %d interpolated consumed -> "
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
                    "[RTCAsync] Interpolated: %d waypoints -> %d actions  "
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
