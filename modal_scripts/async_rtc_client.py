"""Asynchronous RTC inference client for overlapping inference with action execution.

This module provides ``RTCAsyncInferenceClient``, a client that runs inference
in a background thread while the main thread executes actions at the robot's
control frequency.  It handles both cases:

* **Inference faster than execution** -- action chunks queue up so there is
  never a gap.
* **Inference slower than execution** -- the client sends the next observation
  to the server *before* the current chunk runs out, minimising the window
  where the robot has no fresh actions.

The current ``rtc_eval.py`` approach estimates latency and requests new actions
based on that estimate, but this breaks when the latency exceeds the chunk
execution time.  ``RTCAsyncInferenceClient`` solves this by always keeping at
most one inference in flight and triggering the next request as soon as the
buffer runs low -- rather than relying on a fixed latency estimate.

Usage (see also ``examples/async_rtc_example.py``)::

    client = RTCAsyncInferenceClient(
        server_url="https://…modal.run",
        model_config={
            "hf_repo_id": "user/repo",
            "folder_path": "checkpoints/pi05_rtc",
            "config_name": "pi05_rtc",
        },
        control_freq=10.0,
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
          └─ return buffer.popleft()          │  skip stale actions
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
        """
        self._server_url = server_url.rstrip("/")
        self._model_config = dict(model_config)
        self._control_freq = control_freq
        self._action_horizon = action_horizon
        self._latency_buf_mult = latency_buffer_multiplier
        self._http_timeout = http_timeout
        self._default_latency = default_latency
        self._infer_fn = infer_fn

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
        self._collect_if_done()

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
        threshold = int(
            self._avg_latency() * self._control_freq * self._latency_buf_mult
        ) + 1
        if remaining <= threshold:
            self._trigger_event.set()

    def _collect_if_done(self) -> None:
        """No-op -- results are pushed by the inference thread directly."""

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

            try:
                t0 = time.monotonic()
                result = self._do_infer(obs)
                latency = time.monotonic() - t0

                self._last_latency = latency
                self._latencies.append(latency)
                self._chunks_received += 1

                self._ingest_chunk(result, latency)

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

    def _ingest_chunk(self, result: dict, latency: float) -> None:
        actions = result.get("actions")
        if actions is None:
            logger.warning("[RTCAsync] Server result missing 'actions' key")
            return

        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        chunk_size = actions.shape[0]

        # The first ``actions_elapsed`` steps in the chunk correspond to the
        # time the robot was busy while the server was computing.  They are
        # "stale" because the robot has already moved past that point.
        actions_elapsed = int(latency * self._control_freq)
        start = min(actions_elapsed, chunk_size - 1)

        end = chunk_size
        if self._action_horizon is not None:
            end = min(start + self._action_horizon, chunk_size)

        self._actions_skipped += start

        with self._buffer_lock:
            for i in range(start, end):
                self._action_buffer.append(actions[i])
