"""DRTC Modal Client - async inference with real-time chunking over WebSocket.

Implements the DRTC control loop (Algorithm 1) on the client side:
- Main thread: called via infer() each control tick, pops actions from schedule,
  checks trigger condition, merges incoming chunks.
- Obs sender thread: reads from LWW register, sends observations over WebSocket.
- Action receiver thread: receives action chunks from WebSocket, writes to LWW register.

Usage:
    client = DRTCModalClient(
        app_name="openpi-policy-server-3",
        function_name="endpoint",
        hf_repo_id="user/repo",
        folder_path="checkpoints/pi05_tcr",
        config_name="pi05_tcr",
        action_horizon=50,
        fps=50,
    )
    client.start()
    for obs in observations:
        action = client.infer(obs)
        robot.send_action(action)
    client.stop()
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
import modal
import websockets.exceptions
import websockets.sync.client

from .lww_register import LWWRegister
from .latency_estimation import make_latency_estimator
from .action_schedule import ActionSchedule, TimedAction

logger = logging.getLogger(__name__)


@dataclass
class DRTCConfig:
    """Configuration for the DRTC client."""

    action_horizon: int = 50
    s_min: int = 14
    epsilon: int = 1
    fps: int = 50
    rtc_enabled: bool = True
    cooldown_enabled: bool = True
    inference_reset_mode: str = "cooldown"  # "cooldown" or "merge_reset"
    latency_estimator_type: str = "jk"
    latency_alpha: float = 0.125
    latency_beta: float = 0.25
    latency_k: float = 1.5

    @property
    def environment_dt(self) -> float:
        return 1.0 / self.fps


@dataclass
class _ObservationRequest:
    """Observation request written by main thread, read by obs sender."""
    control_step: int
    chunk_start_step: int
    obs: dict
    rtc_meta: dict | None = None


@dataclass
class _ReceivedActionChunk:
    """Action chunk received from server, written by action receiver."""
    actions: list[TimedAction]
    src_control_step: int
    chunk_start_step: int
    measured_latency: float


class DRTCModalClient:
    """DRTC async inference client for Modal WebSocket policy servers.

    Connects to a Modal-deployed policy server using the DRTC protocol,
    sending observations asynchronously and receiving action chunks that
    are merged into an action schedule using freshest-observation-wins.
    """

    def __init__(
        self,
        app_name: str,
        function_name: str,
        hf_repo_id: str,
        folder_path: str,
        config_name: str,
        prompt: str | None = None,
        dataset_repo_id: str | None = None,
        stats_json_path: str | None = None,
        connect_timeout: float | None = None,
        config: DRTCConfig | None = None,
    ):
        self._app_name = app_name
        self._function_name = function_name
        self._hf_repo_id = hf_repo_id
        self._folder_path = folder_path
        self._config_name = config_name
        self._prompt = prompt
        self._dataset_repo_id = dataset_repo_id
        self._stats_json_path = stats_json_path
        self._connect_timeout = connect_timeout

        self.config = config or DRTCConfig()
        self._packer = msgpack_numpy.Packer()
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._server_metadata: dict = {}

        # DRTC state
        self.action_step: int = -1
        self.control_step: int = 0
        self.obs_cooldown: int = 0
        self.action_schedule = ActionSchedule()

        self.latency_estimator = make_latency_estimator(
            kind=self.config.latency_estimator_type,
            fps=self.config.fps,
            alpha=self.config.latency_alpha,
            beta=self.config.latency_beta,
            k=self.config.latency_k,
            action_chunk_size=self.config.action_horizon,
            s_min=self.config.s_min,
        )

        # LWW registers for thread communication
        self._obs_request_reg: LWWRegister[_ObservationRequest | None] = LWWRegister(
            initial_control_step=-1, initial_value=None
        )
        self._action_reg: LWWRegister[_ReceivedActionChunk | None] = LWWRegister(
            initial_control_step=-1, initial_value=None
        )
        self._action_reader = self._action_reg.reader()

        self._shutdown_event = threading.Event()
        self._obs_sender_thread: threading.Thread | None = None
        self._action_receiver_thread: threading.Thread | None = None
        self._started = False
        self._last_action: np.ndarray | None = None

    def _get_modal_endpoint_url(self) -> str:
        func = modal.Function.from_name(self._app_name, self._function_name)
        endpoint_url = func.get_web_url()
        ws_url = endpoint_url.replace("https://", "wss://").replace("http://", "ws://")
        if not ws_url.endswith("/ws"):
            ws_url = ws_url.rstrip("/") + "/ws"
        return ws_url

    def _connect_and_handshake(self):
        """Connect to server and complete DRTC handshake."""
        url = self._get_modal_endpoint_url()
        logger.info(f"DRTC: Connecting to {url}")

        self._ws = websockets.sync.client.connect(
            url, compression=None, max_size=None,
            open_timeout=self._connect_timeout, close_timeout=self._connect_timeout,
        )

        # Phase 1: receive empty metadata
        init_metadata = msgpack_numpy.unpackb(self._ws.recv())
        logger.info(f"DRTC: Received server metadata: {init_metadata}")

        # Phase 2: send DRTC init message with model fields
        init_message: dict[str, Any] = {
            "mode": "drtc",
            "hf_repo_id": self._hf_repo_id,
            "folder_path": self._folder_path,
            "config_name": self._config_name,
            "action_horizon": self.config.action_horizon,
            "fps": self.config.fps,
            "rtc_enabled": self.config.rtc_enabled,
        }
        if self._prompt:
            init_message["prompt"] = self._prompt
        if self._dataset_repo_id:
            init_message["dataset_repo_id"] = self._dataset_repo_id
        if self._stats_json_path:
            init_message["stats_json_path"] = self._stats_json_path

        logger.info(f"DRTC: Sending init (model={self._config_name}, H={self.config.action_horizon})")
        self._ws.send(self._packer.pack(init_message))

        # Phase 4: receive DRTC metadata
        logger.info("DRTC: Waiting for model to load...")
        metadata_data = self._ws.recv()
        if isinstance(metadata_data, str):
            raise RuntimeError(f"DRTC: Server error during init:\n{metadata_data}")
        self._server_metadata = msgpack_numpy.unpackb(metadata_data)
        logger.info(f"DRTC: Server metadata: {self._server_metadata}")

    def start(self) -> bool:
        """Connect and start background threads."""
        try:
            self._connect_and_handshake()
        except Exception as e:
            logger.error(f"DRTC: Failed to connect: {e}")
            return False

        self._shutdown_event.clear()
        self.obs_cooldown = self.config.s_min + self.config.epsilon

        self._obs_sender_thread = threading.Thread(
            target=self._obs_sender_loop, name="drtc_obs_sender", daemon=True
        )
        self._action_receiver_thread = threading.Thread(
            target=self._action_receiver_loop, name="drtc_action_receiver", daemon=True
        )
        self._obs_sender_thread.start()
        self._action_receiver_thread.start()
        self._started = True
        logger.info("DRTC: Client started")
        return True

    def stop(self):
        """Stop background threads and close connection."""
        self._shutdown_event.set()
        if self._obs_sender_thread is not None:
            self._obs_sender_thread.join(timeout=2.0)
        if self._action_receiver_thread is not None:
            self._action_receiver_thread.join(timeout=2.0)
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._started = False
        logger.info("DRTC: Client stopped")

    @property
    def running(self) -> bool:
        return self._started and not self._shutdown_event.is_set()

    @property
    def current_action_step(self) -> int:
        return max(self.action_step, -1)

    # ------------------------------------------------------------------
    # Main thread: infer() implements the DRTC control loop per tick
    # ------------------------------------------------------------------

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Execute one tick of the DRTC control loop.

        Called by the environment at each control step. Returns a single action.

        Steps:
        1. Pop and execute the front action from the schedule.
        2. Check trigger condition; if met, request a new observation.
        3. Merge any incoming action chunks from the server.
        4. Advance the control step.
        """
        if not self.running:
            raise RuntimeError("DRTC client not running. Call start() first.")

        # Step 1: Execute action from schedule
        action_dict: dict[str, Any] = {}
        result = self.action_schedule.pop_front()
        if result is not None:
            step, action, src_control_step, chunk_start_step = result
            self.action_step = step
            self._last_action = action
            action_dict["actions"] = action
            action_dict["action_step"] = step
        elif self._last_action is not None:
            # Starvation: repeat last action
            action_dict["actions"] = self._last_action
            action_dict["action_step"] = self.action_step
            action_dict["starved"] = True
        else:
            # No actions available yet (warmup period)
            action_dict["actions"] = None
            action_dict["action_step"] = -1
            action_dict["starved"] = True

        schedule_size = self.action_schedule.get_size()

        # Step 2: Check inference trigger condition
        latency_steps = self.latency_estimator.estimate_steps
        H = self.config.action_horizon
        s_min = self.config.s_min
        epsilon = self.config.epsilon

        trigger_threshold = H - s_min
        if self.config.cooldown_enabled:
            should_trigger = schedule_size <= trigger_threshold and self.obs_cooldown == 0
        else:
            should_trigger = schedule_size <= trigger_threshold

        if should_trigger:
            current_step = self.current_action_step

            rtc_meta: dict[str, Any] | None = None
            if self.config.rtc_enabled:
                d = int(latency_steps)
                s = max(s_min, d)
                overlap_end = H - s

                action_schedule_spans = self.action_schedule.get_masking_chunk_spans(
                    current_step=current_step, max_len=overlap_end
                )
                rtc_meta = {
                    "enabled": True,
                    "latency_steps": d,
                    "action_schedule_spans": action_schedule_spans,
                    "overlap_end": overlap_end,
                }

            request = _ObservationRequest(
                control_step=self.control_step,
                chunk_start_step=max(current_step, 0),
                obs=obs,
                rtc_meta=rtc_meta,
            )

            if self.config.cooldown_enabled:
                self.obs_cooldown = latency_steps + epsilon

            self._obs_request_reg.update_if_newer(
                control_step=request.control_step, value=request
            )
        else:
            if self.config.cooldown_enabled and self.config.inference_reset_mode == "cooldown":
                self.obs_cooldown = max(self.obs_cooldown - 1, 0)

        # Step 3: Merge incoming action chunks
        state, _, is_new = self._action_reader.read_if_newer()
        chunk = state.value
        if is_new and chunk is not None:
            self.latency_estimator.update(chunk.measured_latency)

            self.action_schedule.merge(
                incoming_actions=chunk.actions,
                src_control_step=chunk.src_control_step,
                chunk_start_step=chunk.chunk_start_step,
                current_action_step=self.current_action_step,
            )

            if self.config.inference_reset_mode == "merge_reset":
                self.obs_cooldown = 0

        # Step 4: Advance control step
        self.control_step += 1

        action_dict["schedule_size"] = self.action_schedule.get_size()
        action_dict["latency_steps"] = self.latency_estimator.estimate_steps
        action_dict["cooldown"] = self.obs_cooldown
        return action_dict

    def reset(self):
        """Reset state for a new episode."""
        self.action_step = -1
        self.control_step = 0
        self.obs_cooldown = self.config.s_min + self.config.epsilon
        self.action_schedule.clear()
        self.latency_estimator.reset()
        self._last_action = None

    # ------------------------------------------------------------------
    # Background thread: observation sender
    # ------------------------------------------------------------------

    def _obs_sender_loop(self):
        """Background thread: reads obs requests from LWW register, sends over WS."""
        reader = self._obs_request_reg.reader()

        while not self._shutdown_event.is_set():
            try:
                state, _, is_new = reader.read_if_newer()
                request = state.value
                if not is_new or request is None:
                    time.sleep(0.005)
                    continue

                # Build wire message
                msg: dict[str, Any] = {"type": "obs"}
                msg["control_step"] = request.control_step
                msg["chunk_start_step"] = request.chunk_start_step
                msg["timestamp"] = time.time()
                if request.rtc_meta is not None:
                    msg["rtc_meta"] = request.rtc_meta

                # Merge observation data
                msg.update(request.obs)

                self._ws.send(self._packer.pack(msg))

            except websockets.exceptions.ConnectionClosed as e:
                if not self._shutdown_event.is_set():
                    logger.warning(f"DRTC obs sender: WebSocket connection closed: {e}")
                    self._shutdown_event.set()
                break
            except Exception as e:
                if self._shutdown_event.is_set():
                    break
                logger.error(f"DRTC obs sender error: {e}\n{traceback.format_exc()}")
                self._shutdown_event.set()
                break

    # ------------------------------------------------------------------
    # Background thread: action receiver
    # ------------------------------------------------------------------

    def _action_receiver_loop(self):
        """Background thread: receives action chunks from WS, writes to LWW register."""
        while not self._shutdown_event.is_set():
            try:
                raw = self._ws.recv(timeout=1.0)
                receive_time = time.time()

                if isinstance(raw, str):
                    logger.warning(f"DRTC: Received text from server: {raw[:200]}")
                    continue

                msg = msgpack_numpy.unpackb(raw)
                if msg.get("type") != "action_chunk":
                    continue

                actions = np.asarray(msg["actions"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)

                src_control_step = int(msg["source_control_step"])
                chunk_start_step = int(msg["chunk_start_step"])
                obs_timestamp = float(msg.get("timestamp", receive_time))
                dt = float(msg.get("dt", 1.0 / self.config.fps))
                num_actions = actions.shape[0]

                measured_latency = receive_time - obs_timestamp

                timed_actions = [
                    TimedAction(
                        action=actions[i],
                        action_step=chunk_start_step + i,
                        src_control_step=src_control_step,
                        chunk_start_step=chunk_start_step,
                    )
                    for i in range(num_actions)
                ]

                chunk = _ReceivedActionChunk(
                    actions=timed_actions,
                    src_control_step=src_control_step,
                    chunk_start_step=chunk_start_step,
                    measured_latency=measured_latency,
                )
                self._action_reg.update_if_newer(
                    control_step=src_control_step, value=chunk
                )

            except TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed as e:
                if not self._shutdown_event.is_set():
                    logger.warning(f"DRTC action receiver: WebSocket connection closed by server: {e}")
                    self._shutdown_event.set()
                break
            except Exception as e:
                if self._shutdown_event.is_set():
                    break
                logger.error(f"DRTC action receiver error: {e}\n{traceback.format_exc()}")
                self._shutdown_event.set()
                break
