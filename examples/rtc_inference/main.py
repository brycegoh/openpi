"""Example: Continuous RTC inference with a mock robot and timing visualization.

Demonstrates the RTCInferenceManager that asynchronously fetches action chunks
from a policy server while a mock robot continuously executes actions.

The key idea is threshold-based pre-fetching: when the action queue drops below
25% of the action horizon, a new inference call is made. This ensures either:
  - Continuous actions when inference is faster than execution
  - Minimized idle time when inference is slower than execution

For RTC mode, the leftover actions from the previous chunk are sent to the
server so it can use inpainting to maintain temporal consistency across chunks.

After the run, a timing dot plot is generated showing when actions were executed,
when inference was called, and when inference responses were received. Statistics
on idle time and overlap (time saved vs synchronous inference) are printed.

Usage (connect to a deployed Modal endpoint by app name):
    python main.py --modal-app-name openpi-policy-server-rtc-1 --action-horizon 50

Usage (connect to a running policy server by URL):
    python main.py --host ws://your-server-url/ws --action-horizon 50

Usage (with a mock policy for local testing without a server):
    python main.py --use-mock-policy --action-horizon 50

Usage (RTC disabled, simple sequential chunking):
    python main.py --modal-app-name openpi-policy-server-rtc-1 --no-rtc
"""

import dataclasses
import logging
import threading
import time
from typing import Optional

import numpy as np
import tyro

logger = logging.getLogger(__name__)


class MockRobot:
    """Simulated robot that produces random observations and logs executed actions.

    Mimics a real robot interface with get_observation() and execute_action() methods.
    Each observation contains a random state vector and random camera images.
    """

    def __init__(self, state_dim: int = 14, image_size: int = 224, num_cameras: int = 2):
        self._state_dim = state_dim
        self._image_size = image_size
        self._num_cameras = num_cameras
        self._step = 0
        self._state = np.random.randn(state_dim).astype(np.float32)

    def get_observation(self) -> dict:
        """Return current observation with state and camera images."""
        obs = {
            "state": self._state.copy(),
            "images": {},
            "prompt": "pick up the object",
        }
        camera_names = ["top", "left", "right"]
        for i in range(self._num_cameras):
            name = camera_names[i % len(camera_names)]
            obs["images"][name] = np.random.randint(
                0, 256, size=(3, self._image_size, self._image_size), dtype=np.uint8
            )
        return obs

    def execute_action(self, action: np.ndarray) -> None:
        """Execute an action on the mock robot (updates internal state)."""
        self._step += 1
        self._state = self._state + 0.01 * action[: self._state_dim]


class MockPolicy:
    """Mock policy for testing without a real server.

    Returns random action chunks with realistic timing simulation.
    Implements the same interface as WebsocketClientPolicy.
    """

    def __init__(self, action_horizon: int = 50, action_dim: int = 14, latency_ms: float = 200.0):
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._latency_ms = latency_ms

    def infer(self, obs: dict) -> dict:
        time.sleep(self._latency_ms / 1000.0)
        actions = np.random.randn(self._action_horizon, self._action_dim).astype(np.float32) * 0.1
        raw_actions = actions.copy()
        return {
            "actions": actions,
            "raw_actions": raw_actions,
            "state": obs.get("state", np.zeros(self._action_dim)),
        }


@dataclasses.dataclass
class TimingEvent:
    """A single timestamped event for the timing visualization."""
    kind: str  # "action_executed", "idle", "inference_start", "inference_end"
    timestamp: float  # time.monotonic() value


class TimingLog:
    """Thread-safe container for timing events recorded from multiple threads."""

    def __init__(self):
        self._events: list[TimingEvent] = []
        self._lock = threading.Lock()

    def record(self, kind: str) -> None:
        ev = TimingEvent(kind=kind, timestamp=time.monotonic())
        with self._lock:
            self._events.append(ev)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    @property
    def events(self) -> list[TimingEvent]:
        with self._lock:
            return list(self._events)


class InstrumentedPolicy:
    """Wraps any policy to record inference_start / inference_end timing events."""

    def __init__(self, inner_policy, timing_log: TimingLog):
        self._inner = inner_policy
        self._timing = timing_log

    def infer(self, obs: dict) -> dict:
        self._timing.record("inference_start")
        result = self._inner.infer(obs)
        self._timing.record("inference_end")
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_modal_endpoint_url(app_name: str, function_name: str = "endpoint") -> str:
    """Fetch the WebSocket URL from a deployed Modal app."""
    import modal

    logger.info("Fetching endpoint URL from Modal app '%s' function '%s'...", app_name, function_name)
    try:
        func = modal.Function.from_name(app_name, function_name)
        endpoint_url = func.get_web_url()
    except Exception as e:
        logger.error("Failed to fetch Modal endpoint URL: %s", e)
        logger.error("Make sure the app '%s' is deployed. Run: modal deploy modal_scripts/modal_serve_policy.py", app_name)
        raise

    ws_url = endpoint_url.replace("https://", "wss://").replace("http://", "ws://")
    if not ws_url.endswith("/ws"):
        ws_url = ws_url.rstrip("/") + "/ws"
    logger.info("Resolved endpoint URL: %s", ws_url)
    return ws_url


@dataclasses.dataclass
class Args:
    """Arguments for the RTC inference example."""

    # Server connection — provide modal_app_name OR host (modal takes precedence)
    modal_app_name: str | None = None
    modal_function_name: str = "endpoint"
    host: str = "0.0.0.0"
    port: int | None = 8000
    api_key: str | None = None

    # Policy configuration (required for Modal server — included in every observation)
    hf_repo_id: str | None = None
    folder_path: str | None = None
    config_name: str | None = None
    prompt: str | None = None
    dataset_repo_id: str | None = None
    stats_json_path: str | None = None

    # Use a mock policy instead of connecting to a server
    use_mock_policy: bool = False
    mock_latency_ms: float = 200.0

    # Action parameters
    action_horizon: int = 50
    action_dim: int = 14
    control_hz: float = 30.0

    # RTC parameters
    rtc: bool = True
    execution_horizon: int = 10
    prefix_attention_schedule: str = "LINEAR"
    max_guidance_weight: float = 5.0
    refill_threshold: float = 0.50

    # Run duration
    duration: float = 30.0

    # Output
    save_plot: str = "timing_plot.png"

    # Logging
    verbose: bool = False


def compute_stats(
    events: list[TimingEvent],
    t0: float,
    action_interval: float,
) -> dict:
    """Derive idle time and overlap statistics from recorded timing events.

    Returns a dict with:
        idle_time_s        – total seconds the robot had no action to execute
        overlap_time_s     – total seconds where inference was in-flight AND
                             actions were being executed concurrently
        total_inference_s  – sum of all inference durations
        sync_duration_s    – hypothetical duration if inference blocked execution
        async_duration_s   – actual wall-clock duration
        time_saved_s       – sync_duration_s - async_duration_s
        time_saved_pct     – percentage of sync duration saved
    """
    action_times = []
    idle_times = []
    inference_intervals: list[tuple[float, float]] = []

    pending_start: Optional[float] = None
    for ev in sorted(events, key=lambda e: e.timestamp):
        t = ev.timestamp - t0
        if ev.kind == "action_executed":
            action_times.append(t)
        elif ev.kind == "idle":
            idle_times.append(t)
        elif ev.kind == "inference_start":
            pending_start = t
        elif ev.kind == "inference_end":
            if pending_start is not None:
                inference_intervals.append((pending_start, t))
                pending_start = None

    idle_time_s = len(idle_times) * action_interval

    # Overlap: for each inference interval, count how many action ticks fall inside it
    overlap_time_s = 0.0
    for inf_start, inf_end in inference_intervals:
        for at in action_times:
            if inf_start <= at <= inf_end:
                overlap_time_s += action_interval

    total_inference_s = sum(end - start for start, end in inference_intervals)

    async_duration_s = max(
        (ev.timestamp - t0 for ev in events),
        default=0.0,
    )

    # In a synchronous model, each inference blocks execution entirely, so
    # the total time = action execution time + total inference time (no overlap).
    action_execution_s = len(action_times) * action_interval
    sync_duration_s = action_execution_s + total_inference_s

    time_saved_s = sync_duration_s - async_duration_s
    time_saved_pct = (time_saved_s / sync_duration_s * 100) if sync_duration_s > 0 else 0.0

    return {
        "idle_time_s": idle_time_s,
        "overlap_time_s": overlap_time_s,
        "total_inference_s": total_inference_s,
        "sync_duration_s": sync_duration_s,
        "async_duration_s": async_duration_s,
        "time_saved_s": time_saved_s,
        "time_saved_pct": time_saved_pct,
        "num_inferences": len(inference_intervals),
        "num_actions": len(action_times),
        "num_idles": len(idle_times),
    }


def plot_timeline(
    events: list[TimingEvent],
    t0: float,
    stats: dict,
    save_path: str,
) -> None:
    """Generate a dot-plot timeline of action execution, inference calls, and idle periods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_ts = []
    idle_ts = []
    inf_start_ts = []
    inf_end_ts = []
    inference_intervals: list[tuple[float, float]] = []

    pending_start: Optional[float] = None
    for ev in sorted(events, key=lambda e: e.timestamp):
        t = ev.timestamp - t0
        if ev.kind == "action_executed":
            action_ts.append(t)
        elif ev.kind == "idle":
            idle_ts.append(t)
        elif ev.kind == "inference_start":
            inf_start_ts.append(t)
            pending_start = t
        elif ev.kind == "inference_end":
            inf_end_ts.append(t)
            if pending_start is not None:
                inference_intervals.append((pending_start, t))
                pending_start = None

    fig, ax = plt.subplots(figsize=(16, 4))

    row_labels = ["Action Executed", "Inference Called", "Inference Received"]
    row_y = {label: i for i, label in enumerate(row_labels)}

    # Actions
    if action_ts:
        ax.scatter(action_ts, [row_y["Action Executed"]] * len(action_ts),
                   color="#2196F3", s=8, alpha=0.7, zorder=3, label="Action Executed")

    # Idle ticks – shown as red dots on the action row
    if idle_ts:
        ax.scatter(idle_ts, [row_y["Action Executed"]] * len(idle_ts),
                   color="#F44336", s=8, alpha=0.7, marker="x", zorder=3, label="Idle (no action)")

    # Inference start
    if inf_start_ts:
        ax.scatter(inf_start_ts, [row_y["Inference Called"]] * len(inf_start_ts),
                   color="#4CAF50", s=30, alpha=0.9, zorder=3, label="Inference Called")

    # Inference end
    if inf_end_ts:
        ax.scatter(inf_end_ts, [row_y["Inference Received"]] * len(inf_end_ts),
                   color="#FF9800", s=30, alpha=0.9, zorder=3, label="Inference Received")

    # Horizontal bars connecting inference start->end (spanning the two rows)
    for start, end in inference_intervals:
        ax.plot([start, end], [row_y["Inference Called"], row_y["Inference Received"]],
                color="#9E9E9E", linewidth=1.5, alpha=0.5, zorder=2)

    # Shade idle regions on the action row
    if idle_ts and len(idle_ts) >= 2:
        idle_sorted = sorted(idle_ts)
        gap_threshold = (idle_sorted[1] - idle_sorted[0]) * 3 if len(idle_sorted) > 1 else 0.1
        region_start = idle_sorted[0]
        prev = idle_sorted[0]
        for t in idle_sorted[1:]:
            if t - prev > gap_threshold:
                ax.axvspan(region_start, prev, ymin=0, ymax=0.4, color="#F44336", alpha=0.12)
                region_start = t
            prev = t
        ax.axvspan(region_start, prev, ymin=0, ymax=0.4, color="#F44336", alpha=0.12)

    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Time (s)")
    ax.set_title("RTC Inference Timeline")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)

    # Stats text box
    stats_text = (
        f"Idle: {stats['idle_time_s']:.2f}s  |  "
        f"Overlap: {stats['overlap_time_s']:.2f}s  |  "
        f"Avg inference: {stats['total_inference_s'] / max(stats['num_inferences'], 1) * 1000:.0f}ms  |  "
        f"Time saved vs sync: {stats['time_saved_s']:.2f}s ({stats['time_saved_pct']:.1f}%)"
    )
    fig.text(0.5, -0.02, stats_text, ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved timing plot to %s", save_path)
    plt.close(fig)


def main(args: Args) -> None:
    from openpi_client.rtc_client import RTCInferenceManager

    timing_log = TimingLog()
    robot = MockRobot(state_dim=args.action_dim, num_cameras=2)

    if args.use_mock_policy:
        logger.info("Using mock policy (latency=%.0fms)", args.mock_latency_ms)
        raw_policy = MockPolicy(
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
            latency_ms=args.mock_latency_ms,
        )
    else:
        from openpi_client import websocket_client_policy as _wcp

        if args.modal_app_name:
            ws_url = get_modal_endpoint_url(args.modal_app_name, args.modal_function_name)
            raw_policy = _wcp.WebsocketClientPolicy(host=ws_url, api_key=args.api_key)
        else:
            logger.info("Connecting to server at %s:%s", args.host, args.port)
            raw_policy = _wcp.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
            )
        logger.info("Connected. Server metadata: %s", raw_policy.get_server_metadata())

    policy = InstrumentedPolicy(raw_policy, timing_log)

    # Build extra fields injected into every observation for the Modal server.
    server_obs_fields: dict = {}
    if args.hf_repo_id:
        server_obs_fields["hf_repo_id"] = args.hf_repo_id
    if args.folder_path:
        server_obs_fields["folder_path"] = args.folder_path
    if args.config_name:
        server_obs_fields["config_name"] = args.config_name
    if args.prompt:
        server_obs_fields["prompt"] = args.prompt
    if args.dataset_repo_id:
        server_obs_fields["dataset_repo_id"] = args.dataset_repo_id
    if args.stats_json_path:
        server_obs_fields["stats_json_path"] = args.stats_json_path

    def get_observation() -> dict:
        obs = robot.get_observation()
        obs.update(server_obs_fields)
        return obs

    rtc_config = {
        "enabled": args.rtc,
        "execution_horizon": args.execution_horizon,
        "prefix_attention_schedule": args.prefix_attention_schedule,
        "max_guidance_weight": args.max_guidance_weight,
    }

    manager = RTCInferenceManager(
        policy=policy,
        get_observation_fn=get_observation,
        action_horizon=args.action_horizon,
        refill_threshold=args.refill_threshold,
        rtc_enabled=args.rtc,
        rtc_config=rtc_config,
    )

    logger.info("Starting RTC inference manager...")
    logger.info("  Action horizon: %d", args.action_horizon)
    logger.info("  Control Hz: %.1f", args.control_hz)
    logger.info("  Refill threshold: %.0f%% (%d actions)", args.refill_threshold * 100, int(args.action_horizon * args.refill_threshold))
    logger.info("  RTC enabled: %s", args.rtc)
    if args.rtc:
        logger.info("  Execution horizon: %d", args.execution_horizon)
        logger.info("  Attention schedule: %s", args.prefix_attention_schedule)
    logger.info("  Duration: %.1fs", args.duration)

    manager.start()

    timing_log.clear()

    action_interval = 1.0 / args.control_hz
    t0 = time.monotonic()
    start_time = time.time()
    actions_executed = 0
    actions_missed = 0
    last_report_time = start_time

    try:
        while (time.time() - start_time) < args.duration:
            loop_start = time.perf_counter()

            action = manager.get_action()

            if action is not None:
                robot.execute_action(action)
                actions_executed += 1
                timing_log.record("action_executed")
            else:
                actions_missed += 1
                timing_log.record("idle")

            # Periodic status report
            now = time.time()
            if now - last_report_time >= 5.0:
                elapsed = now - start_time
                queue_remaining = manager.action_queue.remaining()
                hit_rate = actions_executed / max(actions_executed + actions_missed, 1) * 100
                logger.info(
                    "[%.1fs] executed=%d missed=%d hit_rate=%.1f%% queue=%d inferences=%d",
                    elapsed,
                    actions_executed,
                    actions_missed,
                    hit_rate,
                    queue_remaining,
                    manager._inference_count,
                )
                last_report_time = now

            # Maintain control frequency
            dt = time.perf_counter() - loop_start
            sleep_time = action_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        manager.stop()

    elapsed = time.time() - start_time
    total_steps = actions_executed + actions_missed
    hit_rate = actions_executed / max(total_steps, 1) * 100

    # --- Timing statistics ---
    stats = compute_stats(timing_log.events, t0, action_interval)

    logger.info("=" * 60)
    logger.info("Run Summary")
    logger.info("=" * 60)
    logger.info("  Duration: %.1fs", elapsed)
    logger.info("  Total control steps: %d", total_steps)
    logger.info("  Actions executed: %d", actions_executed)
    logger.info("  Actions missed (empty queue): %d", actions_missed)
    logger.info("  Hit rate: %.1f%%", hit_rate)
    logger.info("  Effective Hz: %.1f", actions_executed / elapsed)
    logger.info("  Cold start inference: %.1fms (excluded from stats)", manager._cold_start_time * 1000)
    logger.info("  Inferences (excl. cold start): %d", manager._inference_count)
    if manager._inference_count > 0:
        avg_infer_ms = (manager._total_inference_time / manager._inference_count) * 1000
        logger.info("  Avg inference time: %.1fms", avg_infer_ms)
    logger.info("-" * 60)
    logger.info("Timing Analysis (async vs sync)")
    logger.info("-" * 60)
    logger.info("  Idle time (no action available): %.3fs", stats["idle_time_s"])
    logger.info("  Overlap time (inference during execution): %.3fs", stats["overlap_time_s"])
    logger.info("  Total inference time: %.3fs", stats["total_inference_s"])
    logger.info("  Hypothetical sync duration: %.3fs", stats["sync_duration_s"])
    logger.info("  Actual async duration: %.3fs", stats["async_duration_s"])
    logger.info("  Time saved by async: %.3fs (%.1f%%)", stats["time_saved_s"], stats["time_saved_pct"])
    logger.info("=" * 60)

    # --- Plot ---
    plot_timeline(timing_log.events, t0, stats, args.save_plot)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
