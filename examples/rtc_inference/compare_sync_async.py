"""Compare sync (blocking) vs async+RTC inference modes.

Runs both modes sequentially on the same server connection, then generates
comparison plots for timing and per-dimension action values.

Sync mode:  blocking infer -> execute full chunk -> repeat (no overlap)
Async+RTC:  RTCInferenceManager with background pre-fetching and inpainting

Usage (mock policy for local testing):
    python compare_sync_async.py --use-mock-policy --duration 10

Usage (Modal endpoint):
    python examples/rtc_inference/compare_sync_async.py \
        --modal-app-name openpi-policy-server-rtc-1 \
        --hf-repo-id griffinlabs/pi05_412ep_pytorch \
        --folder-path "pi05_tcr_full_finetune_pytorch/pi05_412ep/20000" \
        --config-name pi05_tcr_full_finetune_pytorch \
        --prompt "Clean the countertop" \
        --dataset-repo-id griffinlabs/tcr-data \
        --stats-json-path ./norm_stats.json \
        --duration 60

Usage (direct WebSocket server):
    python compare_sync_async.py --host ws://your-server/ws --duration 20
"""

import dataclasses
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rtc-client"))

from main import (
    MockRobot,
    MockPolicy,
    TimingEvent,
    TimingLog,
    InstrumentedPolicy,
    get_modal_endpoint_url,
    compute_stats,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Arguments for the sync vs async+RTC comparison."""

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

    # RTC parameters (for async+RTC mode only)
    execution_horizon: int = 10
    prefix_attention_schedule: str = "LINEAR"
    max_guidance_weight: float = -1.0  # -1 = auto (num_steps). Original RTC paper: 5.0
    sigma_d: float = 0.2  # Prior data std dev. Original RTC paper: 1.0 (Soare optimization: 0.2)
    refill_threshold: float = 0.50

    # Duration per mode (each mode runs for this many seconds)
    duration: float = 30.0

    # Output
    save_prefix: str = "comparison"

    # Logging
    verbose: bool = False


def _build_server_obs_fields(args: Args) -> dict:
    """Build the dict of extra observation fields for Modal servers."""
    fields: dict = {}
    if args.hf_repo_id:
        fields["hf_repo_id"] = args.hf_repo_id
    if args.folder_path:
        fields["folder_path"] = args.folder_path
    if args.config_name:
        fields["config_name"] = args.config_name
    if args.prompt:
        fields["prompt"] = args.prompt
    if args.dataset_repo_id:
        fields["dataset_repo_id"] = args.dataset_repo_id
    if args.stats_json_path:
        fields["stats_json_path"] = args.stats_json_path
    return fields


def run_sync_mode(
    raw_policy,
    robot: MockRobot,
    server_obs_fields: dict,
    action_horizon: int,
    duration: float,
    control_hz: float,
    timing_log: TimingLog,
) -> tuple[list[np.ndarray], float]:
    """Run blocking sync inference: infer -> execute full chunk -> repeat.

    Returns (action_log, t0).
    """
    action_log: list[np.ndarray] = []
    action_interval = 1.0 / control_hz

    t0 = time.monotonic()
    start_time = time.time()

    while (time.time() - start_time) < duration:
        obs = robot.get_observation()
        obs.update(server_obs_fields)

        timing_log.record("inference_start")
        result = raw_policy.infer(obs)
        timing_log.record("inference_end")

        actions = result.get("actions")
        if actions is None:
            logger.error("Sync: no actions returned")
            continue
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        for i in range(len(actions)):
            if (time.time() - start_time) >= duration:
                break
            loop_start = time.perf_counter()

            robot.execute_action(actions[i])
            action_log.append(actions[i].copy())
            timing_log.record("action_executed")

            dt = time.perf_counter() - loop_start
            sleep_time = action_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    return action_log, t0


def run_async_rtc_mode(
    raw_policy,
    robot: MockRobot,
    server_obs_fields: dict,
    action_horizon: int,
    duration: float,
    control_hz: float,
    refill_threshold: float,
    rtc_config: dict,
    timing_log: TimingLog,
) -> tuple[list[np.ndarray], float]:
    """Run async+RTC inference using RTCInferenceManager.

    Returns (action_log, t0).
    """
    from rtc_client import RTCInferenceManager

    action_log: list[np.ndarray] = []
    action_interval = 1.0 / control_hz

    instrumented = InstrumentedPolicy(raw_policy, timing_log)

    def get_obs() -> dict:
        obs = robot.get_observation()
        obs.update(server_obs_fields)
        return obs

    manager = RTCInferenceManager(
        policy=instrumented,
        get_observation_fn=get_obs,
        action_horizon=action_horizon,
        refill_threshold=refill_threshold,
        rtc_enabled=True,
        rtc_config=rtc_config,
    )

    manager.start()
    timing_log.clear()

    t0 = time.monotonic()
    start_time = time.time()

    try:
        while (time.time() - start_time) < duration:
            loop_start = time.perf_counter()

            action = manager.get_action()
            if action is not None:
                robot.execute_action(action)
                action_log.append(action.copy())
                timing_log.record("action_executed")
            else:
                timing_log.record("idle")

            dt = time.perf_counter() - loop_start
            sleep_time = action_interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        manager.stop()

    return action_log, t0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_timeline_on_ax(
    ax,
    events: list[TimingEvent],
    t0: float,
    title: str,
    stats: dict,
) -> None:
    """Draw the dot-plot timeline on a given matplotlib Axes."""
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

    row_labels = ["Action Executed", "Inference Called", "Inference Received"]
    row_y = {label: i for i, label in enumerate(row_labels)}

    if action_ts:
        ax.scatter(
            action_ts, [row_y["Action Executed"]] * len(action_ts),
            color="#2196F3", s=8, alpha=0.7, zorder=3, label="Action Executed",
        )
    if idle_ts:
        ax.scatter(
            idle_ts, [row_y["Action Executed"]] * len(idle_ts),
            color="#F44336", s=8, alpha=0.7, marker="x", zorder=3, label="Idle",
        )
    if inf_start_ts:
        ax.scatter(
            inf_start_ts, [row_y["Inference Called"]] * len(inf_start_ts),
            color="#4CAF50", s=30, alpha=0.9, zorder=3, label="Inference Called",
        )
    if inf_end_ts:
        ax.scatter(
            inf_end_ts, [row_y["Inference Received"]] * len(inf_end_ts),
            color="#FF9800", s=30, alpha=0.9, zorder=3, label="Inference Received",
        )

    for start, end in inference_intervals:
        ax.plot(
            [start, end],
            [row_y["Inference Called"], row_y["Inference Received"]],
            color="#9E9E9E", linewidth=1.5, alpha=0.5, zorder=2,
        )

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
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    avg_inf_ms = stats["total_inference_s"] / max(stats["num_inferences"], 1) * 1000
    subtitle = (
        f"Actions: {stats['num_actions']}  |  "
        f"Idle: {stats['idle_time_s']:.2f}s  |  "
        f"Avg inference: {avg_inf_ms:.0f}ms  |  "
        f"Inferences: {stats['num_inferences']}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)


def plot_timing_comparison(
    sync_events: list[TimingEvent],
    sync_t0: float,
    sync_stats: dict,
    async_events: list[TimingEvent],
    async_t0: float,
    async_stats: dict,
    save_path: str,
) -> None:
    """Generate a two-row timing comparison: sync (top) vs async+RTC (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_sync, ax_async) = plt.subplots(2, 1, figsize=(16, 7), sharex=False)

    _draw_timeline_on_ax(ax_sync, sync_events, sync_t0, "Sync (Blocking)", sync_stats)
    _draw_timeline_on_ax(ax_async, async_events, async_t0, "Async + RTC", async_stats)

    ax_async.set_xlabel("Time (s)")
    fig.suptitle("Timing Comparison: Sync vs Async+RTC", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved timing comparison plot to %s", save_path)
    plt.close(fig)


def plot_action_comparison(
    sync_actions: list[np.ndarray],
    async_actions: list[np.ndarray],
    action_dim: int,
    save_path: str,
) -> None:
    """Generate a grid of per-dimension subplots comparing sync vs async+RTC actions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = math.ceil(action_dim / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    sync_arr = np.array(sync_actions) if sync_actions else np.empty((0, action_dim))
    async_arr = np.array(async_actions) if async_actions else np.empty((0, action_dim))

    for dim in range(action_dim):
        row, col = divmod(dim, ncols)
        ax = axes[row][col]

        if len(sync_arr) > 0:
            ax.plot(sync_arr[:, dim], color="#2196F3", alpha=0.8, linewidth=0.8, label="Sync")
        if len(async_arr) > 0:
            ax.plot(async_arr[:, dim], color="#FF9800", alpha=0.8, linewidth=0.8, label="Async+RTC")

        ax.set_title(f"Dim {dim}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

        if dim == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplots
    for idx in range(action_dim, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Action Values: Sync vs Async+RTC (by step index)", fontsize=13, fontweight="bold")
    fig.supxlabel("Step", fontsize=10)
    fig.supylabel("Action Value", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved action comparison plot to %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    # --- Create policy ---
    if args.use_mock_policy:
        logger.info("Using mock policy (latency=%.0fms)", args.mock_latency_ms)
        raw_policy = MockPolicy(
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
            latency_ms=args.mock_latency_ms,
        )
    else:
        from rtc_client import websocket_policy as _wcp

        if args.modal_app_name:
            ws_url = get_modal_endpoint_url(args.modal_app_name, args.modal_function_name)
            raw_policy = _wcp.WebsocketClientPolicy(host=ws_url, api_key=args.api_key)
        else:
            logger.info("Connecting to server at %s:%s", args.host, args.port)
            raw_policy = _wcp.WebsocketClientPolicy(
                host=args.host, port=args.port, api_key=args.api_key,
            )
        logger.info("Connected. Server metadata: %s", raw_policy.get_server_metadata())

    server_obs_fields = _build_server_obs_fields(args)

    # Save a deterministic initial state so both modes start identically
    initial_state = np.random.randn(args.action_dim).astype(np.float32)

    rtc_config = {
        "enabled": True,
        "execution_horizon": args.execution_horizon,
        "prefix_attention_schedule": args.prefix_attention_schedule,
        "max_guidance_weight": args.max_guidance_weight if args.max_guidance_weight >= 0 else None,
        "sigma_d": args.sigma_d,
    }

    action_interval = 1.0 / args.control_hz

    # --- Warmup (excluded from all stats) ---
    logger.info("Running warmup inference to eliminate cold start...")
    warmup_robot = MockRobot(state_dim=args.action_dim, num_cameras=2)
    warmup_obs = warmup_robot.get_observation()
    warmup_obs.update(server_obs_fields)
    warmup_start = time.monotonic()
    raw_policy.infer(warmup_obs)
    logger.info("Warmup complete in %.1fms", (time.monotonic() - warmup_start) * 1000)

    # =====================================================================
    # MODE 1: Sync (blocking)
    # =====================================================================
    logger.info("=" * 60)
    logger.info("MODE 1: Sync (blocking) — %.1fs", args.duration)
    logger.info("=" * 60)

    sync_robot = MockRobot(state_dim=args.action_dim, num_cameras=2)
    sync_robot._state = initial_state.copy()
    sync_timing_log = TimingLog()

    sync_actions, sync_t0 = run_sync_mode(
        raw_policy=raw_policy,
        robot=sync_robot,
        server_obs_fields=server_obs_fields,
        action_horizon=args.action_horizon,
        duration=args.duration,
        control_hz=args.control_hz,
        timing_log=sync_timing_log,
    )

    sync_stats = compute_stats(sync_timing_log.events, sync_t0, action_interval)
    logger.info("Sync results: %d actions, %d inferences, idle=%.2fs",
                sync_stats["num_actions"], sync_stats["num_inferences"], sync_stats["idle_time_s"])

    # =====================================================================
    # MODE 2: Async + RTC
    # =====================================================================
    logger.info("=" * 60)
    logger.info("MODE 2: Async + RTC — %.1fs", args.duration)
    logger.info("=" * 60)

    async_robot = MockRobot(state_dim=args.action_dim, num_cameras=2)
    async_robot._state = initial_state.copy()
    async_timing_log = TimingLog()

    async_actions, async_t0 = run_async_rtc_mode(
        raw_policy=raw_policy,
        robot=async_robot,
        server_obs_fields=server_obs_fields,
        action_horizon=args.action_horizon,
        duration=args.duration,
        control_hz=args.control_hz,
        refill_threshold=args.refill_threshold,
        rtc_config=rtc_config,
        timing_log=async_timing_log,
    )

    async_stats = compute_stats(async_timing_log.events, async_t0, action_interval)
    logger.info("Async+RTC results: %d actions, %d inferences, idle=%.2fs",
                async_stats["num_actions"], async_stats["num_inferences"], async_stats["idle_time_s"])

    # =====================================================================
    # Summary
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    logger.info("                      %-14s %s", "Sync", "Async+RTC")
    logger.info("  Actions executed:   %-14d %d", sync_stats["num_actions"], async_stats["num_actions"])
    logger.info("  Inferences:         %-14d %d", sync_stats["num_inferences"], async_stats["num_inferences"])
    logger.info("  Idle time:          %-14s %s",
                f"{sync_stats['idle_time_s']:.3f}s", f"{async_stats['idle_time_s']:.3f}s")
    logger.info("  Total infer time:   %-14s %s",
                f"{sync_stats['total_inference_s']:.3f}s", f"{async_stats['total_inference_s']:.3f}s")

    sync_avg_inf = sync_stats["total_inference_s"] / max(sync_stats["num_inferences"], 1) * 1000
    async_avg_inf = async_stats["total_inference_s"] / max(async_stats["num_inferences"], 1) * 1000
    logger.info("  Avg inference:      %-14s %s", f"{sync_avg_inf:.0f}ms", f"{async_avg_inf:.0f}ms")

    sync_hz = sync_stats["num_actions"] / max(sync_stats["async_duration_s"], 0.001)
    async_hz = async_stats["num_actions"] / max(async_stats["async_duration_s"], 0.001)
    logger.info("  Effective Hz:       %-14s %s", f"{sync_hz:.1f}", f"{async_hz:.1f}")
    logger.info("=" * 60)

    # =====================================================================
    # Plots
    # =====================================================================
    timing_path = f"{args.save_prefix}_timing.png"
    actions_path = f"{args.save_prefix}_actions.png"

    plot_timing_comparison(
        sync_events=sync_timing_log.events,
        sync_t0=sync_t0,
        sync_stats=sync_stats,
        async_events=async_timing_log.events,
        async_t0=async_t0,
        async_stats=async_stats,
        save_path=timing_path,
    )

    plot_action_comparison(
        sync_actions=sync_actions,
        async_actions=async_actions,
        action_dim=args.action_dim,
        save_path=actions_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
