"""Test script for the Modal policy server endpoint.

Sends random observations (images + state) and verifies the server returns
valid action chunks. Tests both sync and DRTC async modes.

Usage:
    # Test sync mode (default):
    python test_endpoint.py

    # Test DRTC async mode:
    python test_endpoint.py --drtc

    # Custom model / number of steps:
    python test_endpoint.py --steps 5 --config pi05_tcr_full_finetune_pytorch
"""

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_endpoint")

# ============================================================
# Configuration defaults (override via CLI args)
# ============================================================
DEFAULTS = dict(
    app_name="openpi-policy-server-rtc-1",
    function_name="endpoint",
    hf_repo_id="griffinlabs/pi05_412ep_pytorch",
    folder_path="pi05_tcr_full_finetune_pytorch/pi05_412ep/20000",
    config_name="pi05_tcr_full_finetune_pytorch",
    prompt="pick up the object",
    dataset_repo_id="griffinlabs/tcr-data",
    stats_json_path="./norm_stats.json",
)

IMG_H, IMG_W = 480, 640
STATE_DIM = 14
NUM_CAMERAS = 3  # top, left, right


def make_random_obs(prompt: str = "pick up the object") -> dict:
    """Generate a random observation matching TCR 3-camera format."""
    return {
        "images": {
            "top": np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "left": np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "right": np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8),
        },
        "state": np.random.randn(STATE_DIM).astype(np.float32),
        "prompt": prompt,
    }


def test_sync(args):
    """Test the synchronous (legacy) inference mode."""
    from modal_client import ModalClientPolicy

    logger.info("=== SYNC MODE TEST ===")
    logger.info(f"Connecting to {args.app_name}/{args.function_name}...")

    client = ModalClientPolicy(
        app_name=args.app_name,
        function_name=args.function_name,
        hf_repo_id=args.hf_repo_id,
        folder_path=args.folder_path,
        config_name=args.config_name,
        prompt=args.prompt,
        dataset_repo_id=args.dataset_repo_id,
        stats_json_path=args.stats_json_path,
    )

    logger.info(f"Connected. Running {args.steps} inference steps...")

    latencies = []
    for step in range(args.steps):
        obs = make_random_obs(args.prompt)

        t0 = time.monotonic()
        result = client.infer(obs)
        elapsed_ms = (time.monotonic() - t0) * 1000
        latencies.append(elapsed_ms)

        actions = result.get("actions")
        timing = result.get("server_timing", {})

        if actions is not None:
            actions = np.asarray(actions)
            logger.info(
                f"Step {step}: actions shape={actions.shape}, "
                f"server_infer={timing.get('infer_ms', 0):.0f}ms, "
                f"round_trip={elapsed_ms:.0f}ms"
            )
        else:
            logger.warning(f"Step {step}: No actions returned! keys={list(result.keys())}")

    client.close()

    latencies = np.array(latencies)
    logger.info("--- Sync Results ---")
    logger.info(f"  Steps:  {len(latencies)}")
    logger.info(f"  Mean:   {latencies.mean():.0f} ms")
    logger.info(f"  Median: {np.median(latencies):.0f} ms")
    logger.info(f"  Min:    {latencies.min():.0f} ms")
    logger.info(f"  Max:    {latencies.max():.0f} ms")
    if len(latencies) > 1:
        logger.info(f"  Mean (excl. first): {latencies[1:].mean():.0f} ms")
    logger.info("=== SYNC TEST DONE ===")


def test_drtc(args):
    """Test the DRTC async inference mode."""
    from drtc.drtc_modal_client import DRTCModalClient, DRTCConfig

    logger.info("=== DRTC ASYNC MODE TEST ===")

    drtc_config = DRTCConfig(
        action_horizon=args.action_horizon,
        s_min=14,
        epsilon=1,
        fps=args.fps,
        rtc_enabled=args.rtc,
    )

    client = DRTCModalClient(
        app_name=args.app_name,
        function_name=args.function_name,
        hf_repo_id=args.hf_repo_id,
        folder_path=args.folder_path,
        config_name=args.config_name,
        prompt=args.prompt,
        dataset_repo_id=args.dataset_repo_id,
        stats_json_path=args.stats_json_path,
        config=drtc_config,
    )

    logger.info("Starting DRTC client...")
    ok = client.start()
    if not ok:
        logger.error("Failed to start DRTC client")
        return

    dt = drtc_config.environment_dt
    actions_received = 0
    starved_count = 0

    logger.info(f"Running {args.steps} control steps at {args.fps} Hz (dt={dt*1000:.0f}ms)...")

    for step in range(args.steps):
        obs = make_random_obs(args.prompt)

        t0 = time.monotonic()
        result = client.infer(obs)
        infer_ms = (time.monotonic() - t0) * 1000

        action = result.get("actions")
        sched_size = result.get("schedule_size", 0)
        latency_steps = result.get("latency_steps", 0)
        cooldown = result.get("cooldown", 0)
        starved = result.get("starved", False)

        if action is not None:
            actions_received += 1
        if starved:
            starved_count += 1

        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            action_shape = np.asarray(action).shape if action is not None else None
            logger.info(
                f"Step {step}: action={action_shape}, "
                f"sched={sched_size}, lat_steps={latency_steps}, "
                f"cd={cooldown}, starved={starved}, infer_ms={infer_ms:.1f}"
            )

        # Rate-limit to target fps
        elapsed = time.monotonic() - t0
        sleep_time = max(0, dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    client.stop()

    logger.info("--- DRTC Results ---")
    logger.info(f"  Steps:           {args.steps}")
    logger.info(f"  Actions received: {actions_received}/{args.steps}")
    logger.info(f"  Starvations:     {starved_count}/{args.steps}")
    logger.info(f"  Final latency:   {client.latency_estimator.estimate_steps} steps "
                f"({client.latency_estimator.estimate_seconds:.3f}s)")
    logger.info("=== DRTC TEST DONE ===")


def main():
    parser = argparse.ArgumentParser(description="Test Modal policy server endpoint")
    parser.add_argument("--drtc", action="store_true", help="Test DRTC async mode instead of sync")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
    parser.add_argument("--app-name", default=DEFAULTS["app_name"])
    parser.add_argument("--function-name", default=DEFAULTS["function_name"])
    parser.add_argument("--hf-repo-id", default=DEFAULTS["hf_repo_id"])
    parser.add_argument("--folder-path", default=DEFAULTS["folder_path"])
    parser.add_argument("--config-name", default=DEFAULTS["config_name"])
    parser.add_argument("--prompt", default=DEFAULTS["prompt"])
    parser.add_argument("--dataset-repo-id", default=DEFAULTS["dataset_repo_id"])
    parser.add_argument("--stats-json-path", default=DEFAULTS["stats_json_path"])
    # DRTC-specific
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--rtc", action="store_true", help="Enable RTC guidance (DRTC mode only)")

    args = parser.parse_args()

    if args.drtc:
        test_drtc(args)
    else:
        test_sync(args)


if __name__ == "__main__":
    main()
