"""Async RTC inference example with a mock robot.

Demonstrates how to use ``RTCAsyncInferenceClient`` to run model inference in
the background while executing actions in a continuous control loop.  This
decouples inference latency from the control frequency so that:

* When inference is **faster** than action execution, fresh action chunks
  queue up and the robot always has actions available.
* When inference is **slower** than action execution, the client sends the
  next observation *before* the current chunk runs out, minimising gaps.

The example ships with a built-in ``--mock`` mode that simulates server
latency locally so you can test the pipeline without a deployed endpoint.

Usage
-----

With a deployed Modal server::

    python examples/async_rtc_example.py \\
        --server-url https://YOUR_ORG--openpi-policy-server-3-endpoint.modal.run \\
        --hf-repo-id your-user/your-model \\
        --folder-path checkpoints/pi05_rtc \\
        --config-name pi05_rtc

Local mock mode (no server required)::

    python examples/async_rtc_example.py --mock
    python examples/async_rtc_example.py --mock --mock-latency 0.8
    python examples/async_rtc_example.py --mock --mock-latency 0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "modal_scripts"))
from async_rtc_client import RTCAsyncInferenceClient  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock robot
# ---------------------------------------------------------------------------

class MockRobot:
    """Simulated robot that produces random observations and logs actions."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 7,
        image_shape: tuple[int, ...] = (224, 224, 3),
        num_cameras: int = 1,
        prompt: str = "pick up the red block",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_shape = image_shape
        self.num_cameras = num_cameras
        self.prompt = prompt

        self.state = np.zeros(state_dim, dtype=np.float64)
        self.steps_executed = 0

    def get_observation(self) -> dict:
        obs: dict = {
            "observation/state": self.state.copy(),
            "prompt": self.prompt,
        }
        for i in range(self.num_cameras):
            obs[f"observation/image_{i}"] = np.random.randint(
                0, 256, size=self.image_shape, dtype=np.uint8
            )
        return obs

    def execute_action(self, action: np.ndarray) -> None:
        n = min(len(action), self.state_dim)
        self.state[:n] += action[:n] * 0.01
        self.steps_executed += 1


# ---------------------------------------------------------------------------
# Mock inference function (for --mock mode)
# ---------------------------------------------------------------------------

def make_mock_infer_fn(
    action_dim: int = 7,
    chunk_size: int = 50,
    latency: float = 0.3,
) -> callable:
    """Return a callable that fakes server inference with configurable latency."""

    def _mock_infer(obs_with_config: dict) -> dict:
        time.sleep(latency)
        actions = np.random.randn(chunk_size, action_dim).astype(np.float32) * 0.1
        return {
            "actions": actions,
            "state": obs_with_config.get("observation/state", np.zeros(action_dim)),
            "server_timing": {"infer_ms": latency * 1000},
        }

    return _mock_infer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async RTC inference example with a mock robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--server-url",
        default="https://YOUR_ORG--openpi-policy-server-3-endpoint.modal.run",
        help="Base URL of the deployed Modal inference server.",
    )
    p.add_argument("--hf-repo-id", default="", help="HuggingFace repo ID for the model.")
    p.add_argument("--folder-path", default="", help="Checkpoint folder path in the repo.")
    p.add_argument("--config-name", default="", help="Training config name.")
    p.add_argument("--dataset-repo-id", default=None, help="HF dataset repo for norm stats.")
    p.add_argument("--stats-json-path", default=None, help="Path to stats.json in dataset repo.")
    p.add_argument("--prompt", default="pick up the red block", help="Task prompt.")

    p.add_argument("--control-freq", type=float, default=10.0, help="Control frequency (Hz).")
    p.add_argument("--action-horizon", type=int, default=None, help="Actions to use per chunk.")
    p.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds.")

    p.add_argument("--mock", action="store_true", help="Use simulated inference (no server).")
    p.add_argument("--mock-latency", type=float, default=0.3, help="Simulated inference latency (s).")
    p.add_argument("--mock-chunk-size", type=int, default=50, help="Simulated action chunk size.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    robot = MockRobot(prompt=args.prompt)
    action_dim = robot.action_dim
    control_period = 1.0 / args.control_freq

    model_config: dict = {}
    infer_fn = None

    if args.mock:
        logger.info(
            "Running in MOCK mode  (latency=%.2fs, chunk_size=%d)",
            args.mock_latency, args.mock_chunk_size,
        )
        infer_fn = make_mock_infer_fn(
            action_dim=action_dim,
            chunk_size=args.mock_chunk_size,
            latency=args.mock_latency,
        )
    else:
        if not all([args.hf_repo_id, args.folder_path, args.config_name]):
            logger.error(
                "When not using --mock, you must provide "
                "--hf-repo-id, --folder-path, and --config-name"
            )
            sys.exit(1)

        model_config = {
            "hf_repo_id": args.hf_repo_id,
            "folder_path": args.folder_path,
            "config_name": args.config_name,
        }
        if args.dataset_repo_id:
            model_config["dataset_repo_id"] = args.dataset_repo_id
        if args.stats_json_path:
            model_config["stats_json_path"] = args.stats_json_path

    with RTCAsyncInferenceClient(
        server_url=args.server_url,
        model_config=model_config,
        control_freq=args.control_freq,
        action_horizon=args.action_horizon,
        infer_fn=infer_fn,
    ) as client:

        # -- Warmup: block until the first action chunk is ready. -----------
        logger.info("Warming up (waiting for first inference)...")
        obs = robot.get_observation()
        first_action = client.warmup(obs)
        robot.execute_action(first_action)
        logger.info("Warmup done. Entering control loop.")

        # -- Main control loop ---------------------------------------------
        start_time = time.monotonic()
        loop_count = 0

        try:
            while True:
                loop_start = time.monotonic()
                elapsed_total = loop_start - start_time

                if elapsed_total >= args.duration:
                    break

                obs = robot.get_observation()
                action = client.step(obs)

                if action is not None:
                    robot.execute_action(action)
                else:
                    logger.debug("Buffer empty -- holding last action")

                loop_count += 1
                if loop_count % int(args.control_freq * 5) == 0:
                    stats = client.get_stats()
                    logger.info(
                        "[%.1fs] steps=%d  buf=%d  chunks=%d  "
                        "avg_lat=%.0fms  empty=%d",
                        elapsed_total,
                        robot.steps_executed,
                        stats["buffer_size"],
                        stats["chunks_received"],
                        stats["avg_latency_ms"],
                        stats["empty_buffer_hits"],
                    )

                # Maintain control frequency
                dt = time.monotonic() - loop_start
                sleep_time = control_period - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        # -- Summary -------------------------------------------------------
        stats = client.get_stats()
        total_time = time.monotonic() - start_time
        logger.info("=" * 60)
        logger.info("Run finished in %.1fs", total_time)
        logger.info("Robot steps executed: %d", robot.steps_executed)
        logger.info("Action chunks received: %d", stats["chunks_received"])
        logger.info("Actions returned: %d", stats["actions_returned"])
        logger.info("Actions skipped (stale): %d", stats["actions_skipped"])
        logger.info("Empty-buffer hits: %d", stats["empty_buffer_hits"])
        logger.info(
            "Inference latency: avg=%.0fms  last=%.0fms",
            stats["avg_latency_ms"],
            stats["last_latency_ms"],
        )
        logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
