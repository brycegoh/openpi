"""Example: Continuous RTC inference with a mock robot.

Demonstrates the RTCInferenceManager that asynchronously fetches action chunks
from a policy server while a mock robot continuously executes actions.

The key idea is threshold-based pre-fetching: when the action queue drops below
25% of the action horizon, a new inference call is made. This ensures either:
  - Continuous actions when inference is faster than execution
  - Minimized idle time when inference is slower than execution

For RTC mode, the leftover actions from the previous chunk are sent to the
server so it can use inpainting to maintain temporal consistency across chunks.

Usage (connect to a running policy server):
    python main.py --host ws://your-server-url/ws --action-horizon 50

Usage (with a mock policy for local testing without a server):
    python main.py --use-mock-policy --action-horizon 50

Usage (RTC disabled, simple sequential chunking):
    python main.py --host ws://your-server-url/ws --no-rtc
"""

import dataclasses
import logging
import time

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
        camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
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
class Args:
    """Arguments for the RTC inference example."""

    # Server connection
    host: str = "0.0.0.0"
    port: int | None = 8000
    api_key: str | None = None

    # Use a mock policy instead of connecting to a server
    use_mock_policy: bool = False
    mock_latency_ms: float = 200.0

    # Action parameters
    action_horizon: int = 50
    action_dim: int = 14
    control_hz: float = 50.0

    # RTC parameters
    rtc: bool = True
    execution_horizon: int = 10
    prefix_attention_schedule: str = "LINEAR"
    max_guidance_weight: float = 5.0
    refill_threshold: float = 0.25

    # Run duration
    duration: float = 30.0

    # Logging
    verbose: bool = False


def main(args: Args) -> None:
    from openpi_client.rtc_client import RTCInferenceManager

    robot = MockRobot(state_dim=args.action_dim, num_cameras=2)

    if args.use_mock_policy:
        logger.info("Using mock policy (latency=%.0fms)", args.mock_latency_ms)
        policy = MockPolicy(
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
            latency_ms=args.mock_latency_ms,
        )
    else:
        from openpi_client import websocket_client_policy as _wcp
        logger.info("Connecting to server at %s:%s", args.host, args.port)
        policy = _wcp.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        logger.info("Connected. Server metadata: %s", policy.get_server_metadata())

    rtc_config = {
        "enabled": args.rtc,
        "execution_horizon": args.execution_horizon,
        "prefix_attention_schedule": args.prefix_attention_schedule,
        "max_guidance_weight": args.max_guidance_weight,
    }

    manager = RTCInferenceManager(
        policy=policy,
        get_observation_fn=robot.get_observation,
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

    action_interval = 1.0 / args.control_hz
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
            else:
                actions_missed += 1

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

    logger.info("=" * 60)
    logger.info("Run Summary")
    logger.info("=" * 60)
    logger.info("  Duration: %.1fs", elapsed)
    logger.info("  Total control steps: %d", total_steps)
    logger.info("  Actions executed: %d", actions_executed)
    logger.info("  Actions missed (empty queue): %d", actions_missed)
    logger.info("  Hit rate: %.1f%%", hit_rate)
    logger.info("  Effective Hz: %.1f", actions_executed / elapsed)
    logger.info("  Inferences: %d", manager._inference_count)
    if manager._inference_count > 0:
        avg_infer_ms = (manager._total_inference_time / manager._inference_count) * 1000
        logger.info("  Avg inference time: %.1fms", avg_infer_ms)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
