"""Standalone WebSocket server for OpenPI policy serving.

This is a self-contained version of the Modal deployment (modal_serve_policy.py)
that runs as a regular FastAPI/Uvicorn server with no Modal dependency.

KEY FEATURE: Dynamic Model Loading in Inference Loop
=====================================================
Model loading happens during the inference loop (Phase 5), allowing clients to
dynamically switch models by including model specification in any observation message.

The server implements two-level caching to optimize performance:
1. Checkpoint Cache: Avoids re-downloading model files if already on disk
2. Policy Cache: Avoids re-loading and re-compiling models if already in memory

Protocol:
1. Client connects via WebSocket at /ws
2. Server sends empty metadata {}
3. Client sends initialization message (can be empty)
4. Server sends empty metadata {}
5. Client sends observation with model specification and data:
    {
        "hf_repo_id": "username/repo-name",
        "folder_path": "checkpoints/pi05_tcr",
        "config_name": "pi05_tcr",
        "prompt": "optional prompt text",
        "dataset_repo_id": "username/dataset-repo",
        "stats_json_path": "stats.json",
        "model_action_horizon": 25,
        ... other observation data ...
    }
6. Server loads model (if not already loaded) and returns action
7. Subsequent observations can omit model fields to reuse loaded model,
   or include them to switch to a different model

Usage:
    python serve_policy.py
    python serve_policy.py --port 8080
    python serve_policy.py --checkpoint-dir /data/checkpoints
"""

import argparse
import dataclasses
import logging
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import openpi_client.msgpack_numpy as msgpack_numpy
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

_checkpoint_cache: dict[tuple[str, str], Path] = {}
_policy_cache: dict[tuple[str, ...], object] = {}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATASET_DIR = Path(os.environ.get("OPENPI_LOCAL_DATASET_DIR", PROJECT_ROOT / "dataset"))
CHECKPOINT_BASE_DIR = os.environ.get("CHECKPOINT_BASE_DIR", "/inference-checkpoints")


def download_checkpoint(hf_repo_id: str, folder_path: str) -> Path:
    """Download checkpoint from HuggingFace with caching."""
    cache_key = (hf_repo_id, folder_path)

    if cache_key in _checkpoint_cache:
        cached_path = _checkpoint_cache[cache_key]
        if cached_path.exists():
            logger.info(f"Using cached checkpoint from {cached_path}")
            return cached_path
        else:
            logger.warning(f"Cached path {cached_path} gone, re-downloading")
            del _checkpoint_cache[cache_key]

    logger.info(f"Downloading checkpoint from {hf_repo_id}/{folder_path}...")

    checkpoint_base = Path(CHECKPOINT_BASE_DIR)
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    if folder_path and folder_path != "/":
        clean_path = folder_path.strip("/")
        checkpoint_dir = checkpoint_base / clean_path

        if (
            checkpoint_dir.exists()
            and any(checkpoint_dir.iterdir())
            and any(f.name.endswith(".safetensors") for f in checkpoint_dir.iterdir())
        ):
            logger.info(f"Checkpoint directory {checkpoint_dir} exists, skipping download.")
            _checkpoint_cache[cache_key] = checkpoint_dir
            return checkpoint_dir

        cmd = [
            "huggingface-cli", "download", hf_repo_id,
            "--include", f"{clean_path}/*",
            "--local-dir", str(checkpoint_base),
        ]
    else:
        checkpoint_dir = checkpoint_base

        if (
            checkpoint_dir.exists()
            and any(checkpoint_dir.iterdir())
            and any(f.name.endswith(".safetensors") for f in checkpoint_dir.iterdir())
        ):
            logger.info(f"Checkpoint directory {checkpoint_dir} exists, skipping download.")
            _checkpoint_cache[cache_key] = checkpoint_dir
            return checkpoint_dir

        cmd = [
            "huggingface-cli", "download", hf_repo_id,
            "--local-dir", str(checkpoint_base),
        ]

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info(f"Successfully downloaded checkpoint to {checkpoint_dir}")

    for file in os.listdir(checkpoint_dir):
        logger.info(f"[checkpoint_dir] File: {file}")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Folder {folder_path} not found in repo {hf_repo_id}")

    _checkpoint_cache[cache_key] = checkpoint_dir
    return checkpoint_dir


def download_norm_stats(dataset_repo_id: str, stats_json_path: str):
    """Download and load norm_stats from HuggingFace dataset.

    Also saves to the local project dataset directory so config's
    _load_norm_stats can find it for configs that still reference the
    dataset path directly.
    """
    logger.info(f"Downloading norm_stats.json from {dataset_repo_id}/{stats_json_path}...")

    stats_dest_dir = Path("/tmp/norm_stats")
    stats_dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli", "download", dataset_repo_id,
        stats_json_path,
        "--repo-type", "dataset",
        "--local-dir", str(stats_dest_dir),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    norm_stats_file = stats_dest_dir / stats_json_path
    logger.info(f"Loading norm stats from {norm_stats_file}")
    norm_stats = _normalize.deserialize_json(norm_stats_file.read_text())
    logger.info(f"Successfully loaded norm stats with keys: {list(norm_stats.keys())}")

    config_expected_dir = PROJECT_DATASET_DIR
    config_expected_dir.mkdir(parents=True, exist_ok=True)
    config_expected_file = config_expected_dir / "norm_stats.json"
    shutil.copy(norm_stats_file, config_expected_file)
    logger.info(f"Copied norm_stats to {config_expected_file} for config loading")

    return norm_stats


def load_policy(
    hf_repo_id: str,
    folder_path: str,
    config_name: str,
    prompt: str | None = None,
    dataset_repo_id: str | None = None,
    stats_json_path: str | None = None,
    model_action_horizon: int | None = None,
    use_quantile_norm: bool | None = None,
    use_delta_joint_actions: bool | None = None,
):
    """Load a policy with two-level caching (checkpoint + policy)."""
    cache_key = (
        hf_repo_id,
        folder_path,
        config_name,
        prompt or "",
        dataset_repo_id or "",
        stats_json_path or "",
        model_action_horizon or "",
        use_quantile_norm if use_quantile_norm is not None else "",
        use_delta_joint_actions if use_delta_joint_actions is not None else "",
    )

    logger.info(f"Loading new policy: {config_name} from {hf_repo_id}/{folder_path}")

    norm_stats = None
    if dataset_repo_id and stats_json_path:
        try:
            norm_stats = download_norm_stats(dataset_repo_id, stats_json_path)
        except Exception as e:
            logger.error(f"Failed to download/load norm_stats.json: {e}")
            logger.error(traceback.format_exc())
            raise
    else:
        logger.warning(
            "dataset_repo_id and stats_json_path not provided. "
            "For TCR configs (pi05_tcr_*), you MUST provide these parameters "
            "to download norm_stats from HuggingFace."
        )

    if cache_key in _policy_cache:
        logger.info(
            f"Using cached policy for {config_name} from {hf_repo_id}/{folder_path}. "
            f"{len(_policy_cache)} policies cached."
        )
        return _policy_cache[cache_key]

    try:
        checkpoint_dir = download_checkpoint(hf_repo_id, folder_path)
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise

    try:
        logger.info(f"Loading policy from {checkpoint_dir}...")

        logger.info(f"Getting config: {config_name}")
        train_config = _config.get_config(config_name)
        logger.info(f"Using config '{train_config.name}' with pi05={train_config.model.pi05}")

        if model_action_horizon is not None:
            logger.info(
                f"Overriding action_horizon: {train_config.model.action_horizon} -> {model_action_horizon}"
            )
            new_model_cfg = dataclasses.replace(
                train_config.model, action_horizon=model_action_horizon
            )
            train_config = dataclasses.replace(train_config, model=new_model_cfg)

        logger.info("Creating trained policy...")
        policy = _policy_config.create_trained_policy(
            train_config,
            str(checkpoint_dir),
            default_prompt=prompt,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
            use_delta_joint_actions=use_delta_joint_actions,
        )

        _policy_cache[cache_key] = policy
        logger.info(f"Successfully loaded and cached policy {config_name}")
        return policy
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        logger.error(traceback.format_exc())
        raise


app = FastAPI()


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket) -> None:
    """WebSocket handler for policy inference.

    Protocol (follows serve_policy.py / WebsocketPolicyServer pattern):
    1. Client connects
    2. Server sends empty metadata message {}
    3. Client sends initialization message (can be empty or specify model)
    4. Server sends empty metadata (model will be loaded in Phase 5)
    5. Inference loop - client sends observations with model specification:
        - hf_repo_id, folder_path, config_name (required for model loading)
        - prompt, dataset_repo_id, stats_json_path (optional)
        - model_action_horizon (optional int, overrides config's action_horizon)
        - Model is loaded/switched dynamically when these fields are present
        - Server responds with actions
    """
    await websocket.accept()
    logger.info(f"Connection from {websocket.client} opened")

    packer = msgpack_numpy.Packer()
    policy = None
    current_model_key = None

    try:
        # PHASE 1: Send initial empty metadata
        logger.info("PHASE 1: Sending initial empty metadata")
        await websocket.send_bytes(packer.pack({}))

        # PHASE 2: Receive client initialization
        logger.info("PHASE 2: Waiting for client initialization")
        init_data = await websocket.receive_bytes()
        init_message = msgpack_numpy.unpackb(init_data)
        logger.info(f"Received initialization message: {init_message}")

        # PHASE 3: Skip - model loading moved to Phase 5
        logger.info("PHASE 3: Skipped - model will be loaded in inference loop")

        # PHASE 4: Send empty metadata
        logger.info("PHASE 4: Sending empty metadata")
        await websocket.send_bytes(packer.pack({}))

        # PHASE 5: Inference loop with dynamic model loading
        logger.info("PHASE 5: Starting inference loop")
        prev_total_time = None
        step_count = 0
        while True:
            try:
                start_time = time.monotonic()

                logger.info(f"Step {step_count}: Waiting for observation...")
                obs = msgpack_numpy.unpackb(await websocket.receive_bytes())
                logger.info(f"Step {step_count}: Received observation with keys: {list(obs.keys())}")

                rtc_prev_actions = obs.pop("rtc_prev_actions", None)
                rtc_config = obs.pop("rtc_config", None)
                if rtc_prev_actions is not None:
                    obs["_rtc_prev_actions"] = rtc_prev_actions
                if rtc_config is not None:
                    obs["_rtc_config"] = rtc_config

                hf_repo_id = obs.pop("hf_repo_id", None)
                folder_path = obs.pop("folder_path", None)
                config_name = obs.pop("config_name", None)
                prompt = obs.pop("prompt", None)
                dataset_repo_id = obs.pop("dataset_repo_id", None)
                stats_json_path = obs.pop("stats_json_path", None)
                model_action_horizon = obs.pop("model_action_horizon", None)
                if model_action_horizon is not None:
                    model_action_horizon = int(model_action_horizon)
                    if model_action_horizon < 1:
                        raise ValueError(
                            f"model_action_horizon must be a positive integer, got {model_action_horizon}"
                        )
                use_quantile_norm = obs.pop("use_quantile_norm", None)
                if use_quantile_norm is not None:
                    use_quantile_norm = bool(use_quantile_norm)
                use_delta_joint_actions = obs.pop("use_delta_joint_actions", None)
                if use_delta_joint_actions is not None:
                    use_delta_joint_actions = bool(use_delta_joint_actions)

                if hf_repo_id and folder_path and config_name:
                    new_model_key = (
                        hf_repo_id,
                        folder_path,
                        config_name,
                        prompt or "",
                        dataset_repo_id or "",
                        stats_json_path or "",
                        model_action_horizon or "",
                        use_quantile_norm if use_quantile_norm is not None else "",
                        use_delta_joint_actions if use_delta_joint_actions is not None else "",
                    )

                    if current_model_key != new_model_key:
                        logger.info(
                            f"Step {step_count}: Loading model {config_name} from {hf_repo_id}/{folder_path}"
                        )
                        try:
                            policy = load_policy(
                                hf_repo_id,
                                folder_path,
                                config_name,
                                prompt,
                                dataset_repo_id,
                                stats_json_path,
                                model_action_horizon,
                                use_quantile_norm,
                                use_delta_joint_actions,
                            )
                            current_model_key = new_model_key
                            logger.info(f"Step {step_count}: Model loaded successfully")
                        except Exception as e:
                            logger.error(f"Step {step_count}: Failed to load policy: {e}")
                            logger.error(traceback.format_exc())
                            await websocket.send_text(
                                f"Error loading policy:\n{traceback.format_exc()}"
                            )
                            await websocket.close(code=1011, reason="Policy loading failed")
                            return

                if policy is None:
                    error_msg = "No policy loaded. Include hf_repo_id, folder_path, and config_name in observation."
                    logger.error(f"Step {step_count}: {error_msg}")
                    await websocket.send_text(error_msg)
                    await websocket.close(code=1011, reason=error_msg)
                    return

                logger.info(f"Step {step_count}: Running inference...")
                infer_time = time.monotonic()
                try:
                    action = policy.infer(obs)
                    logger.info(f"Step {step_count}: Inference completed, action keys: {list(action.keys())}")
                except Exception as e:
                    logger.error(f"Step {step_count}: Inference failed: {e}")
                    logger.error(traceback.format_exc())
                    raise
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                logger.info(f"Step {step_count}: Sending action back to client...")
                await websocket.send_bytes(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
                logger.info(f"Step {step_count}: Completed in {prev_total_time*1000:.1f}ms")
                step_count += 1

            except WebSocketDisconnect:
                logger.info(f"Connection from {websocket.client} closed (after {step_count} steps)")
                break
            except Exception as e:
                logger.error(f"Error in inference loop at step {step_count}: {e}")
                logger.error(traceback.format_exc())
                try:
                    await websocket.send_text(traceback.format_exc())
                    await websocket.close(
                        code=1011,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except Exception:
                    logger.error("Failed to send error to client")
                raise

    except WebSocketDisconnect:
        logger.info(f"Connection from {websocket.client} closed")
    except Exception:
        logger.error(f"Error in websocket handler:\n{traceback.format_exc()}")
        raise


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone WebSocket policy server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Base directory for downloaded checkpoints (default: /inference-checkpoints or CHECKPOINT_BASE_DIR env)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint_dir:
        CHECKPOINT_BASE_DIR = args.checkpoint_dir
    uvicorn.run(app, host=args.host, port=args.port)
