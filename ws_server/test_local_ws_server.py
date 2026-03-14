#!/usr/bin/env python3
"""Minimal end-to-end test for the local standalone WebSocket server.

Edit the config block below, then run:

    uv run python ws_server/test_local_ws_server.py
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import websockets.sync.client

from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config: edit these values and then run the script with no flags.
# -----------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8888
CHECK_HEALTHZ = True
PRINT_FULL_ACTION = False
RNG_SEED = 7

# GriffinLabs model defaults taken from `examples/rtc_inference/main.py`.
HF_REPO_ID = "griffinlabs/pi05_B017_1877_ckpt"
FOLDER_PATH = "pi05_tcr_full_finetune_pytorch/pi05_B017/4385"
CONFIG_NAME = "pi05_tcr_full_finetune_pytorch"
PROMPT = "Clean the countertop"
DATASET_REPO_ID: str | None = "griffinlabs/B017_dataset"
STATS_JSON_PATH: str | None = "./norm_stats.json"
MODEL_ACTION_HORIZON: int | None = 50

# Random observation shape for the test request.
STATE_DIM = 14
IMAGE_SIZE = 224
CAMERA_NAMES = ("top", "left", "right")


def build_http_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def build_ws_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/ws"


def check_healthz(host: str, port: int) -> float:
    url = build_http_url(host, port, "/healthz")
    logger.info("Checking %s", url)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Health check failed for {url}: {exc}") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    if payload.get("status") != "ok":
        raise RuntimeError(f"Unexpected /healthz response from {url}: {payload}")

    logger.info("Health check passed in %.1f ms: %s", elapsed_ms, payload)
    return elapsed_ms


def build_model_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {
        "hf_repo_id": HF_REPO_ID,
        "folder_path": FOLDER_PATH,
        "config_name": CONFIG_NAME,
        "prompt": PROMPT,
    }
    if DATASET_REPO_ID:
        fields["dataset_repo_id"] = DATASET_REPO_ID
    if STATS_JSON_PATH:
        fields["stats_json_path"] = STATS_JSON_PATH
    if MODEL_ACTION_HORIZON is not None:
        fields["model_action_horizon"] = MODEL_ACTION_HORIZON
    return fields


def random_observation() -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    return {
        "state": rng.standard_normal(STATE_DIM, dtype=np.float32),
        "images": {
            camera_name: rng.integers(
                0,
                256,
                size=(3, IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.uint8,
            )
            for camera_name in CAMERA_NAMES
        },
        "prompt": PROMPT,
    }


def summarize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {key: summarize_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_value(item) for item in value]
    return value


def run_inference_test() -> dict[str, Any]:
    ws_url = build_ws_url(HOST, PORT)
    logger.info("Connecting to %s", ws_url)

    packer = msgpack_numpy.Packer()
    model_fields = build_model_fields()
    observation = random_observation()

    connect_start = time.perf_counter()
    with websockets.sync.client.connect(
        ws_url,
        compression=None,
        max_size=None,
        open_timeout=None,
        close_timeout=None,
    ) as ws:
        connect_ms = (time.perf_counter() - connect_start) * 1000

        initial_metadata_raw = ws.recv()
        initial_metadata = msgpack_numpy.unpackb(initial_metadata_raw)
        logger.info("Initial metadata: %s", initial_metadata)

        init_start = time.perf_counter()
        ws.send(packer.pack(model_fields))
        policy_metadata_raw = ws.recv()
        init_roundtrip_ms = (time.perf_counter() - init_start) * 1000
        if isinstance(policy_metadata_raw, str):
            raise RuntimeError(f"Server returned an initialization error:\n{policy_metadata_raw}")
        policy_metadata = msgpack_numpy.unpackb(policy_metadata_raw)
        logger.info("Policy metadata: %s", policy_metadata)

        # This server loads the model in the inference loop, so repeat the model
        # fields inside the observation payload as the minimal working example.
        observation.update(model_fields)
        inference_start = time.perf_counter()
        ws.send(packer.pack(observation))
        action_raw = ws.recv()
        inference_roundtrip_ms = (time.perf_counter() - inference_start) * 1000
        if isinstance(action_raw, str):
            raise RuntimeError(f"Server returned an inference error:\n{action_raw}")
        action = msgpack_numpy.unpackb(action_raw)

    action_summary = {
        "keys": sorted(action.keys()),
        "server_timing": summarize_value(action.get("server_timing", {})),
    }
    for preferred_key in ("actions", "action", "terminate_episode"):
        if preferred_key in action:
            action_summary[preferred_key] = summarize_value(action[preferred_key])

    result = {
        "server": {
            "healthz_url": build_http_url(HOST, PORT, "/healthz"),
            "ws_url": ws_url,
        },
        "model": summarize_value(model_fields),
        "timing_ms": {
            "ws_connect_ms": connect_ms,
            "init_roundtrip_ms": init_roundtrip_ms,
            "inference_roundtrip_ms": inference_roundtrip_ms,
        },
        "policy_metadata": summarize_value(policy_metadata),
        "action_summary": action_summary,
    }

    if PRINT_FULL_ACTION:
        result["full_action"] = summarize_value(action)

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result: dict[str, Any] = {}
    if CHECK_HEALTHZ:
        result["timing_ms"] = {
            "healthz_ms": check_healthz(HOST, PORT),
        }

    inference_result = run_inference_test()
    if "timing_ms" in result:
        inference_result["timing_ms"] = {
            **result["timing_ms"],
            **inference_result["timing_ms"],
        }

    print(json.dumps(inference_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
