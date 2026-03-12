#!/usr/bin/env python3
"""
Programmatically spawn a RunPod GPU pod and start the WebSocket policy server.

Reads configuration from a YAML file (default: ws_server_configs/ws_server_config.yaml)
and secrets from the project .env file (RUNPOD_API_KEY, HF_TOKEN).
Shell environment variables take precedence over .env values.

The pod will:
1. Clone the openpi repository
2. Set up the environment (dependencies, custom transformers)
3. Launch the WebSocket policy server in a tmux session

Usage:
    python launch_ws_server.py
    python launch_ws_server.py --config ws_server_configs/my_config.yaml
"""

import argparse
import os
import pathlib
import sys
import time

import requests
import yaml
from dotenv import load_dotenv

RUNPOD_API_BASE = "https://rest.runpod.io/v1"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Error: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return value


def notify_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return
    base_url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    requests.get(base_url, params=params, timeout=300)


def build_workflow_cmd(cfg: dict, env_var_names: list[str]) -> str:
    """Build the &&-chained shell command that clones, sets up, and starts the WS server."""
    git_repo = cfg["git_repo"]
    git_branch = cfg["git_branch"]
    ws_port = cfg.get("ws_port", 8000)

    serve_cmd = (
        f"python ws_server/serve_policy.py "
        f"--host 0.0.0.0 --port {ws_port}"
    )

    tg_curl = (
        'curl -sf "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage"'
        ' -d chat_id=$TELEGRAM_CHAT_ID'
    )
    serve_with_notify = (
        f"{{ {tg_curl} --data-urlencode 'text=WS server starting on port {ws_port}'; }} 2>/dev/null; "
        f"{{ {serve_cmd}; }}"
        f" || {{ {tg_curl} --data-urlencode 'text=WS server FAILED'; exit 1; }}"
    )

    export_cmds = [f"export {name}=${name}" for name in env_var_names]

    setup_script = cfg.get("setup_script", "")
    setup_step = f"bash {setup_script}" if setup_script else "echo 'No extra setup script'"

    inner_cmds = " && ".join(export_cmds + [
        "cd /workspace",
        f"git clone {git_repo}",
        "cd openpi",
        f"git checkout {git_branch}",
        setup_step,
        serve_with_notify,
    ])

    return inner_cmds


def build_startup_cmd(cfg: dict, env_var_names: list[str]) -> str:
    """Build the dockerStartCmd that writes /post_start.sh and execs /start.sh.

    The generated command writes a /post_start.sh script that installs tmux and
    runs the full workflow inside a detached tmux session named 'ws_server'.
    SSH in and run ``tmux attach -t ws_server`` to monitor.
    """
    workflow = build_workflow_cmd(cfg, env_var_names)

    post_start_body = "\n".join([
        "#!/bin/bash",
        "apt-get update && apt-get install -y tmux",
        "cat > /tmp/ws_workflow.sh << 'WS_WORKFLOW_EOF'",
        "#!/bin/bash",
        workflow,
        "WS_WORKFLOW_EOF",
        "chmod +x /tmp/ws_workflow.sh",
        "tmux new-session -d -s ws_server 'bash /tmp/ws_workflow.sh; exec bash'",
        "exit 0",
    ])

    startup = "\n".join([
        "cat > /post_start.sh << 'POSTSTART'",
        post_start_body,
        "POSTSTART",
        "chmod +x /post_start.sh",
        "exec /start.sh",
    ])

    return startup


def build_pod_payload(cfg: dict, hf_token: str) -> dict:
    env_var_names = ["HF_TOKEN"]

    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if telegram_token and telegram_chat_id:
        env_var_names += ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

    startup_cmd = build_startup_cmd(cfg, env_var_names)

    env = {"HF_TOKEN": hf_token}
    if telegram_token:
        env["TELEGRAM_BOT_TOKEN"] = telegram_token
    if telegram_chat_id:
        env["TELEGRAM_CHAT_ID"] = telegram_chat_id

    ws_port = cfg.get("ws_port", 8000)

    payload = {
        "name": cfg["pod_name"],
        "templateId": cfg["template_id"],
        "gpuTypeIds": [cfg["gpu_type"]],
        "gpuCount": cfg.get("gpu_count", 1),
        "cloudType": cfg.get("cloud_type", "SECURE"),
        "containerDiskInGb": cfg.get("container_disk_gb", 50),
        "volumeInGb": cfg.get("volume_gb", 100),
        "volumeMountPath": cfg.get("volume_mount_path", "/workspace"),
        "ports": f"{ws_port}/http,22/tcp",
        "env": env,
        "dockerStartCmd": ["bash", "-c", startup_cmd],
    }
    return payload


def create_pod(api_key: str, payload: dict) -> dict:
    resp = requests.post(
        f"{RUNPOD_API_BASE}/pods",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if resp.status_code not in (200, 201):
        print(f"Error creating pod: {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        notify_telegram(f"WS server pod creation FAILED (HTTP {resp.status_code})")
        sys.exit(1)
    return resp.json()


def get_pod(api_key: str, pod_id: str) -> dict:
    resp = requests.get(
        f"{RUNPOD_API_BASE}/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return resp.json()


def poll_until_running(api_key: str, pod_id: str, timeout: int = 600) -> dict:
    """Poll pod status until it reaches RUNNING or timeout (seconds)."""
    start = time.time()
    interval = 5
    while time.time() - start < timeout:
        pod = get_pod(api_key, pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")
        runtime_status = pod.get("runtime", {})
        if runtime_status and runtime_status.get("uptimeInSeconds"):
            return pod
        if status == "RUNNING":
            return pod
        print(f"  Pod status: {status} (waiting...)")
        time.sleep(interval)
        interval = min(interval * 1.5, 30)
    print(f"Warning: timed out after {timeout}s waiting for pod to reach RUNNING.", file=sys.stderr)
    return get_pod(api_key, pod_id)


def main():
    parser = argparse.ArgumentParser(description="Launch a RunPod WebSocket policy server")
    parser.add_argument(
        "--config",
        default="ws_server_configs/ws_server_config.yaml",
        help="Path to YAML config file (default: ws_server_configs/ws_server_config.yaml)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: .env next to this script)",
    )
    args = parser.parse_args()

    env_path = args.env_file or (pathlib.Path(__file__).parent / ".env")
    load_dotenv(env_path, override=False)

    cfg = load_config(args.config)

    api_key = require_env("RUNPOD_API_KEY")
    hf_token = require_env("HF_TOKEN")

    payload = build_pod_payload(cfg, hf_token)

    ws_port = cfg.get("ws_port", 8000)
    print(f"Creating RunPod pod '{cfg['pod_name']}' for WS policy server...")
    print(f"  Template:  {cfg['template_id']}")
    print(f"  GPU:       {cfg.get('gpu_count', 1)}x {cfg['gpu_type']}")
    print(f"  Cloud:     {cfg.get('cloud_type', 'SECURE')}")
    print(f"  WS port:   {ws_port}")
    print()

    pod = create_pod(api_key, payload)
    pod_id = pod.get("id", "unknown")
    print(f"Pod created: {pod_id}")
    cost = pod.get("costPerHr") or pod.get("adjustedCostPerHr")
    if cost:
        print(f"  Cost: ${cost}/hr")
    print()

    cost_str = f" (${cost}/hr)" if cost else ""
    notify_telegram(
        f"WS server pod created: {pod_id}{cost_str}\n"
        f"GPU: {cfg.get('gpu_count', 1)}x {cfg['gpu_type']}"
    )

    print("Waiting for pod to start...")
    pod = poll_until_running(api_key, pod_id)

    status = pod.get("desiredStatus", "UNKNOWN")
    print(f"\nPod {pod_id} is now {status}")
    if pod.get("publicIp"):
        print(f"  Public IP: {pod['publicIp']}")
    port_mappings = pod.get("portMappings") or {}
    if port_mappings:
        print(f"  Port mappings: {port_mappings}")
    dashboard = f"https://www.runpod.io/console/pods/{pod_id}"
    print(f"\n  Dashboard: {dashboard}")

    if pod.get("publicIp"):
        print(f"\n  WebSocket URL: ws://{pod['publicIp']}:{ws_port}/ws")
        print(f"  Health check:  http://{pod['publicIp']}:{ws_port}/healthz")

    ip_str = f"\nIP: {pod['publicIp']}" if pod.get("publicIp") else ""
    notify_telegram(
        f"WS server pod {pod_id} is now {status}{ip_str}\n"
        f"Dashboard: {dashboard}"
    )


if __name__ == "__main__":
    main()
