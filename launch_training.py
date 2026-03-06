#!/usr/bin/env python3
"""
Programmatically spawn a RunPod GPU pod and start a training job.

Reads all configuration from a YAML file (default: training_config.yaml)
and secrets from the project .env file (RUNPOD_API_KEY, HF_TOKEN, WANDB_API_KEY,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
Shell environment variables take precedence over .env values.

Usage:
    python launch_training.py                        # uses training_config.yaml
    python launch_training.py --config my_run.yaml   # uses a custom config
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
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return
    base_url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    resp = requests.get(base_url, params=params, timeout=300)
    if resp.status_code != 200:
        return


def build_workflow_cmd(
    cfg: dict, env_var_names: list[str], stop_after_training: bool = False,
) -> str:
    """Build the &&-chained shell command for the full setup + training workflow."""
    git_repo = cfg["git_repo"]
    git_branch = cfg["git_branch"]
    dataset_repo = cfg["dataset_repo"]
    base_model_repo = cfg["base_model_repo"]
    base_model_repo_path = cfg.get("base_model_repo_path", "")
    num_gpus = cfg["num_gpus"]
    exp_name = cfg["exp_name"]
    checkpoint_base_dir = cfg["checkpoint_base_dir"]
    base_checkpoint = cfg["base_checkpoint"]
    batch_size = cfg["batch_size"]
    save_interval = cfg["save_interval"]
    keep_period = cfg["keep_period"]
    num_train_steps = cfg["num_train_steps"]
    action_horizon = cfg["action_horizon"]
    train_config_name = cfg.get("train_config_name", "pi05_tcr_full_finetune_pytorch")
    wandb_flag = "--wandb-enabled" if cfg.get("wandb_enabled", True) else ""

    weight_path = base_checkpoint
    if base_model_repo_path:
        weight_path = f"{base_checkpoint}/{base_model_repo_path}"

    ttac_args = ""
    ttac = cfg.get("ttac")
    if ttac and ttac.get("enabled", False):
        ttac_args = (
            f" --model.ttac-config.enabled True"
            f" --model.ttac-config.min-delay {ttac['min_delay']}"
            f" --model.ttac-config.max-delay {ttac['max_delay']}"
            f" --model.ttac-config.delay-distribution {ttac.get('delay_distribution', 'UNIFORM')}"
            f" --model.ttac-config.exp-decay {ttac.get('exp_decay', 1.0)}"
        )

    train_cmd = (
        f"uv run torchrun --standalone --nnodes=1 --nproc_per_node={num_gpus} "
        f"scripts/train_pytorch.py {train_config_name} "
        f"--exp_name {exp_name} "
        f"--checkpoint-base-dir {checkpoint_base_dir} "
        f"--data.repo-id /workspace/dataset "
        f"--pytorch_weight_path {weight_path} "
        f"{wandb_flag} "
        f"--batch-size {batch_size} "
        f"--save-interval {save_interval} "
        f"--keep-period {keep_period} "
        f"--num-train-steps {num_train_steps} "
        f"--model.action-horizon {action_horizon}"
        f"{ttac_args}"
    )

    tg_curl = (
        'curl -sf "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage"'
        ' -d chat_id=$TELEGRAM_CHAT_ID'
    )
    train_with_notify = (
        f"{{ {train_cmd} && {tg_curl} --data-urlencode 'text=Training {exp_name} completed'; }}"
        f" || {{ {tg_curl} --data-urlencode 'text=Training {exp_name} FAILED'; exit 1; }}"
    )

    if stop_after_training:
        train_with_notify += (
            " ; curl -s -X POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop "
            '-H "Authorization: Bearer $RUNPOD_API_KEY" || true'
        )

    export_cmds = [f"export {name}=${name}" for name in env_var_names]

    inner_cmds = " && ".join(export_cmds + [
        "cd /workspace",
        f"git clone {git_repo}",
        "cd openpi",
        f"git checkout {git_branch}",
        f'bash setup.sh {dataset_repo} {base_model_repo} "{base_model_repo_path}"',
        train_with_notify,
    ])

    return inner_cmds


def build_startup_cmd(
    cfg: dict, env_var_names: list[str], stop_after_training: bool = False,
) -> str:
    """Build the dockerStartCmd that writes /post_start.sh and execs /start.sh.

    The generated command writes a /post_start.sh script that installs tmux and
    runs the full workflow (clone, setup, training) inside a detached tmux
    session named 'train'.  The template's /start.sh entrypoint is preserved,
    providing SSH, Jupyter, nginx, and env export.

    SSH in and run ``tmux attach -t train`` to monitor progress.
    """
    workflow = build_workflow_cmd(cfg, env_var_names, stop_after_training)

    # /post_start.sh is called by /start.sh after nginx, SSH, Jupyter and
    # env-export are already set up.  We install tmux, write the workflow
    # to a helper script (avoiding nested quoting issues), and launch it
    # in a detached tmux session.  ``exit 0`` ensures /start.sh is never
    # killed by set -e even if something above fails.
    post_start_body = "\n".join([
        "#!/bin/bash",
        "apt-get update && apt-get install -y tmux",
        "cat > /tmp/train_workflow.sh << 'TRAIN_WORKFLOW_EOF'",
        "#!/bin/bash",
        workflow,
        "TRAIN_WORKFLOW_EOF",
        "chmod +x /tmp/train_workflow.sh",
        "tmux new-session -d -s train 'bash /tmp/train_workflow.sh; exec bash'",
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


def build_pod_payload(
    cfg: dict, hf_token: str, wandb_api_key: str,
    telegram_token: str, telegram_chat_id: str,
) -> dict:
    env_var_names = ["HF_TOKEN", "WANDB_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    stop_after = cfg.get("stop_after_training", False)
    if stop_after:
        env_var_names.append("RUNPOD_API_KEY")

    startup_cmd = build_startup_cmd(
        cfg, env_var_names, stop_after_training=stop_after,
    )

    payload = {
        "name": cfg["pod_name"],
        "templateId": cfg["template_id"],
        "gpuTypeIds": [cfg["gpu_type"]],
        "gpuCount": cfg["gpu_count"],
        "cloudType": cfg["cloud_type"],
        "containerDiskInGb": cfg["container_disk_gb"],
        "volumeInGb": cfg["volume_gb"],
        "volumeMountPath": cfg["volume_mount_path"],
        "env": {
            "HF_TOKEN": hf_token,
            "WANDB_API_KEY": wandb_api_key,
            "TELEGRAM_BOT_TOKEN": telegram_token,
            "TELEGRAM_CHAT_ID": telegram_chat_id,
        },
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
        notify_telegram(f"Pod creation FAILED (HTTP {resp.status_code})")
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
    parser = argparse.ArgumentParser(description="Launch a RunPod training job")
    parser.add_argument(
        "--config",
        default="training_configs/training_config.yaml",
        help="Path to YAML config file (default: training_config.yaml)",
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
    wandb_api_key = require_env("WANDB_API_KEY")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    payload = build_pod_payload(cfg, hf_token, wandb_api_key, telegram_token, telegram_chat_id)

    train_config_name = cfg.get("train_config_name", "pi05_tcr_full_finetune_pytorch")
    print(f"Creating RunPod pod '{cfg['pod_name']}'...")
    print(f"  Template:  {cfg['template_id']}")
    print(f"  GPU:       {cfg['gpu_count']}x {cfg['gpu_type']}")
    print(f"  Cloud:     {cfg['cloud_type']}")
    print(f"  Config:    {train_config_name}")
    print(f"  Dataset:   {cfg['dataset_repo']}")
    print(f"  Base model:{cfg['base_model_repo']}")
    if cfg.get("base_model_repo_path"):
        print(f"  Model path:{cfg['base_model_repo_path']}")
    print(f"  Exp name:  {cfg['exp_name']}")
    print(f"  Steps:     {cfg['num_train_steps']}")
    print(f"  Batch:     {cfg['batch_size']}")
    print(f"  Horizon:   {cfg['action_horizon']}")
    print(f"  Save every {cfg['save_interval']} steps, keep every {cfg['keep_period']} steps")
    ttac = cfg.get("ttac")
    if ttac and ttac.get("enabled", False):
        print(f"  TTAC:      enabled, delay=[{ttac['min_delay']}, {ttac['max_delay']}], "
              f"distribution={ttac.get('delay_distribution', 'UNIFORM')}")
    else:
        print(f"  TTAC:      disabled")
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
        f"Pod created: {pod_id}{cost_str}\n"
        f"Exp: {cfg['exp_name']} | {cfg['gpu_count']}x {cfg['gpu_type']}"
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

    ip_str = f"\nIP: {pod['publicIp']}" if pod.get("publicIp") else ""
    notify_telegram(
        f"Pod {pod_id} is now {status}{ip_str}\n"
        f"Dashboard: {dashboard}"
    )


if __name__ == "__main__":
    main()
