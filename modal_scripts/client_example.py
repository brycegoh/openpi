from drtc.drtc_modal_client import DRTCModalClient, DRTCConfig

# ============================================================
# CONFIGURATION
# ============================================================

MODAL_APP_NAME = "openpi-policy-server-3"
MODAL_FUNCTION_NAME = "endpoint"

HF_REPO_ID = "griffinlabs/pi05_412ep_pytorch"
FOLDER_PATH = "pi05_tcr_full_finetune_pytorch/pi05_412ep/20000"
CONFIG_NAME = "pi05_tcr_full_finetune_pytorch"
PROMPT = "pick up the object"

DATASET_REPO_ID = "griffinlabs/tcr-data"
STATS_JSON_PATH = "./norm_stats.json"

# DRTC-specific config
drtc_config = DRTCConfig(
    action_horizon=50,     # H: actions per chunk from the model
    s_min=14,              # minimum execution horizon
    epsilon=1,             # cooldown buffer
    fps=50,                # control loop frequency
    rtc_enabled=True,      # enable RTC inpainting guidance
    latency_estimator_type="jk",
)

# ============================================================
# DRTC ASYNC USAGE
# ============================================================

client = DRTCModalClient(
    app_name=MODAL_APP_NAME,
    function_name=MODAL_FUNCTION_NAME,
    hf_repo_id=HF_REPO_ID,
    folder_path=FOLDER_PATH,
    config_name=CONFIG_NAME,
    prompt=PROMPT,
    dataset_repo_id=DATASET_REPO_ID,
    stats_json_path=STATS_JSON_PATH,
    config=drtc_config,
)

# Connect and start background threads
client.start()

import time

for step in range(1000):
    obs = {}  # your observation dict (state, images, etc.)

    result = client.infer(obs)

    action = result["actions"]       # np.ndarray or None during warmup
    starved = result.get("starved")  # True if no action was available

    if action is not None:
        pass  # robot.send_action(action)

    # Rate-limit to match the configured fps
    time.sleep(drtc_config.environment_dt)

# Clean shutdown
client.stop()