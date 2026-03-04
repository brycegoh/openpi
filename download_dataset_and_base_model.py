from huggingface_hub import snapshot_download
import os

while True:
    try:
        snapshot_download(
            repo_id=os.environ.get("DATASET_REPO_ID"), local_dir="/workspace/dataset", repo_type="dataset"
        )
        break
    except Exception as e:
        print(f"Error: {e}, retrying...")

base_model_kwargs = {}
base_model_repo_path = os.environ.get("BASE_MODEL_REPO_PATH", "")
if base_model_repo_path:
    base_model_kwargs["allow_patterns"] = [f"{base_model_repo_path}/*", f"{base_model_repo_path}/**"]

while True:
    try:
        snapshot_download(
            repo_id=os.environ.get("BASE_MODEL_REPO_ID"),
            local_dir="/workspace/base_checkpoints",
            repo_type="model",
            **base_model_kwargs,
        )
        break
    except Exception as e:
        print(f"Error: {e}, retrying...")