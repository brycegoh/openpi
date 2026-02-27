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

while True:
    try:
        snapshot_download(
            repo_id="griffinlabs/pi05_pytorch_evaluated_base", local_dir="/workspace/base_checkpoints/pi05_base_pytorch", repo_type="model"
        )
        break
    except Exception as e:
        print(f"Error: {e}, retrying...")