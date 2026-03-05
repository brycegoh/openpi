"""Modal serverless deployment for OpenPI policy serving.

This script deploys a WebSocket server on Modal that can dynamically load
and serve different policy checkpoints from HuggingFace repositories.

KEY FEATURE: Dynamic Model Loading in Inference Loop
=====================================================
Model loading happens during the inference loop (Phase 5), allowing clients to
dynamically switch models by including model specification in any observation message.
This is different from the original protocol where models were loaded at connection time.

The server implements two-level caching to optimize performance:
1. Checkpoint Cache: Avoids re-downloading model files if already on disk
2. Policy Cache: Avoids re-loading and re-compiling models if already in memory

This allows multiple clients to use different models without conflicts, while
still benefiting from caching when the same model is requested multiple times.

Example usage:
    modal deploy modal_serve_policy.py

Protocol:
1. Client connects
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
        ... other observation data ...
    }
6. Server loads model (if not already loaded) and returns action
7. Subsequent observations can omit model fields to reuse loaded model,
   or include them to switch to a different model

CRITICAL: Custom Transformers Implementation
============================================
This deployment requires custom modifications to the transformers library to support:
1. AdaRMS (Adaptive RMS normalization) - needed for pi0.5 models
2. Correct precision control for activations
3. KV cache support without automatic updates

The custom files are in src/openpi/models_pytorch/transformers_replace/ and must replace
the standard transformers package files during the build.

Why the special handling is needed:
- Standard transformers 4.53.2 doesn't support AdaRMS (use_adarms parameter)
- Without AdaRMS, GemmaRMSNorm only has a 'weight' parameter
- With AdaRMS, GemmaRMSNorm has a 'dense' layer (weight + bias) instead
- This creates a state_dict mismatch when loading pi0.5 checkpoints

The build process:
1. Dockerfile installs dependencies and copies custom transformers
2. Modal installs additional web server packages (fastapi, websockets)
3. Custom transformers are re-applied AFTER pip installs (critical!)
4. Python bytecode cache is cleared to force reimport
5. Verification checks ensure AdaRMS support is present

Key techniques to avoid conda-like reinstallation issues:
- Use --no-deps for packages that don't need transformers dependencies
- Clear __pycache__ after copying custom files
- Clear sys.modules cache before verification imports
- Verify actual model structure (GemmaRMSNorm.dense) not just config
"""

import dataclasses
import logging
import time
import traceback

import modal
from pathlib import Path

# Note: Specific imports are done inside the function to avoid import issues with Modal

# Get the project root directory (parent of scripts/)
project_root = Path(__file__).parent.parent

# Create Modal image from the existing serve_policy.Dockerfile
# This uses uv.lock for dependency management, ensuring consistency with local development
print(project_root)
image = (
    modal.Image.from_dockerfile(
        path=str(
            project_root / "modal_scripts/v21_openpi.Dockerfile"
        ),
        context_dir=str(project_root),
    )
    .uv_pip_install("fastapi", "websockets")
    .env({"PATH": "/.venv/bin:$PATH"})
)
app = modal.App(
    "openpi-policy-server-rtc-1",
    image=image,
)

# Checkpoint cache to avoid re-downloading the same checkpoint files
_checkpoint_cache: dict[tuple[str, str], Path] = {}

# Policy cache to avoid reloading the same policy (includes compiled models)
_policy_cache: dict[tuple[str, ...], any] = {}

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@app.function(
    gpu="A10G",
    timeout=3600,
    concurrency_limit=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/inference-checkpoints": modal.Volume.from_name(
            "openpi-checkpoint-volume", create_if_missing=True
        )
    },
    # region="ap-southeast",
)
@modal.concurrent(max_inputs=5)
@modal.asgi_app()
def endpoint():
    """FastAPI endpoint with WebSocket support for policy serving."""
    import os
    import sys
    import importlib.util
    from pathlib import Path
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import openpi_client.msgpack_numpy as msgpack_numpy

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    web_app = FastAPI()

    def download_checkpoint(hf_repo_id: str, folder_path: str) -> Path:
        """Download checkpoint from HuggingFace with caching.

        Returns the path to the downloaded checkpoint directory.
        This function checks if the checkpoint is already downloaded before fetching.
        """
        # Check cache first - this prevents re-downloading on every request
        cache_key = (hf_repo_id, folder_path)

        if cache_key in _checkpoint_cache:
            cached_path = _checkpoint_cache[cache_key]
            if cached_path.exists():
                logger.info(f"Using cached checkpoint from {cached_path}")
                return cached_path
            else:
                # Path was cached but files are gone, remove from cache
                logger.warning(
                    f"Cached checkpoint path {cached_path} no longer exists, re-downloading"
                )
                del _checkpoint_cache[cache_key]

        import subprocess

        logger.info(f"Downloading checkpoint from {hf_repo_id}/{folder_path}...")

        # Download the specific folder from HuggingFace to /inference-checkpoints
        checkpoint_base = Path("/inference-checkpoints")
        checkpoint_base.mkdir(parents=True, exist_ok=True)

        # Use huggingface-cli download to get the checkpoint folder
        # If folder_path is "/" or empty, download everything; otherwise download specific path
        if folder_path and folder_path != "/":
            # Strip leading/trailing slashes for cleaner paths
            clean_path = folder_path.strip("/")
            checkpoint_dir = checkpoint_base / clean_path
            # skip if exists
            if (
                checkpoint_dir.exists()
                and any(checkpoint_dir.iterdir())
                and any(
                    f.name.endswith(".safetensors") for f in checkpoint_dir.iterdir()
                )
            ):
                logger.info(
                    f"Checkpoint directory {checkpoint_dir} exists, skipping download."
                )
                _checkpoint_cache[cache_key] = checkpoint_dir
                return checkpoint_dir
            cmd = [
                "huggingface-cli",
                "download",
                hf_repo_id,
                "--include",
                f"{clean_path}/*",
                "--local-dir",
                str(checkpoint_base),
            ]
        else:
            checkpoint_dir = checkpoint_base
            # skip if exists and there is a safetensors file
            if (
                checkpoint_dir.exists()
                and any(checkpoint_dir.iterdir())
                and any(f.endswith(".safetensors") for f in checkpoint_dir.iterdir())
            ):
                logger.info(
                    f"Checkpoint directory {checkpoint_dir} exists, skipping download."
                )
                _checkpoint_cache[cache_key] = checkpoint_dir
                return checkpoint_dir
            cmd = [
                "huggingface-cli",
                "download",
                hf_repo_id,
                "--local-dir",
                str(checkpoint_base),
            ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully downloaded checkpoint to {checkpoint_dir}")

        logger.info(f"[checkpoint_dir] Checkpoint dir: {str(checkpoint_dir)}")

        # list all files in the checkpoint_dir
        for file in os.listdir(checkpoint_dir):
            logger.info(f"[checkpoint_dir] File: {file}")

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Folder {folder_path} not found in repo {hf_repo_id}"
            )

        # Cache the checkpoint directory path
        _checkpoint_cache[cache_key] = checkpoint_dir
        logger.info(f"Cached checkpoint path for future requests")

        return checkpoint_dir

    def download_norm_stats(dataset_repo_id: str, stats_json_path: str):
        """Download and load norm_stats from HuggingFace dataset.

        Returns the loaded norm_stats dictionary.
        
        Also saves to /workspace/dataset/norm_stats.json so that config's
        _load_norm_stats can find it when repo_id="/workspace/dataset".
        """
        import subprocess
        import shutil
        from openpi.shared import normalize as _normalize

        logger.info(
            f"Downloading norm_stats.json from {dataset_repo_id}/{stats_json_path}..."
        )

        stats_dest_dir = Path("/tmp/norm_stats")
        stats_dest_dir.mkdir(parents=True, exist_ok=True)

        # Use huggingface-cli download to get stats.json
        cmd = [
            "huggingface-cli",
            "download",
            dataset_repo_id,
            stats_json_path,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(stats_dest_dir),
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Load the norm stats directly from the downloaded file
        norm_stats_file = stats_dest_dir / stats_json_path
        logger.info(f"Loading norm stats from {norm_stats_file}")
        norm_stats = _normalize.deserialize_json(norm_stats_file.read_text())
        logger.info(
            f"Successfully loaded norm stats with keys: {list(norm_stats.keys())}"
        )

        # Also save to /workspace/dataset/ so config's _load_norm_stats can find it
        # when repo_id="/workspace/dataset" (the asset_id falls back to repo_id)
        config_expected_dir = Path("/workspace/dataset")
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
    ):
        """Load a policy with caching.

        This function is called on every request and allows each request to specify
        its own model. It checks caches at two levels:
        1. Checkpoint cache - avoids re-downloading model files
        2. Policy cache - avoids re-loading and re-compiling the model

        Args:
            hf_repo_id: HuggingFace repo ID (e.g., "username/repo-name")
            folder_path: Path to checkpoint folder within the repo
            config_name: Config name to use for loading the policy
            prompt: Optional default prompt for the policy
            dataset_repo_id: Optional HuggingFace dataset repo for norm_stats
            stats_json_path: Path to stats.json within the dataset repo
        """
        # Cache key for the fully loaded policy (including compiled model)
        # Note: We include all parameters since they affect the policy behavior
        cache_key = (
            hf_repo_id,
            folder_path,
            config_name,
            prompt or "",
            dataset_repo_id or "",
            stats_json_path or "",
        )

        logger.info(
            f"Loading new policy: {config_name} from {hf_repo_id}/{folder_path}"
        )

        # Download norm_stats if provided (not cached separately since it's small)
        # For TCR configs with repo_id="/workspace/dataset", norm_stats MUST be provided
        # via dataset_repo_id and stats_json_path since the local path doesn't exist in Modal
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
                "to download norm_stats from HuggingFace, e.g.:\n"
                '  "dataset_repo_id": "your-username/your-dataset",\n'
                '  "stats_json_path": "norm_stats.json"'
            )

        if cache_key in _policy_cache:
            logger.info(
                f"✓ Using cached policy for {config_name} from {hf_repo_id}/{folder_path}.\n\n There are {len(_policy_cache)} policies cached."
            )
            return _policy_cache[cache_key]

        # Download checkpoint (uses checkpoint cache to avoid re-downloading)
        try:
            checkpoint_dir = download_checkpoint(hf_repo_id, folder_path)
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            raise

        # Load the policy from the checkpoint
        try:
            logger.info(f"Loading policy from {checkpoint_dir}...")

            # Use the config specified by the client
            logger.info(f"Getting config: {config_name}")
            train_config = _config.get_config(config_name)
            logger.info(
                f"Using config '{train_config.name}' with pi05={train_config.model.pi05}"
            )

            # Create policy with norm_stats
            logger.info("Creating trained policy...")
            policy = _policy_config.create_trained_policy(
                train_config,
                str(checkpoint_dir),
                default_prompt=prompt,
                norm_stats=norm_stats,
            )

            # Cache the loaded policy for future requests with the same parameters
            _policy_cache[cache_key] = policy
            logger.info(f"✓ Successfully loaded and cached policy {config_name}")
            return policy
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            logger.error(traceback.format_exc())
            raise

    @web_app.websocket("/ws")
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
            - Model is loaded/switched dynamically when these fields are present
            - Server responds with actions

        Note: Model loading happens in the inference loop, allowing dynamic model switching.
        Caching at two levels (checkpoint + policy) prevents redundant downloads/loads.
        """
        await websocket.accept()
        logger.info(f"Connection from {websocket.client} opened")

        packer = msgpack_numpy.Packer()
        policy = None
        current_model_key = None  # Track currently loaded model

        try:
            # PHASE 1: Send initial empty metadata (signals server is ready)
            logger.info("PHASE 1: Sending initial empty metadata")
            await websocket.send_bytes(packer.pack({}))
            logger.info("✓ Sent initial empty metadata")

            # PHASE 2: Receive client initialization (can be empty - model loaded in Phase 5)
            logger.info("PHASE 2: Waiting for client initialization")
            init_data = await websocket.receive_bytes()
            init_message = msgpack_numpy.unpackb(init_data)
            logger.info(f"✓ Received initialization message: {init_message}")

            # PHASE 3: Skip - model loading moved to Phase 5
            logger.info("PHASE 3: Skipped - model will be loaded in inference loop")

            # PHASE 4: Send empty metadata (policy metadata will be sent after first model load)
            logger.info("PHASE 4: Sending empty metadata (model loads in Phase 5)")
            await websocket.send_bytes(packer.pack({}))
            logger.info("✓ Sent empty metadata, ready for inference loop")

            # PHASE 5: Inference loop with dynamic model loading
            logger.info("PHASE 5: Starting inference loop with dynamic model loading")
            prev_total_time = None
            step_count = 0
            while True:
                try:
                    start_time = time.monotonic()

                    # Receive observation (line 58)
                    logger.info(f"Step {step_count}: Waiting for observation...")
                    obs = msgpack_numpy.unpackb(await websocket.receive_bytes())
                    logger.info(
                        f"Step {step_count}: Received observation with keys: {list(obs.keys())}"
                    )

                    # Extract RTC parameters (kept in obs as _rtc_* keys for Policy.infer)
                    rtc_prev_actions = obs.pop("rtc_prev_actions", None)
                    rtc_config = obs.pop("rtc_config", None)
                    if rtc_prev_actions is not None:
                        obs["_rtc_prev_actions"] = rtc_prev_actions
                    if rtc_config is not None:
                        obs["_rtc_config"] = rtc_config

                    # Check if model specification is included in the observation
                    hf_repo_id = obs.pop("hf_repo_id", None)
                    folder_path = obs.pop("folder_path", None)
                    config_name = obs.pop("config_name", None)
                    prompt = obs.pop("prompt", None)
                    dataset_repo_id = obs.pop("dataset_repo_id", None)
                    stats_json_path = obs.pop("stats_json_path", None)

                    # Create model key to check if we need to load/switch model
                    if hf_repo_id and folder_path and config_name:
                        new_model_key = (
                            hf_repo_id,
                            folder_path,
                            config_name,
                            prompt or "",
                            dataset_repo_id or "",
                            stats_json_path or "",
                        )

                        # Load model if not loaded or if different model requested
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
                                )
                                current_model_key = new_model_key
                                logger.info(
                                    f"Step {step_count}: ✓ Model loaded successfully"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Step {step_count}: Failed to load policy: {e}"
                                )
                                logger.error(traceback.format_exc())
                                await websocket.send_text(
                                    f"Error loading policy:\n{traceback.format_exc()}"
                                )
                                await websocket.close(
                                    code=1011, reason="Policy loading failed"
                                )
                                return

                    # Check if policy is loaded
                    if policy is None:
                        error_msg = "No policy loaded. Include hf_repo_id, folder_path, and config_name in observation."
                        logger.error(f"Step {step_count}: {error_msg}")
                        await websocket.send_text(error_msg)
                        await websocket.close(code=1011, reason=error_msg)
                        return

                    # Run inference (lines 60-62)
                    logger.info(f"Step {step_count}: Running inference...")
                    infer_time = time.monotonic()
                    try:
                        action = policy.infer(obs)
                        logger.info(
                            f"Step {step_count}: Inference completed, action keys: {list(action.keys())}"
                        )
                    except Exception as e:
                        logger.error(f"Step {step_count}: Inference failed: {e}")
                        logger.error(traceback.format_exc())
                        raise
                    infer_time = time.monotonic() - infer_time

                    # Add timing information (lines 64-69)
                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"]["prev_total_ms"] = (
                            prev_total_time * 1000
                        )

                    # Send action (line 71)
                    logger.info(f"Step {step_count}: Sending action back to client...")
                    await websocket.send_bytes(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time
                    logger.info(
                        f"Step {step_count}: ✓ Completed in {prev_total_time*1000:.1f}ms"
                    )
                    step_count += 1

                except WebSocketDisconnect:
                    logger.info(
                        f"Connection from {websocket.client} closed (after {step_count} steps)"
                    )
                    break
                except Exception as e:
                    # Send traceback as text (lines 78-82)
                    logger.error(f"Error in inference loop at step {step_count}: {e}")
                    logger.error(traceback.format_exc())
                    try:
                        await websocket.send_text(traceback.format_exc())
                        await websocket.close(
                            code=1011,
                            reason="Internal server error. Traceback included in previous frame.",
                        )
                    except:
                        logger.error("Failed to send error to client")
                    raise

        except WebSocketDisconnect:
            logger.info(f"Connection from {websocket.client} closed")
        except Exception:
            logger.error(f"Error in websocket handler:\n{traceback.format_exc()}")
            raise

    @web_app.get("/healthz")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok"}

    return web_app
