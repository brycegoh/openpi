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
1. Clone the openpi repo (v2.1 branch) from GitHub into the Modal image
2. Install all dependencies via uv sync (uses uv.lock for reproducibility)
3. Apply custom transformers patches from the repo
4. Python bytecode cache is cleared to force reimport
5. Install web server packages (fastapi, websockets, msgpack-numpy)
"""

import asyncio
import logging
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import modal

image = modal.Image.from_dockerfile(
    "v21_openpi.Dockerfile",
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
    from pathlib import Path
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from openpi_client import msgpack_numpy

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

    # ------------------------------------------------------------------
    # Inline LWW Register for DRTC (server-side, thread-safe)
    # ------------------------------------------------------------------
    class _LWWState:
        __slots__ = ("control_step", "value")
        def __init__(self, control_step: int, value):
            self.control_step = control_step
            self.value = value

    class _LWWRegister:
        """Minimal thread-safe LWW register for server-side obs/action handoff."""
        def __init__(self, initial_control_step: int = -(2**63), initial_value=None):
            self._lock = threading.Lock()
            self._state = _LWWState(initial_control_step, initial_value)

        def read(self):
            with self._lock:
                return self._state

        def update_if_newer(self, control_step: int, value):
            with self._lock:
                if control_step > self._state.control_step:
                    self._state = _LWWState(control_step, value)
                    return self._state, True
                return self._state, False

        def reader(self, initial_watermark: int = -(2**63)):
            return _LWWReader(self, initial_watermark)

    class _LWWReader:
        def __init__(self, register, watermark: int):
            self._register = register
            self._watermark = watermark

        def read_if_newer(self):
            state = self._register.read()
            is_new = state.control_step > self._watermark
            if is_new:
                self._watermark = state.control_step
            return state, is_new

    # ------------------------------------------------------------------
    # Action Chunk Cache for RTC inpainting (server-side)
    # ------------------------------------------------------------------
    class _ActionChunkCache:
        """LRU cache for raw action chunks, keyed by source control step."""
        def __init__(self, max_size: int = 10):
            self._cache: OrderedDict = OrderedDict()
            self._max_size = max_size

        def put(self, src_step: int, raw_actions):
            if src_step in self._cache:
                del self._cache[src_step]
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[src_step] = raw_actions

        def get(self, src_step: int):
            return self._cache.get(src_step)

        def clear(self):
            self._cache.clear()

    # ------------------------------------------------------------------
    # DRTC async inference handler
    # ------------------------------------------------------------------
    async def _handle_drtc_mode(
        websocket: WebSocket,
        packer,
        policy,
        drtc_config: dict,
    ) -> None:
        """Run DRTC async bidirectional inference loop.

        Two concurrent tasks:
        - _drtc_receive_obs: receives observations from WS, writes to LWW register
        - _drtc_infer_and_send: polls register, runs inference in thread pool, sends chunks
        """
        import numpy as np

        action_horizon = drtc_config.get("action_horizon", 50)
        fps = drtc_config.get("fps", 50)
        rtc_enabled = drtc_config.get("rtc_enabled", False)
        dt = 1.0 / fps

        obs_reg = _LWWRegister()
        action_cache = _ActionChunkCache(max_size=action_horizon)
        shutdown_event = asyncio.Event()

        async def _drtc_receive_obs():
            """Receive observations from client and write to LWW register."""
            try:
                while not shutdown_event.is_set():
                    raw = await websocket.receive_bytes()
                    msg = msgpack_numpy.unpackb(raw)
                    if msg.get("type") != "obs":
                        continue
                    control_step = msg.get("control_step", 0)
                    obs_reg.update_if_newer(control_step, msg)
            except WebSocketDisconnect:
                logger.info("DRTC: Client disconnected (obs receiver)")
            except Exception as e:
                logger.error(f"DRTC obs receiver error: {e}")
            finally:
                shutdown_event.set()

        async def _drtc_infer_and_send():
            """Poll obs register, run inference, send action chunks."""
            import numpy as np

            reader = obs_reg.reader()
            infer_count = 0

            try:
                while not shutdown_event.is_set():
                    state, is_new = reader.read_if_newer()
                    if not is_new or state.value is None:
                        await asyncio.sleep(0.005)
                        continue

                    obs_msg = state.value
                    control_step = obs_msg.get("control_step", 0)
                    chunk_start_step = obs_msg.get("chunk_start_step", 0)
                    obs_timestamp = obs_msg.get("timestamp", time.time())

                    # Extract observation data (remove DRTC protocol fields)
                    obs = {k: v for k, v in obs_msg.items()
                           if k not in ("type", "control_step", "chunk_start_step",
                                        "timestamp", "rtc_meta")}

                    # Extract RTC metadata if present
                    rtc_meta = obs_msg.get("rtc_meta")

                    # Build sample_kwargs for RTC if enabled
                    rtc_sample_kwargs = {}
                    if rtc_enabled and rtc_meta is not None:
                        try:
                            d = int(rtc_meta.get("latency_steps", 0))
                            H = action_horizon
                            overlap_end = int(rtc_meta.get("overlap_end", H - d))
                            action_schedule_spans = rtc_meta.get("action_schedule_spans")

                            if action_schedule_spans:
                                slices = []
                                for src_step, start_idx, end_idx in action_schedule_spans:
                                    cached = action_cache.get(int(src_step))
                                    if cached is not None:
                                        slices.append(cached[start_idx:end_idx])

                                if slices:
                                    import torch
                                    prefix_np = np.concatenate(slices, axis=0)
                                    prefix_tensor = torch.from_numpy(prefix_np).unsqueeze(0)
                                    T_prefix = prefix_tensor.shape[1]
                                    effective_overlap_end = min(overlap_end, T_prefix)

                                    rtc_sample_kwargs = {
                                        "inference_delay": d,
                                        "prev_chunk_left_over": prefix_tensor,
                                        "overlap_end": effective_overlap_end,
                                    }
                        except Exception as e:
                            logger.warning(f"DRTC: RTC metadata processing error: {e}")

                    # Run blocking inference in thread pool
                    t_infer_start = time.monotonic()
                    try:
                        if rtc_sample_kwargs:
                            action = await asyncio.to_thread(
                                policy.infer, obs, **rtc_sample_kwargs
                            )
                        else:
                            action = await asyncio.to_thread(policy.infer, obs)
                    except Exception as e:
                        logger.error(f"DRTC: Inference failed at step {infer_count}: {e}")
                        logger.error(traceback.format_exc())
                        continue
                    t_infer_ms = (time.monotonic() - t_infer_start) * 1000

                    # Extract actions array from policy output
                    actions = action.get("actions")
                    if actions is None:
                        logger.warning("DRTC: No 'actions' key in inference output")
                        continue

                    actions = np.asarray(actions, dtype=np.float32)
                    if actions.ndim == 1:
                        actions = actions.reshape(1, -1)

                    # Cache raw actions for future RTC inpainting
                    action_cache.put(control_step, actions)

                    num_actions, action_dim = actions.shape

                    # Build response
                    response = {
                        "type": "action_chunk",
                        "actions": actions,
                        "source_control_step": control_step,
                        "chunk_start_step": chunk_start_step,
                        "timestamp": obs_timestamp,
                        "dt": dt,
                        "num_actions": int(num_actions),
                        "action_dim": int(action_dim),
                        "server_timing": {"infer_ms": t_infer_ms},
                    }

                    try:
                        await websocket.send_bytes(packer.pack(response))
                    except Exception:
                        logger.info("DRTC: Failed to send action chunk (client may have disconnected)")
                        break

                    infer_count += 1
                    if infer_count % 50 == 0:
                        logger.info(f"DRTC: Completed {infer_count} inferences, last infer_ms={t_infer_ms:.1f}")

            except Exception as e:
                logger.error(f"DRTC infer+send error: {e}")
                logger.error(traceback.format_exc())
            finally:
                shutdown_event.set()

        logger.info(f"DRTC: Entering async mode (action_horizon={action_horizon}, fps={fps}, rtc={rtc_enabled})")

        await asyncio.gather(
            _drtc_receive_obs(),
            _drtc_infer_and_send(),
            return_exceptions=True,
        )
        logger.info("DRTC: Async loop ended")

    # ------------------------------------------------------------------
    # WebSocket handler (supports both sync and DRTC modes)
    # ------------------------------------------------------------------
    @web_app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        """WebSocket handler for policy inference.

        Supports two modes:
        - Sync mode (default): Client sends obs, blocks, receives action chunk.
        - DRTC mode: Client sends {"mode": "drtc", ...} in init message.
          Server and client exchange obs/actions asynchronously over the same WS.

        Protocol:
        1. Client connects
        2. Server sends empty metadata {}
        3. Client sends initialization message:
           - For sync: can be empty, model loaded in Phase 5
           - For DRTC: must include "mode": "drtc" + model fields + drtc config
        4. Server sends metadata (empty for sync, drtc config for DRTC)
        5. Inference loop (sync or DRTC)
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
            logger.info(f"Received init message keys: {list(init_message.keys()) if isinstance(init_message, dict) else 'not a dict'}")

            # Detect DRTC mode
            is_drtc = isinstance(init_message, dict) and init_message.get("mode") == "drtc"

            if is_drtc:
                # ---- DRTC MODE ----
                logger.info("PHASE 3: DRTC mode detected, loading model during handshake")

                hf_repo_id = init_message.get("hf_repo_id")
                folder_path = init_message.get("folder_path")
                config_name = init_message.get("config_name")
                prompt = init_message.get("prompt")
                dataset_repo_id = init_message.get("dataset_repo_id")
                stats_json_path = init_message.get("stats_json_path")

                if not (hf_repo_id and folder_path and config_name):
                    error_msg = "DRTC mode requires hf_repo_id, folder_path, and config_name in init message."
                    logger.error(error_msg)
                    await websocket.send_text(error_msg)
                    await websocket.close(code=1011, reason=error_msg)
                    return

                try:
                    policy = load_policy(
                        hf_repo_id, folder_path, config_name,
                        prompt, dataset_repo_id, stats_json_path,
                    )
                except Exception as e:
                    logger.error(f"DRTC: Failed to load policy: {e}")
                    await websocket.send_text(f"Error loading policy:\n{traceback.format_exc()}")
                    await websocket.close(code=1011, reason="Policy loading failed")
                    return

                drtc_config = {
                    "action_horizon": init_message.get("action_horizon", 50),
                    "fps": init_message.get("fps", 50),
                    "rtc_enabled": init_message.get("rtc_enabled", False),
                }

                # PHASE 4: Send DRTC metadata
                logger.info("PHASE 4: Sending DRTC metadata")
                await websocket.send_bytes(packer.pack({
                    "mode": "drtc",
                    "action_horizon": drtc_config["action_horizon"],
                    "rtc_enabled": drtc_config["rtc_enabled"],
                }))

                # PHASE 5: DRTC async loop
                logger.info("PHASE 5: Starting DRTC async inference loop")
                await _handle_drtc_mode(websocket, packer, policy, drtc_config)

            else:
                # ---- LEGACY SYNC MODE ----
                logger.info("PHASE 3: Sync mode - model will be loaded in inference loop")

                # PHASE 4: Send empty metadata
                logger.info("PHASE 4: Sending empty metadata (model loads in Phase 5)")
                await websocket.send_bytes(packer.pack({}))

                # PHASE 5: Synchronous inference loop
                logger.info("PHASE 5: Starting sync inference loop")
                prev_total_time = None
                step_count = 0
                while True:
                    try:
                        start_time = time.monotonic()

                        logger.info(f"Step {step_count}: Waiting for observation...")
                        obs = msgpack_numpy.unpackb(await websocket.receive_bytes())
                        logger.info(
                            f"Step {step_count}: Received observation with keys: {list(obs.keys())}"
                        )

                        hf_repo_id = obs.pop("hf_repo_id", None)
                        folder_path = obs.pop("folder_path", None)
                        config_name = obs.pop("config_name", None)
                        prompt = obs.pop("prompt", None)
                        dataset_repo_id = obs.pop("dataset_repo_id", None)
                        stats_json_path = obs.pop("stats_json_path", None)

                        if hf_repo_id and folder_path and config_name:
                            new_model_key = (
                                hf_repo_id, folder_path, config_name,
                                prompt or "", dataset_repo_id or "", stats_json_path or "",
                            )
                            if current_model_key != new_model_key:
                                logger.info(
                                    f"Step {step_count}: Loading model {config_name} from {hf_repo_id}/{folder_path}"
                                )
                                try:
                                    policy = load_policy(
                                        hf_repo_id, folder_path, config_name,
                                        prompt, dataset_repo_id, stats_json_path,
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
                            logger.info(
                                f"Step {step_count}: Inference completed, action keys: {list(action.keys())}"
                            )
                        except Exception as e:
                            logger.error(f"Step {step_count}: Inference failed: {e}")
                            logger.error(traceback.format_exc())
                            raise
                        infer_time = time.monotonic() - infer_time

                        action["server_timing"] = {"infer_ms": infer_time * 1000}
                        if prev_total_time is not None:
                            action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                        logger.info(f"Step {step_count}: Sending action back to client...")
                        await websocket.send_bytes(packer.pack(action))
                        prev_total_time = time.monotonic() - start_time
                        logger.info(f"Step {step_count}: Completed in {prev_total_time*1000:.1f}ms")
                        step_count += 1

                    except WebSocketDisconnect:
                        logger.info(f"Connection closed (after {step_count} steps)")
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
