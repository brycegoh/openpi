#!/usr/bin/env python

import logging
import pathlib
import random
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from openpi.policies import rtc_processor
import tyro

from openpi.policies import policy_config
from openpi.training import config as train_config
import openpi.training.data_loader as _data_loader
from openpi.policies.rtc_processor import RTCConfig
from openpi.policies import policy_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # JAX uses explicit PRNG keys, but we can set the default behavior
    key = jax.random.PRNGKey(seed)
    logging.info(f"Random seed set to: {seed}")
    logging.info(f"JAX PRNG key initialized with seed: {seed}")
    return key


class RTCDatasetEvaluator:
    """Evaluator for RTC on dataset samples."""

    def __init__(self, cfg: "Args"):
        self.cfg = cfg

        # Load the training config to get model configuration
        logging.info(f"Loading checkpoint from {cfg.checkpoint_path}")
        self.train_cfg = train_config.get_config(cfg.train_config_name)

        logging.info(f"Model config: {self.train_cfg.model}")

        # Don't load policies yet - we'll load them one at a time during evaluation
        # to avoid GPU memory issues
        self.policy_with_rtc = None
        self.policy_without_rtc = None

        # Create raw dataset without transforms for inference
        # We'll use policy.infer() which applies transforms itself
        self.data_config = self.train_cfg.data.create(self.train_cfg.assets_dirs, self.train_cfg.model)
        self.raw_dataset = _data_loader.create_torch_dataset(
            self.data_config,
            self.train_cfg.model.action_horizon,
            self.train_cfg.model
        )

        # Apply only repack transforms to get the right key structure
        # Skip data_transforms and model_transforms since policy.infer() will apply them
        self.dataset = _data_loader.TransformedDataset(
            self.raw_dataset,
            list(self.data_config.repack_transforms.inputs)
        )

        logging.info(f"Dataloader created successfully")

    def _load_policy_with_rtc(self):
        """Load policy with RTC enabled."""
        import torch
        import gc

        logging.info("Creating policy WITH RTC...")
        self.policy_with_rtc = policy_config.create_trained_policy(
            train_config=self.train_cfg,
            checkpoint_dir=pathlib.Path(self.cfg.checkpoint_path),
            sample_kwargs=self.cfg.sample_kwargs or {},
        )
        self.policy_with_rtc._model.init_rtc_processor(self.cfg.rtc_config)
        logging.info("Policy WITH RTC loaded successfully")

    def _load_policy_without_rtc(self):
        """Load policy without RTC."""
        import torch
        import gc

        logging.info("Creating policy WITHOUT RTC...")
        self.policy_without_rtc = policy_config.create_trained_policy(
            train_config=self.train_cfg,
            checkpoint_dir=pathlib.Path(self.cfg.checkpoint_path),
            sample_kwargs=self.cfg.sample_kwargs or {},
        )
        # Initialize with disabled RTC
        disabled_rtc_config = replace(self.cfg.rtc_config, enabled=False)
        self.policy_without_rtc._model.init_rtc_processor(disabled_rtc_config)
        self.policy_without_rtc._model.rtc_processor.rtc_config.enabled = False
        logging.info("Policy WITHOUT RTC loaded successfully")

    def _free_policy(self, which="both"):
        """Free GPU memory by deleting policy."""
        import torch
        import gc

        if which in ("with_rtc", "both") and self.policy_with_rtc is not None:
            del self.policy_with_rtc
            self.policy_with_rtc = None
            logging.info("Freed policy WITH RTC")

        if which in ("without_rtc", "both") and self.policy_without_rtc is not None:
            del self.policy_without_rtc
            self.policy_without_rtc = None
            logging.info("Freed policy WITHOUT RTC")

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logging.info("Cleared CUDA cache")

    def run_evaluation(self) -> dict:
        """Run full evaluation on dataset.

        Returns:
            Dictionary with aggregated metrics and detailed results
        """
        # Randomly select 2 samples from the dataset
        dataset_size = len(self.dataset)
        logging.info(f"Dataset size: {dataset_size}")

        if dataset_size < 2:
            logging.error(f"Not enough samples in dataset. Found {dataset_size}, need at least 2")
            return {}

        selected_indices = random.sample(range(dataset_size), 2)
        logging.info(f"Selected samples at indices: {selected_indices}")

        # Get the two selected samples
        first_sample = self.dataset[selected_indices[0]]
        second_sample = self.dataset[selected_indices[1]]

        # Extract actions from first sample
        # Take only first half based on model's action_horizon, not cfg
        model_action_horizon = self.train_cfg.model.action_horizon
        prev_chunk_left_over = np.array(first_sample["actions"])
        if len(prev_chunk_left_over.shape) > 1:
            # Take first half based on model's action horizon
            prev_chunk_left_over = prev_chunk_left_over[:model_action_horizon // 2]
            # Add batch dimension to match observation batch size
            prev_chunk_left_over = prev_chunk_left_over[np.newaxis, ...]  # Shape: (1, time, action_dim)
            logging.info(f"prev_chunk_left_over shape after adding batch dim: {prev_chunk_left_over.shape}")
        else:
            logging.warning("Actions have unexpected shape, skipping evaluation")
            return {}

        # Prepare observation dict for policy.infer()
        # Remove actions key since policy.infer() doesn't need it
        obs = {k: v for k, v in second_sample.items() if k != "actions"}

        # Debug: print observation keys
        logging.info(f"Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, (np.ndarray, list)):
                logging.info(f"  {key}: shape={np.array(value).shape if hasattr(value, 'shape') or isinstance(value, list) else 'N/A'}, type={type(value)}")
            elif isinstance(value, dict):
                logging.info(f"  {key}: nested dict with keys {list(value.keys())}")
            else:
                logging.info(f"  {key}: type={type(value)}")

        # Generate noise for inference
        # Use model's action_horizon and action_dim, not cfg values
        model_action_horizon = self.train_cfg.model.action_horizon
        model_action_dim = self.train_cfg.model.action_dim
        noise = np.random.randn(model_action_horizon, model_action_dim).astype(np.float32)

        # ========== Run inference WITHOUT RTC ==========
        logging.info("=" * 80)
        logging.info("Running inference WITHOUT RTC")
        logging.info("=" * 80)

        # Load policy without RTC
        self._load_policy_without_rtc()

        # Use the policy without RTC
        rtc_processor_no_rtc = self.policy_without_rtc._model.rtc_processor
        if rtc_processor_no_rtc:
            logging.info(f"Policy WITHOUT RTC - rtc_enabled: {rtc_processor_no_rtc.rtc_enabled()}")
            logging.info(f"Policy WITHOUT RTC - config: {rtc_processor_no_rtc.rtc_config}")
        else:
            logging.info("Policy WITHOUT RTC - rtc_processor is None")

        result_no_rtc = self.policy_without_rtc.infer(
            obs,
            noise=noise,
            inference_delay=self.cfg.inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=self.cfg.execution_horizon
        )
        actions_no_rtc = result_no_rtc["actions"]
        # Get tracking data from result dictionary
        tracking_no_rtc = result_no_rtc.get("tracking_history", None)
        logging.info(f"result_no_rtc keys: {list(result_no_rtc.keys())}")
        if tracking_no_rtc is not None:
            logging.info(f"tracking_no_rtc keys: {list(tracking_no_rtc.keys())}")
        else:
            logging.warning("tracking_no_rtc is None")

        # Free the non-RTC policy to save GPU memory
        self._free_policy(which="without_rtc")

        # ========== Run inference WITH RTC ==========
        logging.info("=" * 80)
        logging.info("Running inference WITH RTC")
        logging.info("=" * 80)

        # Load policy with RTC
        self._load_policy_with_rtc()

        # Use the policy with RTC
        rtc_processor_with_rtc = self.policy_with_rtc._model.rtc_processor
        if rtc_processor_with_rtc:
            logging.info(f"Policy WITH RTC - rtc_enabled: {rtc_processor_with_rtc.rtc_enabled()}")
            logging.info(f"Policy WITH RTC - config: {rtc_processor_with_rtc.rtc_config}")
            logging.info(f"Policy WITH RTC - execution_horizon: {rtc_processor_with_rtc.rtc_config.execution_horizon}")
            logging.info(f"Policy WITH RTC - prefix_attention_schedule: {rtc_processor_with_rtc.rtc_config.prefix_attention_schedule}")
            logging.info(f"Policy WITH RTC - max_guidance_weight: {rtc_processor_with_rtc.rtc_config.max_guidance_weight}")
        else:
            logging.info("Policy WITH RTC - rtc_processor is None (ERROR: should not happen!)")

        result_rtc = self.policy_with_rtc.infer(
            obs,
            noise=noise,
            inference_delay=self.cfg.inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=self.cfg.execution_horizon
        )

        logging.info(f"result_rtc keys: {list(result_rtc.keys())}")
        logging.info(f"result_rtc['actions'] shape: {result_rtc['actions'].shape}")

        actions_rtc = result_rtc["actions"]
        # Get tracking data from result dictionary
        tracking_rtc = result_rtc.get("tracking_history", None)

        if tracking_rtc is not None:
            logging.info(f"tracking_rtc keys: {list(tracking_rtc.keys())}")
            if 'weights' in tracking_rtc:
                weights = np.array(tracking_rtc['weights'])
                logging.info(f"Weights shape: {weights.shape}")
                logging.info(f"First timestep weights: {weights[0] if weights.shape[0] > 0 else 'empty'}")
                logging.info(f"Weights stats - min: {np.min(weights)}, max: {np.max(weights)}, mean: {np.mean(weights)}")
                logging.info(f"Non-zero weights count: {np.sum(weights > 0)}")
            if 'guidance_weight' in tracking_rtc:
                gw = np.array(tracking_rtc['guidance_weight'])
                logging.info(f"Guidance weights: min={np.min(gw)}, max={np.max(gw)}, mean={np.mean(gw)}")
        else:
            logging.warning("tracking_rtc is None")

        # ========== Create side-by-side visualization ==========
        # Use min of 6 and model's action_dim for plots
        num_plots = min(6, model_action_dim)
        fig, axes = plt.subplots(num_plots, 2, figsize=(20, 12))

        # Handle single row case
        if num_plots == 1:
            axes = axes.reshape(1, 2)

        fig.suptitle(f"Episodes {selected_indices[0]} & {selected_indices[1]} - RTC Comparison", fontsize=18)

        # Remove batch dimension for plotting
        prev_chunk_to_plot = prev_chunk_left_over[0] if prev_chunk_left_over.ndim == 3 else prev_chunk_left_over
        actions_no_rtc_plot = actions_no_rtc[0] if actions_no_rtc.ndim == 3 else actions_no_rtc
        actions_rtc_plot = actions_rtc[0] if actions_rtc.ndim == 3 else actions_rtc

        # Plot NO RTC (left column)
        self.axs = axes[:, 0]
        axes[0, 0].set_title("Without RTC", fontsize=16, fontweight='bold')
        self.plot_waypoints(prev_chunk_to_plot, label="Previous Actions", color="red")
        self.plot_waypoints(actions_no_rtc_plot, label="Predicted Actions", color="blue")

        # Plot WITH RTC (right column)
        self.axs = axes[:, 1]
        axes[0, 1].set_title("With RTC", fontsize=16, fontweight='bold')
        self.plot_waypoints(prev_chunk_to_plot, label="Previous Actions", color="red")
        self.plot_waypoints(actions_rtc_plot, label="Predicted Actions", color="blue")

        plt.tight_layout()
        filename = f"rtc_comparison_episodes_{selected_indices[0]}_{selected_indices[1]}.png"
        plt.savefig(filename, dpi=150)
        logging.info(f"Saved RTC comparison to {filename}")
        plt.close(fig)

        # ========== Create denoising step visualizations ==========
        logging.info("Creating denoising step visualizations...")
        self.visualize_denoising_steps(tracking_no_rtc, f"no_rtc_{selected_indices[0]}_{selected_indices[1]}")
        self.visualize_denoising_steps(tracking_rtc, f"with_rtc_{selected_indices[0]}_{selected_indices[1]}")

        # ========== Combine denoising visualizations side-by-side ==========
        self.combine_denoising_visualizations()

        logging.info("Evaluation completed")
        return {}

    def plot_waypoints(self, chunk, start_from: int = 0, color: str | None = None, label: str | None = None):
        # Only plot as many dimensions as we have subplots
        num_dims_to_plot = min(len(self.axs), chunk.shape[-1])
        for j in range(num_dims_to_plot):
            self.axs[j].plot(
                np.arange(start_from, start_from + chunk.shape[0]),
                chunk[:, j],
                color=color,
                label=label,
            )
            self.axs[j].set_ylabel("Joint angle", fontsize=14)
            self.axs[j].grid()
            plt.tick_params(labelsize=14)
            self.axs[j].legend(loc="upper right", fontsize=14)
            if j == 2:
                self.axs[j].set_xlabel("Step #", fontsize=16)

    def visualize_rtc_tracking(self, tracking_rtc: dict, tracking_no_rtc: dict, selected_indices: list):
        """Create detailed visualization of RTC tracking data."""
        logging.info("Creating detailed RTC tracking visualizations...")

        logging.info(f"tracking_rtc: {tracking_rtc}")
        logging.info(f"tracking_no_rtc: {tracking_no_rtc}")
        # Check if we have RTC tracking data
        if tracking_rtc is None or "x_t" not in tracking_rtc:
            logging.warning("No RTC tracking data available for visualization")
            return

        # Extract arrays from tracking dictionary
        # Shape: [num_steps, batch_size, action_horizon, action_dim]
        x_t_rtc = np.array(tracking_rtc["x_t"])
        v_t_rtc = np.array(tracking_rtc["v_t"])
        time_rtc = np.array(tracking_rtc["time"])

        # Create figure
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(f"RTC Tracking Details - Episodes {selected_indices[0]} & {selected_indices[1]}", fontsize=20)

        # 1. X_t evolution over timesteps
        ax1 = plt.subplot(3, 3, 1)
        num_dims = min(3, x_t_rtc.shape[-1])
        for i in range(num_dims):
            ax1.plot(time_rtc, x_t_rtc[:, 0, 0, i], label=f"Dim {i}")
        ax1.set_xlabel("Time (t)")
        ax1.set_ylabel("X_t value")
        ax1.set_title("X_t Evolution (first 3 dims)")
        ax1.legend()
        ax1.grid(True)

        # 2. V_t (velocity) evolution
        ax2 = plt.subplot(3, 3, 2)
        for i in range(num_dims):
            ax2.plot(time_rtc, v_t_rtc[:, 0, 0, i], label=f"Dim {i}")
        ax2.set_xlabel("Time (t)")
        ax2.set_ylabel("V_t value")
        ax2.set_title("V_t (Velocity) Evolution")
        ax2.legend()
        ax2.grid(True)

        # 3. Guidance weights over time (if RTC)
        if "guidance_weight" in tracking_rtc:
            ax3 = plt.subplot(3, 3, 3)
            guidance_weights = np.array(tracking_rtc["guidance_weight"])
            if guidance_weights.ndim > 1:
                guidance_weights = guidance_weights[:, 0]  # Take first batch
            ax3.plot(time_rtc, guidance_weights)
            ax3.set_xlabel("Time (t)")
            ax3.set_ylabel("Guidance Weight")
            ax3.set_title("Guidance Weight Evolution")
            ax3.grid(True)

        # 4. Error magnitudes (if RTC)
        if "error" in tracking_rtc:
            ax4 = plt.subplot(3, 3, 4)
            errors = np.array(tracking_rtc["error"])
            error_norms = np.linalg.norm(errors.reshape(errors.shape[0], -1), axis=-1)
            ax4.plot(time_rtc, error_norms)
            ax4.set_xlabel("Time (t)")
            ax4.set_ylabel("Error Norm")
            ax4.set_title("Error Magnitude Over Time")
            ax4.grid(True)

        # 5. Prefix weights visualization (if RTC)
        if "weights" in tracking_rtc:
            ax5 = plt.subplot(3, 3, 5)
            weights = np.array(tracking_rtc["weights"])
            if weights.ndim > 1:
                weights = weights[0, 0] if weights.ndim > 2 else weights[0]
            ax5.plot(weights)
            ax5.set_xlabel("Action Dimension")
            ax5.set_ylabel("Weight")
            ax5.set_title("Prefix Attention Weights")
            ax5.grid(True)

        # 6. Pinv correction magnitudes (if RTC)
        if "pinv_correction" in tracking_rtc:
            ax6 = plt.subplot(3, 3, 6)
            pinv_corrections = np.array(tracking_rtc["pinv_correction"])
            pinv_norms = np.linalg.norm(pinv_corrections.reshape(pinv_corrections.shape[0], -1), axis=-1)
            ax6.plot(time_rtc, pinv_norms)
            ax6.set_xlabel("Time (t)")
            ax6.set_ylabel("Correction Norm")
            ax6.set_title("Pinv Correction Magnitude")
            ax6.grid(True)

        # 7. X_1 predictions (if RTC)
        if "x_1" in tracking_rtc:
            ax7 = plt.subplot(3, 3, 7)
            x_1_history = np.array(tracking_rtc["x_1"])
            for i in range(num_dims):
                ax7.plot(time_rtc, x_1_history[:, 0, 0, i], label=f"Dim {i}")
            ax7.set_xlabel("Time (t)")
            ax7.set_ylabel("X_1 value")
            ax7.set_title("X_1 Predictions (first 3 dims)")
            ax7.legend()
            ax7.grid(True)

        # 8. Convergence plot - difference between consecutive x_t
        if x_t_rtc.shape[0] > 1:
            ax8 = plt.subplot(3, 3, 8)
            x_t_diffs = np.linalg.norm(np.diff(x_t_rtc, axis=0).reshape(x_t_rtc.shape[0]-1, -1), axis=-1)
            ax8.plot(time_rtc[1:], x_t_diffs)
            ax8.set_xlabel("Time (t)")
            ax8.set_ylabel("||X_t - X_{t-1}||")
            ax8.set_title("Convergence (Step Size)")
            ax8.grid(True)
            ax8.set_yscale("log")

        # 9. Compare RTC vs non-RTC if available
        if tracking_no_rtc is not None and "v_t" in tracking_no_rtc:
            ax9 = plt.subplot(3, 3, 9)
            v_t_no_rtc = np.array(tracking_no_rtc["v_t"])
            v_t_rtc_norm = np.linalg.norm(v_t_rtc.reshape(v_t_rtc.shape[0], -1), axis=-1)
            v_t_no_rtc_norm = np.linalg.norm(v_t_no_rtc.reshape(v_t_no_rtc.shape[0], -1), axis=-1)
            ax9.plot(time_rtc, v_t_rtc_norm, label="RTC", color="red")
            ax9.plot(time_rtc, v_t_no_rtc_norm, label="No RTC", color="blue", linestyle="--")
            ax9.set_xlabel("Time (t)")
            ax9.set_ylabel("||V_t||")
            ax9.set_title("Velocity Magnitude Comparison")
            ax9.legend()
            ax9.grid(True)

        plt.tight_layout()
        tracking_filename = f"rtc_tracking_details_{selected_indices[0]}_{selected_indices[1]}.png"
        plt.savefig(tracking_filename, dpi=150)
        logging.info(f"Saved RTC tracking details to {tracking_filename}")
        plt.close(fig)

    def visualize_denoising_steps(self, tracking_data: dict, filename: str):
        """Create denoising step visualizations from tracking data.

        Args:
            tracking_data: Dictionary with tracking history from RTCTracker
            filename: Name of the file to save the visualizations
        """
        if tracking_data is None or "x_t" not in tracking_data:
            logging.warning(f"No tracking data available for {filename} visualization")
            return

        # Extract data
        x_t_history = np.array(tracking_data["x_t"])  # [num_steps, batch, action_horizon, action_dim]
        v_t_history = np.array(tracking_data["v_t"])
        time_history = np.array(tracking_data["time"])

        # Get dimensions
        num_steps = x_t_history.shape[0]
        action_dim = min(6, x_t_history.shape[-1])  # Plot first 6 dimensions

        # Create x_t visualization
        fig_xt, axes_xt = plt.subplots(action_dim, 1, figsize=(12, 12))
        if action_dim == 1:
            axes_xt = [axes_xt]

        fig_xt.suptitle(f"X_t Denoising Trajectory ({filename})", fontsize=16)

        for dim in range(action_dim):
            ax = axes_xt[dim]
            # Plot trajectory for each denoising step
            for step in range(num_steps):
                x_t_step = x_t_history[step, 0, :, dim]  # [action_horizon]
                alpha = 0.3 + 0.7 * (step / max(1, num_steps - 1))
                ax.plot(x_t_step, alpha=alpha, linewidth=1, color='blue')

            # Highlight first and last
            ax.plot(x_t_history[0, 0, :, dim], color='green', linewidth=2, label='Initial', alpha=0.8)
            ax.plot(x_t_history[-1, 0, :, dim], color='red', linewidth=2, label='Final', alpha=0.8)

            ax.set_ylabel(f"Dim {dim}", fontsize=12)
            ax.grid(True, alpha=0.3)
            if dim == 0:
                ax.legend(loc='upper right')
            if dim == action_dim - 1:
                ax.set_xlabel("Action Horizon", fontsize=12)

        plt.tight_layout()
        xt_filename = f"pi0_pytorch_x_t_{filename}_denoise_steps.png"
        plt.savefig(xt_filename, dpi=150)
        logging.info(f"Saved x_t denoising visualization to {xt_filename}")
        plt.close(fig_xt)

        # Create v_t visualization
        fig_v, axes_v = plt.subplots(action_dim, 1, figsize=(12, 12))
        if action_dim == 1:
            axes_v = [axes_v]

        fig_v.suptitle(f"V_t Velocity Trajectory ({filename})", fontsize=16)

        for dim in range(action_dim):
            ax = axes_v[dim]
            # Plot velocity for each denoising step
            for step in range(num_steps):
                v_t_step = v_t_history[step, 0, :, dim]  # [action_horizon]
                alpha = 0.3 + 0.7 * (step / max(1, num_steps - 1))
                ax.plot(v_t_step, alpha=alpha, linewidth=1, color='purple')

            # Highlight first and last
            ax.plot(v_t_history[0, 0, :, dim], color='green', linewidth=2, label='Initial', alpha=0.8)
            ax.plot(v_t_history[-1, 0, :, dim], color='red', linewidth=2, label='Final', alpha=0.8)

            ax.set_ylabel(f"Dim {dim}", fontsize=12)
            ax.grid(True, alpha=0.3)
            if dim == 0:
                ax.legend(loc='upper right')
            if dim == action_dim - 1:
                ax.set_xlabel("Action Horizon", fontsize=12)

        plt.tight_layout()
        v_filename = f"pi0_pytorch_v_{filename}_denoise_steps.png"
        plt.savefig(v_filename, dpi=150)
        logging.info(f"Saved v_t velocity visualization to {v_filename}")
        plt.close(fig_v)

    def combine_denoising_visualizations(self):
        """Combine non-RTC and RTC denoising visualizations side-by-side."""
        import os

        # Check if visualization files exist
        xt_no_rtc_file = "pi0_pytorch_x_t_no_rtc_denoise_steps.png"
        xt_rtc_file = "pi0_pytorch_x_t_with_rtc_denoise_steps.png"
        v_no_rtc_file = "pi0_pytorch_v_no_rtc_denoise_steps.png"
        v_rtc_file = "pi0_pytorch_v_with_rtc_denoise_steps.png"

        files_exist = all(os.path.exists(f) for f in [xt_no_rtc_file, xt_rtc_file, v_no_rtc_file, v_rtc_file])

        if not files_exist:
            logging.warning("Some denoising visualization files not found, skipping combination")
            return

        logging.info("Combining denoising visualizations side-by-side...")

        # Load images
        xt_no_rtc_img = mpimg.imread(xt_no_rtc_file)
        xt_rtc_img = mpimg.imread(xt_rtc_file)
        v_no_rtc_img = mpimg.imread(v_no_rtc_file)
        v_rtc_img = mpimg.imread(v_rtc_file)

        # Create combined figure for x_t
        fig_xt, axes_xt = plt.subplots(1, 2, figsize=(24, 12))
        fig_xt.suptitle("X_t Denoising Trajectories Comparison", fontsize=18, fontweight='bold')

        axes_xt[0].imshow(xt_no_rtc_img)
        axes_xt[0].set_title("Without RTC", fontsize=16, fontweight='bold')
        axes_xt[0].axis('off')

        axes_xt[1].imshow(xt_rtc_img)
        axes_xt[1].set_title("With RTC", fontsize=16, fontweight='bold')
        axes_xt[1].axis('off')

        plt.tight_layout()
        combined_xt_file = "pi0_pytorch_x_t_comparison.png"
        plt.savefig(combined_xt_file, dpi=150, bbox_inches='tight')
        logging.info(f"Saved combined x_t visualization to {combined_xt_file}")
        plt.close(fig_xt)

        # Create combined figure for v_t
        fig_v, axes_v = plt.subplots(1, 2, figsize=(24, 12))
        fig_v.suptitle("V_t Velocity Trajectories Comparison", fontsize=18, fontweight='bold')

        axes_v[0].imshow(v_no_rtc_img)
        axes_v[0].set_title("Without RTC", fontsize=16, fontweight='bold')
        axes_v[0].axis('off')

        axes_v[1].imshow(v_rtc_img)
        axes_v[1].set_title("With RTC", fontsize=16, fontweight='bold')
        axes_v[1].axis('off')

        plt.tight_layout()
        combined_v_file = "pi0_pytorch_v_comparison.png"
        plt.savefig(combined_v_file, dpi=150, bbox_inches='tight')
        logging.info(f"Saved combined v_t visualization to {combined_v_file}")
        plt.close(fig_v)

@dataclass
class Args:
    """Arguments for RTC dataset evaluation."""

    # Training config name (e.g., "pi0_libero", "pi05_droid", etc.)
    train_config_name: str = field(
        metadata={"help": "Name of the training config to use (from openpi.training.config)"}
    )

    # Path to the checkpoint directory
    checkpoint_path: str = field(
        metadata={"help": "Path to the checkpoint directory"}
    )

    # Dataset configuration
    dataset_repo_id: str = field(
        metadata={"help": "HuggingFace repo ID for the LeRobot dataset"}
    )

    action_horizon: int = field(
        default=50,
        metadata={"help": "Action horizon (chunk size)"},
    )

    # If provided, will be used as default prompt
    default_prompt: str | None = field(
        default=None,
        metadata={"help": "Default prompt to use if not present in the data"},
    )

    # Seed configuration
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )

    # Action dimension
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension"},
    )

    # Additional sample kwargs to pass to the model
    sample_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Additional kwargs to pass to sample_actions"},
    )

    # Inference delay
    inference_delay: int = 1

    # Execution horizon for RTC
    execution_horizon: int = field(
        default=10,
        metadata={"help": "Execution horizon for RTC (number of timesteps for prefix weights)"},
    )

    rtc_config: RTCConfig = field(
        default_factory=lambda: RTCConfig(debug=True),  # Enable debug for tracking in evaluation
        metadata={"help": "RTC configuration for real-time control"},
    )
    
def main(args: Args):
    """Main entry point for RTC dataset evaluation."""
    # Set random seed for reproducibility
    _ = set_seed(args.seed)

    logging.info("=" * 80)
    logging.info(f"Pi0 Dataset Evaluation with config {args}")
    logging.info("=" * 80)

    evaluator = RTCDatasetEvaluator(args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Starting RTC dataset evaluation")
    main(tyro.cli(Args))
