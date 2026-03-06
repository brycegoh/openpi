# Training-Time Action Conditioning (TTAC)

Training-Time Action Conditioning teaches the model to handle inference delay
during training. It feeds the **ground-truth action prefix** to the model and
trains only on the remaining postfix actions. This keeps chunk transitions
smooth without doing any inference-time inpainting (VJP/gradient computation).

Based on: [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964)
(Physical Intelligence, 2024).

---

## Why TTAC?

Standard action chunking policies generate a full chunk of future actions, then
execute them while generating the next chunk. The **inference delay** (time to
generate a new chunk) means the first few actions in each new chunk overlap with
already-committed actions from the previous chunk. Without conditioning, the new
chunk may be inconsistent with the committed prefix.

**Inference-time RTC** (already in this repo as `sample_actions_rtc`) solves this
with VJP-based correction at each denoising step, but this adds compute cost.

**TTAC** instead teaches the model at training time what a "committed prefix"
looks like, so at inference the model naturally produces consistent continuations
with **zero extra compute** -- just replace the prefix and set its timestep.

---

## How It Works

### Flow Matching Convention

OpenPI uses the **1-to-0** flow matching convention:

```
x_t = t * noise + (1 - t) * actions
```

- `t = 0` means **clean** (fully denoised) actions
- `t = 1` means **pure noise**
- Denoising proceeds from `t = 1` down to `t = 0`

> **Note:** The original kinetix repo uses the opposite 0-to-1 convention where
> `t = 1` is clean. All timestep values in this implementation are adapted
> accordingly. If you see the kinetix paper or code reference `t = 1.0` for
> prefix tokens, that corresponds to our `t = 0.0`.

### At Training Time

For each batch element:

1. **Sample a delay** `d` from `[min_delay, max_delay]` (uniform or exponential).
2. **Create per-token timesteps:** the first `d` positions (prefix) get `t = 0.0`
   (ground truth); the remaining positions (postfix) get the normally sampled `t`.
3. **Interpolate per token:** `x_t[i] = t[i] * noise[i] + (1 - t[i]) * action[i]`.
   Prefix positions become the exact ground-truth actions (since `t = 0`).
4. **Forward pass** with per-token timesteps through the model.
5. **Mask the loss** to only backpropagate through postfix positions.

### At Inference Time

When generating a new chunk with previous committed actions available:

1. At each denoising step, **replace** the first `d` positions of `x_t` with the
   previous chunk's leftover actions.
2. **Set their timestep** to `0.0` (fully denoised).
3. Run the model forward with per-token timesteps.
4. Apply the Euler step to all positions. The prefix gets replaced again on the
   next step.

The model has learned during training that tokens at `t = 0` are ground truth
context, so it naturally produces smooth continuations.

---

## Configuration

TTAC is configured through `TTACConfig` on the model's `Pi0Config`:

```python
from openpi.policies.ttac import TTACConfig
import openpi.models.pi0_config as pi0_config

model_cfg = pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=50,
    ttac_config=TTACConfig(
        enabled=True,       # Toggle TTAC on/off
        min_delay=0,         # Minimum delay (inclusive)
        max_delay=6,         # Maximum delay (inclusive)
        delay_distribution="UNIFORM",  # "UNIFORM" or "EXP"
        exp_decay=1.0,       # Decay rate for EXP distribution
    ),
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Enable TTAC during training |
| `min_delay` | `int` | `0` | Minimum number of prefix actions |
| `max_delay` | `int` | `6` | Maximum number of prefix actions |
| `delay_distribution` | `str` | `"UNIFORM"` | `"UNIFORM"` or `"EXP"` |
| `exp_decay` | `float` | `1.0` | Exponential decay factor (EXP only) |

### Delay Distribution

The delay distribution controls how often each delay value `d` is sampled during
training. This directly affects how much training signal the model receives for
each prefix length.

#### `UNIFORM`

Every delay in `[min_delay, max_delay]` is sampled with equal probability.

```
delay:  0   1   2   3   4   5
prob:  1/6 1/6 1/6 1/6 1/6 1/6    (for min_delay=0, max_delay=5)
```

**When to use:** Good general-purpose default. Gives equal training signal to
all delay values, which is appropriate when you don't know your inference latency
in advance or want the model to handle any delay equally well.

#### `EXP`

Smaller delays are exponentially more likely, with probability proportional to
`exp(-exp_decay * d)`. Larger `exp_decay` values concentrate more probability
mass on small delays.

```
delay:  0      1      2      3      4       (exp_decay=1.0)
weight: 1.000  0.368  0.135  0.050  0.018
prob:   63.6%  23.4%  8.6%   3.2%   1.2%
```

This is what the original paper uses. The authors explain:

> *"We sample delays from {0, 1, 2, 3, 4} with exponentially decreasing
> weights, as we found that higher delays need less training supervision.
> Better results could likely be obtained by spending more training compute
> training individual checkpoints for each delay."*

The reasoning is that **delay=0 is the hardest case** for the model: it receives
no prefix context at all and must generate the entire chunk from scratch (this is
standard unconditional generation). As the delay increases, the model gets more
ground-truth prefix context to condition on, making the prediction task easier.
Therefore:

- **delay=0** (no prefix): The model must learn full unconditional generation.
  This is the baseline capability and needs the most training signal.
- **delay=1-2** (short prefix): The model gets minimal context. Still
  challenging, still needs significant training signal.
- **delay=3-4+** (long prefix): The model gets substantial context. The
  remaining postfix is short and strongly constrained by the prefix, so the
  model learns this regime with relatively few examples.

The exponential weighting reflects this diminishing marginal value: each
additional prefix action makes the task easier, so the model needs proportionally
less training time on that delay value.

**When to use:** Recommended when you want to match the paper's approach. Also
good when your expected inference delay is usually small (1-3 steps) but you
want coverage up to a worst-case delay. The exponential distribution naturally
allocates training budget in proportion to task difficulty.

#### `exp_decay` parameter

Controls the steepness of the exponential. Only used with `EXP` distribution.

| `exp_decay` | Effect | delay=0 vs delay=4 ratio |
|---|---|---|
| 0.5 | Gentle decay, closer to uniform | 7.4x |
| 1.0 | Paper default | 54.6x |
| 2.0 | Aggressive, mostly delay=0 | 2981x |

- `exp_decay=1.0` matches the original kinetix paper.
- Lower values (0.3-0.7) give a softer version of EXP, more like a compromise
  between UNIFORM and the paper's approach.
- Higher values (1.5+) concentrate almost all training on delay=0.

### Guidelines

- Set `max_delay` to your expected **worst-case inference delay** in timesteps
  (typically 4-8 for most setups).
- `min_delay=0` is important: it ensures the model still trains on the
  unconditional case (no prefix), preserving its ability to generate the first
  chunk where no previous actions exist.
- `max_delay` must be strictly less than `action_horizon`.
- Start with `UNIFORM` for simplicity. Switch to `EXP` with `exp_decay=1.0` if
  you want to match the paper or if you observe that the model underperforms at
  small delays.

---

## Training

### Pre-built Configs

Several TTAC configs are available in `src/openpi/training/config.py`:

| Config Name | Model | Description |
|---|---|---|
| `debug_ttac` | Pi0 (dummy) | Quick local test, 10 steps |
| `debug_pi05_ttac` | Pi0.5 (dummy) | Quick local test, 10 steps |
| `pi05_tcr_ttac_pytorch` | Pi0.5 (3-cam) | Production TCR training with TTAC |
| `pi05_tcr_4cam_ttac_pytorch` | Pi0.5 (4-cam) | Production 4-camera TCR with TTAC |

### Running Training

Single GPU:

```bash
python scripts/train_pytorch.py pi05_tcr_ttac_pytorch \
    --exp_name my_ttac_run
```

Multi-GPU:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_tcr_ttac_pytorch \
    --exp_name my_ttac_run
```

### Custom Config

To add TTAC to any existing config, add `ttac_config` to the model:

```python
TrainConfig(
    name="my_ttac_config",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        ttac_config=TTACConfig(
            enabled=True,
            min_delay=0,
            max_delay=6,
        ),
    ),
    data=YourDataConfig(...),
    pytorch_weight_path="/path/to/base/weights",
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    num_train_steps=30_000,
    batch_size=256,
)
```

No changes to the training loop or data loading are required. TTAC is activated
automatically inside `model.forward()` when `ttac_config.enabled=True` and the
model is in training mode.

### Verifying TTAC is Active

The training script logs TTAC status at startup:

```
TTAC enabled: delay=[0, 6], distribution=UNIFORM
```

And in the training info block:

```
TTAC: enabled, delay=[0, 6], distribution=UNIFORM
```

---

## Inference

### Using TTAC at Inference

After training with TTAC, use `sample_actions_ttac()` for inference with
previous-chunk conditioning:

```python
# First chunk: no previous actions, falls back to standard sampling
actions = model.sample_actions(device, observation)

# Subsequent chunks: condition on previous chunk's leftover
actions = model.sample_actions_ttac(
    device,
    observation,
    prev_chunk_leftover=previous_actions,  # (B, action_horizon, action_dim)
    inference_delay=5,                      # number of committed prefix actions
)
```

When `prev_chunk_leftover` is `None` or `inference_delay <= 0`,
`sample_actions_ttac` falls back to standard `sample_actions`.

### TTAC vs Inference-Time RTC

| | TTAC (`sample_actions_ttac`) | RTC (`sample_actions_rtc`) |
|---|---|---|
| Requires special training | Yes (TTAC-enabled) | No (works with any model) |
| Inference compute overhead | **None** | VJP + guidance per step |
| Gradient computation | Not needed | Required (`torch.enable_grad`) |
| Memory overhead | None | Stores grad graph |
| Chunk transition quality | Comparable (per paper) | Comparable |

---

## What Changed (Implementation Details)

### New Files

- **`src/openpi/policies/ttac.py`** -- Core TTAC module:
  - `TTACConfig` -- Configuration dataclass
  - `sample_ttac_delay()` -- Per-batch delay sampling (UNIFORM/EXP)
  - `apply_ttac_training()` -- Creates per-token timesteps and postfix mask
  - `masked_mean()` -- Loss masking that normalizes by valid positions (matches
    kinetix reference)
  - `apply_ttac_inference()` -- Replaces prefix in `x_t` and creates per-token
    timesteps for inference

- **`tests/test_ttac.py`** -- 27 unit tests covering config validation, delay
  sampling, per-token time construction, masked loss, inference conditioning,
  sinusoidal embedding with 2D time, and flow matching integration.

### Modified Files

- **`src/openpi/models/pi0_config.py`** -- Added `ttac_config: TTACConfig | None`
  field to `Pi0Config`.

- **`src/openpi/models_pytorch/pi0_pytorch.py`**:
  - `create_sinusoidal_pos_embedding()` now accepts `(B,)` or `(B, T)` time
    tensors, returning `(B, D)` or `(B, T, D)` respectively.
  - `embed_suffix()` guards the `time_emb` expansion with `if time_emb.ndim == 2`
    so per-token `(B, T, D)` embeddings pass through without incorrect reshaping.
  - `forward()` checks `self.config.ttac_config` during training to sample delays,
    create per-token timesteps, and apply masked loss.
  - New `sample_actions_ttac()` method for inference with previous-chunk
    conditioning.

- **`src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`**
  -- Fixed `GemmaRMSNorm.forward()` to only unsqueeze the modulation tensor when
  it is 2D (global conditioning). Per-token 3D conditioning from TTAC already has
  the sequence dimension and must not be unsqueezed again. This is required for
  Pi0.5's adaRMS to work correctly with per-token timesteps.

- **`scripts/train_pytorch.py`**:
  - Model config conversion now propagates `ttac_config` and `rtc_config`.
  - Validates `max_delay < action_horizon` at startup.
  - Logs TTAC status in the training info block.

- **`src/openpi/training/config.py`** -- Added debug and production TTAC configs.

---

## Reference Implementation Comparison

This implementation was cross-referenced against two sources:

1. **[Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)**
   (JAX, original paper code)
2. **[huggingface/lerobot PR #2830](https://github.com/huggingface/lerobot/pull/2830)**
   (PyTorch, in-progress port)

### Convention Adaptation

The kinetix repo uses the 0-to-1 convention (`t=1` is clean), while OpenPI and
LeRobot use the 1-to-0 convention (`t=0` is clean). All timestep values are
adapted accordingly:

| | Kinetix | OpenPI / LeRobot |
|---|---|---|
| Prefix time (training) | 1.0 | 0.0 |
| Prefix time (inference) | 1.0 | 0.0 |

### Known Issue in LeRobot PR #2830

The LeRobot PR sets prefix timestep to `0.0` at training but `1.0` at inference.
In the 1-to-0 convention, `t=1.0` means pure noise, creating a train/inference
mismatch. Our implementation correctly uses `0.0` (clean) in both paths,
matching the kinetix behavior after convention adaptation.

### Feature Comparison

| Feature | Kinetix | LeRobot | Ours |
|---|---|---|---|
| Delay distribution | EXP only | UNIFORM + EXP | UNIFORM + EXP |
| Delay range | `[0, d)` | `[min, max]` | `[min, max]` |
| Loss normalization | By positions | By positions | By positions |
| Inference prefix time | Correct | **Bug (1.0)** | Correct (0.0) |
| Per-token adaLN/adaRMS | MLP-Mixer adaLN | Not shown | GemmaRMSNorm fix |
| Final prefix clamp | No | No | Yes (safety) |
