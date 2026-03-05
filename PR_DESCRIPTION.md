## Real-Time Chunking (RTC) for PyTorch with Threshold-Based Async Inference

Implements real-time chunking for pi0.5 VLA models with a threshold-based asynchronous inference system that handles both short and long inference times. Instead of predicting latency (which fails when inference takes longer than action execution), the system monitors the action queue and triggers new inference when remaining actions drop below a configurable threshold (default 25% of action horizon).

### Architecture

**Server-side RTC inpainting** — The policy server accepts previous actions alongside new observations and uses VJP-based velocity correction during the flow-matching denoising process to produce temporally consistent action chunks.

**Client-side async inference** — A background thread monitors the action queue and proactively starts inference before the queue is exhausted, decoupling inference timing from action execution.

### Files Changed

| File | Change |
|---|---|
| `src/openpi/policies/rtc_processor_pytorch.py` | **New.** PyTorch RTCConfig and `get_prefix_weights` matching the kinetix reference implementation |
| `src/openpi/models_pytorch/pi0_pytorch.py` | **Modified.** Added `sample_actions_rtc()` with VJP-based inpainting correction during denoising |
| `src/openpi/policies/policy.py` | **Modified.** Routes to `sample_actions_rtc` when RTC params present; returns `raw_actions` for feedback |
| `modal_scripts/modal_serve_policy.py` | **Modified.** Extracts and passes through RTC parameters from WebSocket observations |
| `packages/openpi-client/src/openpi_client/rtc_client.py` | **New.** `RTCInferenceManager` with threshold-based async inference and `RTCActionQueue` |
| `examples/rtc_inference/main.py` | **New.** Example with mock robot demonstrating continuous action execution |

### Verification Against Reference Implementations

The RTC math was verified against the [original kinetix implementation](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py) and the discussion in [huggingface/lerobot#2511](https://github.com/huggingface/lerobot/issues/2511). Three issues were found and corrected:

**1. `get_prefix_weights` formula (off-by-one in denominator)**

The kinetix reference uses `clip((end - i) / (end - start + 1), 0, 1)`. Our initial implementation used `(end - i) / (end - start)` (missing +1). For `start=2, end=6, total=10`:
- Kinetix: `[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, ...]`
- Before fix: `[1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, ...]`
- After fix: matches kinetix exactly

**2. Denoiser formula (wrong time convention)**

Pi0 trains `v_t` to predict `noise - action` with time going 1→0. The correct predicted clean sample is `x_0 = x_t - time * v_t`. Our initial code used `x_t + v_t * (1 - time)` which computes the noise endpoint instead of the clean endpoint. This was independently confirmed by @shlyang's analysis in lerobot#2511:
```
pi0: time 1→0, v_t = noise - actions → actions = x_t - t * v_t
RTC: time 0→1, v_t = action - noise  → action  = x_t + (1-t) * v_t
```

**3. Correction sign and guidance weight (reversed for pi0 convention)**

Since pi0's Euler step uses negative `dt` (time 1→0), the velocity correction must use *subtraction* `v_t - guidance * correction` (not addition) so the `+|dt| * guidance * correction` term in the Euler update pushes `x_t` toward reducing the error. The guidance weight formula also needed adaptation via `tau = 1 - time` before applying the kinetix formulas. All three implementations (kinetix, lerobot, ours) now produce identical guidance weights at every timestep.

**One thing we get right that lerobot gets wrong:** Our `x_t.requires_grad_(True)` is placed *before* the model forward pass, so `torch.autograd.grad` computes the true VJP through the model's Jacobian. LeRobot places it *after*, making `correction = error` (identity Jacobian). @shlyang's experiments in the issue showed that proper VJP correction produces significantly better trajectory matching (verified with cosine similarity analysis and MAE sweep experiments).

### Test Results (Mock Policy)

| Inference Latency | Hit Rate | Effective Hz | Notes |
|---|---|---|---|
| 20ms | 99.2% | 49.2 Hz | Near-continuous actions |
| 100ms | 98.0% | 48.7 Hz | Nearly no idle time |
| 150ms | 98.0% | 48.7 Hz | Pre-fetching eliminates most gaps |
| 800ms | 51.0% | 25.3 Hz | Graceful degradation, no crashes |
