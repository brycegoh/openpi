"""Tests for Training-Time Action Conditioning (TTAC)."""

import torch
import pytest

from openpi.policies.ttac import (
    TTACConfig,
    TTACDelayDistribution,
    apply_ttac_inference,
    apply_ttac_training,
    masked_mean,
    sample_ttac_delay,
)


class TestTTACConfig:
    def test_defaults(self):
        cfg = TTACConfig()
        assert cfg.enabled is False
        assert cfg.min_delay == 0
        assert cfg.max_delay == 6
        assert cfg.delay_distribution == TTACDelayDistribution.UNIFORM
        assert cfg.exp_decay == 1.0

    def test_invalid_min_delay(self):
        with pytest.raises(ValueError, match="min_delay must be >= 0"):
            TTACConfig(min_delay=-1)

    def test_invalid_max_lt_min(self):
        with pytest.raises(ValueError, match="max_delay .* must be >= min_delay"):
            TTACConfig(min_delay=5, max_delay=3)

    def test_invalid_exp_decay(self):
        with pytest.raises(ValueError, match="exp_decay must be positive"):
            TTACConfig(exp_decay=0.0)
        with pytest.raises(ValueError, match="exp_decay must be positive"):
            TTACConfig(exp_decay=-1.0)


class TestSampleDelay:
    def test_uniform_range(self):
        cfg = TTACConfig(min_delay=2, max_delay=8)
        delays = sample_ttac_delay(cfg, batch_size=1000, device=torch.device("cpu"))
        assert delays.shape == (1000,)
        assert delays.min() >= 2
        assert delays.max() <= 8

    def test_exp_range(self):
        cfg = TTACConfig(min_delay=0, max_delay=5, delay_distribution=TTACDelayDistribution.EXP)
        delays = sample_ttac_delay(cfg, batch_size=1000, device=torch.device("cpu"))
        assert delays.shape == (1000,)
        assert delays.min() >= 0
        assert delays.max() <= 5

    def test_exp_weights_by_absolute_delay(self):
        """EXP distribution weights by absolute delay value, not relative index."""
        cfg = TTACConfig(min_delay=2, max_delay=4, delay_distribution=TTACDelayDistribution.EXP, exp_decay=1.0)
        delays = sample_ttac_delay(cfg, batch_size=5000, device=torch.device("cpu"))
        counts = torch.bincount(delays, minlength=5)
        assert counts[2] > counts[3] > counts[4]

    def test_single_value(self):
        cfg = TTACConfig(min_delay=3, max_delay=3)
        delays = sample_ttac_delay(cfg, batch_size=10, device=torch.device("cpu"))
        assert (delays == 3).all()
        assert delays.dtype == torch.long


class TestApplyTTACTraining:
    def test_basic_masks(self):
        time = torch.tensor([0.5, 0.7])
        delay = torch.tensor([2, 3])
        seq_len = 5

        time_tokens, postfix_mask = apply_ttac_training(time, delay, seq_len)

        assert time_tokens.shape == (2, 5)
        assert postfix_mask.shape == (2, 5)

        # Batch 0: delay=2, so prefix=[0,1], postfix=[2,3,4]
        assert time_tokens[0, 0].item() == 0.0
        assert time_tokens[0, 1].item() == 0.0
        assert time_tokens[0, 2].item() == pytest.approx(0.5)
        assert time_tokens[0, 3].item() == pytest.approx(0.5)
        assert time_tokens[0, 4].item() == pytest.approx(0.5)

        assert postfix_mask[0].tolist() == [False, False, True, True, True]

        # Batch 1: delay=3, so prefix=[0,1,2], postfix=[3,4]
        assert time_tokens[1, 0].item() == 0.0
        assert time_tokens[1, 1].item() == 0.0
        assert time_tokens[1, 2].item() == 0.0
        assert time_tokens[1, 3].item() == pytest.approx(0.7)
        assert time_tokens[1, 4].item() == pytest.approx(0.7)

        assert postfix_mask[1].tolist() == [False, False, False, True, True]

    def test_zero_delay(self):
        time = torch.tensor([0.5])
        delay = torch.tensor([0])
        time_tokens, postfix_mask = apply_ttac_training(time, delay, 4)

        assert postfix_mask[0].tolist() == [True, True, True, True]
        assert (time_tokens[0] == 0.5).all()

    def test_full_delay(self):
        time = torch.tensor([0.5])
        delay = torch.tensor([4])
        time_tokens, postfix_mask = apply_ttac_training(time, delay, 4)

        assert postfix_mask[0].tolist() == [False, False, False, False]
        assert (time_tokens[0] == 0.0).all()

    def test_delay_exceeds_seq_len_is_clamped(self):
        """Delay > seq_len should be clamped to seq_len (all prefix)."""
        time = torch.tensor([0.5])
        delay = torch.tensor([10])
        time_tokens, postfix_mask = apply_ttac_training(time, delay, 4)

        assert postfix_mask[0].tolist() == [False, False, False, False]
        assert (time_tokens[0] == 0.0).all()


class TestMaskedMean:
    def test_all_true(self):
        losses = torch.ones(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        result = masked_mean(losses, mask)
        # 8 valid positions, each summing 3 elements = total 24, denom = 8 (positions)
        assert result.item() == pytest.approx(3.0)

    def test_partial_mask(self):
        losses = torch.ones(1, 4, 2) * 2.0
        mask = torch.tensor([[False, False, True, True]])
        result = masked_mean(losses, mask)
        # 2 valid positions, each summing 2*2=4, total=8, denom=2 (positions)
        assert result.item() == pytest.approx(4.0)

    def test_all_false(self):
        losses = torch.ones(1, 4, 2)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        result = masked_mean(losses, mask)
        assert result.item() == pytest.approx(0.0)

    def test_none_mask(self):
        losses = torch.ones(2, 4, 3) * 3.0
        result = masked_mean(losses, None)
        assert result.item() == pytest.approx(3.0)

    def test_reduce_dims(self):
        losses = torch.ones(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[0, :2] = False
        result = masked_mean(losses, mask, reduce_dims=(1, 2))
        assert result.shape == (2,)
        # Batch 0: 2 valid positions, sum=6, denom=2 → 3.0
        assert result[0].item() == pytest.approx(3.0)
        # Batch 1: 4 valid positions, sum=12, denom=4 → 3.0
        assert result[1].item() == pytest.approx(3.0)

    def test_matches_kinetix_normalization(self):
        """Verify normalization matches kinetix: sum(loss*mask)/sum(mask) where mask is (B,T,1)."""
        B, T, D = 2, 6, 4
        losses = torch.randn(B, T, D)
        mask_2d = torch.tensor([[True, True, False, False, True, True],
                                [False, True, True, True, True, False]])
        result = masked_mean(losses, mask_2d)

        # Manual kinetix-style computation
        mask_3d = mask_2d.unsqueeze(-1).float()  # (B, T, 1)
        expected = (losses * mask_3d).sum() / mask_3d.sum()
        assert result.item() == pytest.approx(expected.item(), rel=1e-5)


class TestApplyTTACInference:
    def test_no_prev_actions(self):
        x_t = torch.randn(2, 5, 3)
        x_t_cond, time_tokens = apply_ttac_inference(x_t, 0.8, 3, None, 5)
        assert torch.equal(x_t_cond, x_t)
        assert (time_tokens == 0.8).all()

    def test_zero_delay(self):
        x_t = torch.randn(2, 5, 3)
        prev = torch.randn(2, 5, 3)
        x_t_cond, time_tokens = apply_ttac_inference(x_t, 0.8, 0, prev, 5)
        assert torch.equal(x_t_cond, x_t)
        assert (time_tokens == 0.8).all()

    def test_with_conditioning(self):
        x_t = torch.randn(2, 5, 3)
        prev = torch.ones(2, 5, 3) * 99.0
        x_t_cond, time_tokens = apply_ttac_inference(x_t, 0.6, 2, prev, 5)

        # Prefix replaced with prev actions
        assert (x_t_cond[:, :2] == 99.0).all()
        # Postfix unchanged
        assert torch.equal(x_t_cond[:, 2:], x_t[:, 2:])

        # Time tokens
        assert (time_tokens[:, :2] == 0.0).all()
        assert (time_tokens[:, 2:] == 0.6).all()


def _create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    """Standalone copy for testing without full pi0_pytorch import chain."""
    import math
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    reshape_back = None
    if time.ndim == 2:
        B, T = time.shape
        reshape_back = (B, T, dimension)
        time = time.reshape(-1)
    elif time.ndim != 1:
        raise ValueError(f"Expected 1D or 2D time tensor, got {time.ndim}D.")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    result = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    if reshape_back is not None:
        result = result.reshape(reshape_back)
    return result


class TestSinusoidalEmbedding:
    def test_1d_time(self):
        time = torch.tensor([0.3, 0.7])
        result = _create_sinusoidal_pos_embedding(time, 16, 4e-3, 4.0, device=torch.device("cpu"))
        assert result.shape == (2, 16)

    def test_2d_time(self):
        time = torch.tensor([[0.0, 0.3, 0.5], [0.0, 0.0, 0.7]])
        result = _create_sinusoidal_pos_embedding(time, 16, 4e-3, 4.0, device=torch.device("cpu"))
        assert result.shape == (2, 3, 16)

    def test_2d_matches_1d(self):
        time_2d = torch.tensor([[0.3, 0.5, 0.7]])
        result_2d = _create_sinusoidal_pos_embedding(time_2d, 16, 4e-3, 4.0, device=torch.device("cpu"))

        time_1d = torch.tensor([0.3, 0.5, 0.7])
        result_1d = _create_sinusoidal_pos_embedding(time_1d, 16, 4e-3, 4.0, device=torch.device("cpu"))

        torch.testing.assert_close(result_2d[0], result_1d)

    def test_prefix_at_zero_time(self):
        """Verify that prefix positions at t=0 produce a consistent embedding."""
        time = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        result = _create_sinusoidal_pos_embedding(time, 16, 4e-3, 4.0, device=torch.device("cpu"))

        torch.testing.assert_close(result[0, 0], result[0, 1])
        torch.testing.assert_close(result[0, 2], result[0, 3])
        assert not torch.equal(result[0, 0], result[0, 2])


class TestFlowMatchingIntegration:
    """Verify TTAC integrates correctly with the flow matching interpolation."""

    def test_prefix_actions_are_clean(self):
        """With TTAC, prefix positions (t=0) should have x_t = actions (no noise)."""
        batch_size, seq_len, action_dim = 2, 8, 4
        actions = torch.randn(batch_size, seq_len, action_dim)
        noise = torch.randn(batch_size, seq_len, action_dim)
        time = torch.tensor([0.5, 0.8])
        delay = torch.tensor([3, 2])

        time_tokens, postfix_mask = apply_ttac_training(time, delay, seq_len)
        time_expanded = time_tokens[:, :, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # Prefix positions should be clean (x_t = actions at t=0)
        assert torch.allclose(x_t[0, :3], actions[0, :3])
        assert torch.allclose(x_t[1, :2], actions[1, :2])

        # Postfix positions should be noisy
        assert not torch.allclose(x_t[0, 3:], actions[0, 3:])
        assert not torch.allclose(x_t[1, 2:], actions[1, 2:])

    def test_loss_mask_excludes_prefix(self):
        """Verify masked_mean only counts postfix positions."""
        batch_size, seq_len, action_dim = 1, 6, 4
        delay = torch.tensor([2])
        time = torch.tensor([0.5])
        _, postfix_mask = apply_ttac_training(time, delay, seq_len)

        losses = torch.ones(batch_size, seq_len, action_dim)
        losses[:, :2] = 999.0  # Large values in prefix (should be masked)

        result = masked_mean(losses, postfix_mask)
        # 4 valid positions, each with D=4 ones, sum=16, denom=4 positions → 4.0
        assert result.item() == pytest.approx(float(action_dim))
