"""Tests for session 15 improvements: seed reproducibility, weight decay,
LR warmup, gradient accumulation, and training summary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

# ===================================================================
# Seed reproducibility
# ===================================================================


class TestSeedReproducibility:
    """MAPPOTrainer produces deterministic results for the same seed."""

    def test_same_seed_same_weights(self) -> None:
        """Two trainers with the same seed have identical initial weights."""
        cfg = MAPPOConfig(rollout_length=8)
        t1 = MAPPOTrainer(mappo_config=cfg, seed=42)
        t2 = MAPPOTrainer(mappo_config=cfg, seed=42)
        for name in ("vessel", "port", "coordinator"):
            p1 = list(t1.actor_critics[name].parameters())
            p2 = list(t2.actor_critics[name].parameters())
            for a, b in zip(p1, p2):
                assert torch.allclose(a, b), f"{name} params differ with same seed"

    def test_different_seed_different_weights(self) -> None:
        """Two trainers with different seeds have different initial weights."""
        cfg = MAPPOConfig(rollout_length=8)
        t1 = MAPPOTrainer(mappo_config=cfg, seed=42)
        t2 = MAPPOTrainer(mappo_config=cfg, seed=999)
        any_diff = False
        for name in ("vessel", "port", "coordinator"):
            p1 = list(t1.actor_critics[name].parameters())
            p2 = list(t2.actor_critics[name].parameters())
            for a, b in zip(p1, p2):
                if not torch.allclose(a, b):
                    any_diff = True
                    break
        assert any_diff, "Different seeds should produce different weights"

    def test_same_seed_same_rollout(self) -> None:
        """Same seed → same rollout rewards (within float tolerance)."""
        cfg = MAPPOConfig(rollout_length=8)
        t1 = MAPPOTrainer(mappo_config=cfg, seed=77)
        info1 = t1.collect_rollout()

        t2 = MAPPOTrainer(mappo_config=cfg, seed=77)
        info2 = t2.collect_rollout()

        assert abs(info1["mean_reward"] - info2["mean_reward"]) < 1e-6

    def test_seed_stored_on_trainer(self) -> None:
        """The seed value is stored for introspection."""
        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=123)
        assert trainer._seed == 123


# ===================================================================
# Weight decay
# ===================================================================


class TestWeightDecay:
    """MAPPOConfig.weight_decay is applied to the Adam optimizer."""

    def test_default_zero(self) -> None:
        """Default weight_decay is 0."""
        cfg = MAPPOConfig()
        assert cfg.weight_decay == 0.0

    def test_weight_decay_applied_to_optimizer(self) -> None:
        """When weight_decay > 0, optimizer param groups reflect it."""
        cfg = MAPPOConfig(rollout_length=8, weight_decay=1e-4)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        for name, opt in trainer.optimizers.items():
            for pg in opt.param_groups:
                assert pg["weight_decay"] == 1e-4, f"{name} optimizer missing weight_decay"

    def test_training_with_weight_decay(self) -> None:
        """Training completes successfully with weight_decay enabled."""
        cfg = MAPPOConfig(
            rollout_length=8, num_epochs=1, minibatch_size=8, weight_decay=1e-3
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=2)
        assert len(history) == 2
        assert all("mean_reward" in h for h in history)


# ===================================================================
# LR warmup
# ===================================================================


class TestLRWarmup:
    """LR warmup phase ramps from 0 to lr before annealing."""

    def test_default_no_warmup(self) -> None:
        """Default lr_warmup_fraction is 0."""
        cfg = MAPPOConfig()
        assert cfg.lr_warmup_fraction == 0.0

    def test_warmup_ramps_lr(self) -> None:
        """During warmup, LR increases from ~0 toward lr."""
        cfg = MAPPOConfig(
            rollout_length=8,
            num_epochs=1,
            minibatch_size=8,
            lr=3e-4,
            lr_end=0.0,
            lr_warmup_fraction=0.5,
            total_iterations=10,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)

        # After 1 iteration (warmup: 5 iterations total)
        trainer.collect_rollout()
        trainer.update()
        lr_1 = trainer.current_lr
        # LR should be lr * (1/5) = 3e-4 * 0.2 = 6e-5
        assert lr_1 < cfg.lr, f"LR {lr_1} should be less than {cfg.lr} during warmup"
        assert lr_1 > 0.0, "LR should be > 0 after first warmup step"

        # After 5 iterations (end of warmup), LR should be near lr
        for _ in range(4):
            trainer.collect_rollout()
            trainer.update()
        lr_5 = trainer.current_lr
        assert abs(lr_5 - cfg.lr) < 1e-6, f"LR should equal lr at warmup end, got {lr_5}"

    def test_warmup_then_annealing(self) -> None:
        """After warmup, LR anneals toward lr_end."""
        cfg = MAPPOConfig(
            rollout_length=8,
            num_epochs=1,
            minibatch_size=8,
            lr=1e-3,
            lr_end=1e-5,
            lr_warmup_fraction=0.2,
            total_iterations=10,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)

        # Run 10 iterations
        lrs: list[float] = []
        for _ in range(10):
            trainer.collect_rollout()
            trainer.update()
            lrs.append(trainer.current_lr)

        # Warmup: iterations 1-2 (warmup_fraction=0.2, total=10 → 2 warmup iters)
        # LR at iteration 2 should be near lr
        assert lrs[1] >= lrs[0], "LR should increase during warmup"

        # After warmup, LR should decrease
        assert lrs[-1] < lrs[2], "LR should decrease during annealing phase"

    def test_no_warmup_annealing_only(self) -> None:
        """With warmup_fraction=0, behaves as pure linear annealing."""
        cfg = MAPPOConfig(
            rollout_length=8,
            num_epochs=1,
            minibatch_size=8,
            lr=1e-3,
            lr_end=1e-5,
            lr_warmup_fraction=0.0,
            total_iterations=4,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)

        lrs: list[float] = []
        for _ in range(4):
            trainer.collect_rollout()
            trainer.update()
            lrs.append(trainer.current_lr)

        # Pure annealing: should be monotonically decreasing
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1], "LR should monotonically decrease"


# ===================================================================
# Gradient accumulation
# ===================================================================


class TestGradientAccumulation:
    """Gradient accumulation simulates larger effective batch sizes."""

    def test_default_no_accumulation(self) -> None:
        """Default grad_accumulation_steps is 1 (no accumulation)."""
        cfg = MAPPOConfig()
        assert cfg.grad_accumulation_steps == 1

    def test_accumulation_trains_successfully(self) -> None:
        """Training completes with grad_accumulation_steps > 1."""
        cfg = MAPPOConfig(
            rollout_length=16,
            num_epochs=2,
            minibatch_size=4,
            grad_accumulation_steps=2,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=3)
        assert len(history) == 3
        # All entries should have losses
        for h in history:
            assert "vessel_value_loss" in h
            assert "port_policy_loss" in h

    def test_accumulation_produces_valid_metrics(self) -> None:
        """Update returns finite metrics with accumulation enabled."""
        cfg = MAPPOConfig(
            rollout_length=16,
            num_epochs=1,
            minibatch_size=4,
            grad_accumulation_steps=4,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        trainer.collect_rollout()
        results = trainer.update()
        for agent_type, res in results.items():
            assert np.isfinite(res.policy_loss), f"{agent_type} infinite policy_loss"
            assert np.isfinite(res.value_loss), f"{agent_type} infinite value_loss"
            assert np.isfinite(res.grad_norm), f"{agent_type} infinite grad_norm"
            assert np.isfinite(res.explained_variance), f"{agent_type} infinite ev"

    def test_accumulation_step_1_equivalent(self) -> None:
        """grad_accumulation_steps=1 should match default behaviour."""
        cfg1 = MAPPOConfig(
            rollout_length=8, num_epochs=1, minibatch_size=8,
            grad_accumulation_steps=1,
        )
        cfg_default = MAPPOConfig(
            rollout_length=8, num_epochs=1, minibatch_size=8,
        )
        t1 = MAPPOTrainer(mappo_config=cfg1, seed=42)
        t1.collect_rollout()
        r1 = t1.update()

        t2 = MAPPOTrainer(mappo_config=cfg_default, seed=42)
        t2.collect_rollout()
        r2 = t2.update()

        # Loss values should be close (same seed, same effective config)
        for agent in ("vessel", "port", "coordinator"):
            diff = abs(r1[agent].value_loss - r2[agent].value_loss)
            assert diff < 1e-4, f"{agent} value_loss differs: {diff}"


# ===================================================================
# Training summary
# ===================================================================


class TestTrainingSummary:
    """MAPPOTrainer.training_summary() aggregates training stats."""

    def test_empty_history(self) -> None:
        """Empty history returns minimal summary."""
        summary = MAPPOTrainer.training_summary([])
        assert summary["total_iterations"] == 0

    def test_summary_fields(self) -> None:
        """Summary includes all expected aggregate fields."""
        cfg = MAPPOConfig(rollout_length=8, num_epochs=1, minibatch_size=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=5)
        summary = MAPPOTrainer.training_summary(history)

        assert summary["total_iterations"] == 5
        assert "final_mean_reward" in summary
        assert "best_mean_reward" in summary
        assert "best_iteration" in summary
        assert "reward_improvement" in summary
        assert "final_lr" in summary
        assert "final_entropy_coeff" in summary

    def test_summary_per_agent_stats(self) -> None:
        """Summary includes per-agent mean losses and explained variance."""
        cfg = MAPPOConfig(rollout_length=8, num_epochs=1, minibatch_size=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=3)
        summary = MAPPOTrainer.training_summary(history)

        for agent in ("vessel", "port", "coordinator"):
            assert f"mean_{agent}_value_loss" in summary
            assert f"mean_{agent}_policy_loss" in summary
            assert f"mean_{agent}_explained_variance" in summary
            assert f"final_{agent}_explained_variance" in summary

    def test_best_iteration_correct(self) -> None:
        """Best iteration tracks the highest mean_reward."""
        history: list[dict[str, Any]] = [
            {"mean_reward": 1.0},
            {"mean_reward": 5.0},
            {"mean_reward": 3.0},
        ]
        summary = MAPPOTrainer.training_summary(history)
        assert summary["best_mean_reward"] == 5.0
        assert summary["best_iteration"] == 1

    def test_reward_improvement(self) -> None:
        """reward_improvement = final - first."""
        history: list[dict[str, Any]] = [
            {"mean_reward": 1.0},
            {"mean_reward": 4.0},
        ]
        summary = MAPPOTrainer.training_summary(history)
        assert abs(summary["reward_improvement"] - 3.0) < 1e-6


# ===================================================================
# Combined integration test
# ===================================================================


class TestCombinedFeatures:
    """All new features work together."""

    def test_full_training_with_all_features(self) -> None:
        """Training works with warmup + weight_decay + grad_accum + seed."""
        cfg = MAPPOConfig(
            rollout_length=16,
            num_epochs=2,
            minibatch_size=4,
            lr=1e-3,
            lr_end=1e-5,
            lr_warmup_fraction=0.3,
            weight_decay=1e-4,
            grad_accumulation_steps=2,
            total_iterations=6,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=6)
        summary = MAPPOTrainer.training_summary(history)

        assert summary["total_iterations"] == 6
        assert np.isfinite(summary["final_mean_reward"])
        assert summary["final_lr"] < cfg.lr  # annealed
        assert summary["final_lr"] > 0  # not zero yet

    def test_curriculum_with_warmup(self) -> None:
        """Curriculum + LR warmup combination works."""
        from hmarl_mvp.curriculum import CurriculumScheduler

        target = {"num_vessels": 4, "num_ports": 3, "rollout_steps": 15}
        curriculum = CurriculumScheduler(
            target_config=target, warmup_fraction=0.5
        )
        cfg = MAPPOConfig(
            rollout_length=8,
            num_epochs=1,
            minibatch_size=8,
            lr=3e-4,
            lr_warmup_fraction=0.3,
            total_iterations=4,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=4, curriculum=curriculum)
        assert len(history) == 4
