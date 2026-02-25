"""Tests for LR scheduling, value clipping, checkpointing, and early stopping."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hmarl_mvp.checkpointing import EarlyStopping, TrainingCheckpoint
from hmarl_mvp.config import get_default_config
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

# =====================================================================
# LR scheduler
# =====================================================================


class TestLRScheduler:
    """Test linear learning-rate annealing."""

    def test_lr_anneals_linearly(self) -> None:
        """LR should decay from lr to lr_end over total_iterations."""
        cfg = MAPPOConfig(
            lr=1e-3,
            lr_end=1e-5,
            total_iterations=10,
            rollout_length=4,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=1)
        initial_lr = trainer.current_lr
        assert abs(initial_lr - 1e-3) < 1e-8

        # Run 5 iterations (halfway)
        for _ in range(5):
            trainer.collect_rollout()
            trainer.update()

        mid_lr = trainer.current_lr
        expected_mid = 1e-5 + 0.5 * (1e-3 - 1e-5)
        assert abs(mid_lr - expected_mid) < 1e-7, f"Expected ~{expected_mid}, got {mid_lr}"

        # Run remaining 5 iterations
        for _ in range(5):
            trainer.collect_rollout()
            trainer.update()

        final_lr = trainer.current_lr
        assert abs(final_lr - 1e-5) < 1e-7, f"Expected ~1e-5, got {final_lr}"

    def test_lr_does_not_anneal_when_total_iterations_zero(self) -> None:
        """When total_iterations=0, LR should stay constant."""
        cfg = MAPPOConfig(lr=3e-4, total_iterations=0, rollout_length=4)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=2)
        for _ in range(3):
            trainer.collect_rollout()
            trainer.update()
        assert abs(trainer.current_lr - 3e-4) < 1e-8

    def test_lr_clamps_at_end(self) -> None:
        """LR should not go below lr_end even after total_iterations."""
        cfg = MAPPOConfig(
            lr=1e-3, lr_end=1e-4, total_iterations=5, rollout_length=4,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=3)
        for _ in range(10):  # 2x total_iterations
            trainer.collect_rollout()
            trainer.update()
        assert trainer.current_lr >= 1e-4 - 1e-8


# =====================================================================
# Value clipping
# =====================================================================


class TestValueClipping:
    """Test PPO2-style value clipping."""

    def test_value_clip_produces_valid_result(self) -> None:
        """Training with value_clip_eps > 0 should not crash."""
        cfg = MAPPOConfig(
            value_clip_eps=0.2,
            rollout_length=8,
            num_epochs=2,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=10)
        trainer.collect_rollout()
        result = trainer.update()
        for agent_type, ppo in result.items():
            assert np.isfinite(ppo.value_loss), f"{agent_type} value_loss not finite"
            assert np.isfinite(ppo.policy_loss), f"{agent_type} policy_loss not finite"

    def test_value_clip_disabled(self) -> None:
        """Training with value_clip_eps=0 falls back to plain MSE."""
        cfg = MAPPOConfig(
            value_clip_eps=0.0,
            rollout_length=8,
            num_epochs=2,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=11)
        trainer.collect_rollout()
        result = trainer.update()
        assert all(np.isfinite(r.value_loss) for r in result.values())


# =====================================================================
# Checkpointing
# =====================================================================


class TestCheckpointing:
    """Test TrainingCheckpoint save/load cycle."""

    def test_step_saves_best_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MAPPOConfig(rollout_length=4)
            trainer = MAPPOTrainer(mappo_config=cfg, seed=20)
            ckpt = TrainingCheckpoint(save_dir=tmp, save_every=0)

            # First step should always be best
            is_best = ckpt.step(trainer, iteration=1, metric=1.0)
            assert is_best
            assert ckpt.best_iteration == 1
            assert ckpt.best_metric == 1.0

            # Worse metric should not be best
            is_best = ckpt.step(trainer, iteration=2, metric=0.5)
            assert not is_best
            assert ckpt.best_iteration == 1

            # New best
            is_best = ckpt.step(trainer, iteration=3, metric=2.0)
            assert is_best
            assert ckpt.best_iteration == 3

            # Verify best model files exist
            best_files = list(Path(tmp).glob("best_model_*.pt"))
            assert len(best_files) == 3  # vessel, port, coordinator

    def test_periodic_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MAPPOConfig(rollout_length=4)
            trainer = MAPPOTrainer(mappo_config=cfg, seed=21)
            ckpt = TrainingCheckpoint(save_dir=tmp, save_every=5)

            for i in range(1, 11):
                ckpt.step(trainer, iteration=i, metric=float(i))

            # Should have periodic checkpoints at 5 and 10
            ckpt_5 = list(Path(tmp).glob("ckpt_000005_*.pt"))
            ckpt_10 = list(Path(tmp).glob("ckpt_000010_*.pt"))
            assert len(ckpt_5) == 3
            assert len(ckpt_10) == 3

    def test_save_and_load_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MAPPOConfig(rollout_length=4)
            trainer = MAPPOTrainer(mappo_config=cfg, seed=22)
            ckpt = TrainingCheckpoint(save_dir=tmp, save_every=0)
            ckpt.step(trainer, iteration=1, metric=1.0)
            ckpt.step(trainer, iteration=2, metric=2.0)
            out = ckpt.save_history()
            assert out.exists()
            import json

            history = json.loads(out.read_text())
            assert len(history) == 2
            assert history[0]["iteration"] == 1

    def test_load_best_restores_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MAPPOConfig(rollout_length=4)
            trainer = MAPPOTrainer(mappo_config=cfg, seed=23)

            ckpt = TrainingCheckpoint(save_dir=tmp, save_every=0)
            ckpt.step(trainer, iteration=1, metric=5.0)

            # Modify weights
            trainer.collect_rollout()
            trainer.update()

            # Reload best — should not crash
            ckpt.load_best(trainer)

    def test_cleanup_old_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = MAPPOConfig(rollout_length=4)
            trainer = MAPPOTrainer(mappo_config=cfg, seed=24)
            ckpt = TrainingCheckpoint(save_dir=tmp, save_every=5)

            for i in range(1, 21):
                ckpt.step(trainer, iteration=i, metric=float(i))

            # Before cleanup: 4 periodic checkpoints (5, 10, 15, 20)
            all_periodic = list(Path(tmp).glob("ckpt_*_*.pt"))
            assert len(all_periodic) == 12  # 4 iterations × 3 agent types

            ckpt.cleanup_old_checkpoints(keep_last=2)

            # After cleanup: only 15 and 20 should remain
            remaining = list(Path(tmp).glob("ckpt_*_*.pt"))
            assert len(remaining) == 6  # 2 iterations × 3 agent types


# =====================================================================
# Early stopping
# =====================================================================


class TestEarlyStopping:
    """Test patience-based early stopping."""

    def test_triggers_after_patience(self) -> None:
        es = EarlyStopping(patience=3, higher_is_better=True)
        assert not es.step(1.0, iteration=1)
        assert not es.step(0.9, iteration=2)
        assert not es.step(0.9, iteration=3)
        assert es.step(0.9, iteration=4)  # 3 non-improving steps
        assert es.stopped_iteration == 4

    def test_resets_on_improvement(self) -> None:
        es = EarlyStopping(patience=3, higher_is_better=True)
        assert not es.step(1.0)
        assert not es.step(0.9)
        assert not es.step(0.9)
        # Improvement resets counter
        assert not es.step(1.1)
        assert es.wait_count == 0
        assert not es.step(1.0)
        assert not es.step(1.0)
        assert es.step(1.0)  # 3 non-improving again

    def test_lower_is_better(self) -> None:
        es = EarlyStopping(patience=2, higher_is_better=False)
        assert not es.step(1.0)
        assert not es.step(1.0)
        assert es.step(1.0)  # 2 non-improving

    def test_min_delta(self) -> None:
        es = EarlyStopping(patience=2, min_delta=0.5, higher_is_better=True)
        assert not es.step(1.0)
        # 1.3 is better than 1.0 but not by min_delta=0.5
        assert not es.step(1.3)
        assert es.step(1.3)

    def test_reset(self) -> None:
        es = EarlyStopping(patience=2, higher_is_better=True)
        es.step(5.0)
        es.step(1.0)
        es.step(1.0)  # triggers
        es.reset()
        assert es.wait_count == 0
        assert not es.step(0.1)  # fresh start


# =====================================================================
# MAPPO on larger topologies (scenario integration)
# =====================================================================


class TestMAPPOLargerTopology:
    """Verify MAPPO trainer works on non-default topologies."""

    def test_mappo_10_vessels_4_ports(self) -> None:
        env_cfg = get_default_config(
            num_ports=4, num_vessels=10, rollout_steps=12,
        )
        mappo_cfg = MAPPOConfig(rollout_length=8, num_epochs=1)
        trainer = MAPPOTrainer(
            env_config=env_cfg, mappo_config=mappo_cfg, seed=50,
        )
        rollout = trainer.collect_rollout()
        assert np.isfinite(rollout["mean_reward"])
        update = trainer.update()
        assert all(np.isfinite(r.total_loss) for r in update.values())

    def test_mappo_2_coordinators(self) -> None:
        env_cfg = get_default_config(
            num_ports=4, num_vessels=8, num_coordinators=2, rollout_steps=12,
        )
        mappo_cfg = MAPPOConfig(rollout_length=8, num_epochs=1)
        trainer = MAPPOTrainer(
            env_config=env_cfg, mappo_config=mappo_cfg, seed=51,
        )
        rollout = trainer.collect_rollout()
        assert np.isfinite(rollout["mean_reward"])
        result = trainer.update()
        assert "coordinator" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
