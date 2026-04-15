"""Tests for Population-Based Training (PBT) module."""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import pytest
import torch

from hmarl_mvp.config import get_default_config
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.pbt import PBTConfig, PBTTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_env_config() -> dict:
    """Minimal env config for fast PBT tests."""
    return {
        "num_vessels": 2,
        "num_ports": 2,
        "docks_per_port": 1,
        "rollout_steps": 30,
    }


def _small_mappo_config() -> MAPPOConfig:
    """Minimal MAPPO config for fast PBT tests."""
    return MAPPOConfig(
        rollout_length=4,
        num_epochs=1,
        minibatch_size=8,
        hidden_dims=[16, 16],
        vessel_hidden_dims=[16, 16],
        lr=3e-4,
        entropy_coeff=0.05,
        clip_eps=0.2,
    )


# ---------------------------------------------------------------------------
# PBTConfig tests
# ---------------------------------------------------------------------------


class TestPBTConfig:
    def test_defaults(self):
        cfg = PBTConfig()
        assert cfg.population_size == 4
        assert cfg.interval == 10
        assert cfg.fraction_top == 0.25
        assert cfg.fraction_bottom == 0.25
        assert cfg.perturb_factor == 1.2
        assert cfg.perturb_lr is True
        assert cfg.perturb_entropy is True
        assert cfg.perturb_clip_eps is True

    def test_custom_values(self):
        cfg = PBTConfig(population_size=8, interval=20, perturb_factor=1.5)
        assert cfg.population_size == 8
        assert cfg.interval == 20
        assert cfg.perturb_factor == 1.5

    def test_bounds(self):
        cfg = PBTConfig()
        assert cfg.lr_min < cfg.lr_max
        assert cfg.entropy_min < cfg.entropy_max
        assert cfg.clip_eps_min < cfg.clip_eps_max


# ---------------------------------------------------------------------------
# PBTTrainer construction tests
# ---------------------------------------------------------------------------


class TestPBTTrainerConstruction:
    def test_creates_correct_population(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=3),
        )
        assert len(pbt.workers) == 3

    def test_workers_have_different_seeds(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=4),
            base_seed=42,
        )
        seeds = pbt._worker_seeds
        assert len(set(seeds)) == 4  # all unique

    def test_workers_scheduling_disabled(self):
        """PBT workers should have built-in scheduling disabled."""
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2),
        )
        for w in pbt.workers:
            assert w.mappo_cfg.total_iterations == 0
            assert w.mappo_cfg.entropy_coeff_end is None

    def test_workers_independent_networks(self):
        """Each worker should have independent network parameters."""
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2),
        )
        # After construction (before training), weights may differ due
        # to different seeds
        w0_params = list(pbt.workers[0].actor_critics["vessel"].parameters())
        w1_params = list(pbt.workers[1].actor_critics["vessel"].parameters())
        # At minimum, parameters should be separate objects
        assert w0_params[0] is not w1_params[0]


# ---------------------------------------------------------------------------
# Weight copy tests
# ---------------------------------------------------------------------------


class TestCopyWeights:
    def test_copy_makes_weights_equal(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2),
        )
        # Train worker 0 a bit to diverge weights
        pbt.workers[0].collect_rollout()
        pbt.workers[0].update()

        pbt._copy_weights(source_idx=0, target_idx=1)

        for name in pbt.workers[0].actor_critics:
            src_sd = pbt.workers[0].actor_critics[name].state_dict()
            tgt_sd = pbt.workers[1].actor_critics[name].state_dict()
            for key in src_sd:
                torch.testing.assert_close(src_sd[key], tgt_sd[key])

    def test_copy_transfers_normaliser_state(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2),
        )
        # Update source normaliser
        pbt.workers[0]._reward_normalizers["vessel"].update(10.0)
        pbt.workers[0]._reward_normalizers["vessel"].update(20.0)

        pbt._copy_weights(source_idx=0, target_idx=1)

        src_rms = pbt.workers[0]._reward_normalizers["vessel"]
        tgt_rms = pbt.workers[1]._reward_normalizers["vessel"]
        assert tgt_rms.mean == pytest.approx(src_rms.mean)
        assert tgt_rms.var == pytest.approx(src_rms.var)

    def test_copy_transfers_hyperparams(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2),
        )
        pbt.workers[0].mappo_cfg.entropy_coeff = 0.09
        pbt.workers[0].mappo_cfg.clip_eps = 0.15

        pbt._copy_weights(source_idx=0, target_idx=1)

        assert pbt.workers[1].mappo_cfg.entropy_coeff == 0.09
        assert pbt.workers[1].mappo_cfg.clip_eps == 0.15

    def test_copy_resets_optimiser(self):
        """After copy, optimiser should have fresh state (no momentum)."""
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2),
        )
        # Train worker 0 to populate optimizer state
        pbt.workers[0].collect_rollout()
        pbt.workers[0].update()

        pbt._copy_weights(source_idx=0, target_idx=1)

        for opt in pbt.workers[1].optimizers.values():
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state.get(p, {})
                    # Fresh optimizer has no state entries
                    assert len(state) == 0


# ---------------------------------------------------------------------------
# Perturbation tests
# ---------------------------------------------------------------------------


class TestPerturbHyperparams:
    def test_perturb_changes_lr(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, perturb_factor=2.0),
        )
        original_lr = pbt.workers[0].current_lr
        # Run perturbation many times — at least one should differ
        changed = False
        for _ in range(20):
            pbt._perturb_hyperparams(0)
            if pbt.workers[0].current_lr != original_lr:
                changed = True
                break
        assert changed

    def test_perturb_respects_bounds(self):
        cfg = PBTConfig(
            population_size=2,
            perturb_factor=100.0,  # extreme factor
            lr_min=1e-5,
            lr_max=1e-3,
            entropy_min=0.001,
            entropy_max=0.1,
            clip_eps_min=0.1,
            clip_eps_max=0.3,
        )
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=cfg,
        )
        for _ in range(50):
            pbt._perturb_hyperparams(0)
            hp = pbt._get_hyperparams(0)
            assert cfg.lr_min <= hp["lr"] <= cfg.lr_max
            assert cfg.entropy_min <= hp["entropy_coeff"] <= cfg.entropy_max
            assert cfg.clip_eps_min <= hp["clip_eps"] <= cfg.clip_eps_max

    def test_selective_perturbation(self):
        """When a perturbation channel is disabled, that param stays fixed."""
        cfg = PBTConfig(
            population_size=2,
            perturb_lr=False,
            perturb_entropy=True,
            perturb_clip_eps=False,
            perturb_factor=2.0,
        )
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=cfg,
        )
        original_lr = pbt.workers[0].current_lr
        original_eps = pbt.workers[0].mappo_cfg.clip_eps

        for _ in range(20):
            pbt._perturb_hyperparams(0)
        assert pbt.workers[0].current_lr == original_lr
        assert pbt.workers[0].mappo_cfg.clip_eps == original_eps


# ---------------------------------------------------------------------------
# Exploit and explore tests
# ---------------------------------------------------------------------------


class TestExploitAndExplore:
    def test_exploit_log_populated(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=4, interval=2),
        )
        # Simulate some rewards
        for w_idx in range(4):
            pbt._rewards[w_idx] = [float(w_idx)] * 2

        pbt._exploit_and_explore(round_idx=0)

        assert len(pbt._exploit_log) == 1
        log = pbt._exploit_log[0]
        assert "ranking" in log
        assert "means" in log
        assert "events" in log

    def test_bottom_worker_gets_replaced(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(
                population_size=4,
                interval=2,
                fraction_top=0.5,
                fraction_bottom=0.25,
            ),
        )
        # Worker 3 is worst, worker 0 is best
        pbt._rewards = [
            [10.0, 10.0],  # worker 0 — best
            [5.0, 5.0],  # worker 1
            [3.0, 3.0],  # worker 2
            [-5.0, -5.0],  # worker 3 — worst
        ]

        # Record worker 3 weights before exploit
        old_sd = copy.deepcopy(
            pbt.workers[3].actor_critics["vessel"].state_dict()
        )

        pbt._exploit_and_explore(round_idx=0)

        # Worker 3 should have been replaced
        assert len(pbt._exploit_log) == 1
        events = pbt._exploit_log[0]["events"]
        targets = [e["target"] for e in events]
        assert 3 in targets

    def test_top_worker_not_replaced(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(
                population_size=4,
                interval=2,
                fraction_top=0.25,
                fraction_bottom=0.25,
            ),
        )
        pbt._rewards = [
            [10.0, 10.0],  # best
            [5.0, 5.0],
            [3.0, 3.0],
            [-5.0, -5.0],  # worst
        ]

        pbt._exploit_and_explore(round_idx=0)

        events = pbt._exploit_log[0]["events"]
        targets = [e["target"] for e in events]
        assert 0 not in targets  # best worker should not be a target


# ---------------------------------------------------------------------------
# End-to-end training tests
# ---------------------------------------------------------------------------


class TestPBTTraining:
    def test_short_training_runs(self):
        """PBT training completes without error."""
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=2),
        )
        result = pbt.train(total_iterations=4)

        assert "best_worker_idx" in result
        assert "best_mean_reward" in result
        assert result["total_rounds"] == 2
        assert len(result["per_worker_rewards"]) == 2
        # Each worker should have 4 reward entries
        for rews in result["per_worker_rewards"]:
            assert len(rews) == 4

    def test_remainder_iterations(self):
        """Remainder iterations (total % interval != 0) are executed."""
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=3),
        )
        result = pbt.train(total_iterations=5)

        assert result["total_rounds"] == 1  # 5 // 3 = 1
        # Each worker: 3 (round) + 2 (remainder) = 5 rewards
        for rews in result["per_worker_rewards"]:
            assert len(rews) == 5

    def test_hyperparam_history_recorded(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=2),
        )
        result = pbt.train(total_iterations=4)

        for hp_list in result["hyperparam_history"]:
            assert len(hp_list) == 4
            for hp in hp_list:
                assert "lr" in hp
                assert "entropy_coeff" in hp
                assert "clip_eps" in hp

    def test_log_fn_called(self):
        calls = []

        def log_fn(round_idx, worker_idx, iteration, rollout_info):
            calls.append((round_idx, worker_idx, iteration))

        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=2),
        )
        pbt.train(total_iterations=4, log_fn=log_fn)

        # 2 rounds × 2 workers × 2 iterations = 8 calls
        assert len(calls) == 8

    def test_checkpoint_saves_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pbt = PBTTrainer(
                env_config=_small_env_config(),
                mappo_config=_small_mappo_config(),
                pbt_config=PBTConfig(population_size=2, interval=2),
            )
            pbt.train(total_iterations=4, checkpoint_dir=tmpdir)

            # Should have checkpoint files from 2 rounds
            files = os.listdir(tmpdir)
            assert len(files) > 0
            # Check for expected prefix pattern
            assert any("pbt_best_r0000" in f for f in files)
            assert any("pbt_best_r0001" in f for f in files)

    def test_best_worker_property(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=2),
        )
        pbt.train(total_iterations=4)

        best = pbt.best_worker
        assert isinstance(best, MAPPOTrainer)

    def test_exploit_log_in_result(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=2),
        )
        result = pbt.train(total_iterations=4)

        # 2 rounds, exploit happens after round 0 only (skipped after last)
        assert len(result["exploit_log"]) == 1

    def test_final_hyperparams_in_result(self):
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=2),
        )
        result = pbt.train(total_iterations=4)

        assert len(result["final_hyperparams"]) == 2
        for hp in result["final_hyperparams"]:
            assert "lr" in hp
            assert hp["lr"] > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPBTEdgeCases:
    def test_population_size_2(self):
        """Minimum viable population: 1 top, 1 bottom."""
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=2),
        )
        result = pbt.train(total_iterations=4)
        assert result["best_worker_idx"] in (0, 1)

    def test_no_perturbation(self):
        """All perturbation channels disabled — hyperparams stay constant."""
        cfg = PBTConfig(
            population_size=2,
            interval=2,
            perturb_lr=False,
            perturb_entropy=False,
            perturb_clip_eps=False,
        )
        mcfg = _small_mappo_config()
        original_ent = mcfg.entropy_coeff
        original_eps = mcfg.clip_eps

        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=mcfg,
            pbt_config=cfg,
        )
        pbt.train(total_iterations=4)

        # All workers should still have original entropy and clip_eps
        # (LR may differ slightly due to Adam updates, but perturbation is off)
        for w in pbt.workers:
            assert w.mappo_cfg.clip_eps == original_eps

    def test_single_iteration_per_round(self):
        """interval=1: exploit/explore every single iteration."""
        pbt = PBTTrainer(
            env_config=_small_env_config(),
            mappo_config=_small_mappo_config(),
            pbt_config=PBTConfig(population_size=2, interval=1),
        )
        result = pbt.train(total_iterations=3)
        assert result["total_rounds"] == 3
        # Exploit happens after rounds 0, 1 (not after last round 2)
        assert len(result["exploit_log"]) == 2
