"""Tests for the MAPPO training module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hmarl_mvp.config import get_default_config
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    PPOUpdateResult,
    _nn_to_coordinator_action,
    _nn_to_port_action,
    _nn_to_vessel_action,
    global_state_dim_from_config,
)


@pytest.fixture()
def small_config() -> dict[str, object]:
    return get_default_config(num_ports=3, num_vessels=4, rollout_steps=30)


@pytest.fixture()
def small_mappo_cfg() -> MAPPOConfig:
    return MAPPOConfig(
        rollout_length=8,
        num_epochs=2,
        minibatch_size=4,
        lr=1e-3,
        hidden_dims=[32, 32],
    )


class TestGlobalStateDim:
    """Test global state dimension computation."""

    def test_default_config(self) -> None:
        cfg = get_default_config()
        dim = global_state_dim_from_config(cfg)
        env = MaritimeEnv(config=cfg)
        env.reset()
        actual = env.get_global_state().shape[0]
        assert dim == actual, f"Expected {dim}, got {actual}"

    def test_small_config(self, small_config: dict[str, object]) -> None:
        dim = global_state_dim_from_config(small_config)
        env = MaritimeEnv(config=small_config)
        env.reset()
        actual = env.get_global_state().shape[0]
        assert dim == actual


class TestActionTranslation:
    """Test NN output → env action dict conversion."""

    def test_vessel_action_clamps(self) -> None:
        cfg = get_default_config()
        # High value should be clamped to speed_max
        action = _nn_to_vessel_action(torch.tensor([100.0]), cfg)
        assert action["target_speed"] == cfg["speed_max"]
        # Low value should be clamped to speed_min
        action = _nn_to_vessel_action(torch.tensor([-100.0]), cfg)
        assert action["target_speed"] == cfg["speed_min"]
        assert action["request_arrival_slot"] is True

    def test_port_action_discrete(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4)
        env = MaritimeEnv(config=cfg)
        env.reset()
        action = _nn_to_port_action(torch.tensor(2), 0, env)
        assert action["service_rate"] == 2
        assert "accept_requests" in action

    def test_coordinator_action_discrete(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4)
        env = MaritimeEnv(config=cfg)
        env.reset()
        assignments = {0: [0, 1, 2, 3]}
        action = _nn_to_coordinator_action(torch.tensor(1), 0, env, assignments)
        assert action["dest_port"] == 1
        assert "per_vessel_dest" in action
        assert "emission_budget" in action


class TestMAPPOTrainer:
    """End-to-end tests for the MAPPO trainer."""

    def test_init(self, small_config: dict[str, object], small_mappo_cfg: MAPPOConfig) -> None:
        trainer = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        assert "vessel" in trainer.actor_critics
        assert "port" in trainer.actor_critics
        assert "coordinator" in trainer.actor_critics

    def test_collect_rollout(
        self, small_config: dict[str, object], small_mappo_cfg: MAPPOConfig
    ) -> None:
        trainer = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        info = trainer.collect_rollout()
        assert "mean_reward" in info
        assert "total_reward" in info
        assert isinstance(info["mean_reward"], float)

    def test_update_runs(
        self, small_config: dict[str, object], small_mappo_cfg: MAPPOConfig
    ) -> None:
        trainer = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        trainer.collect_rollout()
        results = trainer.update()
        assert "vessel" in results
        assert "port" in results
        assert "coordinator" in results
        for agent_type, result in results.items():
            assert isinstance(result, PPOUpdateResult)
            # After one rollout + update, losses should be finite
            assert np.isfinite(result.policy_loss), f"{agent_type} policy_loss not finite"
            assert np.isfinite(result.value_loss), f"{agent_type} value_loss not finite"

    def test_evaluate(
        self, small_config: dict[str, object], small_mappo_cfg: MAPPOConfig
    ) -> None:
        trainer = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        metrics = trainer.evaluate(num_steps=5)
        assert "mean_vessel_reward" in metrics
        assert "mean_port_reward" in metrics
        assert "mean_coordinator_reward" in metrics
        assert "total_reward" in metrics

    def test_multiple_iterations(
        self, small_config: dict[str, object], small_mappo_cfg: MAPPOConfig
    ) -> None:
        """Run 3 full collect→update iterations without error."""
        trainer = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        for _ in range(3):
            trainer.collect_rollout()
            trainer.update()
        assert len(trainer.reward_history) == 3

    def test_reward_history(
        self, small_config: dict[str, object], small_mappo_cfg: MAPPOConfig
    ) -> None:
        trainer = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        trainer.collect_rollout()
        trainer.update()
        assert len(trainer.reward_history) == 1

    def test_save_load_models(
        self, small_config: dict[str, object], small_mappo_cfg: MAPPOConfig, tmp_path: object
    ) -> None:
        trainer = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        trainer.collect_rollout()
        trainer.update()

        prefix = str(tmp_path) + "/model"  # type: ignore[operator]
        trainer.save_models(prefix)
        eval_before = trainer.evaluate(num_steps=5)

        # Load into a new trainer
        trainer2 = MAPPOTrainer(
            env_config=small_config,
            mappo_config=small_mappo_cfg,
        )
        trainer2.load_models(prefix)
        eval_after = trainer2.evaluate(num_steps=5)

        # Same seed + same weights should produce same results
        assert eval_before["total_reward"] == pytest.approx(
            eval_after["total_reward"], abs=0.01
        )


class TestPPOLossDirection:
    """Sanity check that PPO loss values are in reasonable ranges."""

    def test_value_loss_positive(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=30)
        mappo_cfg = MAPPOConfig(
            rollout_length=16,
            num_epochs=2,
            minibatch_size=8,
            hidden_dims=[32, 32],
        )
        trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg)
        trainer.collect_rollout()
        results = trainer.update()
        # Value loss should be positive (MSE)
        for agent_type, result in results.items():
            assert result.value_loss >= 0, f"{agent_type} value_loss < 0"

    def test_clip_fraction_bounded(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=30)
        mappo_cfg = MAPPOConfig(
            rollout_length=16,
            num_epochs=2,
            minibatch_size=8,
            hidden_dims=[32, 32],
        )
        trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg)
        trainer.collect_rollout()
        results = trainer.update()
        for agent_type, result in results.items():
            assert 0.0 <= result.clip_fraction <= 1.0, (
                f"{agent_type} clip_frac={result.clip_fraction}"
            )
