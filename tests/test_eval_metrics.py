"""Tests for operational metrics in MAPPO evaluate() and per-agent rewards in collect_rollout()."""

from __future__ import annotations

import pytest

from hmarl_mvp.config import get_default_config
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer


@pytest.fixture()
def trainer() -> MAPPOTrainer:
    cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
    mappo_cfg = MAPPOConfig(
        rollout_length=8,
        num_epochs=2,
        minibatch_size=4,
        lr=1e-3,
        hidden_dims=[32, 32],
    )
    return MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=42)


# -----------------------------------------------------------------------
# collect_rollout per-agent reward breakdown
# -----------------------------------------------------------------------


class TestCollectRolloutRewards:
    """Test that collect_rollout returns per-agent-type reward breakdown."""

    def test_has_per_agent_keys(self, trainer: MAPPOTrainer) -> None:
        info = trainer.collect_rollout()
        assert "vessel_mean_reward" in info
        assert "port_mean_reward" in info
        assert "coordinator_mean_reward" in info

    def test_per_agent_rewards_are_finite(self, trainer: MAPPOTrainer) -> None:
        info = trainer.collect_rollout()
        for key in ("vessel_mean_reward", "port_mean_reward", "coordinator_mean_reward"):
            val = info[key]
            assert isinstance(val, float)
            assert val == val  # not NaN

    def test_backward_compatible_keys(self, trainer: MAPPOTrainer) -> None:
        """mean_reward and total_reward still present."""
        info = trainer.collect_rollout()
        assert "mean_reward" in info
        assert "total_reward" in info


# -----------------------------------------------------------------------
# evaluate with operational metrics
# -----------------------------------------------------------------------


class TestEvaluateOperationalMetrics:
    """Test that evaluate() returns operational & economic metrics."""

    def test_has_vessel_metrics(self, trainer: MAPPOTrainer) -> None:
        metrics = trainer.evaluate(num_steps=5)
        for key in (
            "avg_speed",
            "avg_fuel_remaining",
            "total_fuel_used",
            "total_emissions_co2",
            "avg_delay_hours",
            "on_time_rate",
        ):
            assert key in metrics, f"Missing vessel metric: {key}"

    def test_has_port_metrics(self, trainer: MAPPOTrainer) -> None:
        metrics = trainer.evaluate(num_steps=5)
        for key in (
            "avg_queue",
            "dock_utilization",
            "total_wait_hours",
            "total_vessels_served",
            "avg_wait_per_vessel",
        ):
            assert key in metrics, f"Missing port metric: {key}"

    def test_has_economic_metrics(self, trainer: MAPPOTrainer) -> None:
        metrics = trainer.evaluate(num_steps=5)
        for key in (
            "fuel_cost_usd",
            "delay_cost_usd",
            "carbon_cost_usd",
            "total_ops_cost_usd",
            "price_per_vessel_usd",
            "cost_reliability",
        ):
            assert key in metrics, f"Missing economic metric: {key}"

    def test_still_has_reward_keys(self, trainer: MAPPOTrainer) -> None:
        """Original reward-based keys remain present."""
        metrics = trainer.evaluate(num_steps=5)
        for key in (
            "mean_vessel_reward",
            "mean_port_reward",
            "mean_coordinator_reward",
            "total_reward",
        ):
            assert key in metrics

    def test_metric_values_are_finite(self, trainer: MAPPOTrainer) -> None:
        metrics = trainer.evaluate(num_steps=5)
        for key, val in metrics.items():
            assert isinstance(val, (int, float)), f"{key}: expected numeric, got {type(val)}"
            assert val == val, f"{key} is NaN"

    def test_on_time_rate_bounded(self, trainer: MAPPOTrainer) -> None:
        metrics = trainer.evaluate(num_steps=5)
        assert 0.0 <= metrics["on_time_rate"] <= 1.0

    def test_cost_reliability_bounded(self, trainer: MAPPOTrainer) -> None:
        metrics = trainer.evaluate(num_steps=5)
        assert metrics["cost_reliability"] <= 1.0


# -----------------------------------------------------------------------
# evaluate_episodes inherits operational metrics
# -----------------------------------------------------------------------


class TestEvaluateEpisodesMetrics:
    """Test that evaluate_episodes() aggregates operational metrics."""

    def test_episode_metrics_include_operational(self, trainer: MAPPOTrainer) -> None:
        result = trainer.evaluate_episodes(num_episodes=2, num_steps=5)
        assert "episodes" in result
        ep0 = result["episodes"][0]
        assert "total_fuel_used" in ep0
        assert "avg_queue" in ep0
        assert "total_ops_cost_usd" in ep0

    def test_mean_std_include_operational(self, trainer: MAPPOTrainer) -> None:
        result = trainer.evaluate_episodes(num_episodes=2, num_steps=5)
        assert "total_fuel_used" in result["mean"]
        assert "total_fuel_used" in result["std"]

    def test_multi_episode_consistency(self, trainer: MAPPOTrainer) -> None:
        result = trainer.evaluate_episodes(num_episodes=3, num_steps=5)
        assert len(result["episodes"]) == 3
        # Mean should be between min and max
        for key in ("total_fuel_used", "avg_queue", "total_ops_cost_usd"):
            if key in result["mean"]:
                assert result["min"][key] <= result["mean"][key] <= result["max"][key]


# -----------------------------------------------------------------------
# train() loop propagates per-agent rewards
# -----------------------------------------------------------------------


class TestTrainPerAgentRewards:
    """Per-agent rewards appear in the training history log entries."""

    def test_history_has_per_agent_rewards(self, trainer: MAPPOTrainer) -> None:
        history = trainer.train(num_iterations=2)
        assert len(history) == 2
        entry = history[0]
        assert "vessel_mean_reward" in entry or "mean_reward" in entry
