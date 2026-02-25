"""Tests for per-step global state storage, reward normalisation, and comparison runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hmarl_mvp.buffer import MultiAgentRolloutBuffer, RolloutBuffer
from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import run_mappo_comparison
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer, RunningMeanStd
from hmarl_mvp.plotting import plot_mappo_comparison, plot_training_curves

# ---------------------------------------------------------------------------
# Per-step global state storage
# ---------------------------------------------------------------------------


class TestGlobalStateBuffer:
    """Verify that RolloutBuffer stores and retrieves per-step global states."""

    def test_global_state_stored(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=3, act_dim=1, global_state_dim=5)
        gs = np.ones(5, dtype=np.float32) * 2.0
        buf.add(obs=np.zeros(3), action=0.0, reward=1.0, done=False, global_state=gs)
        data = buf.get_tensors()
        assert "global_states" in data
        assert data["global_states"].shape == (1, 5)
        np.testing.assert_allclose(data["global_states"][0].numpy(), 2.0)

    def test_no_global_state_key_when_dim_zero(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=3, act_dim=1)
        buf.add(obs=np.zeros(3), action=0.0, reward=1.0, done=False)
        data = buf.get_tensors()
        assert "global_states" not in data

    def test_multi_agent_global_state(self) -> None:
        mbuf = MultiAgentRolloutBuffer(
            num_agents=2, capacity=4, obs_dim=3, act_dim=1, global_state_dim=5
        )
        gs = np.arange(5, dtype=np.float32)
        mbuf[0].add(obs=np.zeros(3), action=0.0, reward=1.0, done=False, global_state=gs)
        mbuf[1].add(obs=np.zeros(3), action=0.0, reward=1.0, done=False, global_state=gs)
        for i in range(2):
            data = mbuf[i].get_tensors()
            assert "global_states" in data
            np.testing.assert_allclose(data["global_states"][0].numpy(), gs)


class TestGlobalStateInMAPPO:
    """Verify that MAPPO trainer stores per-step global states during rollout."""

    def test_buffers_have_global_dim(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        mappo_cfg = MAPPOConfig(rollout_length=4, hidden_dims=[16, 16])
        trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg)
        assert trainer.vessel_buf.global_state_dim == trainer.global_dim
        assert trainer.port_buf.global_state_dim == trainer.global_dim
        assert trainer.coordinator_buf.global_state_dim == trainer.global_dim

    def test_global_states_populated_after_rollout(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        mappo_cfg = MAPPOConfig(rollout_length=8, hidden_dims=[16, 16])
        trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg)
        trainer.collect_rollout()
        # Check vessel buffer agent 0
        data = trainer.vessel_buf[0].get_tensors()
        assert "global_states" in data
        assert data["global_states"].shape[0] == 8
        assert data["global_states"].shape[1] == trainer.global_dim
        # Global states should vary across steps (not all identical)
        gs_np = data["global_states"].numpy()
        # At minimum, the first and last should differ (env evolves)
        assert not np.allclose(gs_np[0], gs_np[-1])


# ---------------------------------------------------------------------------
# Reward normalisation
# ---------------------------------------------------------------------------


class TestRunningMeanStd:
    """Test the Welford online statistics tracker."""

    def test_single_value(self) -> None:
        rms = RunningMeanStd()
        rms.update(5.0)
        assert rms.mean == pytest.approx(5.0, abs=0.1)

    def test_batch_mean(self) -> None:
        rms = RunningMeanStd()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        rms.update_batch(values)
        assert rms.mean == pytest.approx(3.0, abs=0.1)

    def test_normalize_centers(self) -> None:
        rms = RunningMeanStd()
        rms.update_batch([10.0] * 100)
        # Normalizing the mean should give ~0
        assert abs(rms.normalize(10.0)) < 0.5

    def test_std_positive(self) -> None:
        rms = RunningMeanStd()
        rms.update_batch([1.0, 10.0, 5.0])
        assert rms.std > 0


class TestRewardNormInTrainer:
    """Test that reward normalisation integrates into training."""

    def test_normalize_rewards_flag(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        mappo_cfg = MAPPOConfig(
            rollout_length=8, hidden_dims=[16, 16], normalize_rewards=True
        )
        trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg)
        trainer.collect_rollout()
        # After collection, reward normalizers should have been updated
        for name, rms in trainer._reward_normalizers.items():
            assert rms.count > 1.0, f"{name} normalizer not updated"

    def test_no_normalize_flag(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        mappo_cfg = MAPPOConfig(
            rollout_length=8, hidden_dims=[16, 16], normalize_rewards=False
        )
        trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg)
        trainer.collect_rollout()
        # Normalizers still get updated (tracking stats) but rewards are raw
        data = trainer.vessel_buf[0].get_tensors()
        # Raw rewards should be negative (from reward functions)
        assert data["rewards"].min().item() <= 0.0


# ---------------------------------------------------------------------------
# MAPPO vs heuristic comparison
# ---------------------------------------------------------------------------


class TestMAPPOComparison:
    """Test the comparison runner end-to-end."""

    def test_run_mappo_comparison(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        results = run_mappo_comparison(
            train_iterations=3,
            rollout_length=8,
            eval_steps=10,
            baselines=["independent", "forecast"],
            seed=42,
            config=cfg,
            mappo_kwargs={"hidden_dims": [16, 16]},
        )
        assert "mappo" in results
        assert "independent" in results
        assert "forecast" in results
        assert "_train_log" in results
        assert isinstance(results["mappo"], pd.DataFrame)
        assert len(results["mappo"]) == 10
        assert "avg_queue" in results["mappo"].columns
        assert (results["mappo"]["policy"] == "mappo").all()

    def test_train_log_has_iterations(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        results = run_mappo_comparison(
            train_iterations=5,
            rollout_length=8,
            eval_steps=5,
            baselines=["independent"],
            config=cfg,
            mappo_kwargs={"hidden_dims": [16, 16]},
        )
        train_log = results["_train_log"]
        assert len(train_log) == 5
        assert "mean_reward" in train_log.columns


# ---------------------------------------------------------------------------
# Plotting smoke tests
# ---------------------------------------------------------------------------


class TestPlotting:
    """Smoke test new plotting functions (non-interactive)."""

    def test_plot_training_curves_no_crash(self, tmp_path: object) -> None:
        import matplotlib
        matplotlib.use("Agg")
        df = pd.DataFrame({
            "iteration": [1, 2, 3, 4, 5],
            "mean_reward": [-10.0, -9.0, -8.0, -7.5, -7.0],
            "vessel_value_loss": [100.0, 90.0, 80.0, 70.0, 60.0],
        })
        out = str(tmp_path) + "/train.png"  # type: ignore[operator]
        plot_training_curves(df, out_path=out)
        from pathlib import Path
        assert Path(out).exists()

    def test_plot_mappo_comparison_no_crash(self, tmp_path: object) -> None:
        import matplotlib
        matplotlib.use("Agg")
        results = {
            "mappo": pd.DataFrame({
                "t": [0, 1, 2],
                "avg_queue": [3.0, 2.5, 2.0],
                "dock_utilization": [0.5, 0.6, 0.7],
                "total_emissions_co2": [10, 20, 30],
                "avg_vessel_reward": [-5, -4, -3],
                "total_ops_cost_usd": [1000, 2000, 3000],
                "coordinator_reward": [-8, -7, -6],
            }),
            "forecast": pd.DataFrame({
                "t": [0, 1, 2],
                "avg_queue": [4.0, 3.5, 3.0],
                "dock_utilization": [0.4, 0.5, 0.6],
                "total_emissions_co2": [15, 25, 35],
                "avg_vessel_reward": [-6, -5, -4],
                "total_ops_cost_usd": [1200, 2200, 3200],
                "coordinator_reward": [-9, -8, -7],
            }),
        }
        out = str(tmp_path) + "/cmp.png"  # type: ignore[operator]
        plot_mappo_comparison(results, out_path=out)
        from pathlib import Path
        assert Path(out).exists()
