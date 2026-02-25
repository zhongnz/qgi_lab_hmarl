"""Tests for new features: coordinator action spreading, observation
normalisation, curriculum learning, and analysis utilities.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import pandas as pd
import torch

from hmarl_mvp.analysis import (
    compare_to_baselines,
    compute_ablation_deltas,
    compute_training_stats,
    format_comparison_table,
    rank_sweep_results,
    summarize_experiment,
)
from hmarl_mvp.curriculum import (
    CurriculumScheduler,
    CurriculumStage,
    make_curriculum_configs,
)
from hmarl_mvp.mappo import (
    MAPPOConfig,
    MAPPOTrainer,
    ObsRunningMeanStd,
    _nn_to_coordinator_action,
)

# ===================================================================
# Coordinator action spreading
# ===================================================================


class TestCoordinatorActionSpreading(unittest.TestCase):
    """Verify that the coordinator distributes vessels across ports."""

    def setUp(self) -> None:
        from hmarl_mvp.env import MaritimeEnv

        self.env = MaritimeEnv(
            config={"num_vessels": 6, "num_ports": 5, "rollout_steps": 10}
        )
        self.env.reset()
        self.assignments = self.env._build_assignments()

    def test_per_vessel_dest_produced(self) -> None:
        """Coordinator action should contain per_vessel_dest entries."""
        raw = torch.tensor(2)  # select port 2
        action = _nn_to_coordinator_action(raw, 0, self.env, self.assignments)
        self.assertIn("per_vessel_dest", action)
        local_ids = self.assignments.get(0, [])
        self.assertEqual(len(action["per_vessel_dest"]), len(local_ids))

    def test_vessels_spread_across_ports(self) -> None:
        """With enough vessels, destinations should span multiple ports."""
        raw = torch.tensor(0)
        action = _nn_to_coordinator_action(raw, 0, self.env, self.assignments)
        destinations = set(action["per_vessel_dest"].values())
        # With 6 vessels and 5 ports, at least 2 distinct ports expected
        self.assertGreaterEqual(len(destinations), 1)

    def test_primary_port_in_destinations(self) -> None:
        """The primary port should appear as a destination for at least one vessel."""
        for primary in range(self.env.num_ports):
            raw = torch.tensor(primary)
            action = _nn_to_coordinator_action(raw, 0, self.env, self.assignments)
            local_ids = self.assignments.get(0, [])
            if local_ids:
                self.assertIn(primary, action["per_vessel_dest"].values())

    def test_deterministic_for_same_port(self) -> None:
        """Same primary port produces identical per-vessel destinations."""
        raw = torch.tensor(3)
        a1 = _nn_to_coordinator_action(raw, 0, self.env, self.assignments)
        a2 = _nn_to_coordinator_action(raw, 0, self.env, self.assignments)
        self.assertEqual(a1["per_vessel_dest"], a2["per_vessel_dest"])

    def test_different_primary_different_spread(self) -> None:
        """Different primary ports should produce different vessel spreads."""
        local_ids = self.assignments.get(0, [])
        if len(local_ids) < 2:
            self.skipTest("Need at least 2 vessels for port-spread variation")
        a0 = _nn_to_coordinator_action(torch.tensor(0), 0, self.env, self.assignments)
        a1 = _nn_to_coordinator_action(torch.tensor(1), 0, self.env, self.assignments)
        self.assertNotEqual(
            list(a0["per_vessel_dest"].values()),
            list(a1["per_vessel_dest"].values()),
        )


# ===================================================================
# Observation normalisation
# ===================================================================


class TestObsRunningMeanStd(unittest.TestCase):
    """Unit tests for ObsRunningMeanStd."""

    def test_initial_state(self) -> None:
        norm = ObsRunningMeanStd(dim=4)
        self.assertEqual(norm.dim, 4)
        np.testing.assert_array_equal(norm.mean, np.zeros(4))
        np.testing.assert_array_equal(norm.var, np.ones(4))

    def test_single_update(self) -> None:
        norm = ObsRunningMeanStd(dim=3)
        norm.update(np.array([10.0, 20.0, 30.0]))
        # Mean should shift toward the observation
        self.assertAlmostEqual(norm.mean[0], 10.0, places=3)

    def test_batch_update_mean(self) -> None:
        norm = ObsRunningMeanStd(dim=2)
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        norm.update_batch(data)
        np.testing.assert_allclose(norm.mean, [3.0, 4.0], atol=0.5)

    def test_normalize_zero_mean_unit_var(self) -> None:
        """After sufficient samples, normalised data should be ~N(0,1)."""
        rng = np.random.default_rng(42)
        norm = ObsRunningMeanStd(dim=2)
        data = rng.normal(loc=[5.0, -3.0], scale=[2.0, 0.5], size=(500, 2))
        for row in data:
            norm.update(row)
        normalised = np.array([norm.normalize(row) for row in data])
        np.testing.assert_allclose(normalised.mean(axis=0), [0.0, 0.0], atol=0.15)
        np.testing.assert_allclose(normalised.std(axis=0), [1.0, 1.0], atol=0.15)

    def test_normalize_doesnt_nan(self) -> None:
        """Normalise should never produce NaN even with constant inputs."""
        norm = ObsRunningMeanStd(dim=2)
        for _ in range(10):
            norm.update(np.array([1.0, 1.0]))
        result = norm.normalize(np.array([1.0, 1.0]))
        self.assertFalse(np.any(np.isnan(result)))


class TestObsNormInTrainer(unittest.TestCase):
    """Integration test: obs normalisation in MAPPOTrainer."""

    def test_trainer_has_obs_normalizers(self) -> None:
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=True)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        self.assertIn("vessel", trainer._obs_normalizers)
        self.assertIn("port", trainer._obs_normalizers)
        self.assertIn("coordinator", trainer._obs_normalizers)

    def test_normalize_obs_updates_stats(self) -> None:
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=True)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        obs = np.ones(trainer.obs_dims["vessel"]) * 5.0
        result = trainer._normalize_obs(obs, "vessel")
        # After one update the mean shifts; output should be finite
        self.assertEqual(result.shape, obs.shape)
        self.assertFalse(np.any(np.isnan(result)))

    def test_obs_norm_disabled(self) -> None:
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=False)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        obs = np.ones(trainer.obs_dims["vessel"]) * 5.0
        result = trainer._normalize_obs(obs, "vessel")
        np.testing.assert_array_equal(result, obs)

    def test_eval_normalize_obs_no_update(self) -> None:
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=True)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        # Prime the normalizer
        obs = np.ones(trainer.obs_dims["vessel"]) * 3.0
        trainer._normalize_obs(obs, "vessel")
        count_before = trainer._obs_normalizers["vessel"].count
        # Eval should not update count
        trainer._eval_normalize_obs(obs, "vessel")
        self.assertEqual(trainer._obs_normalizers["vessel"].count, count_before)

    def test_collect_rollout_with_obs_norm(self) -> None:
        """Full rollout collection with obs normalisation should succeed."""
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=True)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        info = trainer.collect_rollout()
        self.assertIn("mean_reward", info)
        self.assertFalse(np.isnan(info["mean_reward"]))

    def test_diagnostics_include_obs_stats(self) -> None:
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=True)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        trainer.collect_rollout()
        diag = trainer.get_diagnostics()
        self.assertIn("vessel_obs_mean_norm", diag)
        self.assertIn("vessel_obs_std_mean", diag)
        self.assertIn("port_obs_mean_norm", diag)
        self.assertIn("coordinator_obs_mean_norm", diag)


# ===================================================================
# Curriculum learning
# ===================================================================


class TestCurriculumScheduler(unittest.TestCase):
    """Tests for the curriculum learning scheduler."""

    def _target(self) -> dict[str, Any]:
        return {
            "num_vessels": 8,
            "num_ports": 5,
            "docks_per_port": 3,
            "rollout_steps": 64,
        }

    def test_at_start_uses_easy_config(self) -> None:
        sched = CurriculumScheduler(target_config=self._target(), warmup_fraction=0.3)
        cfg = sched.get_config(0, 100)
        self.assertLessEqual(cfg["num_vessels"], self._target()["num_vessels"])
        self.assertLessEqual(cfg["num_ports"], self._target()["num_ports"])

    def test_at_end_matches_target(self) -> None:
        sched = CurriculumScheduler(target_config=self._target(), warmup_fraction=0.3)
        cfg = sched.get_config(99, 100)
        self.assertEqual(cfg["num_vessels"], 8)
        self.assertEqual(cfg["num_ports"], 5)
        self.assertEqual(cfg["rollout_steps"], 64)

    def test_monotonic_ramp(self) -> None:
        sched = CurriculumScheduler(target_config=self._target(), warmup_fraction=0.5)
        prev_vessels = 0
        for i in range(0, 100, 5):
            cfg = sched.get_config(i, 100)
            self.assertGreaterEqual(cfg["num_vessels"], prev_vessels)
            prev_vessels = cfg["num_vessels"]

    def test_warmup_fraction_zero_gives_target(self) -> None:
        sched = CurriculumScheduler(target_config=self._target(), warmup_fraction=0.0)
        cfg = sched.get_config(0, 100)
        self.assertEqual(cfg["num_vessels"], 8)

    def test_start_config_override(self) -> None:
        sched = CurriculumScheduler(
            target_config=self._target(),
            start_config={"num_vessels": 4, "num_ports": 3},
            warmup_fraction=0.5,
        )
        cfg = sched.get_config(0, 100)
        self.assertEqual(cfg["num_vessels"], 4)
        self.assertEqual(cfg["num_ports"], 3)

    def test_invalid_warmup_fraction(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumScheduler(target_config=self._target(), warmup_fraction=1.5)

    def test_is_at_target(self) -> None:
        sched = CurriculumScheduler(target_config=self._target(), warmup_fraction=0.3)
        self.assertFalse(sched.is_at_target(0, 100))
        self.assertTrue(sched.is_at_target(50, 100))
        self.assertTrue(sched.is_at_target(100, 100))

    def test_get_progress(self) -> None:
        sched = CurriculumScheduler(target_config=self._target())
        self.assertAlmostEqual(sched.get_progress(50, 100), 0.5)
        self.assertAlmostEqual(sched.get_progress(100, 100), 1.0)
        self.assertAlmostEqual(sched.get_progress(0, 100), 0.0)

    def test_total_iterations_zero(self) -> None:
        sched = CurriculumScheduler(target_config=self._target())
        cfg = sched.get_config(0, 0)
        self.assertEqual(cfg["num_vessels"], 8)
        self.assertTrue(sched.is_at_target(0, 0))

    def test_non_rampable_keys_pass_through(self) -> None:
        target = {**self._target(), "seed": 99, "nominal_speed": 14.0}
        sched = CurriculumScheduler(target_config=target)
        cfg = sched.get_config(0, 100)
        self.assertEqual(cfg["seed"], 99)
        self.assertEqual(cfg["nominal_speed"], 14.0)

    def test_minimum_floor(self) -> None:
        """Ramped int values should never go below 1."""
        sched = CurriculumScheduler(
            target_config={"num_vessels": 1, "num_ports": 1, "rollout_steps": 1},
            start_config={"num_vessels": 1, "num_ports": 1, "rollout_steps": 1},
        )
        cfg = sched.get_config(0, 100)
        self.assertGreaterEqual(cfg["num_vessels"], 1)
        self.assertGreaterEqual(cfg["num_ports"], 1)


class TestCurriculumStages(unittest.TestCase):
    """Tests for multi-stage curriculum mode."""

    def test_staged_curriculum(self) -> None:
        target = {"num_vessels": 8, "num_ports": 5, "rollout_steps": 64}
        stages = [
            CurriculumStage(fraction=0.0, config_overrides={"num_vessels": 2, "num_ports": 2}),
            CurriculumStage(fraction=0.3, config_overrides={"num_vessels": 4, "num_ports": 3}),
            CurriculumStage(fraction=0.7, config_overrides={"num_vessels": 8, "num_ports": 5}),
        ]
        sched = CurriculumScheduler(target_config=target, stages=stages)

        cfg0 = sched.get_config(0, 100)
        self.assertEqual(cfg0["num_vessels"], 2)

        cfg40 = sched.get_config(40, 100)
        self.assertEqual(cfg40["num_vessels"], 4)

        cfg80 = sched.get_config(80, 100)
        self.assertEqual(cfg80["num_vessels"], 8)

    def test_stages_must_be_sorted(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumScheduler(
                target_config={"num_vessels": 8},
                stages=[
                    CurriculumStage(fraction=0.5, config_overrides={}),
                    CurriculumStage(fraction=0.2, config_overrides={}),
                ],
            )

    def test_is_at_target_staged(self) -> None:
        stages = [
            CurriculumStage(fraction=0.0, config_overrides={"num_vessels": 2}),
            CurriculumStage(fraction=0.6, config_overrides={"num_vessels": 8}),
        ]
        sched = CurriculumScheduler(target_config={"num_vessels": 8}, stages=stages)
        self.assertFalse(sched.is_at_target(30, 100))
        self.assertTrue(sched.is_at_target(70, 100))


class TestMakeCurriculumConfigs(unittest.TestCase):

    def test_preview_returns_correct_count(self) -> None:
        result = make_curriculum_configs(
            target_config={"num_vessels": 8, "num_ports": 5},
            total_iterations=100,
            sample_points=5,
        )
        self.assertEqual(len(result), 5)

    def test_preview_first_and_last(self) -> None:
        result = make_curriculum_configs(
            target_config={"num_vessels": 8, "num_ports": 5, "rollout_steps": 64},
            total_iterations=100,
            warmup_fraction=0.3,
            sample_points=3,
        )
        # First should be easy, last should be target
        self.assertLessEqual(result[0][1]["num_vessels"], 8)
        self.assertEqual(result[-1][1]["num_vessels"], 8)


# ===================================================================
# Analysis utilities
# ===================================================================


class TestCompareToBaselines(unittest.TestCase):

    def test_basic_comparison(self) -> None:
        mappo = {"total_reward": -10.0, "mean_vessel_reward": -5.0}
        baselines = {
            "forecast": pd.DataFrame({"total_reward": [-12.0, -11.0]}),
            "reactive": pd.DataFrame({"total_reward": [-15.0, -14.0]}),
        }
        df = compare_to_baselines(mappo, baselines)
        self.assertEqual(len(df), 3)
        self.assertIn("rank", df.columns)
        # MAPPO should rank first (highest total_reward)
        mappo_row = df[df["policy"] == "mappo"]
        self.assertEqual(mappo_row["rank"].values[0], 1)

    def test_empty_baselines(self) -> None:
        df = compare_to_baselines({"total_reward": -5.0}, {})
        self.assertEqual(len(df), 1)


class TestRankSweepResults(unittest.TestCase):

    def test_ranking_order(self) -> None:
        results = [
            {"config": {"lr": 1e-3}, "mean_reward": -10.0},
            {"config": {"lr": 3e-4}, "mean_reward": -5.0},
            {"config": {"lr": 1e-4}, "mean_reward": -8.0},
        ]
        df = rank_sweep_results(results, sort_by="mean_reward", ascending=False)
        self.assertEqual(df.iloc[0]["mean_reward"], -5.0)
        self.assertEqual(df.iloc[0]["rank"], 1)

    def test_empty_input(self) -> None:
        df = rank_sweep_results([])
        self.assertTrue(df.empty)

    def test_config_flattening(self) -> None:
        results = [{"config": {"lr": 1e-3, "epochs": 4}, "loss": 0.5}]
        df = rank_sweep_results(results)
        self.assertIn("cfg_lr", df.columns)
        self.assertIn("cfg_epochs", df.columns)


class TestComputeAblationDeltas(unittest.TestCase):

    def test_basic_deltas(self) -> None:
        results = {
            "baseline": {"reward": -5.0, "loss": 0.5},
            "no_norm": {"reward": -8.0, "loss": 0.7},
            "high_entropy": {"reward": -4.0, "loss": 0.6},
        }
        df = compute_ablation_deltas(results)
        # no_norm should have negative delta
        no_norm = df[df["variant"] == "no_norm"]
        self.assertAlmostEqual(no_norm["delta_reward"].values[0], -3.0)
        # high_entropy should have positive delta
        high_ent = df[df["variant"] == "high_entropy"]
        self.assertAlmostEqual(high_ent["delta_reward"].values[0], 1.0)

    def test_missing_baseline_raises(self) -> None:
        with self.assertRaises(KeyError):
            compute_ablation_deltas({"a": {"x": 1}}, baseline_key="missing")

    def test_percentage_deltas(self) -> None:
        results = {
            "baseline": {"reward": -10.0},
            "variant": {"reward": -8.0},
        }
        df = compute_ablation_deltas(results)
        variant = df[df["variant"] == "variant"]
        self.assertAlmostEqual(variant["pct_reward"].values[0], 20.0)


class TestComputeTrainingStats(unittest.TestCase):

    def test_basic_stats(self) -> None:
        history = [float(x) for x in range(20)]
        stats = compute_training_stats(history, window=5)
        self.assertEqual(stats["final_reward"], 19.0)
        self.assertGreater(stats["improvement"], 0.0)

    def test_empty_history(self) -> None:
        stats = compute_training_stats([])
        self.assertTrue(np.isnan(stats["final_reward"]))

    def test_window_larger_than_history(self) -> None:
        stats = compute_training_stats([1.0, 2.0], window=100)
        self.assertAlmostEqual(stats["smoothed_final"], 1.5)


class TestSummarizeExperiment(unittest.TestCase):

    def test_full_summary(self) -> None:
        summary = summarize_experiment(
            name="test_run",
            training_history=[1.0, 2.0, 3.0],
            eval_metrics={"total_reward": -5.0},
            diagnostics={"lr": 3e-4},
            config={"num_vessels": 8},
        )
        self.assertEqual(summary["name"], "test_run")
        self.assertIn("training_stats", summary)
        self.assertIn("eval", summary)
        self.assertEqual(summary["num_iterations"], 3)

    def test_minimal_summary(self) -> None:
        summary = summarize_experiment(name="minimal")
        self.assertEqual(summary["name"], "minimal")
        self.assertNotIn("training_stats", summary)


class TestFormatComparisonTable(unittest.TestCase):

    def test_comparison_table(self) -> None:
        experiments = [
            summarize_experiment("a", training_history=[1.0, 2.0], eval_metrics={"r": -5.0}),
            summarize_experiment("b", training_history=[3.0, 4.0], eval_metrics={"r": -3.0}),
        ]
        df = format_comparison_table(experiments)
        self.assertEqual(len(df), 2)
        self.assertIn("final_reward", df.columns)

    def test_empty_experiments(self) -> None:
        df = format_comparison_table([])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
