"""Tests for new features: gradient diagnostics, multi-seed MAPPO evaluation."""

from __future__ import annotations

import unittest

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import run_multi_seed_mappo_comparison
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer, PPOUpdateResult


class GradientDiagnosticsTests(unittest.TestCase):
    """Tests for PPOUpdateResult gradient/weight norm tracking."""

    def setUp(self) -> None:
        cfg = get_default_config(
            num_ports=3, num_vessels=3, rollout_steps=15, num_coordinators=1
        )
        mappo_cfg = MAPPOConfig(rollout_length=8, num_epochs=2, minibatch_size=8)
        self.trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=42)

    def test_update_returns_grad_and_weight_norms(self) -> None:
        self.trainer.collect_rollout()
        results = self.trainer.update()
        for agent_type, result in results.items():
            self.assertIsInstance(result, PPOUpdateResult)
            self.assertGreater(result.grad_norm, 0.0, f"{agent_type} grad_norm should be > 0")
            self.assertGreater(result.weight_norm, 0.0, f"{agent_type} weight_norm should be > 0")

    def test_get_diagnostics_returns_expected_keys(self) -> None:
        self.trainer.collect_rollout()
        self.trainer.update()
        diag = self.trainer.get_diagnostics()
        for agent_type in ("vessel", "port", "coordinator"):
            self.assertIn(f"{agent_type}_weight_norm", diag)
            self.assertIn(f"{agent_type}_param_count", diag)
            self.assertIn(f"{agent_type}_reward_mean", diag)
            self.assertIn(f"{agent_type}_reward_std", diag)
        self.assertIn("iteration", diag)
        self.assertIn("lr", diag)
        self.assertEqual(diag["iteration"], 1.0)

    def test_diagnostics_weight_norms_positive(self) -> None:
        self.trainer.collect_rollout()
        self.trainer.update()
        diag = self.trainer.get_diagnostics()
        for agent_type in ("vessel", "port", "coordinator"):
            self.assertGreater(diag[f"{agent_type}_weight_norm"], 0.0)
            self.assertGreater(diag[f"{agent_type}_param_count"], 0.0)

    def test_reward_normaliser_stats_update_after_rollout(self) -> None:
        self.trainer.collect_rollout()
        self.trainer.update()
        diag = self.trainer.get_diagnostics()
        # After collecting one rollout, reward_std should be > 0 (non-trivial)
        for agent_type in ("vessel", "port", "coordinator"):
            self.assertGreater(diag[f"{agent_type}_reward_std"], 0.0)


class MultiSeedMAPPOTests(unittest.TestCase):
    """Tests for run_multi_seed_mappo_comparison."""

    def test_basic_multi_seed_comparison(self) -> None:
        df = run_multi_seed_mappo_comparison(
            train_iterations=3,
            rollout_length=8,
            eval_steps=5,
            baselines=["independent"],
            seeds=[42, 123],
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 15},
        )
        self.assertIn("seed", df.columns)
        self.assertIn("policy", df.columns)
        policies = set(df["policy"].unique())
        self.assertIn("mappo", policies)
        self.assertIn("independent", policies)
        seeds_seen = set(df["seed"].unique())
        self.assertEqual(seeds_seen, {42, 123})

    def test_all_seeds_have_both_policies(self) -> None:
        df = run_multi_seed_mappo_comparison(
            train_iterations=2,
            rollout_length=8,
            eval_steps=3,
            baselines=["forecast"],
            seeds=[42, 256],
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 15},
        )
        for seed in [42, 256]:
            seed_df = df[df["seed"] == seed]
            policies = set(seed_df["policy"].unique())
            self.assertIn("mappo", policies)
            self.assertIn("forecast", policies)

    def test_compatible_with_summarize_multi_seed(self) -> None:
        from hmarl_mvp.experiment import summarize_multi_seed

        df = run_multi_seed_mappo_comparison(
            train_iterations=2,
            rollout_length=8,
            eval_steps=3,
            baselines=["independent"],
            seeds=[42, 123],
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 15},
        )
        summary = summarize_multi_seed(df)
        self.assertGreater(len(summary), 0)
        policies_in_summary = summary["policy"].unique()
        self.assertIn("mappo", policies_in_summary)


if __name__ == "__main__":
    unittest.main()
