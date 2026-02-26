"""Close three identified test coverage gaps.

1. compare_to_baselines end-to-end with realistic baseline DataFrames
2. evaluate_episodes seed variation (different episodes → different results)
3. _rebuild_env with dimension change (networks rebuilt from scratch)
"""

from __future__ import annotations

import unittest
from typing import Any

import pandas as pd
import torch

from hmarl_mvp.analysis import compare_to_baselines
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

# -----------------------------------------------------------------------
# Gap 1: compare_to_baselines end-to-end
# -----------------------------------------------------------------------


class TestCompareToBaselinesEndToEnd(unittest.TestCase):
    """End-to-end test of compare_to_baselines with realistic DataFrames."""

    @staticmethod
    def _make_baseline_df(rows: list[dict[str, float]]) -> pd.DataFrame:
        """Create a DataFrame mimicking run_experiment() output."""
        return pd.DataFrame(rows)

    def test_ranking_correct_order(self) -> None:
        """Best total_reward should rank first."""
        mappo_metrics: dict[str, float] = {
            "mean_vessel_reward": -2.0,
            "mean_port_reward": -1.0,
            "mean_coordinator_reward": -0.5,
            "total_reward": -3.5,
        }
        baselines = {
            "independent": self._make_baseline_df([
                {"mean_vessel_reward": -6.0, "mean_port_reward": -4.0,
                 "mean_coordinator_reward": -3.0},
                {"mean_vessel_reward": -5.0, "mean_port_reward": -3.5,
                 "mean_coordinator_reward": -2.5},
            ]),
            "reactive": self._make_baseline_df([
                {"mean_vessel_reward": -3.0, "mean_port_reward": -2.0,
                 "mean_coordinator_reward": -1.5},
            ]),
        }
        result = compare_to_baselines(mappo_metrics, baselines)
        self.assertEqual(len(result), 3)  # mappo + 2 baselines
        self.assertIn("rank", result.columns)
        # MAPPO has -3.5 total, reactive synthesizes to ~-6.5, independent ~-12.25
        first_policy = result.iloc[0]["policy"]
        self.assertEqual(first_policy, "mappo")

    def test_no_nan_in_output(self) -> None:
        """Output should have no NaN for metrics present in all sources."""
        mappo_metrics: dict[str, float] = {
            "mean_vessel_reward": -1.0,
            "mean_port_reward": -2.0,
            "mean_coordinator_reward": -3.0,
            "total_reward": -6.0,
        }
        baselines = {
            "forecast": self._make_baseline_df([
                {"mean_vessel_reward": -1.5, "mean_port_reward": -2.5,
                 "mean_coordinator_reward": -3.5},
            ]),
        }
        result = compare_to_baselines(mappo_metrics, baselines)
        # total_reward is synthesized for baselines, so all rows should have it
        self.assertFalse(result["total_reward"].isna().any())

    def test_multiple_baselines_all_present(self) -> None:
        """All baselines should appear as rows, plus MAPPO."""
        mappo_metrics: dict[str, float] = {"total_reward": -5.0}
        baselines = {
            "independent": self._make_baseline_df([{"total_reward": -10.0}]),
            "reactive": self._make_baseline_df([{"total_reward": -8.0}]),
            "forecast": self._make_baseline_df([{"total_reward": -7.0}]),
            "oracle": self._make_baseline_df([{"total_reward": -6.0}]),
        }
        result = compare_to_baselines(
            mappo_metrics, baselines, metric_keys=["total_reward"],
        )
        policies = set(result["policy"])
        self.assertEqual(policies, {"mappo", "independent", "reactive", "forecast", "oracle"})
        self.assertEqual(len(result), 5)

    def test_synthesized_total_from_components(self) -> None:
        """When baseline lacks total_reward, it should be synthesized."""
        mappo_metrics: dict[str, float] = {"total_reward": -5.0}
        baselines = {
            "heuristic": self._make_baseline_df([
                {"mean_vessel_reward": -1.0, "mean_port_reward": -2.0,
                 "mean_coordinator_reward": -3.0},
            ]),
        }
        result = compare_to_baselines(mappo_metrics, baselines)
        heur_row = result[result["policy"] == "heuristic"].iloc[0]
        self.assertAlmostEqual(heur_row["total_reward"], -6.0)


# -----------------------------------------------------------------------
# Gap 2: evaluate_episodes seed variation
# -----------------------------------------------------------------------


class TestEvaluateEpisodesSeedVariation(unittest.TestCase):
    """Verify that different episodes produce distinct results."""

    def _make_trainer(self) -> MAPPOTrainer:
        cfg = MAPPOConfig(rollout_length=4)
        return MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 3, "rollout_steps": 10},
            mappo_config=cfg,
            seed=42,
        )

    def test_episodes_differ(self) -> None:
        """Multiple episodes should NOT all be identical (std > 0 somewhere)."""
        trainer = self._make_trainer()
        result = trainer.evaluate_episodes(num_episodes=5)
        episodes = result["episodes"]
        self.assertEqual(len(episodes), 5)
        # At least one metric should have non-zero std across episodes
        any_variation = any(v > 0 for v in result["std"].values())
        self.assertTrue(
            any_variation,
            "Expected at least one metric with non-zero std across episodes "
            f"but got all-zero stds: {result['std']}",
        )

    def test_seed_restored_after_evaluation(self) -> None:
        """Original env seed should be restored after evaluate_episodes."""
        trainer = self._make_trainer()
        original_seed = trainer.env.seed
        trainer.evaluate_episodes(num_episodes=3)
        self.assertEqual(trainer.env.seed, original_seed)

    def test_per_episode_results_are_complete(self) -> None:
        """Each episode dict should have all expected reward keys."""
        trainer = self._make_trainer()
        result = trainer.evaluate_episodes(num_episodes=2)
        expected_keys = {"mean_vessel_reward", "mean_port_reward", "mean_coordinator_reward"}
        for ep in result["episodes"]:
            self.assertTrue(expected_keys.issubset(ep.keys()), f"Missing keys in {ep.keys()}")


# -----------------------------------------------------------------------
# Gap 3: _rebuild_env with dimension change
# -----------------------------------------------------------------------


class TestRebuildEnvDimensionChange(unittest.TestCase):
    """Verify _rebuild_env rebuilds networks when observation dims change."""

    def test_dimension_change_rebuilds_networks(self) -> None:
        """Changing num_ports should change obs dims and rebuild networks."""
        cfg = MAPPOConfig(rollout_length=4)
        small_env: dict[str, Any] = {"num_vessels": 2, "num_ports": 2, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=small_env, mappo_config=cfg, seed=42)

        old_coord_param_id = id(list(trainer.actor_critics["coordinator"].parameters())[0])
        old_obs_dims = dict(trainer.obs_dims)

        # Increase num_ports → changes coordinator obs dim
        new_env: dict[str, Any] = {"num_vessels": 2, "num_ports": 5, "rollout_steps": 10}
        trainer._rebuild_env(new_env)

        # Coordinator observation dimensions should have changed
        self.assertNotEqual(
            trainer.obs_dims["coordinator"], old_obs_dims["coordinator"],
            "Expected coordinator obs dim to change when num_ports changes",
        )

        # Network parameter objects should be NEW (rebuilt from scratch)
        new_coord_param_id = id(list(trainer.actor_critics["coordinator"].parameters())[0])
        self.assertNotEqual(
            old_coord_param_id, new_coord_param_id,
            "Expected coordinator network to be rebuilt with new parameter objects",
        )

    def test_dimension_change_new_optimizers(self) -> None:
        """After dim change, optimizers should track the new parameters."""
        cfg = MAPPOConfig(rollout_length=4)
        small_env: dict[str, Any] = {"num_vessels": 2, "num_ports": 2, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=small_env, mappo_config=cfg)

        new_env: dict[str, Any] = {"num_vessels": 2, "num_ports": 5, "rollout_steps": 10}
        trainer._rebuild_env(new_env)

        # Optimizers should reference the new network params
        for name, opt in trainer.optimizers.items():
            ac = trainer.actor_critics[name]
            opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
            net_param_ids = {id(p) for p in ac.parameters()}
            self.assertEqual(opt_param_ids, net_param_ids,
                             f"Optimizer for {name} does not track rebuilt network params")

    def test_rebuilt_trainer_can_evaluate(self) -> None:
        """After rebuild with new dims, evaluate() should still work."""
        cfg = MAPPOConfig(rollout_length=4)
        small_env: dict[str, Any] = {"num_vessels": 2, "num_ports": 2, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=small_env, mappo_config=cfg)

        new_env: dict[str, Any] = {"num_vessels": 3, "num_ports": 4, "rollout_steps": 10}
        trainer._rebuild_env(new_env)

        # Should not crash
        result = trainer.evaluate(num_steps=5)
        self.assertIn("mean_vessel_reward", result)
        self.assertIn("mean_port_reward", result)

    def test_same_dims_preserves_networks(self) -> None:
        """Sanity: same dims should keep existing network weights."""
        cfg = MAPPOConfig(rollout_length=4)
        env_cfg: dict[str, Any] = {"num_vessels": 2, "num_ports": 3, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg)
        old_params = {
            name: list(ac.parameters())[0].data.clone()
            for name, ac in trainer.actor_critics.items()
        }
        trainer._rebuild_env(env_cfg)
        for name, ac in trainer.actor_critics.items():
            new_param = list(ac.parameters())[0].data
            self.assertTrue(
                torch.equal(old_params[name], new_param),
                f"Expected {name} params preserved for same-dim rebuild",
            )


if __name__ == "__main__":
    unittest.main()
