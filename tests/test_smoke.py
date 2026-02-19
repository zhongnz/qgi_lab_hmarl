"""Smoke tests for module-first HMARL refactor."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.experiment import run_experiment


class SmokeTests(unittest.TestCase):
    def test_run_experiment_returns_expected_columns(self) -> None:
        df = run_experiment(policy_type="forecast", steps=5, seed=42)
        expected_cols = {
            "t",
            "policy",
            "avg_queue",
            "total_emissions_co2",
            "total_ops_cost_usd",
            "avg_vessel_reward",
            "coordinator_reward",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        self.assertEqual(len(df), 5)

    def test_env_reset_step_shapes(self) -> None:
        cfg = get_default_config(rollout_steps=3)
        env = MaritimeEnv(config=cfg, seed=42)
        obs = env.reset()
        self.assertIn("coordinator", obs)
        self.assertIn("vessels", obs)
        self.assertIn("ports", obs)
        self.assertEqual(len(obs["vessels"]), cfg["num_vessels"])
        self.assertEqual(len(obs["ports"]), cfg["num_ports"])

        actions = env.sample_stub_actions()
        next_obs, rewards, done, info = env.step(actions)
        self.assertIn("coordinator", next_obs)
        self.assertIn("coordinator", rewards)
        self.assertIn("port_metrics", info)
        self.assertFalse(done)

    def test_global_state_is_stable_without_step(self) -> None:
        env = MaritimeEnv(config=get_default_config(rollout_steps=2), seed=42)
        env.reset()
        state_1 = env.get_global_state()
        state_2 = env.get_global_state()
        self.assertTrue(np.array_equal(state_1, state_2))


if __name__ == "__main__":
    unittest.main()
