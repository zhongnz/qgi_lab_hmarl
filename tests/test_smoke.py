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
        self.assertIn("coordinators", obs)
        self.assertIn("vessels", obs)
        self.assertIn("ports", obs)
        self.assertEqual(len(obs["vessels"]), cfg["num_vessels"])
        self.assertEqual(len(obs["ports"]), cfg["num_ports"])

        actions = env.sample_stub_actions()
        next_obs, rewards, done, info = env.step(actions)
        self.assertIn("coordinator", next_obs)
        self.assertIn("coordinators", next_obs)
        self.assertIn("coordinator", rewards)
        self.assertIn("coordinators", rewards)
        self.assertIn("port_metrics", info)
        self.assertFalse(done)

    def test_global_state_is_stable_without_step(self) -> None:
        env = MaritimeEnv(config=get_default_config(rollout_steps=2), seed=42)
        env.reset()
        state_1 = env.get_global_state()
        state_2 = env.get_global_state()
        self.assertTrue(np.array_equal(state_1, state_2))

    def test_async_latency_delays_dispatch(self) -> None:
        cfg = get_default_config(
            num_ports=2,
            num_vessels=1,
            num_coordinators=2,
            rollout_steps=8,
            coord_decision_interval_steps=3,
            vessel_decision_interval_steps=1,
            port_decision_interval_steps=1,
            message_latency_steps=2,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()
        vessel = env.vessels[0]
        vessel.location = 0
        vessel.destination = 0
        vessel.at_sea = False
        vessel.position_nm = 0.0
        env.ports[1].occupied = 0
        env.ports[1].docks = max(env.ports[1].docks, 1)

        fixed_actions = {
            "coordinators": [
                {"dest_port": 1, "departure_window_hours": 12, "emission_budget": 50.0},
                {"dest_port": 1, "departure_window_hours": 12, "emission_budget": 50.0},
            ],
            "coordinator": {"dest_port": 1, "departure_window_hours": 12, "emission_budget": 50.0},
            "vessels": [{"target_speed": cfg["nominal_speed"], "request_arrival_slot": True}],
            "ports": [
                {"service_rate": 1, "accept_requests": 0},
                {"service_rate": 1, "accept_requests": 1},
            ],
        }

        at_sea_history = []
        for _ in range(5):
            env.step(fixed_actions)
            at_sea_history.append(env.vessels[0].at_sea)

        self.assertEqual(at_sea_history[:4], [False, False, False, False])
        self.assertTrue(at_sea_history[4])

    def test_run_experiment_supports_multi_coordinator_rollout(self) -> None:
        cfg = get_default_config(
            num_coordinators=2,
            rollout_steps=4,
            coord_decision_interval_steps=2,
            message_latency_steps=1,
        )
        df = run_experiment(policy_type="forecast", steps=4, seed=42, config=cfg)
        self.assertEqual(len(df), 4)
        self.assertIn("num_coordinators", df.columns)
        self.assertIn("pending_arrival_requests", df.columns)
        self.assertTrue((df["num_coordinators"] == 2).all())

    def test_run_experiment_supports_non_default_num_ports(self) -> None:
        cfg = get_default_config(num_ports=6, num_vessels=4, rollout_steps=3)
        df = run_experiment(policy_type="forecast", steps=3, seed=42, config=cfg)
        self.assertEqual(len(df), 3)
        self.assertIn("avg_queue", df.columns)

    def test_forecast_and_oracle_generate_accepted_requests(self) -> None:
        cfg = get_default_config(
            num_ports=3,
            num_vessels=6,
            docks_per_port=5,
            rollout_steps=12,
            message_latency_steps=2,
        )
        forecast_df = run_experiment(policy_type="forecast", steps=12, seed=42, config=cfg)
        oracle_df = run_experiment(policy_type="oracle", steps=12, seed=42, config=cfg)

        self.assertGreater(float(forecast_df["total_vessel_requests"].iloc[-1]), 0.0)
        self.assertGreater(float(oracle_df["total_vessel_requests"].iloc[-1]), 0.0)
        self.assertGreater(float(forecast_df["total_port_accepted"].iloc[-1]), 0.0)
        self.assertGreater(float(oracle_df["total_port_accepted"].iloc[-1]), 0.0)

        self.assertTrue((forecast_df["policy_agreement_rate"] >= 0.0).all())
        self.assertTrue((forecast_df["policy_agreement_rate"] <= 1.0).all())
        self.assertTrue((oracle_df["policy_agreement_rate"] >= 0.0).all())
        self.assertTrue((oracle_df["policy_agreement_rate"] <= 1.0).all())

    def test_independent_baseline_generates_operational_activity(self) -> None:
        cfg = get_default_config(
            num_ports=3,
            num_vessels=6,
            docks_per_port=6,
            rollout_steps=12,
            coord_decision_interval_steps=1,
            message_latency_steps=1,
        )
        df = run_experiment(policy_type="independent", steps=12, seed=42, config=cfg)

        self.assertGreater(float(df["total_vessel_requests"].iloc[-1]), 0.0)
        self.assertGreater(float(df["total_port_accepted"].iloc[-1]), 0.0)
        self.assertGreater(float(df["total_fuel_used"].iloc[-1]), 0.0)
        self.assertGreater(float(df["total_emissions_co2"].iloc[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
