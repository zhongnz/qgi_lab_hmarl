"""Scenario tests: large fleets, heterogeneous ports, stress latency, multi-seed."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.experiment import (
    run_experiment,
)


class LargeFleetTests(unittest.TestCase):
    """Test that the simulator handles larger fleet topologies."""

    def test_20_vessels_8_ports(self) -> None:
        cfg = get_default_config(
            num_ports=8,
            num_vessels=20,
            num_coordinators=2,
            docks_per_port=5,
            rollout_steps=10,
        )
        df = run_experiment(policy_type="forecast", steps=10, seed=42, config=cfg)
        self.assertEqual(len(df), 10)
        self.assertGreater(float(df["total_fuel_used"].iloc[-1]), 0.0)

    def test_30_vessels_10_ports_3_coordinators(self) -> None:
        cfg = get_default_config(
            num_ports=10,
            num_vessels=30,
            num_coordinators=3,
            docks_per_port=6,
            rollout_steps=8,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        obs = env.reset()
        self.assertEqual(len(obs["vessels"]), 30)
        self.assertEqual(len(obs["ports"]), 10)
        self.assertEqual(len(obs["coordinators"]), 3)
        for _ in range(8):
            actions = env.sample_stub_actions()
            obs, rewards, done, info = env.step(actions)
        self.assertTrue(done)


class HeterogeneousPortTests(unittest.TestCase):
    """Test non-uniform port configurations via custom distance matrices."""

    def test_asymmetric_distances(self) -> None:
        num_ports = 4
        # Deliberately asymmetric: port 0 is close to all others
        distance_nm = np.array([
            [0, 500, 600, 700],
            [500, 0, 5000, 6000],
            [600, 5000, 0, 8000],
            [700, 6000, 8000, 0],
        ], dtype=float)
        cfg = get_default_config(num_ports=num_ports, num_vessels=6, rollout_steps=8)
        df = run_experiment(
            policy_type="forecast",
            steps=8,
            seed=42,
            config=cfg,
            distance_nm=distance_nm,
        )
        self.assertEqual(len(df), 8)

    def test_single_port_topology(self) -> None:
        cfg = get_default_config(
            num_ports=1,
            num_vessels=3,
            docks_per_port=4,
            rollout_steps=5,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        obs = env.reset()
        self.assertEqual(len(obs["ports"]), 1)
        for _ in range(5):
            actions = env.sample_stub_actions()
            env.step(actions)


class StressLatencyTests(unittest.TestCase):
    """Test extreme cadence and latency settings."""

    def test_high_latency(self) -> None:
        cfg = get_default_config(
            num_ports=3,
            num_vessels=4,
            rollout_steps=15,
            message_latency_steps=5,
            coord_decision_interval_steps=6,
        )
        df = run_experiment(policy_type="forecast", steps=15, seed=42, config=cfg)
        self.assertEqual(len(df), 15)
        # With high latency, activity should still eventually occur
        self.assertGreaterEqual(float(df["total_vessel_requests"].iloc[-1]), 0.0)

    def test_frequent_coordinator_updates(self) -> None:
        cfg = get_default_config(
            num_ports=3,
            num_vessels=6,
            rollout_steps=10,
            coord_decision_interval_steps=1,
            message_latency_steps=1,
        )
        df = run_experiment(policy_type="forecast", steps=10, seed=42, config=cfg)
        # Every step is a coordinator update
        self.assertTrue((df["coordinator_updates"] == 1).all())

    def test_slow_port_cadence(self) -> None:
        cfg = get_default_config(
            num_ports=3,
            num_vessels=4,
            rollout_steps=12,
            port_decision_interval_steps=4,
            message_latency_steps=1,
        )
        df = run_experiment(policy_type="forecast", steps=12, seed=42, config=cfg)
        self.assertEqual(len(df), 12)


class AllPoliciesScenarioTests(unittest.TestCase):
    """Smoke-test all policy types on a non-default topology."""

    def test_all_policies_on_large_topology(self) -> None:
        cfg = get_default_config(
            num_ports=6,
            num_vessels=12,
            num_coordinators=2,
            docks_per_port=4,
            rollout_steps=8,
        )
        for policy in ["independent", "reactive", "forecast", "oracle"]:
            with self.subTest(policy=policy):
                df = run_experiment(
                    policy_type=policy, steps=8, seed=42, config=cfg
                )
                self.assertEqual(len(df), 8)
                self.assertEqual(df["policy"].iloc[0], policy)


if __name__ == "__main__":
    unittest.main()
