"""Tests for MAPPO hyperparam sweep, ablation runner, policies, and forecasts."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import run_mappo_ablation, run_mappo_hyperparam_sweep
from hmarl_mvp.forecasts import MediumTermForecaster, OracleForecaster, ShortTermForecaster
from hmarl_mvp.policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from hmarl_mvp.state import PortState, VesselState, make_rng

# -----------------------------------------------------------------------
# Hyperparameter sweep tests
# -----------------------------------------------------------------------


class HyperparamSweepTests(unittest.TestCase):
    """Tests for run_mappo_hyperparam_sweep."""

    def test_single_param_sweep(self) -> None:
        """Sweep over one parameter with two values produces two rows."""
        df = run_mappo_hyperparam_sweep(
            param_grid={"lr": [1e-4, 3e-4]},
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        self.assertEqual(len(df), 2)
        self.assertIn("lr", df.columns)
        self.assertIn("final_mean_reward", df.columns)
        self.assertIn("best_mean_reward", df.columns)
        self.assertIn("total_reward", df.columns)

    def test_multi_param_grid(self) -> None:
        """Sweep over two params produces cartesian product rows."""
        df = run_mappo_hyperparam_sweep(
            param_grid={"lr": [1e-4, 3e-4], "entropy_coeff": [0.01, 0.05]},
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        self.assertEqual(len(df), 4)  # 2 × 2
        self.assertIn("entropy_coeff", df.columns)

    def test_eval_metrics_present(self) -> None:
        """Evaluation metrics from trainer.evaluate() are in output."""
        df = run_mappo_hyperparam_sweep(
            param_grid={"lr": [3e-4]},
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        for col in ["mean_vessel_reward", "mean_port_reward",
                     "mean_coordinator_reward", "total_reward"]:
            self.assertIn(col, df.columns)

    def test_values_are_finite(self) -> None:
        """All numeric values in sweep output are finite."""
        df = run_mappo_hyperparam_sweep(
            param_grid={"lr": [3e-4]},
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        for col in df.select_dtypes(include="number").columns:
            self.assertTrue(df[col].apply(np.isfinite).all(), f"Non-finite in {col}")


# -----------------------------------------------------------------------
# Ablation runner tests
# -----------------------------------------------------------------------


class AblationRunnerTests(unittest.TestCase):
    """Tests for run_mappo_ablation."""

    def test_basic_ablation(self) -> None:
        """Two named ablations produce two rows with labels."""
        df = run_mappo_ablation(
            ablations={
                "baseline": {},
                "no_reward_norm": {"normalize_rewards": False},
            },
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        self.assertEqual(len(df), 2)
        self.assertIn("ablation", df.columns)
        self.assertSetEqual(set(df["ablation"]), {"baseline", "no_reward_norm"})

    def test_ablation_has_diagnostics(self) -> None:
        """Ablation output includes diagnostics columns."""
        df = run_mappo_ablation(
            ablations={"baseline": {}},
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        diag_cols = [c for c in df.columns if c.startswith("diag_")]
        self.assertGreater(len(diag_cols), 0)
        self.assertIn("diag_vessel_weight_norm", df.columns)
        self.assertIn("diag_lr", df.columns)

    def test_single_ablation(self) -> None:
        """A single ablation produces one row."""
        df = run_mappo_ablation(
            ablations={"only_one": {"entropy_coeff": 0.05}},
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ablation"], "only_one")

    def test_ablation_values_finite(self) -> None:
        """All numeric ablation values are finite."""
        df = run_mappo_ablation(
            ablations={"test": {}},
            train_iterations=2,
            rollout_length=16,
            eval_steps=8,
            seed=42,
            config={"num_vessels": 3, "num_ports": 3},
        )
        for col in df.select_dtypes(include="number").columns:
            self.assertTrue(df[col].apply(np.isfinite).all(), f"Non-finite in {col}")


# -----------------------------------------------------------------------
# Policy tests (comprehensive — previously only smoke-tested)
# -----------------------------------------------------------------------


class FleetCoordinatorPolicyTests(unittest.TestCase):
    """Thorough tests for FleetCoordinatorPolicy across all modes."""

    def setUp(self) -> None:
        self.cfg = get_default_config(num_vessels=5, num_ports=4)
        self.ports = [
            PortState(port_id=i, queue=i * 2, docks=3, occupied=min(i, 3))
            for i in range(4)
        ]
        self.vessels = [
            VesselState(vessel_id=i, location=i % 4, destination=(i + 1) % 4)
            for i in range(5)
        ]
        self.rng = make_rng(42)

    def test_independent_returns_required_keys(self) -> None:
        policy = FleetCoordinatorPolicy(self.cfg, mode="independent")
        action = policy.propose_action(
            np.zeros((4, 7)), self.vessels, self.ports, self.rng
        )
        for key in ["dest_port", "departure_window_hours", "emission_budget"]:
            self.assertIn(key, action)

    def test_independent_requires_rng(self) -> None:
        policy = FleetCoordinatorPolicy(self.cfg, mode="independent")
        with self.assertRaises(ValueError):
            policy.propose_action(np.zeros((4, 7)), self.vessels, self.ports, None)

    def test_independent_per_vessel_dest_avoids_self_loop(self) -> None:
        policy = FleetCoordinatorPolicy(self.cfg, mode="independent")
        action = policy.propose_action(
            np.zeros((4, 7)), self.vessels, self.ports, self.rng
        )
        per_vessel = action.get("per_vessel_dest", {})
        for vid, dest in per_vessel.items():
            vessel = next(v for v in self.vessels if v.vessel_id == vid)
            self.assertNotEqual(dest, vessel.location)

    def test_reactive_picks_min_queue(self) -> None:
        policy = FleetCoordinatorPolicy(self.cfg, mode="reactive")
        action = policy.propose_action(
            np.zeros((4, 7)), self.vessels, self.ports, self.rng
        )
        # Port 0 has queue=0, so reactive should pick port 0
        self.assertEqual(action["dest_port"], 0)

    def test_reactive_no_rng_needed(self) -> None:
        """Reactive mode works without rng."""
        policy = FleetCoordinatorPolicy(self.cfg, mode="reactive")
        action = policy.propose_action(
            np.zeros((4, 7)), self.vessels, self.ports, None
        )
        self.assertIn("dest_port", action)

    def test_forecast_uses_forecast_data(self) -> None:
        """Forecast mode picks port with lowest mean forecast score."""
        policy = FleetCoordinatorPolicy(self.cfg, mode="forecast")
        forecast = np.ones((4, 7))
        forecast[2, :] = 0.0  # Port 2 is least congested
        action = policy.propose_action(forecast, self.vessels, self.ports, self.rng)
        self.assertEqual(action["dest_port"], 2)

    def test_forecast_per_vessel_destinations(self) -> None:
        """Forecast mode distributes vessels across sorted ports."""
        policy = FleetCoordinatorPolicy(self.cfg, mode="forecast")
        forecast = np.array([[5, 5, 5, 5, 5, 5, 5],
                             [1, 1, 1, 1, 1, 1, 1],
                             [3, 3, 3, 3, 3, 3, 3],
                             [2, 2, 2, 2, 2, 2, 2]], dtype=float)
        action = policy.propose_action(forecast, self.vessels, self.ports, self.rng)
        per_vessel = action.get("per_vessel_dest", {})
        self.assertGreater(len(per_vessel), 0)

    def test_emission_budget_clamp(self) -> None:
        """Forecast mode clamps emission budget to minimum 10."""
        for v in self.vessels:
            v.emissions = 1000.0
        policy = FleetCoordinatorPolicy(self.cfg, mode="forecast")
        action = policy.propose_action(
            np.zeros((4, 7)), self.vessels, self.ports, self.rng
        )
        self.assertGreaterEqual(action["emission_budget"], 10.0)


class VesselPolicyTests(unittest.TestCase):
    """Thorough tests for VesselPolicy across all modes."""

    def setUp(self) -> None:
        self.cfg = get_default_config()
        self.directive = {"dest_port": 1, "departure_window_hours": 12}
        self.forecast = np.ones((5, 12))

    def test_independent_uses_nominal_speed(self) -> None:
        policy = VesselPolicy(self.cfg, mode="independent")
        action = policy.propose_action(self.forecast, self.directive)
        self.assertAlmostEqual(action["target_speed"], self.cfg["nominal_speed"])

    def test_reactive_uses_nominal_speed(self) -> None:
        policy = VesselPolicy(self.cfg, mode="reactive")
        action = policy.propose_action(self.forecast, self.directive)
        self.assertAlmostEqual(action["target_speed"], self.cfg["nominal_speed"])

    def test_independent_reactive_identical(self) -> None:
        """Independent and reactive produce identical output."""
        p1 = VesselPolicy(self.cfg, mode="independent")
        p2 = VesselPolicy(self.cfg, mode="reactive")
        a1 = p1.propose_action(self.forecast, self.directive)
        a2 = p2.propose_action(self.forecast, self.directive)
        self.assertEqual(a1, a2)

    def test_forecast_high_congestion_slows_down(self) -> None:
        """When congestion > 5, vessel slows to speed_min."""
        policy = VesselPolicy(self.cfg, mode="forecast")
        forecast = np.full((5, 12), 10.0)
        action = policy.propose_action(forecast, self.directive)
        self.assertAlmostEqual(action["target_speed"], self.cfg["speed_min"])

    def test_forecast_low_congestion_speeds_up(self) -> None:
        """When congestion < 3, vessel uses speed_max."""
        policy = VesselPolicy(self.cfg, mode="forecast")
        forecast = np.full((5, 12), 1.0)
        action = policy.propose_action(forecast, self.directive)
        self.assertAlmostEqual(action["target_speed"], self.cfg["speed_max"])

    def test_forecast_medium_congestion_nominal(self) -> None:
        """When 3 < congestion < 5, vessel uses nominal speed."""
        policy = VesselPolicy(self.cfg, mode="forecast")
        forecast = np.full((5, 12), 4.0)
        action = policy.propose_action(forecast, self.directive)
        self.assertAlmostEqual(action["target_speed"], self.cfg["nominal_speed"])

    def test_always_requests_arrival_slot(self) -> None:
        """All modes always request arrival slot."""
        for mode in ["independent", "reactive", "forecast"]:
            policy = VesselPolicy(self.cfg, mode=mode)
            action = policy.propose_action(self.forecast, self.directive)
            self.assertTrue(action["request_arrival_slot"], f"mode={mode}")


class PortPolicyTests(unittest.TestCase):
    """Thorough tests for PortPolicy across all modes."""

    def setUp(self) -> None:
        self.cfg = get_default_config()
        self.forecast_row = np.ones(12)

    def test_independent_accepts_up_to_available(self) -> None:
        port = PortState(port_id=0, queue=3, docks=4, occupied=2)
        policy = PortPolicy(self.cfg, mode="independent")
        action = policy.propose_action(port, incoming_requests=5, short_forecast_row=self.forecast_row)
        self.assertLessEqual(action["accept_requests"], port.docks - port.occupied)

    def test_independent_service_rate_one(self) -> None:
        port = PortState(port_id=0, queue=3, docks=4, occupied=2)
        policy = PortPolicy(self.cfg, mode="independent")
        action = policy.propose_action(port, incoming_requests=1, short_forecast_row=self.forecast_row)
        self.assertEqual(action["service_rate"], 1)

    def test_reactive_scales_with_queue(self) -> None:
        port = PortState(port_id=0, queue=5, docks=4, occupied=1)
        policy = PortPolicy(self.cfg, mode="reactive")
        action = policy.propose_action(port, incoming_requests=2, short_forecast_row=self.forecast_row)
        self.assertGreater(action["service_rate"], 1)

    def test_forecast_high_pressure_max_service(self) -> None:
        """High forecast pressure → service_rate = docks."""
        port = PortState(port_id=0, queue=1, docks=4, occupied=1)
        policy = PortPolicy(self.cfg, mode="forecast")
        high_forecast = np.full(12, 10.0)
        action = policy.propose_action(port, incoming_requests=2, short_forecast_row=high_forecast)
        self.assertEqual(action["service_rate"], port.docks)

    def test_forecast_low_pressure_conservative(self) -> None:
        """Low forecast pressure and low queue → conservative service rate."""
        port = PortState(port_id=0, queue=0, docks=4, occupied=1)
        policy = PortPolicy(self.cfg, mode="forecast")
        low_forecast = np.full(12, 1.0)
        action = policy.propose_action(port, incoming_requests=0, short_forecast_row=low_forecast)
        self.assertLessEqual(action["service_rate"], port.occupied + 1)

    def test_accept_requests_never_negative(self) -> None:
        """Accept requests is non-negative even when docks are full."""
        port = PortState(port_id=0, queue=10, docks=3, occupied=3)
        for mode in ["independent", "reactive", "forecast"]:
            policy = PortPolicy(self.cfg, mode=mode)
            action = policy.propose_action(port, incoming_requests=5, short_forecast_row=self.forecast_row)
            self.assertGreaterEqual(action["accept_requests"], 0, f"mode={mode}")


# -----------------------------------------------------------------------
# Forecaster tests (comprehensive — previously only shape-tested)
# -----------------------------------------------------------------------


class MediumTermForecasterTests(unittest.TestCase):
    """Tests for MediumTermForecaster."""

    def setUp(self) -> None:
        self.ports = [PortState(port_id=i, queue=i * 2, docks=3, occupied=0)
                      for i in range(4)]
        self.rng = make_rng(42)

    def test_output_shape(self) -> None:
        f = MediumTermForecaster(horizon_days=7)
        result = f.predict(self.ports, self.rng)
        self.assertEqual(result.shape, (4, 7))

    def test_non_negative(self) -> None:
        f = MediumTermForecaster(horizon_days=7)
        result = f.predict(self.ports, self.rng)
        self.assertTrue((result >= 0).all())

    def test_deterministic_with_same_rng(self) -> None:
        f = MediumTermForecaster(horizon_days=7)
        r1 = f.predict(self.ports, make_rng(99))
        r2 = f.predict(self.ports, make_rng(99))
        np.testing.assert_array_equal(r1, r2)

    def test_reflects_current_queue(self) -> None:
        """Higher queue ports produce higher baseline forecast."""
        f = MediumTermForecaster(horizon_days=7)
        result = f.predict(self.ports, self.rng)
        # Port 3 (queue 6) should have higher mean than port 0 (queue 0)
        self.assertGreater(result[3].mean(), result[0].mean())

    def test_different_horizons(self) -> None:
        for h in [1, 3, 14]:
            f = MediumTermForecaster(horizon_days=h)
            result = f.predict(self.ports, make_rng(42))
            self.assertEqual(result.shape, (4, h))

    def test_single_port(self) -> None:
        ports = [PortState(port_id=0, queue=5, docks=2, occupied=0)]
        f = MediumTermForecaster(horizon_days=7)
        result = f.predict(ports, self.rng)
        self.assertEqual(result.shape, (1, 7))


class ShortTermForecasterTests(unittest.TestCase):
    """Tests for ShortTermForecaster."""

    def setUp(self) -> None:
        self.ports = [PortState(port_id=i, queue=i, docks=3, occupied=0)
                      for i in range(5)]
        self.rng = make_rng(42)

    def test_output_shape(self) -> None:
        f = ShortTermForecaster(horizon_hours=12)
        result = f.predict(self.ports, self.rng)
        self.assertEqual(result.shape, (5, 12))

    def test_non_negative(self) -> None:
        f = ShortTermForecaster(horizon_hours=12)
        result = f.predict(self.ports, self.rng)
        self.assertTrue((result >= 0).all())

    def test_deterministic_with_same_rng(self) -> None:
        f = ShortTermForecaster(horizon_hours=12)
        r1 = f.predict(self.ports, make_rng(99))
        r2 = f.predict(self.ports, make_rng(99))
        np.testing.assert_array_equal(r1, r2)

    def test_reflects_current_queue(self) -> None:
        f = ShortTermForecaster(horizon_hours=12)
        result = f.predict(self.ports, self.rng)
        self.assertGreater(result[4].mean(), result[0].mean())

    def test_different_horizons(self) -> None:
        for h in [1, 6, 24]:
            f = ShortTermForecaster(horizon_hours=h)
            result = f.predict(self.ports, make_rng(42))
            self.assertEqual(result.shape, (5, h))


class OracleForecasterTests(unittest.TestCase):
    """Tests for OracleForecaster."""

    def setUp(self) -> None:
        self.ports = [PortState(port_id=i, queue=i * 3, docks=3, occupied=0)
                      for i in range(4)]

    def test_output_shapes(self) -> None:
        f = OracleForecaster(medium_horizon_days=7, short_horizon_hours=12)
        medium, short = f.predict(self.ports)
        self.assertEqual(medium.shape, (4, 7))
        self.assertEqual(short.shape, (4, 12))

    def test_oracle_repeats_current_queue(self) -> None:
        """Oracle forecast is a constant repeat of current queue."""
        f = OracleForecaster(medium_horizon_days=5, short_horizon_hours=6)
        medium, short = f.predict(self.ports)
        for i, p in enumerate(self.ports):
            np.testing.assert_array_equal(medium[i], p.queue)
            np.testing.assert_array_equal(short[i], p.queue)

    def test_no_rng_required(self) -> None:
        """Oracle does not need an rng argument."""
        f = OracleForecaster(medium_horizon_days=7, short_horizon_hours=12)
        # Should accept no rng
        medium, short = f.predict(self.ports)
        self.assertIsInstance(medium, np.ndarray)

    def test_different_api_from_heuristic(self) -> None:
        """Oracle returns a tuple, unlike the other forecasters."""
        f = OracleForecaster(medium_horizon_days=7, short_horizon_hours=12)
        result = f.predict(self.ports)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
