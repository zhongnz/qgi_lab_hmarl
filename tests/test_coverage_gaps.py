"""Tests closing coverage gaps: agents, config utilities, dynamics, env helpers, sweeps."""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

import numpy as np

from hmarl_mvp.agents import (
    FleetCoordinatorAgent,
    FleetCoordinatorState,
    PortAgent,
    assign_vessels_to_coordinators,
)
from hmarl_mvp.config import (
    DISTANCE_NM,
    generate_distance_matrix,
    get_default_config,
    resolve_distance_matrix,
    validate_distance_matrix,
)
from hmarl_mvp.dynamics import compute_fuel_and_emissions, dispatch_vessel, step_ports, step_vessels
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.experiment import (
    run_horizon_sweep,
    run_noise_sweep,
    run_policy_sweep,
    run_sharing_sweep,
    save_result_dict,
    summarize_policy_results,
)
from hmarl_mvp.state import PortState, VesselState, initialize_vessels, make_rng

# ── Agent unit tests ─────────────────────────────────────────────────────


class FleetCoordinatorAgentTests(unittest.TestCase):
    """Direct tests for FleetCoordinatorAgent."""

    def setUp(self) -> None:
        self.cfg = get_default_config(num_ports=5, num_vessels=3)
        self.rng = make_rng(42)

    def test_state_initialises_correctly(self) -> None:
        agent = FleetCoordinatorAgent(config=self.cfg, coordinator_id=2)
        self.assertEqual(agent.state.coordinator_id, 2)
        self.assertEqual(agent.state.cumulative_emissions, 0.0)
        self.assertEqual(agent.state.emission_budget, 50.0)

    def test_get_obs_returns_expected_shape(self) -> None:
        agent = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        vessels = initialize_vessels(3, 5, self.cfg["nominal_speed"], self.rng)
        medium = np.ones((5, self.cfg["medium_horizon_days"]))
        obs = agent.get_obs(medium, vessels)
        # medium_forecast flattened + vessel_summaries flattened + 1 (emissions)
        expected = 5 * self.cfg["medium_horizon_days"] + 3 * 4 + 1
        self.assertEqual(obs.shape, (expected,))

    def test_get_obs_updates_cumulative_emissions(self) -> None:
        agent = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        v1 = VesselState(vessel_id=0, location=0, destination=1, emissions=3.5)
        v2 = VesselState(vessel_id=1, location=1, destination=2, emissions=1.5)
        medium = np.zeros((5, self.cfg["medium_horizon_days"]))
        agent.get_obs(medium, [v1, v2])
        self.assertAlmostEqual(agent.state.cumulative_emissions, 5.0)

    def test_apply_action_normalises_and_updates_state(self) -> None:
        agent = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        result = agent.apply_action({
            "dest_port": 3,
            "departure_window_hours": 24,
            "emission_budget": 30.0,
        })
        self.assertEqual(result["dest_port"], 3)
        self.assertEqual(result["departure_window_hours"], 24)
        self.assertAlmostEqual(result["emission_budget"], 30.0)
        self.assertEqual(agent.state.last_dest_port, 3)
        self.assertEqual(agent.last_action["dest_port"], 3)

    def test_apply_action_uses_defaults_when_keys_missing(self) -> None:
        agent = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        result = agent.apply_action({})
        self.assertEqual(result["dest_port"], 0)
        self.assertEqual(result["departure_window_hours"], 12)
        self.assertAlmostEqual(result["emission_budget"], 50.0)


class PortAgentTests(unittest.TestCase):
    """Direct tests for PortAgent."""

    def setUp(self) -> None:
        self.cfg = get_default_config(short_horizon_hours=4)

    def test_get_obs_returns_expected_shape(self) -> None:
        state = PortState(port_id=0, queue=5, docks=4, occupied=2)
        agent = PortAgent(state, self.cfg)
        forecast_row = np.ones(self.cfg["short_horizon_hours"])
        obs = agent.get_obs(forecast_row, incoming_requests=3)
        expected = 3 + self.cfg["short_horizon_hours"] + 1
        self.assertEqual(obs.shape, (expected,))

    def test_obs_values_are_correct(self) -> None:
        state = PortState(port_id=0, queue=7, docks=4, occupied=2)
        agent = PortAgent(state, self.cfg)
        forecast_row = np.array([0.1, 0.2, 0.3, 0.4])
        obs = agent.get_obs(forecast_row, incoming_requests=5)
        self.assertAlmostEqual(obs[0], 7.0)   # queue
        self.assertAlmostEqual(obs[1], 4.0)   # docks
        self.assertAlmostEqual(obs[2], 2.0)   # occupied
        self.assertAlmostEqual(obs[-1], 5.0)  # incoming_requests

    def test_apply_action_clips_negatives(self) -> None:
        state = PortState(port_id=0, queue=0, docks=3, occupied=0)
        agent = PortAgent(state, self.cfg)
        result = agent.apply_action({"service_rate": -1, "accept_requests": -5})
        self.assertEqual(result["service_rate"], 0)
        self.assertEqual(result["accept_requests"], 0)


class FleetCoordinatorStateTests(unittest.TestCase):
    """Direct tests for the coordinator state dataclass."""

    def test_defaults(self) -> None:
        s = FleetCoordinatorState()
        self.assertEqual(s.coordinator_id, 0)
        self.assertEqual(s.cumulative_emissions, 0.0)

    def test_custom_values(self) -> None:
        s = FleetCoordinatorState(coordinator_id=5, emission_budget=25.0)
        self.assertEqual(s.coordinator_id, 5)
        self.assertAlmostEqual(s.emission_budget, 25.0)


class AssignVesselsEdgeCaseTests(unittest.TestCase):
    """Edge-case tests for assign_vessels_to_coordinators."""

    def test_single_coordinator_gets_all_vessels(self) -> None:
        rng = make_rng(1)
        vessels = initialize_vessels(5, 3, 12.0, rng)
        groups = assign_vessels_to_coordinators(vessels, 1)
        self.assertEqual(len(groups[0]), 5)

    def test_more_coordinators_than_vessels(self) -> None:
        rng = make_rng(1)
        vessels = initialize_vessels(2, 3, 12.0, rng)
        groups = assign_vessels_to_coordinators(vessels, 5)
        total = sum(len(v) for v in groups.values())
        self.assertEqual(total, 2)
        self.assertEqual(len(groups), 5)

    def test_zero_coordinators_raises(self) -> None:
        rng = make_rng(1)
        vessels = initialize_vessels(3, 3, 12.0, rng)
        with self.assertRaises(ValueError):
            assign_vessels_to_coordinators(vessels, 0)


# ── Config utility tests ─────────────────────────────────────────────────


class GenerateDistanceMatrixTests(unittest.TestCase):
    """Tests for generate_distance_matrix."""

    def test_output_is_square_and_symmetric(self) -> None:
        m = generate_distance_matrix(7)
        self.assertEqual(m.shape, (7, 7))
        np.testing.assert_array_almost_equal(m, m.T)

    def test_diagonal_is_zero(self) -> None:
        m = generate_distance_matrix(4)
        np.testing.assert_array_equal(np.diag(m), 0.0)

    def test_off_diagonal_positive(self) -> None:
        m = generate_distance_matrix(6)
        np.fill_diagonal(m, np.inf)
        self.assertTrue(np.all(m > 0))

    def test_single_port(self) -> None:
        m = generate_distance_matrix(1)
        self.assertEqual(m.shape, (1, 1))
        self.assertAlmostEqual(m[0, 0], 0.0)

    def test_zero_ports_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_distance_matrix(0)


class ResolveDistanceMatrixTests(unittest.TestCase):
    """Tests for resolve_distance_matrix."""

    def test_default_5_port_uses_builtin(self) -> None:
        m = resolve_distance_matrix(5)
        np.testing.assert_array_equal(m, DISTANCE_NM)

    def test_non_5_port_generates_matrix(self) -> None:
        m = resolve_distance_matrix(7)
        self.assertEqual(m.shape, (7, 7))
        self.assertAlmostEqual(m[0, 0], 0.0)

    def test_custom_matrix_passes_through(self) -> None:
        custom = np.array([[0, 100], [100, 0]], dtype=float)
        m = resolve_distance_matrix(2, custom)
        np.testing.assert_array_equal(m, custom)

    def test_invalid_matrix_raises(self) -> None:
        bad = np.array([[0, 100], [100, 0], [50, 50]], dtype=float)
        with self.assertRaises(ValueError):
            resolve_distance_matrix(2, bad)


class ValidateDistanceMatrixTests(unittest.TestCase):
    """Tests for validate_distance_matrix."""

    def test_non_square_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_distance_matrix(np.ones((3, 4)), 3)

    def test_wrong_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_distance_matrix(np.zeros((3, 3)), 4)

    def test_negative_values_raise(self) -> None:
        m = np.array([[0, -1], [-1, 0]], dtype=float)
        with self.assertRaises(ValueError):
            validate_distance_matrix(m, 2)

    def test_nonzero_diagonal_raises(self) -> None:
        m = np.array([[1, 100], [100, 0]], dtype=float)
        with self.assertRaises(ValueError):
            validate_distance_matrix(m, 2)

    def test_valid_matrix_passes(self) -> None:
        m = np.array([[0, 50], [50, 0]], dtype=float)
        result = validate_distance_matrix(m, 2)
        np.testing.assert_array_equal(result, m)


# ── Dynamics unit tests ──────────────────────────────────────────────────


class ComputeFuelAndEmissionsTests(unittest.TestCase):
    """Direct tests for compute_fuel_and_emissions."""

    def setUp(self) -> None:
        self.cfg = get_default_config()

    def test_zero_speed_yields_zero(self) -> None:
        fuel, co2 = compute_fuel_and_emissions(0.0, self.cfg)
        self.assertAlmostEqual(fuel, 0.0)
        self.assertAlmostEqual(co2, 0.0)

    def test_cubic_scaling(self) -> None:
        fuel_1, _ = compute_fuel_and_emissions(1.0, self.cfg)
        fuel_2, _ = compute_fuel_and_emissions(2.0, self.cfg)
        self.assertAlmostEqual(fuel_2 / fuel_1, 8.0, places=5)

    def test_emission_factor_applied(self) -> None:
        fuel, co2 = compute_fuel_and_emissions(10.0, self.cfg)
        self.assertAlmostEqual(co2, fuel * self.cfg["emission_factor"])

    def test_hours_scaling(self) -> None:
        fuel_1, _ = compute_fuel_and_emissions(10.0, self.cfg, hours=1.0)
        fuel_3, _ = compute_fuel_and_emissions(10.0, self.cfg, hours=3.0)
        self.assertAlmostEqual(fuel_3 / fuel_1, 3.0, places=5)


class StepVesselsTests(unittest.TestCase):
    """Direct tests for step_vessels."""

    def setUp(self) -> None:
        self.cfg = get_default_config()
        self.distance_nm = resolve_distance_matrix(self.cfg["num_ports"])

    def test_docked_vessel_unchanged(self) -> None:
        v = VesselState(vessel_id=0, location=0, destination=1, speed=12.0, at_sea=False)
        stats = step_vessels([v], self.distance_nm, self.cfg)
        self.assertAlmostEqual(stats[0]["fuel_used"], 0.0)
        self.assertFalse(stats[0]["arrived"])
        self.assertFalse(v.at_sea)

    def test_in_transit_vessel_advances(self) -> None:
        v = VesselState(vessel_id=0, location=0, destination=1, speed=12.0,
                        at_sea=True, fuel=1000.0, position_nm=0.0)
        stats = step_vessels([v], self.distance_nm, self.cfg)
        self.assertGreater(stats[0]["fuel_used"], 0.0)
        self.assertGreater(v.position_nm, 0.0)

    def test_arrival_detection(self) -> None:
        dist = self.distance_nm[0, 1]
        v = VesselState(vessel_id=0, location=0, destination=1, speed=12.0,
                        at_sea=True, position_nm=dist - 1.0, fuel=1000.0)
        stats = step_vessels([v], self.distance_nm, self.cfg, dt_hours=1000.0)
        self.assertTrue(stats[0]["arrived"])
        self.assertFalse(v.at_sea)
        self.assertEqual(v.location, 1)
        self.assertAlmostEqual(v.position_nm, 0.0)


class StepPortsTests(unittest.TestCase):
    """Direct tests for step_ports."""

    def test_queue_admitted_to_berth(self) -> None:
        port = PortState(port_id=0, queue=3, docks=4, occupied=0)
        step_ports([port], service_rates=[2])
        self.assertEqual(port.queue, 1)
        self.assertEqual(port.occupied, 2)
        self.assertEqual(port.vessels_served, 2)

    def test_service_completion_frees_berth(self) -> None:
        port = PortState(port_id=0, queue=0, docks=2, occupied=1,
                         service_times=[0.5])
        step_ports([port], service_rates=[0], dt_hours=1.0)
        self.assertEqual(port.occupied, 0)

    def test_cannot_exceed_dock_capacity(self) -> None:
        port = PortState(port_id=0, queue=10, docks=2, occupied=2)
        step_ports([port], service_rates=[5])
        self.assertEqual(port.occupied, 2)
        self.assertEqual(port.queue, 10)

    def test_cumulative_wait_hours_accumulate(self) -> None:
        port = PortState(port_id=0, queue=4, docks=1, occupied=0)
        step_ports([port], service_rates=[0], dt_hours=2.0)
        self.assertAlmostEqual(port.cumulative_wait_hours, 8.0)


class DispatchVesselTests(unittest.TestCase):
    """Direct tests for dispatch_vessel."""

    def setUp(self) -> None:
        self.cfg = get_default_config()

    def test_dispatch_sets_at_sea(self) -> None:
        v = VesselState(vessel_id=0, location=0, destination=0, at_sea=False)
        dispatch_vessel(v, destination=2, speed=12.0, config=self.cfg)
        self.assertTrue(v.at_sea)
        self.assertEqual(v.destination, 2)
        self.assertAlmostEqual(v.position_nm, 0.0)

    def test_self_loop_noop(self) -> None:
        v = VesselState(vessel_id=0, location=3, destination=3, at_sea=False)
        dispatch_vessel(v, destination=3, speed=12.0, config=self.cfg)
        self.assertFalse(v.at_sea)

    def test_speed_clipped(self) -> None:
        v = VesselState(vessel_id=0, location=0, destination=0, at_sea=False)
        dispatch_vessel(v, destination=1, speed=999.0, config=self.cfg)
        self.assertAlmostEqual(v.speed, self.cfg["speed_max"])


# ── Env helper method tests ──────────────────────────────────────────────


class EnvHelperTests(unittest.TestCase):
    """Tests for MaritimeEnv helper methods."""

    def setUp(self) -> None:
        self.cfg: dict[str, Any] = {"num_ports": 3, "num_vessels": 4,
                                     "rollout_steps": 10, "num_coordinators": 2}
        self.env = MaritimeEnv(config=self.cfg, seed=42)
        self.env.reset()

    def test_build_assignments_covers_all_vessels(self) -> None:
        assignments = self.env._build_assignments()
        all_ids = [vid for ids in assignments.values() for vid in ids]
        self.assertEqual(sorted(all_ids), sorted(v.vessel_id for v in self.env.vessels))

    def test_peek_step_context_keys(self) -> None:
        ctx = self.env.peek_step_context()
        self.assertIn("assignments", ctx)
        self.assertIn("latest_directive_by_vessel", ctx)
        self.assertIn("pending_port_requests", ctx)

    def test_get_directive_for_vessel_returns_dict(self) -> None:
        assignments = self.env._build_assignments()
        directive = self.env.get_directive_for_vessel(
            vessel_id=0,
            assignments=assignments,
            latest_directive_by_vessel={},
        )
        self.assertIsInstance(directive, dict)

    def test_sample_stub_actions_structure(self) -> None:
        actions = self.env.sample_stub_actions()
        self.assertIn("coordinator", actions)
        self.assertIn("coordinators", actions)
        self.assertIn("vessels", actions)
        self.assertIn("ports", actions)
        self.assertEqual(len(actions["vessels"]), self.env.num_vessels)
        self.assertEqual(len(actions["ports"]), self.env.num_ports)

    def test_global_state_dim_matches(self) -> None:
        from hmarl_mvp.mappo import global_state_dim_from_config
        gs = self.env.get_global_state()
        expected_dim = global_state_dim_from_config(self.env.cfg)
        self.assertEqual(len(gs), expected_dim)


# ── Experiment sweep tests ───────────────────────────────────────────────


class PolicySweepTests(unittest.TestCase):
    """Smoke tests for experiment sweep functions."""

    def test_run_policy_sweep(self) -> None:
        results = run_policy_sweep(
            policies=["independent", "reactive"],
            steps=5,
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 10},
        )
        self.assertEqual(set(results.keys()), {"independent", "reactive"})
        for df in results.values():
            self.assertEqual(len(df), 5)

    def test_run_horizon_sweep(self) -> None:
        results = run_horizon_sweep(
            horizons=[4, 8],
            steps=5,
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 10},
        )
        self.assertEqual(set(results.keys()), {4, 8})

    def test_run_noise_sweep(self) -> None:
        results = run_noise_sweep(
            noise_levels=[0.0, 1.0],
            steps=5,
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 10},
        )
        self.assertEqual(set(results.keys()), {0.0, 1.0})

    def test_run_sharing_sweep(self) -> None:
        results = run_sharing_sweep(
            steps=5,
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 10},
        )
        self.assertIn("shared", results)
        self.assertIn("coordinator_only", results)

    def test_summarize_policy_results(self) -> None:
        results = run_policy_sweep(
            policies=["independent", "forecast"],
            steps=5,
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 10},
        )
        summary = summarize_policy_results(results)
        self.assertEqual(len(summary), 2)
        self.assertIn("avg_queue", summary.columns)

    def test_save_result_dict(self) -> None:
        results = run_policy_sweep(
            policies=["independent"],
            steps=3,
            config={"num_ports": 3, "num_vessels": 3, "rollout_steps": 10},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_result_dict(results, tmpdir, "test")
            files = os.listdir(tmpdir)
            self.assertTrue(any("test_independent" in f for f in files))


if __name__ == "__main__":
    unittest.main()
