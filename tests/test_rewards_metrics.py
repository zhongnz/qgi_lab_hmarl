"""Direct unit tests for reward functions and metric helpers."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.dynamics import dispatch_vessel
from hmarl_mvp.metrics import (
    compute_economic_metrics,
    compute_economic_step_deltas,
    compute_port_metrics,
    compute_vessel_metrics,
    forecast_mae,
    forecast_rmse,
)
from hmarl_mvp.rewards import (
    compute_coordinator_reward_step,
    compute_port_reward,
    compute_vessel_reward_step,
)
from hmarl_mvp.state import PortState, VesselState


class VesselRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = get_default_config()
        self.vessel = VesselState(vessel_id=0, location=0, destination=1)

    def test_zero_deltas_yield_zero_reward(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel, self.cfg, fuel_used=0.0, co2_emitted=0.0, delay_hours=0.0
        )
        self.assertEqual(reward, 0.0)

    def test_positive_fuel_gives_negative_reward(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel, self.cfg, fuel_used=5.0, co2_emitted=0.0, delay_hours=0.0
        )
        self.assertLess(reward, 0.0)
        expected = -(self.cfg["fuel_weight"] * 5.0)
        self.assertAlmostEqual(reward, expected)

    def test_positive_delay_gives_negative_reward(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel, self.cfg, fuel_used=0.0, co2_emitted=0.0, delay_hours=2.0
        )
        expected = -(self.cfg["delay_weight"] * 2.0)
        self.assertAlmostEqual(reward, expected)

    def test_positive_emissions_gives_negative_reward(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel, self.cfg, fuel_used=0.0, co2_emitted=3.0, delay_hours=0.0
        )
        expected = -(self.cfg["emission_weight"] * 3.0)
        self.assertAlmostEqual(reward, expected)

    def test_combined_costs_are_additive(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel, self.cfg, fuel_used=1.0, co2_emitted=2.0, delay_hours=3.0
        )
        expected = -(
            self.cfg["fuel_weight"] * 1.0
            + self.cfg["emission_weight"] * 2.0
            + self.cfg["delay_weight"] * 3.0
        )
        self.assertAlmostEqual(reward, expected)

    def test_negative_deltas_are_clamped_to_zero(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel, self.cfg, fuel_used=-1.0, co2_emitted=-1.0, delay_hours=-1.0
        )
        self.assertEqual(reward, 0.0)


class PortRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = get_default_config()

    def test_full_port_no_queue_gives_zero_idle_penalty(self) -> None:
        port = PortState(port_id=0, queue=0, docks=3, occupied=3)
        reward = compute_port_reward(port, self.cfg)
        self.assertEqual(reward, 0.0)

    def test_empty_port_penalizes_idle_docks(self) -> None:
        port = PortState(port_id=0, queue=0, docks=3, occupied=0)
        reward = compute_port_reward(port, self.cfg)
        expected = -(self.cfg["dock_idle_weight"] * 3)
        self.assertAlmostEqual(reward, expected)

    def test_queue_increases_penalty(self) -> None:
        port = PortState(port_id=0, queue=5, docks=3, occupied=3)
        reward = compute_port_reward(port, self.cfg)
        self.assertAlmostEqual(reward, -5.0)


class CoordinatorRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = get_default_config()
        self.ports = [
            PortState(port_id=0, queue=2, docks=3, occupied=1),
            PortState(port_id=1, queue=4, docks=3, occupied=2),
        ]

    def test_zero_deltas_still_penalize_queue(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports, self.cfg, fuel_used=0.0, co2_emitted=0.0
        )
        avg_queue = 3.0  # (2+4)/2
        self.assertAlmostEqual(reward, -avg_queue)

    def test_emission_lambda_amplifies_co2(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports, self.cfg, fuel_used=0.0, co2_emitted=10.0
        )
        avg_queue = 3.0
        expected = -(avg_queue + self.cfg["emission_lambda"] * 10.0)
        self.assertAlmostEqual(reward, expected)

    def test_empty_ports_list(self) -> None:
        reward = compute_coordinator_reward_step(
            [], self.cfg, fuel_used=1.0, co2_emitted=0.0
        )
        self.assertAlmostEqual(reward, -1.0)


class ForecastMetricTests(unittest.TestCase):
    def test_mae_perfect(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(forecast_mae(a, a), 0.0)

    def test_mae_known(self) -> None:
        pred = np.array([1.0, 3.0])
        actual = np.array([2.0, 5.0])
        self.assertAlmostEqual(forecast_mae(pred, actual), 1.5)

    def test_rmse_perfect(self) -> None:
        a = np.array([1.0, 2.0])
        self.assertAlmostEqual(forecast_rmse(a, a), 0.0)

    def test_rmse_known(self) -> None:
        pred = np.array([1.0, 3.0])
        actual = np.array([2.0, 5.0])
        # errors: 1, 2 → mse = (1+4)/2 = 2.5 → rmse = sqrt(2.5)
        self.assertAlmostEqual(forecast_rmse(pred, actual), np.sqrt(2.5))


class VesselMetricTests(unittest.TestCase):
    def test_empty_vessels(self) -> None:
        m = compute_vessel_metrics([])
        self.assertEqual(m["avg_speed"], 0.0)
        self.assertEqual(m["total_emissions_co2"], 0.0)

    def test_basic_aggregation(self) -> None:
        vessels = [
            VesselState(vessel_id=0, location=0, destination=1, speed=10.0, fuel=80.0,
                        initial_fuel=100.0, emissions=5.0, delay_hours=1.0),
            VesselState(vessel_id=1, location=1, destination=0, speed=14.0, fuel=60.0,
                        initial_fuel=100.0, emissions=15.0, delay_hours=3.0),
        ]
        m = compute_vessel_metrics(vessels)
        self.assertAlmostEqual(m["avg_speed"], 12.0)
        self.assertAlmostEqual(m["total_fuel_used"], 60.0)  # 20+40
        self.assertAlmostEqual(m["total_emissions_co2"], 20.0)
        self.assertAlmostEqual(m["avg_delay_hours"], 2.0)
        self.assertAlmostEqual(m["on_time_rate"], 0.5)  # vessel 0 < 2h


class PortMetricTests(unittest.TestCase):
    def test_empty_ports(self) -> None:
        m = compute_port_metrics([])
        self.assertEqual(m["avg_queue"], 0.0)

    def test_basic_aggregation(self) -> None:
        ports = [
            PortState(port_id=0, queue=2, docks=4, occupied=2,
                      cumulative_wait_hours=10.0, vessels_served=5),
            PortState(port_id=1, queue=6, docks=4, occupied=4,
                      cumulative_wait_hours=20.0, vessels_served=5),
        ]
        m = compute_port_metrics(ports)
        self.assertAlmostEqual(m["avg_queue"], 4.0)
        self.assertAlmostEqual(m["dock_utilization"], 0.75)
        self.assertAlmostEqual(m["total_wait_hours"], 30.0)
        self.assertEqual(m["total_vessels_served"], 10)
        self.assertAlmostEqual(m["avg_wait_per_vessel"], 3.0)


class EconomicMetricTests(unittest.TestCase):
    def test_empty_vessels(self) -> None:
        cfg = get_default_config()
        m = compute_economic_metrics([], cfg)
        self.assertEqual(m["total_ops_cost_usd"], 0.0)

    def test_cost_components(self) -> None:
        cfg = get_default_config(
            fuel_price_per_ton=600.0,
            delay_penalty_per_hour=5000.0,
            carbon_price_per_ton=90.0,
            cargo_value_per_vessel=1_000_000.0,
        )
        vessels = [
            VesselState(vessel_id=0, location=0, destination=1,
                        fuel=90.0, initial_fuel=100.0, emissions=10.0, delay_hours=2.0),
        ]
        m = compute_economic_metrics(vessels, cfg)
        self.assertAlmostEqual(m["fuel_cost_usd"], 10.0 * 600.0)
        self.assertAlmostEqual(m["delay_cost_usd"], 2.0 * 5000.0)
        self.assertAlmostEqual(m["carbon_cost_usd"], 10.0 * 90.0)
        expected_total = 6000.0 + 10000.0 + 900.0
        self.assertAlmostEqual(m["total_ops_cost_usd"], expected_total)
        self.assertAlmostEqual(m["price_per_vessel_usd"], expected_total)
        self.assertAlmostEqual(m["cost_reliability"], 1.0 - expected_total / 1_000_000.0)


class DispatchSelfLoopTests(unittest.TestCase):
    """Verify that dispatching a vessel to its current location is a no-op."""

    def test_dispatch_to_same_port_is_noop(self) -> None:
        cfg = get_default_config()
        vessel = VesselState(vessel_id=0, location=2, destination=3, at_sea=False)
        dispatch_vessel(vessel, destination=2, speed=cfg["nominal_speed"], config=cfg)
        self.assertFalse(vessel.at_sea)
        self.assertEqual(vessel.destination, 3)  # unchanged

    def test_dispatch_to_different_port_works(self) -> None:
        cfg = get_default_config()
        vessel = VesselState(vessel_id=0, location=0, destination=0, at_sea=False)
        dispatch_vessel(vessel, destination=1, speed=cfg["nominal_speed"], config=cfg)
        self.assertTrue(vessel.at_sea)
        self.assertEqual(vessel.destination, 1)


class EconomicStepDeltaTests(unittest.TestCase):
    def test_zero_deltas(self) -> None:
        cfg = get_default_config()
        m = compute_economic_step_deltas(0.0, 0.0, 0.0, cfg)
        self.assertEqual(m["step_fuel_cost_usd"], 0.0)
        self.assertEqual(m["step_delay_cost_usd"], 0.0)
        self.assertEqual(m["step_carbon_cost_usd"], 0.0)
        self.assertEqual(m["step_total_ops_cost_usd"], 0.0)

    def test_known_deltas(self) -> None:
        cfg = get_default_config(
            fuel_price_per_ton=600.0,
            delay_penalty_per_hour=5000.0,
            carbon_price_per_ton=90.0,
        )
        m = compute_economic_step_deltas(
            step_fuel_used=2.0, step_co2_emitted=5.0, step_delay_hours=1.0, config=cfg
        )
        self.assertAlmostEqual(m["step_fuel_cost_usd"], 1200.0)
        self.assertAlmostEqual(m["step_delay_cost_usd"], 5000.0)
        self.assertAlmostEqual(m["step_carbon_cost_usd"], 450.0)
        self.assertAlmostEqual(m["step_total_ops_cost_usd"], 6650.0)


if __name__ == "__main__":
    unittest.main()
