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
    compute_coordinator_reward_breakdown,
    compute_coordinator_reward_step,
    compute_port_reward,
    compute_port_reward_breakdown,
    compute_vessel_reward_breakdown,
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
            self.vessel,
            self.cfg,
            fuel_used=1.0,
            co2_emitted=2.0,
            delay_hours=3.0,
            transit_hours=1.0,
        )
        expected = -(
            self.cfg["fuel_weight"] * 1.0
            + self.cfg["emission_weight"] * 2.0
            + self.cfg["delay_weight"] * 3.0
            + self.cfg["transit_time_weight"] * 1.0
        )
        self.assertAlmostEqual(reward, expected)

    def test_negative_deltas_are_clamped_to_zero(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel, self.cfg, fuel_used=-1.0, co2_emitted=-1.0, delay_hours=-1.0
        )
        self.assertEqual(reward, 0.0)

    def test_transit_time_penalty_applies_only_when_provided(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            delay_hours=0.0,
            transit_hours=1.5,
        )
        expected = -(self.cfg["transit_time_weight"] * 1.5)
        self.assertAlmostEqual(reward, expected)

    def test_arrival_bonus_offsets_vessel_cost(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel,
            self.cfg,
            fuel_used=1.0,
            co2_emitted=0.0,
            delay_hours=0.0,
            transit_hours=0.0,
            arrived=True,
        )
        expected = self.cfg["arrival_reward"] - self.cfg["fuel_weight"] * 1.0
        self.assertAlmostEqual(reward, expected)

    def test_schedule_delay_penalty_applies(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            delay_hours=0.0,
            schedule_delay_hours=1.25,
        )
        expected = -(self.cfg["schedule_delay_weight"] * 1.25)
        self.assertAlmostEqual(reward, expected)

    def test_on_time_arrival_bonus_applies(self) -> None:
        reward = compute_vessel_reward_step(
            self.vessel,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            delay_hours=0.0,
            arrived=True,
            arrived_on_time=True,
        )
        expected = self.cfg["arrival_reward"] + self.cfg["on_time_arrival_reward"]
        self.assertAlmostEqual(reward, expected)

    def test_vessel_reward_breakdown_sums_to_reward(self) -> None:
        parts = compute_vessel_reward_breakdown(
            self.vessel,
            self.cfg,
            fuel_used=1.0,
            co2_emitted=2.0,
            delay_hours=3.0,
            transit_hours=0.5,
            schedule_delay_hours=0.25,
            arrived=True,
            arrived_on_time=True,
        )
        reward = compute_vessel_reward_step(
            self.vessel,
            self.cfg,
            fuel_used=1.0,
            co2_emitted=2.0,
            delay_hours=3.0,
            transit_hours=0.5,
            schedule_delay_hours=0.25,
            arrived=True,
            arrived_on_time=True,
        )
        self.assertAlmostEqual(parts["total"], reward)


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

    def test_serving_vessels_gives_port_bonus(self) -> None:
        port = PortState(port_id=0, queue=0, docks=3, occupied=1)
        reward = compute_port_reward(port, self.cfg, served_vessels=2.0)
        expected = self.cfg["port_service_reward"] * 2.0 - self.cfg["dock_idle_weight"] * 2.0
        self.assertAlmostEqual(reward, expected)

    def test_accepting_requests_gives_port_bonus(self) -> None:
        port = PortState(port_id=0, queue=0, docks=3, occupied=1)
        reward = compute_port_reward(port, self.cfg, accepted_requests=2.0)
        expected = self.cfg["port_accept_reward"] * 2.0 - self.cfg["dock_idle_weight"] * 2.0
        self.assertAlmostEqual(reward, expected)

    def test_rejecting_requests_penalizes_port(self) -> None:
        port = PortState(port_id=0, queue=0, docks=3, occupied=1)
        reward = compute_port_reward(port, self.cfg, rejected_requests=2.0)
        expected = -(self.cfg["dock_idle_weight"] * 2.0 + self.cfg["port_reject_penalty"] * 2.0)
        self.assertAlmostEqual(reward, expected)

    def test_port_reward_breakdown_sums_to_reward(self) -> None:
        port = PortState(port_id=0, queue=2, docks=3, occupied=1)
        parts = compute_port_reward_breakdown(
            port,
            self.cfg,
            served_vessels=1.0,
            accepted_requests=2.0,
            rejected_requests=1.0,
        )
        reward = compute_port_reward(
            port,
            self.cfg,
            served_vessels=1.0,
            accepted_requests=2.0,
            rejected_requests=1.0,
        )
        self.assertAlmostEqual(parts["total"], reward)


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
        avg_idle_docks = 1.5
        avg_occupied_docks = 1.5
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = (
            self.cfg["coordinator_utilization_reward"] * avg_occupied_docks
            - (
                self.cfg["coordinator_queue_weight"] * avg_queue
                + self.cfg["coordinator_idle_dock_weight"] * avg_idle_docks
                + queue_imbalance
            )
        )
        self.assertAlmostEqual(reward, expected)

    def test_emission_lambda_amplifies_co2(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports, self.cfg, fuel_used=0.0, co2_emitted=10.0
        )
        avg_queue = 3.0
        avg_idle_docks = 1.5
        avg_occupied_docks = 1.5
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = -(
            self.cfg["coordinator_queue_weight"] * avg_queue
            + self.cfg["coordinator_idle_dock_weight"] * avg_idle_docks
            + self.cfg["coordinator_emission_weight"] * 10.0
            + queue_imbalance
        ) + self.cfg["coordinator_utilization_reward"] * avg_occupied_docks
        self.assertAlmostEqual(reward, expected)

    def test_delay_penalty_applies_to_coordinator(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            delay_hours=2.0,
        )
        avg_queue = 3.0
        avg_idle_docks = 1.5
        avg_occupied_docks = 1.5
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = -(
            self.cfg["coordinator_queue_weight"] * avg_queue
            + self.cfg["coordinator_idle_dock_weight"] * avg_idle_docks
            + self.cfg["coordinator_delay_weight"] * 2.0
            + queue_imbalance
        ) + self.cfg["coordinator_utilization_reward"] * avg_occupied_docks
        self.assertAlmostEqual(reward, expected)

    def test_schedule_delay_penalty_applies_to_coordinator(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            schedule_delay_hours=1.5,
        )
        avg_queue = 3.0
        avg_idle_docks = 1.5
        avg_occupied_docks = 1.5
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = -(
            self.cfg["coordinator_queue_weight"] * avg_queue
            + self.cfg["coordinator_idle_dock_weight"] * avg_idle_docks
            + self.cfg["coordinator_schedule_delay_weight"] * 1.5
            + queue_imbalance
        ) + self.cfg["coordinator_utilization_reward"] * avg_occupied_docks
        self.assertAlmostEqual(reward, expected)

    def test_service_bonus_offsets_coordinator_cost(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports,
            self.cfg,
            fuel_used=1.0,
            co2_emitted=0.0,
            delay_hours=0.0,
            served_vessels=3.0,
        )
        avg_queue = 3.0
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = (
            self.cfg["coordinator_service_reward"] * 3.0
            + self.cfg["coordinator_utilization_reward"] * 1.5
            - (
                self.cfg["coordinator_fuel_weight"] * 1.0
                + self.cfg["coordinator_queue_weight"] * avg_queue
                + self.cfg["coordinator_idle_dock_weight"] * 1.5
                + queue_imbalance
            )
        )
        self.assertAlmostEqual(reward, expected)

    def test_accept_bonus_offsets_coordinator_cost(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            accepted_requests=2.0,
        )
        avg_queue = 3.0
        avg_idle_docks = 1.5
        avg_occupied_docks = 1.5
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = (
            self.cfg["coordinator_accept_reward"] * 2.0
            + self.cfg["coordinator_utilization_reward"] * avg_occupied_docks
            - (
                self.cfg["coordinator_queue_weight"] * avg_queue
                + self.cfg["coordinator_idle_dock_weight"] * avg_idle_docks
                + queue_imbalance
            )
        )
        self.assertAlmostEqual(reward, expected)

    def test_reject_penalty_applies_to_coordinator(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            rejected_requests=2.0,
        )
        avg_queue = 3.0
        avg_idle_docks = 1.5
        avg_occupied_docks = 1.5
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = -(
            self.cfg["coordinator_queue_weight"] * avg_queue
            + self.cfg["coordinator_idle_dock_weight"] * avg_idle_docks
            + self.cfg["coordinator_reject_penalty"] * 2.0
            + queue_imbalance
        ) + self.cfg["coordinator_utilization_reward"] * avg_occupied_docks
        self.assertAlmostEqual(reward, expected)

    def test_idle_dock_penalty_applies_to_coordinator(self) -> None:
        reward = compute_coordinator_reward_step(
            self.ports,
            self.cfg,
            fuel_used=0.0,
            co2_emitted=0.0,
            delay_hours=0.0,
            served_vessels=0.0,
        )
        avg_queue = 3.0
        avg_idle_docks = 1.5
        avg_occupied_docks = 1.5
        queue_imbalance = self.cfg["coordinator_queue_imbalance_weight"] * float(
            __import__("numpy").std([2.0, 4.0])
        )
        expected = (
            self.cfg["coordinator_utilization_reward"] * avg_occupied_docks
            - (
                self.cfg["coordinator_queue_weight"] * avg_queue
                + self.cfg["coordinator_idle_dock_weight"] * avg_idle_docks
                + queue_imbalance
            )
        )
        self.assertAlmostEqual(reward, expected)

    def test_empty_ports_list(self) -> None:
        reward = compute_coordinator_reward_step(
            [], self.cfg, fuel_used=1.0, co2_emitted=0.0
        )
        self.assertAlmostEqual(reward, -self.cfg["coordinator_fuel_weight"])

    def test_coordinator_reward_breakdown_sums_to_reward(self) -> None:
        parts = compute_coordinator_reward_breakdown(
            self.ports,
            self.cfg,
            fuel_used=1.5,
            co2_emitted=4.0,
            delay_hours=2.0,
            schedule_delay_hours=0.5,
            served_vessels=2.0,
            accepted_requests=3.0,
            rejected_requests=1.0,
        )
        reward = compute_coordinator_reward_step(
            self.ports,
            self.cfg,
            fuel_used=1.5,
            co2_emitted=4.0,
            delay_hours=2.0,
            schedule_delay_hours=0.5,
            served_vessels=2.0,
            accepted_requests=3.0,
            rejected_requests=1.0,
        )
        self.assertAlmostEqual(parts["total"], reward)

    def test_coordinator_queue_imbalance_penalty(self) -> None:
        """Queue imbalance penalty penalises uneven distribution across ports."""
        from hmarl_mvp.state import PortState

        cfg = dict(self.cfg)
        cfg["coordinator_queue_imbalance_weight"] = 1.0
        # Balanced: all queues = 2 → std = 0
        balanced = [PortState(port_id=i, docks=3, queue=2, occupied=1) for i in range(3)]
        parts_balanced = compute_coordinator_reward_breakdown(
            balanced, cfg, fuel_used=0, co2_emitted=0,
        )
        self.assertAlmostEqual(parts_balanced["queue_imbalance_penalty"], 0.0)

        # Imbalanced: queues = [0, 0, 6] → std ≈ 2.83
        imbalanced = [
            PortState(port_id=0, docks=3, queue=0, occupied=0),
            PortState(port_id=1, docks=3, queue=0, occupied=0),
            PortState(port_id=2, docks=3, queue=6, occupied=3),
        ]
        parts_imb = compute_coordinator_reward_breakdown(
            imbalanced, cfg, fuel_used=0, co2_emitted=0,
        )
        self.assertGreater(parts_imb["queue_imbalance_penalty"], 2.0)
        # Imbalanced total should be worse (more negative)
        self.assertLess(parts_imb["total"], parts_balanced["total"])

    def test_coordinator_queue_imbalance_zero_weight(self) -> None:
        """Penalty is zero when weight is zero."""
        cfg = dict(self.cfg)
        cfg["coordinator_queue_imbalance_weight"] = 0.0
        from hmarl_mvp.state import PortState
        ports = [
            PortState(port_id=0, docks=3, queue=0, occupied=0),
            PortState(port_id=1, docks=3, queue=5, occupied=3),
        ]
        parts = compute_coordinator_reward_breakdown(ports, cfg, fuel_used=0, co2_emitted=0)
        self.assertAlmostEqual(parts["queue_imbalance_penalty"], 0.0)


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

    def test_cumulative_fuel_used_survives_refuel(self) -> None:
        vessel = VesselState(
            vessel_id=0,
            location=0,
            destination=1,
            fuel=100.0,
            initial_fuel=100.0,
            cumulative_fuel_used=35.0,
        )
        m = compute_vessel_metrics([vessel])
        self.assertAlmostEqual(m["total_fuel_used"], 35.0)

    def test_stalled_vessels_are_counted(self) -> None:
        vessels = [
            VesselState(vessel_id=0, location=0, destination=1, stalled=True),
            VesselState(vessel_id=1, location=1, destination=0, stalled=False),
        ]
        m = compute_vessel_metrics(vessels)
        self.assertAlmostEqual(m["stalled_vessels"], 1.0)
        self.assertAlmostEqual(m["stalled_rate"], 0.5)

    def test_schedule_metrics_override_legacy_on_time_rate(self) -> None:
        vessels = [
            VesselState(
                vessel_id=0,
                location=0,
                destination=1,
                completed_arrivals=2,
                completed_scheduled_arrivals=2,
                on_time_arrivals=1,
                schedule_delay_hours=3.0,
                delay_hours=10.0,
            ),
            VesselState(
                vessel_id=1,
                location=1,
                destination=0,
                completed_arrivals=1,
                completed_scheduled_arrivals=1,
                on_time_arrivals=1,
                schedule_delay_hours=1.0,
                delay_hours=10.0,
            ),
        ]
        m = compute_vessel_metrics(vessels)
        self.assertAlmostEqual(m["completed_arrivals"], 3.0)
        self.assertAlmostEqual(m["scheduled_arrivals"], 3.0)
        self.assertAlmostEqual(m["on_time_arrivals"], 2.0)
        self.assertAlmostEqual(m["on_time_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(m["avg_schedule_delay_hours"], 4.0 / 3.0)


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
                        fuel=100.0, initial_fuel=100.0, cumulative_fuel_used=10.0,
                        emissions=10.0, delay_hours=2.0),
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
        vessel = VesselState(vessel_id=0, location=0, destination=0, at_sea=False, stalled=True)
        dispatch_vessel(
            vessel,
            destination=1,
            speed=cfg["nominal_speed"],
            config=cfg,
            requested_arrival_time=9.5,
        )
        self.assertTrue(vessel.at_sea)
        self.assertFalse(vessel.stalled)
        self.assertEqual(vessel.destination, 1)
        self.assertAlmostEqual(vessel.requested_arrival_time, 9.5)


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
