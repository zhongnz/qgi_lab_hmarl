"""Correctness tests for reward, dynamics, and topology invariants."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.agents import assign_vessels_to_coordinators
from hmarl_mvp.config import get_default_config
from hmarl_mvp.dynamics import step_ports, step_vessels
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.state import PortState, VesselState


class ModelCorrectnessTests(unittest.TestCase):
    def test_docked_vessel_has_zero_step_reward_when_no_delay(self) -> None:
        cfg = get_default_config(
            num_ports=2,
            num_vessels=1,
            rollout_steps=2,
            message_latency_steps=1,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()
        vessel = env.vessels[0]
        vessel.at_sea = False
        vessel.location = 0
        vessel.destination = 0
        vessel.speed = cfg["nominal_speed"]
        vessel.delay_hours = 0.0

        actions = {
            "coordinators": [{"dest_port": 0, "departure_window_hours": 12, "emission_budget": 50.0}],
            "vessels": [{"target_speed": cfg["nominal_speed"], "request_arrival_slot": False}],
            "ports": [{"service_rate": 0, "accept_requests": 0} for _ in range(cfg["num_ports"])],
        }
        _, rewards, _, _ = env.step(actions)
        self.assertEqual(rewards["vessels"][0], 0.0)

    def test_port_occupancy_releases_after_service_time(self) -> None:
        port = PortState(
            port_id=0,
            queue=0,
            docks=2,
            occupied=2,
            service_times=[1.0, 2.0],
        )
        step_ports([port], [0], dt_hours=1.0, service_time_hours=3.0)
        self.assertEqual(port.occupied, 1)
        self.assertEqual(port.service_times, [1.0])
        step_ports([port], [0], dt_hours=1.0, service_time_hours=3.0)
        self.assertEqual(port.occupied, 0)
        self.assertEqual(port.service_times, [])

    def test_waiting_delay_scales_with_dt_hours(self) -> None:
        cfg = get_default_config(
            num_ports=2,
            num_vessels=1,
            rollout_steps=3,
            message_latency_steps=2,
            dt_hours=0.5,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()

        actions = {
            "coordinators": [{"dest_port": 1, "departure_window_hours": 0, "emission_budget": 50.0}],
            "vessels": [{"target_speed": cfg["nominal_speed"], "request_arrival_slot": True}],
            "ports": [{"service_rate": 0, "accept_requests": 0} for _ in range(cfg["num_ports"])],
        }
        _, _, _, info = env.step(actions)
        self.assertAlmostEqual(env.vessels[0].delay_hours, 0.5, places=6)
        self.assertAlmostEqual(float(info["step_delay_hours"]), 0.5, places=6)

    def test_in_transit_reward_penalizes_travel_time(self) -> None:
        cfg = get_default_config(
            num_ports=2,
            num_vessels=1,
            rollout_steps=2,
            fuel_weight=0.0,
            emission_weight=0.0,
            transit_time_weight=5.0,
            dt_hours=1.0,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()
        vessel = env.vessels[0]
        vessel.location = 0
        vessel.destination = 1
        vessel.speed = cfg["nominal_speed"]
        vessel.at_sea = True
        vessel.position_nm = 0.0

        actions = {
            "coordinators": [{"dest_port": 1, "departure_window_hours": 0, "emission_budget": 50.0}],
            "vessels": [{"target_speed": cfg["nominal_speed"], "request_arrival_slot": False}],
            "ports": [{"service_rate": 0, "accept_requests": 0} for _ in range(cfg["num_ports"])],
        }
        _, rewards, _, info = env.step(actions)
        self.assertAlmostEqual(rewards["vessels"][0], -5.0, places=6)
        self.assertAlmostEqual(float(info["step_delay_hours"]), 0.0, places=6)

    def test_out_of_fuel_vessel_stalls_mid_leg(self) -> None:
        cfg = get_default_config(num_ports=2, fuel_rate_coeff=0.002, dt_hours=1.0)
        distance_nm = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=float)
        hourly_fuel = cfg["fuel_rate_coeff"] * (cfg["nominal_speed"] ** 3)
        vessel = VesselState(
            vessel_id=0,
            location=0,
            destination=1,
            speed=cfg["nominal_speed"],
            fuel=hourly_fuel / 2.0,
            at_sea=True,
            position_nm=0.0,
        )

        stats = step_vessels([vessel], distance_nm, cfg, dt_hours=1.0)
        self.assertTrue(vessel.at_sea)
        self.assertTrue(vessel.stalled)
        self.assertAlmostEqual(vessel.fuel, 0.0, places=6)
        self.assertAlmostEqual(vessel.position_nm, cfg["nominal_speed"] * 0.5, places=6)
        self.assertAlmostEqual(float(stats[0]["travel_hours"]), 0.5, places=6)
        self.assertAlmostEqual(float(stats[0]["stall_hours"]), 0.5, places=6)
        self.assertFalse(bool(stats[0]["arrived"]))

        stats = step_vessels([vessel], distance_nm, cfg, dt_hours=1.0)
        self.assertAlmostEqual(vessel.position_nm, cfg["nominal_speed"] * 0.5, places=6)
        self.assertAlmostEqual(float(stats[0]["travel_hours"]), 0.0, places=6)
        self.assertAlmostEqual(float(stats[0]["stall_hours"]), 1.0, places=6)

    def test_stalled_vessel_delay_accrues_in_env_step(self) -> None:
        cfg = get_default_config(
            num_ports=2,
            num_vessels=1,
            rollout_steps=2,
            fuel_weight=0.0,
            emission_weight=0.0,
            transit_time_weight=0.0,
            delay_weight=2.0,
            dt_hours=1.0,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()
        vessel = env.vessels[0]
        vessel.location = 0
        vessel.destination = 1
        vessel.speed = cfg["nominal_speed"]
        vessel.fuel = 0.0
        vessel.at_sea = True
        vessel.position_nm = 0.0
        vessel.delay_hours = 0.0
        vessel.stalled = False

        actions = {
            "coordinators": [{"dest_port": 1, "departure_window_hours": 0, "emission_budget": 50.0}],
            "vessels": [{"target_speed": cfg["nominal_speed"], "request_arrival_slot": False}],
            "ports": [{"service_rate": 0, "accept_requests": 0} for _ in range(cfg["num_ports"])],
        }
        _, rewards, _, info = env.step(actions)
        self.assertTrue(env.vessels[0].stalled)
        self.assertAlmostEqual(env.vessels[0].position_nm, 0.0, places=6)
        self.assertAlmostEqual(env.vessels[0].delay_hours, 1.0, places=6)
        self.assertAlmostEqual(float(info["step_stall_hours"]), 1.0, places=6)
        self.assertAlmostEqual(float(info["step_delay_hours"]), 1.0, places=6)
        self.assertAlmostEqual(rewards["vessels"][0], -2.0, places=6)

    def test_full_port_can_accept_advance_reservation(self) -> None:
        cfg = get_default_config(
            num_ports=2,
            num_vessels=1,
            rollout_steps=2,
            message_latency_steps=1,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()
        env.ports[1].occupied = env.ports[1].docks
        env.ports[1].queue = 0
        env.bus.enqueue_arrival_request(0, 0, 1, 0.0)

        actions = {
            "coordinators": [{"dest_port": 1, "departure_window_hours": 0, "emission_budget": 50.0}],
            "vessels": [{"target_speed": cfg["nominal_speed"], "request_arrival_slot": False}],
            "ports": [{"service_rate": 0, "accept_requests": 0}, {"service_rate": 0, "accept_requests": 1}],
        }
        _, _, _, info = env.step(actions)
        response = env.bus.deliver_due(env.t)

        self.assertEqual(float(info["requests_accepted"]), 1.0)
        self.assertEqual(float(info["requests_rejected"]), 0.0)
        self.assertEqual(response[0]["accepted"], True)
        self.assertEqual(response[0]["dest_port"], 1)

    def test_non_default_port_topology_builds_valid_distance_matrix(self) -> None:
        cfg = get_default_config(num_ports=6, num_vessels=2, rollout_steps=2)
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()
        self.assertEqual(env.distance_nm.shape, (6, 6))

    def test_invalid_distance_matrix_shape_raises(self) -> None:
        cfg = get_default_config(num_ports=4)
        with self.assertRaises(ValueError):
            MaritimeEnv(
                config=cfg,
                seed=42,
                distance_nm=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            )

    def test_in_transit_assignment_uses_vessel_id_partition(self) -> None:
        vessels = [
            VesselState(vessel_id=0, location=4, destination=1, at_sea=True),
            VesselState(vessel_id=1, location=4, destination=1, at_sea=True),
            VesselState(vessel_id=2, location=3, destination=1, at_sea=True),
        ]
        groups = assign_vessels_to_coordinators(vessels, num_coordinators=2)
        self.assertEqual(groups[0], [0, 2])
        self.assertEqual(groups[1], [1])


if __name__ == "__main__":
    unittest.main()
