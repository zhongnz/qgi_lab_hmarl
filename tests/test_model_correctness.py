"""Correctness tests for reward, dynamics, and topology invariants."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.dynamics import step_ports
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.multi_coordinator import assign_vessels_to_coordinators
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
