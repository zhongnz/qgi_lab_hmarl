"""Tests for new agent/policy/forecaster class interfaces."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.agents import FleetCoordinatorAgent, PortAgent, VesselAgent
from hmarl_mvp.config import get_default_config
from hmarl_mvp.forecasts import MediumTermForecaster, OracleForecaster, ShortTermForecaster
from hmarl_mvp.policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from hmarl_mvp.state import PortState, VesselState, initialize_ports, make_rng


class AgentPolicyForecasterTests(unittest.TestCase):
    def test_vessel_agent_observation_and_action_clip(self) -> None:
        cfg = get_default_config(short_horizon_hours=6, speed_min=8.0, speed_max=18.0)
        state = VesselState(vessel_id=1, location=0, destination=2, speed=12.0)
        vessel = VesselAgent(state=state, config=cfg)

        obs = vessel.get_obs(
            short_forecast_row=np.ones(cfg["short_horizon_hours"]),
            directive={"dest_port": 2, "departure_window_hours": 12, "emission_budget": 42.0},
        )
        self.assertEqual(len(obs), 4 + cfg["short_horizon_hours"] + 3)

        action = vessel.apply_action({"target_speed": 100.0, "request_arrival_slot": True})
        self.assertEqual(action["target_speed"], cfg["speed_max"])
        self.assertTrue(action["request_arrival_slot"])

    def test_policy_classes_return_expected_keys(self) -> None:
        cfg = get_default_config(num_ports=5, short_horizon_hours=4)
        coordinator_agent = FleetCoordinatorAgent(cfg, coordinator_id=0)
        vessel_agent = VesselAgent(
            VesselState(vessel_id=0, location=0, destination=1, speed=cfg["nominal_speed"]),
            cfg,
        )
        port_agent = PortAgent(PortState(port_id=0, queue=2, docks=3, occupied=1), cfg)

        medium = np.zeros((cfg["num_ports"], cfg["medium_horizon_days"]), dtype=float)
        short = np.zeros((cfg["num_ports"], cfg["short_horizon_hours"]), dtype=float)
        ports = [port_agent.state]
        vessels = [vessel_agent.state]
        rng = make_rng(42)

        coord_action = FleetCoordinatorPolicy(cfg, mode="forecast").act(
            coordinator_agent,
            medium_forecast=medium,
            vessels=vessels,
            ports=ports,
            rng=rng,
        )
        self.assertIn("dest_port", coord_action)

        vessel_action = VesselPolicy(cfg, mode="forecast").act(
            vessel_agent,
            short_forecast=short,
            directive=coord_action,
        )
        self.assertIn("target_speed", vessel_action)

        port_action = PortPolicy(cfg, mode="forecast").act(
            port_agent,
            incoming_requests=3,
            short_forecast_row=short[0],
        )
        self.assertIn("service_rate", port_action)

    def test_forecaster_classes_shape(self) -> None:
        cfg = get_default_config(num_ports=5, medium_horizon_days=3, short_horizon_hours=6)
        rng = make_rng(42)
        ports = initialize_ports(
            num_ports=cfg["num_ports"],
            docks_per_port=cfg["docks_per_port"],
            rng=rng,
        )

        medium = MediumTermForecaster(cfg["medium_horizon_days"]).predict(cfg["num_ports"], rng)
        short = ShortTermForecaster(cfg["short_horizon_hours"]).predict(cfg["num_ports"], rng)
        self.assertEqual(medium.shape, (cfg["num_ports"], cfg["medium_horizon_days"]))
        self.assertEqual(short.shape, (cfg["num_ports"], cfg["short_horizon_hours"]))

        oracle_medium, oracle_short = OracleForecaster(
            medium_horizon_days=cfg["medium_horizon_days"],
            short_horizon_hours=cfg["short_horizon_hours"],
        ).predict(ports)
        self.assertEqual(oracle_medium.shape, medium.shape)
        self.assertEqual(oracle_short.shape, short.shape)


if __name__ == "__main__":
    unittest.main()

