"""Tests for proposal-alignment fixes: dock availability obs, port reward,
coordinator metrics, coordination metrics, trip tracking, and cadence defaults."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.agents import VesselAgent
from hmarl_mvp.config import HMARLConfig, get_default_config
from hmarl_mvp.dynamics import dispatch_vessel
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.metrics import (
    compute_coordination_metrics,
    compute_coordinator_metrics,
)
from hmarl_mvp.rewards import compute_port_reward
from hmarl_mvp.state import PortState, VesselState

# -----------------------------------------------------------------------
# Cadence defaults match proposal §4.1
# -----------------------------------------------------------------------


class TestCadenceDefaults(unittest.TestCase):
    """Verify decision cadence defaults align with proposal ranges."""

    def test_port_cadence_default_within_proposal_range(self) -> None:
        """Proposal says port decisions every 2-6h; default should be >= 2."""
        cfg = HMARLConfig()
        self.assertGreaterEqual(cfg.port_decision_interval_steps, 2)
        self.assertLessEqual(cfg.port_decision_interval_steps, 6)

    def test_vessel_cadence_default_within_proposal_range(self) -> None:
        """Proposal says vessel decisions every 1-4h; default 1 is the minimum."""
        cfg = HMARLConfig()
        self.assertGreaterEqual(cfg.vessel_decision_interval_steps, 1)
        self.assertLessEqual(cfg.vessel_decision_interval_steps, 4)

    def test_coordinator_cadence_default_within_proposal_range(self) -> None:
        """Proposal says coordinator decisions every 12-24h."""
        cfg = HMARLConfig()
        self.assertGreaterEqual(cfg.coord_decision_interval_steps, 12)
        self.assertLessEqual(cfg.coord_decision_interval_steps, 24)


# -----------------------------------------------------------------------
# Dock availability in vessel observations (proposal §4.1)
# -----------------------------------------------------------------------


class TestVesselDockAvailabilityObs(unittest.TestCase):
    """Vessel observation includes dock availability from destination port."""

    def test_obs_length_includes_dock_avail(self) -> None:
        cfg = get_default_config(short_horizon_hours=6)
        state = VesselState(vessel_id=0, location=0, destination=1, speed=12.0)
        agent = VesselAgent(state, cfg)
        obs = agent.get_obs(
            short_forecast_row=np.ones(6),
            directive={"dest_port": 1, "departure_window_hours": 12, "emission_budget": 50},
            dock_availability=0.67,
        )
        # 5 local + 6 forecast + 3 directive = 14
        self.assertEqual(len(obs), 5 + 6 + 3)

    def test_dock_avail_value_in_obs(self) -> None:
        cfg = get_default_config(short_horizon_hours=4)
        state = VesselState(vessel_id=0, location=0, destination=1, speed=10.0)
        agent = VesselAgent(state, cfg)
        obs = agent.get_obs(
            short_forecast_row=np.zeros(4),
            dock_availability=0.75,
        )
        # dock_availability is the 5th element (index 4) of the local block
        self.assertAlmostEqual(obs[4], 0.75, places=5)

    def test_env_passes_dock_avail_to_vessel_obs(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=2, rollout_steps=10)
        env = MaritimeEnv(config=cfg, seed=42)
        obs = env.reset()
        # Vessel obs dimension should be 5 + short_h + 3
        short_h = cfg["short_horizon_hours"]
        expected_dim = 5 + short_h + 3
        for v_obs in obs["vessels"]:
            self.assertEqual(len(v_obs), expected_dim)


# -----------------------------------------------------------------------
# Port reward uses waiting time (proposal §4.2)
# -----------------------------------------------------------------------


class TestPortRewardWaitTime(unittest.TestCase):
    """Port reward penalises queue waiting time, not just count."""

    def test_queue_penalty_is_time_weighted(self) -> None:
        """With dt=1.0, queue=5 → penalty = 5*1.0 = 5.0 (not just queue count)."""
        port = PortState(port_id=0, queue=5, docks=3, occupied=1)
        cfg = get_default_config()
        r = compute_port_reward(port, cfg)
        # wait_penalty = queue * dt_hours = 5 * 1.0 = 5.0
        # idle = 3 - 1 = 2, idle_penalty = 0.5 * 2 = 1.0
        expected = -(5.0 + 1.0)
        self.assertAlmostEqual(r, expected, places=5)

    def test_custom_dt_affects_reward(self) -> None:
        """If dt_hours is set, wait penalty scales accordingly."""
        port = PortState(port_id=0, queue=4, docks=2, occupied=2)
        cfg = dict(get_default_config())
        cfg["dt_hours"] = 2.0  # 2-hour steps
        r = compute_port_reward(port, cfg)
        # wait_penalty = 4 * 2.0 = 8.0, idle = 0
        expected = -8.0
        self.assertAlmostEqual(r, expected, places=5)


# -----------------------------------------------------------------------
# Coordinator metrics (proposal §6.3)
# -----------------------------------------------------------------------


class TestCoordinatorMetrics(unittest.TestCase):
    """Test emission budget compliance, route efficiency, trip duration."""

    def test_emission_compliance_all_within_budget(self) -> None:
        vessels = [
            VesselState(vessel_id=0, location=0, destination=1, emissions=10.0),
            VesselState(vessel_id=1, location=1, destination=2, emissions=20.0),
        ]
        cfg = {"emission_budget": 50.0}
        m = compute_coordinator_metrics(vessels, cfg)
        self.assertAlmostEqual(m["emission_budget_compliance"], 1.0)

    def test_emission_compliance_half_exceed(self) -> None:
        vessels = [
            VesselState(vessel_id=0, location=0, destination=1, emissions=10.0),
            VesselState(vessel_id=1, location=1, destination=2, emissions=60.0),
        ]
        cfg = {"emission_budget": 50.0}
        m = compute_coordinator_metrics(vessels, cfg)
        self.assertAlmostEqual(m["emission_budget_compliance"], 0.5)

    def test_route_efficiency_defaults_to_one(self) -> None:
        vessels = [
            VesselState(vessel_id=0, location=0, destination=0),
        ]
        cfg: dict = {"emission_budget": 50.0}
        m = compute_coordinator_metrics(vessels, cfg)
        self.assertAlmostEqual(m["avg_route_efficiency"], 1.0)

    def test_trip_duration_computed(self) -> None:
        v = VesselState(
            vessel_id=0, location=1, destination=1,
            at_sea=False, trip_start_step=5,
        )
        cfg = {"emission_budget": 50.0, "_current_step": 15}
        m = compute_coordinator_metrics([v], cfg)
        self.assertAlmostEqual(m["avg_trip_duration_hours"], 10.0)


# -----------------------------------------------------------------------
# Coordination metrics (proposal §6.3)
# -----------------------------------------------------------------------


class TestCoordinationMetrics(unittest.TestCase):
    """Test policy agreement rate and communication overhead."""

    def test_full_agreement(self) -> None:
        m = compute_coordination_metrics(
            requests_submitted=10, requests_accepted=10, messages_exchanged=30,
        )
        self.assertAlmostEqual(m["policy_agreement_rate"], 1.0)

    def test_partial_agreement(self) -> None:
        m = compute_coordination_metrics(
            requests_submitted=10, requests_accepted=7, messages_exchanged=20,
        )
        self.assertAlmostEqual(m["policy_agreement_rate"], 0.7)

    def test_zero_requests(self) -> None:
        m = compute_coordination_metrics(
            requests_submitted=0, requests_accepted=0, messages_exchanged=5,
        )
        self.assertAlmostEqual(m["policy_agreement_rate"], 0.0)

    def test_communication_overhead(self) -> None:
        m = compute_coordination_metrics(
            requests_submitted=5, requests_accepted=3, messages_exchanged=42,
        )
        self.assertAlmostEqual(m["communication_overhead"], 42.0)


# -----------------------------------------------------------------------
# Trip start tracking in dispatch
# -----------------------------------------------------------------------


class TestTripStartTracking(unittest.TestCase):
    """VesselState.trip_start_step is set on dispatch."""

    def test_dispatch_sets_trip_start_step(self) -> None:
        cfg = dict(get_default_config())
        v = VesselState(vessel_id=0, location=0, destination=0, speed=12.0)
        dispatch_vessel(v, destination=2, speed=12.0, config=cfg, current_step=7)
        self.assertEqual(v.trip_start_step, 7)

    def test_dispatch_same_port_no_change(self) -> None:
        cfg = dict(get_default_config())
        v = VesselState(vessel_id=0, location=3, destination=3, speed=12.0, trip_start_step=0)
        dispatch_vessel(v, destination=3, speed=12.0, config=cfg, current_step=10)
        # Dispatching to same port is a no-op
        self.assertEqual(v.trip_start_step, 0)

    def test_default_trip_start_step(self) -> None:
        v = VesselState(vessel_id=0, location=0, destination=1)
        self.assertEqual(v.trip_start_step, 0)
