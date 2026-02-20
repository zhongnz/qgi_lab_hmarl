"""Gym-style environment skeleton for HMARL maritime simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent
from .config import DEFAULT_CONFIG, DISTANCE_NM, SEED, get_default_config
from .dynamics import dispatch_vessel, observe_port_metrics, step_ports, step_vessels
from .forecasts import MediumTermForecaster, ShortTermForecaster
from .policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from .rewards import (
    compute_coordinator_reward,
    compute_port_reward,
    compute_vessel_reward,
)
from .state import PortState, VesselState, initialize_ports, initialize_vessels, make_rng


class MaritimeEnv:
    """
    Gymnasium-style multi-agent maritime environment skeleton.

    The class intentionally uses plain dictionaries and numpy arrays to keep
    integration easy for both notebooks and future gymnasium wrappers.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        seed: int = SEED,
        distance_nm: np.ndarray | None = None,
    ) -> None:
        user_cfg = dict(config or {})
        if "seed" not in user_cfg:
            user_cfg["seed"] = seed
        self.cfg = get_default_config(**user_cfg)
        self.seed = int(self.cfg["seed"])
        self.rng = make_rng(self.seed)
        self.num_ports = self.cfg["num_ports"]
        self.num_vessels = self.cfg["num_vessels"]
        self.distance_nm = DISTANCE_NM if distance_nm is None else distance_nm
        self.t = 0

        self.ports: list[PortState] = []
        self.vessels: list[VesselState] = []
        self.coordinator: FleetCoordinatorAgent | None = None
        self.port_agents: list[PortAgent] = []
        self.vessel_agents: list[VesselAgent] = []
        self.medium_forecast: np.ndarray | None = None
        self.short_forecast: np.ndarray | None = None
        self.medium_forecaster = MediumTermForecaster(self.cfg["medium_horizon_days"])
        self.short_forecaster = ShortTermForecaster(self.cfg["short_horizon_hours"])
        self.coordinator_policy = FleetCoordinatorPolicy(self.cfg, mode="forecast")
        self.vessel_policy = VesselPolicy(self.cfg, mode="forecast")
        self.port_policy = PortPolicy(self.cfg, mode="forecast")

        self.obs_shapes = {
            "coordinator": (
                self.num_ports * self.cfg["medium_horizon_days"]
                + self.num_vessels * 4
                + 1
            ),
            "vessel": 1 + 1 + 1 + 1 + self.cfg["short_horizon_hours"] + 3,
            "port": 1 + 1 + 1 + self.cfg["short_horizon_hours"] + 1,
        }
        self.action_shapes = {
            "coordinator": self.num_ports + 2,
            "vessel": 2,
            "port": 2,
        }

    def reset(self) -> dict[str, Any]:
        """Reset state and return initial observations."""
        self.t = 0
        self.rng = make_rng(self.seed)
        self.ports = initialize_ports(
            num_ports=self.num_ports,
            docks_per_port=self.cfg["docks_per_port"],
            rng=self.rng,
        )
        self.vessels = initialize_vessels(
            num_vessels=self.num_vessels,
            num_ports=self.num_ports,
            nominal_speed=self.cfg["nominal_speed"],
            rng=self.rng,
        )
        self.coordinator = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        self.vessel_agents = [VesselAgent(v, self.cfg) for v in self.vessels]
        self.port_agents = [PortAgent(p, self.cfg) for p in self.ports]
        self._refresh_forecasts()
        return self._get_observations()

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool, dict[str, Any]]:
        """Execute one environment tick."""
        coord_action = actions["coordinator"]
        vessel_actions = actions["vessels"]
        port_actions = actions["ports"]
        if self.coordinator is None:
            self.coordinator = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        coord_action = self.coordinator.apply_action(coord_action)

        normalized_vessel_actions = []
        for vessel_agent, action in zip(self.vessel_agents, vessel_actions):
            normalized = vessel_agent.apply_action(action)
            normalized_vessel_actions.append(normalized)
            if not vessel_agent.state.at_sea:
                dispatch_vessel(
                    vessel=vessel_agent.state,
                    destination=coord_action["dest_port"],
                    speed=normalized["target_speed"],
                    config=self.cfg,
                )

        step_vessels(
            vessels=self.vessels,
            distance_nm=self.distance_nm,
            config=self.cfg,
            dt_hours=1.0,
        )

        for vessel_agent in self.vessel_agents:
            vessel = vessel_agent.state
            if not vessel.at_sea and vessel.location == vessel.destination:
                self.ports[vessel.location].queue += 1

        normalized_port_actions = [
            port_agent.apply_action(action)
            for port_agent, action in zip(self.port_agents, port_actions)
        ]
        service_rates = [int(a["service_rate"]) for a in normalized_port_actions]
        step_ports(self.ports, service_rates, dt_hours=1.0)

        rewards = self._compute_rewards()

        self.t += 1
        done = self.t >= self.cfg["rollout_steps"]
        self._refresh_forecasts()
        obs = self._get_observations()
        info = {"port_metrics": observe_port_metrics(self.ports)}
        return obs, rewards, done, info

    def sample_stub_actions(self) -> dict[str, Any]:
        """Helper for smoke tests and demos."""
        if self.medium_forecast is None or self.short_forecast is None:
            self._refresh_forecasts()
        medium = self.medium_forecast
        short = self.short_forecast
        if self.coordinator is None:
            self.coordinator = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        directive = self.coordinator_policy.act(
            agent=self.coordinator,
            medium_forecast=medium,
            vessels=self.vessels,
            ports=self.ports,
            rng=self.rng,
        )
        vessel_actions = [
            self.vessel_policy.act(vessel_agent, short, directive)
            for vessel_agent in self.vessel_agents
        ]
        incoming = sum(1 for a in vessel_actions if a["request_arrival_slot"])
        port_actions = [
            self.port_policy.act(port_agent, incoming, short[i])
            for i, port_agent in enumerate(self.port_agents)
        ]
        return {
            "coordinator": directive,
            "vessels": vessel_actions,
            "ports": port_actions,
        }

    def _refresh_forecasts(self) -> None:
        """Refresh cached forecasts exactly once per environment tick."""
        self.medium_forecast = self.medium_forecaster.predict(self.num_ports, self.rng)
        self.short_forecast = self.short_forecaster.predict(self.num_ports, self.rng)

    def _get_observations(self) -> dict[str, Any]:
        if self.medium_forecast is None or self.short_forecast is None:
            self._refresh_forecasts()
        medium = self.medium_forecast
        short = self.short_forecast

        if self.coordinator is None:
            self.coordinator = FleetCoordinatorAgent(config=self.cfg, coordinator_id=0)
        coord_obs = self.coordinator.get_obs(medium, self.vessels)

        vessel_obs = []
        directive = self.coordinator.last_action if self.coordinator is not None else None
        for vessel_agent in self.vessel_agents:
            vessel = vessel_agent.state
            dest = vessel.destination if vessel.destination < self.num_ports else 0
            v_obs = vessel_agent.get_obs(short[dest], directive=directive)
            vessel_obs.append(v_obs)

        port_obs = []
        for i, port_agent in enumerate(self.port_agents):
            p_obs = port_agent.get_obs(short[i], incoming_requests=0)
            port_obs.append(p_obs)

        return {"coordinator": coord_obs, "vessels": vessel_obs, "ports": port_obs}

    def _compute_rewards(self) -> dict[str, Any]:
        vessel_rewards = [compute_vessel_reward(v, self.cfg) for v in self.vessels]
        port_rewards = [compute_port_reward(p, self.cfg) for p in self.ports]
        coordinator_reward = compute_coordinator_reward(self.vessels, self.ports, self.cfg)
        return {
            "coordinator": coordinator_reward,
            "vessels": vessel_rewards,
            "ports": port_rewards,
        }

    def get_global_state(self) -> np.ndarray:
        """Flatten all observations and global stats for CTDE critic inputs."""
        obs = self._get_observations()
        global_congestion = np.array([p.queue for p in self.ports], dtype=float)
        total_emissions = np.array([sum(v.emissions for v in self.vessels)], dtype=float)
        return np.concatenate(
            [
                obs["coordinator"],
                *(obs["vessels"] if obs["vessels"] else [np.array([])]),
                *(obs["ports"] if obs["ports"] else [np.array([])]),
                global_congestion,
                total_emissions,
            ]
        )


def make_default_env() -> MaritimeEnv:
    """Convenience factory used by scripts and tests."""
    return MaritimeEnv(config=DEFAULT_CONFIG)
