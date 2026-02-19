"""Gym-style environment skeleton for HMARL maritime simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG, DISTANCE_NM, SEED, get_default_config
from .dynamics import dispatch_vessel, observe_port_metrics, step_ports, step_vessels
from .forecasts import medium_term_forecast, short_term_forecast
from .policies import fleet_coordinator_policy, port_policy, vessel_policy
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
        self.medium_forecast: np.ndarray | None = None
        self.short_forecast: np.ndarray | None = None

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

        for vessel, action in zip(self.vessels, vessel_actions):
            if not vessel.at_sea:
                dispatch_vessel(
                    vessel=vessel,
                    destination=coord_action["dest_port"],
                    speed=action["target_speed"],
                    config=self.cfg,
                )

        step_vessels(
            vessels=self.vessels,
            distance_nm=self.distance_nm,
            config=self.cfg,
            dt_hours=1.0,
        )

        for vessel in self.vessels:
            if not vessel.at_sea and vessel.location == vessel.destination:
                self.ports[vessel.location].queue += 1

        service_rates = [int(a["service_rate"]) for a in port_actions]
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
        directive = fleet_coordinator_policy(medium, self.vessels)
        vessel_actions = [
            vessel_policy(v, short, directive, config=self.cfg) for v in self.vessels
        ]
        incoming = sum(1 for a in vessel_actions if a["request_arrival_slot"])
        port_actions = [
            port_policy(p, incoming, short[i]) for i, p in enumerate(self.ports)
        ]
        return {
            "coordinator": directive,
            "vessels": vessel_actions,
            "ports": port_actions,
        }

    def _refresh_forecasts(self) -> None:
        """Refresh cached forecasts exactly once per environment tick."""
        self.medium_forecast = medium_term_forecast(
            num_ports=self.num_ports,
            horizon_days=self.cfg["medium_horizon_days"],
            rng=self.rng,
        )
        self.short_forecast = short_term_forecast(
            num_ports=self.num_ports,
            horizon_hours=self.cfg["short_horizon_hours"],
            rng=self.rng,
        )

    def _get_observations(self) -> dict[str, Any]:
        if self.medium_forecast is None or self.short_forecast is None:
            self._refresh_forecasts()
        medium = self.medium_forecast
        short = self.short_forecast

        vessel_summaries = np.array(
            [[v.location, v.speed, v.fuel, v.emissions] for v in self.vessels]
        )
        total_emissions = sum(v.emissions for v in self.vessels)
        coord_obs = np.concatenate(
            [
                medium.flatten(),
                vessel_summaries.flatten() if vessel_summaries.size else np.array([]),
                np.array([total_emissions], dtype=float),
            ]
        )

        vessel_obs = []
        for vessel in self.vessels:
            dest = vessel.destination if vessel.destination < self.num_ports else 0
            dest_forecast = short[dest]
            v_obs = np.concatenate(
                [
                    np.array(
                        [vessel.location, vessel.speed, vessel.fuel, vessel.emissions],
                        dtype=float,
                    ),
                    dest_forecast,
                    np.array([0.0, 0.0, 0.0], dtype=float),
                ]
            )
            vessel_obs.append(v_obs)

        port_obs = []
        for i, port in enumerate(self.ports):
            p_obs = np.concatenate(
                [
                    np.array([port.queue, port.docks, port.occupied], dtype=float),
                    short[i],
                    np.array([0.0], dtype=float),
                ]
            )
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
