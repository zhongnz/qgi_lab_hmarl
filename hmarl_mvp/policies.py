"""Policy stubs and policy classes for hierarchy and baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent
from .state import PortState, VesselState


class FleetCoordinatorPolicy:
    """Coordinator policy container (heuristic placeholder for PPO actor)."""

    def __init__(self, config: dict[str, Any], mode: str = "forecast") -> None:
        self.cfg = config
        self.mode = mode

    def act(
        self,
        agent: FleetCoordinatorAgent,
        medium_forecast: np.ndarray,
        vessels: list[VesselState],
        ports: list[PortState],
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Produce strategic directive for one step."""
        if self.mode == "independent":
            if rng is None:
                raise ValueError("rng is required for independent coordinator mode")
            action = {
                "dest_port": int(rng.integers(0, self.cfg["num_ports"])),
                "departure_window_hours": 12,
                "emission_budget": 50.0,
            }
        elif self.mode == "reactive":
            dest_port = int(np.argmin([p.queue for p in ports])) if ports else 0
            action = {
                "dest_port": dest_port,
                "departure_window_hours": 12,
                "emission_budget": 50.0,
            }
        else:
            port_scores = medium_forecast.mean(axis=1)
            dest_port = int(np.argmin(port_scores))
            total_emissions = sum(v.emissions for v in vessels)
            action = {
                "dest_port": dest_port,
                "departure_window_hours": 12,
                "emission_budget": max(50.0 - total_emissions * 0.1, 10.0),
            }
        return agent.apply_action(action)


class VesselPolicy:
    """Vessel policy container (heuristic placeholder for PPO actor)."""

    def __init__(self, config: dict[str, Any], mode: str = "forecast") -> None:
        self.cfg = config
        self.mode = mode

    def act(
        self,
        agent: VesselAgent,
        short_forecast: np.ndarray,
        directive: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce local vessel action for one step."""
        if self.mode == "independent":
            action = {
                "target_speed": float(self.cfg["nominal_speed"]),
                "request_arrival_slot": False,
            }
        elif self.mode == "reactive":
            action = {
                "target_speed": float(self.cfg["nominal_speed"]),
                "request_arrival_slot": True,
            }
        else:
            dest_port = int(directive["dest_port"])
            congestion = float(short_forecast[dest_port].mean())
            if congestion > 5.0:
                speed = self.cfg["speed_min"]
            elif congestion > 3.0:
                speed = self.cfg["nominal_speed"]
            else:
                speed = self.cfg["speed_max"]
            action = {
                "target_speed": float(speed),
                "request_arrival_slot": True,
            }
        return agent.apply_action(action)


class PortPolicy:
    """Port policy container (heuristic placeholder for PPO actor)."""

    def __init__(self, config: dict[str, Any], mode: str = "forecast") -> None:
        self.cfg = config
        self.mode = mode

    def act(
        self,
        agent: PortAgent,
        incoming_requests: int,
        short_forecast_row: np.ndarray,
    ) -> dict[str, Any]:
        """Produce local port action for one step."""
        if self.mode == "independent":
            action = {"service_rate": 1, "accept_requests": 0}
        elif self.mode == "reactive":
            service_rate = min(agent.state.docks, max(agent.state.queue, 1))
            action = {
                "service_rate": int(service_rate),
                "accept_requests": int(max(agent.state.docks - agent.state.occupied, 0)),
            }
        else:
            pressure = float(short_forecast_row.mean())
            if pressure > 4.0:
                service_rate = agent.state.docks
            elif agent.state.queue > 2:
                service_rate = min(agent.state.docks, agent.state.queue)
            else:
                service_rate = min(agent.state.docks, agent.state.occupied + 1)
            action = {
                "service_rate": int(service_rate),
                "accept_requests": int(
                    min(incoming_requests, max(agent.state.docks - agent.state.occupied, 0))
                ),
            }
        return agent.apply_action(action)


def fleet_coordinator_policy(
    medium_forecast: np.ndarray,
    vessels: list[VesselState],
) -> dict[str, Any]:
    """Backward-compatible function wrapper for forecast coordinator mode."""
    cfg = {"num_ports": int(medium_forecast.shape[0])}
    policy = FleetCoordinatorPolicy(config=cfg, mode="forecast")
    agent = FleetCoordinatorAgent(config=cfg)
    ports: list[PortState] = []
    return policy.act(
        agent=agent,
        medium_forecast=medium_forecast,
        vessels=vessels,
        ports=ports,
    )


def reactive_coordinator_policy(ports: list[PortState]) -> dict[str, Any]:
    """Backward-compatible function wrapper for reactive coordinator mode."""
    cfg = {"num_ports": len(ports)}
    policy = FleetCoordinatorPolicy(config=cfg, mode="reactive")
    agent = FleetCoordinatorAgent(config=cfg)
    medium = np.zeros((len(ports), 1), dtype=float)
    return policy.act(
        agent=agent,
        medium_forecast=medium,
        vessels=[],
        ports=ports,
    )


def vessel_policy(
    vessel: VesselState,
    short_forecast: np.ndarray,
    directive: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible function wrapper for forecast vessel mode."""
    agent = VesselAgent(state=vessel, config=config)
    policy = VesselPolicy(config=config, mode="forecast")
    return policy.act(agent, short_forecast, directive)


def reactive_vessel_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible function wrapper for reactive vessel mode."""
    vessel = VesselState(vessel_id=0, location=0, destination=0, speed=config["nominal_speed"])
    agent = VesselAgent(state=vessel, config=config)
    policy = VesselPolicy(config=config, mode="reactive")
    short = np.zeros((config.get("num_ports", 1), config.get("short_horizon_hours", 1)))
    directive = {"dest_port": 0, "departure_window_hours": 12, "emission_budget": 50.0}
    return policy.act(agent, short, directive)


def independent_vessel_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible function wrapper for independent vessel mode."""
    vessel = VesselState(vessel_id=0, location=0, destination=0, speed=config["nominal_speed"])
    agent = VesselAgent(state=vessel, config=config)
    policy = VesselPolicy(config=config, mode="independent")
    short = np.zeros((config.get("num_ports", 1), config.get("short_horizon_hours", 1)))
    directive = {"dest_port": 0, "departure_window_hours": 12, "emission_budget": 50.0}
    return policy.act(agent, short, directive)


def port_policy(
    port_state: PortState,
    incoming_requests: int,
    short_forecast_row: np.ndarray,
) -> dict[str, Any]:
    """Backward-compatible function wrapper for forecast port mode."""
    cfg = {"docks_per_port": port_state.docks}
    agent = PortAgent(state=port_state, config=cfg)
    policy = PortPolicy(config=cfg, mode="forecast")
    return policy.act(agent, incoming_requests, short_forecast_row)


def reactive_port_policy(port_state: PortState) -> dict[str, Any]:
    """Backward-compatible function wrapper for reactive port mode."""
    cfg = {"docks_per_port": port_state.docks}
    agent = PortAgent(state=port_state, config=cfg)
    policy = PortPolicy(config=cfg, mode="reactive")
    return policy.act(agent, incoming_requests=0, short_forecast_row=np.zeros(1, dtype=float))


def independent_port_policy() -> dict[str, Any]:
    """Backward-compatible function wrapper for independent port mode."""
    state = PortState(port_id=0, queue=0, docks=1, occupied=0)
    cfg = {"docks_per_port": 1}
    agent = PortAgent(state=state, config=cfg)
    policy = PortPolicy(config=cfg, mode="independent")
    return policy.act(agent, incoming_requests=0, short_forecast_row=np.zeros(1, dtype=float))
