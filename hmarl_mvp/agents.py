"""Agent containers for vessel, port, and fleet-coordinator logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .state import PortState, VesselState


@dataclass
class FleetCoordinatorState:
    """Minimal state tracked by the fleet coordinator."""

    coordinator_id: int = 0
    cumulative_emissions: float = 0.0
    last_dest_port: int = 0
    emission_budget: float = 50.0
    departure_window_hours: int = 12


class VesselAgent:
    """Vessel agent wrapper around ``VesselState``."""

    def __init__(self, state: VesselState, config: dict[str, Any]) -> None:
        self.state = state
        self.cfg = config
        self.last_action: dict[str, Any] = {
            "target_speed": float(self.state.speed),
            "request_arrival_slot": False,
            "requested_arrival_time": 0.0,
        }

    def get_obs(
        self,
        short_forecast_row: np.ndarray,
        directive: dict[str, Any] | None = None,
        dock_availability: float = 0.0,
    ) -> np.ndarray:
        """Build local vessel observation vector.

        Parameters
        ----------
        short_forecast_row:
            Short-term congestion forecast for the vessel's destination port.
        directive:
            Latest strategic directive from the fleet coordinator.
        dock_availability:
            Fraction of available (unoccupied) docks at the destination port.
            This corresponds to the "dock availability signals from ports"
            specified in the proposal (Section 4.1).
        """
        directive = directive or {}
        directive_vec = np.array(
            [
                float(directive.get("dest_port", 0)),
                float(directive.get("departure_window_hours", 0)),
                float(directive.get("emission_budget", 0)),
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                np.array(
                    [
                        self.state.location,
                        self.state.speed,
                        self.state.fuel,
                        self.state.emissions,
                        dock_availability,
                    ],
                    dtype=float,
                ),
                short_forecast_row.astype(float),
                directive_vec,
            ]
        )

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply local control action (speed target, arrival-slot request, and requested arrival time)."""
        speed = float(
            np.clip(
                action.get("target_speed", self.state.speed),
                self.cfg["speed_min"],
                self.cfg["speed_max"],
            )
        )
        normalized = {
            "target_speed": speed,
            "request_arrival_slot": bool(action.get("request_arrival_slot", False)),
            # Explicit arrival time (t_arr) requested by the vessel agent.
            # A value of 0.0 means "no preference"; positive values are absolute
            # simulation steps by which the vessel wants to arrive.
            "requested_arrival_time": float(action.get("requested_arrival_time", 0.0)),
        }
        self.state.speed = speed
        self.last_action = normalized
        return normalized


class PortAgent:
    """Port agent wrapper around ``PortState``."""

    def __init__(self, state: PortState, config: dict[str, Any]) -> None:
        self.state = state
        self.cfg = config
        self.last_action: dict[str, Any] = {"service_rate": 1, "accept_requests": 0}

    def get_obs(
        self,
        short_forecast_row: np.ndarray,
        incoming_requests: int = 0,
    ) -> np.ndarray:
        """Build local port observation vector."""
        return np.concatenate(
            [
                np.array(
                    [self.state.queue, self.state.docks, self.state.occupied],
                    dtype=float,
                ),
                short_forecast_row.astype(float),
                np.array([float(incoming_requests)], dtype=float),
            ]
        )

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply local port control action."""
        service_rate = int(max(action.get("service_rate", 1), 0))
        accept_requests = int(max(action.get("accept_requests", 0), 0))
        normalized = {
            "service_rate": service_rate,
            "accept_requests": accept_requests,
        }
        self.last_action = normalized
        return normalized


class FleetCoordinatorAgent:
    """Coordinator wrapper for strategic observations and directives."""

    def __init__(
        self,
        config: dict[str, Any],
        coordinator_id: int = 0,
    ) -> None:
        self.cfg = config
        self.state = FleetCoordinatorState(coordinator_id=coordinator_id)
        self.last_action: dict[str, Any] = {
            "dest_port": 0,
            "departure_window_hours": 12,
            "emission_budget": 50.0,
        }

    def get_obs(
        self,
        medium_forecast: np.ndarray,
        vessels: list[VesselState],
    ) -> np.ndarray:
        """Build coordinator observation vector."""
        vessel_summaries = np.array(
            [[v.location, v.speed, v.fuel, v.emissions] for v in vessels],
            dtype=float,
        )
        total_emissions = float(sum(v.emissions for v in vessels))
        self.state.cumulative_emissions = total_emissions
        return np.concatenate(
            [
                medium_forecast.flatten().astype(float),
                vessel_summaries.flatten() if vessel_summaries.size else np.array([]),
                np.array([total_emissions], dtype=float),
            ]
        )

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply strategic directive."""
        normalized = {
            "dest_port": int(action.get("dest_port", self.state.last_dest_port)),
            "departure_window_hours": int(
                action.get("departure_window_hours", self.state.departure_window_hours)
            ),
            "emission_budget": float(action.get("emission_budget", self.state.emission_budget)),
        }
        self.state.last_dest_port = int(normalized["dest_port"])
        self.state.departure_window_hours = int(normalized["departure_window_hours"])
        self.state.emission_budget = float(normalized["emission_budget"])
        self.last_action = normalized
        return normalized


def assign_vessels_to_coordinators(
    vessels: list[VesselState],
    num_coordinators: int,
) -> dict[int, list[int]]:
    """Partition vessels into coordinator groups.

    - Docked vessels: assignment by current location modulo coordinator count.
    - In-transit vessels: assignment by vessel id modulo coordinator count.
    """
    if num_coordinators <= 0:
        raise ValueError("num_coordinators must be >= 1")

    groups: dict[int, list[int]] = {i: [] for i in range(num_coordinators)}
    for vessel in vessels:
        if vessel.at_sea:
            coordinator_id = int(vessel.vessel_id % num_coordinators)
        else:
            coordinator_id = int(vessel.location % num_coordinators)
        groups[coordinator_id].append(vessel.vessel_id)
    return groups
