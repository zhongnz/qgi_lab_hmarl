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
    departure_window_hours: int = 0


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
        remaining_range_nm: float = 0.0,
        deadline_delta_hours: float = 0.0,
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
        remaining_range_nm:
            Nautical miles remaining to destination (0.0 when docked).
        deadline_delta_hours:
            Hours until requested arrival deadline (0.0 when no deadline).
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
        # Vessel identity: normalized index [0, 1] so parameter-shared
        # vessels can differentiate and break herding symmetry.
        num_vessels = max(int(self.cfg.get("num_vessels", 1)), 1)
        vessel_id_norm = float(self.state.vessel_id) / max(num_vessels - 1, 1)
        return np.concatenate(
            [
                np.array(
                    [
                        vessel_id_norm,
                        self.state.location,
                        self.state.position_nm,
                        self.state.speed,
                        self.state.fuel,
                        self.state.emissions,
                        float(bool(self.state.stalled)),
                        float(getattr(self.state, "port_service_state", 0)),
                        dock_availability,
                        float(bool(self.state.at_sea)),
                        remaining_range_nm,
                        deadline_delta_hours,
                    ],
                    dtype=float,
                ),
                short_forecast_row.astype(float),
                directive_vec,
            ]
        )

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply local control action (speed target, arrival-slot request, and requested arrival time)."""
        raw_speed = action.get("target_speed", self.state.speed)
        raw_arrival = action.get("requested_arrival_time", 0.0)
        # Guard against NaN/Inf from corrupted network outputs
        speed_val = float(raw_speed) if np.isfinite(float(raw_speed)) else float(self.cfg["nominal_speed"])
        arrival_val = float(raw_arrival) if np.isfinite(float(raw_arrival)) else 0.0
        speed = float(
            np.clip(
                speed_val,
                self.cfg["speed_min"],
                self.cfg["speed_max"],
            )
        )
        normalized = {
            "target_speed": speed,
            "request_arrival_slot": bool(action.get("request_arrival_slot", False)),
            "requested_arrival_time": arrival_val,
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
        booked_arrivals: float = 0.0,
        imminent_arrivals: float = 0.0,
        weather_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build local port observation vector.

        ``weather_features`` is an optional compact weather summary
        for inbound routes to this port.
        """
        weather_vec = (
            np.asarray(weather_features, dtype=float).ravel()
            if weather_features is not None
            else np.array([], dtype=float)
        )
        return np.concatenate(
            [
                np.array(
                    [
                        self.state.queue,
                        self.state.docks,
                        self.state.occupied,
                        float(booked_arrivals),
                        float(imminent_arrivals),
                        float(self.state.occupied) / max(float(self.state.docks), 1.0),
                    ],
                    dtype=float,
                ),
                short_forecast_row.astype(float),
                np.array([float(incoming_requests)], dtype=float),
                weather_vec,
            ]
        )

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply local port control action."""
        raw_sr = action.get("service_rate", 1)
        raw_ar = action.get("accept_requests", 0)
        # Guard against NaN/Inf from corrupted network outputs
        service_rate = int(max(raw_sr, 0)) if np.isfinite(float(raw_sr)) else 1
        accept_requests = int(max(raw_ar, 0)) if np.isfinite(float(raw_ar)) else 0
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
            "departure_window_hours": 0,
            "emission_budget": 50.0,
        }

    def get_obs(
        self,
        medium_forecast: np.ndarray,
        vessels: list[VesselState],
        port_load_summary: np.ndarray | None = None,
        weather: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build coordinator observation vector.

        When ``weather_enabled=True`` in config, appends a flattened
        ``(num_ports x num_ports)`` weather matrix. If weather is unavailable,
        appends a zero vector to keep observation dimensions fixed.
        """
        vessel_summaries = np.array(
            [
                [
                    v.location,
                    v.position_nm,
                    v.speed,
                    v.fuel,
                    v.emissions,
                    float(bool(v.stalled)),
                    float(getattr(v, "port_service_state", 0)),
                ]
                for v in vessels
            ],
            dtype=float,
        )
        num_ports = int(self.cfg.get("num_ports", 0))
        expected_port_summary = num_ports * 5
        if port_load_summary is None:
            port_load_features = np.zeros(expected_port_summary, dtype=float)
        else:
            port_load_features = np.asarray(port_load_summary, dtype=float).ravel()
            if port_load_features.size < expected_port_summary:
                port_load_features = np.concatenate(
                    [port_load_features, np.zeros(expected_port_summary - port_load_features.size)]
                )
            else:
                port_load_features = port_load_features[:expected_port_summary]
        total_emissions = float(sum(v.emissions for v in vessels))
        weather_features = np.array([], dtype=float)
        if bool(self.cfg.get("weather_enabled", True)):
            expected = num_ports * num_ports
            if weather is not None:
                weather_arr = np.asarray(weather, dtype=float)
                if weather_arr.shape == (num_ports, num_ports):
                    weather_features = weather_arr.flatten()
                else:
                    weather_features = np.zeros(expected, dtype=float)
            else:
                weather_features = np.zeros(expected, dtype=float)
        return np.concatenate(
            [
                medium_forecast.flatten().astype(float),
                port_load_features,
                vessel_summaries.flatten() if vessel_summaries.size else np.array([]),
                np.array([total_emissions], dtype=float),
                weather_features,
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
