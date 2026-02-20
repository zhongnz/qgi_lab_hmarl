"""Agent containers for vessel, port, and fleet-coordinator logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics import compute_fuel_and_emissions
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
        }

    def reset(self) -> None:
        """Reset local control history for a new rollout."""
        self.last_action = {
            "target_speed": float(self.state.speed),
            "request_arrival_slot": False,
        }

    def get_obs(
        self,
        short_forecast_row: np.ndarray,
        directive: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Build local vessel observation vector."""
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
                    ],
                    dtype=float,
                ),
                short_forecast_row.astype(float),
                directive_vec,
            ]
        )

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply local control action (speed target and arrival-slot request)."""
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
        }
        self.state.speed = speed
        self.last_action = normalized
        return normalized

    def local_emission_cost(self, hours: float = 1.0) -> float:
        """Local emission-weighted cost estimate for diagnostics."""
        _, co2 = compute_fuel_and_emissions(self.state.speed, self.cfg, hours=hours)
        return float(self.cfg["emission_weight"] * co2)

    def metrics(self) -> dict[str, float]:
        """Return compact vessel diagnostics."""
        return {
            "speed": float(self.state.speed),
            "fuel": float(self.state.fuel),
            "emissions": float(self.state.emissions),
            "delay_hours": float(self.state.delay_hours),
        }


class PortAgent:
    """Port agent wrapper around ``PortState``."""

    def __init__(self, state: PortState, config: dict[str, Any]) -> None:
        self.state = state
        self.cfg = config
        self.last_action: dict[str, Any] = {"service_rate": 1, "accept_requests": 0}

    def reset(self) -> None:
        """Reset local control history for a new rollout."""
        self.last_action = {"service_rate": 1, "accept_requests": 0}

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

    def local_emission_cost(self) -> float:
        """
        Port-local emission proxy.

        Port agent itself does not emit directly in this MVP; kept for
        symmetry with other agents and future extension.
        """
        return 0.0

    def metrics(self) -> dict[str, float]:
        """Return compact port diagnostics."""
        utilization = (self.state.occupied / self.state.docks) if self.state.docks else 0.0
        return {
            "queue": float(self.state.queue),
            "dock_utilization": float(utilization),
            "cumulative_wait_hours": float(self.state.cumulative_wait_hours),
        }


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

    def reset(self) -> None:
        """Reset strategic state for a new rollout."""
        self.state.cumulative_emissions = 0.0
        self.state.last_dest_port = 0
        self.state.emission_budget = 50.0
        self.state.departure_window_hours = 12
        self.last_action = {
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
        self.state.last_dest_port = normalized["dest_port"]
        self.state.departure_window_hours = normalized["departure_window_hours"]
        self.state.emission_budget = normalized["emission_budget"]
        self.last_action = normalized
        return normalized

    def local_emission_cost(self) -> float:
        """Coordinator-level emission penalty component."""
        return float(self.cfg["emission_lambda"] * self.state.cumulative_emissions)

    def metrics(self) -> dict[str, float]:
        """Return compact coordinator diagnostics."""
        return {
            "cumulative_emissions": float(self.state.cumulative_emissions),
            "emission_budget": float(self.state.emission_budget),
            "last_dest_port": float(self.state.last_dest_port),
        }

