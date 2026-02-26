"""Policy stubs and policy classes for hierarchy and baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dynamics import weather_fuel_multiplier
from .state import PortState, VesselState


class FleetCoordinatorPolicy:
    """Coordinator policy container (heuristic placeholder for PPO actor)."""

    def __init__(self, config: dict[str, Any], mode: str = "forecast") -> None:
        self.cfg = config
        self.mode = mode

    def propose_action(
        self,
        medium_forecast: np.ndarray,
        vessels: list[VesselState],
        ports: list[PortState],
        rng: np.random.Generator | None = None,
        weather: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Produce a coordinator action without mutating agent state.

        Parameters
        ----------
        weather:
            Optional ``(num_ports, num_ports)`` sea-state matrix.
            When provided and mode is ``"forecast"`` or ``"oracle"``,
            the routing heuristic penalises routes through rough seas.
        """
        if self.mode == "independent":
            if rng is None:
                raise ValueError("rng is required for independent coordinator mode")
            num_ports = int(self.cfg["num_ports"])
            default_dest = int(rng.integers(0, num_ports))
            independent_destinations: dict[int, int] = {}
            if num_ports > 1:
                # Keep independent behavior random while avoiding immediate self-loops.
                for vessel in vessels:
                    destination = int(rng.integers(0, num_ports - 1))
                    if destination >= vessel.location:
                        destination += 1
                    independent_destinations[vessel.vessel_id] = destination
            return {
                "dest_port": default_dest,
                "per_vessel_dest": independent_destinations,
                "departure_window_hours": 12,
                "emission_budget": 50.0,
            }
        if self.mode == "reactive":
            dest_port = int(np.argmin([p.queue for p in ports])) if ports else 0
            return {
                "dest_port": dest_port,
                "departure_window_hours": 12,
                "emission_budget": 50.0,
            }
        port_scores = medium_forecast.mean(axis=1)

        # Weather-aware routing: penalise ports reachable only through
        # rough seas so the coordinator prefers calmer routes.
        if weather is not None:
            weather_penalty = float(self.cfg.get("weather_penalty_factor", 0.15))
            # Mean sea-state across all routes *to* each port (column mean).
            mean_sea_to_port = np.asarray(weather).mean(axis=0)
            # Normalise to [0,1] relative to the configured maximum.
            sea_max = float(self.cfg.get("sea_state_max", 3.0))
            normalised = np.clip(mean_sea_to_port / max(sea_max, 1e-6), 0.0, 1.0)
            # Higher sea state → higher score → less attractive destination.
            port_scores = port_scores + weather_penalty * normalised * port_scores.max()

        sorted_ports = [int(p) for p in np.argsort(port_scores)]
        dest_port = sorted_ports[0]
        total_emissions = sum(v.emissions for v in vessels)
        forecast_destinations: dict[int, int] = {}
        if vessels and len(sorted_ports) > 1:
            for i, v in enumerate(vessels):
                forecast_destinations[v.vessel_id] = sorted_ports[i % len(sorted_ports)]
        return {
            "dest_port": dest_port,
            "per_vessel_dest": forecast_destinations,
            "departure_window_hours": 12,
            "emission_budget": max(50.0 - total_emissions * 0.1, 10.0),
        }


class VesselPolicy:
    """Vessel policy container (heuristic placeholder for PPO actor)."""

    def __init__(self, config: dict[str, Any], mode: str = "forecast") -> None:
        self.cfg = config
        self.mode = mode

    def propose_action(
        self,
        short_forecast: np.ndarray,
        directive: dict[str, Any],
        sea_state: float = 0.0,
    ) -> dict[str, Any]:
        """Produce a vessel action without mutating agent state.

        Parameters
        ----------
        sea_state:
            Local sea-state value for this vessel's current route segment.
            When positive, ``forecast`` mode reduces speed to save fuel in
            rough conditions using :func:`weather_fuel_multiplier`.
        """
        if self.mode == "independent":
            return {
                "target_speed": float(self.cfg["nominal_speed"]),
                "request_arrival_slot": True,
            }
        if self.mode == "reactive":
            return {
                "target_speed": float(self.cfg["nominal_speed"]),
                "request_arrival_slot": True,
            }
        dest_port = int(directive["dest_port"])
        congestion = float(short_forecast[dest_port].mean())
        if congestion > 5.0:
            speed = self.cfg["speed_min"]
        elif congestion > 3.0:
            speed = self.cfg["nominal_speed"]
        else:
            speed = self.cfg["speed_max"]

        # Weather-aware speed adjustment: slow down in rough seas to
        # reduce the fuel-consumption multiplier from weather.
        if sea_state > 0.0:
            fuel_mult = weather_fuel_multiplier(
                sea_state, float(self.cfg.get("weather_penalty_factor", 0.15))
            )
            if fuel_mult > 1.3:
                # Very rough – drop to minimum speed regardless of congestion.
                speed = self.cfg["speed_min"]
            elif fuel_mult > 1.1:
                # Moderately rough – cap at nominal speed.
                speed = min(speed, self.cfg["nominal_speed"])

        return {
            "target_speed": float(speed),
            "request_arrival_slot": True,
        }


class PortPolicy:
    """Port policy container (heuristic placeholder for PPO actor)."""

    def __init__(self, config: dict[str, Any], mode: str = "forecast") -> None:
        self.cfg = config
        self.mode = mode

    def propose_action(
        self,
        port_state: PortState,
        incoming_requests: int,
        short_forecast_row: np.ndarray,
    ) -> dict[str, Any]:
        """Produce a port action without mutating agent state."""
        if self.mode == "independent":
            available_slots = max(port_state.docks - port_state.occupied, 0)
            return {"service_rate": 1, "accept_requests": int(min(incoming_requests, available_slots))}
        if self.mode == "reactive":
            service_rate = min(port_state.docks, max(port_state.queue, 1))
            return {
                "service_rate": int(service_rate),
                "accept_requests": int(max(port_state.docks - port_state.occupied, 0)),
            }
        pressure = float(short_forecast_row.mean())
        if pressure > 4.0:
            service_rate = port_state.docks
        elif port_state.queue > 2:
            service_rate = min(port_state.docks, port_state.queue)
        else:
            service_rate = min(port_state.docks, port_state.occupied + 1)
        return {
            "service_rate": int(service_rate),
            "accept_requests": int(
                min(incoming_requests, max(port_state.docks - port_state.occupied, 0))
            ),
        }
