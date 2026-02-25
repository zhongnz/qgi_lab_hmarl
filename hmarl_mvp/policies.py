"""Policy stubs and policy classes for hierarchy and baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

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
    ) -> dict[str, Any]:
        """Produce a coordinator action without mutating agent state."""
        if self.mode == "independent":
            if rng is None:
                raise ValueError("rng is required for independent coordinator mode")
            return {
                "dest_port": int(rng.integers(0, self.cfg["num_ports"])),
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
        sorted_ports = [int(p) for p in np.argsort(port_scores)]
        dest_port = sorted_ports[0]
        total_emissions = sum(v.emissions for v in vessels)
        per_vessel_dest: dict[int, int] = {}
        if vessels and len(sorted_ports) > 1:
            for i, v in enumerate(vessels):
                per_vessel_dest[v.vessel_id] = sorted_ports[i % len(sorted_ports)]
        return {
            "dest_port": dest_port,
            "per_vessel_dest": per_vessel_dest,
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
    ) -> dict[str, Any]:
        """Produce a vessel action without mutating agent state."""
        if self.mode == "independent":
            return {
                "target_speed": float(self.cfg["nominal_speed"]),
                "request_arrival_slot": False,
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
            return {"service_rate": 1, "accept_requests": 0}
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

