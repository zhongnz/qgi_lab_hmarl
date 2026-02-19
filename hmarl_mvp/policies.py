"""Policy stubs for hierarchy and baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import PortState, VesselState


def fleet_coordinator_policy(
    medium_forecast: np.ndarray,
    vessels: list[VesselState],
) -> dict[str, Any]:
    """Forecast-informed strategic directive."""
    port_scores = medium_forecast.mean(axis=1)
    dest_port = int(np.argmin(port_scores))
    total_emissions = sum(v.emissions for v in vessels)
    return {
        "dest_port": dest_port,
        "departure_window_hours": 12,
        "emission_budget": max(50.0 - total_emissions * 0.1, 10.0),
    }


def reactive_coordinator_policy(ports: list[PortState]) -> dict[str, Any]:
    """Reactive strategic directive from current queue state only."""
    dest_port = int(np.argmin([p.queue for p in ports]))
    return {
        "dest_port": dest_port,
        "departure_window_hours": 12,
        "emission_budget": 50.0,
    }


def vessel_policy(
    vessel: VesselState,
    short_forecast: np.ndarray,
    directive: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Forecast-aware vessel policy with speed thresholds."""
    del vessel  # placeholder for learned policy features
    dest_port = int(directive["dest_port"])
    congestion = float(short_forecast[dest_port].mean())
    if congestion > 5.0:
        speed = config["speed_min"]
    elif congestion > 3.0:
        speed = config["nominal_speed"]
    else:
        speed = config["speed_max"]
    return {
        "target_speed": float(speed),
        "request_arrival_slot": True,
    }


def reactive_vessel_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Reactive policy without forecasts."""
    return {
        "target_speed": float(config["nominal_speed"]),
        "request_arrival_slot": True,
    }


def independent_vessel_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Independent policy with no explicit coordination request."""
    return {
        "target_speed": float(config["nominal_speed"]),
        "request_arrival_slot": False,
    }


def port_policy(
    port_state: PortState,
    incoming_requests: int,
    short_forecast_row: np.ndarray,
) -> dict[str, Any]:
    """Forecast-aware service policy."""
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


def reactive_port_policy(port_state: PortState) -> dict[str, Any]:
    """Reactive policy from queue/occupancy only."""
    service_rate = min(port_state.docks, max(port_state.queue, 1))
    return {
        "service_rate": int(service_rate),
        "accept_requests": int(max(port_state.docks - port_state.occupied, 0)),
    }


def independent_port_policy() -> dict[str, Any]:
    """Independent baseline with fixed minimal service behavior."""
    return {"service_rate": 1, "accept_requests": 0}

