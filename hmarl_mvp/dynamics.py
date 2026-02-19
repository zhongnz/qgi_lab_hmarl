"""Environment dynamics and physics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import PortState, VesselState


def compute_fuel_and_emissions(
    speed: float,
    config: dict[str, Any],
    hours: float = 1.0,
) -> tuple[float, float]:
    """Cubic fuel model with linear emission factor."""
    fuel = config["fuel_rate_coeff"] * (speed**3) * hours
    co2 = fuel * config["emission_factor"]
    return fuel, co2


def step_vessels(
    vessels: list[VesselState],
    distance_nm: np.ndarray,
    config: dict[str, Any],
    dt_hours: float = 1.0,
) -> None:
    """Advance all in-transit vessels by one tick."""
    for vessel in vessels:
        if not vessel.at_sea:
            continue
        vessel.position_nm += vessel.speed * dt_hours
        fuel_used, co2 = compute_fuel_and_emissions(vessel.speed, config, dt_hours)
        vessel.fuel = max(vessel.fuel - fuel_used, 0.0)
        vessel.emissions += co2

        leg_distance = distance_nm[vessel.location, vessel.destination]
        if vessel.position_nm >= leg_distance:
            vessel.position_nm = 0.0
            vessel.location = vessel.destination
            vessel.at_sea = False


def step_ports(
    ports: list[PortState],
    service_rates: list[int],
    dt_hours: float = 1.0,
) -> None:
    """Serve queued vessels and accumulate waiting time."""
    for port, rate in zip(ports, service_rates):
        served = min(port.queue, max(int(rate), 0))
        port.queue = max(port.queue - served, 0)
        port.occupied = min(port.docks, port.occupied + served)
        port.vessels_served += served
        port.cumulative_wait_hours += port.queue * dt_hours


def dispatch_vessel(
    vessel: VesselState,
    destination: int,
    speed: float,
    config: dict[str, Any],
) -> None:
    """Send a vessel to a destination at a clipped speed."""
    vessel.destination = int(destination)
    vessel.speed = float(np.clip(speed, config["speed_min"], config["speed_max"]))
    vessel.position_nm = 0.0
    vessel.at_sea = True


def observe_port_metrics(ports: list[PortState]) -> dict[str, float]:
    """Compact helper used by env info outputs."""
    avg_queue = float(np.mean([p.queue for p in ports])) if ports else 0.0
    dock_util = (
        float(np.mean([p.occupied / p.docks for p in ports])) if ports else 0.0
    )
    total_wait = float(sum(p.cumulative_wait_hours for p in ports))
    return {
        "avg_queue": avg_queue,
        "dock_utilization": dock_util,
        "total_wait_hours": total_wait,
    }

