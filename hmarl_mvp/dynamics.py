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
) -> dict[int, dict[str, float | bool]]:
    """Advance all in-transit vessels by one tick and return per-vessel step deltas."""
    step_stats: dict[int, dict[str, float | bool]] = {}
    for vessel in vessels:
        fuel_used = 0.0
        co2 = 0.0
        arrived = False
        if not vessel.at_sea:
            step_stats[vessel.vessel_id] = {
                "fuel_used": fuel_used,
                "co2_emitted": co2,
                "arrived": arrived,
            }
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
            arrived = True
        step_stats[vessel.vessel_id] = {
            "fuel_used": float(fuel_used),
            "co2_emitted": float(co2),
            "arrived": arrived,
        }
    return step_stats


def step_ports(
    ports: list[PortState],
    service_rates: list[int],
    dt_hours: float = 1.0,
    service_time_hours: float = 6.0,
) -> None:
    """Advance port service state, admit queued vessels, and accumulate waiting time.

    ``service_rates`` acts as a per-port *admission cap* — the maximum
    number of queued vessels that may be admitted to a berth this tick,
    subject to available dock capacity.
    """
    for port, rate in zip(ports, service_rates):
        # Normalize potentially stale manual test overrides of occupied/service_times.
        if len(port.service_times) != int(port.occupied):
            port.service_times = list(port.service_times[: max(int(port.occupied), 0)])
            if len(port.service_times) < int(port.occupied):
                missing = int(port.occupied) - len(port.service_times)
                port.service_times.extend([float(service_time_hours)] * missing)

        # Complete ongoing services and free berths.
        remaining = [max(t - dt_hours, 0.0) for t in port.service_times]
        port.service_times = [t for t in remaining if t > 0.0]
        port.occupied = len(port.service_times)

        port.cumulative_wait_hours += port.queue * dt_hours
        available_slots = max(port.docks - port.occupied, 0)
        served = min(port.queue, max(int(rate), 0), available_slots)
        port.queue = max(port.queue - served, 0)
        port.vessels_served += served
        if served > 0:
            port.service_times.extend([float(service_time_hours)] * int(served))
            port.occupied = len(port.service_times)


def dispatch_vessel(
    vessel: VesselState,
    destination: int,
    speed: float,
    config: dict[str, Any],
    current_step: int = 0,
) -> None:
    """Send a vessel to a destination at a clipped speed.

    If *destination* equals the vessel's current location the call is a
    no-op — dispatching to the same port would produce a zero-distance
    leg that arrives instantly and re-queues the vessel.

    Parameters
    ----------
    vessel:
        Vessel to dispatch.
    destination:
        Target port index.
    speed:
        Requested speed (will be clipped to [speed_min, speed_max]).
    config:
        Validated config dict.
    current_step:
        Current simulation step, recorded in ``vessel.trip_start_step``.
    """
    if int(destination) == vessel.location:
        return
    vessel.destination = int(destination)
    vessel.speed = float(np.clip(speed, config["speed_min"], config["speed_max"]))
    vessel.position_nm = 0.0
    vessel.at_sea = True
    vessel.trip_start_step = int(current_step)

