"""Environment dynamics and physics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import PortState, VesselState

# ---------------------------------------------------------------------------
# Weather generation
# ---------------------------------------------------------------------------


def generate_weather(
    num_ports: int,
    rng: np.random.Generator,
    sea_state_max: float = 3.0,
) -> np.ndarray:
    """Generate a symmetric sea-state matrix for the current step.

    Returns a ``(num_ports, num_ports)`` matrix where entry ``[i, j]``
    is the sea state on the route from port *i* to port *j*.  Values
    are drawn uniformly from ``[0, sea_state_max]`` and the matrix is
    made symmetric (same conditions in both directions).  The diagonal
    is zero (no route from a port to itself).
    """
    raw = rng.uniform(0, sea_state_max, size=(num_ports, num_ports))
    symmetric = (raw + raw.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


def update_weather_ar1(
    prev: np.ndarray,
    rng: np.random.Generator,
    autocorrelation: float = 0.7,
    sea_state_max: float = 3.0,
) -> np.ndarray:
    """Advance weather one tick using an AR(1) process.

    ``next = autocorrelation * prev + (1 - autocorrelation) * noise``

    where *noise* is drawn from ``generate_weather``.  The result is
    clipped to ``[0, sea_state_max]`` and made symmetric with a zero
    diagonal.  When ``autocorrelation == 0`` this degenerates to pure
    i.i.d. noise (backward compatible).
    """
    autocorrelation = float(np.clip(autocorrelation, 0.0, 1.0))
    noise = generate_weather(prev.shape[0], rng, sea_state_max)
    updated = autocorrelation * prev + (1.0 - autocorrelation) * noise
    updated = np.clip(updated, 0.0, sea_state_max)
    # Re-symmetrise and zero diagonal
    updated = (updated + updated.T) / 2.0
    np.fill_diagonal(updated, 0.0)
    return updated


def weather_fuel_multiplier(sea_state: float, penalty_factor: float = 0.15) -> float:
    """Return the fuel multiplier caused by sea state.

    ``multiplier = 1 + penalty_factor * sea_state``

    With defaults (penalty 0.15, max sea state 3), worst case is a
    1.45× fuel burn which is consistent with IMO rough-weather
    guidelines.
    """
    return 1.0 + penalty_factor * max(float(sea_state), 0.0)


def weather_speed_factor(sea_state: float, penalty_factor: float = 0.15) -> float:
    """Return effective speed fraction under weather.

    ``factor = 1 / (1 + penalty_factor * sea_state)``

    This reduces effective distance travelled per tick in rough seas.
    """
    return 1.0 / weather_fuel_multiplier(sea_state, penalty_factor)


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------


def compute_fuel_and_emissions(
    speed: float,
    config: dict[str, Any],
    hours: float = 1.0,
    sea_state: float = 0.0,
) -> tuple[float, float]:
    """Cubic fuel model with linear emission factor and optional weather penalty.

    When ``sea_state > 0`` and the config has ``weather_penalty_factor``,
    fuel consumption is multiplied by ``1 + penalty_factor * sea_state``
    to model increased resistance in rough seas.
    """
    penalty = float(config.get("weather_penalty_factor", 0.15))
    multiplier = weather_fuel_multiplier(sea_state, penalty)
    fuel = config["fuel_rate_coeff"] * (speed**3) * hours * multiplier
    co2 = fuel * config["emission_factor"]
    return fuel, co2


def step_vessels(
    vessels: list[VesselState],
    distance_nm: np.ndarray,
    config: dict[str, Any],
    dt_hours: float = 1.0,
    weather: np.ndarray | None = None,
    current_step: int = 0,
) -> dict[int, dict[str, float | bool]]:
    """Advance all in-transit vessels by one tick and return per-vessel step deltas.

    If *weather* is provided (a ``num_ports × num_ports`` sea-state matrix),
    fuel consumption is increased and effective distance covered is reduced
    proportionally.

    Vessels with ``pending_departure=True`` are activated (``at_sea=True``) once
    ``current_step >= vessel.depart_at_step`` (departure-window enforcement).
    """
    penalty = float(config.get("weather_penalty_factor", 0.15))
    step_stats: dict[int, dict[str, float | bool]] = {}
    for vessel in vessels:
        # Activate pending departures when the coordinator's departure window opens.
        if vessel.pending_departure and current_step >= vessel.depart_at_step:
            vessel.at_sea = True
            vessel.pending_departure = False

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

        # Determine sea state for this vessel's route
        sea_state = 0.0
        if weather is not None:
            src = vessel.location
            dst = vessel.destination
            if 0 <= src < weather.shape[0] and 0 <= dst < weather.shape[1]:
                sea_state = float(weather[src, dst])

        # Weather reduces effective speed (less distance per tick)
        speed_frac = weather_speed_factor(sea_state, penalty)
        effective_advance = vessel.speed * dt_hours * speed_frac
        vessel.position_nm += effective_advance

        fuel_used, co2 = compute_fuel_and_emissions(
            vessel.speed, config, dt_hours, sea_state=sea_state,
        )
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
    departure_window_hours: int = 0,
    dt_hours: float = 1.0,
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
    departure_window_hours:
        Minimum number of hours to wait before departing, as instructed by
        the fleet coordinator's directive.  When > 0 the vessel is placed in
        ``pending_departure`` mode and ``step_vessels`` will activate it once
        ``current_step >= depart_at_step``.
    dt_hours:
        Simulation time step in hours, used to convert the window to steps.
    """
    if int(destination) == vessel.location:
        return
    vessel.destination = int(destination)
    vessel.speed = float(np.clip(speed, config["speed_min"], config["speed_max"]))
    vessel.position_nm = 0.0
    vessel.trip_start_step = int(current_step)
    window_steps = int(departure_window_hours / dt_hours) if dt_hours > 0 else 0
    if window_steps > 0:
        vessel.pending_departure = True
        vessel.depart_at_step = int(current_step) + window_steps
        vessel.at_sea = False
    else:
        vessel.at_sea = True
        vessel.pending_departure = False

