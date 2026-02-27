"""Reward functions for coordinator, vessel, and port agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dynamics import weather_fuel_multiplier
from .state import PortState, VesselState


def compute_vessel_reward_step(
    vessel: VesselState,
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
    delay_hours: float,
) -> float:
    """Per-step vessel reward using step-level deltas.

    Returns ``-(fuel_weight * fuel + delay_weight * delay + emission_weight * co2)``.
    With default weights (1.0, 1.5, 0.7) and typical per-step values
    (fuel ~ 0–7, co2 ~ 0–22, delay 0–1 h) rewards range roughly from
    0 (docked, no delay) to about −20 (fast transit).

    ``vessel`` is accepted for API stability; future reward shaping may
    condition on position or remaining fuel.
    """
    _ = vessel
    fuel_cost = config["fuel_weight"] * float(max(fuel_used, 0.0))
    delay_cost = config["delay_weight"] * float(max(delay_hours, 0.0))
    emission_cost = config["emission_weight"] * float(max(co2_emitted, 0.0))
    return -(fuel_cost + delay_cost + emission_cost)


def compute_port_reward(port: PortState, config: dict[str, Any]) -> float:
    """Per-step port reward as negative queue wait rate + idle dock penalty.

    Uses the queue-length-weighted waiting rate (``queue * dt``) as a
    proxy for waiting-time accumulation, matching the proposal's
    :math:`R_P = -(\\text{Queue waiting time} + \\text{Dock idle time})`.
    With default 3 docks and weight 0.5, range is roughly [−1.5, −10+]
    depending on queue buildup.
    """
    # queue * dt_hours measures per-step waiting-time accumulation
    dt_hours = float(config.get("dt_hours", 1.0))
    wait_penalty = float(port.queue) * dt_hours
    idle_docks = max(port.docks - port.occupied, 0)
    idle_penalty = config["dock_idle_weight"] * idle_docks
    return -(wait_penalty + idle_penalty)


def compute_coordinator_reward_step(
    ports: list[PortState],
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
) -> float:
    """System-level coordinator reward using step-level deltas.

    Returns ``-(fuel_used + avg_queue + emission_lambda * co2_emitted)``.
    The emission_lambda (default 2.0) intentionally amplifies the
    CO2 signal at the coordinator level relative to vessel rewards.
    """
    avg_queue = float(np.mean([p.queue for p in ports])) if ports else 0.0
    voyage_cost = float(max(fuel_used, 0.0)) + avg_queue
    emission_penalty = config["emission_lambda"] * float(max(co2_emitted, 0.0))
    return -(voyage_cost + emission_penalty)


# ---------------------------------------------------------------------------
# Weather-aware reward shaping (opt-in)
# ---------------------------------------------------------------------------


def weather_vessel_shaping(
    speed: float,
    sea_state: float,
    config: dict[str, Any],
) -> float:
    """Bonus for vessels that reduce speed in rough weather.

    Returns a small positive reward when the vessel's speed is at or
    below nominal while the fuel-consumption multiplier from weather
    exceeds 1.1.  This encourages fuel-efficient weather routing.

    Returns 0.0 when ``sea_state <= 0`` or weather is calm.
    """
    if sea_state <= 0.0:
        return 0.0
    fuel_mult = weather_fuel_multiplier(
        sea_state, float(config.get("weather_penalty_factor", 0.15))
    )
    if fuel_mult <= 1.1:
        return 0.0
    # Bonus proportional to how much fuel was *saved* by slowing down.
    nominal = float(config.get("nominal_speed", 12.0))
    if speed <= nominal:
        bonus_weight = float(config.get("weather_shaping_weight", 0.3))
        return bonus_weight * (fuel_mult - 1.0)
    return 0.0


def weather_coordinator_shaping(
    weather: np.ndarray | None,
    routes: list[tuple[int, int]] | dict[int, int],
    config: dict[str, Any],
) -> float:
    """Bonus for a coordinator that routes fleet through calmer seas.

    Computes mean sea-state along the chosen routes and rewards lower
    values.  Returns 0.0 when weather is ``None`` or no routes.

    *routes* may be a list of ``(src_port, dst_port)`` tuples or a
    legacy ``{src: dst}`` dict.
    """
    if weather is None or not routes:
        return 0.0
    n = weather.shape[0]
    pairs: list[tuple[int, int]]
    if isinstance(routes, dict):
        pairs = list(routes.items())
    else:
        pairs = list(routes)
    total_sea = 0.0
    count = 0
    for src, dst in pairs:
        if 0 <= src < n and 0 <= dst < n:
            total_sea += float(weather[src, dst])
            count += 1
    if count == 0:
        return 0.0
    mean_route_sea = total_sea / count
    sea_max = float(config.get("sea_state_max", 3.0))
    # Reward calm routes: bonus = weight * (1 - normalised_sea)
    normalised = min(mean_route_sea / max(sea_max, 1e-6), 1.0)
    bonus_weight = float(config.get("weather_shaping_weight", 0.3))
    return bonus_weight * (1.0 - normalised)
