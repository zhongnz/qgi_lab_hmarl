"""Reward functions for coordinator, vessel, and port agents.

Reward Design
=============
The HMARL reward structure uses **three independent per-step reward
signals** — one per agent type — plus two optional **weather-aware
shaping terms** that encourage fuel-efficient behaviour in rough seas.

Vessel reward (per step):
    ``r_V(t) = -(fuel_weight * Δfuel_t + delay_weight * Δdelay_t + emission_weight * ΔCO2_t
                 + transit_time_weight * Δtransit_t) + arrival_reward * 1[arrived_t]``
    With defaults (1.0, 1.5, 0.7) rewards range from 0 (docked) to ~-20 (fast transit).

Port reward (per step):
    ``r_P(t) = port_accept_reward * accepted_t + port_service_reward * served_t
                 - port_reject_penalty * rejected_t
                 - (queue_t * dt_hours + dock_idle_weight * idle_docks_t)``
    Penalises both accumulated waiting time and wasted berth capacity while
    rewarding prompt admissions and actual berth service starts.

Coordinator reward (per step):
    ``r_C(t) = step_accept_reward_t + step_served_reward_t - step_reject_penalty_t
                 - (Δfuel_total_t + avg_queue_t
                 + coordinator_idle_dock_weight * avg_idle_docks_t
                 + coordinator_delay_weight * Δdelay_t
                 + emission_lambda * ΔCO2_total_t)``
    System-level signal; adds direct pressure against waiting/rejection delay
    and a small positive bonus for actual berth service throughput.

Weather shaping (opt-in, additive):
    * **Vessel**: bonus for slowing down when sea state raises fuel multiplier > 1.1.
    * **Coordinator**: bonus for routing fleet through calmer seas.
    Both are gated on ``weather_enabled=True`` and scale with ``weather_shaping_weight``.

The rewards are consumed by ``env._compute_rewards()`` in Phase 5 of the
transition kernel, using ``W_t`` (the weather active during the current
tick).  Rewards are normalised in MAPPO via Welford running statistics
when ``normalize_rewards=True``.
"""

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
    transit_hours: float = 0.0,
    arrived: bool = False,
) -> float:
    """Per-step vessel reward using step-level deltas.

    Returns
    ``-(fuel_weight * Δfuel + delay_weight * Δdelay + emission_weight * Δco2
       + transit_time_weight * Δtransit) + arrival_reward * 1[arrived]``.
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
    transit_cost = config.get("transit_time_weight", 0.0) * float(max(transit_hours, 0.0))
    arrival_bonus = config.get("arrival_reward", 0.0) if arrived else 0.0
    return arrival_bonus - (fuel_cost + delay_cost + emission_cost + transit_cost)


def compute_port_reward(
    port: PortState,
    config: dict[str, Any],
    served_vessels: float = 0.0,
    accepted_requests: float = 0.0,
    rejected_requests: float = 0.0,
) -> float:
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
    accept_bonus = config.get("port_accept_reward", 0.0) * float(max(accepted_requests, 0.0))
    reject_penalty = config.get("port_reject_penalty", 0.0) * float(max(rejected_requests, 0.0))
    service_bonus = config.get("port_service_reward", 0.0) * float(max(served_vessels, 0.0))
    return accept_bonus + service_bonus - (wait_penalty + idle_penalty + reject_penalty)


def compute_coordinator_reward_step(
    ports: list[PortState],
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
    delay_hours: float = 0.0,
    served_vessels: float = 0.0,
    accepted_requests: float = 0.0,
    rejected_requests: float = 0.0,
) -> float:
    """System-level coordinator reward using step-level deltas.

    Returns ``step_accept_reward + step_served_reward - step_reject_penalty - (Δfuel_total + avg_queue
    + coordinator_idle_dock_weight * avg_idle_docks
    + coordinator_delay_weight * Δdelay + emission_lambda * Δco2_total)``.
    The emission_lambda intentionally amplifies the CO2 signal at the
    coordinator level relative to vessel rewards, while the delay term
    aligns the coordinator with slot-allocation latency and missed service.
    """
    avg_queue = float(np.mean([p.queue for p in ports])) if ports else 0.0
    avg_idle_docks = (
        float(np.mean([max(p.docks - p.occupied, 0) for p in ports])) if ports else 0.0
    )
    voyage_cost = float(max(fuel_used, 0.0)) + avg_queue
    idle_penalty = config.get("coordinator_idle_dock_weight", 0.0) * avg_idle_docks
    delay_penalty = config.get("coordinator_delay_weight", 0.0) * float(max(delay_hours, 0.0))
    emission_penalty = config["emission_lambda"] * float(max(co2_emitted, 0.0))
    accept_bonus = config.get("coordinator_accept_reward", 0.0) * float(
        max(accepted_requests, 0.0)
    )
    reject_penalty = config.get("coordinator_reject_penalty", 0.0) * float(
        max(rejected_requests, 0.0)
    )
    throughput_bonus = config.get("coordinator_service_reward", 0.0) * float(
        max(served_vessels, 0.0)
    )
    return accept_bonus + throughput_bonus - (
        voyage_cost + idle_penalty + delay_penalty + emission_penalty + reject_penalty
    )


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
