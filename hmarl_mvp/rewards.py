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
    ``r_C(t) = step_accept_reward_t + step_served_reward_t + utilization_reward_t
                 - step_reject_penalty_t - (fuel_weight * Δfuel_total_t
                 + queue_weight * avg_queue_t
                 + coordinator_idle_dock_weight * avg_idle_docks_t
                 + coordinator_delay_weight * Δdelay_t
                 + coordinator_emission_weight * ΔCO2_total_t)``
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


def compute_vessel_reward_breakdown(
    vessel: VesselState,
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
    delay_hours: float,
    transit_hours: float = 0.0,
    schedule_delay_hours: float = 0.0,
    arrived: bool = False,
    arrived_on_time: bool = False,
) -> dict[str, float]:
    """Return named vessel reward components for one step."""
    _ = vessel
    fuel_cost = config["fuel_weight"] * float(max(fuel_used, 0.0))
    delay_cost = config["delay_weight"] * float(max(delay_hours, 0.0))
    emission_cost = config["emission_weight"] * float(max(co2_emitted, 0.0))
    transit_cost = config.get("transit_time_weight", 0.0) * float(max(transit_hours, 0.0))
    schedule_delay_cost = config.get("schedule_delay_weight", 0.0) * float(
        max(schedule_delay_hours, 0.0)
    )
    arrival_bonus = config.get("arrival_reward", 0.0) if arrived else 0.0
    on_time_bonus = config.get("on_time_arrival_reward", 0.0) if arrived_on_time else 0.0
    weather_shaping_bonus = 0.0
    total = arrival_bonus + on_time_bonus - (
        fuel_cost + delay_cost + emission_cost + transit_cost + schedule_delay_cost
    )
    return {
        "fuel_cost": float(fuel_cost),
        "delay_cost": float(delay_cost),
        "emission_cost": float(emission_cost),
        "transit_cost": float(transit_cost),
        "schedule_delay_cost": float(schedule_delay_cost),
        "arrival_bonus": float(arrival_bonus),
        "on_time_bonus": float(on_time_bonus),
        "weather_shaping_bonus": float(weather_shaping_bonus),
        "total": float(total),
    }


def compute_port_reward_breakdown(
    port: PortState,
    config: dict[str, Any],
    served_vessels: float = 0.0,
    accepted_requests: float = 0.0,
    rejected_requests: float = 0.0,
) -> dict[str, float]:
    """Return named port reward components for one step."""
    dt_hours = float(config.get("dt_hours", 1.0))
    wait_penalty = float(port.queue) * dt_hours
    idle_docks = max(port.docks - port.occupied, 0)
    idle_penalty = config["dock_idle_weight"] * idle_docks
    accept_bonus = config.get("port_accept_reward", 0.0) * float(max(accepted_requests, 0.0))
    reject_penalty = config.get("port_reject_penalty", 0.0) * float(max(rejected_requests, 0.0))
    service_bonus = config.get("port_service_reward", 0.0) * float(max(served_vessels, 0.0))
    total = accept_bonus + service_bonus - (wait_penalty + idle_penalty + reject_penalty)
    return {
        "wait_penalty": float(wait_penalty),
        "idle_penalty": float(idle_penalty),
        "accept_bonus": float(accept_bonus),
        "reject_penalty": float(reject_penalty),
        "service_bonus": float(service_bonus),
        "total": float(total),
    }


def compute_coordinator_reward_breakdown(
    ports: list[PortState],
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
    delay_hours: float = 0.0,
    schedule_delay_hours: float = 0.0,
    served_vessels: float = 0.0,
    accepted_requests: float = 0.0,
    rejected_requests: float = 0.0,
) -> dict[str, float]:
    """Return named coordinator reward components for one step."""
    avg_queue = float(np.mean([p.queue for p in ports])) if ports else 0.0
    avg_idle_docks = (
        float(np.mean([max(p.docks - p.occupied, 0) for p in ports])) if ports else 0.0
    )
    avg_occupied_docks = float(np.mean([max(p.occupied, 0) for p in ports])) if ports else 0.0
    fuel_penalty = config.get("coordinator_fuel_weight", 1.0) * float(max(fuel_used, 0.0))
    queue_penalty = config.get("coordinator_queue_weight", 1.0) * avg_queue
    idle_penalty = config.get("coordinator_idle_dock_weight", 0.0) * avg_idle_docks
    utilization_bonus = config.get("coordinator_utilization_reward", 0.0) * avg_occupied_docks
    delay_penalty = config.get("coordinator_delay_weight", 0.0) * float(max(delay_hours, 0.0))
    schedule_delay_penalty = config.get("coordinator_schedule_delay_weight", 0.0) * float(
        max(schedule_delay_hours, 0.0)
    )
    emission_penalty = config.get(
        "coordinator_emission_weight",
        config.get("emission_lambda", 0.0),
    ) * float(max(co2_emitted, 0.0))
    accept_bonus = config.get("coordinator_accept_reward", 0.0) * float(
        max(accepted_requests, 0.0)
    )
    reject_penalty = config.get("coordinator_reject_penalty", 0.0) * float(
        max(rejected_requests, 0.0)
    )
    throughput_bonus = config.get("coordinator_service_reward", 0.0) * float(
        max(served_vessels, 0.0)
    )
    weather_shaping_bonus = 0.0
    total = accept_bonus + throughput_bonus + utilization_bonus - (
        fuel_penalty
        + queue_penalty
        + idle_penalty
        + delay_penalty
        + schedule_delay_penalty
        + emission_penalty
        + reject_penalty
    )
    return {
        "fuel_penalty": float(fuel_penalty),
        "queue_penalty": float(queue_penalty),
        "idle_penalty": float(idle_penalty),
        "utilization_bonus": float(utilization_bonus),
        "delay_penalty": float(delay_penalty),
        "schedule_delay_penalty": float(schedule_delay_penalty),
        "emission_penalty": float(emission_penalty),
        "accept_bonus": float(accept_bonus),
        "reject_penalty": float(reject_penalty),
        "throughput_bonus": float(throughput_bonus),
        "weather_shaping_bonus": float(weather_shaping_bonus),
        "total": float(total),
    }


def compute_vessel_reward_step(
    vessel: VesselState,
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
    delay_hours: float,
    transit_hours: float = 0.0,
    schedule_delay_hours: float = 0.0,
    arrived: bool = False,
    arrived_on_time: bool = False,
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
    return compute_vessel_reward_breakdown(
        vessel=vessel,
        config=config,
        fuel_used=fuel_used,
        co2_emitted=co2_emitted,
        delay_hours=delay_hours,
        transit_hours=transit_hours,
        schedule_delay_hours=schedule_delay_hours,
        arrived=arrived,
        arrived_on_time=arrived_on_time,
    )["total"]


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
    return compute_port_reward_breakdown(
        port=port,
        config=config,
        served_vessels=served_vessels,
        accepted_requests=accepted_requests,
        rejected_requests=rejected_requests,
    )["total"]


def compute_coordinator_reward_step(
    ports: list[PortState],
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
    delay_hours: float = 0.0,
    schedule_delay_hours: float = 0.0,
    served_vessels: float = 0.0,
    accepted_requests: float = 0.0,
    rejected_requests: float = 0.0,
) -> float:
    """System-level coordinator reward using step-level deltas.

    Returns ``step_accept_reward + step_served_reward + utilization_reward
    - step_reject_penalty - (fuel_weight * Δfuel_total + queue_weight * avg_queue
    + coordinator_idle_dock_weight * avg_idle_docks
    + coordinator_delay_weight * Δdelay + coordinator_emission_weight * Δco2_total)``.
    The coordinator objective is intentionally flow-oriented: it still
    penalizes costly routing, but does not double-count fuel and emissions
    as aggressively as the vessel reward. This keeps the coordinator focused
    on berth usage and schedule quality instead of collapsing into overly
    conservative dispatching.
    """
    return compute_coordinator_reward_breakdown(
        ports=ports,
        config=config,
        fuel_used=fuel_used,
        co2_emitted=co2_emitted,
        delay_hours=delay_hours,
        schedule_delay_hours=schedule_delay_hours,
        served_vessels=served_vessels,
        accepted_requests=accepted_requests,
        rejected_requests=rejected_requests,
    )["total"]


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
