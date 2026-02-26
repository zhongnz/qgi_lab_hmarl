"""Metrics for forecasting, operations, coordination, and economics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import PortState, VesselState


def forecast_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error between predicted and actual arrays."""
    return float(np.mean(np.abs(predicted - actual)))


def forecast_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error between predicted and actual arrays."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def compute_vessel_metrics(vessels: list[VesselState]) -> dict[str, float]:
    """Aggregate fleet-level metrics: speed, fuel, emissions, delay, on-time rate."""
    avg_speed = float(np.mean([v.speed for v in vessels])) if vessels else 0.0
    avg_fuel = float(np.mean([v.fuel for v in vessels])) if vessels else 0.0
    total_fuel_used = float(sum(max(v.initial_fuel - v.fuel, 0.0) for v in vessels))
    total_emissions = float(sum(v.emissions for v in vessels))
    avg_delay = float(np.mean([v.delay_hours for v in vessels])) if vessels else 0.0
    on_time_count = sum(1 for v in vessels if v.delay_hours < 2.0)
    on_time_rate = on_time_count / len(vessels) if vessels else 0.0
    return {
        "avg_speed": avg_speed,
        "avg_fuel_remaining": avg_fuel,
        "total_fuel_used": total_fuel_used,
        "total_emissions_co2": total_emissions,
        "avg_delay_hours": avg_delay,
        "on_time_rate": on_time_rate,
    }


def compute_port_metrics(ports: list[PortState]) -> dict[str, float]:
    """Aggregate port-level metrics: queue, utilization, wait, throughput."""
    avg_queue = float(np.mean([p.queue for p in ports])) if ports else 0.0
    dock_util = float(np.mean([p.occupied / p.docks for p in ports])) if ports else 0.0
    total_wait = float(sum(p.cumulative_wait_hours for p in ports))
    total_served = int(sum(p.vessels_served for p in ports))
    avg_wait_per = total_wait / total_served if total_served > 0 else 0.0
    return {
        "avg_queue": avg_queue,
        "dock_utilization": dock_util,
        "total_wait_hours": total_wait,
        "total_vessels_served": total_served,
        "avg_wait_per_vessel": avg_wait_per,
    }

def compute_economic_metrics(
    vessels: list[VesselState],
    config: dict[str, Any],
) -> dict[str, float]:
    """End-of-episode economic costs: fuel, delay, carbon, and reliability."""
    fuel_cost_total = sum(
        max(v.initial_fuel - v.fuel, 0.0) * config["fuel_price_per_ton"]
        for v in vessels
    )
    delay_cost_total = sum(v.delay_hours * config["delay_penalty_per_hour"] for v in vessels)
    carbon_cost_total = sum(v.emissions * config["carbon_price_per_ton"] for v in vessels)
    total_ops_cost = fuel_cost_total + delay_cost_total + carbon_cost_total

    n_vessels = len(vessels)
    price_per_vessel = total_ops_cost / n_vessels if n_vessels > 0 else 0.0
    cargo_total = n_vessels * config["cargo_value_per_vessel"]
    reliability = 1.0 - (total_ops_cost / cargo_total) if cargo_total > 0 else 0.0

    return {
        "fuel_cost_usd": float(fuel_cost_total),
        "delay_cost_usd": float(delay_cost_total),
        "carbon_cost_usd": float(carbon_cost_total),
        "total_ops_cost_usd": float(total_ops_cost),
        "price_per_vessel_usd": float(price_per_vessel),
        "cost_reliability": float(reliability),
    }


def compute_economic_step_deltas(
    step_fuel_used: float,
    step_co2_emitted: float,
    step_delay_hours: float,
    config: dict[str, Any],
) -> dict[str, float]:
    """Per-step economic cost deltas from step-level physical quantities."""
    fuel_cost = step_fuel_used * config["fuel_price_per_ton"]
    delay_cost = step_delay_hours * config["delay_penalty_per_hour"]
    carbon_cost = step_co2_emitted * config["carbon_price_per_ton"]
    return {
        "step_fuel_cost_usd": float(fuel_cost),
        "step_delay_cost_usd": float(delay_cost),
        "step_carbon_cost_usd": float(carbon_cost),
        "step_total_ops_cost_usd": float(fuel_cost + delay_cost + carbon_cost),
    }


# ---------------------------------------------------------------------------
# Coordinator-level metrics (proposal §6.3)
# ---------------------------------------------------------------------------


def compute_coordinator_metrics(
    vessels: list[VesselState],
    config: dict[str, Any],
    distance_nm: np.ndarray | None = None,
) -> dict[str, float]:
    """Fleet-coordinator evaluation metrics from the proposal.

    * **emission_budget_compliance**: fraction of vessels whose cumulative
      emissions are within the coordinator's emission budget.
    * **route_efficiency**: ratio of straight-line (optimal) trip distance
      to actual distance implied by fuel usage, averaged across in-transit
      or arrived vessels.  Falls back to 1.0 when data is insufficient.
    * **avg_trip_duration_hours**: mean trip duration for vessels that have
      completed at least one trip (requires ``trip_start_step`` > 0 and
      current step stored via ``_current_step`` config key).
    """
    emission_budget = float(config.get("emission_budget", 50.0))
    compliant = sum(1 for v in vessels if v.emissions <= emission_budget)
    compliance = compliant / len(vessels) if vessels else 0.0

    # Route efficiency: actual fuel vs minimum theoretical fuel for straight trip
    route_efficiencies: list[float] = []
    if distance_nm is not None:
        for v in vessels:
            if v.location != v.destination and v.at_sea:
                optimal_dist = float(distance_nm[v.location, v.destination])
                actual_dist = float(v.position_nm) if v.position_nm > 0 else optimal_dist
                if actual_dist > 0:
                    route_efficiencies.append(min(optimal_dist / actual_dist, 1.0))
    avg_route_eff = float(np.mean(route_efficiencies)) if route_efficiencies else 1.0

    # Trip duration
    current_step = int(config.get("_current_step", 0))
    trip_durations: list[float] = []
    for v in vessels:
        if v.trip_start_step > 0 and not v.at_sea:
            # Arrived: trip duration = current_step - trip_start_step hours
            dur = float(current_step - v.trip_start_step)
            if dur > 0:
                trip_durations.append(dur)
    avg_trip_duration = float(np.mean(trip_durations)) if trip_durations else 0.0

    return {
        "emission_budget_compliance": float(compliance),
        "avg_route_efficiency": float(avg_route_eff),
        "avg_trip_duration_hours": float(avg_trip_duration),
    }


def compute_coordination_metrics(
    requests_submitted: int,
    requests_accepted: int,
    messages_exchanged: int,
) -> dict[str, float]:
    """Coordination quality metrics from the proposal (§6.3).

    * **policy_agreement_rate**: fraction of vessel arrival requests
      that were accepted by port agents.
    * **communication_overhead**: total inter-agent messages exchanged
      during the episode.
    """
    agreement = (
        requests_accepted / requests_submitted if requests_submitted > 0 else 0.0
    )
    return {
        "policy_agreement_rate": float(agreement),
        "communication_overhead": float(messages_exchanged),
    }

