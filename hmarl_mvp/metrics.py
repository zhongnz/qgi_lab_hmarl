"""Metrics for forecasting, operations, coordination, and economics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import PortState, VesselState


def forecast_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def forecast_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def compute_vessel_metrics(vessels: list[VesselState]) -> dict[str, float]:
    avg_speed = float(np.mean([v.speed for v in vessels])) if vessels else 0.0
    avg_fuel = float(np.mean([v.fuel for v in vessels])) if vessels else 0.0
    total_fuel_used = float(sum(100.0 - v.fuel for v in vessels))
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


def compute_coordination_metrics(
    vessel_actions: list[dict[str, Any]],
    port_actions: list[dict[str, Any]],
) -> dict[str, float]:
    total_requests = sum(1 for a in vessel_actions if a.get("request_arrival_slot", False))
    total_accepted = sum(int(a.get("accept_requests", 0)) for a in port_actions)
    agreement_rate = total_accepted / total_requests if total_requests > 0 else 0.0
    return {
        "total_vessel_requests": float(total_requests),
        "total_port_accepted": float(total_accepted),
        "policy_agreement_rate": agreement_rate,
    }


def compute_economic_metrics(
    vessels: list[VesselState],
    config: dict[str, Any],
) -> dict[str, float]:
    fuel_cost_total = sum((100.0 - v.fuel) * config["fuel_price_per_ton"] for v in vessels)
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

