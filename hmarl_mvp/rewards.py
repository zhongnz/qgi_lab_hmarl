"""Reward functions for coordinator, vessel, and port agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dynamics import compute_fuel_and_emissions
from .state import PortState, VesselState


def compute_vessel_reward(vessel: VesselState, config: dict[str, Any]) -> float:
    """Per-step vessel reward as negative weighted cost."""
    fuel_used, co2 = compute_fuel_and_emissions(vessel.speed, config, hours=1.0)
    fuel_cost = config["fuel_weight"] * fuel_used
    delay_cost = config["delay_weight"] * vessel.delay_hours
    emission_cost = config["emission_weight"] * co2
    return -(fuel_cost + delay_cost + emission_cost)


def compute_port_reward(port: PortState, config: dict[str, Any]) -> float:
    """Per-step port reward as negative queue + idle penalty."""
    queue_penalty = float(port.queue)
    idle_docks = max(port.docks - port.occupied, 0)
    idle_penalty = config["dock_idle_weight"] * idle_docks
    return -(queue_penalty + idle_penalty)


def compute_coordinator_reward(
    vessels: list[VesselState],
    ports: list[PortState],
    config: dict[str, Any],
) -> float:
    """System-level coordinator reward."""
    total_fuel_cost = sum(
        compute_fuel_and_emissions(v.speed, config, 1.0)[0] for v in vessels
    )
    total_emissions = sum(v.emissions for v in vessels)
    avg_queue = float(np.mean([p.queue for p in ports])) if ports else 0.0
    emission_penalty = config["emission_lambda"] * total_emissions
    voyage_cost = total_fuel_cost + avg_queue
    return -(voyage_cost + emission_penalty)

