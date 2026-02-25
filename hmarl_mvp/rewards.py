"""Reward functions for coordinator, vessel, and port agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state import PortState, VesselState


def compute_vessel_reward_step(
    vessel: VesselState,
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
    delay_hours: float,
) -> float:
    """Per-step vessel reward using step-level deltas."""
    _ = vessel
    fuel_cost = config["fuel_weight"] * float(max(fuel_used, 0.0))
    delay_cost = config["delay_weight"] * float(max(delay_hours, 0.0))
    emission_cost = config["emission_weight"] * float(max(co2_emitted, 0.0))
    return -(fuel_cost + delay_cost + emission_cost)


def compute_port_reward(port: PortState, config: dict[str, Any]) -> float:
    """Per-step port reward as negative queue + idle penalty."""
    queue_penalty = float(port.queue)
    idle_docks = max(port.docks - port.occupied, 0)
    idle_penalty = config["dock_idle_weight"] * idle_docks
    return -(queue_penalty + idle_penalty)


def compute_coordinator_reward_step(
    ports: list[PortState],
    config: dict[str, Any],
    fuel_used: float,
    co2_emitted: float,
) -> float:
    """System-level coordinator reward using step-level deltas."""
    avg_queue = float(np.mean([p.queue for p in ports])) if ports else 0.0
    voyage_cost = float(max(fuel_used, 0.0)) + avg_queue
    emission_penalty = config["emission_lambda"] * float(max(co2_emitted, 0.0))
    return -(voyage_cost + emission_penalty)
