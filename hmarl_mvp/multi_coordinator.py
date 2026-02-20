"""Scaffolding utilities for multi-coordinator experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .policies import fleet_coordinator_policy
from .state import VesselState


def assign_vessels_to_coordinators(
    vessels: list[VesselState],
    num_coordinators: int,
) -> dict[int, list[int]]:
    """
    Partition vessels into coordinator groups.

    Current strategy:
    - Docked vessels: assignment by current location modulo coordinator count.
    - In-transit vessels: assignment by vessel id modulo coordinator count.
    """
    if num_coordinators <= 0:
        raise ValueError("num_coordinators must be >= 1")

    groups: dict[int, list[int]] = {i: [] for i in range(num_coordinators)}
    for vessel in vessels:
        if vessel.location >= 0:
            coordinator_id = int(vessel.location % num_coordinators)
        else:
            coordinator_id = int(vessel.vessel_id % num_coordinators)
        groups[coordinator_id].append(vessel.vessel_id)
    return groups


def build_multi_coordinator_directives(
    medium_forecast: np.ndarray,
    vessels: list[VesselState],
    num_coordinators: int,
) -> list[dict[str, Any]]:
    """
    Build one strategic directive per coordinator.

    This is a light wrapper over the current single-coordinator policy to keep
    the code path modular while preserving existing behavior.
    """
    assignments = assign_vessels_to_coordinators(vessels, num_coordinators)
    directives: list[dict[str, Any]] = []
    vessels_by_id = {v.vessel_id: v for v in vessels}

    for coordinator_id in range(num_coordinators):
        local_ids = assignments.get(coordinator_id, [])
        local_vessels = [vessels_by_id[i] for i in local_ids if i in vessels_by_id]
        if not local_vessels:
            # Fallback keeps output shape stable even when a partition is empty.
            local_vessels = vessels
        directive = fleet_coordinator_policy(medium_forecast, local_vessels)
        directive["coordinator_id"] = coordinator_id
        directive["assigned_vessel_ids"] = local_ids
        directives.append(directive)
    return directives

