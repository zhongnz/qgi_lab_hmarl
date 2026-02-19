"""Project configuration and constants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

SEED = 42

DEFAULT_CONFIG: dict[str, Any] = {
    # Fleet topology
    "num_ports": 5,
    "num_vessels": 8,
    "docks_per_port": 3,
    # Forecast horizons
    "medium_horizon_days": 5,
    "short_horizon_hours": 12,
    # Reward weights (alpha, beta, gamma, lambda)
    "fuel_weight": 1.0,
    "delay_weight": 1.5,
    "emission_weight": 0.7,
    "emission_lambda": 2.0,
    "dock_idle_weight": 0.5,
    # Physics
    "fuel_rate_coeff": 0.002,
    "emission_factor": 3.114,
    "speed_min": 8.0,
    "speed_max": 18.0,
    "nominal_speed": 12.0,
    # Economic parameters (RQ4)
    "cargo_value_per_vessel": 1_000_000,
    "fuel_price_per_ton": 600,
    "delay_penalty_per_hour": 5_000,
    "carbon_price_per_ton": 90,
    # Simulation
    "rollout_steps": 20,
    "seed": SEED,
}

# Distance matrix (nautical miles) between 5 ports.
DISTANCE_NM = np.array(
    [
        [0, 8400, 2200, 3400, 7800],
        [8400, 0, 9800, 6100, 5400],
        [2200, 9800, 0, 5500, 5900],
        [3400, 6100, 5500, 0, 8200],
        [7800, 5400, 5900, 8200, 0],
    ],
    dtype=float,
)


def get_default_config(**overrides: Any) -> dict[str, Any]:
    """Return a copy of the default config with optional overrides."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg

