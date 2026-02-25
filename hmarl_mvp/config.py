"""Project configuration and constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

SEED = 42


@dataclass(frozen=True)
class HMARLConfig:
    """Typed project configuration with built-in validation."""

    # Fleet topology
    num_ports: int = 5
    num_vessels: int = 8
    num_coordinators: int = 1
    docks_per_port: int = 3
    # Forecast horizons
    medium_horizon_days: int = 5
    short_horizon_hours: int = 12
    # Asynchronous cadence (simulation steps)
    coord_decision_interval_steps: int = 12
    vessel_decision_interval_steps: int = 1
    port_decision_interval_steps: int = 1
    message_latency_steps: int = 1
    # Reward weights (alpha, beta, gamma, lambda)
    fuel_weight: float = 1.0
    delay_weight: float = 1.5
    emission_weight: float = 0.7
    emission_lambda: float = 2.0
    dock_idle_weight: float = 0.5
    # Physics
    fuel_rate_coeff: float = 0.002
    emission_factor: float = 3.114
    speed_min: float = 8.0
    speed_max: float = 18.0
    nominal_speed: float = 12.0
    initial_fuel: float = 100.0
    service_time_hours: float = 6.0
    # Economic parameters (RQ4)
    cargo_value_per_vessel: float = 1_000_000.0
    fuel_price_per_ton: float = 600.0
    delay_penalty_per_hour: float = 5_000.0
    carbon_price_per_ton: float = 90.0
    # Simulation
    rollout_steps: int = 20
    seed: int = SEED

    def validate(self) -> None:
        """Validate logical and numerical consistency."""
        int_fields = {
            "num_ports": self.num_ports,
            "num_vessels": self.num_vessels,
            "num_coordinators": self.num_coordinators,
            "docks_per_port": self.docks_per_port,
            "medium_horizon_days": self.medium_horizon_days,
            "short_horizon_hours": self.short_horizon_hours,
            "coord_decision_interval_steps": self.coord_decision_interval_steps,
            "vessel_decision_interval_steps": self.vessel_decision_interval_steps,
            "port_decision_interval_steps": self.port_decision_interval_steps,
            "message_latency_steps": self.message_latency_steps,
            "rollout_steps": self.rollout_steps,
            "seed": self.seed,
        }
        for name, value in int_fields.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")

        positive_int_fields = {
            "num_ports": self.num_ports,
            "num_vessels": self.num_vessels,
            "num_coordinators": self.num_coordinators,
            "docks_per_port": self.docks_per_port,
            "medium_horizon_days": self.medium_horizon_days,
            "short_horizon_hours": self.short_horizon_hours,
            "coord_decision_interval_steps": self.coord_decision_interval_steps,
            "vessel_decision_interval_steps": self.vessel_decision_interval_steps,
            "port_decision_interval_steps": self.port_decision_interval_steps,
            "message_latency_steps": self.message_latency_steps,
            "rollout_steps": self.rollout_steps,
        }
        for name, value in positive_int_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

        non_negative_float_fields = {
            "fuel_weight": self.fuel_weight,
            "delay_weight": self.delay_weight,
            "emission_weight": self.emission_weight,
            "emission_lambda": self.emission_lambda,
            "dock_idle_weight": self.dock_idle_weight,
            "fuel_price_per_ton": self.fuel_price_per_ton,
            "delay_penalty_per_hour": self.delay_penalty_per_hour,
            "carbon_price_per_ton": self.carbon_price_per_ton,
            "cargo_value_per_vessel": self.cargo_value_per_vessel,
        }
        for field_name, field_value in non_negative_float_fields.items():
            if field_value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {field_value}")

        positive_float_fields = {
            "fuel_rate_coeff": self.fuel_rate_coeff,
            "emission_factor": self.emission_factor,
            "speed_min": self.speed_min,
            "speed_max": self.speed_max,
            "nominal_speed": self.nominal_speed,
            "initial_fuel": self.initial_fuel,
            "service_time_hours": self.service_time_hours,
        }
        for field_name, field_value in positive_float_fields.items():
            if field_value <= 0:
                raise ValueError(f"{field_name} must be > 0, got {field_value}")

        if self.speed_min > self.speed_max:
            raise ValueError(
                f"speed_min ({self.speed_min}) must be <= speed_max ({self.speed_max})"
            )
        if not (self.speed_min <= self.nominal_speed <= self.speed_max):
            raise ValueError(
                "nominal_speed must be within [speed_min, speed_max], got "
                f"{self.nominal_speed} not in [{self.speed_min}, {self.speed_max}]"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation for runtime code."""
        return asdict(self)

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


def _build_validated_config(overrides: Mapping[str, Any]) -> dict[str, Any]:
    base = HMARLConfig().to_dict()
    unknown_keys = sorted(set(overrides.keys()) - set(base.keys()))
    if unknown_keys:
        raise KeyError(f"Unknown config keys: {unknown_keys}")
    merged = dict(base)
    merged.update(dict(overrides))
    cfg = HMARLConfig(**merged)
    cfg.validate()
    return cfg.to_dict()


DEFAULT_CONFIG: dict[str, Any] = _build_validated_config({})


def validate_distance_matrix(distance_nm: np.ndarray, num_ports: int) -> np.ndarray:
    """Validate and normalize a distance matrix for the configured topology."""
    matrix = np.asarray(distance_nm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("distance_nm must be a 2D matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_nm must be square")
    if matrix.shape[0] != num_ports:
        raise ValueError(
            f"distance_nm shape {matrix.shape} does not match num_ports={num_ports}"
        )
    if np.any(matrix < 0):
        raise ValueError("distance_nm must be non-negative")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("distance_nm diagonal must be all zeros")
    return matrix


def generate_distance_matrix(num_ports: int) -> np.ndarray:
    """Create a deterministic symmetric distance matrix for arbitrary port counts."""
    if num_ports <= 0:
        raise ValueError("num_ports must be >= 1")
    idx = np.arange(num_ports)
    ring_distance = np.abs(np.subtract.outer(idx, idx)).astype(float)
    ring_distance = np.minimum(ring_distance, num_ports - ring_distance)
    variation = (np.add.outer(idx, idx) % 4).astype(float) * 200.0
    matrix = 1200.0 + 900.0 * ring_distance + variation
    np.fill_diagonal(matrix, 0.0)
    return matrix


def resolve_distance_matrix(
    num_ports: int,
    distance_nm: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return a validated distance matrix that matches the configured topology.

    Uses the fixed 5-port matrix for default topology and generates a deterministic
    matrix for any other port count.
    """
    if distance_nm is not None:
        return validate_distance_matrix(distance_nm, num_ports)
    if num_ports == DISTANCE_NM.shape[0]:
        return DISTANCE_NM.copy()
    return generate_distance_matrix(num_ports)


def get_default_config(**overrides: Any) -> dict[str, Any]:
    """Return a copy of the default config with optional overrides."""
    return _build_validated_config(overrides)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a config mapping against the typed schema."""
    return _build_validated_config(config)
