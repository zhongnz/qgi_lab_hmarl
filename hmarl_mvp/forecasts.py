"""Forecast generators for medium-term and short-term congestion."""

from __future__ import annotations

import numpy as np

from .state import PortState


def medium_term_forecast(
    num_ports: int,
    horizon_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mock strategic forecast with trend + noise."""
    base = rng.uniform(2, 8, size=(num_ports, 1))
    trend = np.linspace(0, 1.5, horizon_days)[None, :]
    noise = rng.normal(0, 0.3, size=(num_ports, horizon_days))
    return np.clip(base + trend + noise, 0, None)


def short_term_forecast(
    num_ports: int,
    horizon_hours: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mock operational forecast with local noise."""
    base = rng.uniform(1, 6, size=(num_ports, 1))
    noise = rng.normal(0, 0.5, size=(num_ports, horizon_hours))
    return np.clip(base + noise, 0, None)


def oracle_forecasts(
    ports: list[PortState],
    medium_horizon_days: int,
    short_horizon_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic oracle-like forecast from current realized queue state."""
    current_q = np.array([p.queue for p in ports], dtype=float)[:, None]
    medium = np.repeat(current_q, medium_horizon_days, axis=1)
    short = np.repeat(current_q, short_horizon_hours, axis=1)
    return medium, short

