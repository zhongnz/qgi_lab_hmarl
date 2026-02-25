"""Forecast generators for medium-term and short-term congestion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import PortState


@dataclass
class MediumTermForecaster:
    """State-dependent strategic forecaster (3-7 day horizon)."""

    horizon_days: int

    def predict(self, ports: list[PortState], rng: np.random.Generator) -> np.ndarray:
        """State-dependent strategic forecast: current queue + trend + noise."""
        current_q = np.array([p.queue for p in ports], dtype=float)[:, None]
        trend = np.linspace(0, 1.5, self.horizon_days)[None, :]
        noise = rng.normal(0, 0.3, size=(len(ports), self.horizon_days))
        return np.clip(current_q + trend + noise, 0, None)


@dataclass
class ShortTermForecaster:
    """State-dependent operational forecaster (6-24 hour horizon)."""

    horizon_hours: int

    def predict(self, ports: list[PortState], rng: np.random.Generator) -> np.ndarray:
        """State-dependent operational forecast: current queue + noise."""
        current_q = np.array([p.queue for p in ports], dtype=float)[:, None]
        noise = rng.normal(0, 0.5, size=(len(ports), self.horizon_hours))
        return np.clip(current_q + noise, 0, None)


@dataclass
class OracleForecaster:
    """Oracle forecaster derived from realized queue state."""

    medium_horizon_days: int
    short_horizon_hours: int

    def predict(self, ports: list[PortState]) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic oracle-like forecast from current realized queue state."""
        current_q = np.array([p.queue for p in ports], dtype=float)[:, None]
        medium = np.repeat(current_q, self.medium_horizon_days, axis=1)
        short = np.repeat(current_q, self.short_horizon_hours, axis=1)
        return medium, short
