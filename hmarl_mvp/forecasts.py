"""Forecast generators for medium-term and short-term congestion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics import step_ports, step_vessels
from .state import PortState, VesselState


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


@dataclass
class GroundTruthForecaster:
    """Deterministic forecast from the currently committed system state.

    The forecast rolls forward the existing queues, ongoing service, and vessels
    that are already at sea or committed to depart. It intentionally does not
    invent new future requests or re-routing decisions, which keeps it causal
    and cheap enough for debugging and training ablations.
    """

    medium_horizon_days: int
    short_horizon_hours: int
    config: dict[str, Any]
    distance_nm: np.ndarray

    def _apply_arrivals(
        self,
        *,
        ports: list[PortState],
        vessels: list[VesselState],
        step_stats: dict[int, dict[str, float | bool]],
    ) -> None:
        """Mirror environment arrival semantics onto the copied forecast state."""
        single_mission = str(self.config.get("episode_mode", "continuous")) == "single_mission"
        mission_success_on = str(self.config.get("mission_success_on", "arrival"))
        for vessel in vessels:
            stats = step_stats.get(vessel.vessel_id, {})
            if not bool(stats.get("arrived", False)):
                continue
            destination = int(vessel.location)
            if single_mission and mission_success_on == "arrival":
                vessel.mission_done = True
                vessel.mission_success = True
                vessel.mission_failed = False
                vessel.pending_departure = False
                vessel.pending_requested_arrival_time = 0.0
                vessel.requested_arrival_time = 0.0
                continue
            if 0 <= destination < len(ports):
                port = ports[destination]
                port.queue += 1
                port.queued_vessel_ids.append(int(vessel.vessel_id))

    def _apply_service_completion(
        self,
        *,
        ports: list[PortState],
        vessels: list[VesselState],
        completed_by_port: list[list[int]],
    ) -> None:
        """Mirror the subset of service-completion semantics relevant to forecasting."""
        single_mission = str(self.config.get("episode_mode", "continuous")) == "single_mission"
        mission_success_on = str(self.config.get("mission_success_on", "arrival"))
        for completed in completed_by_port:
            for vessel_id in completed:
                if not (0 <= int(vessel_id) < len(vessels)):
                    continue
                vessel = vessels[int(vessel_id)]
                if single_mission and mission_success_on == "service_complete":
                    vessel.mission_done = True
                    vessel.mission_success = True
                    vessel.mission_failed = False
                    vessel.pending_departure = False
                    vessel.pending_requested_arrival_time = 0.0
                    vessel.requested_arrival_time = 0.0
                else:
                    vessel.fuel = float(getattr(vessel, "initial_fuel", vessel.fuel))

    def predict(
        self,
        ports: list[PortState],
        vessels: list[VesselState],
        current_step: int,
        weather: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project future queue levels from the currently committed state only."""
        max_horizon = max(int(self.medium_horizon_days), int(self.short_horizon_hours))
        current_q = np.array([p.queue for p in ports], dtype=float)[:, None]
        if max_horizon <= 0:
            return current_q.copy(), current_q.copy()

        sim_ports = deepcopy(ports)
        sim_vessels = deepcopy(vessels)
        dt_hours = float(self.config.get("dt_hours", 1.0))
        service_time_hours = float(self.config.get("service_time_hours", 6.0))
        queue_history: list[np.ndarray] = []

        for step_offset in range(max_horizon):
            sim_step = int(current_step) + step_offset
            step_stats = step_vessels(
                sim_vessels,
                distance_nm=self.distance_nm,
                config=self.config,
                dt_hours=dt_hours,
                weather=weather,
                current_step=sim_step,
            )
            self._apply_arrivals(ports=sim_ports, vessels=sim_vessels, step_stats=step_stats)
            completed_by_port = step_ports(
                sim_ports,
                service_rates=[int(port.docks) for port in sim_ports],
                dt_hours=dt_hours,
                service_time_hours=service_time_hours,
            )
            self._apply_service_completion(
                ports=sim_ports,
                vessels=sim_vessels,
                completed_by_port=completed_by_port,
            )
            queue_history.append(np.array([port.queue for port in sim_ports], dtype=float))

        history = np.stack(queue_history, axis=1)
        medium = history[:, : int(self.medium_horizon_days)]
        short = history[:, : int(self.short_horizon_hours)]
        if medium.shape[1] < int(self.medium_horizon_days):
            medium = np.pad(
                medium,
                ((0, 0), (0, int(self.medium_horizon_days) - medium.shape[1])),
                mode="edge",
            )
        if short.shape[1] < int(self.short_horizon_hours):
            short = np.pad(
                short,
                ((0, 0), (0, int(self.short_horizon_hours) - short.shape[1])),
                mode="edge",
            )
        return medium, short
