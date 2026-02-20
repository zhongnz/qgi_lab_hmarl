"""Asynchronous decision cadence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def should_update(step: int, interval: int) -> bool:
    """Return True when an agent with `interval` is due at `step`."""
    if interval <= 0:
        raise ValueError("interval must be >= 1")
    if step < 0:
        raise ValueError("step must be >= 0")
    return step % interval == 0


@dataclass(frozen=True)
class DecisionCadence:
    """Cadence for coordinator, vessel, and port decision loops."""

    coordinator_steps: int
    vessel_steps: int
    port_steps: int
    message_latency_steps: int = 1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DecisionCadence":
        return cls(
            coordinator_steps=max(1, int(config["coord_decision_interval_steps"])),
            vessel_steps=max(1, int(config["vessel_decision_interval_steps"])),
            port_steps=max(1, int(config["port_decision_interval_steps"])),
            message_latency_steps=max(1, int(config.get("message_latency_steps", 1))),
        )

    def due(self, step: int) -> dict[str, bool]:
        """Return which agent classes should update at this step."""
        return {
            "coordinator": should_update(step, self.coordinator_steps),
            "vessel": should_update(step, self.vessel_steps),
            "port": should_update(step, self.port_steps),
        }

