"""State dataclasses and initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SEED


@dataclass
class PortState:
    """State of a single port."""

    port_id: int
    queue: int
    docks: int
    occupied: int
    service_times: list[float] = field(default_factory=list)
    cumulative_wait_hours: float = 0.0
    vessels_served: int = 0


@dataclass
class VesselState:
    """State of a single vessel."""

    vessel_id: int
    location: int
    destination: int
    position_nm: float = 0.0
    speed: float = 12.0
    fuel: float = 100.0
    initial_fuel: float = 100.0
    emissions: float = 0.0
    delay_hours: float = 0.0
    at_sea: bool = False


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Create a fresh RNG for deterministic experiments."""
    return np.random.default_rng(seed)


def initialize_ports(
    num_ports: int,
    docks_per_port: int,
    rng: np.random.Generator,
    service_time_hours: float = 6.0,
) -> list[PortState]:
    """Initialize random port states."""
    ports: list[PortState] = []
    for i in range(num_ports):
        occupied = int(rng.integers(0, docks_per_port))
        ports.append(
            PortState(
                port_id=i,
                queue=int(rng.integers(0, 5)),
                docks=docks_per_port,
                occupied=occupied,
                service_times=[float(service_time_hours) for _ in range(occupied)],
            )
        )
    return ports


def initialize_vessels(
    num_vessels: int,
    num_ports: int,
    nominal_speed: float,
    rng: np.random.Generator,
    initial_fuel: float = 100.0,
) -> list[VesselState]:
    """Initialize random vessel states."""
    return [
        VesselState(
            vessel_id=i,
            location=int(rng.integers(0, num_ports)),
            destination=int(rng.integers(0, num_ports)),
            speed=nominal_speed,
            fuel=initial_fuel,
            initial_fuel=initial_fuel,
        )
        for i in range(num_vessels)
    ]
