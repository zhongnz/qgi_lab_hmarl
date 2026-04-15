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
    # Each port has docks_per_port berths.  Default is 1 to create genuine
    # congestion pressure (5 docks for 8 vessels, guaranteeing queuing).
    # Previous defaults of 2 or 3 resulted in near-zero queue lengths and
    # dock utilisation of ~0.12, leaving no congestion signal to learn from.
    docks_per_port: int = 1
    # Forecast horizons
    medium_horizon_days: int = 5
    short_horizon_hours: int = 12
    forecast_source: str = "ground_truth"
    # Asynchronous cadence (simulation steps)
    coord_decision_interval_steps: int = 12
    vessel_decision_interval_steps: int = 1
    port_decision_interval_steps: int = 2
    message_latency_steps: int = 1
    # Reward weights (alpha, beta, gamma, lambda)
    # Vessel reward = -fuel_weight*fuel - delay_weight*delay - emission_weight*co2
    #                 - transit_time_weight*transit - schedule_delay_weight*schedule_delay
    #                 + arrival_reward*arrived + on_time_arrival_reward*on_time
    # transit_time_weight is intentionally high (8.0) to strongly discourage
    # fuel-wasting detours; schedule_delay_weight mirrors it for deadline adherence.
    fuel_weight: float = 1.0
    delay_weight: float = 1.5
    # emission_weight was 0.7 but emission_factor=3.114 means the effective
    # penalty on the same fuel burn is 2.18× fuel_weight — a double penalty.
    # Set to 0.0; fuel_weight alone captures consumption cost.  Coordinator
    # retains a small emission_weight for system-level carbon signal.
    emission_weight: float = 0.0
    # Transit cost per hour at sea.  At nominal speed (12 kn) fuel_cost ≈
    # 3.5/hr, so 2.5 keeps a meaningful travel disincentive without the
    # previous 2.3× dominance over fuel.
    transit_time_weight: float = 2.5
    arrival_reward: float = 15.0
    on_time_arrival_reward: float = 20.0
    schedule_delay_weight: float = 3.0
    dock_idle_weight: float = 0.5
    port_accept_reward: float = 1.5
    port_reject_penalty: float = 1.0
    port_service_reward: float = 2.0
    coordinator_fuel_weight: float = 0.25
    coordinator_queue_weight: float = 0.75
    coordinator_emission_weight: float = 0.2
    coordinator_delay_weight: float = 3.0
    coordinator_schedule_delay_weight: float = 4.0
    coordinator_accept_reward: float = 2.0
    coordinator_reject_penalty: float = 2.0
    coordinator_service_reward: float = 6.0
    coordinator_idle_dock_weight: float = 0.0  # redundant with utilization_reward (idle = docks - occupied)
    coordinator_utilization_reward: float = 2.0
    coordinator_queue_imbalance_weight: float = 0.5  # penalise std(queue) across ports (anti-herding)
    # Directive compliance: bonus when vessels follow coordinator's per-vessel
    # destination assignment.  Closes the credit-assignment gap between
    # coordinator actions and vessel behaviour.
    coordinator_compliance_weight: float = 1.5
    on_time_tolerance_hours: float = 2.0
    # Tightened from 1.0 to 0.25 so that schedule_delay_weight (8.0) is
    # actually triggered — the old 1.0h slack was too generous and made
    # the schedule_delay component effectively dormant.
    requested_arrival_slack_hours: float = 0.25
    # Physics
    # fuel_rate_coeff: fuel consumption in tonnes per (knot^3 * hour).
    # Cubic speed-fuel law approximates propeller hydrodynamics (Harvald 1983).
    fuel_rate_coeff: float = 0.002
    # emission_factor: CO2 tonnes emitted per tonne of fuel burned.
    # Based on IMO EEDI guidelines for HFO (Heavy Fuel Oil): 3.114 t-CO2/t-fuel.
    emission_factor: float = 3.114
    speed_min: float = 8.0   # minimum safe speed (knots)
    speed_max: float = 18.0  # maximum design speed (knots)
    nominal_speed: float = 12.0  # economic cruising speed (knots)
    initial_fuel: float = 100.0  # fuel capacity per vessel (tonnes)
    service_time_hours: float = 6.0  # berth service duration (hours)
    dt_hours: float = 1.0  # simulation time step (hours)
    # Weather (on by default for realistic stochastic fuel costs)
    weather_enabled: bool = True
    sea_state_max: float = 3.0
    weather_penalty_factor: float = 0.15
    weather_autocorrelation: float = 0.7
    weather_shaping_weight: float = 0.3
    port_weather_features: bool = True
    coordinator_departure_window_options: tuple[int, ...] = (0,)
    # Economic parameters (RQ4)
    cargo_value_per_vessel: float = 1_000_000.0
    fuel_price_per_ton: float = 600.0
    delay_penalty_per_hour: float = 5_000.0
    carbon_price_per_ton: float = 90.0
    # Simulation
    rollout_steps: int = 138
    episode_mode: str = "continuous"
    mission_success_on: str = "arrival"
    seed: int = SEED

    def validate(self) -> None:
        """Validate logical and numerical consistency.

        Each field is checked against one constraint:
        - ``"int>=1"`` — must be int ≥ 1
        - ``"int>=0"`` — must be int ≥ 0
        - ``"float>=0"`` — must be float/int ≥ 0
        - ``"float>0"`` — must be float/int > 0
        """
        _rules: dict[str, str] = {
            # Fleet topology
            "num_ports": "int>=1",
            "num_vessels": "int>=1",
            "num_coordinators": "int>=1",
            "docks_per_port": "int>=1",
            # Forecast horizons
            "medium_horizon_days": "int>=1",
            "short_horizon_hours": "int>=1",
            # Asynchronous cadence
            "coord_decision_interval_steps": "int>=1",
            "vessel_decision_interval_steps": "int>=1",
            "port_decision_interval_steps": "int>=1",
            "message_latency_steps": "int>=1",
            # Simulation
            "rollout_steps": "int>=1",
            "seed": "int>=0",
            # Reward weights
            "fuel_weight": "float>=0",
            "delay_weight": "float>=0",
            "emission_weight": "float>=0",
            "transit_time_weight": "float>=0",
            "arrival_reward": "float>=0",
            "on_time_arrival_reward": "float>=0",
            "schedule_delay_weight": "float>=0",
            "dock_idle_weight": "float>=0",
            "port_accept_reward": "float>=0",
            "port_reject_penalty": "float>=0",
            "port_service_reward": "float>=0",
            "coordinator_fuel_weight": "float>=0",
            "coordinator_queue_weight": "float>=0",
            "coordinator_emission_weight": "float>=0",
            "coordinator_delay_weight": "float>=0",
            "coordinator_schedule_delay_weight": "float>=0",
            "coordinator_accept_reward": "float>=0",
            "coordinator_reject_penalty": "float>=0",
            "coordinator_service_reward": "float>=0",
            "coordinator_idle_dock_weight": "float>=0",
            "coordinator_utilization_reward": "float>=0",
            "coordinator_compliance_weight": "float>=0",
            "on_time_tolerance_hours": "float>=0",
            "requested_arrival_slack_hours": "float>=0",
            # Economic parameters
            "fuel_price_per_ton": "float>=0",
            "delay_penalty_per_hour": "float>=0",
            "carbon_price_per_ton": "float>=0",
            "cargo_value_per_vessel": "float>=0",
            # Physics
            "fuel_rate_coeff": "float>0",
            "emission_factor": "float>0",
            "speed_min": "float>0",
            "speed_max": "float>0",
            "nominal_speed": "float>0",
            "initial_fuel": "float>0",
            "service_time_hours": "float>0",
            "dt_hours": "float>0",
            # Weather
            "sea_state_max": "float>0",
            "weather_penalty_factor": "float>=0",
            "weather_autocorrelation": "float>=0",
            "weather_shaping_weight": "float>=0",
        }
        for name, rule in _rules.items():
            value = getattr(self, name)
            if rule.startswith("int"):
                if not isinstance(value, int):
                    raise TypeError(f"{name} must be int, got {type(value).__name__}")
                bound = int(rule.split(">=")[1])
                if value < bound:
                    raise ValueError(f"{name} must be >= {bound}, got {value}")
            elif rule == "float>=0":
                if value < 0:
                    raise ValueError(f"{name} must be >= 0, got {value}")
            elif rule == "float>0":
                if value <= 0:
                    raise ValueError(f"{name} must be > 0, got {value}")

        if self.speed_min > self.speed_max:
            raise ValueError(
                f"speed_min ({self.speed_min}) must be <= speed_max ({self.speed_max})"
            )
        valid_forecast_sources = {
            "heuristic", "noiseless_current", "ground_truth", "learned_forecast",
        }
        if self.forecast_source not in valid_forecast_sources:
            raise ValueError(
                f"forecast_source must be one of {valid_forecast_sources}, "
                f"got {self.forecast_source!r}"
            )
        if self.episode_mode not in {"continuous", "single_mission"}:
            raise ValueError(
                "episode_mode must be one of {'continuous', 'single_mission'}, "
                f"got {self.episode_mode!r}"
            )
        if self.mission_success_on not in {"arrival", "service_complete"}:
            raise ValueError(
                "mission_success_on must be one of {'arrival', 'service_complete'}, "
                f"got {self.mission_success_on!r}"
            )
        if self.weather_autocorrelation > 1.0:
            raise ValueError(
                "weather_autocorrelation must be <= 1.0, "
                f"got {self.weather_autocorrelation}"
            )
        if not (self.speed_min <= self.nominal_speed <= self.speed_max):
            raise ValueError(
                "nominal_speed must be within [speed_min, speed_max], got "
                f"{self.nominal_speed} not in [{self.speed_min}, {self.speed_max}]"
            )
        # Reward weight sanity: at least one reward weight should be positive
        reward_weights = [
            self.fuel_weight, self.delay_weight, self.emission_weight,
            self.arrival_reward, self.on_time_arrival_reward,
            self.schedule_delay_weight, self.transit_time_weight,
        ]
        if all(w == 0.0 for w in reward_weights):
            raise ValueError(
                "All vessel reward weights are zero — training will produce no learning signal"
            )
        if not isinstance(self.coordinator_departure_window_options, (list, tuple)):
            raise TypeError("coordinator_departure_window_options must be a list/tuple")
        if len(self.coordinator_departure_window_options) == 0:
            raise ValueError("coordinator_departure_window_options must be non-empty")
        for w in self.coordinator_departure_window_options:
            if int(w) < 0:
                raise ValueError(
                    "coordinator_departure_window_options must contain non-negative hours"
                )
        # Cross-parameter feasibility checks
        if self.num_vessels < self.num_coordinators:
            raise ValueError(
                f"num_vessels ({self.num_vessels}) must be >= num_coordinators "
                f"({self.num_coordinators}) — coordinator assignment requires at least "
                f"one vessel per coordinator"
            )
        if self.rollout_steps < self.coord_decision_interval_steps:
            import warnings

            warnings.warn(
                f"rollout_steps ({self.rollout_steps}) < "
                f"coord_decision_interval_steps ({self.coord_decision_interval_steps}) "
                f"— coordinator will never execute a decision step",
                UserWarning,
                stacklevel=2,
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation for runtime code."""
        return asdict(self)

# Distance matrix (nautical miles) between 5 ports.
#
# Values are scaled for the simulator's hourly time step and default
# episode lengths so multiple voyages can complete within one rollout.
DISTANCE_NM = np.array(
    [
        [0, 84, 22, 34, 78],
        [84, 0, 98, 61, 54],
        [22, 98, 0, 55, 59],
        [34, 61, 55, 0, 82],
        [78, 54, 59, 82, 0],
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
    variation = (np.add.outer(idx, idx) % 4).astype(float) * 4.0
    matrix = 24.0 + 18.0 * ring_distance + variation
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


# ---------------------------------------------------------------------------
# Decision cadence
# ---------------------------------------------------------------------------


def should_update(step: int, interval: int) -> bool:
    """Return True when an agent with *interval* is due at *step*."""
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
        """Construct from a validated config dictionary."""
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
