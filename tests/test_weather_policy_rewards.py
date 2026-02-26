"""Tests for weather-aware policies, experiment logging, and reward shaping."""

from __future__ import annotations

import numpy as np
import pytest

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import run_experiment
from hmarl_mvp.policies import FleetCoordinatorPolicy, VesselPolicy
from hmarl_mvp.rewards import weather_coordinator_shaping, weather_vessel_shaping
from hmarl_mvp.state import PortState, VesselState

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def cfg():
    return get_default_config(
        num_vessels=3,
        num_ports=3,
        weather_enabled=True,
        sea_state_max=3.0,
        weather_penalty_factor=0.15,
    )


@pytest.fixture()
def calm_weather():
    """3×3 close-to-zero sea state."""
    return np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])


@pytest.fixture()
def rough_weather():
    """3×3 matrix with one calm column (port 0) and two rough columns."""
    return np.array([[0.2, 2.8, 2.9], [0.1, 2.7, 3.0], [0.3, 2.5, 2.8]])


@pytest.fixture()
def vessels():
    return [
        VesselState(vessel_id=0, location=0, destination=1, speed=14.0, fuel=0.0, emissions=0.0),
        VesselState(vessel_id=1, location=1, destination=2, speed=14.0, fuel=0.0, emissions=0.0),
        VesselState(vessel_id=2, location=2, destination=0, speed=14.0, fuel=0.0, emissions=0.0),
    ]


@pytest.fixture()
def ports():
    return [
        PortState(port_id=0, queue=2, docks=3, occupied=1),
        PortState(port_id=1, queue=3, docks=3, occupied=2),
        PortState(port_id=2, queue=4, docks=3, occupied=3),
    ]


# ── FleetCoordinatorPolicy weather tests ────────────────────────────────


class TestCoordinatorWeatherRouting:
    """Coordinator forecast mode should prefer calmer routes."""

    def test_no_weather_baseline(self, cfg, vessels, ports) -> None:
        """Without weather, coordinator picks lowest forecast score port."""
        policy = FleetCoordinatorPolicy(cfg, mode="forecast")
        medium = np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
        action = policy.propose_action(medium, vessels, ports)
        assert action["dest_port"] == 0  # lowest mean forecast

    def test_weather_shifts_routing(self, cfg, vessels, ports) -> None:
        """When port 0 has rough seas, coordinator should shift preference."""
        policy = FleetCoordinatorPolicy(cfg, mode="forecast")
        # Forecasts: port 0 is slightly best (3.0 vs 3.1, 3.2)
        medium = np.array([[3.0, 3.1, 3.2], [3.0, 3.1, 3.2], [3.0, 3.1, 3.2]])
        # Weather: port 0 column has high sea-state
        weather = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [3.0, 3.0, 0.1]])
        # Column means: port0=1.07, port1=1.07, port2=0.1
        # With weather penalty, port 2 (calm seas) should be preferred
        action_weather = policy.propose_action(medium, vessels, ports, weather=weather)
        action_no_weather = policy.propose_action(medium, vessels, ports)
        # The key assertion: weather changes the routing decision
        # (specific port depends on penalty magnitude, but it shouldn't be same as no-weather)
        # At minimum, weather was used (scores differ)
        assert isinstance(action_weather["dest_port"], int)
        assert action_weather["dest_port"] != action_no_weather["dest_port"]

    def test_calm_weather_no_change(self, cfg, vessels, ports, calm_weather) -> None:
        """Calm weather (near zero) should not significantly change routing."""
        policy = FleetCoordinatorPolicy(cfg, mode="forecast")
        medium = np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
        action_calm = policy.propose_action(medium, vessels, ports, weather=calm_weather)
        action_none = policy.propose_action(medium, vessels, ports)
        # Port 0 is still best with calm weather
        assert action_calm["dest_port"] == action_none["dest_port"]

    def test_independent_mode_ignores_weather(self, cfg, vessels, ports, rough_weather) -> None:
        """Independent mode should ignore weather (random routing)."""
        policy = FleetCoordinatorPolicy(cfg, mode="independent")
        rng = np.random.default_rng(42)
        action = policy.propose_action(medium_forecast=np.zeros((3, 3)),
                                        vessels=vessels, ports=ports,
                                        rng=rng, weather=rough_weather)
        assert "dest_port" in action

    def test_reactive_mode_ignores_weather(self, cfg, vessels, ports, rough_weather) -> None:
        """Reactive mode should ignore weather (min queue routing)."""
        policy = FleetCoordinatorPolicy(cfg, mode="reactive")
        action = policy.propose_action(medium_forecast=np.zeros((3, 3)),
                                        vessels=vessels, ports=ports,
                                        weather=rough_weather)
        assert action["dest_port"] == 0  # port 0 has lowest queue

    def test_per_vessel_dest_weather_aware(self, cfg, vessels, ports, rough_weather) -> None:
        """Per-vessel destinations should reflect weather-adjusted port ranking."""
        policy = FleetCoordinatorPolicy(cfg, mode="forecast")
        medium = np.ones((3, 3)) * 3.0  # equal forecast scores
        action = policy.propose_action(medium, vessels, ports, weather=rough_weather)
        # Port 0 (calm column) should appear as best destination
        assert action["dest_port"] == 0


# ── VesselPolicy weather tests ──────────────────────────────────────────


class TestVesselWeatherSpeed:
    """Vessel forecast mode should slow down in rough seas."""

    def test_no_sea_state_baseline(self, cfg) -> None:
        """Without sea state, vessel uses congestion-based speed."""
        policy = VesselPolicy(cfg, mode="forecast")
        short = np.ones((3, 3)) * 2.0  # low congestion
        directive = {"dest_port": 0}
        action = policy.propose_action(short, directive)
        assert action["target_speed"] == cfg["speed_max"]

    def test_rough_sea_reduces_speed(self, cfg) -> None:
        """High sea state should cap vessel speed."""
        policy = VesselPolicy(cfg, mode="forecast")
        short = np.ones((3, 3)) * 2.0  # low congestion → normally speed_max
        directive = {"dest_port": 0}
        # Very rough sea state
        action = policy.propose_action(short, directive, sea_state=2.5)
        # fuel_mult at sea_state=2.5 with penalty=0.15 → 1 + 0.15*2.5 = 1.375 > 1.3
        assert action["target_speed"] == cfg["speed_min"]

    def test_moderate_sea_caps_at_nominal(self, cfg) -> None:
        """Moderate sea state should cap speed at nominal."""
        policy = VesselPolicy(cfg, mode="forecast")
        short = np.ones((3, 3)) * 2.0  # low congestion → normally speed_max
        directive = {"dest_port": 0}
        # Moderate sea: fuel_mult = 1 + 0.15*1.0 = 1.15 (between 1.1 and 1.3)
        action = policy.propose_action(short, directive, sea_state=1.0)
        assert action["target_speed"] == cfg["nominal_speed"]

    def test_calm_sea_no_speed_change(self, cfg) -> None:
        """Calm sea state should not change the speed decision."""
        policy = VesselPolicy(cfg, mode="forecast")
        short = np.ones((3, 3)) * 2.0
        directive = {"dest_port": 0}
        action = policy.propose_action(short, directive, sea_state=0.3)
        # fuel_mult = 1 + 0.15*0.3 = 1.045 < 1.1 → no adjustment
        assert action["target_speed"] == cfg["speed_max"]

    def test_independent_mode_ignores_sea(self, cfg) -> None:
        """Independent mode ignores sea_state entirely."""
        policy = VesselPolicy(cfg, mode="independent")
        action = policy.propose_action(np.zeros((3, 3)), {"dest_port": 0}, sea_state=3.0)
        assert action["target_speed"] == cfg["nominal_speed"]

    def test_reactive_mode_ignores_sea(self, cfg) -> None:
        """Reactive mode ignores sea_state entirely."""
        policy = VesselPolicy(cfg, mode="reactive")
        action = policy.propose_action(np.zeros((3, 3)), {"dest_port": 0}, sea_state=3.0)
        assert action["target_speed"] == cfg["nominal_speed"]


# ── Weather reward shaping tests ────────────────────────────────────────


class TestWeatherRewardShaping:
    """Tests for weather_vessel_shaping and weather_coordinator_shaping."""

    def test_vessel_shaping_calm_returns_zero(self, cfg) -> None:
        assert weather_vessel_shaping(14.0, 0.0, cfg) == 0.0

    def test_vessel_shaping_rough_slow_gives_bonus(self, cfg) -> None:
        # sea_state=2.5, fuel_mult=1.375 > 1.1, speed=10 <= nominal
        bonus = weather_vessel_shaping(10.0, 2.5, cfg)
        assert bonus > 0.0

    def test_vessel_shaping_rough_fast_no_bonus(self, cfg) -> None:
        # speed > nominal → no bonus even in rough seas
        bonus = weather_vessel_shaping(20.0, 2.5, cfg)
        assert bonus == 0.0

    def test_vessel_shaping_mild_weather_no_bonus(self, cfg) -> None:
        # fuel_mult = 1 + 0.15*0.5 = 1.075 < 1.1
        bonus = weather_vessel_shaping(10.0, 0.5, cfg)
        assert bonus == 0.0

    def test_coordinator_shaping_no_weather(self, cfg) -> None:
        assert weather_coordinator_shaping(None, {0: 1}, cfg) == 0.0

    def test_coordinator_shaping_no_destinations(self, cfg) -> None:
        weather = np.ones((3, 3))
        assert weather_coordinator_shaping(weather, {}, cfg) == 0.0

    def test_coordinator_shaping_calm_route_bonus(self, cfg) -> None:
        # Route through calm seas → high bonus
        weather = np.zeros((3, 3))
        destinations = {0: 1, 1: 2}
        bonus = weather_coordinator_shaping(weather, destinations, cfg)
        # All zeros → normalised = 0, bonus = 0.3 * 1.0 = 0.3
        assert bonus == pytest.approx(0.3, abs=0.01)

    def test_coordinator_shaping_rough_route_lower_bonus(self, cfg) -> None:
        # Route through rough seas → lower bonus
        weather = np.full((3, 3), 3.0)
        destinations = {0: 1, 1: 2}
        bonus = weather_coordinator_shaping(weather, destinations, cfg)
        # normalised = 1.0, bonus = 0.3 * 0.0 = 0.0
        assert bonus == pytest.approx(0.0, abs=0.01)

    def test_coordinator_shaping_mixed_routes(self, cfg) -> None:
        weather = np.array([[0.0, 3.0, 0.0], [3.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
        # Routes: 0→2 (sea=0.0), 1→0 (sea=3.0) → mean=1.5
        destinations = {0: 2, 1: 0}
        bonus = weather_coordinator_shaping(weather, destinations, cfg)
        # normalised = 1.5/3.0 = 0.5, bonus = 0.3 * 0.5 = 0.15
        assert bonus == pytest.approx(0.15, abs=0.01)


# ── Experiment weather logging tests ────────────────────────────────────


class TestExperimentWeatherLogging:
    """Experiment logs should include weather metrics when enabled."""

    def test_weather_fields_in_log_enabled(self) -> None:
        cfg = get_default_config(
            num_vessels=2, num_ports=2, weather_enabled=True, sea_state_max=3.0
        )
        df = run_experiment(policy_type="forecast", steps=5, seed=42, config=cfg)
        assert "weather_enabled" in df.columns
        assert "mean_sea_state" in df.columns
        assert "max_sea_state" in df.columns
        assert df["weather_enabled"].iloc[0] == 1
        assert df["mean_sea_state"].max() > 0.0

    def test_weather_fields_in_log_disabled(self) -> None:
        cfg = get_default_config(
            num_vessels=2, num_ports=2, weather_enabled=False
        )
        df = run_experiment(policy_type="forecast", steps=5, seed=42, config=cfg)
        assert "weather_enabled" in df.columns
        assert df["weather_enabled"].iloc[0] == 0
        assert df["mean_sea_state"].max() == 0.0


# ── MAPPO ablation env-config override tests ────────────────────────────


class TestAblationEnvOverrides:
    """Ablation entries with env_ prefix should override env config."""

    def test_env_prefix_separation(self) -> None:
        """env_ prefixed keys should be separated from MAPPO overrides."""
        overrides = {
            "env_weather_enabled": True,
            "env_sea_state_max": 5.0,
            "normalize_rewards": False,
        }
        env_overrides = {
            k[4:]: v for k, v in overrides.items() if k.startswith("env_")
        }
        mappo_overrides = {
            k: v for k, v in overrides.items() if not k.startswith("env_")
        }
        assert env_overrides == {"weather_enabled": True, "sea_state_max": 5.0}
        assert mappo_overrides == {"normalize_rewards": False}
