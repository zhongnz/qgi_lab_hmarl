"""Tests for curriculum weather ramp, CLI weather flag, gym wrapper weather,
and MAPPO weather speed capping."""

from __future__ import annotations

import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from hmarl_mvp.config import get_default_config
from hmarl_mvp.curriculum import CurriculumScheduler
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.gym_wrapper import MaritimeGymEnv

# ── Curriculum weather ramp tests ────────────────────────────────────────


class TestCurriculumWeatherRamp:
    """CurriculumScheduler should ramp weather parameters."""

    def test_sea_state_max_ramps_linearly(self) -> None:
        """sea_state_max should interpolate from start to target."""
        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 4,
                "num_ports": 3,
                "sea_state_max": 5.0,
            },
            start_config={"sea_state_max": 1.0},
            warmup_fraction=0.5,
        )
        # At start (progress=0), sea_state_max should be start value (1.0)
        cfg_start = scheduler.get_config(0, 100)
        assert cfg_start["sea_state_max"] == pytest.approx(1.0, abs=0.01)

        # At midpoint of warmup (progress=0.25, alpha=0.5), should be 3.0
        cfg_mid = scheduler.get_config(25, 100)
        assert cfg_mid["sea_state_max"] == pytest.approx(3.0, abs=0.01)

        # After warmup (progress=0.5+), should be target (5.0)
        cfg_end = scheduler.get_config(50, 100)
        assert cfg_end["sea_state_max"] == pytest.approx(5.0, abs=0.01)

    def test_weather_penalty_factor_ramps(self) -> None:
        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 4,
                "num_ports": 3,
                "weather_penalty_factor": 0.3,
            },
            start_config={"weather_penalty_factor": 0.0},
            warmup_fraction=1.0,
        )
        # Halfway through training
        cfg = scheduler.get_config(50, 100)
        assert cfg["weather_penalty_factor"] == pytest.approx(0.15, abs=0.01)

    def test_weather_shaping_weight_ramps(self) -> None:
        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 4,
                "num_ports": 3,
                "weather_shaping_weight": 0.6,
            },
            start_config={"weather_shaping_weight": 0.0},
            warmup_fraction=1.0,
        )
        cfg = scheduler.get_config(50, 100)
        assert cfg["weather_shaping_weight"] == pytest.approx(0.3, abs=0.01)

    def test_weather_enabled_bool_ramp(self) -> None:
        """weather_enabled should switch on at alpha >= 0.5."""
        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 4,
                "num_ports": 3,
                "weather_enabled": True,
            },
            start_config={"weather_enabled": False},
            warmup_fraction=1.0,
        )
        # Before halfway: disabled
        cfg_early = scheduler.get_config(24, 100)
        assert cfg_early["weather_enabled"] is False

        # At or after halfway: enabled
        cfg_late = scheduler.get_config(50, 100)
        assert cfg_late["weather_enabled"] is True

    def test_weather_enabled_same_start_and_target(self) -> None:
        """If weather_enabled=True in both start and target, always True."""
        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 4,
                "num_ports": 3,
                "weather_enabled": True,
            },
            start_config={"weather_enabled": True},
            warmup_fraction=0.5,
        )
        cfg = scheduler.get_config(0, 100)
        assert cfg["weather_enabled"] is True

    def test_multi_stage_weather(self) -> None:
        """Multi-stage curriculum can introduce weather at a specific stage."""
        from hmarl_mvp.curriculum import CurriculumStage

        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 6,
                "num_ports": 4,
                "weather_enabled": True,
                "sea_state_max": 5.0,
            },
            stages=[
                CurriculumStage(fraction=0.0, config_overrides={
                    "num_vessels": 2,
                    "num_ports": 2,
                    "weather_enabled": False,
                }),
                CurriculumStage(fraction=0.5, config_overrides={
                    "num_vessels": 4,
                    "num_ports": 3,
                    "weather_enabled": True,
                    "sea_state_max": 3.0,
                }),
                CurriculumStage(fraction=0.8, config_overrides={
                    "weather_enabled": True,
                    "sea_state_max": 5.0,
                }),
            ],
        )
        # Stage 1: no weather
        cfg1 = scheduler.get_config(10, 100)
        assert cfg1.get("weather_enabled") is False

        # Stage 2: moderate weather
        cfg2 = scheduler.get_config(60, 100)
        assert cfg2.get("weather_enabled") is True
        assert cfg2.get("sea_state_max") == 3.0

        # Stage 3: full weather
        cfg3 = scheduler.get_config(85, 100)
        assert cfg3.get("weather_enabled") is True
        assert cfg3.get("sea_state_max") == 5.0


# ── CLI weather flag tests ───────────────────────────────────────────────


class TestCLIWeatherFlag:
    """run_mappo.py should accept --weather and --sea-state-max flags."""

    def test_help_includes_weather(self) -> None:
        """The train subcommand help should mention --weather."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_mappo", "train", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl",
            timeout=30,
        )
        assert "--weather" in result.stdout
        assert "--sea-state-max" in result.stdout

    def test_ablate_help_includes_weather(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_mappo", "ablate", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl",
            timeout=30,
        )
        assert "--weather" in result.stdout

    def test_parse_weather_flag(self) -> None:
        """Parsing --weather should set weather=True in namespace."""
        from scripts.run_mappo import parse_args

        with patch("sys.argv", ["prog", "train", "--weather", "--iterations", "1"]):
            args = parse_args()
        assert args.weather is True
        assert args.sea_state_max == 3.0

    def test_parse_sea_state_max(self) -> None:
        from scripts.run_mappo import parse_args

        with patch("sys.argv", ["prog", "train", "--weather", "--sea-state-max", "5.0"]):
            args = parse_args()
        assert args.sea_state_max == 5.0

    def test_parse_no_weather_default(self) -> None:
        from scripts.run_mappo import parse_args

        with patch("sys.argv", ["prog", "train", "--iterations", "1"]):
            args = parse_args()
        assert args.weather is False


class TestCLIWeatherWiring:
    """Weather flags should be propagated into command execution config."""

    def test_compare_passes_weather_config(self, tmp_path: Any) -> None:
        from scripts.run_mappo import cmd_compare, parse_args

        captured: dict[str, Any] = {}

        def fake_compare(*args: Any, **kwargs: Any) -> dict[str, pd.DataFrame]:
            captured["config"] = kwargs.get("config")
            step_df = pd.DataFrame({"t": [0], "avg_queue": [1.0]})
            train_df = pd.DataFrame({"iteration": [1], "mean_reward": [0.0]})
            return {"mappo": step_df, "independent": step_df.copy(), "_train_log": train_df}

        with patch("scripts.run_mappo.run_mappo_comparison", side_effect=fake_compare):
            with patch(
                "sys.argv",
                [
                    "prog",
                    "compare",
                    "--weather",
                    "--sea-state-max",
                    "4.0",
                    "--no-plots",
                    "--output-dir",
                    str(tmp_path),
                ],
            ):
                args = parse_args()
            cmd_compare(args)

        assert captured["config"]["weather_enabled"] is True
        assert captured["config"]["sea_state_max"] == 4.0

    def test_sweep_passes_weather_config(self, tmp_path: Any) -> None:
        from scripts.run_mappo import cmd_sweep, parse_args

        captured: dict[str, Any] = {}

        def fake_sweep(*args: Any, **kwargs: Any) -> pd.DataFrame:
            captured["config"] = kwargs.get("config")
            return pd.DataFrame(
                {"lr": [3e-4], "entropy_coeff": [0.01], "total_reward": [1.0]}
            )

        with patch("scripts.run_mappo.run_mappo_hyperparam_sweep", side_effect=fake_sweep):
            with patch(
                "sys.argv",
                [
                    "prog",
                    "sweep",
                    "--weather",
                    "--sea-state-max",
                    "3.5",
                    "--output-dir",
                    str(tmp_path),
                ],
            ):
                args = parse_args()
            cmd_sweep(args)

        assert captured["config"]["weather_enabled"] is True
        assert captured["config"]["sea_state_max"] == 3.5

    def test_ablate_passes_weather_config(self, tmp_path: Any) -> None:
        from scripts.run_mappo import cmd_ablate, parse_args

        captured: dict[str, Any] = {}

        def fake_ablate(*args: Any, **kwargs: Any) -> pd.DataFrame:
            captured["config"] = kwargs.get("config")
            return pd.DataFrame(
                {
                    "ablation": ["baseline"],
                    "final_mean_reward": [0.0],
                    "best_mean_reward": [0.0],
                    "total_reward": [0.0],
                }
            )

        with patch("scripts.run_mappo.run_mappo_ablation", side_effect=fake_ablate):
            with patch(
                "sys.argv",
                [
                    "prog",
                    "ablate",
                    "--weather",
                    "--sea-state-max",
                    "6.0",
                    "--output-dir",
                    str(tmp_path),
                ],
            ):
                args = parse_args()
            cmd_ablate(args)

        assert captured["config"]["weather_enabled"] is True
        assert captured["config"]["sea_state_max"] == 6.0


# ── Gym wrapper weather info tests ───────────────────────────────────────


class TestGymWrapperWeather:
    """Gym wrapper should expose weather info."""

    def test_weather_matrix_in_info(self) -> None:
        """When weather is enabled, step info should include weather_matrix."""
        env = MaritimeGymEnv(
            config={"num_vessels": 2, "num_ports": 2, "weather_enabled": True},
        )
        _obs, _info = env.reset()
        action = env.action_space.sample()
        _obs, _reward, _term, _trunc, info = env.step(action)
        assert "weather_matrix" in info
        assert info["weather_matrix"].shape == (2, 2)

    def test_no_weather_matrix_when_disabled(self) -> None:
        env = MaritimeGymEnv(
            config={"num_vessels": 2, "num_ports": 2, "weather_enabled": False},
        )
        _obs, _info = env.reset()
        action = env.action_space.sample()
        _obs, _reward, _term, _trunc, info = env.step(action)
        assert "weather_matrix" not in info

    def test_weather_obs_dim_matches(self) -> None:
        """Observation dimension should account for weather."""
        env_no = MaritimeGymEnv(
            config={"num_vessels": 2, "num_ports": 2, "weather_enabled": False},
        )
        env_w = MaritimeGymEnv(
            config={"num_vessels": 2, "num_ports": 2, "weather_enabled": True},
        )
        # Weather adds +1 per vessel to obs
        diff = env_w.observation_space.shape[0] - env_no.observation_space.shape[0]
        assert diff == 2  # +1 per vessel × 2 vessels


# ── MAPPO weather speed cap tests ────────────────────────────────────────


class TestMAPPOWeatherSpeedCap:
    """_nn_to_vessel_action should respect speed_cap from weather."""

    def test_speed_cap_none_no_effect(self) -> None:
        """Without speed_cap, full speed range is available."""
        import torch

        from hmarl_mvp.mappo import _nn_to_vessel_action

        cfg: dict[str, Any] = {"speed_min": 8.0, "speed_max": 20.0}
        raw = torch.tensor([18.0])
        action = _nn_to_vessel_action(raw, cfg, speed_cap=None)
        assert action["target_speed"] == 18.0

    def test_speed_cap_limits_max(self) -> None:
        """Speed cap should limit the maximum speed."""
        import torch

        from hmarl_mvp.mappo import _nn_to_vessel_action

        cfg: dict[str, Any] = {"speed_min": 8.0, "speed_max": 20.0}
        raw = torch.tensor([18.0])
        action = _nn_to_vessel_action(raw, cfg, speed_cap=14.0)
        assert action["target_speed"] == 14.0

    def test_speed_cap_allows_below(self) -> None:
        """Speed below cap should pass through unchanged."""
        import torch

        from hmarl_mvp.mappo import _nn_to_vessel_action

        cfg: dict[str, Any] = {"speed_min": 8.0, "speed_max": 20.0}
        raw = torch.tensor([12.0])
        action = _nn_to_vessel_action(raw, cfg, speed_cap=14.0)
        assert action["target_speed"] == 12.0

    def test_speed_min_floor_maintained(self) -> None:
        """Speed_min should still be the floor even with a cap."""
        import torch

        from hmarl_mvp.mappo import _nn_to_vessel_action

        cfg: dict[str, Any] = {"speed_min": 8.0, "speed_max": 20.0}
        raw = torch.tensor([5.0])
        action = _nn_to_vessel_action(raw, cfg, speed_cap=14.0)
        assert action["target_speed"] == 8.0

    def test_vessel_weather_speed_cap_no_weather(self) -> None:
        """When weather disabled, cap should be None."""
        from hmarl_mvp.mappo import MAPPOTrainer

        cfg = get_default_config(num_vessels=2, num_ports=2, weather_enabled=False)
        trainer = MAPPOTrainer(env_config=cfg, seed=42)
        trainer.env.reset()
        cap = trainer._vessel_weather_speed_cap(0)
        assert cap is None

    def test_vessel_weather_speed_cap_calm(self) -> None:
        """In calm weather, cap should be None."""
        from hmarl_mvp.mappo import MAPPOTrainer

        cfg = get_default_config(
            num_vessels=2, num_ports=2,
            weather_enabled=True, sea_state_max=0.3,
        )
        trainer = MAPPOTrainer(env_config=cfg, seed=42)
        trainer.env.reset()
        # With sea_state_max=0.3, fuel_mult = 1 + 0.15*0.15 ≈ 1.02 < 1.1
        cap = trainer._vessel_weather_speed_cap(0)
        # Very mild — should always be None for low enough sea states
        # (depends on generated weather; with max=0.3, always below 1.1)
        assert cap is None

    def test_vessel_weather_speed_cap_rough(self) -> None:
        """In rough weather, cap should be speed_min or nominal_speed."""
        from hmarl_mvp.mappo import MAPPOTrainer

        cfg = get_default_config(
            num_vessels=2, num_ports=2,
            weather_enabled=True, sea_state_max=10.0,
            weather_penalty_factor=0.5,
        )
        trainer = MAPPOTrainer(env_config=cfg, seed=42)
        trainer.env.reset()
        cap = trainer._vessel_weather_speed_cap(0)
        # With sea_state_max=10 and penalty=0.5, fuel_mult would be very high
        # Cap should be either nominal_speed or speed_min
        assert cap is not None
        assert cap <= cfg["nominal_speed"]


# ── Integration: weather curriculum + MAPPO ──────────────────────────────


class TestWeatherCurriculumIntegration:
    """Curriculum weather ramp should produce valid env configs."""

    def test_curriculum_config_creates_valid_env(self) -> None:
        """Each curriculum stage config should create a valid MaritimeEnv."""
        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 4,
                "num_ports": 3,
                "weather_enabled": True,
                "sea_state_max": 3.0,
            },
            start_config={
                "weather_enabled": False,
                "sea_state_max": 0.5,
            },
            warmup_fraction=0.5,
        )
        for step in [0, 25, 50, 75, 100]:
            cfg = scheduler.get_config(step, 100)
            full_cfg = get_default_config(**cfg)
            env = MaritimeEnv(config=full_cfg)
            env.reset()
            assert len(env.vessels) > 0
            assert len(env.ports) > 0

    def test_weather_ramp_affects_observations(self) -> None:
        """When weather ramps on, vessel obs should gain extra dimension."""
        from hmarl_mvp.networks import obs_dim_from_env

        scheduler = CurriculumScheduler(
            target_config={
                "num_vessels": 3,
                "num_ports": 2,
                "weather_enabled": True,
            },
            start_config={"weather_enabled": False},
            warmup_fraction=0.5,
        )
        # Before weather (progress < 0.25 → alpha < 0.5)
        cfg_early = get_default_config(**scheduler.get_config(10, 100))
        d_early = obs_dim_from_env(cfg_early)

        # After weather (progress > 0.25 → alpha > 0.5)
        cfg_late = get_default_config(**scheduler.get_config(60, 100))
        d_late = obs_dim_from_env(cfg_late)

        assert d_late["vessel"] == d_early["vessel"] + 1
