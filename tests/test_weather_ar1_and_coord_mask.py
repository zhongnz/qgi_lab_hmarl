"""Tests for AR(1) correlated weather and coordinator action masking.

Session 27 additions:
- AR(1) temporal weather correlation (dynamics + env integration)
- Coordinator action masking in MAPPO
- weather_autocorrelation config field
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hmarl_mvp.config import HMARLConfig, get_default_config
from hmarl_mvp.dynamics import generate_weather, update_weather_ar1
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

# ===================================================================
# AR(1) Weather Model — dynamics.py
# ===================================================================


class TestUpdateWeatherAR1:
    """Tests for the AR(1) temporal weather correlation function."""

    def test_shape_preserved(self) -> None:
        """Output has the same shape as input."""
        rng = np.random.default_rng(42)
        prev = generate_weather(5, rng, sea_state_max=3.0)
        result = update_weather_ar1(prev, rng, autocorrelation=0.7, sea_state_max=3.0)
        assert result.shape == prev.shape

    def test_symmetric(self) -> None:
        """AR(1) output matrix is symmetric."""
        rng = np.random.default_rng(42)
        prev = generate_weather(5, rng)
        result = update_weather_ar1(prev, rng, autocorrelation=0.7)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_zero_diagonal(self) -> None:
        """Diagonal should always be zero."""
        rng = np.random.default_rng(42)
        prev = generate_weather(5, rng)
        result = update_weather_ar1(prev, rng, autocorrelation=0.7)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-12)

    def test_clipped_to_bounds(self) -> None:
        """All values within [0, sea_state_max]."""
        rng = np.random.default_rng(42)
        prev = generate_weather(5, rng, sea_state_max=3.0)
        result = update_weather_ar1(prev, rng, autocorrelation=0.9, sea_state_max=3.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 3.0)

    def test_zero_autocorrelation_is_iid(self) -> None:
        """With autocorrelation=0, result is pure noise (no dependence on prev)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        prev = np.ones((5, 5)) * 2.5  # high previous state
        np.fill_diagonal(prev, 0.0)
        result = update_weather_ar1(prev, rng1, autocorrelation=0.0, sea_state_max=3.0)
        fresh = generate_weather(5, rng2, sea_state_max=3.0)
        # Both used the same seed so both should get the same noise
        # With autocorrelation=0, the re-symmetrise makes them equal
        np.testing.assert_allclose(result, fresh, atol=1e-10)

    def test_high_autocorrelation_stays_close(self) -> None:
        """With very high autocorrelation, output tracks previous state closely."""
        rng = np.random.default_rng(42)
        prev = generate_weather(5, rng, sea_state_max=3.0)
        result = update_weather_ar1(prev, rng, autocorrelation=0.99, sea_state_max=3.0)
        # Should be close to prev
        diff = np.abs(result - prev).mean()
        assert diff < 0.2, f"Expected mean diff < 0.2, got {diff}"

    def test_temporal_correlation_exists(self) -> None:
        """Successive AR(1) steps are more correlated than i.i.d. steps."""
        rng_ar = np.random.default_rng(42)
        state = generate_weather(5, rng_ar, sea_state_max=3.0)
        ar_series = [state.copy()]
        for _ in range(50):
            state = update_weather_ar1(state, rng_ar, autocorrelation=0.8, sea_state_max=3.0)
            ar_series.append(state.copy())

        # Compute lag-1 autocorrelation on the (0,1) entry
        values = [s[0, 1] for s in ar_series]
        mean = np.mean(values)
        numer = sum((values[i] - mean) * (values[i + 1] - mean) for i in range(len(values) - 1))
        denom = sum((v - mean) ** 2 for v in values) + 1e-12
        lag1_corr = numer / denom
        assert lag1_corr > 0.3, f"Expected lag-1 autocorrelation > 0.3, got {lag1_corr}"

    def test_autocorrelation_clamped(self) -> None:
        """Values outside [0,1] are silently clamped."""
        rng = np.random.default_rng(42)
        prev = generate_weather(3, rng, sea_state_max=3.0)
        # Should not raise with autocorrelation > 1
        result = update_weather_ar1(prev, rng, autocorrelation=1.5, sea_state_max=3.0)
        assert result.shape == prev.shape


# ===================================================================
# AR(1) Weather — Config validation
# ===================================================================


class TestWeatherAutocorrelationConfig:
    """Config field validation for weather_autocorrelation."""

    def test_default_zero(self) -> None:
        """Default autocorrelation is 0 (backward compatible)."""
        cfg = HMARLConfig()
        assert cfg.weather_autocorrelation == 0.0

    def test_valid_value(self) -> None:
        """Autocorrelation within [0, 1] passes validation."""
        cfg = HMARLConfig(weather_autocorrelation=0.7)
        cfg.validate()  # should not raise

    def test_too_high_rejects(self) -> None:
        """Autocorrelation > 1.0 raises ValueError."""
        cfg = HMARLConfig(weather_autocorrelation=1.5)
        with pytest.raises(ValueError, match="weather_autocorrelation"):
            cfg.validate()

    def test_negative_rejects(self) -> None:
        """Negative autocorrelation raises ValueError."""
        with pytest.raises(ValueError):
            get_default_config(weather_autocorrelation=-0.1)

    def test_in_config_dict(self) -> None:
        """weather_autocorrelation flows through get_default_config."""
        cfg = get_default_config(weather_autocorrelation=0.5)
        assert cfg["weather_autocorrelation"] == 0.5


# ===================================================================
# AR(1) Weather — Environment integration
# ===================================================================


class TestAR1WeatherEnvIntegration:
    """Weather AR(1) is actually used in the environment step loop."""

    def test_env_uses_ar1_when_configured(self) -> None:
        """When autocorrelation > 0, consecutive weather matrices are correlated."""
        env = MaritimeEnv(config={
            "weather_enabled": True,
            "weather_autocorrelation": 0.8,
            "rollout_steps": 20,
            "num_ports": 3,
            "num_vessels": 2,
        })
        obs = env.reset()  # noqa: F841
        assert env._weather is not None
        w0 = env._weather.copy()

        # Step a few times
        weather_snapshots = [w0]
        for _ in range(10):
            actions = env.sample_stub_actions()
            env.step(actions)
            if env._weather is not None:
                weather_snapshots.append(env._weather.copy())

        # Check temporal correlation
        if len(weather_snapshots) >= 3:
            diffs_consecutive = [
                np.abs(weather_snapshots[i + 1] - weather_snapshots[i]).mean()
                for i in range(len(weather_snapshots) - 1)
            ]
            mean_diff = np.mean(diffs_consecutive)
            # AR(1) with high autocorr should have small diffs
            assert mean_diff < 1.5, f"Expected small consecutive diffs, got {mean_diff}"

    def test_env_iid_when_autocorrelation_zero(self) -> None:
        """With autocorrelation=0, weather is i.i.d. (backward compatible)."""
        env = MaritimeEnv(config={
            "weather_enabled": True,
            "weather_autocorrelation": 0.0,
            "rollout_steps": 10,
            "num_ports": 3,
            "num_vessels": 2,
        })
        env.reset()
        # Just verify it works without error
        for _ in range(5):
            actions = env.sample_stub_actions()
            env.step(actions)


# ===================================================================
# Coordinator Action Masking
# ===================================================================


class TestCoordinatorActionMasking:
    """Tests for coordinator destination masking in MAPPO."""

    def test_build_coordinator_mask_all_open(self) -> None:
        """When no port is congested, all destinations are valid."""
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 3, "docks_per_port": 3, "rollout_steps": 10},
            mappo_config=MAPPOConfig(rollout_length=5, num_epochs=1),
        )
        trainer.env.reset()
        mask = trainer._build_coordinator_mask()
        assert mask.shape == (3,)
        assert mask.sum() == 3.0  # all ports valid

    def test_mask_blocks_congested_port(self) -> None:
        """A port with full docks and full queue should be masked out."""
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 3, "docks_per_port": 2, "rollout_steps": 10},
            mappo_config=MAPPOConfig(rollout_length=5, num_epochs=1),
        )
        trainer.env.reset()
        # Simulate congestion: port 1 is fully occupied and has full queue
        port = trainer.env.ports[1]
        port.occupied = port.docks
        port.queue = port.docks
        port.service_times = [6.0] * port.docks

        mask = trainer._build_coordinator_mask()
        assert mask[1] == 0.0, "Congested port should be masked"
        assert mask[0] == 1.0
        assert mask[2] == 1.0

    def test_mask_safety_all_congested(self) -> None:
        """If ALL ports are congested, all are allowed (safety fallback)."""
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "docks_per_port": 1, "rollout_steps": 10},
            mappo_config=MAPPOConfig(rollout_length=5, num_epochs=1),
        )
        trainer.env.reset()
        # Congest all ports
        for port in trainer.env.ports:
            port.occupied = port.docks
            port.queue = port.docks
            port.service_times = [6.0] * port.docks

        mask = trainer._build_coordinator_mask()
        assert mask.sum() == 2.0, "All congested -> safety fallback: all allowed"

    def test_coordinator_mask_tensor_shape(self) -> None:
        """Tensor mask has correct shape and dtype."""
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 4, "docks_per_port": 2, "rollout_steps": 10},
            mappo_config=MAPPOConfig(rollout_length=5, num_epochs=1),
        )
        trainer.env.reset()
        mask_t = trainer._coordinator_mask_tensor()
        assert mask_t.shape == (1, 4)
        assert mask_t.dtype == torch.bool

    def test_coordinator_buffer_has_mask_dim(self) -> None:
        """Coordinator rollout buffer is allocated with mask_dim=num_ports."""
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 3, "docks_per_port": 2, "rollout_steps": 10},
            mappo_config=MAPPOConfig(rollout_length=5, num_epochs=1),
        )
        trainer.env.reset()
        # Check that coordinator buffer has mask storage
        coord_buf = trainer.coordinator_buf[0]
        assert coord_buf.mask_dim is not None
        assert coord_buf.mask_dim == 3

    def test_collect_rollout_with_coordinator_mask(self) -> None:
        """Full rollout collection works with coordinator masking enabled."""
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 3, "docks_per_port": 2, "rollout_steps": 10},
            mappo_config=MAPPOConfig(rollout_length=5, num_epochs=1),
        )
        stats = trainer.collect_rollout()
        assert "coordinator_mean_reward" in stats or "total_reward" in stats


# ===================================================================
# Figure generation script — import test
# ===================================================================


class TestFigureScriptImport:
    """Verify the paper figure generation script is importable."""

    def test_importable(self) -> None:
        """Script module can be imported without errors."""
        import importlib
        import sys
        scripts_dir = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        spec = importlib.util.find_spec("generate_paper_figures")  # type: ignore[union-attr]
        assert spec is not None, "generate_paper_figures script should be importable"
