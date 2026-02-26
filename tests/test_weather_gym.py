"""Tests for weather effects and Gymnasium wrapper."""

from __future__ import annotations

import unittest

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.dynamics import (
    compute_fuel_and_emissions,
    generate_weather,
    step_vessels,
    weather_fuel_multiplier,
    weather_speed_factor,
)
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.networks import obs_dim_from_env
from hmarl_mvp.state import VesselState

# ===================================================================
# Weather physics unit tests
# ===================================================================


class TestGenerateWeather(unittest.TestCase):
    """Tests for the weather generation function."""

    def test_shape_and_symmetry(self) -> None:
        rng = np.random.default_rng(42)
        w = generate_weather(5, rng, sea_state_max=3.0)
        self.assertEqual(w.shape, (5, 5))
        np.testing.assert_array_almost_equal(w, w.T)

    def test_diagonal_zero(self) -> None:
        rng = np.random.default_rng(42)
        w = generate_weather(4, rng)
        np.testing.assert_array_equal(np.diag(w), 0.0)

    def test_values_in_range(self) -> None:
        rng = np.random.default_rng(42)
        w = generate_weather(6, rng, sea_state_max=5.0)
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(w <= 5.0))

    def test_single_port(self) -> None:
        rng = np.random.default_rng(42)
        w = generate_weather(1, rng)
        self.assertEqual(w.shape, (1, 1))
        self.assertAlmostEqual(w[0, 0], 0.0)

    def test_different_seeds_produce_different_weather(self) -> None:
        w1 = generate_weather(3, np.random.default_rng(1))
        w2 = generate_weather(3, np.random.default_rng(2))
        self.assertFalse(np.allclose(w1, w2))


class TestWeatherMultipliers(unittest.TestCase):
    """Tests for weather_fuel_multiplier and weather_speed_factor."""

    def test_calm_seas_no_penalty(self) -> None:
        self.assertAlmostEqual(weather_fuel_multiplier(0.0), 1.0)
        self.assertAlmostEqual(weather_speed_factor(0.0), 1.0)

    def test_rough_seas_increase_fuel(self) -> None:
        m = weather_fuel_multiplier(3.0, penalty_factor=0.15)
        self.assertAlmostEqual(m, 1.45)

    def test_rough_seas_reduce_speed(self) -> None:
        f = weather_speed_factor(3.0, penalty_factor=0.15)
        self.assertAlmostEqual(f, 1.0 / 1.45, places=4)

    def test_fuel_x_speed_consistent(self) -> None:
        """fuel_multiplier * speed_factor ≈ 1 (they are inverses)."""
        for state in [0.0, 1.0, 2.0, 3.0]:
            m = weather_fuel_multiplier(state, 0.15)
            f = weather_speed_factor(state, 0.15)
            self.assertAlmostEqual(m * f, 1.0, places=10)

    def test_negative_sea_state_clamped(self) -> None:
        self.assertAlmostEqual(weather_fuel_multiplier(-1.0), 1.0)


class TestComputeFuelWithWeather(unittest.TestCase):
    """Weather affects fuel consumption."""

    def test_zero_sea_state_matches_baseline(self) -> None:
        cfg = get_default_config()
        fuel_base, co2_base = compute_fuel_and_emissions(12.0, cfg, hours=1.0)
        fuel_w, co2_w = compute_fuel_and_emissions(12.0, cfg, hours=1.0, sea_state=0.0)
        self.assertAlmostEqual(fuel_base, fuel_w)
        self.assertAlmostEqual(co2_base, co2_w)

    def test_rough_seas_increase_fuel(self) -> None:
        cfg = get_default_config()
        fuel_calm, _ = compute_fuel_and_emissions(12.0, cfg, hours=1.0, sea_state=0.0)
        fuel_rough, _ = compute_fuel_and_emissions(12.0, cfg, hours=1.0, sea_state=3.0)
        self.assertGreater(fuel_rough, fuel_calm)
        # Should be 1.45x with default penalty
        expected_ratio = 1.0 + 0.15 * 3.0
        self.assertAlmostEqual(fuel_rough / fuel_calm, expected_ratio, places=5)


class TestStepVesselsWithWeather(unittest.TestCase):
    """step_vessels with weather matrix."""

    def _make_vessel(self) -> VesselState:
        v = VesselState(vessel_id=0, location=0, destination=1, speed=12.0, fuel=100.0)
        v.at_sea = True
        v.position_nm = 0.0
        return v

    def test_no_weather_backward_compatible(self) -> None:
        cfg = get_default_config(num_ports=3)
        v = self._make_vessel()
        distance = np.array([[0, 1000, 2000], [1000, 0, 1500], [2000, 1500, 0]], dtype=float)
        stats = step_vessels([v], distance, cfg, dt_hours=1.0)
        self.assertGreater(v.position_nm, 0.0)
        self.assertIn(0, stats)

    def test_weather_reduces_distance_covered(self) -> None:
        cfg = get_default_config(num_ports=2)
        distance = np.array([[0, 10000], [10000, 0]], dtype=float)
        weather = np.array([[0, 3.0], [3.0, 0]], dtype=float)  # rough seas

        v_calm = self._make_vessel()
        step_vessels([v_calm], distance, cfg, dt_hours=1.0, weather=None)

        v_rough = self._make_vessel()
        step_vessels([v_rough], distance, cfg, dt_hours=1.0, weather=weather)

        self.assertGreater(v_calm.position_nm, v_rough.position_nm)

    def test_weather_increases_fuel_used(self) -> None:
        cfg = get_default_config(num_ports=2)
        distance = np.array([[0, 10000], [10000, 0]], dtype=float)
        weather = np.array([[0, 2.0], [2.0, 0]], dtype=float)

        v_calm = self._make_vessel()
        stats_calm = step_vessels([v_calm], distance, cfg, dt_hours=1.0, weather=None)

        v_rough = self._make_vessel()
        stats_rough = step_vessels([v_rough], distance, cfg, dt_hours=1.0, weather=weather)

        self.assertGreater(stats_rough[0]["fuel_used"], stats_calm[0]["fuel_used"])


# ===================================================================
# Weather integration in environment
# ===================================================================


class TestEnvWeatherIntegration(unittest.TestCase):
    """Test weather effects in the full environment."""

    def test_weather_disabled_by_default(self) -> None:
        env = MaritimeEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        env.reset()
        self.assertFalse(env._weather_enabled)
        self.assertIsNone(env._weather)

    def test_weather_enabled_produces_matrix(self) -> None:
        env = MaritimeEnv(
            config={"num_vessels": 2, "num_ports": 3, "rollout_steps": 5, "weather_enabled": True},
        )
        env.reset()
        self.assertTrue(env._weather_enabled)
        self.assertIsNotNone(env._weather)
        assert env._weather is not None  # for mypy
        self.assertEqual(env._weather.shape, (3, 3))

    def test_weather_obs_dim_increases(self) -> None:
        cfg_no_weather = get_default_config(num_vessels=2, num_ports=2)
        cfg_weather = get_default_config(num_vessels=2, num_ports=2, weather_enabled=True)
        dims_no = obs_dim_from_env(cfg_no_weather)
        dims_yes = obs_dim_from_env(cfg_weather)
        self.assertEqual(dims_yes["vessel"], dims_no["vessel"] + 1)
        self.assertEqual(dims_yes["port"], dims_no["port"])  # ports unaffected

    def test_weather_env_step_produces_info(self) -> None:
        env = MaritimeEnv(
            config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5, "weather_enabled": True},
        )
        env.reset()
        actions = env.sample_stub_actions()
        _, _, _, info = env.step(actions)
        self.assertTrue(info["weather_enabled"])
        self.assertIn("mean_sea_state", info)
        self.assertIn("max_sea_state", info)
        self.assertGreaterEqual(info["mean_sea_state"], 0)

    def test_weather_vessel_obs_has_extra_dim(self) -> None:
        env_no = MaritimeEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        env_w = MaritimeEnv(
            config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5, "weather_enabled": True},
        )
        obs_no = env_no.reset()
        obs_w = env_w.reset()
        no_dim = len(obs_no["vessels"][0])
        w_dim = len(obs_w["vessels"][0])
        self.assertEqual(w_dim, no_dim + 1)


# ===================================================================
# Gymnasium wrapper tests
# ===================================================================


class TestMaritimeGymEnv(unittest.TestCase):
    """Tests for the Gymnasium-compatible wrapper."""

    def test_creation_and_spaces(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        self.assertIsNotNone(env.observation_space)
        self.assertIsNotNone(env.action_space)
        assert env.observation_space.shape is not None  # for mypy
        self.assertEqual(env.observation_space.shape[0], env._obs_dim)

    def test_reset_returns_tuple(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        result = env.reset()
        self.assertEqual(len(result), 2)  # (obs, info)
        obs, info = result
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIn("raw_obs", info)

    def test_step_returns_5_tuple(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("raw_rewards", info)

    def test_episode_terminates(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 3})
        env.reset()
        terminated = False
        steps = 0
        while not terminated:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if steps > 100:
                break
        self.assertTrue(terminated)
        self.assertEqual(steps, 3)

    def test_reset_with_seed(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 3})
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_obs_in_observation_space(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))

    def test_unwrapped_returns_maritime_env(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        self.assertIsInstance(env.unwrapped, MaritimeEnv)

    def test_with_weather_enabled(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(
            config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 3, "weather_enabled": True},
        )
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        action = env.action_space.sample()
        obs2, reward, done, truncated, info = env.step(action)
        self.assertTrue(env.observation_space.contains(obs2))
        self.assertTrue(info["weather_enabled"])

    def test_info_contains_reward_breakdown(self) -> None:
        from hmarl_mvp.gym_wrapper import MaritimeGymEnv

        env = MaritimeGymEnv(config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 5})
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        # Reward should be sum of components
        expected = info["vessel_reward"] + info["port_reward"] + info["coordinator_reward"]
        self.assertAlmostEqual(reward, expected, places=5)


if __name__ == "__main__":
    unittest.main()
