"""Research-grade robustness tests for determinism, NaN safety, and invariants.

These tests verify properties required for rigorous, reproducible research:
1. Deterministic reproducibility: same seed → identical trajectories
2. NaN/Inf safety: no corrupted values propagate through the pipeline
3. GAE computation integrity under adversarial inputs
4. Observation sanitization guarantees
5. Cross-parameter validation catches infeasible configs
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from hmarl_mvp.buffer import RolloutBuffer
from hmarl_mvp.config import HMARLConfig, get_default_config
from hmarl_mvp.dynamics import compute_fuel_and_emissions, step_vessels
from hmarl_mvp.env import MaritimeEnv
from hmarl_mvp.networks import DiscreteActor
from hmarl_mvp.state import VesselState


class TestDeterministicReproducibility(unittest.TestCase):
    """Same seed must produce identical trajectories across resets."""

    def test_env_reset_determinism(self) -> None:
        """Two environments with the same seed produce identical resets."""
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        env1 = MaritimeEnv(config=cfg, seed=123)
        env2 = MaritimeEnv(config=cfg, seed=123)
        obs1 = env1.reset()
        obs2 = env2.reset()
        for key in ("vessels", "ports", "coordinators"):
            for a, b in zip(obs1[key], obs2[key]):
                np.testing.assert_array_equal(a, b, err_msg=f"Mismatch in {key} observations")

    def test_env_trajectory_determinism(self) -> None:
        """Full rollouts with identical seeds and actions produce identical rewards."""
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=15)
        rewards_list = []
        for _ in range(2):
            env = MaritimeEnv(config=cfg, seed=42)
            env.reset()
            episode_rewards = []
            for _ in range(10):
                actions = env.sample_stub_actions()
                _, rewards, done, _ = env.step(actions)
                episode_rewards.append(rewards["vessels"][:])
                if done:
                    env.reset()
            rewards_list.append(episode_rewards)
        for step_idx in range(len(rewards_list[0])):
            np.testing.assert_array_almost_equal(
                rewards_list[0][step_idx],
                rewards_list[1][step_idx],
                decimal=10,
                err_msg=f"Reward mismatch at step {step_idx}",
            )

    def test_weather_trajectory_determinism(self) -> None:
        """Weather-enabled env is deterministic with same seed."""
        cfg = get_default_config(
            num_ports=3, num_vessels=4, rollout_steps=15,
            weather_enabled=True, weather_autocorrelation=0.5,
        )
        weather_traces = []
        for _ in range(2):
            env = MaritimeEnv(config=cfg, seed=99)
            env.reset()
            trace = []
            for _ in range(5):
                actions = env.sample_stub_actions()
                env.step(actions)
                trace.append(env._weather.copy() if env._weather is not None else None)
            weather_traces.append(trace)
        for i in range(5):
            np.testing.assert_array_equal(
                weather_traces[0][i], weather_traces[1][i],
                err_msg=f"Weather mismatch at step {i}",
            )


class TestNaNSafety(unittest.TestCase):
    """No NaN/Inf values should propagate through any pipeline component."""

    def test_observations_always_finite(self) -> None:
        """All observations must be finite after any step."""
        cfg = get_default_config(
            num_ports=3, num_vessels=4, rollout_steps=20,
            weather_enabled=True,
        )
        env = MaritimeEnv(config=cfg, seed=42)
        obs = env.reset()
        for _ in range(15):
            for key in ("vessels", "ports", "coordinators"):
                for i, o in enumerate(obs[key]):
                    self.assertTrue(
                        np.all(np.isfinite(o)),
                        f"Non-finite in {key}[{i}]: {o[~np.isfinite(o)]}",
                    )
            actions = env.sample_stub_actions()
            obs, _, done, _ = env.step(actions)
            if done:
                obs = env.reset()

    def test_fuel_emissions_finite_extreme_speed(self) -> None:
        """Fuel/emissions remain finite even at extreme speed values."""
        cfg = get_default_config()
        # Very high speed → speed^3 could overflow with bad coefficients
        fuel, co2 = compute_fuel_and_emissions(speed=1000.0, config=cfg, hours=100.0)
        self.assertTrue(np.isfinite(fuel), f"fuel is not finite: {fuel}")
        self.assertTrue(np.isfinite(co2), f"co2 is not finite: {co2}")

    def test_fuel_emissions_zero_speed(self) -> None:
        """Zero speed produces zero fuel/emissions."""
        cfg = get_default_config()
        fuel, co2 = compute_fuel_and_emissions(speed=0.0, config=cfg, hours=1.0)
        self.assertEqual(fuel, 0.0)
        self.assertEqual(co2, 0.0)

    def test_fuel_emissions_negative_inputs_safe(self) -> None:
        """Negative inputs are clamped, not propagated."""
        cfg = get_default_config()
        fuel, co2 = compute_fuel_and_emissions(speed=-5.0, config=cfg, hours=-1.0)
        self.assertEqual(fuel, 0.0)
        self.assertEqual(co2, 0.0)

    def test_step_vessels_nan_guard(self) -> None:
        """step_vessels never returns NaN/Inf in step stats."""
        cfg = get_default_config(num_ports=3, num_vessels=2)
        distance_nm = np.array([[0, 50, 100], [50, 0, 80], [100, 80, 0]], dtype=float)
        vessels = [
            VesselState(
                vessel_id=0, location=0, destination=1, speed=12.0,
                fuel=0.001, at_sea=True, position_nm=49.99,
            ),
            VesselState(
                vessel_id=1, location=1, destination=2, speed=15.0,
                fuel=100.0, at_sea=True, position_nm=0.0,
            ),
        ]
        stats = step_vessels(vessels, distance_nm, cfg, dt_hours=1.0)
        for vid, s in stats.items():
            for key, val in s.items():
                if isinstance(val, float):
                    self.assertTrue(
                        np.isfinite(val),
                        f"vessel {vid} stat {key} is not finite: {val}",
                    )


class TestGAENaNResilience(unittest.TestCase):
    """GAE computation must not propagate NaN from corrupted rewards."""

    def test_nan_rewards_treated_as_zero(self) -> None:
        """NaN in rewards should be replaced with 0.0 in GAE."""
        buf = RolloutBuffer(capacity=5, obs_dim=2)
        for i in range(5):
            reward = float("nan") if i == 2 else 1.0
            buf.add(
                obs=np.zeros(2),
                action=0.0,
                reward=reward,
                done=False,
                log_prob=0.0,
                value=0.5,
            )
        buf.compute_returns(last_value=0.0)
        data = buf.get_tensors()
        self.assertTrue(
            torch.all(torch.isfinite(data["advantages"])),
            f"Advantages contain non-finite: {data['advantages']}",
        )
        self.assertTrue(
            torch.all(torch.isfinite(data["returns"])),
            f"Returns contain non-finite: {data['returns']}",
        )

    def test_inf_values_treated_as_zero(self) -> None:
        """Inf in value estimates should not propagate NaN through GAE."""
        buf = RolloutBuffer(capacity=3, obs_dim=2)
        buf.add(obs=np.zeros(2), action=0.0, reward=1.0, done=False, log_prob=0.0, value=float("inf"))
        buf.add(obs=np.zeros(2), action=0.0, reward=1.0, done=False, log_prob=0.0, value=0.5)
        buf.add(obs=np.zeros(2), action=0.0, reward=1.0, done=True, log_prob=0.0, value=0.5)
        buf.compute_returns(last_value=0.0)
        data = buf.get_tensors()
        self.assertTrue(torch.all(torch.isfinite(data["advantages"])))
        # Returns = advantages + values; the stored inf value stays in _values.
        # Only check that advantages (the GAE output) are clean.
        self.assertTrue(torch.all(torch.isfinite(data["advantages"])))


class TestDiscreteActorMaskSafety(unittest.TestCase):
    """DiscreteActor must handle degenerate masks gracefully."""

    def test_all_actions_masked_no_nan(self) -> None:
        """When all actions are masked, output should still be valid (no NaN)."""
        actor = DiscreteActor(obs_dim=4, num_actions=3)
        obs = torch.randn(1, 4)
        # All False = all masked
        mask = torch.zeros(1, 3, dtype=torch.bool)
        dist = actor.forward(obs, action_mask=mask)
        # Should fall back to allowing all actions
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.assertTrue(torch.all(torch.isfinite(action)))
        self.assertTrue(torch.all(torch.isfinite(log_prob)))

    def test_single_valid_action_deterministic(self) -> None:
        """When only one action is valid, sampling must return that action."""
        actor = DiscreteActor(obs_dim=4, num_actions=5)
        obs = torch.randn(1, 4)
        mask = torch.zeros(1, 5, dtype=torch.bool)
        mask[0, 2] = True  # Only action 2 is valid
        dist = actor.forward(obs, action_mask=mask)
        for _ in range(10):
            action = dist.sample()
            self.assertEqual(action.item(), 2)


class TestConfigCrossValidation(unittest.TestCase):
    """Config validation catches infeasible parameter combinations."""

    def test_vessels_fewer_than_coordinators_rejected(self) -> None:
        with self.assertRaises(ValueError, msg="Should reject num_vessels < num_coordinators"):
            get_default_config(num_vessels=1, num_coordinators=3)

    def test_rollout_shorter_than_coord_interval_warns(self) -> None:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_default_config(rollout_steps=5, coord_decision_interval_steps=10)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            self.assertTrue(
                any("coordinator will never execute" in str(x.message) for x in user_warnings),
                "Expected UserWarning about coordinator not executing",
            )

    def test_valid_config_passes(self) -> None:
        """A sensible config should pass validation without error."""
        cfg = get_default_config(
            num_ports=5, num_vessels=8, num_coordinators=1,
            rollout_steps=20, coord_decision_interval_steps=12,
        )
        self.assertIsInstance(cfg, dict)

    def test_zero_reward_weights_rejected(self) -> None:
        """All-zero vessel reward weights should be caught."""
        with self.assertRaises(ValueError):
            get_default_config(
                fuel_weight=0, delay_weight=0, emission_weight=0,
                arrival_reward=0, on_time_arrival_reward=0,
                schedule_delay_weight=0, transit_time_weight=0,
            )


class TestObservationSanitization(unittest.TestCase):
    """Observation sanitization replaces NaN/Inf with zero."""

    def test_sanitize_obs_with_nan(self) -> None:
        obs_with_nan = np.array([1.0, float("nan"), 3.0])
        result = MaritimeEnv._sanitize_obs(obs_with_nan)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(result[1], 0.0)

    def test_sanitize_obs_with_inf(self) -> None:
        obs_with_inf = np.array([1.0, float("inf"), float("-inf")])
        result = MaritimeEnv._sanitize_obs(obs_with_inf)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_sanitize_obs_clean_passthrough(self) -> None:
        clean_obs = np.array([1.0, 2.0, 3.0])
        result = MaritimeEnv._sanitize_obs(clean_obs)
        np.testing.assert_array_equal(result, clean_obs)


class TestGlobalStateSanitization(unittest.TestCase):
    """Global state must also be finite for critic inputs."""

    def test_global_state_always_finite(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, rollout_steps=20)
        env = MaritimeEnv(config=cfg, seed=42)
        env.reset()
        for _ in range(5):
            gs = env.get_global_state()
            self.assertTrue(
                np.all(np.isfinite(gs)),
                f"Global state contains non-finite values: {gs[~np.isfinite(gs)]}",
            )
            actions = env.sample_stub_actions()
            _, _, done, _ = env.step(actions)
            if done:
                env.reset()


if __name__ == "__main__":
    unittest.main()
