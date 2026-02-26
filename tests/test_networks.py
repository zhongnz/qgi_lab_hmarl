"""Tests for neural network modules: actors, critics, and factory helpers."""

from __future__ import annotations

import unittest

import torch

from hmarl_mvp.config import get_default_config
from hmarl_mvp.networks import (
    ActorCritic,
    ContinuousActor,
    Critic,
    DiscreteActor,
    build_actor_critics,
    obs_dim_from_env,
)


class ContinuousActorTests(unittest.TestCase):
    def test_forward_returns_distribution(self) -> None:
        actor = ContinuousActor(obs_dim=4, act_dim=1)
        obs = torch.randn(2, 4)
        dist = actor(obs)
        self.assertIsInstance(dist, torch.distributions.Normal)
        self.assertEqual(dist.mean.shape, (2, 1))

    def test_get_action_shapes(self) -> None:
        actor = ContinuousActor(obs_dim=6, act_dim=2)
        obs = torch.randn(3, 6)
        action, log_prob = actor.get_action(obs)
        self.assertEqual(action.shape, (3, 2))
        self.assertEqual(log_prob.shape, (3,))

    def test_deterministic_action(self) -> None:
        actor = ContinuousActor(obs_dim=4, act_dim=1)
        obs = torch.randn(1, 4)
        a1, _ = actor.get_action(obs, deterministic=True)
        a2, _ = actor.get_action(obs, deterministic=True)
        self.assertTrue(torch.equal(a1, a2))

    def test_evaluate_shapes(self) -> None:
        actor = ContinuousActor(obs_dim=4, act_dim=2)
        obs = torch.randn(5, 4)
        actions = torch.randn(5, 2)
        log_prob, entropy = actor.evaluate(obs, actions)
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(entropy.shape, (5,))


class DiscreteActorTests(unittest.TestCase):
    def test_forward_returns_categorical(self) -> None:
        actor = DiscreteActor(obs_dim=4, num_actions=3)
        obs = torch.randn(2, 4)
        dist = actor(obs)
        self.assertIsInstance(dist, torch.distributions.Categorical)

    def test_get_action_shapes(self) -> None:
        actor = DiscreteActor(obs_dim=6, num_actions=5)
        obs = torch.randn(3, 6)
        action, log_prob = actor.get_action(obs)
        self.assertEqual(action.shape, (3,))
        self.assertEqual(log_prob.shape, (3,))

    def test_evaluate_shapes(self) -> None:
        actor = DiscreteActor(obs_dim=4, num_actions=3)
        obs = torch.randn(5, 4)
        actions = torch.tensor([0, 1, 2, 0, 1])
        log_prob, entropy = actor.evaluate(obs, actions)
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(entropy.shape, (5,))


class CriticTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        critic = Critic(state_dim=10)
        state = torch.randn(4, 10)
        value = critic(state)
        self.assertEqual(value.shape, (4,))


class ActorCriticTests(unittest.TestCase):
    def test_continuous_actor_critic(self) -> None:
        ac = ActorCritic(obs_dim=8, global_state_dim=20, act_dim=2, discrete=False)
        obs = torch.randn(3, 8)
        gs = torch.randn(3, 20)
        action, log_prob, value = ac.get_action_and_value(obs, gs)
        self.assertEqual(action.shape, (3, 2))
        self.assertEqual(log_prob.shape, (3,))
        self.assertEqual(value.shape, (3,))

    def test_discrete_actor_critic(self) -> None:
        ac = ActorCritic(obs_dim=8, global_state_dim=20, act_dim=5, discrete=True)
        obs = torch.randn(3, 8)
        gs = torch.randn(3, 20)
        action, log_prob, value = ac.get_action_and_value(obs, gs)
        self.assertEqual(action.shape, (3,))
        self.assertEqual(log_prob.shape, (3,))
        self.assertEqual(value.shape, (3,))

    def test_evaluate(self) -> None:
        ac = ActorCritic(obs_dim=4, global_state_dim=10, act_dim=1, discrete=False)
        obs = torch.randn(5, 4)
        gs = torch.randn(5, 10)
        actions = torch.randn(5, 1)
        log_prob, entropy, value = ac.evaluate(obs, gs, actions)
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(entropy.shape, (5,))
        self.assertEqual(value.shape, (5,))


class FactoryTests(unittest.TestCase):
    def test_obs_dim_from_env(self) -> None:
        cfg = get_default_config(
            num_ports=5, num_vessels=8, short_horizon_hours=12, medium_horizon_days=5
        )
        dims = obs_dim_from_env(cfg)
        self.assertEqual(dims["vessel"], 5 + 12 + 3)  # local(5) + forecast + directive
        self.assertEqual(dims["port"], 3 + 12 + 1)  # local + forecast + incoming
        self.assertEqual(dims["coordinator"], 5 * 5 + 8 * 4 + 1)  # forecast + vessels + total_em

    def test_build_actor_critics(self) -> None:
        cfg = get_default_config(num_ports=3, num_vessels=4, docks_per_port=2)
        dims = obs_dim_from_env(cfg)
        global_dim = sum(dims.values()) + cfg["num_ports"] + 1
        ac_dict = build_actor_critics(
            config=cfg,
            vessel_obs_dim=dims["vessel"],
            port_obs_dim=dims["port"],
            coordinator_obs_dim=dims["coordinator"],
            global_state_dim=global_dim,
            hidden_dims=[32, 32],
        )
        self.assertIn("vessel", ac_dict)
        self.assertIn("port", ac_dict)
        self.assertIn("coordinator", ac_dict)
        # Verify they can forward pass
        obs_v = torch.randn(1, dims["vessel"])
        gs = torch.randn(1, global_dim)
        action, lp, val = ac_dict["vessel"].get_action_and_value(obs_v, gs)
        self.assertEqual(action.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
