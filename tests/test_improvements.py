"""Tests for §12.2 recommended improvements implementation."""

from __future__ import annotations

import unittest

import torch

from hmarl_mvp.config import HMARLConfig
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.networks import (
    ActorCritic,
    AgentConditionedCritic,
    AttentionCoordinatorActor,
    ContinuousActor,
    Critic,
    EncodedCritic,
    RecurrentContinuousActor,
    build_actor_critics,
    build_per_agent_actor_critics,
)


# ---------------------------------------------------------------
# Short-term: Default configuration changes
# ---------------------------------------------------------------


class TestEntropyCoeffDefault(unittest.TestCase):
    """entropy_coeff default should be 0.08 for better exploration."""

    def test_default_is_008(self) -> None:
        cfg = MAPPOConfig()
        self.assertAlmostEqual(cfg.entropy_coeff, 0.08)

    def test_entropy_end_still_001(self) -> None:
        cfg = MAPPOConfig()
        self.assertAlmostEqual(cfg.entropy_coeff_end, 0.01)


class TestSlackHoursDefault(unittest.TestCase):
    """requested_arrival_slack_hours default should be 0.25 (tightened)."""

    def test_default_is_025(self) -> None:
        cfg = HMARLConfig()
        self.assertAlmostEqual(cfg.requested_arrival_slack_hours, 0.25)


# ---------------------------------------------------------------
# Short-term: Per-agent-type hidden dims
# ---------------------------------------------------------------


class TestPerAgentHiddenDims(unittest.TestCase):
    """Vessel gets [128,128] by default; per-agent-type overrides work."""

    def test_vessel_default_128(self) -> None:
        cfg = MAPPOConfig()
        self.assertEqual(cfg.vessel_hidden_dims, [128, 128])

    def test_port_coordinator_fallback_to_hidden_dims(self) -> None:
        cfg = MAPPOConfig()
        self.assertIsNone(cfg.port_hidden_dims)
        self.assertIsNone(cfg.coordinator_hidden_dims)

    def test_vessel_has_more_params_than_port(self) -> None:
        cfg = MAPPOConfig(rollout_length=5)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        vessel_params = sum(
            p.numel() for p in trainer.actor_critics["vessel"].parameters()
        )
        port_params = sum(
            p.numel() for p in trainer.actor_critics["port"].parameters()
        )
        self.assertGreater(vessel_params, port_params)

    def test_custom_per_type_dims(self) -> None:
        """Per-type overrides should be respected."""
        env_cfg = {"num_vessels": 2, "num_ports": 2, "docks_per_port": 2}
        cfg = MAPPOConfig(
            rollout_length=5,
            hidden_dims=[32, 32],
            vessel_hidden_dims=[64, 64],
            port_hidden_dims=[16, 16],
            coordinator_hidden_dims=[48, 48],
            # Disable architectural swaps so param counts reflect hidden_dims only
            coordinator_use_attention=False,
            use_encoded_critic=False,
            vessel_use_recurrence=False,
        )
        trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg, seed=42)
        v_params = sum(p.numel() for p in trainer.actor_critics["vessel"].parameters())
        p_params = sum(p.numel() for p in trainer.actor_critics["port"].parameters())
        c_params = sum(p.numel() for p in trainer.actor_critics["coordinator"].parameters())
        # Vessel (64) > coordinator (48) > port (16) in hidden size
        self.assertGreater(v_params, c_params)
        self.assertGreater(c_params, p_params)

    def test_per_type_dims_nonshared(self) -> None:
        """Per-type overrides work with parameter_sharing=False."""
        env_cfg = {"num_vessels": 2, "num_ports": 2, "docks_per_port": 2}
        cfg = MAPPOConfig(
            rollout_length=5,
            hidden_dims=[32, 32],
            vessel_hidden_dims=[64, 64],
            parameter_sharing=False,
        )
        trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg, seed=42)
        v0_params = sum(p.numel() for p in trainer.actor_critics["vessel_0"].parameters())
        p0_params = sum(p.numel() for p in trainer.actor_critics["port_0"].parameters())
        self.assertGreater(v0_params, p0_params)

    def test_build_actor_critics_per_type(self) -> None:
        """build_actor_critics respects per-type dims."""
        from hmarl_mvp.config import get_default_config

        cfg = get_default_config()
        nets = build_actor_critics(
            config=cfg,
            vessel_obs_dim=27,
            port_obs_dim=22,
            coordinator_obs_dim=132,
            global_state_dim=464,
            hidden_dims=[64, 64],
            vessel_hidden_dims=[128, 128],
        )
        # Vessel actor first layer should be 27→128, not 27→64
        first_layer = list(nets["vessel"].actor.mean_net.children())[0]
        self.assertEqual(first_layer.out_features, 128)

        # Port should still use default [64, 64]
        first_layer_p = list(nets["port"].actor.logit_net.children())[0]
        self.assertEqual(first_layer_p.out_features, 64)


# ---------------------------------------------------------------
# Short-term: Gradient norm warmup
# ---------------------------------------------------------------


class TestGradNormWarmup(unittest.TestCase):
    """Gradient norm warmup ramps from max_grad_norm_start → max_grad_norm."""

    def test_default_warmup_values(self) -> None:
        cfg = MAPPOConfig()
        self.assertAlmostEqual(cfg.max_grad_norm_start, 0.5)
        self.assertAlmostEqual(cfg.max_grad_norm, 2.0)
        self.assertAlmostEqual(cfg.grad_norm_warmup_fraction, 0.1)

    def test_warmup_when_total_iterations_zero(self) -> None:
        """With total_iterations=0, warmup is inactive → returns max_grad_norm."""
        cfg = MAPPOConfig(total_iterations=0, rollout_length=5)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        self.assertAlmostEqual(trainer.current_max_grad_norm, cfg.max_grad_norm)

    def test_warmup_ramps_up(self) -> None:
        """During warmup, grad norm starts tight and ramps up to max_grad_norm."""
        cfg = MAPPOConfig(
            total_iterations=100,
            max_grad_norm=2.0,
            max_grad_norm_start=0.5,
            grad_norm_warmup_fraction=0.1,
            rollout_length=5,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        # At iteration 0 (before any update), should be start value
        self.assertAlmostEqual(trainer.current_max_grad_norm, 0.5)

        # Simulate iterations
        trainer._iteration = 5  # 50% through warmup
        val = trainer.current_max_grad_norm
        self.assertGreater(val, 0.5)
        self.assertLess(val, 2.0)

        # After warmup (iteration >= 10)
        trainer._iteration = 10
        self.assertAlmostEqual(trainer.current_max_grad_norm, 2.0)

        # Well past warmup
        trainer._iteration = 50
        self.assertAlmostEqual(trainer.current_max_grad_norm, 2.0)

    def test_disabled_when_start_equals_end(self) -> None:
        """When max_grad_norm_start == max_grad_norm, warmup is effectively disabled."""
        cfg = MAPPOConfig(
            total_iterations=100,
            max_grad_norm=0.5,
            max_grad_norm_start=0.5,
            rollout_length=5,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        self.assertAlmostEqual(trainer.current_max_grad_norm, 0.5)


# ---------------------------------------------------------------
# Medium-term: Attention-based coordinator actor
# ---------------------------------------------------------------


class TestAttentionCoordinatorActor(unittest.TestCase):
    """Tests for the entity-attention coordinator architecture."""

    def setUp(self) -> None:
        self.num_vessels = 8
        self.num_ports = 5
        self.vessel_entity_dim = 7
        self.medium_horizon_days = 5
        self.port_entity_dim = self.medium_horizon_days + 5
        self.num_actions = 5
        self.actor = AttentionCoordinatorActor(
            num_vessels=self.num_vessels,
            num_ports=self.num_ports,
            vessel_entity_dim=self.vessel_entity_dim,
            port_entity_dim=self.port_entity_dim,
            global_feature_dim=1,
            num_actions=self.num_actions,
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            weather_enabled=True,
        )

    def test_forward_shape(self) -> None:
        """Forward pass returns correct distribution."""
        obs_dim = (
            1  # global feature
            + self.num_ports * self.num_ports  # weather
            + self.num_ports * self.port_entity_dim
            + self.num_vessels * self.vessel_entity_dim
        )
        obs = torch.randn(4, obs_dim)
        dist = self.actor(obs)
        self.assertEqual(dist.probs.shape, (4, self.num_actions))

    def test_get_action(self) -> None:
        obs_dim = (
            1
            + self.num_ports * self.num_ports
            + self.num_ports * self.port_entity_dim
            + self.num_vessels * self.vessel_entity_dim
        )
        obs = torch.randn(2, obs_dim)
        action, log_prob = self.actor.get_action(obs)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(log_prob.shape, (2,))

    def test_evaluate(self) -> None:
        obs_dim = (
            1
            + self.num_ports * self.num_ports
            + self.num_ports * self.port_entity_dim
            + self.num_vessels * self.vessel_entity_dim
        )
        obs = torch.randn(4, obs_dim)
        actions = torch.randint(0, self.num_actions, (4,))
        lp, ent = self.actor.evaluate(obs, actions)
        self.assertEqual(lp.shape, (4,))
        self.assertEqual(ent.shape, (4,))

    def test_action_mask(self) -> None:
        obs_dim = (
            1
            + self.num_ports * self.num_ports
            + self.num_ports * self.port_entity_dim
            + self.num_vessels * self.vessel_entity_dim
        )
        obs = torch.randn(2, obs_dim)
        mask = torch.zeros(2, self.num_actions, dtype=torch.bool)
        mask[:, 0] = True  # only action 0 valid
        dist = self.actor(obs, action_mask=mask)
        # All probability should be on action 0
        self.assertAlmostEqual(dist.probs[0, 0].item(), 1.0, places=4)

    def test_without_weather(self) -> None:
        actor = AttentionCoordinatorActor(
            num_vessels=4,
            num_ports=3,
            vessel_entity_dim=7,
            port_entity_dim=10,
            global_feature_dim=1,
            num_actions=3,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            weather_enabled=False,
        )
        obs_dim = 1 + 3 * 10 + 4 * 7
        obs = torch.randn(2, obs_dim)
        dist = actor(obs)
        self.assertEqual(dist.probs.shape, (2, 3))


# ---------------------------------------------------------------
# Medium-term: Encoded critic with per-type encoders
# ---------------------------------------------------------------


class TestEncodedCritic(unittest.TestCase):
    """Tests for the per-type-encoder critic architecture."""

    def setUp(self) -> None:
        self.critic = EncodedCritic(
            coordinator_obs_dim=132,
            vessel_obs_dim=27,
            port_obs_dim=22,
            num_vessels=8,
            num_ports=5,
            extra_dims=6,
            encoder_dim=32,
            hidden_dims=[32, 32],
        )

    def test_forward_shape(self) -> None:
        state_dim = 132 + 8 * 27 + 5 * 22 + 6
        state = torch.randn(4, state_dim)
        values = self.critic(state)
        self.assertEqual(values.shape, (4,))

    def test_gradient_flow(self) -> None:
        """Gradients flow back through all encoders."""
        state_dim = 132 + 8 * 27 + 5 * 22 + 6
        state = torch.randn(2, state_dim)
        values = self.critic(state)
        loss = values.mean()
        loss.backward()
        for name, p in self.critic.named_parameters():
            self.assertIsNotNone(p.grad, msg=f"No gradient for {name}")

    def test_different_dimensions(self) -> None:
        """Works with non-default dimensions."""
        critic = EncodedCritic(
            coordinator_obs_dim=50,
            vessel_obs_dim=10,
            port_obs_dim=8,
            num_vessels=4,
            num_ports=3,
            extra_dims=4,
            encoder_dim=16,
        )
        state_dim = 50 + 4 * 10 + 3 * 8 + 4
        state = torch.randn(3, state_dim)
        values = critic(state)
        self.assertEqual(values.shape, (3,))


# ---------------------------------------------------------------
# Medium-term: GRU-based vessel policy
# ---------------------------------------------------------------


class TestRecurrentContinuousActor(unittest.TestCase):
    """Tests for the GRU-based recurrent vessel actor."""

    def setUp(self) -> None:
        self.actor = RecurrentContinuousActor(
            obs_dim=27,
            act_dim=2,
            hidden_size=32,
        )

    def test_forward_shape(self) -> None:
        obs = torch.randn(4, 27)
        dist, hidden = self.actor(obs)
        self.assertEqual(dist.mean.shape, (4, 2))
        self.assertEqual(hidden.shape, (1, 4, 32))

    def test_forward_with_sequence(self) -> None:
        """Handles sequential observations (B, T, obs_dim)."""
        obs = torch.randn(4, 10, 27)
        dist, hidden = self.actor(obs)
        self.assertEqual(dist.mean.shape, (4, 2))
        self.assertEqual(hidden.shape, (1, 4, 32))

    def test_hidden_state_persistence(self) -> None:
        """Hidden state from one call feeds into the next."""
        obs1 = torch.randn(2, 27)
        obs2 = torch.randn(2, 27)

        # Without hidden
        dist1, h1 = self.actor(obs1)
        # With hidden from previous step
        dist2, h2 = self.actor(obs2, hidden=h1)
        # Hidden should be different
        self.assertFalse(torch.allclose(h1, h2))

    def test_get_action(self) -> None:
        obs = torch.randn(3, 27)
        action, log_prob, hidden = self.actor.get_action(obs)
        self.assertEqual(action.shape, (3, 2))
        self.assertEqual(log_prob.shape, (3,))
        self.assertEqual(hidden.shape, (1, 3, 32))

    def test_evaluate(self) -> None:
        obs = torch.randn(4, 27)
        actions = torch.randn(4, 2)
        log_prob, entropy = self.actor.evaluate(obs, actions)
        self.assertEqual(log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))

    def test_init_hidden(self) -> None:
        hidden = self.actor.init_hidden(batch_size=5)
        self.assertEqual(hidden.shape, (1, 5, 32))
        self.assertTrue(torch.all(hidden == 0))

    def test_deterministic_action(self) -> None:
        obs = torch.randn(2, 27)
        a1, _, _ = self.actor.get_action(obs, deterministic=True)
        a2, _, _ = self.actor.get_action(obs, deterministic=True)
        self.assertTrue(torch.allclose(a1, a2))


# ---------------------------------------------------------------
# Config flag existence
# ---------------------------------------------------------------


class TestMAPPOConfigFlags(unittest.TestCase):
    """New architectural flags exist with correct defaults."""

    def test_coordinator_use_attention(self) -> None:
        cfg = MAPPOConfig()
        self.assertTrue(cfg.coordinator_use_attention)

    def test_use_encoded_critic(self) -> None:
        cfg = MAPPOConfig()
        self.assertTrue(cfg.use_encoded_critic)

    def test_vessel_use_recurrence(self) -> None:
        cfg = MAPPOConfig()
        self.assertTrue(cfg.vessel_use_recurrence)


# ---------------------------------------------------------------
# Integration: training still works with new defaults
# ---------------------------------------------------------------


class TestTrainingWithNewDefaults(unittest.TestCase):
    """Full MAPPO training loop works with updated defaults."""

    def test_collect_and_update(self) -> None:
        """One collect+update cycle completes without error."""
        cfg = MAPPOConfig(
            rollout_length=4,
            num_epochs=2,
            minibatch_size=8,
            # Use new defaults: entropy=0.05, vessel_hidden_dims=[128,128]
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        rollout_info = trainer.collect_rollout()
        self.assertIn("mean_reward", rollout_info)
        update_info = trainer.update()
        self.assertIn("vessel", update_info)

    def test_grad_norm_warmup_in_training(self) -> None:
        """Grad norm warmup integrates with the training loop."""
        cfg = MAPPOConfig(
            rollout_length=4,
            num_epochs=2,
            minibatch_size=8,
            total_iterations=10,
            max_grad_norm_start=5.0,
            max_grad_norm=0.5,
            grad_norm_warmup_fraction=0.5,
        )
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        # Before training, grad norm should be at start
        self.assertAlmostEqual(trainer.current_max_grad_norm, 5.0)
        # Run one iteration
        trainer.collect_rollout()
        trainer.update()
        # After _step_lr increments _iteration, warmup should have progressed
        self.assertGreater(trainer.current_max_grad_norm, 0.5)
        self.assertLess(trainer.current_max_grad_norm, 5.0)


class TestArchitecturalFlagsIntegration(unittest.TestCase):
    """Integration tests: architectural flags wire through MAPPOTrainer."""

    def _make_trainer(self, **kwargs: object) -> MAPPOTrainer:
        cfg = MAPPOConfig(
            rollout_length=4,
            num_epochs=1,
            minibatch_size=8,
            **kwargs,
        )
        return MAPPOTrainer(mappo_config=cfg, seed=42)

    def test_coordinator_use_attention_builds_and_trains(self) -> None:
        trainer = self._make_trainer(coordinator_use_attention=True)
        ac = trainer.actor_critics["coordinator"]
        self.assertIsInstance(ac.actor, AttentionCoordinatorActor)
        # Verify one rollout + update completes without error
        trainer.collect_rollout()
        trainer.update()

    def test_use_encoded_critic_builds_and_trains(self) -> None:
        trainer = self._make_trainer(use_encoded_critic=True)
        # Vessel gets AgentConditionedCritic (wraps EncodedCritic)
        self.assertIsInstance(
            trainer.actor_critics["vessel"].critic, AgentConditionedCritic,
            "vessel critic should be AgentConditionedCritic",
        )
        for name in ("port", "coordinator"):
            self.assertIsInstance(
                trainer.actor_critics[name].critic, EncodedCritic,
                f"{name} critic should be EncodedCritic",
            )
        trainer.collect_rollout()
        trainer.update()

    def test_vessel_use_recurrence_builds_and_trains(self) -> None:
        trainer = self._make_trainer(vessel_use_recurrence=True)
        ac = trainer.actor_critics["vessel"]
        self.assertIsInstance(ac.actor, RecurrentContinuousActor)
        trainer.collect_rollout()
        trainer.update()

    def test_all_three_flags_combined(self) -> None:
        trainer = self._make_trainer(
            coordinator_use_attention=True,
            use_encoded_critic=True,
            vessel_use_recurrence=True,
        )
        self.assertIsInstance(
            trainer.actor_critics["coordinator"].actor, AttentionCoordinatorActor,
        )
        self.assertIsInstance(
            trainer.actor_critics["vessel"].actor, RecurrentContinuousActor,
        )
        # Vessel gets AgentConditionedCritic (wraps EncodedCritic)
        self.assertIsInstance(
            trainer.actor_critics["vessel"].critic, AgentConditionedCritic,
        )
        for name in ("port", "coordinator"):
            self.assertIsInstance(
                trainer.actor_critics[name].critic, EncodedCritic,
            )
        trainer.collect_rollout()
        trainer.update()

    def test_build_actor_critics_attention_flag(self) -> None:
        cfg = HMARLConfig().to_dict()
        nets = build_actor_critics(
            config=cfg,
            vessel_obs_dim=27,
            port_obs_dim=22,
            coordinator_obs_dim=132,
            global_state_dim=464,
            coordinator_use_attention=True,
        )
        self.assertIsInstance(nets["coordinator"].actor, AttentionCoordinatorActor)
        # vessel and port remain standard
        self.assertIsInstance(nets["vessel"].actor, ContinuousActor)

    def test_build_actor_critics_encoded_critic_flag(self) -> None:
        cfg = HMARLConfig().to_dict()
        nets = build_actor_critics(
            config=cfg,
            vessel_obs_dim=27,
            port_obs_dim=22,
            coordinator_obs_dim=132,
            global_state_dim=464,
            use_encoded_critic=True,
        )
        # Vessel gets AgentConditionedCritic, others get EncodedCritic
        self.assertIsInstance(nets["vessel"].critic, AgentConditionedCritic)
        for name in ("port", "coordinator"):
            self.assertIsInstance(nets[name].critic, EncodedCritic)

    def test_build_actor_critics_recurrence_flag(self) -> None:
        cfg = HMARLConfig().to_dict()
        nets = build_actor_critics(
            config=cfg,
            vessel_obs_dim=27,
            port_obs_dim=22,
            coordinator_obs_dim=132,
            global_state_dim=464,
            vessel_use_recurrence=True,
        )
        self.assertIsInstance(nets["vessel"].actor, RecurrentContinuousActor)


if __name__ == "__main__":
    unittest.main()
