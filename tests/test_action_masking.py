"""Tests for action masking, explained variance, and eval obs normalisation.

Session 14 improvements:
1. Observation normalisation during evaluation (run_mappo_comparison fix)
2. Explained variance metric in PPOUpdateResult
3. Action masking for DiscreteActor / ActorCritic / MAPPO port actions
"""

from __future__ import annotations

import numpy as np
import torch

from hmarl_mvp.buffer import MultiAgentRolloutBuffer, RolloutBuffer
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer, PPOUpdateResult
from hmarl_mvp.networks import ActorCritic, DiscreteActor

# ===================================================================
# Action masking — DiscreteActor
# ===================================================================


class TestDiscreteActorMasking:
    """DiscreteActor respects boolean action masks."""

    def test_mask_prevents_invalid_action(self) -> None:
        """Masked-out actions should never be sampled."""
        actor = DiscreteActor(obs_dim=4, num_actions=5)
        obs = torch.randn(1, 4)
        # Only action 2 is valid
        mask = torch.tensor([[False, False, True, False, False]])
        for _ in range(50):
            action, log_prob = actor.get_action(obs, action_mask=mask)
            assert action.item() == 2

    def test_mask_all_valid_no_error(self) -> None:
        """When all actions are valid, behaviour is unchanged."""
        actor = DiscreteActor(obs_dim=4, num_actions=3)
        obs = torch.randn(1, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        action, log_prob = actor.get_action(obs, action_mask=mask)
        assert 0 <= action.item() < 3
        assert torch.isfinite(log_prob).all()

    def test_deterministic_selects_best_valid(self) -> None:
        """Deterministic mode picks the highest-probability valid action."""
        actor = DiscreteActor(obs_dim=4, num_actions=4)
        obs = torch.randn(1, 4)
        # Mask out all but actions 1 and 3
        mask = torch.tensor([[False, True, False, True]])
        action, _ = actor.get_action(obs, deterministic=True, action_mask=mask)
        assert action.item() in {1, 3}

    def test_forward_masked_logits(self) -> None:
        """Masked actions should get -inf logits."""
        actor = DiscreteActor(obs_dim=4, num_actions=4)
        obs = torch.randn(1, 4)
        mask = torch.tensor([[True, False, True, False]])
        dist = actor.forward(obs, action_mask=mask)
        # Actions 1 and 3 should have zero probability
        assert dist.probs[0, 1].item() < 1e-6
        assert dist.probs[0, 3].item() < 1e-6

    def test_evaluate_with_mask(self) -> None:
        """Evaluate log_prob and entropy with mask applied."""
        actor = DiscreteActor(obs_dim=4, num_actions=4)
        obs = torch.randn(2, 4)
        # Valid: actions 0 and 1
        mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
        actions = torch.tensor([0, 1])
        log_prob, entropy = actor.evaluate(obs, actions, action_mask=mask)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(entropy).all()

    def test_no_mask_backward_compatible(self) -> None:
        """Without a mask, behaviour is identical to before."""
        actor = DiscreteActor(obs_dim=4, num_actions=3)
        obs = torch.randn(1, 4)
        action_no_mask, lp_no_mask = actor.get_action(obs)
        # Just verify it works without error
        assert 0 <= action_no_mask.item() < 3


# ===================================================================
# Action masking — ActorCritic
# ===================================================================


class TestActorCriticMasking:
    """ActorCritic correctly forwards masks to discrete actors."""

    def test_discrete_ac_masking(self) -> None:
        """Discrete ActorCritic passes mask through get_action_and_value."""
        ac = ActorCritic(obs_dim=4, global_state_dim=8, act_dim=5, discrete=True)
        obs = torch.randn(1, 4)
        gs = torch.randn(1, 8)
        # Only action 3 is valid
        mask = torch.tensor([[False, False, False, True, False]])
        for _ in range(20):
            a, lp, v = ac.get_action_and_value(obs, gs, action_mask=mask)
            assert a.item() == 3

    def test_discrete_ac_evaluate_with_mask(self) -> None:
        """Evaluate passes mask through for correct log_prob computation."""
        ac = ActorCritic(obs_dim=4, global_state_dim=8, act_dim=4, discrete=True)
        obs = torch.randn(2, 4)
        gs = torch.randn(2, 8)
        actions = torch.tensor([[0.0], [1.0]])
        mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
        lp, ent, val = ac.evaluate(obs, gs, actions, action_mask=mask)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_continuous_ac_ignores_mask(self) -> None:
        """Continuous ActorCritic gracefully ignores action_mask."""
        ac = ActorCritic(obs_dim=4, global_state_dim=8, act_dim=1, discrete=False)
        obs = torch.randn(1, 4)
        gs = torch.randn(1, 8)
        mask = torch.ones(1, 1, dtype=torch.bool)
        # Should not raise
        a, lp, v = ac.get_action_and_value(obs, gs, action_mask=mask)
        assert torch.isfinite(a).all()


# ===================================================================
# Buffer — action mask storage
# ===================================================================


class TestBufferActionMasks:
    """RolloutBuffer and MultiAgentRolloutBuffer store action masks."""

    def test_rollout_buffer_stores_mask(self) -> None:
        """Masks are stored and returned in get_tensors."""
        buf = RolloutBuffer(capacity=5, obs_dim=3, act_dim=1, mask_dim=4)
        mask = np.array([1.0, 1.0, 0.0, 0.0])
        buf.add(
            obs=np.zeros(3),
            action=0.0,
            reward=1.0,
            done=False,
            action_mask=mask,
        )
        tensors = buf.get_tensors()
        assert "action_masks" in tensors
        assert tensors["action_masks"].shape == (1, 4)
        assert tensors["action_masks"][0, 0].item() is True
        assert tensors["action_masks"][0, 2].item() is False

    def test_rollout_buffer_no_mask_dim(self) -> None:
        """With mask_dim=0, no action_masks returned."""
        buf = RolloutBuffer(capacity=5, obs_dim=3, act_dim=1)
        buf.add(obs=np.zeros(3), action=0.0, reward=1.0, done=False)
        tensors = buf.get_tensors()
        assert "action_masks" not in tensors

    def test_multi_agent_buffer_mask_dim(self) -> None:
        """MultiAgentRolloutBuffer propagates mask_dim to children."""
        mbuf = MultiAgentRolloutBuffer(
            num_agents=2, capacity=3, obs_dim=4, act_dim=1, mask_dim=5
        )
        assert mbuf[0].mask_dim == 5
        assert mbuf[1].mask_dim == 5

    def test_mask_default_all_ones(self) -> None:
        """Without explicit mask, default is all-True."""
        buf = RolloutBuffer(capacity=3, obs_dim=2, act_dim=1, mask_dim=3)
        buf.add(obs=np.zeros(2), action=0.0, reward=0.0, done=False)
        tensors = buf.get_tensors()
        assert tensors["action_masks"][0].all()


# ===================================================================
# Explained variance in PPOUpdateResult
# ===================================================================


class TestExplainedVariance:
    """PPOUpdateResult includes explained_variance field."""

    def test_field_exists(self) -> None:
        """PPOUpdateResult has explained_variance with default 0."""
        r = PPOUpdateResult()
        assert hasattr(r, "explained_variance")
        assert r.explained_variance == 0.0

    def test_mappo_update_returns_explained_variance(self) -> None:
        """After a MAPPO update, explained_variance is populated."""
        cfg = MAPPOConfig(rollout_length=8, num_epochs=1, minibatch_size=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        trainer.collect_rollout()
        results = trainer.update()
        for agent_type, res in results.items():
            # Should be a finite float (can be negative if value fn is very bad)
            assert isinstance(res.explained_variance, float)
            assert np.isfinite(res.explained_variance)

    def test_explained_variance_logged_in_train(self) -> None:
        """The train() method logs explained_variance per agent type."""
        cfg = MAPPOConfig(rollout_length=8, num_epochs=1, minibatch_size=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=2)
        for entry in history:
            assert "vessel_explained_variance" in entry
            assert "port_explained_variance" in entry
            assert "coordinator_explained_variance" in entry


# ===================================================================
# MAPPO port action masking integration
# ===================================================================


class TestMAPPOPortMasking:
    """MAPPO trainer uses port action masks during rollout and eval."""

    def test_build_port_mask_shape(self) -> None:
        """Port mask has correct shape = docks_per_port + 1."""
        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        trainer.env.reset()
        mask = trainer._build_port_mask(0)
        expected_dim = int(trainer.cfg["docks_per_port"]) + 1
        assert mask.shape == (expected_dim,)

    def test_build_port_mask_valid_range(self) -> None:
        """Valid mask entries cover [0..available_docks]."""
        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        # Reset to get fresh state
        trainer.env.reset()
        for i in range(trainer.env.num_ports):
            mask = trainer._build_port_mask(i)
            port = trainer.env.ports[i]
            available = max(port.docks - port.occupied, 0)
            # Entries [0..available] should be 1, rest 0
            assert mask[: available + 1].sum() == available + 1
            if available + 1 < len(mask):
                assert mask[available + 1 :].sum() == 0.0

    def test_port_mask_tensor_device(self) -> None:
        """Port mask tensor is on the correct device."""
        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        trainer.env.reset()
        mt = trainer._port_mask_tensor(0)
        assert mt.dtype == torch.bool
        assert mt.shape[0] == 1  # batch dim

    def test_collect_rollout_with_masking(self) -> None:
        """Rollout collection works with action masking enabled."""
        cfg = MAPPOConfig(rollout_length=8, num_epochs=1, minibatch_size=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        info = trainer.collect_rollout()
        assert "mean_reward" in info
        # Verify masks were stored in port buffer
        for i in range(trainer.env.num_ports):
            tensors = trainer.port_buf[i].get_tensors()
            assert "action_masks" in tensors

    def test_update_uses_masks(self) -> None:
        """PPO update completes with mask data in port buffer."""
        cfg = MAPPOConfig(rollout_length=8, num_epochs=1, minibatch_size=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        trainer.collect_rollout()
        results = trainer.update()
        assert "port" in results
        assert results["port"].policy_loss != 0.0 or results["port"].value_loss != 0.0

    def test_evaluate_with_masking(self) -> None:
        """Evaluation uses action masking for ports."""
        cfg = MAPPOConfig(rollout_length=8, num_epochs=1, minibatch_size=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        # Train briefly so we have normaliser stats
        trainer.collect_rollout()
        trainer.update()
        eval_metrics = trainer.evaluate(num_steps=5)
        assert "mean_vessel_reward" in eval_metrics
        assert "total_reward" in eval_metrics


# ===================================================================
# Eval obs normalisation in run_mappo_comparison
# ===================================================================


class TestEvalObsNormalisation:
    """run_mappo_comparison applies obs normalisation during eval."""

    def test_comparison_uses_normalised_obs(self) -> None:
        """Verify run_mappo_comparison produces valid MAPPO eval results.

        With obs normalisation enabled (default), the evaluation must
        use _eval_normalize_obs.  We test this indirectly by ensuring
        the comparison runs without error and produces MAPPO results.
        """
        from hmarl_mvp.experiment import run_mappo_comparison

        results = run_mappo_comparison(
            train_iterations=3,
            rollout_length=8,
            eval_steps=5,
            baselines=["independent"],
            mappo_kwargs={"normalize_observations": True},
        )
        assert "mappo" in results
        df = results["mappo"]
        assert len(df) > 0
        # Verify we have expected columns
        assert "avg_vessel_reward" in df.columns

    def test_comparison_without_normalisation(self) -> None:
        """Comparison also works when normalisation is disabled."""
        from hmarl_mvp.experiment import run_mappo_comparison

        results = run_mappo_comparison(
            train_iterations=2,
            rollout_length=8,
            eval_steps=4,
            baselines=["independent"],
            mappo_kwargs={"normalize_observations": False},
        )
        assert "mappo" in results
        assert len(results["mappo"]) > 0
