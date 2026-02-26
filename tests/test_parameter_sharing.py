"""Tests for parameter sharing toggle and per-agent network builder."""

from __future__ import annotations

import pytest

from hmarl_mvp.config import get_default_config
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.networks import build_actor_critics, build_per_agent_actor_critics, obs_dim_from_env

# ------------------------------------------------------------------
# build_per_agent_actor_critics
# ------------------------------------------------------------------


class TestBuildPerAgentActorCritics:
    """Tests for build_per_agent_actor_critics network factory."""

    @pytest.fixture()
    def cfg(self) -> dict:
        return get_default_config()

    def test_creates_correct_keys(self, cfg: dict) -> None:
        dims = obs_dim_from_env(cfg)
        nets = build_per_agent_actor_critics(
            config=cfg,
            vessel_obs_dim=dims["vessel"],
            port_obs_dim=dims["port"],
            coordinator_obs_dim=dims["coordinator"],
            global_state_dim=10,
        )
        nv = cfg["num_vessels"]
        np_ = cfg["num_ports"]
        nc = cfg.get("num_coordinators", 1)
        expected_keys = (
            [f"vessel_{i}" for i in range(nv)]
            + [f"port_{i}" for i in range(np_)]
            + [f"coordinator_{i}" for i in range(nc)]
        )
        assert set(nets.keys()) == set(expected_keys)

    def test_separate_parameters(self, cfg: dict) -> None:
        dims = obs_dim_from_env(cfg)
        nets = build_per_agent_actor_critics(
            config=cfg,
            vessel_obs_dim=dims["vessel"],
            port_obs_dim=dims["port"],
            coordinator_obs_dim=dims["coordinator"],
            global_state_dim=10,
        )
        if cfg["num_vessels"] >= 2:
            p0 = list(nets["vessel_0"].parameters())
            p1 = list(nets["vessel_1"].parameters())
            # They should be different objects
            assert p0[0] is not p1[0]

    def test_more_networks_than_shared(self, cfg: dict) -> None:
        dims = obs_dim_from_env(cfg)
        shared = build_actor_critics(
            config=cfg,
            vessel_obs_dim=dims["vessel"],
            port_obs_dim=dims["port"],
            coordinator_obs_dim=dims["coordinator"],
            global_state_dim=10,
        )
        per_agent = build_per_agent_actor_critics(
            config=cfg,
            vessel_obs_dim=dims["vessel"],
            port_obs_dim=dims["port"],
            coordinator_obs_dim=dims["coordinator"],
            global_state_dim=10,
        )
        assert len(per_agent) >= len(shared)

    def test_custom_hidden_dims(self, cfg: dict) -> None:
        dims = obs_dim_from_env(cfg)
        nets = build_per_agent_actor_critics(
            config=cfg,
            vessel_obs_dim=dims["vessel"],
            port_obs_dim=dims["port"],
            coordinator_obs_dim=dims["coordinator"],
            global_state_dim=10,
            hidden_dims=[32],
        )
        # Just verify it creates networks without error
        assert len(nets) > 0
        for ac in nets.values():
            assert sum(p.numel() for p in ac.parameters()) > 0


# ------------------------------------------------------------------
# MAPPOConfig parameter_sharing flag
# ------------------------------------------------------------------


class TestParameterSharingConfig:
    """Tests for the parameter_sharing config flag."""

    def test_default_is_shared(self) -> None:
        cfg = MAPPOConfig()
        assert cfg.parameter_sharing is True

    def test_can_disable(self) -> None:
        cfg = MAPPOConfig(parameter_sharing=False)
        assert cfg.parameter_sharing is False


# ------------------------------------------------------------------
# MAPPOTrainer with parameter_sharing=False
# ------------------------------------------------------------------


class TestTrainerWithoutSharing:
    """Integration tests for MAPPOTrainer with per-agent networks."""

    def test_init_creates_per_agent_keys(self) -> None:
        mcfg = MAPPOConfig(parameter_sharing=False, rollout_length=5)
        trainer = MAPPOTrainer(mappo_config=mcfg, seed=42)
        # Should have individual agent keys, not type keys
        keys = set(trainer.actor_critics.keys())
        assert "vessel" not in keys  # no type-level key
        assert any(k.startswith("vessel_") for k in keys)
        assert any(k.startswith("port_") for k in keys)
        assert any(k.startswith("coordinator_") for k in keys)

    def test_collect_rollout_works(self) -> None:
        mcfg = MAPPOConfig(parameter_sharing=False, rollout_length=5)
        trainer = MAPPOTrainer(mappo_config=mcfg, seed=42)
        info = trainer.collect_rollout()
        assert "mean_reward" in info
        assert isinstance(info["mean_reward"], float)

    def test_update_works(self) -> None:
        mcfg = MAPPOConfig(parameter_sharing=False, rollout_length=5)
        trainer = MAPPOTrainer(mappo_config=mcfg, seed=42)
        trainer.collect_rollout()
        results = trainer.update()
        # Should return type-level keys (aggregated)
        assert "vessel" in results
        assert "port" in results
        assert "coordinator" in results
        assert results["vessel"].policy_loss is not None

    def test_evaluate_works(self) -> None:
        mcfg = MAPPOConfig(parameter_sharing=False, rollout_length=5)
        trainer = MAPPOTrainer(mappo_config=mcfg, seed=42)
        metrics = trainer.evaluate(num_steps=3)
        assert "mean_vessel_reward" in metrics

    def test_train_loop_works(self) -> None:
        mcfg = MAPPOConfig(parameter_sharing=False, rollout_length=5)
        trainer = MAPPOTrainer(mappo_config=mcfg, seed=42)
        history = trainer.train(num_iterations=2)
        assert len(history) == 2
        assert "mean_reward" in history[0]
        # Verify per-agent-type loss keys are present
        assert "vessel_policy_loss" in history[0]

    def test_shared_vs_nonshared_param_count(self) -> None:
        shared_trainer = MAPPOTrainer(
            mappo_config=MAPPOConfig(parameter_sharing=True, rollout_length=5),
            seed=42,
        )
        nonshared_trainer = MAPPOTrainer(
            mappo_config=MAPPOConfig(parameter_sharing=False, rollout_length=5),
            seed=42,
        )
        shared_params = sum(
            sum(p.numel() for p in ac.parameters())
            for ac in shared_trainer.actor_critics.values()
        )
        nonshared_params = sum(
            sum(p.numel() for p in ac.parameters())
            for ac in nonshared_trainer.actor_critics.values()
        )
        # Non-shared should have more total params (separate per agent)
        assert nonshared_params >= shared_params
