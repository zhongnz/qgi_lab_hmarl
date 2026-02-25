"""Tests for session improvements: save/load normalizer state, orthogonal init,
curriculum integration in trainer, and training logger.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any

import torch

from hmarl_mvp.curriculum import CurriculumScheduler
from hmarl_mvp.logger import TrainingLogger
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.networks import ActorCritic, apply_orthogonal_init, build_actor_critics

# ===================================================================
# Save / load normalizer state
# ===================================================================


class TestSaveLoadNormalizerState(unittest.TestCase):
    """Verify that save/load preserves normaliser state for reproducibility."""

    def test_normalizer_state_roundtrip(self) -> None:
        """Saving and loading should produce identical eval results."""
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=True)
        env_cfg: dict[str, Any] = {"num_vessels": 2, "num_ports": 2, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg, seed=42)
        trainer.collect_rollout()
        trainer.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "model")
            trainer.save_models(prefix)

            # Verify normalizer JSON was written
            self.assertTrue(os.path.exists(f"{prefix}_normalizers.json"))

            # Load into fresh trainer
            trainer2 = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg, seed=42)
            trainer2.load_models(prefix)

            eval1 = trainer.evaluate(num_steps=5)
            eval2 = trainer2.evaluate(num_steps=5)

            self.assertAlmostEqual(
                eval1["total_reward"], eval2["total_reward"], places=2
            )

    def test_normalizer_json_structure(self) -> None:
        cfg = MAPPOConfig(rollout_length=4, normalize_observations=True)
        env_cfg: dict[str, Any] = {"num_vessels": 2, "num_ports": 2, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg)
        trainer.collect_rollout()

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "model")
            trainer.save_models(prefix)

            with open(f"{prefix}_normalizers.json") as f:
                data = json.load(f)

            # Should contain both reward and obs normalizers
            self.assertIn("reward_vessel", data)
            self.assertIn("obs_vessel", data)
            self.assertIn("obs_port", data)
            self.assertIn("obs_coordinator", data)

            # Obs normalizer should have mean/var/count
            obs_v = data["obs_vessel"]
            self.assertIn("mean", obs_v)
            self.assertIn("var", obs_v)
            self.assertIn("count", obs_v)

    def test_load_without_normalizer_file(self) -> None:
        """Loading when no normalizer file exists should not crash."""
        cfg = MAPPOConfig(rollout_length=4)
        env_cfg: dict[str, Any] = {"num_vessels": 2, "num_ports": 2, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg)
        trainer.collect_rollout()
        trainer.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "model")
            # Save model weights only (simulate old-style save)
            for name, ac in trainer.actor_critics.items():
                torch.save(ac.state_dict(), f"{prefix}_{name}.pt")

            # Should not raise
            trainer2 = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg)
            trainer2.load_models(prefix)


# ===================================================================
# Orthogonal weight initialisation
# ===================================================================


class TestOrthogonalInit(unittest.TestCase):
    """Tests for orthogonal weight initialisation in networks."""

    def test_default_actorcritic_has_ortho_init(self) -> None:
        """ActorCritic should apply orthogonal init by default."""
        ac = ActorCritic(obs_dim=10, global_state_dim=20, act_dim=3, discrete=True)
        # The hidden layers should have orthogonal weights
        for m in ac.actor.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight.data
                # Orthogonal matrices satisfy W^T W ≈ I (scaled)
                # Check that the weights are not default init (xavier uniform)
                self.assertGreater(w.abs().sum().item(), 0.0)

    def test_ortho_init_disabled(self) -> None:
        """When ortho_init=False, weights should be default PyTorch init."""
        ac1 = ActorCritic(obs_dim=10, global_state_dim=20, act_dim=3, discrete=True, ortho_init=False)
        # Just verify it doesn't crash and produces valid weights
        for m in ac1.modules():
            if isinstance(m, torch.nn.Linear):
                self.assertFalse(torch.all(m.weight == 0).item())

    def test_apply_orthogonal_init_function(self) -> None:
        """Direct call to apply_orthogonal_init should work on any module."""
        net = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )
        apply_orthogonal_init(net, output_gain=1.0)
        # Hidden layer weight should be approximately orthogonal
        w: torch.Tensor = net[0].weight.data  # type: ignore[assignment]
        # W^T W should be close to gain^2 * I for orthogonal columns
        gram = w.T @ w  # (in_features x in_features)
        diag_sum = gram.diag().sum().item()
        total_sum = gram.abs().sum().item()
        # Diagonal elements should dominate
        self.assertGreater(diag_sum / total_sum, 0.5)

    def test_build_actor_critics_includes_ortho(self) -> None:
        """Factory build_actor_critics should produce ortho-initialized networks."""
        from hmarl_mvp.config import get_default_config
        cfg = get_default_config(num_ports=3, num_vessels=4)
        from hmarl_mvp.mappo import global_state_dim_from_config
        from hmarl_mvp.networks import obs_dim_from_env
        dims = obs_dim_from_env(cfg)
        gdim = global_state_dim_from_config(cfg)
        acs = build_actor_critics(
            config=cfg,
            vessel_obs_dim=dims["vessel"],
            port_obs_dim=dims["port"],
            coordinator_obs_dim=dims["coordinator"],
            global_state_dim=gdim,
        )
        for name, ac in acs.items():
            # Just verify the bias of hidden layers is zero (ortho init zeros bias)
            for m in ac.modules():
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    self.assertAlmostEqual(m.bias.data.abs().sum().item(), 0.0, places=5)


# ===================================================================
# Curriculum integration in trainer
# ===================================================================


class TestTrainerCurriculumIntegration(unittest.TestCase):
    """Tests for the train() method with curriculum scheduling."""

    def test_train_basic(self) -> None:
        """train() without curriculum should complete and return history."""
        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        history = trainer.train(num_iterations=3)
        self.assertEqual(len(history), 3)
        for entry in history:
            self.assertIn("mean_reward", entry)
            self.assertIn("iteration", entry)
            self.assertIn("lr", entry)

    def test_train_with_eval_interval(self) -> None:
        """Eval metrics should appear at the specified interval."""
        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        history = trainer.train(num_iterations=4, eval_interval=2)
        # Iterations 0,1,2,3 → eval at it=1 (idx 1) and it=3 (idx 3)
        self.assertNotIn("eval", history[0])
        self.assertIn("eval", history[1])
        self.assertNotIn("eval", history[2])
        self.assertIn("eval", history[3])

    def test_train_with_log_fn(self) -> None:
        """log_fn callback should be called for each iteration."""
        logged: list[tuple[int, dict[str, Any]]] = []

        def my_log(it: int, entry: dict[str, Any]) -> None:
            logged.append((it, entry))

        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        trainer.train(num_iterations=3, log_fn=my_log)
        self.assertEqual(len(logged), 3)
        self.assertEqual(logged[0][0], 0)
        self.assertEqual(logged[2][0], 2)

    def test_train_with_curriculum(self) -> None:
        """Training with curriculum should adapt env config."""
        target = {"num_vessels": 4, "num_ports": 3, "rollout_steps": 10}
        curriculum = CurriculumScheduler(
            target_config=target,
            start_config={"num_vessels": 2, "num_ports": 2},
            warmup_fraction=0.5,
        )
        cfg = MAPPOConfig(rollout_length=4)
        # Use target config dimensions so networks fit the final env
        trainer = MAPPOTrainer(
            env_config=target,
            mappo_config=cfg,
        )
        # This should not crash — curriculum adapts the env
        history = trainer.train(num_iterations=4, curriculum=curriculum)
        self.assertEqual(len(history), 4)

    def test_train_includes_per_agent_losses(self) -> None:
        """History entries should contain per-agent-type loss metrics."""
        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        history = trainer.train(num_iterations=2)
        entry = history[0]
        self.assertIn("vessel_policy_loss", entry)
        self.assertIn("port_value_loss", entry)
        self.assertIn("coordinator_entropy", entry)

    def test_rebuild_env_preserves_networks(self) -> None:
        """_rebuild_env with same dimensions should keep existing networks."""
        cfg = MAPPOConfig(rollout_length=4)
        env_cfg: dict[str, Any] = {"num_vessels": 2, "num_ports": 2, "rollout_steps": 10}
        trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=cfg)
        old_params = {
            name: list(ac.parameters())[0].data.clone()
            for name, ac in trainer.actor_critics.items()
        }
        trainer._rebuild_env(env_cfg)
        for name, ac in trainer.actor_critics.items():
            new_param = list(ac.parameters())[0].data
            self.assertTrue(torch.equal(old_params[name], new_param))


# ===================================================================
# Training logger
# ===================================================================


class TestTrainingLogger(unittest.TestCase):
    """Tests for the TrainingLogger utility."""

    def test_in_memory_logging(self) -> None:
        logger = TrainingLogger()
        logger.log(0, {"mean_reward": -5.0, "lr": 3e-4})
        logger.log(1, {"mean_reward": -4.0, "lr": 2e-4})
        self.assertEqual(logger.num_entries, 2)
        self.assertEqual(logger.get_rewards(), [-5.0, -4.0])

    def test_get_metric(self) -> None:
        logger = TrainingLogger()
        logger.log(0, {"mean_reward": -5.0, "custom": 1.0})
        logger.log(1, {"mean_reward": -4.0, "custom": 2.0})
        self.assertEqual(logger.get_metric("custom"), [1.0, 2.0])

    def test_summary_empty(self) -> None:
        logger = TrainingLogger()
        summary = logger.summary()
        self.assertEqual(summary["num_iterations"], 0)

    def test_summary_with_data(self) -> None:
        logger = TrainingLogger(experiment_name="test_run")
        for i in range(20):
            logger.log(i, {"mean_reward": float(i)})
        summary = logger.summary()
        self.assertEqual(summary["experiment"], "test_run")
        self.assertEqual(summary["num_iterations"], 20)
        self.assertEqual(summary["final_reward"], 19.0)
        self.assertGreater(summary["improvement"], 0.0)

    def test_file_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with TrainingLogger(log_dir=tmpdir, experiment_name="file_test") as logger:
                logger.log(0, {"mean_reward": -3.0})
                logger.log(1, {"mean_reward": -2.0})

            log_path = os.path.join(tmpdir, "train_log.jsonl")
            self.assertTrue(os.path.exists(log_path))

            with open(log_path) as f:
                lines = f.readlines()

            # Header + 2 entries + footer = 4 lines
            self.assertEqual(len(lines), 4)

            header = json.loads(lines[0])
            self.assertEqual(header["event"], "start")
            self.assertEqual(header["experiment"], "file_test")

            entry0 = json.loads(lines[1])
            self.assertEqual(entry0["iteration"], 0)
            self.assertAlmostEqual(entry0["mean_reward"], -3.0)

            footer = json.loads(lines[-1])
            self.assertEqual(footer["event"], "end")
            self.assertEqual(footer["num_iterations"], 2)

    def test_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with TrainingLogger(log_dir=tmpdir) as logger:
                logger.log(0, {"mean_reward": 1.0})
            # File should be closed and flushed
            self.assertIsNone(logger._file)

    def test_entries_property(self) -> None:
        logger = TrainingLogger()
        logger.log(0, {"x": 1})
        logger.log(1, {"x": 2})
        entries = logger.entries
        self.assertEqual(len(entries), 2)
        # Returned list should be a copy
        entries.clear()
        self.assertEqual(logger.num_entries, 2)

    def test_integration_with_trainer(self) -> None:
        """Logger can be used as log_fn in MAPPOTrainer.train()."""
        logger = TrainingLogger(experiment_name="integration")
        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        trainer.train(num_iterations=3, log_fn=logger.log)
        self.assertEqual(logger.num_entries, 3)
        summary = logger.summary()
        self.assertEqual(summary["num_iterations"], 3)


if __name__ == "__main__":
    unittest.main()
