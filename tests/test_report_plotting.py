"""Tests for report generation, new plotting functions, and MAPPO run script."""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

import pandas as pd

from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.plotting import (
    plot_ablation_bar,
    plot_sweep_heatmap,
    plot_training_dashboard,
)
from hmarl_mvp.report import (
    generate_ablation_report,
    generate_sweep_report,
    generate_training_report,
)

# ===================================================================
# Report generation tests
# ===================================================================


class TestTrainingReport(unittest.TestCase):
    """Tests for generate_training_report."""

    def _make_history(self, n: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "iteration": i,
                "mean_reward": -1.0 + 0.1 * i,
                "total_reward": -5.0 + 0.5 * i,
                "lr": 3e-4 * (1 - i / n),
                "entropy_coeff": 0.01,
                "vessel_policy_loss": 0.5 - 0.02 * i,
                "vessel_value_loss": 1.0 - 0.05 * i,
                "vessel_entropy": 2.0 - 0.1 * i,
                "vessel_clip_frac": 0.1,
                "vessel_approx_kl": 0.005 + 0.001 * i,
                "vessel_kl_early_stopped": False,
                "port_policy_loss": 0.3,
                "port_value_loss": 0.8,
                "port_entropy": 1.5,
                "port_clip_frac": 0.05,
                "port_approx_kl": 0.003,
                "port_kl_early_stopped": False,
                "coordinator_policy_loss": 0.4,
                "coordinator_value_loss": 0.9,
                "coordinator_entropy": 1.8,
                "coordinator_clip_frac": 0.08,
                "coordinator_approx_kl": 0.004,
                "coordinator_kl_early_stopped": False,
            }
            for i in range(n)
        ]

    def test_basic_report(self) -> None:
        """Report should be a non-empty markdown string."""
        history = self._make_history()
        report = generate_training_report(history)
        self.assertIsInstance(report, str)
        self.assertIn("# MAPPO Training Report", report)
        self.assertIn("Training Summary", report)

    def test_report_with_config(self) -> None:
        """Config section should appear when provided."""
        history = self._make_history()
        report = generate_training_report(
            history,
            config={"env": {"num_vessels": 8}, "mappo": {"lr": 3e-4}},
        )
        self.assertIn("Configuration", report)
        self.assertIn("env.num_vessels", report)

    def test_report_with_eval(self) -> None:
        """Evaluation section should appear with eval_result."""
        history = self._make_history()
        eval_result = {
            "mean": {"total_reward": -3.0},
            "std": {"total_reward": 0.5},
            "min": {"total_reward": -3.8},
            "max": {"total_reward": -2.2},
            "episodes": [{"total_reward": -3.0}],
        }
        report = generate_training_report(history, eval_result=eval_result)
        self.assertIn("Evaluation", report)
        self.assertIn("total_reward", report)

    def test_report_with_elapsed(self) -> None:
        """Duration should appear when elapsed is provided."""
        history = self._make_history()
        report = generate_training_report(history, elapsed_seconds=120.5)
        self.assertIn("Wall-clock time", report)

    def test_per_agent_table(self) -> None:
        """Report should include per-agent metric table."""
        history = self._make_history()
        report = generate_training_report(history)
        self.assertIn("Per-Agent", report)
        self.assertIn("vessel", report)
        self.assertIn("port", report)
        self.assertIn("coordinator", report)

    def test_empty_history(self) -> None:
        """Report should handle empty history gracefully."""
        report = generate_training_report([])
        self.assertIn("Iterations", report)
        self.assertIn("0", report)

    def test_improvement_stats(self) -> None:
        """With enough iterations, early/late improvement should appear."""
        history = self._make_history(20)
        report = generate_training_report(history)
        self.assertIn("Improvement", report)


class TestSweepReport(unittest.TestCase):
    """Tests for generate_sweep_report."""

    def _make_sweep_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"lr": 1e-4, "entropy_coeff": 0.01, "eval_total_reward": -3.0, "final_mean_reward": -1.0},
            {"lr": 3e-4, "entropy_coeff": 0.01, "eval_total_reward": -2.0, "final_mean_reward": -0.5},
            {"lr": 1e-3, "entropy_coeff": 0.05, "eval_total_reward": -4.0, "final_mean_reward": -1.5},
        ])

    def test_basic_sweep_report(self) -> None:
        df = self._make_sweep_df()
        report = generate_sweep_report(df)
        self.assertIn("Sweep Report", report)
        self.assertIn("Configurations tested", report)
        self.assertIn("Best Configuration", report)

    def test_sorted_by_metric(self) -> None:
        df = self._make_sweep_df()
        report = generate_sweep_report(df, sort_by="eval_total_reward")
        # Best config (lr=3e-4) should appear first in results
        lines = report.split("\n")
        rank1_line = [l for l in lines if l.startswith("| 1 |")]
        self.assertEqual(len(rank1_line), 1)
        self.assertIn("-2.0000", rank1_line[0])

    def test_custom_title(self) -> None:
        df = self._make_sweep_df()
        report = generate_sweep_report(df, title="My Custom Sweep")
        self.assertIn("My Custom Sweep", report)


class TestAblationReport(unittest.TestCase):
    """Tests for generate_ablation_report."""

    def _make_ablation_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"ablation": "baseline", "final_mean_reward": -1.0, "eval_total_reward": -3.0},
            {"ablation": "no_reward_norm", "final_mean_reward": -1.5, "eval_total_reward": -4.0},
            {"ablation": "high_entropy", "final_mean_reward": -0.8, "eval_total_reward": -2.5},
        ])

    def test_basic_ablation_report(self) -> None:
        df = self._make_ablation_df()
        report = generate_ablation_report(df)
        self.assertIn("Ablation Study", report)
        self.assertIn("baseline", report)
        self.assertIn("Variants tested", report)

    def test_deltas_from_baseline(self) -> None:
        df = self._make_ablation_df()
        report = generate_ablation_report(df)
        self.assertIn("Deltas from Baseline", report)
        self.assertIn("no_reward_norm", report)

    def test_custom_title(self) -> None:
        df = self._make_ablation_df()
        report = generate_ablation_report(df, title="Custom Ablation")
        self.assertIn("Custom Ablation", report)


# ===================================================================
# New plotting function tests
# ===================================================================


class TestSweepHeatmap(unittest.TestCase):
    """Tests for plot_sweep_heatmap."""

    def test_heatmap_creates_file(self) -> None:
        df = pd.DataFrame([
            {"lr": 1e-4, "entropy_coeff": 0.01, "eval_total_reward": -3.0},
            {"lr": 3e-4, "entropy_coeff": 0.01, "eval_total_reward": -2.0},
            {"lr": 1e-4, "entropy_coeff": 0.05, "eval_total_reward": -3.5},
            {"lr": 3e-4, "entropy_coeff": 0.05, "eval_total_reward": -2.5},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "heatmap.png")
            plot_sweep_heatmap(df, "lr", "entropy_coeff", out_path=out)
            self.assertTrue(os.path.exists(out))

    def test_missing_param_noop(self) -> None:
        """Missing param should do nothing, not raise."""
        df = pd.DataFrame([{"lr": 1e-4, "eval_total_reward": -3.0}])
        plot_sweep_heatmap(df, "lr", "missing_param")  # Should not raise


class TestAblationBar(unittest.TestCase):
    """Tests for plot_ablation_bar."""

    def test_bar_creates_file(self) -> None:
        df = pd.DataFrame([
            {"ablation": "baseline", "final_mean_reward": -1.0, "eval_total_reward": -3.0},
            {"ablation": "variant_a", "final_mean_reward": -1.5, "eval_total_reward": -4.0},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "ablation_bar.png")
            plot_ablation_bar(df, out_path=out)
            self.assertTrue(os.path.exists(out))

    def test_no_ablation_col_noop(self) -> None:
        """Missing ablation column should not raise."""
        df = pd.DataFrame([{"x": 1}])
        plot_ablation_bar(df)  # Should not raise


class TestTrainingDashboard(unittest.TestCase):
    """Tests for plot_training_dashboard."""

    def test_dashboard_creates_file(self) -> None:
        history: list[dict[str, Any]] = [
            {
                "iteration": i,
                "mean_reward": -1.0 + 0.05 * i,
                "vessel_value_loss": 1.0 - 0.02 * i,
                "port_value_loss": 0.8,
                "vessel_approx_kl": 0.005,
                "port_approx_kl": 0.003,
                "vessel_entropy": 2.0,
                "port_entropy": 1.5,
                "entropy_coeff": 0.01,
            }
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "dashboard.png")
            plot_training_dashboard(history, out_path=out)
            self.assertTrue(os.path.exists(out))

    def test_minimal_history(self) -> None:
        """Dashboard with minimal data should not crash."""
        history = [{"iteration": 0, "mean_reward": -1.0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "dash.png")
            plot_training_dashboard(history, out_path=out)
            self.assertTrue(os.path.exists(out))


# ===================================================================
# Integration: train() produces dashboard-compatible output
# ===================================================================


class TestTrainDashboardIntegration(unittest.TestCase):
    """Verify train() output can feed into dashboard and report."""

    def test_train_output_to_dashboard(self) -> None:
        """train() history should be accepted by plot_training_dashboard."""
        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        history = trainer.train(num_iterations=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "dash.png")
            plot_training_dashboard(history, out_path=out)
            self.assertTrue(os.path.exists(out))

    def test_train_output_to_report(self) -> None:
        """train() history should be accepted by generate_training_report."""
        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
        )
        history = trainer.train(num_iterations=3)
        eval_result = trainer.evaluate_episodes(num_episodes=2)
        report = generate_training_report(
            history, eval_result=eval_result, elapsed_seconds=5.0
        )
        self.assertIn("Training Summary", report)
        self.assertIn("Evaluation", report)

    def test_full_pipeline_with_checkpoint(self) -> None:
        """Full pipeline: train with checkpoint + eval + report."""
        cfg = MAPPOConfig(rollout_length=4)
        trainer = MAPPOTrainer(
            env_config={"num_vessels": 2, "num_ports": 2, "rollout_steps": 10},
            mappo_config=cfg,
            seed=42,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            history = trainer.train(
                num_iterations=3,
                checkpoint_dir=ckpt_dir,
                eval_interval=2,
            )
            eval_result = trainer.evaluate_episodes(num_episodes=2)
            report = generate_training_report(
                history,
                eval_result=eval_result,
                config={"env": {"n": 2}, "mappo": {"lr": 3e-4}},
                elapsed_seconds=2.0,
            )
            # Report file
            report_path = os.path.join(tmpdir, "report.md")
            with open(report_path, "w") as f:
                f.write(report)
            self.assertTrue(os.path.exists(report_path))
            # Checkpoint exists
            pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            self.assertGreater(len(pt_files), 0)


# ===================================================================
# MAPPO run script smoke test
# ===================================================================


class TestRunMappoScript(unittest.TestCase):
    """Smoke tests for run_mappo.py CLI module."""

    def test_script_importable(self) -> None:
        """run_mappo.py should be importable without errors."""
        import importlib.util
        import sys
        from pathlib import Path

        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_mappo.py"
        spec = importlib.util.spec_from_file_location("run_mappo", script_path)
        self.assertIsNotNone(spec)
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        # Prevent it from running main()
        sys.modules["run_mappo"] = mod
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, "cmd_train"))
        self.assertTrue(hasattr(mod, "cmd_compare"))
        self.assertTrue(hasattr(mod, "cmd_sweep"))
        self.assertTrue(hasattr(mod, "cmd_ablate"))


if __name__ == "__main__":
    unittest.main()
