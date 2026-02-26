"""Tests for training profiling, early stopping, multi-seed training, and plotting."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Training profiling (timing keys in train() history)
# ---------------------------------------------------------------------------


class TestTrainingProfiling:
    """Verify timing instrumentation in MAPPOTrainer.train()."""

    def _quick_train(self, iters: int = 3, **kwargs: Any) -> list[dict[str, Any]]:
        from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=99)
        return trainer.train(num_iterations=iters, **kwargs)

    def test_timing_keys_present(self) -> None:
        history = self._quick_train()
        for entry in history:
            assert "rollout_time" in entry
            assert "update_time" in entry
            assert "iter_time" in entry

    def test_timing_values_positive(self) -> None:
        history = self._quick_train()
        for entry in history:
            assert entry["rollout_time"] > 0
            assert entry["update_time"] > 0
            assert entry["iter_time"] > 0

    def test_iter_time_geq_sum(self) -> None:
        """iter_time should be >= rollout_time + update_time."""
        history = self._quick_train()
        for entry in history:
            assert entry["iter_time"] >= entry["rollout_time"] + entry["update_time"] - 1e-6

    def test_total_train_time(self) -> None:
        """Last entry should have total_train_time."""
        history = self._quick_train()
        assert "total_train_time" in history[-1]
        assert history[-1]["total_train_time"] > 0

    def test_total_train_time_geq_sum_iters(self) -> None:
        history = self._quick_train()
        sum_iters = sum(e["iter_time"] for e in history)
        assert history[-1]["total_train_time"] >= sum_iters - 1e-6


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStoppingTrain:
    """Verify patience-based early stopping in train()."""

    def test_no_early_stopping_by_default(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=5)
        assert len(history) == 5

    def test_early_stopping_triggers(self) -> None:
        """With patience=2 and unchanging rewards, should stop early."""
        from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(
            num_iterations=100,
            early_stopping_patience=2,
        )
        assert len(history) < 100
        assert history[-1].get("early_stopped", False)

    def test_early_stopping_zero_disabled(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        history = trainer.train(num_iterations=5, early_stopping_patience=0)
        assert len(history) == 5

    def test_early_stopping_patience_respected(self) -> None:
        """Training runs at least patience+1 iterations."""
        from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

        cfg = MAPPOConfig(rollout_length=8)
        trainer = MAPPOTrainer(mappo_config=cfg, seed=42)
        patience = 3
        history = trainer.train(
            num_iterations=200,
            early_stopping_patience=patience,
        )
        assert len(history) >= patience + 1


# ---------------------------------------------------------------------------
# Multi-seed training
# ---------------------------------------------------------------------------


class TestMultiSeedTraining:
    """Verify train_multi_seed() function."""

    def test_basic_multi_seed(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, train_multi_seed

        result = train_multi_seed(
            mappo_config=MAPPOConfig(rollout_length=8),
            num_iterations=3,
            num_seeds=2,
        )
        assert len(result["seeds"]) == 2
        assert len(result["histories"]) == 2
        assert len(result["summaries"]) == 2
        assert "aggregate_summary" in result
        assert "mean_reward_curve" in result
        assert "std_reward_curve" in result

    def test_explicit_seeds(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, train_multi_seed

        result = train_multi_seed(
            mappo_config=MAPPOConfig(rollout_length=8),
            num_iterations=3,
            seeds=[100, 200],
        )
        assert result["seeds"] == [100, 200]

    def test_curve_shapes(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, train_multi_seed

        result = train_multi_seed(
            mappo_config=MAPPOConfig(rollout_length=8),
            num_iterations=4,
            num_seeds=2,
        )
        assert result["mean_reward_curve"].shape == (4,)
        assert result["std_reward_curve"].shape == (4,)

    def test_aggregate_summary_keys(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, train_multi_seed

        result = train_multi_seed(
            mappo_config=MAPPOConfig(rollout_length=8),
            num_iterations=3,
            num_seeds=2,
        )
        agg = result["aggregate_summary"]
        assert "num_seeds" in agg
        assert "mean_best_mean_reward" in agg
        assert "std_best_mean_reward" in agg
        assert "mean_final_mean_reward" in agg

    def test_multi_seed_with_early_stopping(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, train_multi_seed

        result = train_multi_seed(
            mappo_config=MAPPOConfig(rollout_length=8),
            num_iterations=50,
            num_seeds=2,
            early_stopping_patience=2,
        )
        for hist in result["histories"]:
            assert len(hist) < 50

    def test_multi_seed_with_checkpoint(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, train_multi_seed

        with tempfile.TemporaryDirectory() as tmpdir:
            train_multi_seed(
                mappo_config=MAPPOConfig(rollout_length=8),
                num_iterations=3,
                num_seeds=2,
                checkpoint_dir=tmpdir,
            )
            seed_dirs = list(Path(tmpdir).glob("seed_*"))
            assert len(seed_dirs) == 2

    def test_timing_in_aggregate(self) -> None:
        from hmarl_mvp.mappo import MAPPOConfig, train_multi_seed

        result = train_multi_seed(
            mappo_config=MAPPOConfig(rollout_length=8),
            num_iterations=3,
            num_seeds=2,
        )
        agg = result["aggregate_summary"]
        assert "mean_train_time" in agg
        assert "total_train_time" in agg
        assert agg["total_train_time"] > 0


# ---------------------------------------------------------------------------
# Multi-seed curve plotting
# ---------------------------------------------------------------------------


class TestMultiSeedPlotting:
    """Verify plot_multi_seed_curves and plot_timing_breakdown."""

    def _make_result(self) -> dict[str, Any]:
        """Build a fake multi-seed result dict."""
        histories = []
        for seed in range(3):
            hist = []
            for i in range(10):
                hist.append({
                    "iteration": i,
                    "mean_reward": float(i + seed * 0.1),
                    "rollout_time": 0.1 + i * 0.01,
                    "update_time": 0.05 + i * 0.005,
                    "iter_time": 0.16 + i * 0.015,
                })
            histories.append(hist)
        return {
            "seeds": [0, 1, 2],
            "histories": histories,
        }

    def test_plot_multi_seed_curves_file(self) -> None:
        from hmarl_mvp.plotting import plot_multi_seed_curves

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "curves.png")
            plot_multi_seed_curves(result, out_path=out)
            assert Path(out).exists()

    def test_plot_multi_seed_curves_custom_metric(self) -> None:
        from hmarl_mvp.plotting import plot_multi_seed_curves

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "curves.png")
            plot_multi_seed_curves(result, metric="iter_time", out_path=out)
            assert Path(out).exists()

    def test_plot_timing_breakdown_file(self) -> None:
        from hmarl_mvp.plotting import plot_timing_breakdown

        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "timing.png")
            plot_timing_breakdown(result["histories"][0], out_path=out)
            assert Path(out).exists()

    def test_plot_timing_breakdown_no_timing_keys(self) -> None:
        """Should return without error when timing keys are missing."""
        from hmarl_mvp.plotting import plot_timing_breakdown

        history = [{"iteration": i, "mean_reward": float(i)} for i in range(5)]
        plot_timing_breakdown(history)  # Should not raise

    def test_plot_empty_result_noop(self) -> None:
        from hmarl_mvp.plotting import plot_multi_seed_curves

        result = {"seeds": [], "histories": []}
        plot_multi_seed_curves(result)  # Should not raise


# ---------------------------------------------------------------------------
# CLI multiseed subcommand
# ---------------------------------------------------------------------------


class TestCLIMultiseed:
    """Verify CLI parsing for the multiseed subcommand."""

    def test_multiseed_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_mappo", "multiseed", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--num-seeds" in result.stdout
        assert "--early-stopping" in result.stdout

    def test_train_early_stopping_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_mappo", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--early-stopping" in result.stdout

    def test_multiseed_parse_defaults(self) -> None:
        from scripts.run_mappo import parse_args

        with patch("sys.argv", ["prog", "multiseed"]):
            args = parse_args()
        assert args.command == "multiseed"
        assert args.num_seeds == 3
        assert args.early_stopping == 0
        assert args.iterations == 50

    def test_multiseed_parse_custom(self) -> None:
        from scripts.run_mappo import parse_args

        with patch("sys.argv", [
            "prog", "multiseed",
            "--num-seeds", "5",
            "--iterations", "20",
            "--early-stopping", "10",
            "--weather",
        ]):
            args = parse_args()
        assert args.num_seeds == 5
        assert args.iterations == 20
        assert args.early_stopping == 10
        assert args.weather is True
