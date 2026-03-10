"""Tests for plotting helpers writing non-empty PNG artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from hmarl_mvp.plotting import (
    plot_ablation_bar,
    plot_horizon_sweep,
    plot_mappo_comparison,
    plot_multi_seed_curves,
    plot_noise_sweep,
    plot_policy_comparison,
    plot_sharing_sweep,
    plot_sweep_heatmap,
    plot_time_series_diagnostics,
    plot_timing_breakdown,
    plot_training_curves,
    plot_training_dashboard,
)


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [0, 1, 2],
            "avg_queue": [2.0, 1.8, 1.5],
            "dock_utilization": [0.4, 0.5, 0.6],
            "total_emissions_co2": [0.0, 10.0, 22.0],
            "total_fuel_used": [0.0, 3.0, 8.0],
            "avg_vessel_reward": [-1.0, -1.2, -1.4],
            "total_ops_cost_usd": [1000.0, 2200.0, 3500.0],
            "coordinator_reward": [0.5, 0.4, 0.3],
        }
    )


def _make_train_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iteration": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "mean_reward": [-2.0, -1.8, -1.6, -1.5, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8],
            "joint_mean_reward": [-3.0, -2.8, -2.64, -2.46, -2.27, -2.14, -2.0, -1.89, -1.77, -1.66],
            "vessel_mean_reward": [-1.4, -1.3, -1.2, -1.1, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75],
            "port_mean_reward": [-0.7, -0.68, -0.66, -0.62, -0.58, -0.55, -0.5, -0.48, -0.45, -0.42],
            "coordinator_mean_reward": [-0.9, -0.82, -0.78, -0.74, -0.69, -0.64, -0.6, -0.56, -0.52, -0.49],
            "coordinator_value_loss": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25],
            "vessel_value_loss": [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28],
            "port_value_loss": [0.4, 0.38, 0.36, 0.34, 0.32, 0.31, 0.29, 0.28, 0.27, 0.26],
            "coordinator_explained_variance": [0.1, 0.12, 0.18, 0.25, 0.3, 0.38, 0.45, 0.52, 0.58, 0.63],
            "vessel_explained_variance": [0.05, 0.08, 0.11, 0.15, 0.22, 0.28, 0.35, 0.41, 0.48, 0.54],
            "port_explained_variance": [0.02, 0.03, 0.05, 0.07, 0.11, 0.16, 0.2, 0.24, 0.29, 0.33],
            "coordinator_approx_kl": [0.05, 0.04, 0.03, 0.025, 0.02, 0.018, 0.015, 0.012, 0.01, 0.009],
            "vessel_approx_kl": [0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.018, 0.015, 0.012, 0.01],
            "port_approx_kl": [0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.015, 0.013, 0.011, 0.01],
            "coordinator_entropy": [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8],
            "vessel_entropy": [1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.15, 1.1, 1.05],
            "port_entropy": [1.2, 1.18, 1.15, 1.12, 1.09, 1.05, 1.02, 1.0, 0.98, 0.96],
            "rollout_time": [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0],
            "update_time": [0.5, 0.6, 0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.5, 0.5],
        }
    )


class PlottingTests(unittest.TestCase):
    def test_policy_comparison_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "policy.png"
            results = {"forecast": _make_df(), "reactive": _make_df()}
            plot_policy_comparison(results, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_horizon_noise_and_sharing_plots_write_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            horizon_path = Path(tmp_dir) / "horizon.png"
            noise_path = Path(tmp_dir) / "noise.png"
            sharing_path = Path(tmp_dir) / "sharing.png"

            sweep_results_int = {6: _make_df(), 12: _make_df()}
            sweep_results_float = {0.0: _make_df(), 0.5: _make_df()}
            sharing_results = {"shared": _make_df(), "coordinator_only": _make_df()}

            plot_horizon_sweep(sweep_results_int, out_path=str(horizon_path))
            plot_noise_sweep(sweep_results_float, out_path=str(noise_path))
            plot_sharing_sweep(sharing_results, out_path=str(sharing_path))

            self.assertTrue(horizon_path.exists())
            self.assertTrue(noise_path.exists())
            self.assertTrue(sharing_path.exists())
            self.assertGreater(horizon_path.stat().st_size, 0)
            self.assertGreater(noise_path.stat().st_size, 0)
            self.assertGreater(sharing_path.stat().st_size, 0)

    def test_training_curves_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "train_curves.png"
            plot_training_curves(_make_train_log(), out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_training_curves_no_losses(self) -> None:
        """Handles single-panel case when no *_value_loss columns exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "train_curves_no_loss.png"
            df = _make_train_log()[["iteration", "mean_reward"]]
            plot_training_curves(df, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_mappo_comparison_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "mappo_cmp.png"
            results = {"mappo": _make_df(), "reactive": _make_df()}
            plot_mappo_comparison(results, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_mappo_comparison_skips_private_keys(self) -> None:
        """Keys starting with ``_`` are skipped without error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "mappo_cmp2.png"
            results = {"mappo": _make_df(), "_train_log": _make_train_log()}
            plot_mappo_comparison(results, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_sweep_heatmap_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "heatmap.png"
            sweep_df = pd.DataFrame(
                {
                    "lr": [1e-3, 1e-3, 1e-4, 1e-4],
                    "gamma": [0.95, 0.99, 0.95, 0.99],
                    "total_reward": [-5.0, -3.0, -4.0, -2.5],
                }
            )
            plot_sweep_heatmap(sweep_df, "lr", "gamma", out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_sweep_heatmap_missing_param_col(self) -> None:
        """Silently returns when a required column is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "heatmap2.png"
            sweep_df = pd.DataFrame({"lr": [1e-3], "total_reward": [-5.0]})
            plot_sweep_heatmap(sweep_df, "lr", "gamma", out_path=str(out_path))
            self.assertFalse(out_path.exists())

    def test_ablation_bar_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "ablation.png"
            ablation_df = pd.DataFrame(
                {
                    "ablation": ["full", "no_weather", "no_forecast"],
                    "final_mean_reward": [-1.0, -1.5, -1.3],
                    "best_mean_reward": [-0.8, -1.2, -1.1],
                    "total_reward": [-50.0, -75.0, -65.0],
                }
            )
            plot_ablation_bar(ablation_df, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_training_dashboard_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "dashboard.png"
            history = _make_train_log().to_dict("records")
            plot_training_dashboard(history, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_multi_seed_curves_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "multi_seed.png"
            hist1 = [{"mean_reward": -2.0 + 0.1 * i} for i in range(10)]
            hist2 = [{"mean_reward": -2.2 + 0.12 * i} for i in range(10)]
            multi_seed_result = {"histories": [hist1, hist2], "seeds": [42, 99]}
            plot_multi_seed_curves(multi_seed_result, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_timing_breakdown_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "timing.png"
            history = _make_train_log().to_dict("records")
            plot_timing_breakdown(history, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_time_series_diagnostics_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "diagnostics.png"
            df = _make_df().assign(
                step_fuel_used=[0.5, 0.8, 0.3],
                step_co2_emitted=[1.0, 1.4, 0.9],
                step_delay_hours=[0.0, 0.5, 0.25],
                directive_queue_size=[2.0, 1.0, 0.0],
                vessel_0_speed=[10.0, 11.0, 9.0],
                vessel_1_speed=[8.0, 8.5, 9.0],
                port_0_queue=[1.0, 2.0, 1.0],
                coordinator_0_reward=[-1.0, -0.8, -0.9],
                weather_enabled=[1, 1, 1],
            )
            plot_time_series_diagnostics(df, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_time_series_diagnostics_group_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            df = _make_df().assign(
                step_fuel_used=[0.5, 0.8, 0.3],
                vessel_0_speed=[10.0, 11.0, 9.0],
                port_0_queue=[1.0, 2.0, 1.0],
                coordinator_0_reward=[-1.0, -0.8, -0.9],
            )
            for group in ("aggregate", "vessel", "port", "coordinator"):
                out_path = Path(tmp_dir) / f"diagnostics_{group}.png"
                plot_time_series_diagnostics(
                    df,
                    out_path=str(out_path),
                    column_group=group,
                )
                self.assertTrue(out_path.exists(), msg=group)
                self.assertGreater(out_path.stat().st_size, 0, msg=group)

    def test_time_series_diagnostics_handles_constant_only_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "diagnostics_constant.png"
            df = pd.DataFrame({"t": [0, 1, 2], "weather_enabled": [1, 1, 1]})
            plot_time_series_diagnostics(df, out_path=str(out_path))
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
