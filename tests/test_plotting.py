"""Tests for plotting helpers writing non-empty PNG artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from hmarl_mvp.plotting import (
    plot_horizon_sweep,
    plot_noise_sweep,
    plot_policy_comparison,
    plot_sharing_sweep,
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


if __name__ == "__main__":
    unittest.main()
