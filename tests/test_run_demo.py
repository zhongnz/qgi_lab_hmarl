"""Regression tests for scripts/run_demo.py artifact behavior."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.run_demo as run_demo


def _step_df(policy: str = "forecast") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [0, 1, 2],
            "policy": [policy, policy, policy],
            "avg_queue": [1.0, 0.9, 0.8],
            "dock_utilization": [0.4, 0.5, 0.6],
            "total_emissions_co2": [10.0, 20.0, 30.0],
            "total_ops_cost_usd": [100.0, 200.0, 300.0],
            "avg_vessel_reward": [-1.0, -0.9, -0.8],
            "coordinator_reward": [-2.0, -1.9, -1.8],
        }
    )


def _train_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iteration": [1.0, 2.0, 3.0],
            "mean_reward": [-10.0, -9.0, -8.0],
            "rollout_time": [0.1, 0.1, 0.1],
            "update_time": [0.05, 0.05, 0.05],
            "vessel_value_loss": [1.0, 0.9, 0.8],
            "vessel_entropy": [0.5, 0.4, 0.3],
            "vessel_approx_kl": [0.01, 0.01, 0.01],
            "entropy_coeff": [0.01, 0.01, 0.01],
        }
    )


def test_demo_does_not_export_duplicate_train_log(monkeypatch: Any, tmp_path: Path) -> None:
    policy_results = {
        "independent": _step_df("independent"),
        "reactive": _step_df("reactive"),
        "forecast": _step_df("forecast"),
        "oracle": _step_df("oracle"),
    }
    horizon_results = {6: _step_df("forecast"), 12: _step_df("forecast")}
    noise_results = {0.0: _step_df("forecast"), 1.0: _step_df("forecast")}
    sharing_results = {"shared": _step_df("forecast"), "coordinator_only": _step_df("forecast")}
    mappo_results = {
        "mappo": _step_df("mappo"),
        "reactive": _step_df("reactive"),
        "forecast": _step_df("forecast"),
        "_train_log": _train_log_df(),
    }
    ablation_df = pd.DataFrame(
        {
            "ablation": ["full_model"],
            "final_mean_reward": [-1.0],
            "best_mean_reward": [-0.5],
            "total_reward": [-5.0],
        }
    )
    summary_df = pd.DataFrame(
        {
            "policy": ["forecast"],
            "avg_queue": [0.8],
        }
    )

    monkeypatch.setattr(
        run_demo, "parse_args",
        lambda: Namespace(output_dir=str(tmp_path), seed=42),
    )
    monkeypatch.setattr(run_demo, "run_policy_sweep", lambda **_kwargs: policy_results)
    monkeypatch.setattr(run_demo, "summarize_policy_results", lambda _r: summary_df)
    monkeypatch.setattr(run_demo, "run_horizon_sweep", lambda **_kwargs: horizon_results)
    monkeypatch.setattr(run_demo, "run_noise_sweep", lambda **_kwargs: noise_results)
    monkeypatch.setattr(run_demo, "run_sharing_sweep", lambda **_kwargs: sharing_results)
    monkeypatch.setattr(run_demo, "run_mappo_comparison", lambda **_kwargs: mappo_results)
    monkeypatch.setattr(run_demo, "run_mappo_ablation", lambda **_kwargs: ablation_df)

    # Keep test fast: plotting side-effects are not under test here.
    for plot_name in (
        "plot_policy_comparison",
        "plot_horizon_sweep",
        "plot_noise_sweep",
        "plot_sharing_sweep",
        "plot_training_curves",
        "plot_mappo_comparison",
        "plot_training_dashboard",
        "plot_timing_breakdown",
        "plot_ablation_bar",
    ):
        monkeypatch.setattr(run_demo, plot_name, lambda *args, **kwargs: None)

    run_demo.main()

    assert (tmp_path / "train_log.csv").exists()
    assert (tmp_path / "dashboard_log.csv").exists()
    assert not (tmp_path / "mappo__train_log.csv").exists()
