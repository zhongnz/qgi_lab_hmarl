from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _policy_df(scale: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [0, 1, 2],
            "avg_queue": [1.0 * scale, 1.2 * scale, 1.1 * scale],
            "dock_utilization": [0.4, 0.5, 0.6],
            "total_emissions_co2": [2.0 * scale, 2.3 * scale, 2.7 * scale],
            "total_fuel_used": [1.0 * scale, 1.4 * scale, 1.8 * scale],
            "avg_vessel_reward": [-1.0 * scale, -0.8 * scale, -0.6 * scale],
            "total_ops_cost_usd": [1000 * scale, 1100 * scale, 1200 * scale],
            "coordinator_reward": [-0.2 * scale, -0.1 * scale, -0.05 * scale],
        }
    )


def test_generate_paper_figures_cli_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "figures"

    # Full-run data for training + policy comparison
    full_run = runs_dir / "full_run"
    full_run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "mean_reward": [0.1, 0.2, 0.3],
            "vessel_value_loss": [1.0, 0.8, 0.6],
        }
    ).to_csv(full_run / "mappo__train_log.csv", index=False)

    for i, policy in enumerate(["mappo", "independent", "reactive", "forecast", "oracle"]):
        _policy_df(1.0 + i * 0.1).to_csv(full_run / f"mappo_{policy}.csv", index=False)

    # Multi-seed data
    for seed in (42, 49):
        seed_dir = runs_dir / "multi_seed" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "iteration": [0, 1, 2],
                "mean_reward": [0.0 + seed * 1e-4, 0.2, 0.3],
                "vessel_value_loss": [1.0, 0.9, 0.8],
            }
        ).to_csv(seed_dir / "metrics.csv", index=False)

    pd.DataFrame(
        {
            "seed": [42, 49],
            "final_mean_reward": [0.3, 0.35],
            "best_mean_reward": [0.32, 0.36],
            "total_reward": [3.0, 3.3],
        }
    ).to_csv(runs_dir / "multi_seed" / "summary.csv", index=False)

    no_sharing = runs_dir / "no_sharing"
    no_sharing.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "seed": [42, 49],
            "final_mean_reward": [0.2, 0.22],
            "best_mean_reward": [0.23, 0.24],
            "total_reward": [2.0, 2.2],
        }
    ).to_csv(no_sharing / "summary.csv", index=False)

    # Weather dashboard source
    weather_dir = runs_dir / "weather_curriculum"
    weather_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "iteration": [0, 1, 2],
            "mean_reward": [0.1, 0.15, 0.2],
            "vessel_value_loss": [1.0, 0.9, 0.8],
            "vessel_approx_kl": [0.01, 0.012, 0.011],
            "vessel_entropy": [0.7, 0.65, 0.6],
            "entropy_coeff": [0.01, 0.01, 0.01],
        }
    ).to_csv(weather_dir / "metrics.csv", index=False)

    # Hyperparameter sweep source
    sweep_dir = runs_dir / "mappo_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "lr": [1e-4, 3e-4, 1e-4, 3e-4],
            "entropy_coeff": [0.01, 0.01, 0.05, 0.05],
            "total_reward": [0.1, 0.2, 0.15, 0.25],
        }
    ).to_csv(sweep_dir / "sweep_results.csv", index=False)

    # Economic comparison source
    econ_dir = runs_dir / "economic"
    econ_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "policy": ["independent", "mappo"],
            "fuel_cost_usd": [1000.0, 900.0],
            "delay_cost_usd": [400.0, 300.0],
            "carbon_cost_usd": [250.0, 200.0],
        }
    ).to_csv(econ_dir / "cost_breakdown.csv", index=False)

    cmd = [
        sys.executable,
        "-m",
        "scripts.generate_paper_figures",
        "--runs-dir",
        str(runs_dir),
        "--out-dir",
        str(out_dir),
        "--format",
        "png",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    expected = [
        "fig1_training_curves.png",
        "fig2_policy_comparison.png",
        "fig3_multi_seed_curves.png",
        "fig4_parameter_sharing.png",
        "fig5_weather_dashboard.png",
        "fig6_hyperparam_heatmap.png",
        "fig7_economic_comparison.png",
    ]
    for fname in expected:
        assert (out_dir / fname).exists(), f"missing output figure: {fname}"
