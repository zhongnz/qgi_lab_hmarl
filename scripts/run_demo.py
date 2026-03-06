#!/usr/bin/env python3
"""Lightweight demonstration run showing all HMARL effects.

Runs heuristic baselines, MAPPO training, evaluation, and ablations on
a small-scale setup (5 ports, 8 vessels, weather enabled) that completes
in ~5–8 minutes on a 12-core CPU.  Produces 9 publication-quality plots
covering every visualisation in the plotting module.

Usage::

    python -m scripts.run_demo [--output-dir runs/demo]

Outputs (all saved to ``--output-dir``):

    CSVs:
      policy_summary.csv        – aggregated heuristic baseline summary
      policy_*.csv              – per-policy baseline traces
      horizon_*.csv             – forecast-horizon sweep traces
      noise_*.csv               – forecast-noise sweep traces
      sharing_*.csv             – forecast-sharing sweep traces
      mappo_*.csv               – MAPPO and baseline comparison traces
      train_log.csv             – per-iteration training metrics
      dashboard_log.csv         – dashboard metrics (same MAPPO training run)
      ablation_results.csv      – 3-variant ablation summary

    PNGs:
      01_policy_comparison.png  – 2×3 heuristic comparison
      02_horizon_sweep.png      – forecast horizon ablation
      03_noise_sweep.png        – forecast noise sensitivity
      04_sharing_sweep.png      – forecast sharing ablation
      05_training_curves.png    – MAPPO reward + value loss
      06_mappo_comparison.png   – MAPPO vs heuristic baselines
      07_training_dashboard.png – 2×2 reward/loss/KL/entropy
      08_timing_breakdown.png   – rollout vs update time
      09_ablation_bar.png       – ablation comparison
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure package imports work when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import (
    run_horizon_sweep,
    run_mappo_ablation,
    run_mappo_comparison,
    run_noise_sweep,
    run_policy_sweep,
    run_sharing_sweep,
    save_result_dict,
    summarize_policy_results,
)
from hmarl_mvp.plotting import (
    plot_ablation_bar,
    plot_horizon_sweep,
    plot_mappo_comparison,
    plot_noise_sweep,
    plot_policy_comparison,
    plot_sharing_sweep,
    plot_timing_breakdown,
    plot_training_curves,
    plot_training_dashboard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HMARL MVP demo run.")
    parser.add_argument(
        "--output-dir", type=str, default="runs/demo",
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _banner(msg: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    t0 = time.time()

    # ── Environment configuration ────────────────────────────────────────
    # Small-scale but feature-complete: weather + AR(1) + all 3 agent types.
    cfg = get_default_config(
        num_ports=5,
        num_vessels=8,
        docks_per_port=3,
        rollout_steps=30,
        weather_enabled=True,
        weather_autocorrelation=0.7,
        sea_state_max=3.0,
        weather_shaping_weight=0.3,
    )
    steps = cfg["rollout_steps"]

    # ── 1. Heuristic baseline comparison ─────────────────────────────────
    _banner("1/6  Heuristic baselines (4 policies × 30 steps)")
    policy_results = run_policy_sweep(
        policies=["independent", "reactive", "forecast", "oracle"],
        steps=steps, seed=seed, config=cfg,
    )
    summary = summarize_policy_results(policy_results)
    summary.to_csv(out / "policy_summary.csv")
    save_result_dict(policy_results, str(out), "policy")
    print(summary)

    plot_policy_comparison(policy_results, out_path=str(out / "01_policy_comparison.png"))
    print(f"  → saved 01_policy_comparison.png  ({time.time()-t0:.1f}s elapsed)")

    # ── 2. Forecast ablations (horizon, noise, sharing) ──────────────────
    _banner("2/6  Forecast ablations")
    horizon_results = run_horizon_sweep(
        horizons=[6, 12, 24], steps=steps, seed=seed, config=cfg,
    )
    save_result_dict(horizon_results, str(out), "horizon")
    plot_horizon_sweep(horizon_results, out_path=str(out / "02_horizon_sweep.png"))

    noise_results = run_noise_sweep(
        noise_levels=[0.0, 0.5, 1.0, 2.0], steps=steps, seed=seed, config=cfg,
    )
    save_result_dict(noise_results, str(out), "noise")
    plot_noise_sweep(noise_results, out_path=str(out / "03_noise_sweep.png"))

    sharing_results = run_sharing_sweep(steps=steps, seed=seed, config=cfg)
    save_result_dict(sharing_results, str(out), "sharing")
    plot_sharing_sweep(sharing_results, out_path=str(out / "04_sharing_sweep.png"))
    print(f"  → saved 02–04 sweep plots  ({time.time()-t0:.1f}s elapsed)")

    # ── 3. MAPPO training + comparison ───────────────────────────────────
    _banner("3/6  MAPPO training (30 iterations × 32-step rollouts)")
    mappo_results = run_mappo_comparison(
        train_iterations=30,
        rollout_length=32,
        eval_steps=steps,
        baselines=["reactive", "forecast"],
        seed=seed,
        config=cfg,
        mappo_kwargs={
            "lr": 3e-4,
            "hidden_dims": [64, 64],
            "entropy_coeff": 0.01,
            "normalize_rewards": True,
            "normalize_observations": True,
            "device": "cpu",
        },
    )

    # Save comparison CSVs (exclude metadata keys like "_train_log")
    for name, df in mappo_results.items():
        if str(name).startswith("_"):
            continue
        df.to_csv(out / f"mappo_{name}.csv", index=False)

    train_log_df = mappo_results.get("_train_log", pd.DataFrame())
    if not train_log_df.empty:
        train_log_df.to_csv(out / "train_log.csv", index=False)

    plot_training_curves(train_log_df, out_path=str(out / "05_training_curves.png"))
    plot_mappo_comparison(mappo_results, out_path=str(out / "06_mappo_comparison.png"))
    print(f"  → saved 05–06 MAPPO plots  ({time.time()-t0:.1f}s elapsed)")

    # ── 4. Training dashboard (reuse same training run as step 3) ────────
    _banner("4/6  Training dashboard")
    if not train_log_df.empty:
        dashboard_history = train_log_df.to_dict("records")
        pd.DataFrame(dashboard_history).to_csv(out / "dashboard_log.csv", index=False)
        plot_training_dashboard(dashboard_history, out_path=str(out / "07_training_dashboard.png"))
        plot_timing_breakdown(dashboard_history, out_path=str(out / "08_timing_breakdown.png"))
        print(f"  → saved 07–08 dashboard + timing  ({time.time()-t0:.1f}s elapsed)")
    else:
        print("  → skipped 07–08 (empty training log)")

    # ── 5. Ablation study ────────────────────────────────────────────────
    _banner("5/6  Ablation study (3 variants × 20 iterations)")
    ablation_df = run_mappo_ablation(
        ablations={
            "full_model": {},
            "no_weather": {"env_weather_enabled": False},
            "high_entropy": {"entropy_coeff": 0.05},
        },
        train_iterations=20,
        rollout_length=32,
        eval_steps=steps,
        seed=seed,
        config=cfg,
    )
    ablation_df.to_csv(out / "ablation_results.csv", index=False)
    plot_ablation_bar(ablation_df, out_path=str(out / "09_ablation_bar.png"))
    print(f"  → saved 09_ablation_bar.png  ({time.time()-t0:.1f}s elapsed)")

    # ── 6. Summary ───────────────────────────────────────────────────────
    _banner("6/6  Done!")
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  All outputs saved to: {out.resolve()}")
    print("\n  Plots produced:")
    for png in sorted(out.glob("*.png")):
        size_kb = png.stat().st_size / 1024
        print(f"    {png.name:40s}  ({size_kb:.0f} KB)")
    print("\n  CSVs produced:")
    for csv_file in sorted(out.glob("*.csv")):
        with csv_file.open() as handle:
            rows = sum(1 for _ in handle) - 1
        print(f"    {csv_file.name:40s}  ({rows} rows)")


if __name__ == "__main__":
    main()
