#!/usr/bin/env python3
"""Run baseline experiments and ablations from the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Ensure package imports work when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import (
    run_horizon_sweep,
    run_noise_sweep,
    run_policy_sweep,
    run_sharing_sweep,
    save_result_dict,
    summarize_policy_results,
)
from hmarl_mvp.plotting import (
    plot_horizon_sweep,
    plot_noise_sweep,
    plot_policy_comparison,
    plot_sharing_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMARL MVP baselines/ablations.")
    parser.add_argument("--steps", type=int, default=None, help="Override rollout steps.")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/baseline_refactor",
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing plot PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_default_config()
    if args.steps is not None:
        cfg["rollout_steps"] = args.steps

    policy_results = run_policy_sweep(steps=cfg["rollout_steps"], seed=args.seed, config=cfg)
    all_results = pd.concat(policy_results.values(), ignore_index=True)
    all_results.to_csv(out_dir / "policy_all_results.csv", index=False)
    save_result_dict(policy_results, str(out_dir), "policy")

    summary = summarize_policy_results(policy_results)
    summary.to_csv(out_dir / "policy_summary.csv")
    print("\nPolicy summary:")
    print(summary)

    horizon_results = run_horizon_sweep(steps=cfg["rollout_steps"], seed=args.seed, config=cfg)
    save_result_dict(horizon_results, str(out_dir), "horizon")

    noise_results = run_noise_sweep(steps=cfg["rollout_steps"], seed=args.seed, config=cfg)
    save_result_dict(noise_results, str(out_dir), "noise")

    sharing_results = run_sharing_sweep(steps=cfg["rollout_steps"], seed=args.seed, config=cfg)
    save_result_dict(sharing_results, str(out_dir), "sharing")

    if not args.no_plots:
        plot_policy_comparison(policy_results, out_path=str(out_dir / "policy_comparison.png"))
        plot_horizon_sweep(horizon_results, out_path=str(out_dir / "horizon_sweep.png"))
        plot_noise_sweep(noise_results, out_path=str(out_dir / "noise_sweep.png"))
        plot_sharing_sweep(sharing_results, out_path=str(out_dir / "sharing_sweep.png"))

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
