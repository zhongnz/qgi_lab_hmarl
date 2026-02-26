#!/usr/bin/env python3
"""Run MAPPO training, comparison, sweep, and ablation experiments.

Usage
-----
  # Quick comparison (MAPPO vs baselines):
  python scripts/run_mappo.py compare --iterations 50

  # Hyperparameter sweep:
  python scripts/run_mappo.py sweep --iterations 30

  # Ablation study:
  python scripts/run_mappo.py ablate --iterations 30

  # Full training with checkpointing:
  python scripts/run_mappo.py train --iterations 200 --checkpoint-dir runs/mappo_ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure package imports work when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmarl_mvp.experiment import (
    run_mappo_ablation,
    run_mappo_comparison,
    run_mappo_hyperparam_sweep,
)
from hmarl_mvp.logger import TrainingLogger
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.plotting import plot_mappo_comparison, plot_training_curves
from hmarl_mvp.report import generate_training_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAPPO experiments for HMARL maritime scheduling.",
    )
    sub = parser.add_subparsers(dest="command", help="Experiment type")

    # ---- train ----
    train_p = sub.add_parser("train", help="Train MAPPO with logging and checkpointing")
    train_p.add_argument("--iterations", type=int, default=100)
    train_p.add_argument("--rollout-length", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--entropy-coeff", type=float, default=0.01)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--checkpoint-dir", type=str, default=None)
    train_p.add_argument("--eval-interval", type=int, default=10)
    train_p.add_argument("--output-dir", type=str, default="runs/mappo_train")
    train_p.add_argument("--vessels", type=int, default=None)
    train_p.add_argument("--ports", type=int, default=None)

    # ---- compare ----
    cmp_p = sub.add_parser("compare", help="Train MAPPO and compare against baselines")
    cmp_p.add_argument("--iterations", type=int, default=50)
    cmp_p.add_argument("--rollout-length", type=int, default=64)
    cmp_p.add_argument("--seed", type=int, default=42)
    cmp_p.add_argument("--output-dir", type=str, default="runs/mappo_compare")
    cmp_p.add_argument("--no-plots", action="store_true")

    # ---- sweep ----
    sweep_p = sub.add_parser("sweep", help="Hyperparameter grid sweep")
    sweep_p.add_argument("--iterations", type=int, default=30)
    sweep_p.add_argument("--rollout-length", type=int, default=64)
    sweep_p.add_argument("--seed", type=int, default=42)
    sweep_p.add_argument("--output-dir", type=str, default="runs/mappo_sweep")

    # ---- ablate ----
    abl_p = sub.add_parser("ablate", help="Ablation study")
    abl_p.add_argument("--iterations", type=int, default=30)
    abl_p.add_argument("--rollout-length", type=int, default=64)
    abl_p.add_argument("--seed", type=int, default=42)
    abl_p.add_argument("--output-dir", type=str, default="runs/mappo_ablation")

    return parser.parse_args()


def cmd_train(args: argparse.Namespace) -> None:
    """Run a full MAPPO training session."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg: dict[str, Any] = {}
    if args.vessels is not None:
        env_cfg["num_vessels"] = args.vessels
    if args.ports is not None:
        env_cfg["num_ports"] = args.ports
    env_cfg.setdefault("num_vessels", 8)
    env_cfg.setdefault("num_ports", 5)
    env_cfg["rollout_steps"] = args.rollout_length + 5

    mappo_cfg = MAPPOConfig(
        rollout_length=args.rollout_length,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        total_iterations=args.iterations,
    )

    ckpt_dir = args.checkpoint_dir or str(out_dir / "checkpoints")
    logger = TrainingLogger(
        log_dir=str(out_dir / "logs"),
        experiment_name="mappo_train",
        console=True,
        print_every=max(1, args.iterations // 20),
    )

    print(f"Starting MAPPO training: {args.iterations} iterations")
    print(f"  env: {env_cfg['num_vessels']} vessels, {env_cfg['num_ports']} ports")
    print(f"  lr={args.lr}, ent={args.entropy_coeff}, rollout={args.rollout_length}")
    print(f"  output: {out_dir}")
    t0 = time.time()

    trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=mappo_cfg, seed=args.seed)
    history = trainer.train(
        num_iterations=args.iterations,
        eval_interval=args.eval_interval,
        log_fn=logger.log,
        checkpoint_dir=ckpt_dir,
    )
    elapsed = time.time() - t0
    logger.close()

    # Save training history
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "train_history.csv", index=False)

    # Save final model
    trainer.save_models(str(out_dir / "final_model"))

    # Multi-episode eval
    eval_result = trainer.evaluate_episodes(num_episodes=5)
    with open(out_dir / "eval_result.json", "w") as f:
        json.dump(eval_result, f, indent=2, default=str)

    # Generate report
    report = generate_training_report(
        history=history,
        eval_result=eval_result,
        config={"env": env_cfg, "mappo": mappo_cfg.__dict__},
        elapsed_seconds=elapsed,
    )
    (out_dir / "report.md").write_text(report)

    # Plot training curves
    plot_training_curves(df, out_path=str(out_dir / "training_curves.png"))

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  final reward: {history[-1]['mean_reward']:.4f}")
    print(f"  eval mean total: {eval_result['mean']['total_reward']:.4f}")
    print(f"  outputs: {out_dir}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Train MAPPO and compare against baselines."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running MAPPO comparison: {args.iterations} iterations")
    t0 = time.time()

    results = run_mappo_comparison(
        train_iterations=args.iterations,
        rollout_length=args.rollout_length,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # Save per-policy CSVs
    for name, df in results.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    if not args.no_plots:
        plot_mappo_comparison(results, out_path=str(out_dir / "comparison.png"))
        if "_train_log" in results:
            plot_training_curves(
                results["_train_log"],
                out_path=str(out_dir / "training_curves.png"),
            )

    print(f"\nComparison complete in {elapsed:.1f}s → {out_dir}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run hyperparameter grid sweep."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_grid: dict[str, list[Any]] = {
        "lr": [1e-4, 3e-4, 1e-3],
        "entropy_coeff": [0.005, 0.01, 0.05],
    }

    print(f"Running MAPPO sweep: {args.iterations} iters, {len(param_grid)} params")
    t0 = time.time()

    df = run_mappo_hyperparam_sweep(
        param_grid=param_grid,
        train_iterations=args.iterations,
        rollout_length=args.rollout_length,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    df.to_csv(out_dir / "sweep_results.csv", index=False)
    print(f"\nSweep complete in {elapsed:.1f}s ({len(df)} configs)")
    print(df.sort_values("total_reward", ascending=False).head(5).to_string())
    print(f"Results: {out_dir / 'sweep_results.csv'}")


def cmd_ablate(args: argparse.Namespace) -> None:
    """Run ablation study."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablations: dict[str, dict[str, Any]] = {
        "baseline": {},
        "no_reward_norm": {"normalize_rewards": False},
        "no_obs_norm": {"normalize_observations": False},
        "no_value_clip": {"value_clip_eps": 0.0},
        "high_entropy": {"entropy_coeff": 0.05},
        "low_entropy": {"entropy_coeff": 0.001},
        "no_kl_stop": {"target_kl": 0.0},
        "small_net": {"hidden_dims": [32, 32]},
        # Environment ablations (env_ prefix → env-config override)
        "weather_on": {"env_weather_enabled": True},
        "weather_harsh": {"env_weather_enabled": True, "env_sea_state_max": 5.0},
    }

    print(f"Running MAPPO ablation: {args.iterations} iters, {len(ablations)} variants")
    t0 = time.time()

    df = run_mappo_ablation(
        ablations=ablations,
        train_iterations=args.iterations,
        rollout_length=args.rollout_length,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    df.to_csv(out_dir / "ablation_results.csv", index=False)
    print(f"\nAblation complete in {elapsed:.1f}s ({len(df)} variants)")
    print(df[["ablation", "final_mean_reward", "best_mean_reward", "total_reward"]].to_string())
    print(f"Results: {out_dir / 'ablation_results.csv'}")


def main() -> None:
    args = parse_args()
    if args.command is None:
        print("No command specified. Use: train | compare | sweep | ablate")
        sys.exit(1)

    dispatch = {
        "train": cmd_train,
        "compare": cmd_compare,
        "sweep": cmd_sweep,
        "ablate": cmd_ablate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
