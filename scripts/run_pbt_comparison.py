#!/usr/bin/env python3
"""Run fair baseline comparison + PBT-tuned run, then generate figures.

Step 1: Single-seed baseline (200 iters) with queue-imbalance penalty + entropy annealing
Step 2: PBT-tuned run (500 iters) using best hyperparams from PBT
Step 3: Generate comparison figures

Usage:
    python scripts/run_pbt_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

# ---------------------------------------------------------------------------
# Shared environment config (identical for all runs)
# ---------------------------------------------------------------------------
ENV_CONFIG: dict = {
    "num_vessels": 8,
    "num_ports": 5,
    "docks_per_port": 3,
    "rollout_steps": 69,
    "episode_mode": "continuous",
    "forecast_source": "ground_truth",
    "weather_enabled": True,
    "weather_autocorrelation": 0.7,
}


def run_baseline(output_dir: Path) -> list[dict]:
    """Step 1: Standard baseline with entropy annealing (200 iters)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = MAPPOConfig(
        lr=3e-4,
        lr_end=1e-4,
        rollout_length=64,
        num_epochs=4,
        minibatch_size=128,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_dims=[64, 64],
        vessel_hidden_dims=[128, 128],
        normalize_observations=True,
        normalize_rewards=True,
        entropy_coeff=0.05,
        entropy_coeff_end=0.002,
        max_grad_norm=0.5,
        max_grad_norm_start=5.0,
        grad_norm_warmup_fraction=0.1,
    )
    trainer = MAPPOTrainer(env_config=ENV_CONFIG, mappo_config=cfg, seed=42)

    print("=== Step 1: Baseline (200 iters, entropy 0.05→0.002) ===")
    history = trainer.train(
        num_iterations=200,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    # Save reward curve
    rewards = [h["mean_reward"] for h in history]
    pd.DataFrame({"iteration": range(len(rewards)), "reward": rewards}).to_csv(
        output_dir / "baseline_rewards.csv", index=False
    )
    trainer.save_models(str(output_dir / "model"))
    print(f"  Last-20 mean: {np.mean(rewards[-20:]):.2f}")
    return history


def run_pbt_tuned(output_dir: Path) -> list[dict]:
    """Step 2: PBT-discovered hyperparams, 500 iters with entropy annealing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # PBT best worker 2 found: lr=4.32e-4, entropy=0.05, clip_eps=0.139
    cfg = MAPPOConfig(
        lr=4.32e-4,
        lr_end=1e-4,
        rollout_length=64,
        num_epochs=4,
        minibatch_size=128,
        clip_eps=0.139,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_dims=[64, 64],
        vessel_hidden_dims=[128, 128],
        normalize_observations=True,
        normalize_rewards=True,
        entropy_coeff=0.05,
        entropy_coeff_end=0.002,
        max_grad_norm=0.5,
        max_grad_norm_start=5.0,
        grad_norm_warmup_fraction=0.1,
    )
    trainer = MAPPOTrainer(env_config=ENV_CONFIG, mappo_config=cfg, seed=42)

    print("=== Step 2: PBT-tuned (500 iters, lr=4.32e-4, clip=0.139) ===")
    history = trainer.train(
        num_iterations=500,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    rewards = [h["mean_reward"] for h in history]
    pd.DataFrame({"iteration": range(len(rewards)), "reward": rewards}).to_csv(
        output_dir / "pbt_tuned_rewards.csv", index=False
    )
    trainer.save_models(str(output_dir / "model"))
    print(f"  Last-20 mean: {np.mean(rewards[-20:]):.2f}")
    return history


def generate_figures(
    baseline_history: list[dict],
    pbt_tuned_history: list[dict],
    pbt_rewards_csv: Path,
    fig_dir: Path,
) -> None:
    """Step 3: Generate PBT comparison figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_r = [h["mean_reward"] for h in baseline_history]
    tuned_r = [h["mean_reward"] for h in pbt_tuned_history]

    # Load PBT worker curves
    pbt_df = pd.read_csv(pbt_rewards_csv)
    pbt_workers = {
        col: pbt_df[col].dropna().values
        for col in pbt_df.columns if col.startswith("worker_")
    }

    # --- Figure 1: PBT worker reward curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (col, vals) in enumerate(sorted(pbt_workers.items())):
        window = min(10, len(vals))
        smoothed = pd.Series(vals).rolling(window, min_periods=1).mean().values
        ax.plot(smoothed, color=colors[i], alpha=0.8, label=f"Worker {i}")
        ax.plot(vals, color=colors[i], alpha=0.15, linewidth=0.5)
    # Mark exploit events
    exploit_log_path = pbt_rewards_csv.parent / "pbt_exploit_log.json"
    if exploit_log_path.exists():
        exploit_log = json.load(open(exploit_log_path))
        for entry in exploit_log:
            r = entry["round"]
            for ev in entry.get("events", []):
                it_pos = (r + 1) * 10  # exploit happens after round
                if it_pos < 200:
                    ax.axvline(it_pos, color="gray", alpha=0.15, linewidth=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("PBT Training: Per-Worker Reward Curves (4 workers × 200 iters)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_pbt_worker_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'fig_pbt_worker_curves.png'}")

    # --- Figure 2: Baseline vs PBT-tuned comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 10
    bl_smooth = pd.Series(baseline_r).rolling(window, min_periods=1).mean().values
    tu_smooth = pd.Series(tuned_r).rolling(window, min_periods=1).mean().values

    ax.plot(bl_smooth, color="#1f77b4", alpha=0.9, linewidth=1.5,
            label=f"Baseline (lr=3e-4, clip=0.2) last-20={np.mean(baseline_r[-20:]):.1f}")
    ax.plot(baseline_r, color="#1f77b4", alpha=0.12, linewidth=0.5)
    ax.plot(tu_smooth, color="#d62728", alpha=0.9, linewidth=1.5,
            label=f"PBT-tuned (lr=4.3e-4, clip=0.139) last-20={np.mean(tuned_r[-20:]):.1f}")
    ax.plot(tuned_r, color="#d62728", alpha=0.12, linewidth=0.5)
    ax.axvline(200, color="gray", linestyle="--", alpha=0.4, label="Baseline ends (200)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Baseline vs PBT-Tuned Hyperparameters")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_pbt_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'fig_pbt_comparison.png'}")

    # --- Figure 3: PBT hyperparameter evolution ---
    if exploit_log_path.exists():
        exploit_log = json.load(open(exploit_log_path))

        # Load full hyperparam history from PBT summary
        summary_path = pbt_rewards_csv.parent / "pbt_summary.json"
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        hp_names = ["lr", "entropy_coeff", "clip_eps"]
        hp_labels = ["Learning Rate", "Entropy Coeff", "Clip Epsilon"]

        # Reconstruct per-round hyperparams from exploit log
        for ax, hp_name, hp_label in zip(axes, hp_names, hp_labels):
            for i in range(4):
                # Use the means as proxy x-axis (by round)
                rounds = [e["round"] for e in exploit_log]
                # Get HP values from events
                vals_by_round = []
                for entry in exploit_log:
                    for ev in entry.get("events", []):
                        if ev["target"] == i:
                            vals_by_round.append((entry["round"], ev["new_hyperparams"][hp_name]))
                if vals_by_round:
                    rs, vs = zip(*vals_by_round)
                    ax.scatter(rs, vs, color=colors[i], alpha=0.6, s=20, label=f"W{i}" if hp_name == "lr" else None)
            ax.set_xlabel("PBT Round")
            ax.set_ylabel(hp_label)
            ax.set_title(hp_label)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("PBT Hyperparameter Perturbations", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_pbt_hyperparams.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {fig_dir / 'fig_pbt_hyperparams.png'}")


def main() -> None:
    base_dir = Path("runs/pbt_comparison")
    base_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Step 1
    baseline_history = run_baseline(base_dir / "baseline")

    # Step 2
    pbt_tuned_history = run_pbt_tuned(base_dir / "pbt_tuned")

    # Step 3: Figures
    print("\n=== Step 3: Generating figures ===")
    generate_figures(
        baseline_history=baseline_history,
        pbt_tuned_history=pbt_tuned_history,
        pbt_rewards_csv=Path("runs/pbt/pbt_rewards.csv"),
        fig_dir=Path("figures"),
    )

    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Print summary table
    bl_last20 = np.mean([h["mean_reward"] for h in baseline_history[-20:]])
    tu_last20 = np.mean([h["mean_reward"] for h in pbt_tuned_history[-20:]])
    pbt_df = pd.read_csv("runs/pbt/pbt_rewards.csv")
    pbt_best = pbt_df[[c for c in pbt_df.columns if c.startswith("worker_")]].apply(
        lambda c: c.dropna().iloc[-20:].mean()
    ).min()

    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Run':<35} {'Last-20 Mean':>12} {'Iters':>8}")
    print("-" * 65)
    print(f"{'Baseline (entropy annealing)':<35} {bl_last20:>12.2f} {'200':>8}")
    print(f"{'PBT (4 workers, no annealing)':<35} {pbt_best:>12.2f} {'200':>8}")
    print(f"{'PBT-tuned (best HP + annealing)':<35} {tu_last20:>12.2f} {'500':>8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
