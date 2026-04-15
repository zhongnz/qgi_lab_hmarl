#!/usr/bin/env python3
"""Full-scale production run: PBT-tuned HP + all architectural improvements.

Trains 5 seeds × 500 iterations with the best known configuration,
generates publication figures, and prints a summary table.

Estimated wall-clock: ~2.5–3 hours on 12-core CPU (no GPU).

Usage:
    python scripts/run_production.py
"""

from __future__ import annotations

import gc
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
# Configuration
# ---------------------------------------------------------------------------
NUM_ITERS = 500
SEEDS = [42, 49, 56, 63, 70]
OUTPUT_DIR = Path("runs/production")
FIG_DIR = Path("figures")

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

# PBT-tuned hyperparams + all 3 architectural improvements
MAPPO_KWARGS: dict = {
    "lr": 4.32e-4,
    "lr_end": 1e-4,
    "rollout_length": 64,
    "num_epochs": 4,
    "minibatch_size": 128,
    "clip_eps": 0.139,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "hidden_dims": [64, 64],
    "vessel_hidden_dims": [128, 128],
    "normalize_observations": True,
    "normalize_rewards": True,
    "entropy_coeff": 0.05,
    "entropy_coeff_end": 0.002,
    "max_grad_norm": 0.5,
    "max_grad_norm_start": 5.0,
    "grad_norm_warmup_fraction": 0.1,
    "coordinator_use_attention": True,
    "use_encoded_critic": True,
    "vessel_use_recurrence": True,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seed(seed: int, output_dir: Path) -> list[dict]:
    """Train a single seed. Returns full history."""
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cfg = MAPPOConfig(**MAPPO_KWARGS)
    trainer = MAPPOTrainer(env_config=ENV_CONFIG, mappo_config=cfg, seed=seed)

    print(f"  Training seed={seed} ({NUM_ITERS} iters) ...", flush=True)
    t0 = time.perf_counter()
    history = trainer.train(
        num_iterations=NUM_ITERS,
        checkpoint_dir=str(seed_dir / "checkpoints"),
    )
    elapsed = time.perf_counter() - t0

    # Save per-seed metrics
    rewards = [h["mean_reward"] for h in history]
    pd.DataFrame(history).to_csv(seed_dir / "metrics.csv", index=False)
    pd.DataFrame({"iteration": range(len(rewards)), "reward": rewards}).to_csv(
        seed_dir / "rewards.csv", index=False
    )
    trainer.save_models(str(seed_dir / "model"))

    last20 = np.mean(rewards[-20:])
    best = max(rewards)
    print(f"  seed={seed}: last-20={last20:.2f}, best={best:.2f}, "
          f"time={elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Free memory
    del trainer
    gc.collect()

    return history


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(
    all_histories: dict[int, list[dict]],
    fig_dir: Path,
) -> None:
    """Generate production figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(all_histories.keys())
    n_seeds = len(seeds)

    # Build reward matrix (n_seeds × max_iters)
    max_len = max(len(h) for h in all_histories.values())
    reward_mat = np.full((n_seeds, max_len), np.nan)
    for si, seed in enumerate(seeds):
        rewards = [h["mean_reward"] for h in all_histories[seed]]
        reward_mat[si, :len(rewards)] = rewards

    iters = np.arange(max_len)
    mean_r = np.nanmean(reward_mat, axis=0)
    std_r = np.nanstd(reward_mat, axis=0)
    window = 20

    # --- Figure 15: Production multi-seed training curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_seed = plt.cm.tab10(np.linspace(0, 1, n_seeds))

    for si, seed in enumerate(seeds):
        rewards = reward_mat[si]
        valid = ~np.isnan(rewards)
        smoothed = pd.Series(rewards[valid]).rolling(window, min_periods=1).mean().values
        ax.plot(iters[valid], smoothed, color=colors_seed[si], alpha=0.5,
                linewidth=0.8, label=f"Seed {seed}")

    # Mean ± 1σ band
    smoothed_mean = pd.Series(mean_r).rolling(window, min_periods=1).mean().values
    smoothed_lo = pd.Series(mean_r - std_r).rolling(window, min_periods=1).mean().values
    smoothed_hi = pd.Series(mean_r + std_r).rolling(window, min_periods=1).mean().values

    ax.plot(iters, smoothed_mean, color="black", linewidth=2.5,
            label=f"Mean (last-20={np.nanmean(reward_mat[:, -20:]):.1f})")
    ax.fill_between(iters, smoothed_lo, smoothed_hi,
                     alpha=0.15, color="steelblue", label="± 1σ")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Mean Reward", fontsize=11)
    ax.set_title(f"Production Run: {n_seeds} Seeds × {max_len} Iterations\n"
                 f"(PBT-tuned HP + Attention + Encoded Critic + Recurrent Vessel)",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_production_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'fig_production_curves.png'}")

    # --- Figure 16: Final configuration comparison bar chart ---
    # Collect previous run results for comparison
    prev_results = _load_previous_results()
    prev_results["Production (this run)"] = {
        "last20_mean": float(np.nanmean(reward_mat[:, -20:])),
        "last20_std": float(np.nanstd(np.nanmean(reward_mat[:, -20:], axis=1))),
        "best": float(np.nanmax(reward_mat)),
        "iters": max_len,
        "seeds": n_seeds,
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(prev_results.keys())
    means = [prev_results[n]["last20_mean"] for n in names]
    stds = [prev_results[n].get("last20_std", 0.0) for n in names]
    bests = [prev_results[n]["best"] for n in names]
    x = np.arange(len(names))

    bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8,
                  edgecolor="darkblue", linewidth=0.5)
    # Highlight the production bar
    bars[-1].set_color("#2ca02c")
    bars[-1].set_edgecolor("darkgreen")

    # Annotate with best reward
    for i, (m, b) in enumerate(zip(means, bests)):
        ax.annotate(f"best: {b:.1f}", (x[i], m - 0.5),
                    ha="center", va="top", fontsize=8, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Last-20 Mean Reward", fontsize=11)
    ax.set_title("Configuration Comparison: All Experiments", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_production_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'fig_production_comparison.png'}")


def _load_previous_results() -> dict:
    """Load last-20 mean rewards from prior experiments for comparison."""
    results = {}

    # 1. Baseline (from PBT comparison)
    bl_csv = Path("runs/pbt_comparison/baseline/baseline_rewards.csv")
    if bl_csv.exists():
        df = pd.read_csv(bl_csv)
        rewards = df["reward"].values
        results["Baseline MLP\n(200 iters)"] = {
            "last20_mean": float(np.mean(rewards[-20:])),
            "last20_std": 0.0,
            "best": float(np.max(rewards)),
            "iters": len(rewards),
            "seeds": 1,
        }

    # 2. PBT-tuned MLP (from PBT comparison)
    pbt_csv = Path("runs/pbt_comparison/pbt_tuned/pbt_tuned_rewards.csv")
    if pbt_csv.exists():
        df = pd.read_csv(pbt_csv)
        rewards = df["reward"].values
        results["PBT-tuned MLP\n(500 iters)"] = {
            "last20_mean": float(np.mean(rewards[-20:])),
            "last20_std": 0.0,
            "best": float(np.max(rewards)),
            "iters": len(rewards),
            "seeds": 1,
        }

    # 3. Arch benchmark configs
    for name, label in [
        ("baseline", "Arch: MLP\n(200 iters)"),
        ("attention", "Arch: Attention\n(200 iters)"),
        ("encoded_critic", "Arch: EncodedCritic\n(200 iters)"),
        ("all_three", "Arch: All Three\n(200 iters)"),
    ]:
        csv_path = Path(f"runs/arch_benchmark/{name}/{name}_rewards.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rewards = df["reward"].values
            results[label] = {
                "last20_mean": float(np.mean(rewards[-20:])),
                "last20_std": 0.0,
                "best": float(np.max(rewards)),
                "iters": len(rewards),
                "seeds": 1,
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PRODUCTION RUN: PBT-tuned HP + All Architectural Improvements")
    print(f"  Seeds: {SEEDS}")
    print(f"  Iterations: {NUM_ITERS}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70, flush=True)

    t0_total = time.perf_counter()
    all_histories: dict[int, list[dict]] = {}

    for i, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed} ({i+1}/{len(SEEDS)}) ===", flush=True)
        history = train_seed(seed, OUTPUT_DIR)
        all_histories[seed] = history

    total_elapsed = time.perf_counter() - t0_total

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PRODUCTION RUN COMPLETE")
    print(f"Total wall-clock: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print("=" * 70)

    # Per-seed summary
    print(f"\n{'Seed':<8} {'Last-20':>10} {'Best':>10} {'Final':>10}")
    print("-" * 48)
    all_last20 = []
    all_best = []
    for seed in SEEDS:
        rewards = [h["mean_reward"] for h in all_histories[seed]]
        last20 = np.mean(rewards[-20:])
        best = max(rewards)
        final = rewards[-1]
        all_last20.append(last20)
        all_best.append(best)
        print(f"{seed:<8} {last20:>10.2f} {best:>10.2f} {final:>10.2f}")

    print("-" * 48)
    mean_last20 = np.mean(all_last20)
    std_last20 = np.std(all_last20)
    mean_best = np.mean(all_best)
    print(f"{'Mean':<8} {mean_last20:>10.2f} {mean_best:>10.2f}")
    print(f"{'Std':<8} {std_last20:>10.2f}")
    print(f"{'Best':<8} {'':>10} {max(all_best):>10.2f}")

    # Save aggregate summary
    summary = {
        "config": MAPPO_KWARGS,
        "env_config": ENV_CONFIG,
        "seeds": SEEDS,
        "num_iterations": NUM_ITERS,
        "wall_clock_seconds": total_elapsed,
        "per_seed": {
            seed: {
                "last20_mean": float(np.mean(
                    [h["mean_reward"] for h in all_histories[seed]][-20:]
                )),
                "best": float(max(h["mean_reward"] for h in all_histories[seed])),
            }
            for seed in SEEDS
        },
        "aggregate": {
            "mean_last20": float(mean_last20),
            "std_last20": float(std_last20),
            "mean_best": float(mean_best),
            "overall_best": float(max(all_best)),
        },
    }
    with open(OUTPUT_DIR / "production_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {OUTPUT_DIR / 'production_summary.json'}")

    # Save aggregate CSV (all seeds, aligned by iteration)
    max_len = max(len(all_histories[s]) for s in SEEDS)
    agg_data = {"iteration": list(range(max_len))}
    for seed in SEEDS:
        rewards = [h["mean_reward"] for h in all_histories[seed]]
        padded = rewards + [np.nan] * (max_len - len(rewards))
        agg_data[f"seed_{seed}"] = padded
    agg_df = pd.DataFrame(agg_data)
    seed_cols = [f"seed_{s}" for s in SEEDS]
    agg_df["mean"] = agg_df[seed_cols].mean(axis=1)
    agg_df["std"] = agg_df[seed_cols].std(axis=1)
    agg_df.to_csv(OUTPUT_DIR / "production_rewards.csv", index=False)

    # --- Figures ---
    print("\n=== Generating figures ===", flush=True)
    generate_figures(all_histories, FIG_DIR)

    print("\n" + "=" * 70)
    print("ALL DONE.")
    print(f"  Aggregate last-20: {mean_last20:.2f} ± {std_last20:.2f}")
    print(f"  Overall best reward: {max(all_best):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
