#!/usr/bin/env python3
"""Benchmark architectural improvements: attention coordinator, encoded critic,
recurrent vessel actor, and all three combined.

Runs 5 configurations × 200 iterations each using PBT-tuned hyperparams:
  1. Baseline (standard MLP actors + critics)
  2. Attention coordinator only
  3. Encoded critic only
  4. Recurrent vessel actor only
  5. All three combined

Usage:
    python scripts/run_arch_benchmark.py
"""

from __future__ import annotations

import gc
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
# Shared settings
# ---------------------------------------------------------------------------
NUM_ITERS = 200
SEED = 42

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

# PBT-tuned base hyperparams
BASE_MAPPO = dict(
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

CONFIGS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("attention", {"coordinator_use_attention": True}),
    ("encoded_critic", {"use_encoded_critic": True}),
    ("recurrent", {"vessel_use_recurrence": True}),
    ("all_three", {
        "coordinator_use_attention": True,
        "use_encoded_critic": True,
        "vessel_use_recurrence": True,
    }),
]


def run_one(name: str, extra_cfg: dict, output_dir: Path) -> list[float]:
    """Train one configuration, save rewards CSV, return reward list."""
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = {**BASE_MAPPO, **extra_cfg}
    cfg = MAPPOConfig(**merged)
    trainer = MAPPOTrainer(env_config=ENV_CONFIG, mappo_config=cfg, seed=SEED)

    print(f"  Training {name} ({NUM_ITERS} iters) ...", flush=True)
    t0 = time.perf_counter()
    history = trainer.train(
        num_iterations=NUM_ITERS,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    elapsed = time.perf_counter() - t0
    rewards = [h["mean_reward"] for h in history]
    pd.DataFrame({"iteration": range(len(rewards)), "reward": rewards}).to_csv(
        output_dir / f"{name}_rewards.csv", index=False
    )
    last20 = np.mean(rewards[-20:])
    print(f"  {name}: last-20={last20:.2f}, best={max(rewards):.2f}, time={elapsed:.0f}s", flush=True)

    # Free memory between runs
    del trainer, history
    gc.collect()
    return rewards


def generate_figure(all_rewards: dict[str, list[float]], fig_dir: Path) -> None:
    """Generate comparison figure for all architectural configs."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "baseline": "#1f77b4",
        "attention": "#ff7f0e",
        "encoded_critic": "#2ca02c",
        "recurrent": "#d62728",
        "all_three": "#9467bd",
    }
    labels = {
        "baseline": "Baseline (MLP)",
        "attention": "Attention Coordinator",
        "encoded_critic": "Encoded Critic",
        "recurrent": "Recurrent Vessel",
        "all_three": "All Three Combined",
    }
    window = 10
    for name, rewards in all_rewards.items():
        smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean().values
        last20 = np.mean(rewards[-20:])
        ax.plot(smoothed, color=colors[name], alpha=0.9, linewidth=1.5,
                label=f"{labels[name]} (last-20={last20:.1f})")
        ax.plot(rewards, color=colors[name], alpha=0.12, linewidth=0.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Architectural Improvements Comparison (200 iters, PBT-tuned HP)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_arch_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'fig_arch_comparison.png'}")


def main() -> None:
    base_dir = Path("runs/arch_benchmark")
    base_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    all_rewards: dict[str, list[float]] = {}

    for name, extra_cfg in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        rewards = run_one(name, extra_cfg, base_dir / name)
        all_rewards[name] = rewards

    # Generate figure
    print("\n=== Generating figure ===", flush=True)
    generate_figure(all_rewards, Path("figures"))

    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Summary table
    print("\n" + "=" * 70)
    print("ARCHITECTURE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'Last-20':>10} {'Best':>10} {'Δ vs base':>10}")
    print("-" * 70)
    bl_last20 = np.mean(all_rewards["baseline"][-20:])
    for name, rewards in all_rewards.items():
        last20 = np.mean(rewards[-20:])
        best = max(rewards)
        delta = last20 - bl_last20
        print(f"{name:<25} {last20:>10.2f} {best:>10.2f} {delta:>+10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
