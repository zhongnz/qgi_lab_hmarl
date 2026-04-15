#!/usr/bin/env python3
"""Run PBT experiment: 4 workers x 200 iterations.

Usage:
    python scripts/run_pbt.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmarl_mvp.mappo import MAPPOConfig
from hmarl_mvp.pbt import PBTConfig, PBTTrainer

# ---------------------------------------------------------------------------
# Configuration — matches baseline.yaml but PBT manages lr/entropy/clip_eps
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

MAPPO_CONFIG = MAPPOConfig(
    lr=3e-4,
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
    max_grad_norm=0.5,
    max_grad_norm_start=5.0,
    grad_norm_warmup_fraction=0.1,
)

PBT_CONFIG = PBTConfig(
    population_size=4,
    interval=10,
    fraction_top=0.25,
    fraction_bottom=0.25,
    perturb_factor=1.2,
)

TOTAL_ITERATIONS = 200
OUTPUT_DIR = Path("runs/pbt")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PBT experiment: {PBT_CONFIG.population_size} workers x {TOTAL_ITERATIONS} iters")
    print(f"Output: {OUTPUT_DIR}")

    pbt = PBTTrainer(
        env_config=ENV_CONFIG,
        mappo_config=MAPPO_CONFIG,
        pbt_config=PBT_CONFIG,
        base_seed=42,
    )

    # Simple progress logger
    iteration_times: list[float] = []
    last_log = [time.perf_counter()]

    def log_fn(round_idx: int, worker_idx: int, iteration: int, rollout: dict) -> None:
        now = time.perf_counter()
        iteration_times.append(now - last_log[0])
        last_log[0] = now
        total_done = round_idx * PBT_CONFIG.interval * PBT_CONFIG.population_size + \
                     worker_idx * PBT_CONFIG.interval + (iteration % PBT_CONFIG.interval) + 1
        total_all = TOTAL_ITERATIONS * PBT_CONFIG.population_size
        if total_done % 20 == 0 or total_done <= 4:
            pct = 100. * total_done / total_all
            print(
                f"  [{pct:5.1f}%] round={round_idx} worker={worker_idx} "
                f"it={iteration} reward={rollout['mean_reward']:.4f}"
            )

    result = pbt.train(
        total_iterations=TOTAL_ITERATIONS,
        log_fn=log_fn,
        checkpoint_dir=str(OUTPUT_DIR / "checkpoints"),
    )

    # ---- Save results ----
    print(f"\nPBT complete in {result['total_time']:.1f}s")
    print(f"Best worker: {result['best_worker_idx']} "
          f"(mean reward: {result['best_mean_reward']:.4f})")
    print(f"Final hyperparams:")
    for i, hp in enumerate(result["final_hyperparams"]):
        print(f"  worker {i}: {hp}")

    # Save per-worker reward curves as CSV
    max_len = max(len(r) for r in result["per_worker_rewards"])
    reward_df = pd.DataFrame({
        f"worker_{i}": r + [np.nan] * (max_len - len(r))
        for i, r in enumerate(result["per_worker_rewards"])
    })
    reward_df.index.name = "iteration"
    reward_df.to_csv(OUTPUT_DIR / "pbt_rewards.csv")

    # Save exploit log
    with open(OUTPUT_DIR / "pbt_exploit_log.json", "w") as f:
        json.dump(result["exploit_log"], f, indent=2)

    # Save full result summary
    summary = {
        "best_worker_idx": result["best_worker_idx"],
        "best_mean_reward": float(result["best_mean_reward"]),
        "total_rounds": result["total_rounds"],
        "total_time": result["total_time"],
        "final_hyperparams": result["final_hyperparams"],
    }
    with open(OUTPUT_DIR / "pbt_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save best worker models
    best_worker = pbt.workers[result["best_worker_idx"]]
    best_worker.save_models(str(OUTPUT_DIR / "best_model"))

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
