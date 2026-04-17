"""Weather impact experiment: MAPPO weather ON vs OFF, 1000 iterations.

Answers two questions:
1. Does the 3-agent hierarchy learn to coordinate? (all EVs > 0.3, beats baselines)
2. Does weather asymmetrically impact agents by hierarchy level?

Output: training curves, evaluation metrics at checkpoints, baseline comparison,
and significance testing.  Results saved to figures/weather_experiment.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any

sys.path.insert(0, ".")

import numpy as np

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import run_experiment
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.stats import welch_t_test

# ── Experiment config ────────────────────────────────────────────────────
NUM_ITERATIONS = 750
EVAL_STEPS = 69
SEED = 42
LOG_EVERY = 25
EVAL_AT = [250, 500, 750]
EVAL_EPISODES = 5

ENV_KWARGS: dict[str, Any] = dict(
    num_ports=3,
    num_vessels=15,
    num_coordinators=1,
    docks_per_port=2,
    rollout_steps=EVAL_STEPS,
    forecast_source="ground_truth",
    emission_weight=0.0,
    coordinator_compliance_weight=4.5,
)

MAPPO_KWARGS: dict[str, Any] = dict(
    rollout_length=128,
    num_epochs=4,
    minibatch_size=64,
    total_iterations=NUM_ITERATIONS,
)

EVAL_METRICS = [
    "completed_arrivals", "on_time_rate", "total_fuel_used",
    "total_vessels_served", "avg_schedule_delay_hours", "dock_utilization",
]

BASELINES = ["independent", "reactive", "ground_truth"]

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────
def train_mappo(
    weather_enabled: bool,
    label: str,
) -> dict[str, Any]:
    """Train MAPPO for NUM_ITERATIONS and return all logged data."""
    cfg = get_default_config(**ENV_KWARGS, weather_enabled=weather_enabled)
    mappo_cfg = MAPPOConfig(**MAPPO_KWARGS)
    trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=SEED)

    log_rows: list[dict[str, Any]] = []
    eval_checkpoints: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()

    print(f"\n{'=' * 70}")
    print(f"Training MAPPO [{label}]  weather={weather_enabled}  seed={SEED}")
    print(f"{'=' * 70}")

    for i in range(NUM_ITERATIONS):
        rollout_info = trainer.collect_rollout()
        update_results = trainer.update()
        iteration = i + 1

        # ── Log every LOG_EVERY iterations ──
        if i % LOG_EVERY == 0 or iteration == NUM_ITERATIONS:
            v = update_results["vessel"]
            p = update_results["port"]
            c = update_results["coordinator"]
            elapsed = time.perf_counter() - t0

            row = {
                "iteration": iteration,
                "elapsed_s": round(elapsed, 1),
                "mean_reward": rollout_info["mean_reward"],
                "vessel_mean_reward": rollout_info["vessel_mean_reward"],
                "port_mean_reward": rollout_info["port_mean_reward"],
                "coordinator_mean_reward": rollout_info["coordinator_mean_reward"],
                "v_EV": v.explained_variance,
                "p_EV": p.explained_variance,
                "c_EV": c.explained_variance,
                "v_entropy": v.entropy,
                "p_entropy": p.entropy,
                "c_entropy": c.entropy,
                "v_policy_loss": v.policy_loss,
                "p_policy_loss": p.policy_loss,
                "c_policy_loss": c.policy_loss,
                "v_value_loss": v.value_loss,
                "p_value_loss": p.value_loss,
                "c_value_loss": c.value_loss,
                "v_approx_kl": v.approx_kl,
                "p_approx_kl": p.approx_kl,
                "c_approx_kl": c.approx_kl,
            }
            log_rows.append(row)

            print(
                f"[{iteration:4d}/{NUM_ITERATIONS}]  "
                f"reward={rollout_info['mean_reward']:+7.2f}  "
                f"v_EV={v.explained_variance:+.3f}  "
                f"p_EV={p.explained_variance:+.3f}  "
                f"c_EV={c.explained_variance:+.3f}  "
                f"[{elapsed:.0f}s]",
                flush=True,
            )

        # ── Save checkpoint at eval points ──
        if iteration in EVAL_AT:
            ckpt_prefix = os.path.join(OUT_DIR, f"{label}_iter{iteration}")
            trainer.save_models(ckpt_prefix)
            print(f"  → Checkpoint saved: {ckpt_prefix}", flush=True)

        # ── Evaluate at checkpoints ──
        if iteration in EVAL_AT:
            print(f"  → Evaluating at iteration {iteration} ({EVAL_EPISODES} episodes)...", flush=True)
            eval_result = trainer.evaluate_episodes(
                num_episodes=EVAL_EPISODES,
                num_steps=EVAL_STEPS,
            )
            eval_checkpoints[iteration] = eval_result
            m = eval_result["mean"]
            print(
                f"    arrivals={m.get('completed_arrivals', 0):.1f}  "
                f"on_time={m.get('on_time_rate', 0):.3f}  "
                f"fuel={m.get('total_fuel_used', 0):.1f}  "
                f"served={m.get('total_vessels_served', 0):.1f}",
                flush=True,
            )

    train_time = time.perf_counter() - t0
    print(f"\n{label} training complete in {train_time:.0f}s ({train_time / 60:.1f} min)")

    return {
        "label": label,
        "weather_enabled": weather_enabled,
        "seed": SEED,
        "train_time_s": round(train_time, 1),
        "training_log": log_rows,
        "eval_checkpoints": {
            str(k): v for k, v in eval_checkpoints.items()
        },
    }


def run_baselines(weather_enabled: bool) -> dict[str, dict[str, float]]:
    """Run heuristic baselines and return final-step metrics."""
    results = {}
    for name in BASELINES:
        env_kw = dict(ENV_KWARGS, weather_enabled=weather_enabled)
        df = run_experiment(
            policy_type=name,
            steps=EVAL_STEPS,
            seed=SEED,
            config=env_kw,
        )
        last = df.iloc[-1]
        results[name] = {k: float(last.get(k, 0.0)) for k in EVAL_METRICS}
    return results


def print_comparison_table(
    mappo_off: dict[str, Any],
    mappo_on: dict[str, Any],
    baselines_off: dict[str, dict[str, float]],
    baselines_on: dict[str, dict[str, float]],
) -> None:
    """Print a combined comparison table."""
    print(f"\n{'=' * 90}")
    print("COMPARISON TABLE")
    print(f"{'=' * 90}")

    header = (
        f"{'Policy':<20} {'Weather':<8} "
        f"{'arrivals':>10} {'on_time':>10} {'fuel':>10} "
        f"{'served':>10} {'sched_delay':>12} {'dock_util':>10}"
    )
    print(header)
    print("-" * len(header))

    def print_row(name: str, weather: str, m: dict[str, float]) -> None:
        print(
            f"{name:<20} {weather:<8} "
            f"{m.get('completed_arrivals', 0):>10.1f} "
            f"{m.get('on_time_rate', 0):>10.3f} "
            f"{m.get('total_fuel_used', 0):>10.1f} "
            f"{m.get('total_vessels_served', 0):>10.1f} "
            f"{m.get('avg_schedule_delay_hours', 0):>12.2f} "
            f"{m.get('dock_utilization', 0):>10.3f}"
        )

    # Baselines
    for name in BASELINES:
        print_row(name, "OFF", baselines_off[name])
        print_row(name, "ON", baselines_on[name])

    # MAPPO — use the last checkpoint
    last_ckpt = str(max(int(k) for k in mappo_off["eval_checkpoints"]))
    m_off = mappo_off["eval_checkpoints"][last_ckpt]["mean"]
    m_on = mappo_on["eval_checkpoints"][last_ckpt]["mean"]
    print_row("MAPPO", "OFF", m_off)
    print_row("MAPPO", "ON", m_on)

    print(f"{'=' * 90}")


def print_weather_impact(
    mappo_off: dict[str, Any],
    mappo_on: dict[str, Any],
) -> None:
    """Print weather impact analysis with significance testing."""
    last_ckpt = str(max(int(k) for k in mappo_off["eval_checkpoints"]))
    print(f"\n{'=' * 70}")
    print(f"WEATHER IMPACT ON MAPPO (iteration {last_ckpt}, {EVAL_EPISODES}-episode eval)")
    print(f"{'=' * 70}")

    eps_off = mappo_off["eval_checkpoints"][last_ckpt]["episodes"]
    eps_on = mappo_on["eval_checkpoints"][last_ckpt]["episodes"]

    print(f"\n{'Metric':<25} {'OFF mean':>10} {'ON mean':>10} {'diff':>10} {'p-value':>10} {'sig?':>6}")
    print("-" * 71)

    for metric in EVAL_METRICS:
        vals_off = [ep.get(metric, 0.0) for ep in eps_off]
        vals_on = [ep.get(metric, 0.0) for ep in eps_on]
        result = welch_t_test(vals_off, vals_on)
        sig = "***" if result["p_value"] < 0.01 else "**" if result["p_value"] < 0.05 else "*" if result["p_value"] < 0.1 else ""
        print(
            f"{metric:<25} "
            f"{result['mean_a']:>10.3f} "
            f"{result['mean_b']:>10.3f} "
            f"{result['diff']:>+10.3f} "
            f"{result['p_value']:>10.4f} "
            f"{sig:>6}"
        )

    # EV comparison from training logs
    print(f"\n{'=' * 70}")
    print(f"EXPLAINED VARIANCE AT ITERATION {last_ckpt}")
    print(f"{'=' * 70}")
    log_off = mappo_off["training_log"][-1]
    log_on = mappo_on["training_log"][-1]
    print(f"{'Agent':<15} {'OFF':>10} {'ON':>10} {'diff':>10}")
    print("-" * 45)
    for agent in ["v", "p", "c"]:
        ev_off = log_off[f"{agent}_EV"]
        ev_on = log_on[f"{agent}_EV"]
        name = {"v": "Vessel", "p": "Port", "c": "Coordinator"}[agent]
        print(f"{name:<15} {ev_off:>+10.3f} {ev_on:>+10.3f} {ev_on - ev_off:>+10.3f}")


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    overall_t0 = time.perf_counter()

    # Train MAPPO — both conditions
    mappo_off = train_mappo(weather_enabled=False, label="MAPPO-OFF")
    mappo_on = train_mappo(weather_enabled=True, label="MAPPO-ON")

    # Run baselines — both conditions
    print(f"\n{'=' * 70}")
    print("Running heuristic baselines (weather OFF)...")
    print(f"{'=' * 70}")
    baselines_off = run_baselines(weather_enabled=False)

    print(f"\n{'=' * 70}")
    print("Running heuristic baselines (weather ON)...")
    print(f"{'=' * 70}")
    baselines_on = run_baselines(weather_enabled=True)

    # Print combined comparison
    print_comparison_table(mappo_off, mappo_on, baselines_off, baselines_on)

    # Print weather impact analysis with significance testing
    print_weather_impact(mappo_off, mappo_on)

    # Save raw results
    output = {
        "mappo_off": mappo_off,
        "mappo_on": mappo_on,
        "baselines_off": baselines_off,
        "baselines_on": baselines_on,
    }
    out_path = os.path.join(OUT_DIR, "weather_experiment.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    total_time = time.perf_counter() - overall_t0
    print(f"\nTotal experiment time: {total_time:.0f}s ({total_time / 60:.1f} min)")
