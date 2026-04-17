"""Train MAPPO for 1000 iterations and compare against heuristic baselines.

Evaluation metrics (Pareto comparison, no composite score):
  - completed_arrivals: total trips completed (throughput)
  - on_time_rate: fraction of scheduled arrivals within tolerance
  - total_fuel_used: total fuel consumed (tons, efficiency)
  - total_vessels_served: vessels that completed port service (docking)
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from hmarl_mvp.config import get_default_config
from hmarl_mvp.experiment import run_experiment
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer

NUM_ITERATIONS = 1000
EVAL_STEPS = 69
SEED = 42

METRICS = ["completed_arrivals", "on_time_rate", "total_fuel_used", "total_vessels_served"]

# -- Environment config (shared by MAPPO and baselines) --
env_kwargs = dict(
    num_ports=3,
    num_vessels=10,
    num_coordinators=2,
    docks_per_port=2,
    rollout_steps=EVAL_STEPS,
    weather_enabled=True,
    forecast_source="ground_truth",
    emission_weight=0.0,
    coordinator_compliance_weight=1.5,
)
cfg = get_default_config(**env_kwargs)

# -- MAPPO config (all defaults are now correct) --
mappo_cfg = MAPPOConfig(
    rollout_length=128,
    num_epochs=4,
    total_iterations=NUM_ITERATIONS,
)

# ======================================================================
# Train MAPPO
# ======================================================================
print("=" * 70)
print(f"Training MAPPO: {NUM_ITERATIONS} iterations")
print("=" * 70)

trainer = MAPPOTrainer(env_config=cfg, mappo_config=mappo_cfg, seed=SEED)
t0 = time.perf_counter()

for i in range(NUM_ITERATIONS):
    rollout_info = trainer.collect_rollout()
    update_results = trainer.update()

    if i % 50 == 0 or i == NUM_ITERATIONS - 1:
        v = update_results["vessel"]
        p = update_results["port"]
        c = update_results["coordinator"]
        elapsed = time.perf_counter() - t0
        print(
            f"[{i+1:4d}/{NUM_ITERATIONS}]  "
            f"reward={rollout_info['mean_reward']:+7.2f}  "
            f"v_EV={v.explained_variance:+.3f}  "
            f"p_EV={p.explained_variance:+.3f}  "
            f"c_EV={c.explained_variance:+.3f}  "
            f"c_KL={c.approx_kl:.2e}  "
            f"[{elapsed:.0f}s]"
        )

train_time = time.perf_counter() - t0
print(f"\nTraining complete in {train_time:.0f}s ({train_time/60:.1f} min)")

# ======================================================================
# Evaluate MAPPO
# ======================================================================
print(f"\nEvaluating MAPPO ({EVAL_STEPS} steps, deterministic)...")
mappo_metrics = trainer.evaluate(num_steps=EVAL_STEPS)

# ======================================================================
# Run heuristic baselines
# ======================================================================
baselines = ["independent", "reactive", "ground_truth"]
baseline_metrics: dict[str, dict] = {}

for name in baselines:
    print(f"Running baseline: {name}...")
    df = run_experiment(
        policy_type=name,
        steps=EVAL_STEPS,
        seed=SEED,
        config=env_kwargs,
    )
    last = df.iloc[-1]
    baseline_metrics[name] = {k: float(last.get(k, 0.0)) for k in METRICS}

# ======================================================================
# Comparison table
# ======================================================================
print("\n" + "=" * 78)
print("COMPARISON (Pareto evaluation — no composite score)")
print(f"  {EVAL_STEPS}-step episode, seed={SEED}")
print("=" * 78)

header = f"{'Policy':<16} {'arrivals':>10} {'on_time':>10} {'fuel_used':>10} {'served':>10}"
print(header)
print("-" * len(header))

# Print baselines
for name in baselines:
    m = baseline_metrics[name]
    print(
        f"{name:<16} "
        f"{m['completed_arrivals']:>10.0f} "
        f"{m['on_time_rate']:>10.3f} "
        f"{m['total_fuel_used']:>10.1f} "
        f"{m['total_vessels_served']:>10.0f}"
    )

# Print MAPPO
print(
    f"{'mappo':<16} "
    f"{mappo_metrics.get('completed_arrivals', 0.0):>10.0f} "
    f"{mappo_metrics.get('on_time_rate', 0.0):>10.3f} "
    f"{mappo_metrics.get('total_fuel_used', 0.0):>10.1f} "
    f"{mappo_metrics.get('total_vessels_served', 0.0):>10.0f}"
)
print("=" * 78)

# Also print diagnostic metrics
print("\nDiagnostic metrics:")
diag_keys = ["avg_speed", "avg_delay_hours", "avg_schedule_delay_hours",
             "scheduled_arrivals", "on_time_arrivals", "dock_utilization",
             "avg_queue", "total_wait_hours"]
print(f"{'':>16}", end="")
for k in diag_keys:
    print(f" {k:>12s}", end="")
print()
for name in baselines:
    df = run_experiment(policy_type=name, steps=EVAL_STEPS, seed=SEED, config=env_kwargs)
    last = df.iloc[-1]
    print(f"{name:<16}", end="")
    for k in diag_keys:
        print(f" {float(last.get(k, 0.0)):>12.2f}", end="")
    print()
print(f"{'mappo':<16}", end="")
for k in diag_keys:
    print(f" {mappo_metrics.get(k, 0.0):>12.2f}", end="")
print()

# Pareto dominance check against reactive
reactive = baseline_metrics["reactive"]
wins = 0
wins += int(mappo_metrics.get("completed_arrivals", 0) > reactive["completed_arrivals"])
wins += int(mappo_metrics.get("on_time_rate", 0) > reactive["on_time_rate"])
wins += int(mappo_metrics.get("total_fuel_used", float("inf")) < reactive["total_fuel_used"])
wins += int(mappo_metrics.get("total_vessels_served", 0) > reactive["total_vessels_served"])

print(f"\nMAPPO wins on {wins}/4 metrics vs reactive baseline.")
