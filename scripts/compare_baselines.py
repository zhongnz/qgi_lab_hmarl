"""Train MAPPO for 1000 iterations and compare against heuristic baselines.

Evaluation metrics (Pareto comparison, no composite score):
  - on_time_rate: fraction of vessels arriving within schedule tolerance
  - total_fuel_used: total fuel consumed (tons)
  - dock_utilization: fraction of dock capacity used
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

# -- Environment config (shared by MAPPO and baselines) --
env_kwargs = dict(
    num_ports=5,
    num_vessels=8,
    rollout_steps=EVAL_STEPS,
    weather_enabled=True,
    forecast_source="ground_truth",
    docks_per_port=1,
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
        c = update_results["coordinator"]
        elapsed = time.perf_counter() - t0
        print(
            f"[{i+1:4d}/{NUM_ITERATIONS}]  "
            f"reward={rollout_info['mean_reward']:+7.2f}  "
            f"v_EV={v.explained_variance:+.3f}  "
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
    # Extract final-step metrics from the per-step dataframe.
    last = df.iloc[-1]
    baseline_metrics[name] = {
        "on_time_rate": float(last.get("on_time_rate", 0.0)),
        "total_fuel_used": float(last.get("total_fuel_used", 0.0)),
        "dock_utilization": float(last.get("dock_utilization", 0.0)),
    }

# ======================================================================
# Comparison table
# ======================================================================
print("\n" + "=" * 70)
print("COMPARISON (Pareto evaluation — no composite score)")
print(f"  {EVAL_STEPS}-step episode, seed={SEED}")
print("=" * 70)

header = f"{'Policy':<16} {'on_time_rate':>12} {'fuel_used':>12} {'dock_util':>12}"
print(header)
print("-" * len(header))

# Print baselines
for name in baselines:
    m = baseline_metrics[name]
    print(
        f"{name:<16} {m['on_time_rate']:>12.3f} {m['total_fuel_used']:>12.1f} {m['dock_utilization']:>12.3f}"
    )

# Print MAPPO
print(
    f"{'mappo':<16} "
    f"{mappo_metrics.get('on_time_rate', 0.0):>12.3f} "
    f"{mappo_metrics.get('total_fuel_used', 0.0):>12.1f} "
    f"{mappo_metrics.get('dock_utilization', 0.0):>12.3f}"
)
print("=" * 70)

# Pareto dominance check against reactive
reactive = baseline_metrics["reactive"]
mappo_otr = mappo_metrics.get("on_time_rate", 0.0)
mappo_fuel = mappo_metrics.get("total_fuel_used", 0.0)
mappo_dock = mappo_metrics.get("dock_utilization", 0.0)

wins = 0
wins += int(mappo_otr > reactive["on_time_rate"])
wins += int(mappo_fuel < reactive["total_fuel_used"])
wins += int(mappo_dock > reactive["dock_utilization"])

if wins == 3:
    print("MAPPO Pareto-dominates reactive baseline on all 3 metrics.")
elif wins == 0:
    print("Reactive baseline Pareto-dominates MAPPO on all 3 metrics.")
else:
    print(f"Mixed result: MAPPO wins on {wins}/3 metrics vs reactive.")
