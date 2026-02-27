# Experiment Protocol

> Maps each Research Question to concrete experiments, configs, metrics, and acceptance criteria.

---

## Research Questions

| ID  | Question |
|-----|----------|
| RQ1 | How can heterogeneous agents (vessels, ports, coordinator) coordinate using shared congestion forecasts? |
| RQ2 | Does proactive coordination with forecasts improve over independent / reactive baselines? |
| RQ3 | Which forecast horizons and sharing strategies maximize decision quality? |
| RQ4 | How do coordination improvements affect economics (fuel cost, delay penalties, carbon price)? |

---

## Experiment Matrix

### E1 — Baseline Policy Sweep (RQ1, RQ2)

**Goal:** Establish that MAPPO outperforms heuristic and random baselines.

| Property | Value |
|----------|-------|
| Function | `run_mappo_comparison()` / `run_multi_seed_mappo_comparison()` |
| Config   | `configs/baseline.yaml`, `configs/multi_seed.yaml` |
| Baselines | Random, Greedy-Nearest, Round-Robin, Forecast-Aware, MAPPO |
| Seeds    | 5 (42, 49, 56, 63, 70) |
| Iterations | 200 |
| Key Metrics | `total_fuel`, `total_co2`, `mean_delay_hours`, `avg_utilisation`, `mean_vessel_reward`, `mean_port_reward`, `coordinator_reward` |
| Acceptance | MAPPO achieves statistically significant improvement (p < 0.05, Welch's t-test) over at least 2 heuristic baselines on `total_fuel` and `mean_delay_hours` |

**Commands:**
```bash
python scripts/run_experiment.py configs/multi_seed.yaml
python scripts/run_baselines.py
```

---

### E2 — Parameter Sharing Ablation (RQ1)

**Goal:** Quantify benefit of CTDE parameter sharing vs independent per-agent networks.

| Property | Value |
|----------|-------|
| Function | `run_from_config()` on both configs |
| Config   | `configs/multi_seed.yaml` (shared) vs `configs/no_sharing_ablation.yaml` (independent) |
| Seeds    | 3 (42, 49, 56) |
| Iterations | 200 |
| Key Metrics | `mean_vessel_reward`, `total_fuel`, `total_co2`, convergence speed (iteration to reach 90% of final reward) |
| Acceptance | Report effect size (Cohen's d) and 95% bootstrap CI for reward difference |

**Commands:**
```bash
python scripts/run_experiment.py configs/multi_seed.yaml
python scripts/run_experiment.py configs/no_sharing_ablation.yaml
```

---

### E3 — Forecast Horizon Sweep (RQ3)

**Goal:** Identify optimal medium-term and short-term forecast horizons.

| Property | Value |
|----------|-------|
| Function | `run_horizon_sweep()` |
| Horizons | medium: {1, 3, 5, 7, 10} days; short: {4, 8, 12, 24} hours |
| Seeds    | 3 per combination |
| Key Metrics | `total_fuel`, `mean_delay_hours`, `avg_utilisation` |
| Acceptance | Identify horizon pair that minimises fuel+delay, report marginal improvement per extra horizon day |

**Command:**
```python
from hmarl_mvp import run_horizon_sweep
results = run_horizon_sweep(horizons=[1, 3, 5, 7, 10], seeds=3)
```

---

### E4 — Forecast Noise Ablation (RQ3)

**Goal:** Characterise robustness to noisy forecasts.

| Property | Value |
|----------|-------|
| Function | `run_noise_sweep()` |
| Noise levels | {0.0, 0.1, 0.2, 0.5, 1.0} |
| Seeds    | 3 per level |
| Key Metrics | `total_fuel`, `mean_delay_hours` |
| Acceptance | Demonstrate graceful degradation — performance loss < 20% at noise=0.5 vs noise=0.0 |

---

### E5 — Forecast Sharing Ablation (RQ3)

**Goal:** Test which forecast sharing strategy (full, partial, none) works best.

| Property | Value |
|----------|-------|
| Function | `run_sharing_sweep()` |
| Strategies | full, coordinator_only, port_only, none |
| Seeds    | 3 per strategy |
| Key Metrics | `total_fuel`, `mean_delay_hours`, `avg_utilisation` |
| Acceptance | Identify best strategy, report pairwise significance |

---

### E6 — Weather Impact Analysis (RQ2, RQ4)

**Goal:** Measure policy robustness under varying sea-state conditions.

| Property | Value |
|----------|-------|
| Function | `run_weather_sweep()` |
| Config   | `configs/weather_curriculum.yaml` |
| Sea states | {0.0, 1.0, 2.0, 3.0} (max sea state) |
| Autocorrelation | {0.0, 0.5, 0.7} |
| Seeds    | 3 per combination |
| Key Metrics | `total_fuel`, `total_co2`, `mean_delay_hours`, `economic_total_cost` |
| Acceptance | Weather-curriculum-trained policy outperforms non-weather baseline under storm conditions |

**Command:**
```bash
python scripts/run_experiment.py configs/weather_curriculum.yaml
```

---

### E7 — Hyperparameter Sweep (All RQs)

**Goal:** Find best MAPPO hyperparameters for production experiments.

| Property | Value |
|----------|-------|
| Function | `run_mappo_hyperparam_sweep()` |
| Grid     | LR ∈ {1e-4, 3e-4, 1e-3}, entropy ∈ {0.001, 0.01, 0.05}, GAE λ ∈ {0.9, 0.95, 0.99} |
| Seeds    | 3 per combination (81 runs total) |
| Key Metrics | `coordinator_reward`, convergence iteration |
| Acceptance | Top configuration identified; sensitivity analysis reported |

---

### E8 — Economic Analysis (RQ4)

**Goal:** Translate operational metrics into dollar costs using configurable prices.

| Property | Value |
|----------|-------|
| Function | `compute_economic_metrics()` from rewards module |
| Configs  | Default prices: fuel=$600/ton, delay=$5000/hr, carbon=$90/ton |
| Comparison | MAPPO vs best heuristic, per-seed |
| Key Metrics | `fuel_cost_usd`, `delay_cost_usd`, `carbon_cost_usd`, `economic_total_cost` |
| Acceptance | Report total cost savings (%) and 95% CI |

---

### E9 — Forecaster Ablation (RQ3)

**Goal:** Determine whether a trainable forecaster (MLP or GRU) improves
coordination quality over heuristic queue estimates.

| Property | Value |
|----------|-------|
| Function | `run_experiment(policy_type=...)` with each forecaster variant |
| Variants | Heuristic (`forecast`), MLP (`learned_forecast`), GRU (`rnn_forecast`) |
| Seeds    | 3 (42, 49, 56) |
| Iterations | 200 |
| Key Metrics | `total_fuel`, `mean_delay_hours`, forecaster `val_loss` (MAE on held-out traces) |
| Acceptance | MLP and/or GRU achieve statistically significant improvement (p < 0.05, Welch's) over heuristic on `total_fuel` or `mean_delay_hours`; report forecaster validation curves |

**Training commands:**
```bash
# Train MLP forecaster
python scripts/train_forecaster.py --episodes 20 --steps 40 --epochs 200

# Train GRU forecaster
python scripts/train_forecaster.py --model rnn --seq-len 8 --epochs 200
```

**Run ablation:**
```python
from hmarl_mvp import run_forecaster_ablation
results = run_forecaster_ablation(seeds=[42, 49, 56], num_iterations=200)
```

> **Depends on**: E1 complete, YAML configs updated to 8v/5p scale (see Critical Prerequisites).

---

## Statistical Methodology

All multi-seed comparisons use:
- **Welch's t-test** for pairwise significance (α = 0.05)
- **Bootstrap confidence intervals** (10,000 resamples, 95% CI)
- **Cohen's d** for effect size reporting
- Implementation: `hmarl_mvp.stats.compare_methods()` and `multi_method_comparison()`

---

## Critical Prerequisites

> **BEFORE RUNNING ANY EXPERIMENT:** Update all YAML configs from the current
> development scale (`num_vessels: 3, num_ports: 2`) to the proposal-specified
> scale (`num_vessels: 8, num_ports: 5`). Results at 3v/2p are not comparable
> to the research proposal and should not be used in the final report.
>
> Configs to update: `configs/baseline.yaml`, `configs/multi_seed.yaml`,
> `configs/weather_curriculum.yaml`, `configs/no_sharing_ablation.yaml`.

---

## Execution Plan

| Phase | Experiments | Est. Time | Prerequisites |
|-------|-------------|-----------|---------------|
| 0. Scale fix | Update all configs to 8v/5p | 30 min | None |
| 1. Validation | E1 (baseline sweep) | 2 hours | Phase 0 done |
| 2. Ablations  | E2 (param sharing), E4 (noise), E5 (sharing) | 4 hours | E1 done |
| 3. Horizons   | E3 (horizon sweep) | 3 hours | E1 done |
| 4. Weather    | E6 (weather analysis) | 4 hours | E1 done |
| 5. Tuning     | E7 (hyperparam sweep) | 8 hours | E1-E3 analysed |
| 6. Forecaster | E9 (forecaster ablation) | 3 hours | E1, E3 done |
| 7. Economics  | E8 (cost analysis) | 1 hour  | E1, E6 done |

---

## Output Artifacts

All runs save to `runs/<experiment_name>/`:
- `metrics.csv` — per-iteration metrics
- `config.yaml` — frozen experiment config
- `summary.json` — final statistics
- `checkpoints/` — model weights (if enabled)

Statistical comparisons are generated by `scripts/run_experiment.py --compare` and saved as:
- `comparison_table.csv`
- `significance_tests.json`

---

## Reproducibility Checklist

- [x] Fixed seeds per experiment (configurable in YAML)
- [x] `torch.manual_seed()` + `np.random.default_rng()` alignment
- [x] Deterministic evaluation mode (`deterministic=True`)
- [ ] `torch.use_deterministic_algorithms(True)` (to be added)
- [x] Config frozen at run start (`ExperimentConfig.save_yaml()`)
- [x] Git commit hash logged in output
