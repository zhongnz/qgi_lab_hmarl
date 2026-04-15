# Hierarchical Multi-Agent Reinforcement Learning for Congestion-Aware Vessel Scheduling

**Supervised by Prof. Aboussalah - Spring 2026 Independent Study**

## Overview

This project studies hierarchical multi-agent reinforcement learning (MARL) for congestion-aware maritime scheduling. The system models three agent types:

1. Fleet coordinator (strategic guidance)
2. Vessel agents (speed and arrival decisions)
3. Port agents (dock/service decisions)

Coordination is forecast-informed and evaluated on congestion, fuel, emissions, delay, and economic cost metrics.

## Architecture

The codebase now follows a module-first layout. The notebook remains for exploration and visualization.

```
.
├── hmarl_mvp/          # Core simulator package
│   ├── __init__.py     # Public API re-exports
│   ├── config.py       # Typed config, validation, decision cadence
│   ├── state.py        # Port/vessel state dataclasses, initializers
│   ├── agents.py       # Agent wrappers, vessel-coordinator assignment
│   ├── dynamics.py     # Physics: fuel, emissions, vessel/port ticks
│   ├── env.py          # Gym-style multi-agent environment
│   ├── rewards.py      # Reward functions for all agent types
│   ├── metrics.py      # Operational, forecast, and economic metrics
│   ├── forecasts.py    # Medium-term, short-term, noiseless forecasters
│   ├── message_bus.py  # Asynchronous inter-agent message queues
│   ├── policies.py     # Heuristic policy baselines (independent/reactive/forecast/noiseless)
│   ├── networks.py     # Actor-critic neural networks (MAPPO/CTDE)
│   ├── buffer.py       # Rollout buffer for on-policy RL training
│   ├── mappo.py        # MAPPO trainer (PPO + CTDE multi-agent training)
│   ├── experiment.py   # Experiment runner, sweeps, multi-seed eval
│   ├── experiment_config.py # YAML experiment config + TensorBoard + runner
│   ├── stats.py        # Statistical evaluation (Welch t-test, bootstrap CI)
│   ├── plotting.py     # Matplotlib plot helpers
│   ├── report.py       # Markdown report generators
│   ├── logger.py       # Structured JSONL training logger
│   ├── checkpointing.py # Training checkpoints and early stopping
│   ├── curriculum.py   # Curriculum learning scheduler
│   ├── learned_forecaster.py  # Trainable MLP/GRU queue forecaster
│   └── gym_wrapper.py  # Gymnasium-compatible single-agent wrapper
├── configs/
│   ├── baseline.yaml         # Standard MAPPO baseline experiment
│   ├── multi_seed.yaml       # 5-seed statistical evaluation
│   ├── weather_curriculum.yaml # Weather curriculum progressive training
│   ├── no_sharing_ablation.yaml # Per-agent (no sharing) ablation
│   └── production.yaml       # Full-scale production run (PBT-tuned + all arch improvements)
├── scripts/
│   ├── run_baselines.py      # CLI: run heuristic baseline experiments
│   ├── run_experiment.py     # CLI: run experiments from YAML configs
│   ├── run_mappo.py          # CLI: MAPPO compare / sweep / ablate / train
│   ├── train_forecaster.py   # CLI: train the learned forecaster
│   ├── generate_paper_figures.py  # CLI: generate publication-ready figures
│   └── run_production.py     # CLI: full-scale production run (5 seeds × 500 iters)
├── tests/                    # 853 tests (pytest)
│   ├── test_smoke.py
│   ├── test_components.py
│   ├── test_config_schema.py
│   ├── test_state.py
│   ├── test_message_bus.py
│   ├── test_rewards_metrics.py
│   ├── test_coverage_gaps.py
│   ├── test_model_correctness.py
│   ├── test_networks.py
│   ├── test_buffer.py
│   ├── test_mappo.py
│   ├── test_mappo_advanced.py
│   ├── test_action_masking.py
│   ├── test_scenarios.py
│   ├── test_run_demo.py
│   ├── test_learned_forecaster.py
│   ├── test_learned_forecast_integration.py
│   ├── test_training_infra.py
│   ├── test_training_pipeline.py
│   ├── test_training_quality.py
│   ├── test_new_modules.py
│   ├── test_sweep_ablation.py
│   ├── test_report_plotting.py
│   ├── test_research_robustness.py
│   ├── test_plotting.py
│   ├── test_eval_metrics.py
│   ├── test_proposal_alignment.py
│   ├── test_audit_fixes.py
│   ├── test_weather_ar1_and_coord_mask.py
│   ├── test_weather_gym.py
│   ├── test_weather_policy_rewards.py
│   ├── test_weather_integration.py
│   ├── test_profiling_multiseed.py
│   ├── test_experiment_config.py
│   ├── test_stats.py
│   ├── test_parameter_sharing.py
│   ├── test_visualization.py
│   ├── test_improvements.py
│   ├── test_pbt.py
│   └── test_generate_paper_figures.py
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── CONTRIBUTING.md
├── docs/
│   ├── README.md
│   ├── architecture/
│   ├── meetings/
│   ├── decisions/
│   ├── reports/
│   └── templates/
└── colab_mvp_hmarl_maritime.ipynb
```

A full-experiment Colab notebook (`colab_full_experiment_hmarl_maritime.ipynb`) is also provided for GPU-accelerated training.

## Documentation

Project docs live under `docs/`.

- `docs/meetings/`: meeting notes and minutes
- `docs/decisions/`: architecture/project decisions (ADR-style)
- `docs/architecture/`: design diagrams and technical task plans
- `docs/reports/`: experiment writeups and summaries
- `docs/templates/`: reusable documentation templates

Start with:

- `docs/README.md`
- `docs/templates/meeting_minutes_template.md`
- `docs/architecture/meeting-03_task-plan.md`
- `docs/architecture/state_dynamics.md` — formal physics and state update equations for all agent types
- `docs/reports/metrics_dictionary.md`

## Why This Refactor

Notebooks are great for prototyping but weak for reproducibility and testing. This refactor separates concerns:

1. Core logic in importable modules
2. Reproducible experiments in CLI scripts
3. Lightweight analysis and plots in notebooks

## Quick Start

### 1) Run baseline experiments from terminal

```bash
cd qgi_lab_hmarl
python -m pip install -r requirements.txt
python scripts/run_baselines.py --output-dir runs/baseline_refactor
```

This writes per-policy CSVs, ablation CSVs, a summary CSV, and plot PNGs.
If your default `python` is not the project env, run with:
`../.conda/bin/python scripts/run_baselines.py --output-dir runs/baseline_refactor`.

### 2) Run smoke tests

```bash
cd qgi_lab_hmarl
python -m unittest discover -s tests -p "test_*.py"
```

Or run the repo-standard command:

```bash
cd qgi_lab_hmarl
make test
```

### 3) Run quality checks (lint + type + tests)

```bash
cd qgi_lab_hmarl
make install-dev
make check
```

### 4) Train learned forecaster

```bash
cd qgi_lab_hmarl
python scripts/train_forecaster.py --episodes 20 --steps 40 --epochs 200 --verbose
```

This collects queue traces from heuristic rollouts, trains an MLP forecaster,
and writes model weights + evaluation metrics to `runs/forecaster/`.

### 5) Train MAPPO (multi-agent PPO with CTDE)

```bash
cd qgi_lab_hmarl
python scripts/run_mappo.py train --iterations 50 --rollout-length 64
```

This runs the full MAPPO training loop: collecting rollouts with neural-network
policies, computing GAE advantages, and performing PPO clipped updates. Outputs
model checkpoints and reward curves to `outputs/mappo/`.

### 6) Run experiments from YAML configs

```bash
cd qgi_lab_hmarl
# Single experiment:
python scripts/run_experiment.py configs/baseline.yaml

# Compare two experiments:
python scripts/run_experiment.py configs/baseline.yaml configs/no_sharing_ablation.yaml --compare

# Smoke test (2 iterations):
python scripts/run_experiment.py configs/baseline.yaml --smoke
```

Experiment configs specify environment, MAPPO hyper-parameters, curriculum
stages, seed counts, and output paths in a single reproducible YAML file.
The shipped YAML configs now target the current 8-vessel / 5-port simulator
scale with `rollout_steps: 69` and `rollout_length: 64`.

### 7) Run full-scale production experiment

```bash
cd qgi_lab_hmarl
python scripts/run_production.py
```

Runs 5 seeds × 500 iterations with PBT-tuned hyperparameters and all three
architectural improvements (attention coordinator, encoded critic, recurrent
vessels). Generates Figures 15–16 and a comprehensive summary JSON.
Best result: **−15.52** (seed 63), mean last-20: **−22.47 ± 2.35**.

### 8) Run MAPPO comparison vs baselines

```python
from hmarl_mvp.experiment import run_mappo_comparison
results = run_mappo_comparison(train_iterations=50, rollout_length=64, eval_steps=20)
# results contains per-policy DataFrames + training log
```

### 9) Use notebook for analysis

Use `colab_mvp_hmarl_maritime.ipynb` for presentation and visual inspection. Prefer module imports for any new logic.

## Configuration

Project config is now validated through a typed schema (`HMARLConfig`) in
`hmarl_mvp/config.py`. Use:

- `get_default_config(...)` for validated overrides
- `validate_config(...)` for validating arbitrary mappings

## Research Questions

1. **RQ1 (Coordination effectiveness)**: Can hierarchical MARL with shared congestion forecasts learn vessel-scheduling policies that reduce system-wide operational costs relative to rule-based heuristic coordination?
2. **RQ2 (Value of predictive information)**: To what extent does forecast quality affect coordination performance, and how does forecast-induced herding limit the benefit of better predictions?
3. **RQ3 (Parameter sharing)**: Does sharing actor-critic parameters across homogeneous agents improve sample efficiency and asymptotic performance compared to per-agent networks?
4. **RQ4 (Economic implications)**: What are the operational cost differentials (fuel, delay penalties, carbon cost) between MAPPO-trained and heuristic scheduling policies?

## Timeline

| Month | Milestone |
|-------|-----------|
| Feb | ✅ MVP simulator, rewards, metrics, baseline runner, module-first refactor |
| Feb | ✅ Trained forecasting models, heuristic baselines, RL infrastructure (MAPPO/CTDE), curriculum learning |
| Feb | ✅ Proposal alignment audit: dock availability obs, trip duration metrics, coordinator metrics, decision cadence fixes |
| Feb | ✅ Codebase audit: evaluate() early-termination fix, seed variation, metric key consistency, per-agent reward breakdown, dt_hours config, logger robustness |
| Feb | ✅ Weather effects (sea-state fuel/speed penalties), Gymnasium gym.Env wrapper, coverage gap tests |
| Mar | Tune hyperparameters, run ablation experiments, multi-seed evaluation |
| Apr | Full ablation suite, final report |
