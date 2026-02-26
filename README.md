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
│   ├── forecasts.py    # Medium-term, short-term, oracle forecasters
│   ├── message_bus.py  # Asynchronous inter-agent message queues
│   ├── policies.py     # Heuristic policy baselines (independent/reactive/forecast/oracle)
│   ├── networks.py     # Actor-critic neural networks (MAPPO/CTDE)
│   ├── buffer.py       # Rollout buffer for on-policy RL training
│   ├── mappo.py        # MAPPO trainer (PPO + CTDE multi-agent training)
│   ├── experiment.py   # Experiment runner, sweeps, multi-seed eval
│   ├── experiment_config.py # YAML experiment config + TensorBoard + runner
│   ├── stats.py        # Statistical evaluation (Welch t-test, bootstrap CI)
│   ├── plotting.py     # Matplotlib plot helpers
│   ├── analysis.py     # Post-processing: comparisons, ranking, ablation deltas
│   ├── report.py       # Markdown report generators
│   ├── logger.py       # Structured JSONL training logger
│   ├── checkpointing.py # Training checkpoints and early stopping
│   ├── curriculum.py   # Curriculum learning scheduler
│   ├── learned_forecaster.py  # Trainable MLP queue forecaster
│   └── gym_wrapper.py  # Gymnasium-compatible single-agent wrapper
├── configs/
│   ├── baseline.yaml         # Standard MAPPO baseline experiment
│   ├── multi_seed.yaml       # 5-seed statistical evaluation
│   ├── weather_curriculum.yaml # Weather curriculum progressive training
│   └── no_sharing_ablation.yaml # Per-agent (no sharing) ablation
├── scripts/
│   ├── run_baselines.py      # CLI: run heuristic baseline experiments
│   ├── run_experiment.py     # CLI: run experiments from YAML configs
│   ├── run_mappo.py          # CLI: MAPPO compare / sweep / ablate / train
│   ├── train_mappo.py        # CLI: standalone MAPPO training with checkpoints
│   └── train_forecaster.py   # CLI: train the learned forecaster
├── tests/                    # 666 tests (pytest)
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
│   ├── test_learned_forecaster.py
│   ├── test_learned_forecast_integration.py
│   ├── test_training_infra.py
│   ├── test_training_pipeline.py
│   ├── test_training_quality.py
│   ├── test_new_modules.py
│   ├── test_new_features.py
│   ├── test_sweep_ablation.py
│   ├── test_report_plotting.py
│   ├── test_plotting.py
│   ├── test_eval_metrics.py
│   ├── test_proposal_alignment.py
│   ├── test_audit_fixes.py
│   ├── test_coverage_gaps_v2.py
│   ├── test_weather_gym.py
│   ├── test_weather_policy_rewards.py
│   ├── test_weather_integration.py
│   ├── test_profiling_multiseed.py
│   ├── test_experiment_config.py
│   ├── test_stats.py
│   └── test_parameter_sharing.py
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
python scripts/train_mappo.py --iterations 50 --rollout-length 64
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

### 7) Multi-seed evaluation

```python
from hmarl_mvp import run_multi_seed_policy_sweep, summarize_multi_seed
df = run_multi_seed_policy_sweep(seeds=[42, 123, 256, 512, 1024], steps=20)
summary = summarize_multi_seed(df)
```

### 8) Use notebook for analysis

Use `colab_mvp_hmarl_maritime.ipynb` for presentation and visual inspection. Prefer module imports for any new logic.

## Configuration

Project config is now validated through a typed schema (`HMARLConfig`) in
`hmarl_mvp/config.py`. Use:

- `get_default_config(...)` for validated overrides
- `validate_config(...)` for validating arbitrary mappings

## Research Questions

1. RQ1: How can heterogeneous agents coordinate using shared congestion forecasts?
2. RQ2: Does proactive coordination with forecasts improve over independent/reactive baselines?
3. RQ3: Which forecast horizons and sharing strategies maximize decision quality?
4. RQ4: How do coordination improvements affect economics (price/reliability)?

## Timeline

| Month | Milestone |
|-------|-----------|
| Feb | ✅ MVP simulator, rewards, metrics, baseline runner, module-first refactor |
| Mar | ✅ Trained forecasting models, heuristic baselines, RL infrastructure (MAPPO/CTDE), curriculum learning || Mar | ✅ Proposal alignment audit: dock availability obs, trip duration metrics, coordinator metrics, decision cadence fixes |
| Mar | ✅ Codebase audit: evaluate() early-termination fix, seed variation, metric key consistency, per-agent reward breakdown, dt_hours config, logger robustness |
| Mar | ✅ Weather effects (sea-state fuel/speed penalties), Gymnasium gym.Env wrapper, coverage gap tests |
| Apr | Tune hyperparameters, run ablation experiments, multi-seed evaluation |
| May | Full ablation suite, final report |
