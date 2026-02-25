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
├── hmarl_mvp/
│   ├── __init__.py
│   ├── config.py
│   ├── agents.py
│   ├── state.py
│   ├── dynamics.py
│   ├── forecasts.py
│   ├── policies.py
│   ├── rewards.py
│   ├── metrics.py
│   ├── env.py
│   ├── experiment.py
│   ├── scheduling.py
│   ├── multi_coordinator.py
│   └── plotting.py
├── scripts/
│   └── run_baselines.py
├── tests/
│   ├── test_smoke.py
│   ├── test_arch_scaffolding.py
│   ├── test_agent_policy_forecaster.py
│   ├── test_config_schema.py
│   └── test_model_correctness.py
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
├── requirements-dev.txt
├── CONTRIBUTING.md
├── docs/
│   ├── README.md
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

### 4) Use notebook for analysis

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
| Feb | MVP simulator, rewards, metrics, baseline runner |
| Mar | Train forecasting models, add heuristic baselines |
| Apr | MAPPO with CTDE |
| May | Full ablation suite, final report |
