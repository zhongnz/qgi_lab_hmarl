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
│   ├── state.py
│   ├── dynamics.py
│   ├── forecasts.py
│   ├── policies.py
│   ├── rewards.py
│   ├── metrics.py
│   ├── env.py
│   ├── experiment.py
│   └── plotting.py
├── scripts/
│   └── run_baselines.py
├── tests/
│   └── test_smoke.py
└── colab_mvp_hmarl_maritime.ipynb
```

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

### 3) Use notebook for analysis

Use `colab_mvp_hmarl_maritime.ipynb` for presentation and visual inspection. Prefer module imports for any new logic.

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
