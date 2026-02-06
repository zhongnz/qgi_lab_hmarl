# Hierarchical Multi-Agent Reinforcement Learning for Congestion-Aware Vessel Scheduling

**Supervised by Prof. Aboussalah — Spring 2026 Independent Study**

## Overview

This project develops a hierarchical multi-agent reinforcement learning (MARL) framework for maritime vessel scheduling with predictive port coordination. The system models three agent types—a fleet coordinator, vessel agents, and port agents—that coordinate using shared congestion forecasts to minimize fuel consumption, delays, and emissions.

## Repository Structure

```
├── colab_mvp_hmarl_maritime.ipynb   # MVP notebook (fully executable)
├── README.md
├── .gitignore
└── LICENSE
```

## MVP Notebook

The Colab notebook contains a complete, executable prototype:

| Section | Contents |
|---------|----------|
| §1 | Project framing, objectives, research questions (RQ1–RQ4) |
| §2 | Architecture and MDP formulation |
| §3 | Configuration (physics, economics, fleet topology) |
| §4 | Simulator: vessel movement, port operations, emissions, Gymnasium env |
| §5 | Forecasting module (medium-term and short-term, mock) |
| §6 | Agent decision stubs (coordinator, vessel, port policies) |
| §7 | Reward functions (R_C, R_V, R_P) |
| §8 | Metrics: vessel, port, coordination, economic |
| §9 | Experiment protocol, ablation runner |
| §10 | Execution: baseline comparison, ablation visualizations |
| §11–12 | Checklist and next steps |

## Research Questions

1. **RQ1**: How can heterogeneous agents coordinate using shared congestion forecasts?
2. **RQ2**: Does proactive coordination with forecasts improve over independent/reactive baselines?
3. **RQ3**: Which forecast horizons and sharing strategies maximize decision-making quality?
4. **RQ4**: How do coordination improvements affect economic outcomes (price, reliability)?

## Quick Start

Open the notebook in Google Colab or Jupyter:

```bash
# Clone the repo
git clone https://github.com/zhongnz/qgi_lab_hmarl.git
cd qgi_lab_hmarl

# Run in Jupyter
jupyter notebook colab_mvp_hmarl_maritime.ipynb
```

No external dependencies beyond NumPy, Pandas, and Matplotlib.

## Timeline

| Month | Milestone |
|-------|-----------|
| Feb | MVP notebook with simulator, rewards, metrics, ablation framework |
| Mar | Trained forecasting models, heuristic baselines |
| Apr | MAPPO training with centralized critic (CTDE) |
| May | Full ablation studies, final report |
