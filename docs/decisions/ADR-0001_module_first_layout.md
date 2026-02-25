# ADR-0001: Module-First Layout for HMARL MVP

## Context

The project started as a notebook-heavy prototype. That made iteration fast but
created friction for reproducibility, testing, and CI.

## Decision

Adopt a module-first codebase:

- keep simulator/runtime logic in `hmarl_mvp/`
- keep experiments in executable scripts (`scripts/run_baselines.py`)
- keep notebook usage analysis-first (visualization and reporting only)
- enforce quality gates with lint (`ruff`), typing (`mypy`), and tests (`pytest`)

## Consequences

Positive:

- deterministic API surface for environment, policies, rewards, and metrics
- testable behavior and cleaner regression control
- easier future integration of learned forecasting and MAPPO/CTDE agents

Tradeoffs:

- slightly higher upfront structure overhead
- duplication risk if notebook logic drifts from package APIs

## Alternatives Considered

1. Keep notebook-centric workflow and only add smoke checks.
2. Migrate directly to full RL framework before modular cleanup.

Both alternatives increase integration/debug cost for the current MVP phase.

## Status

Accepted
