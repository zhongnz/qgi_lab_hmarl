# 2026-02-25: MVP State Review

## Scope

Repository-wide review of simulator implementation, baseline runner, tests, and
documentation alignment.

## Current State

1. Module-first MVP is implemented and runnable (`hmarl_mvp/`, CLI runner, tests).
2. Async messaging and cadence scaffolding are in place.
3. Quality gate is green (`make check`).

## Fixes Applied in This Review

1. `independent` baseline now produces operational traffic:
   - vessels submit slot requests
   - ports accept requests up to local capacity
   - coordinator emits random per-vessel destinations
2. Added regression test for independent baseline activity.
3. Added plotting output tests to cover figure artifact generation.
4. Corrected architecture docs that referenced non-existent modules.
5. Added ADR record for module-first architecture decision.

## Remaining Gaps

1. Forecast models are heuristic; no learned forecaster integrated yet.
2. PPO/MAPPO training loop and CTDE learner stack are not implemented.
3. Multi-coordinator strategy is baseline modulo partitioning only.
4. No scenario suite yet for large fleets and heterogeneous ports.

## Recommended Next Steps

1. Implement learned forecasting module and compare against heuristic/oracle.
2. Define independent/reactive/forecast baseline contracts in docs with clear invariants.
3. Start MAPPO/CTDE training integration behind a separate experiment entrypoint.
4. Expand scenario tests (fleet scale, port heterogeneity, stress latency settings).
