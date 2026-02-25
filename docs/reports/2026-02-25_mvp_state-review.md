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

1. ~~Forecast models are heuristic; no learned forecaster integrated yet.~~
   **Resolved**: `learned_forecaster.py` provides a trainable MLP forecaster
   with supervised training pipeline, integration tests, and CLI script.
2. ~~PPO/MAPPO training loop and CTDE learner stack are not implemented.~~
   **Resolved**: `mappo.py` implements full `MAPPOTrainer` with CTDE
   (centralised critic, decentralised actors), rollout buffers, PPO clipped
   surrogate, value clipping, reward normalisation, and LR scheduling.
3. Multi-coordinator strategy is baseline modulo partitioning only.
   Shared critic via CTDE in `mappo.py` partially addresses this; further
   coordination strategies remain future work.
4. ~~No scenario suite yet for large fleets and heterogeneous ports.~~
   **Resolved**: `test_scenarios.py` covers large fleets, heterogeneous ports,
   stress latency, multi-seed evaluation, and all-policy scenarios.

## Recommended Next Steps (Updated)

1. ~~Implement learned forecasting module.~~ **Done.**
2. ~~Define baseline contracts in docs with clear invariants.~~ **Done** (architecture docs).
3. ~~Start MAPPO/CTDE training integration.~~ **Done.**
4. ~~Expand scenario tests.~~ **Done.**
5. (New) Add curriculum learning / adaptive difficulty scaling.
6. (New) Investigate parameter-sharing ablations across agent types.
7. (New) Profile training throughput and optimise buffer/network bottlenecks.
