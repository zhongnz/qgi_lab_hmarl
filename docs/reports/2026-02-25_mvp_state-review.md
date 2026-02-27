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
5. ~~Add curriculum learning / adaptive difficulty scaling.~~ **Done** (`curriculum.py`).
6. ~~Proposal alignment audit.~~ **Done** (dock availability, trip duration, coordinator metrics, cadence defaults).
7. ~~Codebase consistency audit.~~ **Done** (evaluate division bug, seed variation, metric key alignment, per-agent reward accumulation, dt_hours config, logger robustness).
8. ~~Weather effects on fuel/speed.~~ **Done** (`dynamics.py` weather generation, fuel multiplier, speed factor; env integration with per-route sea-state observations).
9. ~~Gymnasium gym.Env wrapper.~~ **Done** (`gym_wrapper.py` wraps MaritimeEnv with standard Box observation/action spaces).
10. ~~Investigate parameter-sharing ablations across agent types.~~ **Done** (ablation framework supports `env_` prefix overrides for environment config).
11. ~~Weather-aware routing experiments (compare policies with/without weather).~~ **Done** (`run_weather_sweep()`, weather-aware policies in coordinator and vessel agents, weather reward shaping, `weather_on`/`weather_harsh` MAPPO ablation variants).
12. ~~Curriculum weather ramp (sea-state, penalty factor, shaping weight, bool enable).~~ **Done** (`curriculum.py` rampable float & bool keys).
13. ~~CLI --weather / --sea-state-max flags for MAPPO.~~ **Done** (shared parent parser in `run_mappo.py`).
14. ~~Gym wrapper weather_matrix exposure.~~ **Done** (`gym_wrapper.py` step() info dict).
15. ~~MAPPO vessel weather speed capping.~~ **Done** (`_vessel_weather_speed_cap()` + `speed_cap` param in `_nn_to_vessel_action()`).
16. ~~Training profiling (rollout_time, update_time, iter_time per iteration).~~ **Done** (`train()` timing instrumentation + `total_train_time`).
17. ~~Early stopping with patience in train().~~ **Done** (`early_stopping_patience` parameter).
18. ~~Multi-seed training runner.~~ **Done** (`train_multi_seed()` in `mappo.py`).
19. ~~Multi-seed learning curve + timing plots.~~ **Done** (`plot_multi_seed_curves()`, `plot_timing_breakdown()` in `plotting.py`).
20. ~~CLI multiseed subcommand.~~ **Done** (`run_mappo.py multiseed`).
21. ~~(New) Run full hyperparameter sweeps and ablation experiments for final report.~~
    Infrastructure complete: YAML experiment config system (`experiment_config.py`),
    `scripts/run_experiment.py`, `configs/` example configs, multi-seed runner.
    Actual sweep execution (E1–E9 from `experiment_protocol.md`) is still outstanding
    pending config scale fix (8v/5p) and compute time.
22. ~~(New) End-to-end weather impact analysis across all policy types.~~
    **Resolved**: `run_weather_sweep()` in `experiment.py` compares all policy
    types under weather on/off conditions. Full multi-seed weather analysis
    (E6 in `experiment_protocol.md`) pending final sweep run.
23. (New) Scale experiment configs from 3v/2p to proposal-specified 8v/5p.
    All YAML configs (`baseline.yaml`, `multi_seed.yaml`, etc.) currently use
    `num_vessels: 3, num_ports: 2`. Must update to `num_vessels: 8, num_ports: 5`
    before running E1–E9 results for the final report.
23. ~~(New) YAML experiment configuration with save/load/run.~~ **Done** (`experiment_config.py`).
24. ~~(New) Statistical evaluation module (Welch's t-test, bootstrap CI, method comparison).~~ **Done** (`stats.py`).
25. ~~(New) Parameter sharing toggle for MAPPO ablation.~~ **Done** (`MAPPOConfig.parameter_sharing`, `build_per_agent_actor_critics()`).
26. ~~(New) Experiment runner script for YAML configs.~~ **Done** (`scripts/run_experiment.py`).
