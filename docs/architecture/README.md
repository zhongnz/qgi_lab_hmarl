# Architecture Notes

This folder captures design details requested in Meeting 03.

## Documents

1. `environment_relationship.md`
2. `agent_input_output.md` — per-agent I/O schema, forecast indexing, docks-per-port
3. `emissions_reward_interactions.md`
4. `forecasting_async_communication.md`
5. `multi_coordinator_scaling.md`
6. `meeting-03_task-plan.md`
7. `mappo_ctde_training.md` — MAPPO trainer, CTDE, buffers, checkpointing
8. `learned_forecaster.md` — trainable MLP / GRU queue forecaster pipeline
9. `state_dynamics.md` — formal equations for every state variable (vessel position, fuel, emissions, port queue, weather AR(1), coordinator cadence)

## Purpose

These notes bridge meeting feedback and implementation:

- environment relationships and agent dataflow
- emissions interactions across agent rewards
- forecast horizons and asynchronous communication cadence
- scaling from one to multiple coordinators
- modularization and environment preparation checklist

