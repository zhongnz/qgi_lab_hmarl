# Meeting 03 Task Plan

This checklist maps Meeting 03 feedback to concrete implementation tasks.

## Task checklist

| Task | Deliverable | Status |
|------|-------------|--------|
| Draw environment relationship | `environment_relationship.md` | Done |
| Clarify emissions interaction and rewards | `emissions_reward_interactions.md` | Done |
| Define forecasting horizons and async communication | `forecasting_async_communication.md` | Done |
| Plan scale-up to multiple coordinators | `multi_coordinator_scaling.md` + `hmarl_mvp/agents.py` / `hmarl_mvp/env.py` scaffolding | Done |
| Diagram each agent input/output | `agent_input_output.md` | Done |
| Make things modular | `hmarl_mvp/` package, script runner, tests | Done |
| Prepare the environment | cadence scaffolding + tests + config knobs | Done |

## Immediate next actions

All three items below were addressed in subsequent development:

1. ~~Add visualization for coordinator assignment and cross-coordinator load.~~
   Deferred — `plotting.py` provides reward/utilisation curves; per-coordinator
   assignment heatmaps remain a future enhancement.
2. ~~Evaluate whether to keep request rejections explicit or allow queued request retries.~~
   Resolved: rejections remain explicit; `MessageBus` tracks pending requests
   and `get_pending_requests_sorted()` enables earliest-deadline-first retry logic.
3. ~~Add scenario tests with larger fleets and port heterogeneity.~~
   Done — `test_scenarios.py` covers large fleets, heterogeneous ports, and
   multi-policy runs.
