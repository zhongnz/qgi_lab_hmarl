# Meeting 03: Colab Walkthrough and Feedback

## Meeting

- Title: Meeting 3 - Colab Walkthrough (Previous Version)
- Date: 2026-02-15
- Location/Call: QGI Lab
- Note taker: Peter

## Attendees

- Peter
- Quantum Geometric Intelligence (QGI) lab members

## Agenda

1. Walkthrough of the previous Colab notebook version
2. Feedback on modeling clarity and architecture
3. Prioritization of next implementation steps

## Discussion Notes

- Peter presented the previous Colab notebook version end-to-end.
- The lab provided technical feedback focused on architecture clarity, agent interfaces, and scaling readiness.
- The following points are reconstructed from meeting bullets and converted into implementation guidance.

### Reconstructed feedback details (inferred)

- **Draw the relationship for the environment**
  - Add a clear system diagram: Coordinator(s) <-> Vessel agents <-> Port agents <-> Environment state.
  - Show where forecasts enter and where rewards/metrics are computed.

- **Interaction effect for emissions for different agents and rewards**
  - Make emissions coupling explicit across agent rewards.
  - Document how coordinator emission budget affects vessel actions and aggregate system penalty.
  - Separate local vs global emission objectives in reward definitions.

- **Forecasting horizons and asynchronous communication**
  - Clarify which agents use short-term vs medium-term horizons.
  - Model asynchronous update cadence (e.g., coordinator slower cycle, vessels/ports faster cycle).
  - Specify message timing assumptions (publish/consume latency).

- **Scale up to multiple coordinators**
  - Design interfaces so one coordinator is not hardcoded.
  - Support coordinator partitioning by region/fleet and coordination protocol between coordinators.

- **Diagram input and output of each agent**
  - Provide per-agent I/O schemas:
    - coordinator: forecast/state in -> strategic directives out
    - vessel: directives + forecast + vessel state in -> speed/arrival requests out
    - port: queue/dock state + requests + forecast in -> service/dock assignment out

- **Make things modular**
  - Split environment, policies, rewards, forecasts, metrics, and runner into separate modules.
  - Reduce notebook logic duplication; keep notebook primarily for analysis and visualization.

- **Prepare the environment**
  - Define a clean environment API (`reset`, `step`, observations, global state).
  - Add reproducibility controls (seed handling, deterministic configs).
  - Add smoke tests for environment shape and baseline execution.

## Decisions Made

- Prioritize refactoring the project into modular components before major model scaling.
- Add architecture and agent I/O diagrams to improve communication and implementation alignment.

## Action Items

| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Create environment relationship diagram and agent I/O diagram | Peter | Next meeting | Done |
| Document emission interaction effects in reward design notes | Peter | Next meeting | Done |
| Define asynchronous communication schedule and forecast horizon mapping | Peter | Next meeting | Done |
| Draft multi-coordinator design notes and interface changes | Peter | Next meeting | Done |
| Continue modular refactor and environment preparation | Peter | Next meeting | Done |

## Open Questions

- Should multi-coordinator design share a single global critic or use hierarchical critics?
- What communication latency assumptions should be enforced for asynchronous updates?

## Follow-up Artifacts

- `docs/architecture/environment_relationship.md`
- `docs/architecture/agent_input_output.md`
- `docs/architecture/emissions_reward_interactions.md`
- `docs/architecture/forecasting_async_communication.md`
- `docs/architecture/multi_coordinator_scaling.md`
- `docs/architecture/meeting-03_task-plan.md`
