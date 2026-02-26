# Metrics Dictionary

This file defines the core columns produced by `run_experiment(...)` and the
CSV outputs from `scripts/run_baselines.py`.

## Run Metadata

- `t`: simulation step index.
- `policy`: one of `independent`, `reactive`, `forecast`, `oracle`, `learned_forecast`, `mappo`.
- `forecast_horizon`: short-horizon forecast length used in the run.
- `forecast_noise`: Gaussian noise scale applied to synthetic forecasts.
- `share_forecasts`: `1` if short forecasts are shared beyond coordinator, else `0`.
- `num_coordinators`: number of active coordinator agents.
- `coordinator_updates`: `1` when coordinator cadence triggers at the step.

## Queue / Port State

- `avg_queue`: mean queue length across ports.
- `dock_utilization`: mean `occupied / docks` across ports.
- `total_wait_hours`: cumulative wait-hours across all ports.
- `total_vessels_served`: cumulative served vessels across all ports.
- `avg_wait_per_vessel`: `total_wait_hours / total_vessels_served`.
- `pending_arrival_requests`: requests waiting in port inbox queues.

## Vessel / Emissions State

- `avg_speed`: average vessel speed.
- `avg_fuel_remaining`: average remaining fuel.
- `total_fuel_used`: cumulative fleet fuel consumption from initial fuel levels.
- `total_emissions_co2`: cumulative fleet CO2 emissions.
- `avg_delay_hours`: average vessel delay.
- `on_time_rate`: share of vessels with delay < 2 hours.

## Coordination Counters

- `step_vessel_requests`: slot requests submitted at this step.
- `step_port_accepted`: slot requests accepted at this step.
- `total_vessel_requests`: cumulative slot requests submitted so far.
- `total_port_accepted`: cumulative slot requests accepted so far.
- `policy_agreement_rate`: cumulative `total_port_accepted / total_vessel_requests`.

## Economics

- `fuel_cost_usd`: cumulative fleet fuel cost.
- `delay_cost_usd`: cumulative delay penalty cost.
- `carbon_cost_usd`: cumulative carbon cost.
- `total_ops_cost_usd`: total operating cost (`fuel + delay + carbon`).
- `price_per_vessel_usd`: average ops cost per vessel.
- `cost_reliability`: `1 - total_ops_cost_usd / total_cargo_value`.

## Rewards

- `avg_vessel_reward`: average per-step vessel reward over all vessels.
- `avg_port_reward`: average per-step port reward over all ports.
- `coordinator_reward`: per-step coordinator reward.

## Per-Step Economic Deltas

- `step_fuel_cost_usd`: fuel cost incurred at this step.
- `step_delay_cost_usd`: delay penalty incurred at this step.
- `step_carbon_cost_usd`: carbon cost incurred at this step.
- `step_total_ops_cost_usd`: total ops cost at this step (`fuel + delay + carbon`).

## MAPPO Training Metrics

Produced by `run_mappo_comparison()` under the `_train_log` key and by the
`MAPPOTrainer` during training iterations.

- `iteration`: training iteration index (1-based).
- `mean_reward`: mean episode reward for the iteration.
- `total_reward`: total episode reward for the iteration.
- `vessel_mean_reward`: mean vessel reward per agent per step across the rollout.
- `port_mean_reward`: mean port reward per agent per step across the rollout.
- `coordinator_mean_reward`: mean coordinator reward per agent per step across the rollout.
- `vessel_value_loss`: critic MSE loss for the vessel actor-critic.
- `port_value_loss`: critic MSE loss for the port actor-critic.
- `coordinator_value_loss`: critic MSE loss for the coordinator actor-critic.
- `*_policy_loss`: PPO clipped surrogate loss per agent type.
- `*_entropy`: policy entropy per agent type.
- `*_clip_fraction`: fraction of samples clipped per agent type.
- `*_grad_norm`: gradient norm after clipping per agent type.
- `*_weight_norm`: total parameter L2 norm per agent type.

## MAPPO Evaluation Metrics

Returned by `MAPPOTrainer.evaluate()`.

- `mean_vessel_reward`: average vessel reward per step (over actual steps completed).
- `mean_port_reward`: average port reward per step.
- `mean_coordinator_reward`: average coordinator reward per step.
- `total_reward`: total reward across all agent types and all steps.
- Plus all vessel, port, and economic metrics listed above.

## Coordinator-Level Metrics

Returned by `compute_coordinator_metrics()`.

- `emission_budget_compliance`: fraction of vessels within emission budget.
- `avg_route_efficiency`: average ratio of direct distance to distance travelled.
- `avg_trip_duration_hours`: average trip duration from dispatch to arrival.

## Coordination Metrics

Returned by `compute_coordination_metrics()`.

- `policy_agreement_rate`: fraction of requests accepted vs submitted.
- `communication_overhead`: total messages exchanged.

## Weather Metrics

Available when `weather_enabled=True` in the environment config. Returned in the
step `info` dict.

- `weather_enabled`: boolean flag indicating weather is active.
- `mean_sea_state`: mean sea state across all routes at the current step.
- `max_sea_state`: maximum sea state across all routes at the current step.

### Weather Config Parameters

- `weather_enabled` (bool, default `False`): enable per-route sea-state effects.
- `sea_state_max` (float, default `3.0`): upper bound for uniformly sampled sea state.
- `weather_penalty_factor` (float, default `0.15`): multiplier for fuel increase and speed reduction per unit sea state.

### Weather Physics

- **Fuel multiplier**: `1 + weather_penalty_factor × sea_state` (default worst case: 1.45×).
- **Speed factor**: `1 / (1 + weather_penalty_factor × sea_state)` (effective distance reduced).
- Vessel observations gain one extra dimension (`sea_state`) when weather is enabled.
