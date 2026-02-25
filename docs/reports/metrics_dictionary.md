# Metrics Dictionary

This file defines the core columns produced by `run_experiment(...)` and the
CSV outputs from `scripts/run_baselines.py`.

## Run Metadata

- `t`: simulation step index.
- `policy`: one of `independent`, `reactive`, `forecast`, `oracle`.
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
