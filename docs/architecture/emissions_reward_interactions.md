# Emissions and Reward Interaction Effects

This note clarifies how emissions couple agent rewards.

## Reward definitions in current MVP

1. Vessel reward:
   `R_V = -(alpha * fuel + beta * delay + gamma * emissions)`
   (default weights: alpha=1.0, beta=1.5, gamma=0.7)
2. Port reward:
   `R_P = -(queue * dt_hours + dock_idle_weight * idle_docks)`
   Queue penalty is time-weighted (queue-length × time-step) to measure
   waiting-time accumulation per proposal §4.2.
3. Coordinator reward:
   `R_C = -(voyage_cost + lambda * total_emissions)`
   (default lambda=2.0, amplifies CO2 signal at the strategic level)

Code references:

- `hmarl_mvp/rewards.py::compute_vessel_reward_step`
- `hmarl_mvp/rewards.py::compute_port_reward`
- `hmarl_mvp/rewards.py::compute_coordinator_reward_step`

## Interaction effect summary

1. Vessel speed increase:
   - increases fuel burn
   - increases vessel emissions
   - indirectly raises coordinator emission penalty
2. Port congestion:
   - can delay vessels
   - delayed arrivals may trigger speed-up behavior in learned policies
   - speed-up increases fleet emissions
3. Coordinator budget pressure:
   - should bias route/schedule decisions away from high congestion routes
   - should indirectly reduce vessel emission-heavy action regimes

## Recommended next reward refinements

1. ~~Add explicit penalty for emission budget violation at coordinator level.~~
   **Done** — `compute_coordinator_metrics()` tracks `emission_budget_compliance`.
2. Add coordination term that rewards vessel-port agreement while penalizing high-emission recovery actions.
3. Separate local and global emission costs to avoid over-penalizing a single layer.
4. ~~Track per-agent emission attribution to support diagnostics and ablation.~~
   **Partially done** — `avg_route_efficiency` and `avg_trip_duration_hours` in
   coordinator metrics provide per-vessel attribution proxies.

## Weather effects on emissions (added Mar 2026)

When `weather_enabled=True`, sea-state conditions affect fuel consumption and
effective vessel speed on each route:

- **Fuel multiplier**: `1 + weather_penalty_factor × sea_state` — increases
  fuel burn (and proportionally CO2) in rough seas.
- **Speed factor**: `1 / (1 + weather_penalty_factor × sea_state)` — reduces
  distance covered per tick, extending voyage time.

This creates a new strategic dimension: the coordinator must weigh routing
through calm vs. rough sea lanes. Vessels experience higher emissions and slower
progress in bad weather, which feeds back into all three reward functions.

### Weather-Aware Reward Shaping (added)

Two optional additive shaping terms encourage weather-efficient behaviour:

- **Vessel shaping** (`weather_vessel_shaping`): Positive bonus when a vessel
  reduces speed in rough seas (fuel_multiplier > 1.1 and speed ≤ nominal).
  `bonus = weather_shaping_weight × (fuel_mult - 1.0)`.
- **Coordinator shaping** (`weather_coordinator_shaping`): Positive bonus when
  the fleet is routed through calmer seas.
  `bonus = weather_shaping_weight × (1 - normalised_mean_route_sea)`.
- Default `weather_shaping_weight = 0.3`. Both bonuses are zero when weather
  is disabled or sea conditions are calm.

Code references:

- `hmarl_mvp/rewards.py::weather_vessel_shaping`
- `hmarl_mvp/rewards.py::weather_coordinator_shaping`

### Weather-Aware Heuristic Policies (added)

In `forecast` mode, heuristic policies now condition on weather:

- **Coordinator**: Port scores augmented with sea-state penalty — avoids
  routing to ports reachable only through rough seas.
- **Vessel**: Speed reduced in rough seas (fuel_mult > 1.3 → speed_min;
  fuel_mult > 1.1 → capped at nominal) to save fuel.
- `independent` and `reactive` modes remain weather-agnostic.
