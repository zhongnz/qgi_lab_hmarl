# Emissions and Reward Interaction Effects

This note clarifies how emissions couple agent rewards.

## Reward definitions in current MVP

1. Vessel reward:
   `R_V = -(alpha * fuel + beta * delay + gamma * emissions)`
2. Port reward:
   `R_P = -(queue_wait + dock_idle_penalty)`
3. Coordinator reward:
   `R_C = -(voyage_cost + lambda * total_emissions)`

Code references:

- `hmarl_mvp/rewards.py::compute_vessel_reward`
- `hmarl_mvp/rewards.py::compute_port_reward`
- `hmarl_mvp/rewards.py::compute_coordinator_reward`

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

1. Add explicit penalty for emission budget violation at coordinator level.
2. Add coordination term that rewards vessel-port agreement while penalizing high-emission recovery actions.
3. Separate local and global emission costs to avoid over-penalizing a single layer.
4. Track per-agent emission attribution to support diagnostics and ablation.

