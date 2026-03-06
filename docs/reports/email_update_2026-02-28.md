# Email Update — HMARL Maritime MVP Progress

**Date:** March 4, 2026 (week ending February 28, 2026)  
**To:** Prof. Aboussalah  
**From:** [Student Name]  
**Subject:** HMARL Maritime MVP — Transition Kernel, Reward System, Weather Integration & First Demo Run

---

Dear Prof. Aboussalah,

This note summarises recent progress on the HMARL maritime simulator. I completed a full audit of the reward system, expanded in-code documentation, and ran an end-to-end demonstration (heuristic baselines, lightweight MAPPO training, and ablations). The run produced 9 plot PNGs and 21 CSVs stored locally under `runs/demo/` — I can push these to the remote repo or attach them to an email upon request.

Since that run, I also completed a post-demo hardening pass (configuration strictness, weather-aware port heuristics, and demo-script cleanup) to improve reproducibility and reporting consistency.

## 1. Transition Kernel

The environment's `step()` method implements a 6-phase joint transition kernel T(s' | s, a) (numbered 0-5):

| Phase | Name | What happens |
|-------|------|-------------|
| **0** | Message delivery | `bus.deliver_due(t)` fires all messages whose latency has expired — coordinator directives land in vessel mailboxes; vessel arrival requests land in port pending queues; slot accept/reject responses are returned for processing in Phase 2. |
| **1** | Coordinator action | Each coordinator (on its sub-cadence, every 12 steps by default) converts its action into per-vessel directives and enqueues them with a configurable message latency. Vessels will see these only after delivery in a future Phase 0. |
| **2** | Vessel action + slot negotiation | Each vessel decides speed and whether to request a berth slot. Accepted slot responses trigger `dispatch_vessel()` which commits the destination; with `departure_window_hours=0` the vessel goes to sea immediately, otherwise it enters `pending_departure` until the departure window opens. Rejected or still-waiting vessels accumulate `delay_hours`. |
| **3** | Physics tick | `step_vessels()` advances every in-transit vessel: `position_nm += speed × dt × weather_speed_factor(W_t)`. Vessels completing their leg are flipped to arrived and appended to the destination port's queue. |
| **4** | Port action + service tick | Ports allocate berth slots from their pending request backlog using Earliest-Deadline-First (EDF) ordering, then `step_ports()` drains completed berths and admits queued vessels. |
| **5** | Reward + clock advance | Rewards are computed from the state produced by phases 1–4 using weather W_t. Only then does the clock advance (t += 1) and weather updates to W_{t+1} via AR(1). Observations returned therefore reflect W_{t+1}. |

The three agent types operate on independent sub-cadences (coordinator every 12 steps, vessel every 1 step, port every 2 steps), and all inter-agent communication goes through a `MessageBus` with configurable latency — no direct state sharing.

## 2. Weather System

Weather is a symmetric `(num_ports × num_ports)` sea-state matrix where entry `[i,j]` represents conditions on the route from port i to port j. It evolves via an AR(1) process:

```
W_{t+1} = α · W_t + (1 − α) · noise,   noise ~ Uniform(0, sea_state_max)
```

With `autocorrelation=0.7` and `sea_state_max=3.0`, weather is persistent but stochastic. It affects the simulation through two channels:
- **Fuel consumption**: multiplied by `1 + penalty_factor × sea_state` (up to 1.45× at max)
- **Effective speed**: divided by the same factor, reducing distance covered per tick

Weather is exposed to all three agent types through observations:
- **Vessels**: local sea-state for their current route segment
- **Ports**: compact inbound-weather summary (mean, max, rough-route fraction)
- **Coordinator**: full weather matrix

Heuristic policies adjust behaviour accordingly: vessels slow down in rough seas (fuel_multiplier > 1.3 → minimum speed), while heuristic coordinators penalise routes through high sea-state regions when assigning destinations.

## 3. Reward System

The reward structure has three independent per-step signals plus two optional weather-shaping bonuses:

**Vessel reward** (per step):  
`r_V = −(1.0 × fuel + 1.5 × delay + 0.7 × CO₂)`  
Range: 0 (docked, no delay) to approximately −20 (fast transit in rough weather).

**Port reward** (per step):  
`r_P = −(queue × dt_hours + 0.5 × idle_docks)`  
Penalises both waiting-time accumulation and wasted berth capacity.

**Coordinator reward** (per step):  
`r_C = −(fuel_used + avg_queue + 2.0 × CO₂)`  
System-level signal with amplified emission penalty (λ=2.0) to reflect the coordinator's fleet-wide responsibility.

**Weather shaping** (opt-in, additive):  
- Vessel bonus: small reward for slowing in rough weather (when fuel multiplier > 1.1)
- Coordinator bonus: reward proportional to how calm the chosen routes are

All negative inputs are clamped to zero (no reward hacking from negative fuel). Rewards are normalised in MAPPO via Welford running statistics.

## 4. Demo Run Results

I ran a complete demonstration on our 12-core CPU (no GPU) in ~3 minutes:

**Setup:** 5 ports, 8 vessels, 3 docks/port, weather enabled (AR(1), α=0.7), 30-step episodes.

### Heuristic Baselines (4 policies)

| Policy | Avg Queue | Dock Util. | CO₂ (tons) | Ops Cost ($) | Cost Reliability |
|--------|----------|-----------|-----------|-------------|------------|
| Independent | 0.127 | 0.182 | 2,234 | $982,089 | 87.7% |
| Reactive | 0.073 | 0.182 | 2,344 | $939,641 | 88.3% |
| Forecast | 0.073 | 0.182 | 2,960 | $999,696 | 87.5% |
| Oracle | 0.073 | 0.182 | 3,184 | $1,029,989 | 87.1% |

The reactive policy achieves the lowest ops cost. Reactive/forecast/oracle all achieve similarly low queue levels, while the independent policy's higher queue confirms that coordination matters. (The reliability column above is the economic metric `cost_reliability = 1 − total_ops_cost / total_cargo_value`, not the vessel on-time rate.)

### MAPPO Training (30 iterations × 32-step rollouts)

- Mean reward improved from −22.0 to −21.0 (best: −11.4) over 30 iterations
- Vessel value loss decreased from 34.4 → 19.4
- Coordinator value loss decreased from 60.3 → 18.4
- KL divergence stayed well below the 0.02 target (vessel: 0.0000–0.0035, coordinator: < 0.0001)
- Average rollout time: 1.23s, update time: 0.25s per iteration

The training curves show the critic is learning (value losses trending down) and policy updates are stable (KL well below target). With only 30 iterations the policy has not yet converged — this is expected and a longer run would improve.

### Ablation Study (3 variants × 20 iterations)

| Variant | Final Mean Reward | Best Mean Reward |
|---------|------------------|-----------------|
| Full model | −22.1 | −11.4 |
| No weather | −18.2 | −9.6 |
| High entropy | −22.1 | −11.4 |

The no-weather variant shows higher (less negative) rewards because weather penalties are absent — this confirms weather is correctly affecting the reward signal.

## 5. Plots Produced

Nine publication-quality plots were generated:

1. **01_policy_comparison.png** — 2×3 grid comparing 4 heuristic policies across queue, utilization, emissions, fuel, reward, and ops cost
2. **02_horizon_sweep.png** — Forecast horizon ablation (6h vs 12h vs 24h)
3. **03_noise_sweep.png** — Forecast noise sensitivity (0.0 to 2.0)
4. **04_sharing_sweep.png** — Shared vs coordinator-only forecast access
5. **05_training_curves.png** — MAPPO reward + per-agent value losses
6. **06_mappo_comparison.png** — MAPPO evaluation vs reactive + forecast baselines
7. **07_training_dashboard.png** — 2×2 panel: reward, value loss, KL divergence (with 0.02 target line), policy entropy
8. **08_timing_breakdown.png** — Stacked area of rollout vs update time per iteration
9. **09_ablation_bar.png** — Grouped bar chart comparing ablation variants

## 6. Post-Demo Code Hardening (Completed March 4)

- **MAPPO action space alignment with scheduling logic**
  - Vessel actor now outputs both speed and requested arrival time (`requested_arrival_time`), enabling direct learning of EDF-relevant arrival intent.
  - Coordinator actor now selects both destination and departure-window bin (`coordinator_departure_window_options`), replacing a fixed departure-window assumption.

- **Weather-aware port-side decision quality**
  - Port observations now include compact inbound-weather features (mean/max/rough-route fraction), and forecast-mode port heuristics use these features to adjust request acceptance conservatively during rough conditions.

- **Experiment configuration robustness**
  - YAML/JSON experiment loading now supports strict top-level key validation to prevent silent typos in config files.

- **Demo runner consistency improvements**
  - The demo script now reuses a single MAPPO training run for training curves, dashboard, and timing plots (no duplicate retraining).
  - Metadata entries such as `_train_log` are no longer exported as duplicate CSV artifacts in new demo runs.

- **Validation status**
  - Full test suite currently passes (`681` tests), including new regression tests for demo artifact behavior.

## Next Steps

- Scale training to 200+ iterations to see MAPPO convergence with the updated action spaces
- Multi-seed runs for confidence intervals and significance testing
- Longer episodes (100+ steps) for realistic voyage durations
- Hyperparameter sweep over learning rate, entropy schedule, and departure-window options

Please let me know if you'd like me to focus on any particular aspect or run a longer experiment.

Best regards,  
[Student Name]
