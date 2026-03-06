# Friday Presentation Update — HMARL Maritime Scheduling

**Date:** 2026-03-07  
**Audience:** Friday meeting discussion  
**Goal:** present the current HMARL system, clarify the mathematical model and rewards, and interpret the latest experiment figures.

## Slide 1 — Project status

This week’s update covers four points:

- explicit state dynamics for each agent type
- reward audit, especially the emissions term
- clarification that the current forecasters are heuristic
- longer MAPPO training with agent-level learning curves

The system is now in a condition where we can discuss both the model structure and the first coherent experimental trends.

## Slide 2 — Problem framing

We model congestion-aware maritime scheduling with three agent types:

- **fleet coordinator**: strategic routing, departure windows, emission budget
- **vessels**: speed control and arrival-slot requests
- **ports**: berth admission and service-rate decisions

The learning problem is hierarchical and asynchronous. Each level has a different action cadence, and decisions propagate through a message bus with latency.

## Slide 3 — Agent state dynamics

### Vessel

The vessel state evolves through physical transit, fuel burn, and delay accumulation:

- position update:
  `x_(t+1) = x_t + v_t * dt * weather_speed_factor`
- fuel update:
  `F_(t+1) = max(F_t - ΔF_t, 0)`
- emissions update:
  `E_(t+1) = E_t + ΔE_t`
- delay update:
  `h_(t+1) = h_t + dt` when docked and waiting / rejected

Key interpretation:
- weather couples movement and fuel directly
- faster sailing is more expensive because fuel scales cubically with speed
- departure windows delay vessel release without fuel burn, but they can still increase operational delay

### Port

The port state evolves through queueing and service:

- queue update:
  `Q_(t+1) = max(Q_t - served_t + arrivals_t, 0)`
- berth timers count down by `dt`
- cumulative waiting time increases by `Q_t * dt`

Key interpretation:
- the port reward should penalize **waiting-time accumulation**, not only queue count
- this is why the reward uses `queue * dt`

### Coordinator

The coordinator tracks:

- latest destination directive
- latest departure-window directive
- latest emission budget
- fleet cumulative emissions summary

Key interpretation:
- coordinator state is partly persistent and partly derived from the vessel fleet
- the heuristic coordinator currently uses a simple emission-budget rule based on total observed fleet emissions

Reference:
- [state_dynamics.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/architecture/state_dynamics.md)

## Slide 4 — Reward definitions

The implemented rewards are per-step:

- vessel:
  `R_V^(t) = -(alpha * Δfuel + beta * Δdelay + gamma * ΔCO2)`
- port:
  `R_P^(t) = -(queue * dt + idle_dock_penalty)`
- coordinator:
  `R_C^(t) = -(Δfleet_fuel + avg_queue + lambda * Δfleet_CO2)`

Important clarification for Friday:

- the emissions penalty uses **incremental step-level CO2**
- it does **not** use cumulative CO2 directly in the reward
- cumulative emissions remain part of the state and observations

Why this matters:

- if cumulative emissions were used directly in reward at every step, the same historical emissions would be penalized repeatedly
- the current implementation avoids that error and is consistent with a stepwise additive objective

Reference:
- [emissions_reward_interactions.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/docs/architecture/emissions_reward_interactions.md)

## Slide 5 — Forecasting clarification

For the current experiments, the forecast inputs are **heuristic**, not learned:

- `ShortTermForecaster`
- `MediumTermForecaster`
- `OracleForecaster` only as an upper-bound baseline

Interpretation for the meeting:

- the current HMARL results test whether hierarchical coordination helps when agents receive heuristic congestion signals
- they do **not** yet answer whether learned forecasting improves the hierarchy

That distinction should be stated explicitly to avoid overstating the forecasting contribution.

## Slide 6 — Experimental setup for the updated run

Updated training run:

- 8 vessels
- 5 ports
- 3 docks per port
- weather enabled
- AR(1) weather persistence with `autocorrelation = 0.7`
- departure-window options: `{0, 6, 12, 24}` hours
- MAPPO training: 80 iterations
- rollout length: 64 steps

Artifacts:

- [training_curves.png](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/training_curves.png)
- [report.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/report.md)

## Slide 7 — Diagram interpretation: updated training curves

Figure:
- [training_curves.png](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/training_curves.png)

What the figure shows:

- panel 1: total training reward over iterations
- panel 2: vessel, port, and coordinator reward curves
- panel 3: value-loss curves for each agent type

How to interpret it:

- training remains noisy, which is expected in a weather-enabled multi-agent setting with heterogeneous rewards
- even with noise, the run improved from an early average reward of `-56.05` to a late average of `-45.26`
- the best reward reached `-19.46` at iteration `38`
- the **port reward is relatively stable** near `-1.4`, so most volatility comes from vessel and especially coordinator terms
- the **coordinator reward dominates magnitude**, which is expected because it combines fleet fuel, average queue, and amplified fleet emissions
- value-loss curves do not indicate divergence; critic learning remains active through the run

Main message:

- this is not a converged result yet, but it is materially stronger evidence than the earlier short demo run
- the agent-level curves are now visible, so we can discuss which part of the hierarchy is driving reward variability

## Slide 8 — Diagram interpretation: heuristic policy comparison

Figure:
- `runs/demo/01_policy_comparison.png`

Supporting table from `runs/demo/policy_summary.csv`:

- `reactive` has the lowest total operating cost: `$939,640`
- `independent` has the highest average queue: `0.127`
- `forecast`, `reactive`, and `oracle` reduce average queue to `0.073`
- `forecast` and `oracle` incur higher emissions than `reactive`

Interpretation:

- coordination clearly matters for congestion, because the independent baseline has the worst queue outcome
- however, **forecast use is not automatically beneficial** in this current heuristic setup
- the reactive heuristic currently dominates forecast/oracle on cost and emissions

Main message:

- the architecture can reduce congestion through coordination
- but forecast-informed heuristics still need to justify themselves economically

## Slide 9 — Diagram interpretation: horizon sweep

Figure:
- `runs/demo/02_horizon_sweep.png`

Final-step values:

- `12h`: emissions `2960`, total ops cost `$999,696`
- `24h`: emissions `4267`, total ops cost `$1,119,268`
- `6h`: emissions `3279`, total ops cost `$1,058,209`

Interpretation:

- in this heuristic setup, `12h` is the best trade-off among the three tested short horizons
- `24h` appears too aggressive or too noisy for the current rule-based downstream decisions
- `6h` gives the worst vessel and coordinator rewards

Main message:

- the horizon choice matters
- longer is not automatically better

## Slide 10 — Diagram interpretation: noise sweep

Figure:
- `runs/demo/03_noise_sweep.png`

Final-step values:

- noise `0.0`: total ops cost `$941,762`, emissions `2433`
- noise `0.5`: total ops cost `$999,696`, emissions `2960`
- noise `1.0`: total ops cost `$1,079,566`, emissions `3717`
- noise `2.0`: total ops cost `$1,094,836`, emissions `3879`

Interpretation:

- the expected degradation is visible: noisier forecasts drive higher emissions and higher cost
- queue remains low at the end of these short runs, so the main effect here is economic and environmental rather than congestion collapse

Main message:

- forecast quality matters, even when queue outcomes look superficially similar

## Slide 11 — Diagram interpretation: sharing sweep

Figure:
- `runs/demo/04_sharing_sweep.png`

Observed result:

- `shared` and `coordinator_only` are effectively identical in the current heuristic experiment

Interpretation:

- in the current heuristic stack, forecast sharing is not yet the limiting factor
- this suggests the bottleneck is likely in the decision logic itself rather than who receives the forecast

Main message:

- forecast sharing becomes more meaningful once the downstream policies can use it more effectively

## Slide 12 — Diagram interpretation: MAPPO vs heuristic baselines

Figure:
- `runs/demo/06_mappo_comparison.png`

Final-step values:

- MAPPO: emissions `314`, total ops cost `$718,740`
- reactive: emissions `2344`, total ops cost `$939,641`
- forecast: emissions `2960`, total ops cost `$999,696`

Interpretation:

- in the demo comparison, MAPPO is substantially better than the two heuristic baselines on emissions and total cost
- MAPPO also yields a much less negative vessel reward than the heuristic alternatives
- the result is promising, but it must still be presented carefully because the demo training budget was short and single-seed

Main message:

- the learned hierarchical policy shows a strong signal of benefit
- but Friday should frame this as **promising preliminary evidence**, not a finalized benchmark result

## Slide 13 — Diagram interpretation: training dashboard

Figure:
- `runs/demo/07_training_dashboard.png`

Interpretation:

- reward is noisy but not unstable
- KL stays well below the `0.02` threshold, so PPO updates are conservative
- critic losses decrease from their initial values, showing learning signal
- entropy remains positive, so the policies have not collapsed prematurely

Main message:

- training is stable enough to extend
- the next issue is sample efficiency and convergence, not catastrophic instability

## Slide 14 — Diagram interpretation: timing breakdown

Figure:
- `runs/demo/08_timing_breakdown.png`

Updated full-scale run:

- wall-clock time: about `5.9 min` for 80 iterations
- time per iteration: `4.44 s`

Interpretation:

- the current implementation is computationally practical for longer runs on CPU
- that removes a major blocker for scaling beyond the original short demo

Main message:

- longer experiments are feasible now

## Slide 15 — Diagram interpretation: ablation results

Figure:
- `runs/demo/09_ablation_bar.png`

Key numbers:

- `full_model`: final mean reward `-30.45`
- `no_weather`: final mean reward `-25.62`
- `high_entropy`: identical to `full_model` in this short run

Interpretation:

- removing weather improves reward because the environment becomes easier
- that confirms the weather term is active and materially affecting the objective
- the high-entropy variant did not separate from baseline in this short run, which likely means the training horizon was too short to expose a difference

Main message:

- weather is a real source of difficulty and is being captured by the environment and rewards

## Slide 16 — Current takeaways

What I would say clearly in the meeting:

- the agent dynamics and reward structure are now fully specified mathematically
- the emissions term is consistent with the intended stepwise reward design
- the current forecasting inputs are heuristic
- longer training is now in place, and agent-level learning curves are available
- preliminary learned-policy results are encouraging, but they are still not the final experimental benchmark

## Slide 17 — Limitations to state explicitly

- single long run, not yet multi-seed
- reward remains noisy
- current presentation evidence is stronger on feasibility and alignment than on final statistical claims
- forecast-learning contribution is not yet isolated, because the presented forecast inputs are heuristic

## Slide 18 — Proposed closing statement

The correct framing for Friday is:

> We now have a mathematically explicit HMARL environment, aligned reward definitions, a clear statement that current forecasters are heuristic, and longer MAPPO training with agent-level learning curves. The current results support a serious discussion of architecture and experimental direction, while the next step is stronger multi-seed validation rather than reworking the model definition.
