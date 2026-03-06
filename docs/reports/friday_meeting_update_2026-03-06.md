# Friday Meeting Update — HMARL Maritime Scheduling

**Date:** 2026-03-06  
**Purpose:** self-contained technical briefing for the Friday discussion.  
**What this document covers:** model structure, transition kernel, time-evolving variables for each agent, reward definitions and verification, forecasting assumptions, experiment setup, latest results, figure interpretation, and discussion points.

## 1. Executive summary

The current HMARL maritime system has three interacting decision layers:

- a **fleet coordinator** that issues destination, departure-window, and emission-budget directives
- **vessel agents** that choose sailing speed and submit arrival-slot requests
- **port agents** that decide how many requests to accept and how many vessels to admit to service

The environment is now documented well enough to discuss it as a complete dynamical system rather than as a set of disconnected modules.

The most important technical clarifications for Friday are:

1. The state dynamics of vessels, ports, weather, and coordinator are now written explicitly as update equations.
2. The environment transition kernel is phase-ordered and deterministic given actions, state, RNG draws, and weather.
3. The reward implementation uses **step-level physical quantities** for vessel and coordinator fuel / emissions terms, not cumulative emissions directly.
4. The forecasters used in the current Friday experiments are **heuristic forecasters**, not learned predictors.
5. A longer full-scale MAPPO run has been completed, and the training plot now includes **per-agent reward curves**.

The latest full-scale run used:

- 8 vessels
- 5 ports
- 3 docks per port
- weather enabled with AR(1) persistence
- 80 MAPPO iterations
- 64-step rollouts

Main quantitative outcomes from that run:

- best training mean reward: `-19.46` at iteration `38`
- final training mean reward: `-46.74`
- final per-agent rewards:
  - vessel: `-2.75`
  - port: `-1.42`
  - coordinator: `-43.99`
- early-average vs late-average training reward:
  - first 8 iterations: `-56.05`
  - last 8 iterations: `-45.26`
  - improvement: `+10.79`
- 5-episode evaluation mean total reward: `-3439.16`
- 5-episode evaluation mean emissions: `1358.77` tons CO2
- 5-episode evaluation mean total operating cost: `$1.031M`
- 5-episode evaluation mean delay: `16.18` hours per vessel

Main artifact files:

- [training_curves.png](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/training_curves.png)
- [report.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/report.md)
- [train_history.csv](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/train_history.csv)
- [eval_result.json](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/eval_result.json)

## 2. System overview

The system is a hierarchical multi-agent reinforcement learning environment for congestion-aware maritime scheduling.

The three decision layers are:

- **Fleet coordinator**
  - operates on the slowest cadence
  - sees fleet-level state and medium-horizon forecast information
  - sends strategic directives to vessels
- **Vessels**
  - operate every step by default
  - see local state, assigned directive, destination-side forecast, and local route weather
  - choose speed and arrival-slot request timing
- **Ports**
  - operate on an intermediate cadence
  - see queue, berth occupancy, pending requests, own-port forecast, and optional inbound weather summary
  - choose acceptance level and service rate

The environment is not synchronous in the trivial sense. There are three distinct sources of asynchrony:

- different decision cadences across agent types
- message-bus latency between sender and receiver
- physical delay between a vessel's departure and eventual arrival at a port

## 3. Notation and default constants

Throughout this note:

- `dt` = simulation step size in hours, default `1.0`
- `c` = fuel-rate coefficient, default `0.002`
- `e` = emissions factor in tons CO2 per ton fuel, default `3.114`
- `p` = weather penalty factor, default `0.15`
- `d_ij` = nautical distance from port `i` to port `j`
- `s_ij^(t)` = sea state on route `i -> j` at time `t`
- `mu(s)` = fuel multiplier due to weather
- `f_w(s)` = effective speed multiplier due to weather

The weather functions are:

`mu(s) = 1 + p * s`

`f_w(s) = 1 / (1 + p * s)`

Interpretation:

- larger sea state increases fuel burn
- larger sea state reduces effective progress per hour
- weather therefore affects both cost and travel time in a coupled way

## 4. Canonical transition kernel

The environment step should be understood as a six-phase transition kernel.

Let the full environment state at time `t` be `S_t`, and let the joint action be `A_t`. Then the environment computes `S_(t+1)` in the following order.

### Phase 0 — Message delivery

All messages whose delivery time has arrived are released from the message bus.

Delivered message types are:

- coordinator directives to vessels
- vessel arrival requests to ports
- port slot accept / reject responses to vessels

Interpretation:

- no agent reads another agent's action directly
- every inter-agent effect is mediated by the bus
- latency is therefore part of the environment dynamics, not just an implementation detail

### Phase 1 — Coordinator action

If the coordinator is due at step `t`, it converts its raw action into per-vessel directives and enqueues them onto the bus.

A directive may contain:

- destination port
- per-vessel destination override
- departure-window hours
- emission budget

These directives are **not** visible to vessels immediately unless the bus latency and cadence make them visible on a later phase-0 delivery.

### Phase 2 — Vessel action and slot negotiation

Each vessel:

- reads the latest available directive
- chooses speed and whether to request an arrival slot
- receives any delivered slot response from Phase 0

If the slot is accepted:

- `dispatch_vessel()` commits destination and speed
- if `departure_window_hours = 0`, the vessel goes to sea immediately
- otherwise the vessel enters pending-departure state until the departure time opens

If the slot is rejected, or the vessel is still waiting on a response:

- the vessel accumulates delay

### Phase 3 — Physics tick

All vessels currently at sea advance physically.

The main operations are:

- compute effective advance using speed and weather
- update distance travelled on current leg
- compute fuel used this step
- compute CO2 emitted this step
- deduct fuel and add cumulative emissions
- if leg distance is completed, mark arrival and place the vessel into the destination-port queue

This is the phase where physical motion and physical resource usage occur.

### Phase 4 — Port action and service tick

If a port is due to act, it decides:

- how many pending requests to accept
- how many queued vessels to admit into service

Then the port-service dynamics advance:

- existing service timers count down
- completed services free berths
- queued vessels are admitted subject to berth availability and service rate
- cumulative waiting time is incremented by queue length times `dt`

### Phase 5 — Reward computation and clock advance

Rewards are computed from the state produced by phases 1–4, using the weather active during that same step.

Only after rewards are computed does the environment:

- increment `t`
- update weather to the next weather state
- rebuild forecasts and observations

This ordering matters. In particular:

- reward at time `t` depends on `W_t`
- returned observations are for the next decision point and therefore reflect post-update weather / forecast state

### Why this kernel is now tidy and clear

Yes, the transition kernel is now reasonably tidy and clear in code and in documentation.

Reasons:

- the execution order is explicit and stable
- message passing, physical updates, port service, and reward computation are separated by phase
- the environment docstring in `hmarl_mvp/env.py` already explains the order
- this document now restates that logic in a standalone form, so readers no longer need the old email note to understand the kernel

## 5. Time-evolving variables by agent type

## 5.1 Vessel state and dynamics

For vessel `k`, define the time-evolving state as:

`S_v,k^(t) = (x_k^(t), v_k^(t), F_k^(t), E_k^(t), ell_k^(t), d_k^(t), h_k^(t), p_k^(t), Tdep_k^(t), asea_k^(t))`

where:

- `x_k^(t)` = distance travelled on the current leg
- `v_k^(t)` = current commanded speed
- `F_k^(t)` = remaining fuel
- `E_k^(t)` = cumulative CO2 emissions
- `ell_k^(t)` = current location port index
- `d_k^(t)` = current committed destination port index
- `h_k^(t)` = accumulated delay hours
- `p_k^(t)` = pending-departure indicator
- `Tdep_k^(t)` = scheduled departure step if pending departure is active
- `asea_k^(t)` = whether the vessel is currently at sea

### Transit update

If `asea_k^(t) = 1`, the physical position update is:

`x_k^(t+1) = x_k^(t) + v_k^(t) * dt * f_w(s_(ell_k,d_k)^(t))`

Interpretation:

- the vessel advances at its commanded speed only after scaling by weather
- in calm weather, `f_w = 1`
- in rough weather, effective progress per hour decreases

### Fuel update

The step fuel burn is:

`DeltaF_k^(t) = c * (v_k^(t))^3 * dt * mu(s_(ell_k,d_k)^(t))`

Then:

`F_k^(t+1) = max(F_k^(t) - DeltaF_k^(t), 0)`

Interpretation:

- the cubic speed term is the main source of economic pressure against excessive speed
- weather worsens the same speed choice by multiplying burn upward

### Emissions update

The step emissions are:

`DeltaE_k^(t) = e * DeltaF_k^(t)`

Then:

`E_k^(t+1) = E_k^(t) + DeltaE_k^(t)`

Interpretation:

- emissions are a derived physical quantity from fuel burn
- there is no separate emissions dynamics beyond fuel burn times emission factor

### Arrival event

If `x_k^(t+1) >= d_(ell_k,d_k)`, then the vessel arrives:

- `x_k^(t+1) = 0`
- `ell_k^(t+1) = d_k^(t)`
- `asea_k^(t+1) = 0`
- destination-port queue increments by 1

### Delay update

If a vessel is docked and either:

- receives a rejected slot response, or
- is still awaiting a slot response,

then:

`h_k^(t+1) = h_k^(t) + dt`

### Pending-departure update

If a vessel is accepted for dispatch with coordinator departure window `W_k`, define:

`Tdep_k = Taccept_k + floor(W_k / dt)`

Then:

- until `t < Tdep_k`, the vessel remains docked and `p_k = 1`
- once `t >= Tdep_k`, the vessel transitions to `asea = 1` and `p_k = 0`

Interpretation:

- departure windows delay vessel release intentionally
- this is how the coordinator can spread departures over time

## 5.2 Port state and dynamics

For port `j`, define the time-evolving state as:

`S_p,j^(t) = (Q_j^(t), D_j, O_j^(t), tau_j^(t), W_j^(t), N_j^(t))`

where:

- `Q_j^(t)` = queue length
- `D_j` = total number of docks
- `O_j^(t)` = occupied docks
- `tau_j^(t)` = vector of service-time remainders for occupied berths
- `W_j^(t)` = cumulative waiting time in vessel-hours
- `N_j^(t)` = cumulative number of vessels served

### Service countdown

For each occupied berth `b`:

`tau_(j,b)^(t+1) = max(tau_(j,b)^(t) - dt, 0)`

If a timer reaches zero, that berth becomes free.

### Occupancy definition

`O_j^(t) = | { b : tau_(j,b)^(t) > 0 } |`

### Queue update

Let `served_j^(t)` be the number admitted from queue this step, and `arrivals_j^(t)` the number of newly arriving vessels that enter the queue this step.

Then:

`Q_j^(t+1) = max(Q_j^(t) - served_j^(t) + arrivals_j^(t), 0)`

### Cumulative waiting-time update

`W_j^(t+1) = W_j^(t) + Q_j^(t) * dt`

Interpretation:

- this is not just counting how many vessels are waiting
- it accumulates how long they wait in total
- this is why the port reward uses queue length times time-step

### Throughput update

Let `C_j^(t)` be the number of services that complete during the current step.

Then:

`N_j^(t+1) = N_j^(t) + C_j^(t)`

## 5.3 Weather state and dynamics

Let `S^(t)` be the route-weather matrix of size `P x P`.

For each route `i -> j`, `S_ij^(t)` is the sea state seen on that route.

The weather follows an AR(1) process:

`S^(t+1) = alpha * S^(t) + (1 - alpha) * eps^(t)`

where:

- `eps^(t)` is a symmetric random matrix with entries drawn from `Uniform(0, s_max)`
- the diagonal is zero
- values are clipped into `[0, s_max]`

Interpretation:

- when `alpha = 0`, weather is i.i.d.
- when `alpha > 0`, weather is persistent
- persistence is what makes short-term forecasting meaningful

In the Friday run:

- `alpha = 0.7`
- weather is persistent enough to create structured periods of rough and calm routes

## 5.4 Fleet coordinator state and dynamics

For coordinator `c`, define the tracked state:

`S_c^(t) = (dC^(t), WC^(t), Be^(t), Etot^(t))`

where:

- `dC^(t)` = most recent primary destination directive
- `WC^(t)` = most recent departure-window directive
- `Be^(t)` = most recent emission-budget directive
- `Etot^(t)` = fleet cumulative emissions summary visible to the coordinator

The fleet-emissions summary is derived from vessel state:

`Etot^(t) = sum_k E_k^(t)`

If the coordinator acts at step `t` and chooses action:

`Ac^(t) = (dC_star^(t), WC_star^(t), Be_star^(t))`

then the persistent coordinator directive state is updated by assignment:

- `dC^(t+1) = dC_star^(t)`
- `WC^(t+1) = WC_star^(t)`
- `Be^(t+1) = Be_star^(t)`

If the coordinator is not due to act, those directive values remain unchanged.

### Heuristic coordinator rule currently used in baselines

For the heuristic `forecast` / `oracle` coordinator, the emission-budget directive is:

`Be_star^(t) = max(50.0 - 0.1 * Etot^(t), 10.0)`

and the heuristic departure window is fixed to:

`WC_star^(t) = 0`

Interpretation:

- the heuristic coordinator does not currently exploit non-zero departure windows
- the learned coordinator can choose among multiple departure-window bins

## 6. Observations, actions, and cadences

## 6.1 Decision cadence

Default cadences are:

- coordinator: every 12 steps
- vessel: every 1 step
- port: every 2 steps

If an agent is not due to act, its previous action is reused.

Interpretation:

- the vessel layer is the most reactive
- the coordinator layer is intentionally slower and strategic
- the port layer sits in between

## 6.2 Coordinator observations and actions

Coordinator observes:

- medium-term congestion forecast
- vessel summaries
- cumulative fleet emissions
- full weather matrix when weather is enabled

Coordinator outputs:

- destination directive
- optional per-vessel destination overrides
- departure-window directive
- emission budget

## 6.3 Vessel observations and actions

Vessel observes:

- latest coordinator directive
- short-term forecast for relevant destination
- local state: position, speed, fuel, emissions, delay, dock-availability signal
- local route sea state when weather is enabled

Vessel outputs:

- target speed
- whether to request arrival slot
- requested arrival time

## 6.4 Port observations and actions

Port observes:

- queue length
- total docks and occupied docks
- number of incoming requests
- own-port short forecast row
- compact inbound-weather summary when enabled

Port outputs:

- service rate
- acceptance limit for pending requests

## 7. Reward system and code-level verification

This is the section that matters most for the emissions question.

## 7.1 Reward formulas actually implemented

### Vessel reward

For vessel `k` at step `t`:

`Rv_k^(t) = -( fuel_weight * fuel_used_k^(t) + delay_weight * delay_k^(t) + emission_weight * co2_k^(t) ) + weather_shaping_k^(t)`

Important point:

- `fuel_used_k^(t)` here is the **fuel burned during the current step**, not remaining fuel and not cumulative fuel used over the whole episode
- `co2_k^(t)` here is the **CO2 emitted during the current step**, not cumulative vessel emissions
- `delay_k^(t)` here is the additional delay accumulated during the current step

### Port reward

For port `j` at step `t`:

`Rp_j^(t) = -( Q_j^(t) * dt + dock_idle_weight * idle_docks_j^(t) )`

Important point:

- this reward does not use a queue delta
- it uses current queue snapshot times `dt` as a proxy for waiting-time accumulation during the current step

### Coordinator reward

For the coordinator at step `t`:

`Rc^(t) = -( fleet_fuel_used^(t) + avg_queue^(t) + emission_lambda * fleet_co2^(t) ) + weather_shaping_coord^(t)`

Important point:

- `fleet_fuel_used^(t)` is the sum of **step-level fuel used** across vessels
- `fleet_co2^(t)` is the sum of **step-level CO2 emitted** across vessels
- `avg_queue^(t)` is the average current queue across ports

## 7.2 Why the statement about step-level deltas is correct

Yes, the statement is correct.

The exact code path is:

1. `step_vessels()` computes per-step physical usage and returns a dictionary `step_stats`.
   - for each vessel it stores:
     - `fuel_used`
     - `co2_emitted`
     - `arrived`
2. Inside `step_vessels()`, those values come from `compute_fuel_and_emissions()` for the current tick.
3. `env._compute_rewards()` receives this structure as `vessel_step_stats`.
4. Vessel reward is computed by passing:
   - `fuel_used = vessel_step_stats[vessel_id]['fuel_used']`
   - `co2_emitted = vessel_step_stats[vessel_id]['co2_emitted']`
   - `delay_hours = step_delay_by_vessel[vessel_id]`
5. Coordinator reward is computed from:
   - `step_fuel_used = sum(stats['fuel_used'])`
   - `step_co2_emitted = sum(stats['co2_emitted'])`

Therefore:

- vessel reward uses step-level fuel / CO2 / delay quantities
- coordinator reward uses step-level fleet aggregates of fuel / CO2
- cumulative emissions `vessel.emissions` are tracked in the state, but not used directly as the vessel or coordinator reward argument

## 7.3 What is cumulative and what is not

### Cumulative variables

These are cumulative over time:

- `vessel.emissions`
- `vessel.delay_hours`
- `port.cumulative_wait_hours`
- `port.vessels_served`
- coordinator-visible fleet total emissions summary

### Step-level variables used directly in reward

These are current-step quantities:

- `fuel_used`
- `co2_emitted`
- `step_delay_by_vessel`
- `step_fuel_used`
- `step_co2_emitted`
- current queue snapshot `Q_j^(t)` in port reward

## 7.4 Why this distinction matters

If reward used cumulative emissions directly at every step, then once emissions were incurred they would be penalized repeatedly on every later step.

That would distort learning because:

- early emissions would dominate all later decisions
- reward would no longer behave like a standard additive per-step objective
- vessel and coordinator penalties would effectively double-count history every step

The current implementation avoids that problem.

## 7.5 Weather shaping terms

In addition to base rewards, there are optional shaping bonuses:

- vessel bonus for slowing down in rough weather
- coordinator bonus for routing through calmer routes

These are additive and only active when weather is enabled.

Interpretation:

- shaping does not redefine the main reward objective
- it nudges learning toward weather-efficient behaviors

## 8. Forecasting assumptions

The current Friday experiments use **heuristic forecasters**.

Specifically:

- `ShortTermForecaster` provides heuristic short-horizon congestion predictions
- `MediumTermForecaster` provides heuristic medium-horizon congestion predictions
- `OracleForecaster` is an analysis-only upper bound with privileged future knowledge

What is **not** being presented as part of the Friday results:

- learned MLP forecasting as the main forecast source
- learned GRU forecasting as the main forecast source

Interpretation:

- the current evidence is about hierarchical coordination with heuristic forecast signals
- the learned-forecast question remains a separate experiment axis

## 9. Friday experiment setup

The main full-scale run for Friday is stored in:

- [runs/friday_meeting_2026-03-06](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06)

Configuration:

- ports: `5`
- vessels: `8`
- docks per port: `3`
- weather enabled: `True`
- weather autocorrelation: `0.7`
- sea-state max: `3.0`
- departure-window options: `(0, 6, 12, 24)`
- rollout length: `64`
- training iterations: `80`
- parameter sharing: `True`
- reward normalization: `True`
- observation normalization: `True`

Wall-clock training summary:

- total training time: about `5.9` minutes
- time per iteration: about `4.44` seconds

## 10. Main results from the full-scale run

From [report.md](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/report.md):

### Training reward summary

- iterations: `80`
- final mean reward: `-46.742864`
- best mean reward: `-19.455224`
- worst mean reward: `-68.874162`
- reward standard deviation: `12.363379`
- first-10-percent average reward: `-56.046598`
- last-10-percent average reward: `-45.255020`
- late minus early improvement: `+10.791578`

### Final per-agent training rewards

- vessel mean reward: `-2.749038`
- port mean reward: `-1.418750`
- coordinator mean reward: `-43.993825`

### Evaluation summary across 5 episodes

- mean vessel reward: `-2.8223`
- mean port reward: `-1.4333`
- mean coordinator reward: `-45.5872`
- mean total reward: `-3439.1551`
- mean total fuel used: `436.3409`
- mean total emissions: `1358.7655` tons CO2
- mean average delay: `16.1750` hours
- mean total operating cost: `$1,031,093.42`
- mean cost reliability: `0.8711`

## 11. Interpretation of the updated training curves

Figure:

- [training_curves.png](/home/ptz/dev/hmarl/qgi_lab_hmarl/qgi_lab_hmarl/runs/friday_meeting_2026-03-06/training_curves.png)

The updated figure contains:

- overall training reward curve
- per-agent reward curves for vessel, port, and coordinator
- critic value-loss curves

Interpretation:

1. The run is still noisy.
   - This is expected because the environment is weather-enabled, multi-agent, and has heterogeneous reward scales.
2. Training does show improvement on average.
   - Early average reward: `-56.05`
   - Late average reward: `-45.26`
3. The best reward occurs well before the end.
   - Best iteration: `38`
   - Best reward: `-19.46`
   - This suggests the training trajectory is not monotonically improving and remains sensitive to exploration and environment variability.
4. Port reward is relatively stable.
   - The port term stays near `-1.4`.
   - Most reward volatility comes from vessel and especially coordinator components.
5. Coordinator reward dominates magnitude.
   - This is expected because the coordinator objective includes fleet fuel, average queue, and fleet emissions.
6. Critic losses do not indicate catastrophic divergence.
   - Learning is still noisy, but the run remains numerically stable.

What this means for Friday:

- it is reasonable to claim that longer training is now in place and that agent-specific learning curves are available
- it is not yet reasonable to claim clean convergence or final policy optimality

## 12. Interpretation of the existing demo figures

The older `runs/demo` figures are still useful for discussion because they show the qualitative behavior of the system and the ablations.

## 12.1 Heuristic policy comparison

Source table: `runs/demo/policy_summary.csv`

Final summary values:

- independent:
  - avg queue `0.127`
  - emissions `2234`
  - total cost `$982,089`
- reactive:
  - avg queue `0.073`
  - emissions `2344`
  - total cost `$939,641`
- forecast:
  - avg queue `0.073`
  - emissions `2960`
  - total cost `$999,696`
- oracle:
  - avg queue `0.073`
  - emissions `3184`
  - total cost `$1,029,989`

Interpretation:

- coordination clearly improves congestion relative to the independent baseline
- however, forecast-informed heuristics do not currently dominate the reactive heuristic economically
- the reactive heuristic is the strongest of the hand-designed policies in this demo set

## 12.2 Horizon sweep

Final-step results:

- `6h`: emissions `3279`, total cost `$1.058M`
- `12h`: emissions `2960`, total cost `$0.999M`
- `24h`: emissions `4267`, total cost `$1.119M`

Interpretation:

- `12h` is the best of the tested heuristic horizon choices
- longer horizon is not automatically better

## 12.3 Noise sweep

Final-step results:

- noise `0.0`: emissions `2433`, total cost `$0.942M`
- noise `0.5`: emissions `2960`, total cost `$1.000M`
- noise `1.0`: emissions `3717`, total cost `$1.080M`
- noise `2.0`: emissions `3879`, total cost `$1.095M`

Interpretation:

- forecast degradation increases emissions and cost as expected
- in these short runs the queue remains low, so the main visible effect is economic rather than queue explosion

## 12.4 Sharing sweep

Final-step results for `shared` and `coordinator_only` are essentially identical in the current demo.

Interpretation:

- in the current heuristic stack, sharing itself is not the limiting factor
- the limiting factor is more likely the quality of the downstream policy logic

## 12.5 MAPPO vs heuristic baselines

From `runs/demo/mappo_mappo.csv`, `mappo_reactive.csv`, `mappo_forecast.csv`:

- MAPPO:
  - emissions `314`
  - total ops cost `$718,740`
- reactive:
  - emissions `2344`
  - total ops cost `$939,641`
- forecast:
  - emissions `2960`
  - total ops cost `$999,696`

Interpretation:

- MAPPO shows a strong advantage over the heuristic baselines in the demo comparison
- however, that comparison is still limited by short training and single-seed evidence
- so the correct claim is that MAPPO is promising, not that the case is already closed

## 12.6 Ablation results

From `runs/demo/ablation_results.csv`:

- full model final mean reward: `-30.45`
- no-weather final mean reward: `-25.62`
- high-entropy final mean reward: `-30.45`

Interpretation:

- removing weather makes the task easier, which confirms that weather is materially active in the reward and transition dynamics
- the short-run entropy ablation does not separate from baseline, likely because the budget is too small to reveal the difference

## 13. What is ready for Friday

The following discussion points are now on solid ground:

- the full environment dynamics can be stated mathematically
- the transition kernel is explicit and phase-ordered
- the reward definitions can be defended precisely
- the emissions term can be explained correctly and unambiguously
- the role of heuristic forecasting is now explicit
- longer training has been run and agent-level learning curves are available

## 14. What is still not final

The following caveats should be stated clearly:

- the long run is still effectively single-seed evidence
- training is noisy and not yet cleanly converged
- forecast-learning has not yet been isolated, because the Friday runs use heuristic forecasters
- stronger experimental claims would require multi-seed validation and likely longer training budgets

## 15. Bottom-line message for the meeting

The correct overall framing is:

- the HMARL model is now well specified mathematically
- the reward design, including emissions, is internally consistent with the current implementation
- the current forecasts used in the experiments are heuristic
- the new longer MAPPO run provides better learning evidence than the earlier short demo
- the project is ready for a substantive Friday discussion on architecture, reward structure, and experimental direction
- the next technical step is stronger validation, not a redefinition of the underlying model
