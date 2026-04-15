# HMARL Maritime Scheduling — Presentation Materials

**Audience:** project presentation for professor and general technical audience  
**Date:** 2026-03-31  
**Prepared from:** project status as of 2026-03-31  
**Recommended format:** 12 main slides + 4 backup slides  
**Recommended length:** 10-12 minutes

## 1. Presentation goal

This presentation should feel like a complete project story, not only a model update.

By the end, the audience should understand:

1. what real scheduling problem the project is studying
2. how the simulator and MARL setup are currently defined
3. what changed since the Friday meeting on 2026-03-06
4. what the best current baseline is
5. what the next research decision should be

## 2. Main message

The core message for the audience is:

> We built a hierarchical multi-agent maritime scheduling simulator, made the environment and diagnostics substantially more trustworthy, and now have a clear working baseline (`v3`) for studying the tradeoff between throughput, schedule quality, and operating cost.

Updated framing:

> We now have a strong continuous baseline (`v3`), a cleaner final run path using `continuous + ground_truth`, and enough diagnostics to study control quality directly without conflating it with forecast error.

## 3. Recommended slide deck

## Slide 1 — Title and framing

**Title:**  
Hierarchical Multi-Agent Reinforcement Learning for Congestion-Aware Maritime Scheduling

**Bullets:**

- Spring 2026 independent study
- hierarchical MARL for maritime coordination
- project status update and current baseline

**Speaker notes:**

Start by positioning this as both a simulator-building and decision-making project. The audience should know immediately that the work is not just “training a model,” but building a trustworthy testbed for coordinated scheduling.

## Slide 2 — Real-world context

**Title:**  
Why this problem matters

**Bullets:**

- ports have limited berth capacity and service time
- vessels must trade off speed, fuel, emissions, and delay
- poor local decisions can create downstream congestion
- the system is naturally multi-agent, asynchronous, and resource-constrained

**Speaker notes:**

Keep this slide intuitive. Explain that maritime scheduling is hard because decisions interact across space and time. This gives the audience the motivation before any RL detail appears.

## Slide 3 — Research questions

**Title:**  
What we are trying to learn

**Bullets:**

- umbrella question:
  - can hierarchical MARL improve maritime scheduling under congestion and resource constraints?
- explicit project research questions:
  - `RQ1`: how can heterogeneous agents coordinate using shared congestion forecasts?
  - `RQ2`: does proactive coordination improve over independent and reactive baselines?
  - `RQ3`: which forecast horizons and sharing strategies maximize decision quality?
  - `RQ4`: how do coordination improvements affect economics such as fuel, delay, and carbon cost?

**Speaker notes:**

This slide should make the scope more honest. Present one umbrella question, then explain that the project is really organized around four linked research questions: coordination, baseline comparison, forecast design, and economics.

## Slide 4 — Environment setup

**Title:**  
Current simulator setup

**Bullets:**

- `5` ports, `8` vessels, `3` docks per port
- `1` hour time step, `6` hour default service time
- synthetic port-distance matrix: `22–98 nm`
- vessels start at ports, not at sea
- default fuel: `100`, with refueling after actual service completion
- asynchronous decision cadence with message latency

**Speaker notes:**

This is one of the missing background slides. Make it concrete. Mention that the geography is currently synthetic and scaled so multiple voyages can complete within a rollout.

## Slide 5 — Experiment setup

**Title:**  
How the current baseline is trained and evaluated

**Bullets:**

- algorithm: MAPPO with centralized training and decentralized execution
- parameter sharing across vessel agents and across port agents
- main operating baseline:
  - `100` training iterations
  - `rollout_length = 64`
  - environment horizon `= 69`
  - `seed = 42`
  - local CPU training for the reported run
- final completion path:
  - `continuous`
  - `ground_truth` forecasts
  - one artifact run plus one five-seed run
- primary evaluation metrics:
  - on-time rate
  - completed arrivals
  - port service events
  - dock utilization
  - total operating cost

**Speaker notes:**

Do not overload this slide with every hyperparameter. The important point now is that the final completion path is deliberately simpler: keep the continuous environment, remove forecast error with `ground_truth`, and use one artifact run plus a five-seed run.

## Slide 6 — System architecture

**Title:**  
Three decision layers with delayed coordination

**Bullets:**

- fleet coordinator:
  - destination guidance
  - slowest decision cadence
- vessel agents:
  - speed control
  - arrival-slot requests
- port agents:
  - request acceptance
  - service and berth decisions
- asynchronous message bus connects all layers

**Speaker notes:**

This is the mechanism slide. Emphasize that information flows through delayed messages and physical movement, which makes the problem more realistic than a simple synchronized toy environment.

## Slide 7 — Transition kernel and rewards

**Title:**  
How one simulation step works

**Bullets:**

- the transition kernel advances the system in a fixed sequence:
  - apply due actions and deliver delayed messages
  - update vessel motion, fuel, emissions, and arrivals
  - update port service, queue, berth occupancy, and refueling
  - compute rewards and log diagnostics
- reward formulas:
  - vessel:
    - `r_V(t) = -(w_f Δfuel_t + w_d Δdelay_t + w_e ΔCO2_t + w_t Δtransit_t + w_s Δsched_t) + r_arr 1[arrived_t] + r_on 1[on_time_t]`
  - port:
    - `r_P(t) = r_acc accepted_t + r_srv served_t - r_rej rejected_t - (queue_t · dt + w_idle idle_docks_t)`
  - coordinator:
    - `r_C(t) = r_acc accepted_t + r_srv served_t + r_util avg_occupied_t - (w_f Δfuel_t + w_q avg_queue_t + w_i avg_idle_t + w_d Δdelay_t + w_s Δsched_t + w_e ΔCO2_t + r_rej rejected_t)`

**Speaker notes:**

This slide now carries the actual reward equations, but they should be presented visually as math rather than read token by token. Explain the three formulas at a high level: vessel for local operating efficiency and punctuality, port for queue and berth management, and coordinator for system throughput, utilization, and cost.

## Slide 8 — What changed since March 6

**Title:**  
Simulator realism, rewards, and validation changes

**Bullets:**

- corrected vessel action semantics so speeds no longer collapse to the minimum
- requested arrival times now become real schedule deadlines
- fuel exhaustion now causes real mid-route stalling
- departures are checked for fuel feasibility
- ports track actual vessels through queueing, service, and refueling
- rewards were redesigned from mostly aggregate penalties to event-driven, role-specific signals
- moved the final run plan to `continuous + ground_truth`
- fixed training resets so episode starts now vary reproducibly across resets

**Speaker notes:**

This is a strong progress slide. It shows that recent work was not only reward tuning. We improved simulator correctness, made the reward signal more informative, and cleaned up the final training path so it matches the actual project goal.

## Slide 9 — Why we trust the results more now

**Title:**  
Diagnostics and transparency upgrades

**Bullets:**

- grouped plots for aggregate, vessel, port, and coordinator behavior
- per-step `eval_trace.csv` with operational state
- per-step action log and event log
- reward-component decomposition
- policy-confidence diagnostics beyond raw entropy

**Visual suggestion:**

- [diagnostics_trace.png](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/diagnostics_trace.png)

**Speaker notes:**

This slide is important because it explains why the current claims are more defensible. Earlier, it was hard to separate simulator issues from learning issues. Now the project is much more inspectable.

## Slide 10 — Current best baseline

**Title:**  
Current recommended baseline: transit-rebalanced v3

**Bullets:**

- preferred run: `local_full_train_2026-03-11_transit_rebalanced_v3`
- `total_reward = -923.38`
- `on_time_rate = 0.928`
- `completed_arrivals = 26.8`
- `port_service_events = 36.2`
- `dock_utilization = 0.28`
- `total_ops_cost_usd = $1.330M`

**Visual suggestion:**

- [training_curves.png](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/training_curves.png)

**Speaker notes:**

This is the “where we are now” slide. Make it clear that `v3` is recommended because it is the best balance, not because it is perfect.

## Slide 11 — Final run plan

**Title:**  
Final full-scale run plan

**Bullets:**

- artifact run:
  - `continuous + ground_truth`
  - `100` iterations
  - seed `42`
  - full plots, traces, report, and saved model
- multi-seed run:
  - same environment and training setup
  - seeds `42, 49, 56, 63, 70`
  - used for the final quantitative claim
- key lesson:
  - the project now finishes on one clean task definition instead of mixing a main environment with a separate validation benchmark

**Metric note:**

- `port_service_events` counts berth admissions across the full system, including seeded background port load
- the artifact run gives the diagnostic trace; the multi-seed run gives the stability summary

**Speaker notes:**

This slide should make the finish line very concrete. We are now using one continuous environment, one reliable forecast source, and two complementary run types: one for artifacts and one for statistical stability.

## Slide 12 — Limitations and next steps

**Title:**  
What is simplified, and what comes next

**Bullets:**

- current port matrix is synthetic, not real geography
- route distances were scaled so multiple trips fit in one rollout
- current fuel level is generous relative to route length
- next steps:
  - run the final `continuous + ground_truth` local plan
  - use the artifact run for figures, traces, and narrative examples
  - use the five-seed run for the final quantitative result
  - optionally reintroduce imperfect forecasts as follow-on work
  - decide whether to move to real ports and nautical distances

**Speaker notes:**

End with maturity and momentum. The audience should leave with confidence that the current baseline is defensible and that the next research step is clear.

## 4. Suggested talk track

If the audience is mixed technical / non-technical, a clean story arc is:

1. why this problem matters
2. how the environment is set up
3. why hierarchical agents make sense
4. what we fixed to make the simulator trustworthy
5. what the current best result is
6. what remains simplified
7. what we will do next

## 5. Backup slides

## Backup 1 — Detailed run comparison

**Title:**  
Recent March run comparison

| Run | Total Reward | On-Time Rate | Avg Schedule Delay (h) | Completed Arrivals | Vessels Served | Dock Utilization | Total Ops Cost |
|-----|--------------|--------------|-------------------------|--------------------|----------------|------------------|----------------|
| `reward_balance_v2` | `-1137.46` | `0.9448` | `0.4651` | `23.8` | `33.2` | `0.1467` | `$1.382M` |
| `transit_rebalanced_v3` | `-923.38` | `0.9282` | `0.3301` | `26.8` | `36.2` | `0.2800` | `$1.330M` |
| `ontime_rebalanced_v4` | `-864.58` | `1.0000` | `0.0000` | `24.0` | `33.4` | `0.0667` | `$1.373M` |

## Backup 2 — Reward notation and interpretation

**Title:**  
Technical backup: reward notation and interpretation

**Bullets:**

- notation:
  - `Δfuel_t`, `Δdelay_t`, `ΔCO2_t`, `Δtransit_t`, `Δsched_t` are per-step increments
  - `accepted_t`, `rejected_t`, and `served_t` are per-step event counts
  - `avg_queue_t`, `avg_idle_t`, and `avg_occupied_t` are fleet-level averages across ports
- interpretation:
  - vessel reward captures local operating efficiency and punctuality
  - port reward captures berth responsiveness and queue management
  - coordinator reward captures system throughput, utilization, and global cost control
- optional detail:
  - weather-shaping terms exist but are omitted from the main formula for readability

**Talking point:**

- the reward is not a single scalar heuristic; it is a structured multi-level objective tied to the actual transition dynamics

## Backup 3 — Current simulator assumptions

**Title:**  
Current default environment values

**Bullets:**

- `num_ports = 5`
- `num_vessels = 8`
- `docks_per_port = 3`
- `rollout_length = 64`
- `env.rollout_steps = 69`
- `initial_fuel = 100`
- synthetic distance matrix with routes from `22` to `98 nm`

**Talking point:**

- the port-distance matrix is a symmetric nautical-mile lookup table between abstract ports
- it is synthetic rather than geographic, because the current environment is scaled for learnability and for multiple completed voyages per rollout

## Backup 4 — Fuel interpretation

**Title:**  
How to interpret the current fuel setting

**Bullets:**

- `initial_fuel = 100`
- rough one-tank range:
  - `8 kn`: about `781 nm`
  - `12 kn`: about `347 nm`
  - `18 kn`: about `154 nm`
- current route distances are only `22–98 nm`

**Talking point:**

- fuel logic is now modeled correctly, but in the current default topology fuel is rarely the dominant operational constraint
