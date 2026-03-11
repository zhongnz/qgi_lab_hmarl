# Friday Meeting Update — HMARL Maritime Scheduling

**Date:** 2026-03-13  
**Generated:** 2026-03-11  
**Purpose:** comprehensive technical update for the Friday discussion, covering all work completed since the Friday meeting on 2026-03-06.  
**What this document covers:** environment and reward fixes, diagnostics and logging upgrades, training and evaluation results, recommended baseline, current simulator assumptions, and next steps.

## 1. Executive summary

Since the 2026-03-06 Friday meeting, the project has moved from “a working training loop with unclear diagnostics” to “a substantially more trustworthy simulator and a much more interpretable MAPPO training/evaluation workflow.”

The most important outcomes are:

1. The simulator mechanics are now materially more realistic and internally consistent.
2. The evaluation pipeline now exposes what the model is actually doing, step by step.
3. The current recommended baseline is the **transit-rebalanced v3** run, not the later schedule-perfect v4 variant.
4. The code defaults have been switched back to match the **v3** baseline.

The recommended baseline run is:

- [local_full_train_2026-03-11_transit_rebalanced_v3/report.md](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/report.md)

Main evaluation metrics for the recommended baseline:

- `total_reward = -923.38`
- `avg_speed = 13.19`
- `avg_delay_hours = 27.63`
- `avg_schedule_delay_hours = 0.33`
- `on_time_rate = 0.928`
- `completed_arrivals = 26.8`
- `total_vessels_served = 36.2`
- `dock_utilization = 0.28`
- `total_ops_cost_usd = $1.330M`
- `stalled_vessels = 0.0`

Interpretation:

- throughput is materially better than earlier March runs
- schedule performance is still strong
- operating cost improved
- the model no longer appears to be benefiting from simulator bugs or opaque plotting

## 2. Work completed since 2026-03-06

The work since the previous Friday meeting falls into four main categories:

- environment correctness
- diagnostics and transparency
- training / reward tuning
- reporting and documentation

### 2.1 Environment correctness fixes

Several issues that affected realism or interpretability were corrected.

#### 2.1.1 Vessel action semantics

The vessel policy output was corrected so that the learned action is interpreted as a latent control around nominal speed, rather than as a raw physical speed value that immediately clipped to the minimum speed.

Result:

- the earlier “all vessels collapse to `8.0 kn`” failure mode was removed
- vessel speed traces now vary meaningfully during evaluation

#### 2.1.2 Schedule-aware arrivals

Requested arrival times are now part of the real environment dynamics rather than just an informal concept. The simulator now tracks arrival commitments, accumulates schedule lateness, and logs on-time arrivals explicitly.

Result:

- `on_time_rate` became a real operational metric
- schedule delay is now visible in both rewards and diagnostics

#### 2.1.3 Fuel exhaustion and stalling

Mid-route fuel depletion is now modeled as an actual stall instead of allowing vessels to keep moving with zero fuel. When a vessel runs out of fuel partway through a step, it only travels as far as its remaining fuel allows, then becomes stranded until serviced.

Result:

- no more “free travel at zero fuel”
- stall behavior now appears in metrics and diagnostics

#### 2.1.4 Fuel-feasible departures

The environment now checks whether a departure is fuel-feasible before dispatch:

- if necessary, departure speed is capped to a fuel-feasible value
- if the route is infeasible even at minimum speed, departure is blocked

Result:

- the simulator avoids obviously impossible legs at dispatch time

#### 2.1.5 Port service and refueling

Ports now track actual vessel IDs through queueing and service, instead of only maintaining aggregate counts. Vessels are refueled only after real port service completion.

Result:

- service completion is tied to specific vessels
- refueling is now a real operational event
- fuel accounting remains cumulative for metrics

#### 2.1.6 Reservation and future-load handling

Reservation handling was improved so ports and the coordinator see booked future load, not just current queue/occupancy. This included advance reservation accounting and ETA-aware future-load features.

Result:

- policies now see pending, booked, and imminent arrivals
- decisions are less myopic than in the early March configuration

## 3. Diagnostics and transparency improvements

One of the main themes since the Friday meeting was making the system inspectable enough to support meaningful debugging.

### 3.1 Training curves and grouped diagnostics

The plotting pipeline was expanded so that training no longer relies on a single overloaded reward plot.

Artifacts now include:

- aggregate training curves
- separate vessel / port / coordinator diagnostics plots
- `eval_trace.csv` with step-level metrics
- grouped diagnostics PNGs

### 3.2 Full trace logging

The evaluation pipeline now records:

- per-step aggregate operational metrics
- per-vessel state trajectories
- per-port state trajectories
- per-step action logs
- per-step event logs

Artifacts include:

- [eval_trace.csv](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/eval_trace.csv)
- `eval_action_trace.csv`
- `eval_event_log.csv`

This makes it possible to inspect:

- who moved
- who requested slots
- who was accepted or rejected
- when service completed
- when refueling happened

### 3.3 Policy-confidence diagnostics

Additional training diagnostics were added to make entropy curves interpretable:

- `vessel_log_std_0`
- `vessel_log_std_1`
- `port_top1_prob`
- `coordinator_top1_prob`
- entropy gaps from uniform for discrete policies

This was necessary because raw entropy alone was not enough to tell whether the coordinator was truly specializing or simply staying close to a uniform policy.

### 3.4 Reward-component tracing

The latest diagnostics now decompose rewards directly into named components. The trace can show, per step:

- vessel fuel / emission / delay / transit / schedule terms
- port wait / idle / service / accept / reject terms
- coordinator fuel / queue / idle / utilization / delay / schedule / throughput terms

This was used directly to identify that the vessel transit-time penalty was dominating the reward landscape more than intended.

## 4. Training and reward tuning sequence

The training-side work since 2026-03-06 was iterative. The important changes are summarized below in the order they were made.

### 4.1 Coordinator action-space simplification

The coordinator was simplified from `destination × departure-window` to a default destination-only action choice, while keeping the infrastructure available for richer timing control if needed later.

Result:

- coordinator behavior became much less random
- throughput and on-time performance improved
- training became easier to interpret

### 4.2 MAPPO schedule changes

The training schedule was adjusted so that:

- learning rate does not decay all the way to zero
- entropy coefficient now decays instead of staying constant

Result:

- the coordinator policy became more decisive
- training remained active later in the run instead of going fully inert

### 4.3 Coordinator reward rebalance

The coordinator objective was reworked so it is less dominated by duplicated fuel/emission costs and more explicitly tied to throughput and utilization.

Key additions:

- explicit coordinator fuel weight
- explicit coordinator queue weight
- explicit coordinator emission weight
- explicit coordinator utilization bonus

Result:

- reward alignment improved substantially
- throughput stayed competitive
- the coordinator signal became less punitive

### 4.4 Transit penalty rebalance — recommended v3 baseline

Using the new reward-component trace, we found that the largest persistent negative term was the **vessel transit-time penalty**. That term was reduced:

- `transit_time_weight: 12.0 -> 8.0`

This produced the current recommended baseline:

- [local_full_train_2026-03-11_transit_rebalanced_v3/report.md](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/report.md)

This was the best operating balance found in the current tuning pass.

### 4.5 On-time reward experiment — v4 variant

A follow-up experiment increased the explicit on-time arrival reward. This produced:

- [local_full_train_2026-03-11_ontime_rebalanced_v4/report.md](../../runs/local_full_train_2026-03-11_ontime_rebalanced_v4/report.md)

`v4` achieved:

- `on_time_rate = 1.0`
- `avg_schedule_delay_hours = 0.0`

But it also became too conservative:

- lower throughput than `v3`
- much lower dock utilization
- higher operating cost

Conclusion:

- `v4` is useful as a schedule-perfect comparison point
- `v3` remains the preferred baseline for balanced operations

## 5. Current recommended baseline

### 5.1 Preferred run

The preferred run for discussion and follow-on work is:

- [local_full_train_2026-03-11_transit_rebalanced_v3/report.md](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/report.md)

Supporting artifacts:

- [train_history.csv](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/train_history.csv)
- [eval_result.json](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/eval_result.json)
- [eval_trace.csv](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/eval_trace.csv)
- [training_curves.png](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/training_curves.png)
- [diagnostics_trace.png](../../runs/local_full_train_2026-03-11_transit_rebalanced_v3/diagnostics_trace.png)

### 5.2 Why v3 is preferred

`v3` is the best current balance between:

- throughput
- schedule quality
- berth utilization
- operating cost

Compared with `reward_balance_v2`, `v3` achieved:

- better total reward
- better dock utilization
- more completed arrivals
- more vessels served
- lower total operating cost

Compared with `v4`, `v3` avoided the over-conservative behavior that produced perfect on-time performance at the expense of port usage and throughput.

## 6. Comparison table for recent March runs

| Run | Total Reward | On-Time Rate | Avg Schedule Delay (h) | Completed Arrivals | Vessels Served | Dock Utilization | Total Ops Cost |
|-----|--------------|--------------|-------------------------|--------------------|----------------|------------------|----------------|
| `reward_balance_v2` | `-1137.46` | `0.9448` | `0.4651` | `23.8` | `33.2` | `0.1467` | `$1.382M` |
| `transit_rebalanced_v3` | `-923.38` | `0.9282` | `0.3301` | `26.8` | `36.2` | `0.2800` | `$1.330M` |
| `ontime_rebalanced_v4` | `-864.58` | `1.0000` | `0.0000` | `24.0` | `33.4` | `0.0667` | `$1.373M` |

Interpretation:

- `v4` wins on strict schedule adherence
- `v3` wins on overall operational balance

## 7. Current simulator assumptions

These are useful to state clearly during the Friday discussion.

### 7.1 Port distances are synthetic

The current default 5-port matrix is:

```python
[
    [0, 84, 22, 34, 78],
    [84, 0, 98, 61, 54],
    [22, 98, 0, 55, 59],
    [34, 61, 55, 0, 82],
    [78, 54, 59, 82, 0],
]
```

Important note:

- these are **not** real geographic port distances
- they were intentionally scaled so multiple voyages can finish within a single 64-step rollout

### 7.2 Vessels do not start in the middle of the sea

At reset:

- vessels start at ports
- destinations are assigned from ports
- they do not begin mid-route unless a test manually creates that condition

### 7.3 Fuel assumptions

Default `initial_fuel = 100.0`.

At current physics settings, that is enough to traverse any default route comfortably. Rough order-of-magnitude range on a full tank:

- at `8 kn`: about `781 nm`
- at `12 kn`: about `347 nm`
- at `18 kn`: about `154 nm`

Since current route distances are only `22–98 nm`, fuel depletion does not usually happen under the default setup.

### 7.4 Refueling

Vessels are refueled after real port service completion, not continuously and not automatically on simple arrival.

## 8. Validation and testing status

The codebase was repeatedly revalidated through this tuning cycle.

Current status:

- full suite passing: `729 passed`

This includes:

- reward tests
- transition-kernel tests
- observation / masking tests
- plotting tests
- MAPPO integration tests

## 9. Documentation updates

Additional documentation and notes added during this period include:

- meeting 4 and 5 notes
- expanded Friday update content
- improved diagnostics/reporting artifacts in run directories

## 10. Recommended discussion points for Friday 2026-03-13

The most useful items to discuss are:

1. Whether the project should keep the current synthetic 5-port topology or move to real ports / real nautical distances.
2. Whether the objective should prioritize overall operations (`v3`) or strict schedule adherence (`v4`-style behavior).
3. Whether the next experiment should be:
   - a multi-seed comparison between `v3` and `v4`, or
   - a migration from abstract ports to a real port-distance dataset.

My recommendation is:

- use `v3` as the default working baseline
- do a short multi-seed comparison of `v3` vs `v4`
- then decide whether to invest the next cycle in real-port data

## 11. Bottom line

The main result since 2026-03-06 is not just that the score improved. The more important result is that the simulator and diagnostics are now much easier to trust.

At this point:

- the environment is substantially cleaner
- the training artifacts are much more transparent
- the recommended baseline is clear
- the remaining choices are now modeling and research choices, not hidden implementation bugs
