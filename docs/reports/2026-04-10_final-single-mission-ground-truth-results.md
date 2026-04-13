# Final Single-Mission + Ground-Truth Results

This note turns the completed local single-mission benchmark run into
report-ready language.

Canonical output root:

```text
runs/final_single_mission_ground_truth_2026-04-10/
```

The completed benchmark path used:

- `episode_mode = single_mission`
- `mission_success_on = arrival`
- `forecast_source = ground_truth`
- `num_vessels = 8`
- `num_ports = 5`
- `rollout_length = 64`
- `rollout_steps = 69`
- `iterations = 100`

Primary artifacts:

- artifact run:
  `runs/final_single_mission_ground_truth_2026-04-10/single_mission_seed42_artifacts/`
- multi-seed run:
  `runs/final_single_mission_ground_truth_2026-04-10/single_mission_multiseed/`

## Final Result Summary

The single-mission benchmark was retained as a controlled comparison setting
for the project. Under this setup, each vessel is limited to one mission and is
marked successful on arrival. The benchmark therefore isolates mission
completion behavior more directly than the continuous environment, but it is not
the final project task.

The artifact run completed successfully and produced the full report bundle:

- `report.md`
- `train_history.csv`
- `eval_result.json`
- `eval_trace.csv`
- `eval_action_trace.csv`
- `eval_event_log.csv`
- `training_curves.png`
- `diagnostics_trace*.png`

The multi-seed run also completed successfully across 5 seeds:

- `42`
- `49`
- `56`
- `63`
- `70`

## Table-Ready Metrics

### Table 1. Artifact Run Evaluation Summary

Source:
`runs/final_single_mission_ground_truth_2026-04-10/single_mission_seed42_artifacts/eval_result.json`

These values are the mean and standard deviation over 5 deterministic
evaluation episodes for the `seed=42` artifact model.

| Metric | Mean | Std |
| --- | ---: | ---: |
| Total reward | -353.65 | 97.66 |
| Mean vessel reward | -3.02 | 0.52 |
| Mean port reward | -0.98 | 0.12 |
| Mean coordinator reward | -10.32 | 2.46 |
| On-time rate | 0.775 | 0.229 |
| Completed arrivals | 8.0 | 0.0 |
| On-time arrivals | 6.2 | 1.83 |
| Mission success rate | 1.0 | 0.0 |
| Average delay hours | 7.15 | 0.94 |
| Average schedule delay hours | 1.212 | 1.210 |
| Port service events | 9.4 | 3.2 |
| Average queue | 0.0 | 0.0 |
| Dock utilization | 0.0 | 0.0 |
| Total fuel used | 98.07 | 10.11 |
| Total operations cost (USD) | 372,327.64 | 44,129.35 |

### Table 2. Multi-Seed Training Summary

Source:
`runs/final_single_mission_ground_truth_2026-04-10/single_mission_multiseed/summary.csv`

These values summarize the final training outcome for each seed after 100
iterations.

| Seed | Final mean reward | Best mean reward | Best iteration | Reward improvement |
| ---: | ---: | ---: | ---: | ---: |
| 42 | -17.39 | -7.52 | 71 | 6.95 |
| 49 | -9.09 | -9.09 | 99 | 13.51 |
| 56 | -15.41 | -8.83 | 81 | -0.90 |
| 63 | -11.57 | -7.85 | 35 | 11.33 |
| 70 | -14.39 | -8.56 | 93 | 7.00 |

### Table 3. Multi-Seed Aggregate Training Summary

Source:
`runs/final_single_mission_ground_truth_2026-04-10/single_mission_multiseed/experiment_summary.json`

| Metric | Value |
| --- | ---: |
| Number of seeds | 5 |
| Mean final mean reward | -13.57 |
| Std final mean reward | 2.92 |
| Mean best mean reward | -8.37 |
| Std best mean reward | 0.59 |
| Mean reward improvement | 7.58 |
| Std reward improvement | 4.94 |
| Mean training time per seed (s) | 329.01 |
| Total multi-seed training time (s) | 1645.05 |

## Report-Ready Language

### Benchmark Results Paragraph

We also evaluated a controlled single-mission benchmark using the same
ground-truth congestion information. In this setup, each of the 8 vessels was
limited to one mission and was considered successful upon arrival. In the
canonical artifact run (`seed=42`), evaluation over 5 deterministic episodes
yielded a mean total reward of `-353.65 ± 97.66`, an on-time rate of
`0.775 ± 0.229`, and `8.0 ± 0.0` completed arrivals, corresponding to a
mission success rate of `1.0`. No stalled vessels were observed.

### Multi-Seed Stability Paragraph

We repeated the single-mission + ground-truth benchmark across 5 seeds
(`42, 49, 56, 63, 70`). The mean final training reward across seeds was
`-13.57 ± 2.92`, with a mean reward improvement of `7.58 ± 4.94`. Four of the
five seeds improved over training, while one seed (`56`) regressed slightly
relative to early training. This indicates that the benchmark is learnable, but
less stable than the final continuous configuration.

### Comparison Paragraph

Compared with the final continuous + ground-truth controller, the
single-mission benchmark produced perfect mission completion but weaker overall
scheduling quality. In particular, the benchmark achieved lower on-time
performance (`0.775` versus `0.968` in the continuous result) and lower
throughput by construction, since each vessel can complete at most one trip.
These results support using single-mission as a diagnostic benchmark rather than
as the final project environment.

## Table Caption Language

### Caption for Artifact Evaluation Table

Evaluation summary for the single-mission + ground-truth HMARL benchmark.
Values are mean ± standard deviation over 5 deterministic evaluation episodes
for the canonical `seed=42` artifact run.

### Caption for Multi-Seed Table

Training summary across 5 seeds for the single-mission + ground-truth HMARL
benchmark. `Final mean reward` is the training reward at iteration 100,
`best mean reward` is the best training reward observed during the run, and
`reward improvement` is the difference between late and early training reward.

## Recommended Wording for Scope

The single-mission benchmark should be presented as a controlled validation
setting rather than the main project result. It removes repeated mission
cycling, guarantees a clean mission-success signal, and is therefore useful for
debugging and comparison. However, it is less representative of the intended
continuous maritime scheduling problem than the final continuous controller.

## Referenced Output Files

- `runs/final_single_mission_ground_truth_2026-04-10/single_mission_seed42_artifacts/report.md`
- `runs/final_single_mission_ground_truth_2026-04-10/single_mission_seed42_artifacts/eval_result.json`
- `runs/final_single_mission_ground_truth_2026-04-10/single_mission_seed42_artifacts/train_history.csv`
- `runs/final_single_mission_ground_truth_2026-04-10/single_mission_multiseed/summary.csv`
- `runs/final_single_mission_ground_truth_2026-04-10/single_mission_multiseed/experiment_summary.json`
- `runs/final_single_mission_ground_truth_2026-04-10/final_run_manifest.json`
