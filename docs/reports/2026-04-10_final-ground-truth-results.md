# Final Continuous + Ground-Truth Results

This note turns the completed local full-scale run into report-ready language.

Canonical output root:

```text
runs/final_local_2026-04-09/
```

The final completion path used:

- `episode_mode = continuous`
- `forecast_source = ground_truth`
- `num_vessels = 8`
- `num_ports = 5`
- `rollout_length = 64`
- `rollout_steps = 69`
- `iterations = 100`

Primary artifacts:

- artifact run: `runs/final_local_2026-04-09/main_seed42_artifacts/`
- multi-seed run: `runs/final_local_2026-04-09/main_multiseed/`

## Final Result Summary

The final project result is a continuous HMARL controller trained and evaluated
with ground-truth congestion information in the 8-vessel / 5-port simulator.
This setup was chosen to reduce forecast-induced error and isolate control
performance in the final project phase.

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
`runs/final_local_2026-04-09/main_seed42_artifacts/eval_result.json`

These values are the mean and standard deviation over 5 deterministic
evaluation episodes for the `seed=42` artifact model.

| Metric | Mean | Std |
| --- | ---: | ---: |
| Total reward | -956.33 | 70.19 |
| Mean vessel reward | -3.10 | 0.42 |
| Mean port reward | -0.87 | 0.04 |
| Mean coordinator reward | -9.89 | 0.97 |
| On-time rate | 0.968 | 0.064 |
| Completed arrivals | 21.4 | 4.03 |
| On-time arrivals | 20.6 | 3.61 |
| Average delay hours | 28.03 | 3.75 |
| Average schedule delay hours | 0.153 | 0.278 |
| Total vessels served | 30.8 | 5.31 |
| Average queue | 0.00 | 0.00 |
| Dock utilization | 0.12 | 0.07 |
| Total fuel used | 234.10 | 43.89 |
| Total operations cost (USD) | 1,327,073.05 | 115,982.69 |

### Table 2. Multi-Seed Training Summary

Source:
`runs/final_local_2026-04-09/main_multiseed/summary.csv`

These values summarize the final training outcome for each seed after 100
iterations.

| Seed | Final mean reward | Best mean reward | Best iteration | Reward improvement |
| ---: | ---: | ---: | ---: | ---: |
| 42 | -12.85 | -12.59 | 97 | 15.78 |
| 49 | -16.92 | -15.24 | 96 | 9.29 |
| 56 | -16.54 | -13.41 | 93 | 11.12 |
| 63 | -11.54 | -9.66 | 98 | 15.16 |
| 70 | -12.07 | -11.76 | 89 | 16.73 |

### Table 3. Multi-Seed Aggregate Training Summary

Source:
`runs/final_local_2026-04-09/main_multiseed/experiment_summary.json`

| Metric | Value |
| --- | ---: |
| Number of seeds | 5 |
| Mean final mean reward | -13.99 |
| Std final mean reward | 2.28 |
| Mean best mean reward | -12.53 |
| Std best mean reward | 1.84 |
| Mean reward improvement | 13.61 |
| Std reward improvement | 2.89 |
| Mean training time per seed (s) | 323.20 |
| Total multi-seed training time (s) | 1616.01 |

## Report-Ready Language

### Main Results Paragraph

We evaluated the final HMARL controller in the continuous 8-vessel / 5-port
simulator using ground-truth congestion information to reduce forecast-induced
error and isolate control performance. In the canonical artifact run
(`seed=42`), evaluation over 5 deterministic episodes yielded a mean total
reward of `-956.33 ± 70.19`, an on-time rate of `0.968 ± 0.064`, and
`21.4 ± 4.0` completed arrivals. The controller maintained zero stalled vessels
and near-zero average queue under this setup, while serving `30.8 ± 5.3` port
service events.

### Multi-Seed Stability Paragraph

To test whether the result was stable across random initialization, we repeated
training for 5 seeds (`42, 49, 56, 63, 70`) using the same continuous +
ground-truth configuration. The mean final training reward across seeds was
`-13.99 ± 2.28`, with a mean reward improvement of `13.61 ± 2.89` relative to
early training. All five seeds improved substantially over the course of the
100-iteration run, which indicates that the final configuration is reasonably
stable at the current project scale.

### Short Interpretation Paragraph

Taken together, these results suggest that the continuous controller can learn
coherent routing and coordination behavior when reliable congestion information
is available. The final setup produced strong on-time performance, no vessel
stalling, and consistent reward improvement across seeds, making it a suitable
final project result for the current simulator scope.

## Table Caption Language

### Caption for Artifact Evaluation Table

Evaluation summary for the final continuous + ground-truth HMARL controller.
Values are mean ± standard deviation over 5 deterministic evaluation episodes
for the canonical `seed=42` artifact run.

### Caption for Multi-Seed Table

Training summary across 5 seeds for the final continuous + ground-truth HMARL
configuration. `Final mean reward` is the training reward at iteration 100,
`best mean reward` is the best training reward observed during the run, and
`reward improvement` is the difference between late and early training reward.

## Recommended Wording for Limitations

The final result should be presented with two explicit scope constraints:

1. The simulator uses a synthetic 8-vessel / 5-port environment rather than a
   real port network.
2. The final experiment uses ground-truth congestion information, so the claim
   is about control quality under reliable forecasts rather than robustness to
   forecast error.

## Referenced Output Files

- `runs/final_local_2026-04-09/main_seed42_artifacts/report.md`
- `runs/final_local_2026-04-09/main_seed42_artifacts/eval_result.json`
- `runs/final_local_2026-04-09/main_seed42_artifacts/train_history.csv`
- `runs/final_local_2026-04-09/main_multiseed/summary.csv`
- `runs/final_local_2026-04-09/main_multiseed/experiment_summary.json`
- `runs/final_local_2026-04-09/final_run_manifest.json`
