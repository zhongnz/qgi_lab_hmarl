# Final Results Section

This section consolidates the completed continuous + ground-truth result and
the completed single-mission + ground-truth comparison benchmark into one
report-ready narrative.

## Final Experimental Setup

The final project result is based on the continuous HMARL controller evaluated
in the synthetic 8-vessel / 5-port simulator using ground-truth congestion
information. This setup was chosen in the final phase of the project to reduce
forecast-induced error and isolate the control problem.

The canonical final configuration used:

- `episode_mode = continuous`
- `forecast_source = ground_truth`
- `num_vessels = 8`
- `num_ports = 5`
- `rollout_length = 64`
- `rollout_steps = 69`
- `iterations = 100`

To test whether the simpler episodic benchmark still added insight, we also ran
a matching single-mission + ground-truth comparison in which each vessel was
limited to one mission and marked successful on arrival.

## Main Result: Continuous + Ground Truth

The continuous controller is the main project result.

In the canonical artifact run (`seed=42`), evaluation over 5 deterministic
episodes produced:

| Metric | Mean | Std |
| --- | ---: | ---: |
| Total reward | -956.33 | 70.19 |
| On-time rate | 0.968 | 0.064 |
| Completed arrivals | 21.4 | 4.03 |
| On-time arrivals | 20.6 | 3.61 |
| Average delay hours | 28.03 | 3.75 |
| Port service events | 30.8 | 5.31 |
| Total fuel used | 234.10 | 43.89 |
| Total operations cost (USD) | 1,327,073.05 | 115,982.69 |

The corresponding 5-seed training sweep (`42, 49, 56, 63, 70`) showed:

| Metric | Value |
| --- | ---: |
| Mean final mean reward | -13.99 |
| Std final mean reward | 2.28 |
| Mean best mean reward | -12.53 |
| Std best mean reward | 1.84 |
| Mean reward improvement | 13.61 |
| Std reward improvement | 2.89 |

These results show that the continuous controller can learn stable coordination
behavior under reliable congestion information. The final setup achieved strong
on-time performance, zero stalled vessels, and substantial continuous
throughput, while also showing consistent improvement across seeds.

## Comparison Benchmark: Single Mission + Ground Truth

The single-mission benchmark was kept as a controlled comparison rather than as
the final project environment.

In the canonical artifact run (`seed=42`), evaluation over 5 deterministic
episodes produced:

| Metric | Mean | Std |
| --- | ---: | ---: |
| Total reward | -353.65 | 97.66 |
| On-time rate | 0.775 | 0.229 |
| Completed arrivals | 8.0 | 0.0 |
| On-time arrivals | 6.2 | 1.83 |
| Mission success rate | 1.0 | 0.0 |
| Average delay hours | 7.15 | 0.94 |
| Port service events | 9.4 | 3.2 |
| Total fuel used | 98.07 | 10.11 |
| Total operations cost (USD) | 372,327.64 | 44,129.35 |

The matching 5-seed sweep showed:

| Metric | Value |
| --- | ---: |
| Mean final mean reward | -13.57 |
| Std final mean reward | 2.92 |
| Mean best mean reward | -8.37 |
| Std best mean reward | 0.59 |
| Mean reward improvement | 7.58 |
| Std reward improvement | 4.94 |

The single-mission benchmark achieved perfect mission completion and no stalled
vessels, but it remained weaker than the continuous result as a final project
outcome. It produced lower on-time performance and lower throughput by
construction, and it was also less stable across seeds.

## Comparative Interpretation

Taken together, the two completed result paths suggest the following:

1. The HMARL controller can learn coherent vessel-port coordination when
   reliable congestion information is available.
2. The continuous environment remains the stronger final task, because it tests
   repeated coordination and sustained throughput rather than one-shot mission
   completion.
3. The single-mission benchmark is still useful as a diagnostic comparison, but
   it should not replace the continuous environment in the final report.

The final project claim should therefore be centered on the continuous +
ground-truth result. The single-mission benchmark can be presented as supporting
evidence that the controller can solve a simpler controlled version of the
problem, but not as the main project endpoint.

## Report-Ready Final Interpretation

The final results show that the HMARL controller can learn stable continuous
coordination behavior in the current 8-vessel / 5-port simulator when
ground-truth congestion information is provided. In the main continuous
configuration, the controller achieved strong on-time performance, meaningful
throughput, zero stalled vessels, and consistent improvement across five seeds.
The accompanying single-mission benchmark confirmed that the controller can also
solve a simpler one-mission setting, but that benchmark was less representative
of the intended maritime scheduling problem and produced weaker overall
scheduling quality than the continuous result.

## Recommended Scope Wording

The final results should be presented with two explicit scope constraints:

1. The simulator is synthetic and uses an 8-vessel / 5-port environment rather
   than a real maritime network.
2. The final result uses ground-truth congestion information, so the claim is
   about control quality under reliable forecasts rather than robustness to
   forecasting error.

## Referenced Sources

- `docs/reports/2026-04-10_final-ground-truth-results.md`
- `docs/reports/2026-04-10_final-single-mission-ground-truth-results.md`
- `runs/final_local_2026-04-09/main_seed42_artifacts/eval_result.json`
- `runs/final_local_2026-04-09/main_multiseed/experiment_summary.json`
- `runs/final_single_mission_ground_truth_2026-04-10/single_mission_seed42_artifacts/eval_result.json`
- `runs/final_single_mission_ground_truth_2026-04-10/single_mission_multiseed/experiment_summary.json`
