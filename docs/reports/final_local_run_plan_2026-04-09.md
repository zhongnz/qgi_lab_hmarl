# Final Local Run Plan

This is the exact local run plan to complete the current HMARL project end to end
without relying on HPC.

## Goal

Produce:

- one main continuous + ground-truth artifact run with full logs, plots, trace, and report
- one main continuous + ground-truth multi-seed result for the final quantitative claim
- one machine-readable manifest tying the outputs together

## Canonical command

From the repo root:

```bash
cd qgi_lab_hmarl
/home/ptz/dev/hmarl/qgi_lab_hmarl/.conda/bin/python scripts/run_final_local_plan.py --device cpu
```

This writes to:

```text
runs/final_local_2026-04-09/
```

## Stages

### 1. Main artifact run

Output:

```text
runs/final_local_2026-04-09/main_seed42_artifacts/
```

Settings:

- `episode_mode = continuous`
- `forecast_source = ground_truth`
- `num_vessels = 8`
- `num_ports = 5`
- `rollout_length = 64`
- `rollout_steps = 69`
- `iterations = 100`
- `seed = 42`

Purpose:

- final plots
- final trace
- final `report.md`
- final saved model
- main result with forecast error removed

### 2. Main multi-seed run

Output:

```text
runs/final_local_2026-04-09/main_multiseed/
```

Settings:

- same environment as the main artifact run
- `iterations = 100`
- seeds: `42, 49, 56, 63, 70`

Purpose:

- final quantitative evidence for the main continuous + ground-truth result
- `summary.csv`
- `experiment_summary.json`

### 3. Final manifest

Output:

```text
runs/final_local_2026-04-09/final_run_manifest.json
```

Purpose:

- one file pointing to the final main outputs
- captures the multi-seed summaries in one place

## Expected local runtime

Based on the saved local runs already in the repo:

- main artifact run: about `6` to `8` minutes
- main multi-seed run (`5 x 100` iterations): about `30` to `40` minutes
Total expected wall-clock time:

- about `35` to `45` minutes on the local machine

## Dry-run mode

To print the plan without running anything:

```bash
cd qgi_lab_hmarl
/home/ptz/dev/hmarl/qgi_lab_hmarl/.conda/bin/python scripts/run_final_local_plan.py --dry-run
```

## Recommended completion path

1. Run the full local plan.
2. Use `main_seed42_artifacts` for plots, trace, and narrative examples.
3. Use `main_multiseed` as the main quantitative result.
4. Refresh the report/slides from those finalized outputs.

## Current interpretation

This plan assumes we are deliberately using `ground_truth` to reduce forecast
error while finishing the project. In practice that means:

- the main result now focuses on control quality under reliable forecasts
- the continuous environment remains the only final task in the completion path

If we later want an imperfect-forecast comparison, we can add it as a separate
follow-on experiment rather than part of the required completion path.
