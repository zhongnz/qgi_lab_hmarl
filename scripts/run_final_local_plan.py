#!/usr/bin/env python3
"""Run the final local HMARL experiment plan end to end.

This script packages the recommended local workflow for the current project:

1. A single-seed continuous + ground_truth "hero" run with full artifacts
2. A multi-seed continuous + ground_truth run for the main quantitative result
3. A compact summary manifest

It is intentionally conservative and CPU-friendly so the whole plan can be run
on the local machine without HPC.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from hmarl_mvp.experiment import run_trained_mappo_trace
from hmarl_mvp.experiment_config import ExperimentConfig, run_from_config
from hmarl_mvp.logger import TrainingLogger
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer
from hmarl_mvp.plotting import plot_time_series_diagnostics, plot_training_curves
from hmarl_mvp.report import generate_training_report


def _parse_seed_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _artifact_env_config(*, episode_mode: str, forecast_source: str, rollout_length: int) -> dict[str, Any]:
    return {
        "num_vessels": 8,
        "num_ports": 5,
        "rollout_steps": rollout_length + 5,
        "episode_mode": episode_mode,
        "mission_success_on": "arrival",
        "forecast_source": forecast_source,
    }


def _run_single_training(
    *,
    out_dir: Path,
    title: str,
    env_cfg: dict[str, Any],
    rollout_length: int,
    iterations: int,
    eval_interval: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    logger = TrainingLogger(
        log_dir=str(out_dir / "logs"),
        experiment_name=title,
        console=True,
        print_every=max(1, iterations // 20),
    )
    mappo_cfg = MAPPOConfig(
        rollout_length=rollout_length,
        lr=3e-4,
        entropy_coeff=0.01,
        total_iterations=iterations,
        device=device,
    )

    trainer = MAPPOTrainer(env_config=env_cfg, mappo_config=mappo_cfg, seed=seed)
    t0 = time.time()
    history = trainer.train(
        num_iterations=iterations,
        eval_interval=eval_interval,
        log_fn=logger.log,
        checkpoint_dir=str(ckpt_dir),
        early_stopping_patience=0,
    )
    elapsed = time.time() - t0
    logger.close()

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "train_history.csv", index=False)

    trainer.save_models(str(out_dir / "final_model"))

    eval_result = trainer.evaluate_episodes(num_episodes=5)
    (out_dir / "eval_result.json").write_text(json.dumps(eval_result, indent=2, default=str))

    trace_result = run_trained_mappo_trace(
        trainer,
        num_steps=env_cfg["rollout_steps"],
        return_logs=True,
    )
    if isinstance(trace_result, tuple):
        eval_trace, eval_action_trace, eval_event_log = trace_result
    else:
        eval_trace = trace_result
        eval_action_trace = pd.DataFrame()
        eval_event_log = pd.DataFrame()
    eval_trace.to_csv(out_dir / "eval_trace.csv", index=False)
    eval_action_trace.to_csv(out_dir / "eval_action_trace.csv", index=False)
    eval_event_log.to_csv(out_dir / "eval_event_log.csv", index=False)

    report = generate_training_report(
        history=history,
        eval_result=eval_result,
        config={"env": env_cfg, "mappo": mappo_cfg.__dict__},
        elapsed_seconds=elapsed,
        title=title,
    )
    (out_dir / "report.md").write_text(report)

    plot_training_curves(history_df, out_path=str(out_dir / "training_curves.png"))
    plot_time_series_diagnostics(
        eval_trace,
        out_path=str(out_dir / "diagnostics_trace.png"),
        column_group="aggregate",
    )
    plot_time_series_diagnostics(
        eval_trace,
        out_path=str(out_dir / "diagnostics_trace_vessels.png"),
        column_group="vessel",
    )
    plot_time_series_diagnostics(
        eval_trace,
        out_path=str(out_dir / "diagnostics_trace_ports.png"),
        column_group="port",
    )
    plot_time_series_diagnostics(
        eval_trace,
        out_path=str(out_dir / "diagnostics_trace_coordinators.png"),
        column_group="coordinator",
    )

    return {
        "out_dir": str(out_dir),
        "elapsed_seconds": elapsed,
        "final_mean_reward": float(history[-1]["mean_reward"]) if history else 0.0,
        "eval_total_reward": float(eval_result["mean"]["total_reward"]),
        "eval_on_time_rate": float(eval_result["mean"]["on_time_rate"]),
    }


def _run_multiseed_experiment(
    *,
    out_dir: Path,
    name: str,
    description: str,
    env_cfg: dict[str, Any],
    rollout_length: int,
    iterations: int,
    seeds: list[int],
) -> dict[str, Any]:
    cfg = ExperimentConfig(
        name=name,
        description=description,
        tags=["final", "local", name],
        env=env_cfg,
        mappo={
            "lr": 3e-4,
            "rollout_length": rollout_length,
            "num_epochs": 4,
            "minibatch_size": 128,
            "clip_eps": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "hidden_dims": [64, 64],
            "normalize_observations": True,
            "normalize_rewards": True,
        },
        num_iterations=iterations,
        num_seeds=len(seeds),
        seeds=seeds,
        eval_interval=10 if iterations >= 50 else 5,
        output_dir=str(out_dir),
        tensorboard=False,
        console_log=True,
        log_every=10,
    )
    return run_from_config(cfg)


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _write_manifest(out_root: Path) -> Path:
    main_multiseed_summary = out_root / "main_multiseed" / "summary.csv"
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(out_root),
        "artifacts": {
            "main_seed42": str(out_root / "main_seed42_artifacts"),
            "main_multiseed": str(out_root / "main_multiseed"),
        },
        "main_multiseed_summary": _read_summary_csv(main_multiseed_summary)
        if main_multiseed_summary.exists()
        else [],
    }
    manifest_path = out_root / "final_run_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))
    return manifest_path


def _print_plan(args: argparse.Namespace, out_root: Path) -> None:
    print("Final local run plan")
    print(f"  output root: {out_root}")
    print(f"  device: {args.device}")
    print(f"  main iterations: {args.main_iterations}")
    print(f"  main seeds: {_parse_seed_list(args.main_seeds)}")
    print("")
    print("Stages")
    print(f"  1. main_seed42_artifacts        -> {out_root / 'main_seed42_artifacts'}")
    print(f"  2. main_multiseed              -> {out_root / 'main_multiseed'}")
    print(f"  3. final_run_manifest.json     -> {out_root / 'final_run_manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final local HMARL experiment plan.")
    parser.add_argument(
        "--output-root",
        default="runs/final_local_2026-04-09",
        help="Directory to store all final local experiment outputs.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--main-iterations", type=int, default=100)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument(
        "--main-seeds",
        default="42,49,56,63,70",
        help="Comma-separated seeds for the main continuous + ground_truth multi-seed run.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "main_artifacts",
            "main_multiseed",
            "summary",
        ],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without executing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _print_plan(args, out_root)
        return

    main_env = _artifact_env_config(
        episode_mode="continuous",
        forecast_source="ground_truth",
        rollout_length=args.rollout_length,
    )
    main_seeds = _parse_seed_list(args.main_seeds)

    if args.stage in {"all", "main_artifacts"}:
        _run_single_training(
            out_dir=out_root / "main_seed42_artifacts",
            title="Final Local Main Continuous + Ground Truth Run",
            env_cfg=main_env,
            rollout_length=args.rollout_length,
            iterations=args.main_iterations,
            eval_interval=10,
            seed=main_seeds[0],
            device=args.device,
        )

    if args.stage in {"all", "main_multiseed"}:
        _run_multiseed_experiment(
            out_dir=out_root / "main_multiseed",
            name="final_main_ground_truth_multiseed",
            description="Final continuous + ground_truth main-result run on the local machine.",
            env_cfg=main_env,
            rollout_length=args.rollout_length,
            iterations=args.main_iterations,
            seeds=main_seeds,
        )

    if args.stage in {"all", "summary"}:
        manifest = _write_manifest(out_root)
        print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
