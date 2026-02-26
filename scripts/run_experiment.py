#!/usr/bin/env python
"""Run experiments from YAML configuration files.

Usage::

    # Run a single experiment:
    python scripts/run_experiment.py configs/baseline.yaml

    # Run and compare two experiments:
    python scripts/run_experiment.py configs/baseline.yaml configs/no_sharing_ablation.yaml

    # Override output directory:
    python scripts/run_experiment.py configs/baseline.yaml --output-dir runs/custom

    # Quick smoke test (2 iterations):
    python scripts/run_experiment.py configs/baseline.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmarl_mvp.experiment_config import (
    load_experiment_config,
    run_from_config,
)
from hmarl_mvp.stats import multi_method_comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HMARL experiments from YAML configs.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        type=str,
        help="Path(s) to YAML experiment config files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for all experiments.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: override to 2 iterations & 1 seed.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="After running, perform statistical comparison of results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_results: dict[str, dict] = {}
    final_rewards: dict[str, list[float]] = {}

    for config_path in args.configs:
        path = Path(config_path)
        if not path.exists():
            print(f"ERROR: Config file not found: {path}", file=sys.stderr)
            sys.exit(1)

        cfg = load_experiment_config(path)

        # Apply overrides
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.smoke:
            cfg.num_iterations = 2
            cfg.num_seeds = 1
            cfg.seeds = None
            cfg.eval_interval = 0
            cfg.early_stopping_patience = 0

        print(f"\n{'='*60}")
        print(f"  Experiment: {cfg.name}")
        print(f"  Config:     {path}")
        print(f"  Seeds:      {cfg.num_seeds}")
        print(f"  Iterations: {cfg.num_iterations}")
        print(f"  Output:     {cfg.output_dir}")
        print(f"{'='*60}\n")

        t0 = time.perf_counter()
        result = run_from_config(cfg)
        elapsed = time.perf_counter() - t0

        all_results[cfg.name] = result
        print(f"\n  Completed in {elapsed:.1f}s")

        # Extract final rewards per seed for comparison
        summaries = result.get("summaries", [])
        rewards = [s.get("final_mean_reward", 0.0) for s in summaries]
        final_rewards[cfg.name] = rewards

        # Print per-seed summary
        for i, s in enumerate(summaries):
            best = s.get("best_mean_reward", 0.0)
            final = s.get("final_mean_reward", 0.0)
            print(f"  Seed {i}: best={best:.4f}  final={final:.4f}")

        if "aggregate_summary" in result:
            agg = result["aggregate_summary"]
            print(f"\n  Aggregate: mean_best={agg.get('mean_best_mean_reward', 0):.4f}"
                  f"  mean_final={agg.get('mean_final_mean_reward', 0):.4f}")

    # Statistical comparison
    if (args.compare or len(args.configs) > 1) and len(final_rewards) > 1:
        # Only compare methods that have at least 2 seeds
        valid = {k: v for k, v in final_rewards.items() if len(v) >= 2}
        if len(valid) >= 2:
            print(f"\n{'='*60}")
            print("  Statistical Comparison")
            print(f"{'='*60}\n")
            comparison = multi_method_comparison(valid)
            print(comparison["summary"])

            # Save comparison
            first_cfg = load_experiment_config(Path(args.configs[0]))
            out = Path(args.output_dir or first_cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "comparison.json", "w") as f:
                json.dump(
                    {k: v for k, v in comparison.items() if k != "summary"},
                    f,
                    indent=2,
                    default=str,
                )
            print(f"\n  Comparison saved to {out / 'comparison.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
