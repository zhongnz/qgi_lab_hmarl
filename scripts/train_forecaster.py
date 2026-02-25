#!/usr/bin/env python3
"""Train a learned queue forecaster on simulator rollout data.

Usage:
    python scripts/train_forecaster.py --episodes 20 --steps 40 --epochs 200
    python scripts/train_forecaster.py --output-dir runs/forecaster
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmarl_mvp.config import get_default_config
from hmarl_mvp.learned_forecaster import (
    LearnedForecaster,
    build_forecast_dataset,
    collect_queue_traces,
    train_forecaster,
)
from hmarl_mvp.metrics import forecast_mae, forecast_rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned queue forecaster.")
    parser.add_argument("--episodes", type=int, default=20, help="Training rollout episodes.")
    parser.add_argument("--steps", type=int, default=40, help="Steps per episode.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (steps).")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/forecaster",
        help="Directory for model and evaluation outputs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch loss.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_default_config()
    num_ports = cfg["num_ports"]
    horizon = args.horizon

    print(f"Collecting {args.episodes} rollout episodes ({args.steps} steps each)...")
    traces = collect_queue_traces(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        config=cfg,
        seed=args.seed,
    )
    print(f"  collected {len(traces)} episodes, "
          f"~{sum(len(ep) for ep in traces)} total snapshots")

    print(f"Building dataset (horizon={horizon})...")
    dataset = build_forecast_dataset(traces, horizon=horizon, features_per_port=3)
    print(f"  dataset size: {len(dataset)} samples")
    if len(dataset) == 0:
        print("ERROR: empty dataset — increase episodes or steps")
        sys.exit(1)

    forecaster = LearnedForecaster(
        num_ports=num_ports,
        horizon=horizon,
        hidden_dims=[128, 64],
    )

    print(f"Training for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr})...")
    result = train_forecaster(
        forecaster=forecaster,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=args.verbose,
    )
    print(f"  final val loss: {result.final_loss:.4f}")

    model_path = out_dir / "forecaster_model.pt"
    forecaster.save(str(model_path))
    print(f"  saved model to {model_path}")

    # Quick evaluation against oracle
    print("\nEvaluation on held-out rollout...")
    eval_traces = collect_queue_traces(
        num_episodes=3,
        steps_per_episode=args.steps,
        config=cfg,
        seed=args.seed + 1000,
    )
    eval_dataset = build_forecast_dataset(eval_traces, horizon=horizon, features_per_port=3)
    if len(eval_dataset) > 0:
        import torch

        inputs_t, targets_t = eval_dataset.to_tensors()
        with torch.no_grad():
            preds = forecaster.net(inputs_t).cpu().numpy()
        targets_np = targets_t.cpu().numpy()
        mae = forecast_mae(preds, targets_np)
        rmse = forecast_rmse(preds, targets_np)
        print(f"  eval MAE:  {mae:.4f}")
        print(f"  eval RMSE: {rmse:.4f}")

        # Save eval summary
        import pandas as pd

        eval_summary = pd.DataFrame(
            {
                "metric": ["final_val_loss", "eval_mae", "eval_rmse", "train_samples"],
                "value": [result.final_loss, mae, rmse, result.num_samples],
            }
        )
        eval_path = out_dir / "forecaster_eval.csv"
        eval_summary.to_csv(eval_path, index=False)
        print(f"  saved eval to {eval_path}")

    # Save loss curve
    import pandas as pd

    loss_df = pd.DataFrame(
        {"epoch": list(range(len(result.epoch_losses))), "val_loss": result.epoch_losses}
    )
    loss_df.to_csv(out_dir / "forecaster_loss_curve.csv", index=False)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
