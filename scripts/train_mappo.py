#!/usr/bin/env python3
"""CLI script for MAPPO training on the maritime HMARL environment.

Usage::

    python scripts/train_mappo.py --iterations 50 --rollout-length 64
    python scripts/train_mappo.py --iterations 200 --lr 1e-4 --hidden 128 128
    python scripts/train_mappo.py --iterations 300 --lr-end 1e-5 --patience 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hmarl_mvp.checkpointing import EarlyStopping, TrainingCheckpoint
from hmarl_mvp.config import get_default_config
from hmarl_mvp.mappo import MAPPOConfig, MAPPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPPO training for HMARL maritime.")
    p.add_argument("--iterations", type=int, default=50, help="Training iterations")
    p.add_argument("--rollout-length", type=int, default=64, help="Steps per rollout")
    p.add_argument("--num-epochs", type=int, default=4, help="PPO epochs per update")
    p.add_argument("--minibatch-size", type=int, default=32, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy bonus")
    p.add_argument("--value-clip-eps", type=float, default=0.2, help="Value clip epsilon (0 = disabled)")
    p.add_argument("--lr-end", type=float, default=0.0, help="Final LR for linear annealing (0 = constant)")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64], help="Hidden dims")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--eval-interval", type=int, default=10, help="Eval every N iterations")
    p.add_argument("--save-dir", type=str, default="outputs/mappo", help="Output directory")
    p.add_argument("--save-every", type=int, default=25, help="Periodic checkpoint interval (0 = off)")
    p.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 = off)")
    p.add_argument("--num-ports", type=int, default=5)
    p.add_argument("--num-vessels", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env_config = get_default_config(
        num_ports=args.num_ports,
        num_vessels=args.num_vessels,
        rollout_steps=args.rollout_length + 10,
    )

    mappo_cfg = MAPPOConfig(
        rollout_length=args.rollout_length,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        value_clip_eps=args.value_clip_eps,
        lr=args.lr,
        lr_end=args.lr_end,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coeff=args.entropy_coeff,
        hidden_dims=args.hidden,
        total_iterations=args.iterations,
    )

    print("=" * 60)
    print("MAPPO Training — HMARL Maritime")
    print("=" * 60)
    print(f"  iterations:     {args.iterations}")
    print(f"  rollout_length: {args.rollout_length}")
    print(f"  lr:             {args.lr} → {args.lr_end}")
    print(f"  value_clip_eps: {args.value_clip_eps}")
    print(f"  hidden_dims:    {args.hidden}")
    print(f"  ports/vessels:  {args.num_ports}/{args.num_vessels}")
    print(f"  save_dir:       {save_dir}")
    if args.patience > 0:
        print(f"  early_stop:     patience={args.patience}")
    if args.save_every > 0:
        print(f"  checkpointing:  every {args.save_every} iterations")
    print("=" * 60)

    trainer = MAPPOTrainer(
        env_config=env_config,
        mappo_config=mappo_cfg,
        seed=args.seed,
    )

    log: list[dict[str, float]] = []
    t0 = time.time()

    # Checkpoint manager
    ckpt = TrainingCheckpoint(
        save_dir=str(save_dir),
        save_every=args.save_every,
    )

    # Early stopping
    es = EarlyStopping(patience=args.patience) if args.patience > 0 else None

    stopped_early = False
    for iteration in range(1, args.iterations + 1):
        rollout_info = trainer.collect_rollout()
        update_info = trainer.update()

        row: dict[str, float] = {
            "iteration": float(iteration),
            "mean_reward": rollout_info["mean_reward"],
            "total_reward": rollout_info["total_reward"],
            "lr": trainer.current_lr,
        }
        for agent_type, result in update_info.items():
            row[f"{agent_type}_policy_loss"] = result.policy_loss
            row[f"{agent_type}_value_loss"] = result.value_loss
            row[f"{agent_type}_entropy"] = result.entropy
            row[f"{agent_type}_clip_frac"] = result.clip_fraction
            row[f"{agent_type}_grad_norm"] = result.grad_norm
            row[f"{agent_type}_weight_norm"] = result.weight_norm

        log.append(row)

        # Checkpoint
        is_best = ckpt.step(
            trainer,
            iteration=iteration,
            metric=rollout_info["mean_reward"],
            extra={"lr": trainer.current_lr},
        )

        # Early stopping check
        if es is not None and es.step(rollout_info["mean_reward"], iteration=iteration):
            print(f"\n  Early stopping at iteration {iteration} (patience={args.patience})")
            stopped_early = True

        if iteration % args.eval_interval == 0 or iteration == 1:
            eval_metrics = trainer.evaluate(num_steps=args.rollout_length)
            elapsed = time.time() - t0
            best_tag = " *BEST*" if is_best else ""
            print(
                f"  iter {iteration:4d} | "
                f"train_r={rollout_info['mean_reward']:+.3f} | "
                f"eval_r={eval_metrics['total_reward']:+.1f} | "
                f"v_loss={update_info['vessel'].value_loss:.4f} | "
                f"p_loss={update_info['vessel'].policy_loss:.4f} | "
                f"clip={update_info['vessel'].clip_fraction:.3f} | "
                f"lr={trainer.current_lr:.2e} | "
                f"time={elapsed:.1f}s{best_tag}"
            )

        if stopped_early:
            break

    # Save results
    trainer.save_models(str(save_dir / "model"))
    ckpt.save_history()

    # Save training log
    log_path = save_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # Save reward curve as CSV
    rewards = trainer.reward_history
    reward_path = save_dir / "reward_curve.csv"
    with open(reward_path, "w") as f:
        f.write("iteration,mean_reward\n")
        for i, r in enumerate(rewards, 1):
            f.write(f"{i},{r:.6f}\n")

    elapsed = time.time() - t0
    print("=" * 60)
    if stopped_early:
        print(f"Training stopped early at iteration {iteration} (patience={args.patience})")
    else:
        print(f"Training complete — {iteration} iterations in {elapsed:.1f}s")
    print(f"  Final mean reward: {rewards[-1]:+.4f}")
    print(f"  Best mean reward:  {ckpt.best_metric:+.4f} (iter {ckpt.best_iteration})")
    if len(rewards) >= 10:
        recent = float(np.mean(rewards[-10:]))
        early = float(np.mean(rewards[:10]))
        print(f"  First-10 avg:      {early:+.4f}")
        print(f"  Last-10 avg:       {recent:+.4f}")
    print(f"  Models saved to:   {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
