"""Population-Based Training (PBT) for MAPPO.

Implements a lightweight **sequential** PBT orchestrator on top of
``MAPPOTrainer``.  Workers take turns training for *interval*
iterations each, then the bottom-performing fraction copies weights
from randomly selected top performers and perturbs hyperparameters.

Sequential PBT produces identical learning dynamics to parallel PBT —
only wall-clock time differs.

Design
------
* Workers are full ``MAPPOTrainer`` instances with independent seeds,
  environments, and normaliser state.
* Built-in LR / entropy schedules are **disabled** on PBT workers
  (``total_iterations=0``, ``entropy_coeff_end=None``).  PBT manages
  these hyperparameters directly via exploit/explore.
* **Exploit**: bottom workers copy weights, normaliser state, and
  hyperparameters from a randomly selected top worker.  Optimiser
  momentum is reset.
* **Explore**: copied hyperparameters (``lr``, ``entropy_coeff``,
  ``clip_eps``) are randomly multiplied or divided by
  ``perturb_factor``, then clamped to configured bounds.
* Memory: each worker is ~60 MB (tiny networks + Python env).
  4 workers ≈ 240 MB — well within 7.5 GiB.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .mappo import MAPPOConfig, MAPPOTrainer

logger = logging.getLogger(__name__)


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training."""

    population_size: int = 4
    interval: int = 10  # iterations between exploit/explore rounds
    fraction_top: float = 0.25  # top fraction eligible as exploit source
    fraction_bottom: float = 0.25  # bottom fraction gets replaced

    # Perturbation
    perturb_factor: float = 1.2  # multiply or divide by this
    perturb_lr: bool = True
    perturb_entropy: bool = True
    perturb_clip_eps: bool = True

    # Hyperparameter bounds
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    entropy_min: float = 0.001
    entropy_max: float = 0.1
    clip_eps_min: float = 0.1
    clip_eps_max: float = 0.3


class PBTTrainer:
    """Sequential Population-Based Training orchestrator.

    All workers share the same environment configuration but have
    different random seeds, producing diverse initial trajectories.
    After every ``interval`` iterations, the bottom workers are
    replaced by perturbed copies of top workers.

    Usage::

        pbt = PBTTrainer(
            env_config={...},
            mappo_config=MAPPOConfig(hidden_dims=[64, 64]),
            pbt_config=PBTConfig(population_size=4),
        )
        result = pbt.train(total_iterations=200)
        best = pbt.best_worker  # MAPPOTrainer with best performance
    """

    def __init__(
        self,
        env_config: dict[str, Any] | None = None,
        mappo_config: MAPPOConfig | None = None,
        pbt_config: PBTConfig | None = None,
        base_seed: int = 42,
    ) -> None:
        self.pbt_cfg = pbt_config or PBTConfig()
        self.env_config = dict(env_config or {})
        base_cfg = mappo_config or MAPPOConfig()
        self._rng = np.random.default_rng(base_seed)

        # Create population — each worker gets a unique seed
        self.workers: list[MAPPOTrainer] = []
        self._worker_seeds: list[int] = []
        for i in range(self.pbt_cfg.population_size):
            cfg = copy.deepcopy(base_cfg)
            # Disable built-in scheduling — PBT manages hyperparams directly
            cfg.total_iterations = 0
            cfg.entropy_coeff_end = None
            worker_seed = base_seed + i * 1000
            worker = MAPPOTrainer(
                env_config=self.env_config,
                mappo_config=cfg,
                seed=worker_seed,
            )
            self.workers.append(worker)
            self._worker_seeds.append(worker_seed)

        # Per-worker tracking
        self._rewards: list[list[float]] = [
            [] for _ in range(self.pbt_cfg.population_size)
        ]
        self._hyperparam_history: list[list[dict[str, float]]] = [
            [] for _ in range(self.pbt_cfg.population_size)
        ]
        self._exploit_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_iterations: int,
        log_fn: Any | None = None,
        checkpoint_dir: str | None = None,
    ) -> dict[str, Any]:
        """Run sequential PBT training.

        Parameters
        ----------
        total_iterations:
            Number of collect+update iterations **per worker**.
        log_fn:
            Optional callback
            ``(round_idx, worker_idx, iteration, rollout_info) -> None``.
        checkpoint_dir:
            If set, save best worker's model after each exploit/explore
            round.

        Returns
        -------
        dict with ``best_worker_idx``, ``best_mean_reward``,
        ``per_worker_rewards``, ``hyperparam_history``,
        ``exploit_log``, ``total_rounds``, ``total_time``,
        ``final_hyperparams``.
        """
        import os
        import time

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        interval = self.pbt_cfg.interval
        num_rounds = total_iterations // interval
        remainder = total_iterations % interval

        start_time = time.perf_counter()

        for round_idx in range(num_rounds):
            # Train each worker sequentially for `interval` iterations
            for w_idx, worker in enumerate(self.workers):
                for local_it in range(interval):
                    global_it = round_idx * interval + local_it
                    rollout = worker.collect_rollout()
                    worker.update()
                    self._rewards[w_idx].append(rollout["mean_reward"])
                    self._hyperparam_history[w_idx].append(
                        self._get_hyperparams(w_idx)
                    )
                    if log_fn is not None:
                        log_fn(round_idx, w_idx, global_it, rollout)

            # Exploit + Explore (skip after final round)
            if round_idx < num_rounds - 1:
                self._exploit_and_explore(round_idx)

            # Checkpoint best worker
            if checkpoint_dir is not None:
                best_idx = self._best_worker_idx()
                prefix = os.path.join(checkpoint_dir, f"pbt_best_r{round_idx:04d}")
                self.workers[best_idx].save_models(prefix)

            logger.info(
                "PBT round %d/%d — best worker %d (%.4f)",
                round_idx + 1,
                num_rounds,
                self._best_worker_idx(),
                self._recent_mean(self._best_worker_idx()),
            )

        # Train any remaining iterations (no exploit/explore after)
        if remainder > 0:
            for w_idx, worker in enumerate(self.workers):
                for _ in range(remainder):
                    rollout = worker.collect_rollout()
                    worker.update()
                    self._rewards[w_idx].append(rollout["mean_reward"])
                    self._hyperparam_history[w_idx].append(
                        self._get_hyperparams(w_idx)
                    )

        total_time = time.perf_counter() - start_time
        best_idx = self._best_worker_idx()

        return {
            "best_worker_idx": best_idx,
            "best_mean_reward": self._recent_mean(best_idx),
            "per_worker_rewards": [list(r) for r in self._rewards],
            "hyperparam_history": [list(h) for h in self._hyperparam_history],
            "exploit_log": list(self._exploit_log),
            "total_rounds": num_rounds,
            "total_time": total_time,
            "final_hyperparams": [
                self._get_hyperparams(i) for i in range(len(self.workers))
            ],
        }

    # ------------------------------------------------------------------
    # Exploit / explore
    # ------------------------------------------------------------------

    def _exploit_and_explore(self, round_idx: int) -> None:
        """Replace bottom workers with perturbed copies of top workers."""
        n = len(self.workers)
        n_top = max(1, int(n * self.pbt_cfg.fraction_top))
        n_bottom = max(1, int(n * self.pbt_cfg.fraction_bottom))

        # Rank by recent mean reward (descending)
        means = [self._recent_mean(i) for i in range(n)]
        ranked = list(np.argsort(means)[::-1])  # best first

        top_indices = ranked[:n_top]
        bottom_indices = ranked[-n_bottom:]

        events: list[dict[str, Any]] = []

        for bot_idx in bottom_indices:
            if bot_idx in top_indices:
                continue  # don't replace a top worker with itself
            source_idx = int(self._rng.choice(top_indices))
            old_hp = self._get_hyperparams(bot_idx)
            self._copy_weights(source_idx, bot_idx)
            self._perturb_hyperparams(bot_idx)
            new_hp = self._get_hyperparams(bot_idx)
            events.append(
                {
                    "round": round_idx,
                    "target": int(bot_idx),
                    "source": int(source_idx),
                    "old_hyperparams": old_hp,
                    "new_hyperparams": new_hp,
                    "target_reward": means[bot_idx],
                    "source_reward": means[source_idx],
                }
            )
            logger.debug(
                "PBT exploit: worker %d (%.4f) <- worker %d (%.4f), hp: %s",
                bot_idx,
                means[bot_idx],
                source_idx,
                means[source_idx],
                new_hp,
            )

        self._exploit_log.append(
            {
                "round": round_idx,
                "ranking": [int(r) for r in ranked],
                "means": [float(m) for m in means],
                "events": events,
            }
        )

    def _copy_weights(self, source_idx: int, target_idx: int) -> None:
        """Copy network weights, normaliser state, and hyperparams."""
        src = self.workers[source_idx]
        tgt = self.workers[target_idx]

        # Copy actor-critic state dicts
        for name in src.actor_critics:
            sd = src.actor_critics[name].state_dict()
            tgt.actor_critics[name].load_state_dict(sd)

        # Copy reward normalisers
        for name in src._reward_normalizers:
            tgt._reward_normalizers[name].mean = src._reward_normalizers[name].mean
            tgt._reward_normalizers[name].var = src._reward_normalizers[name].var
            tgt._reward_normalizers[name].count = src._reward_normalizers[name].count

        # Copy observation normalisers
        for name in src._obs_normalizers:
            tgt._obs_normalizers[name].mean = src._obs_normalizers[name].mean.copy()
            tgt._obs_normalizers[name].var = src._obs_normalizers[name].var.copy()
            tgt._obs_normalizers[name].count = src._obs_normalizers[name].count

        # Copy hyperparameters from source
        tgt.mappo_cfg.entropy_coeff = src.mappo_cfg.entropy_coeff
        tgt.mappo_cfg.clip_eps = src.mappo_cfg.clip_eps

        # Reset optimisers with source's current LR (fresh momentum)
        src_lr = src.current_lr
        for name in tgt.actor_critics:
            tgt.optimizers[name] = torch.optim.Adam(
                tgt.actor_critics[name].parameters(),
                lr=src_lr,
                weight_decay=tgt.mappo_cfg.weight_decay,
            )

    def _perturb_hyperparams(self, worker_idx: int) -> None:
        """Randomly perturb tunable hyperparameters.

        Each parameter is independently multiplied or divided by
        ``perturb_factor`` with equal probability, then clamped to
        configured bounds.
        """
        worker = self.workers[worker_idx]
        cfg = self.pbt_cfg
        factor = cfg.perturb_factor

        if cfg.perturb_lr:
            mult = float(self._rng.choice([1.0 / factor, factor]))
            new_lr = float(np.clip(worker.current_lr * mult, cfg.lr_min, cfg.lr_max))
            for opt in worker.optimizers.values():
                for pg in opt.param_groups:
                    pg["lr"] = new_lr

        if cfg.perturb_entropy:
            mult = float(self._rng.choice([1.0 / factor, factor]))
            new_ent = float(
                np.clip(
                    worker.mappo_cfg.entropy_coeff * mult,
                    cfg.entropy_min,
                    cfg.entropy_max,
                )
            )
            worker.mappo_cfg.entropy_coeff = new_ent

        if cfg.perturb_clip_eps:
            mult = float(self._rng.choice([1.0 / factor, factor]))
            new_eps = float(
                np.clip(
                    worker.mappo_cfg.clip_eps * mult,
                    cfg.clip_eps_min,
                    cfg.clip_eps_max,
                )
            )
            worker.mappo_cfg.clip_eps = new_eps

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def best_worker(self) -> MAPPOTrainer:
        """Return the worker with the highest recent mean reward."""
        return self.workers[self._best_worker_idx()]

    def _best_worker_idx(self) -> int:
        """Index of the worker with the highest recent mean reward."""
        means = [self._recent_mean(i) for i in range(len(self.workers))]
        return int(np.argmax(means))

    def _recent_mean(self, worker_idx: int, window: int | None = None) -> float:
        """Mean reward over the last ``window`` iterations for a worker."""
        window = window or self.pbt_cfg.interval
        rews = self._rewards[worker_idx]
        if not rews:
            return float("-inf")
        return float(np.mean(rews[-window:]))

    def _get_hyperparams(self, worker_idx: int) -> dict[str, float]:
        """Snapshot current hyperparameters for a worker."""
        w = self.workers[worker_idx]
        return {
            "lr": w.current_lr,
            "entropy_coeff": w.mappo_cfg.entropy_coeff,
            "clip_eps": w.mappo_cfg.clip_eps,
        }
