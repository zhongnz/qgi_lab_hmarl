"""Checkpoint management and early-stopping for MAPPO training.

Provides:

* ``TrainingCheckpoint`` — periodic model saving with best-model tracking.
* ``EarlyStopping`` — patience-based training termination.

Both are designed to be called once per training iteration inside the
standard ``collect_rollout() → update()`` loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mappo import MAPPOTrainer


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


@dataclass
class TrainingCheckpoint:
    """Periodic model saver with best-model tracking.

    Usage::

        ckpt = TrainingCheckpoint(save_dir="outputs/mappo")
        for it in range(1, n_iter + 1):
            rollout = trainer.collect_rollout()
            update  = trainer.update()
            ckpt.step(trainer, iteration=it, metric=rollout["mean_reward"])
    """

    save_dir: str = "outputs/mappo"
    save_every: int = 25
    metric_name: str = "mean_reward"
    higher_is_better: bool = True

    # Internal state
    _best_metric: float = field(init=False, default=float("-inf"))
    _best_iteration: int = field(init=False, default=0)
    _history: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.higher_is_better:
            self._best_metric = float("inf")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def step(
        self,
        trainer: MAPPOTrainer,
        iteration: int,
        metric: float,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Record *metric* and optionally save models.

        Returns ``True`` when a new best model is saved.
        """
        is_best = False
        if self.higher_is_better:
            is_best = metric > self._best_metric
        else:
            is_best = metric < self._best_metric

        if is_best:
            self._best_metric = metric
            self._best_iteration = iteration
            trainer.save_models(str(Path(self.save_dir) / "best_model"))

        # Periodic checkpoint
        if self.save_every > 0 and iteration % self.save_every == 0:
            trainer.save_models(str(Path(self.save_dir) / f"ckpt_{iteration:06d}"))

        # Record history
        row: dict[str, Any] = {
            "iteration": iteration,
            self.metric_name: metric,
            "is_best": is_best,
        }
        if extra:
            row.update(extra)
        self._history.append(row)

        return is_best

    @property
    def best_metric(self) -> float:
        return self._best_metric

    @property
    def best_iteration(self) -> int:
        return self._best_iteration

    def save_history(self, path: str | Path | None = None) -> Path:
        """Write checkpoint history to JSON."""
        out = Path(path) if path else Path(self.save_dir) / "ckpt_history.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self._history, f, indent=2)
        return out

    def load_best(self, trainer: MAPPOTrainer) -> None:
        """Reload the best model weights into *trainer*."""
        trainer.load_models(str(Path(self.save_dir) / "best_model"))

    def cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """Remove all but the *keep_last* most recent periodic checkpoints."""
        ckpt_dir = Path(self.save_dir)
        # Find all checkpoint prefixes (e.g. ckpt_000025_vessel.pt)
        ckpt_files = sorted(ckpt_dir.glob("ckpt_*_*.pt"))
        # Group by prefix
        prefixes: dict[str, list[Path]] = {}
        for f in ckpt_files:
            # ckpt_000025_vessel.pt -> ckpt_000025
            parts = f.stem.split("_")
            if len(parts) >= 2:
                prefix = "_".join(parts[:2])
                prefixes.setdefault(prefix, []).append(f)

        sorted_prefixes = sorted(prefixes.keys())
        to_remove = sorted_prefixes[:-keep_last] if len(sorted_prefixes) > keep_last else []
        for prefix in to_remove:
            for f in prefixes[prefix]:
                f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopping:
    """Patience-based early stopping.

    Tracks a metric over iterations and signals when training should stop.

    Usage::

        es = EarlyStopping(patience=20)
        for it in range(1, n_iter + 1):
            ...
            if es.step(metric=rollout["mean_reward"]):
                print("Early stopping triggered")
                break
    """

    patience: int = 20
    min_delta: float = 0.0
    higher_is_better: bool = True

    _best: float = field(init=False, default=float("-inf"))
    _wait: int = field(init=False, default=0)
    _stopped_iteration: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.higher_is_better:
            self._best = float("inf")

    def step(self, metric: float, iteration: int = 0) -> bool:
        """Record *metric*; return ``True`` when patience is exhausted."""
        improved = False
        if self.higher_is_better:
            improved = metric > self._best + self.min_delta
        else:
            improved = metric < self._best - self.min_delta

        if improved:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1

        if self._wait >= self.patience:
            self._stopped_iteration = iteration
            return True
        return False

    @property
    def wait_count(self) -> int:
        return self._wait

    @property
    def stopped_iteration(self) -> int:
        return self._stopped_iteration

    def reset(self) -> None:
        """Reset state for a new training run."""
        self._best = float("-inf") if self.higher_is_better else float("inf")
        self._wait = 0
        self._stopped_iteration = 0
