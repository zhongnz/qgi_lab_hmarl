"""Structured training logger for MAPPO experiments.

Provides a ``TrainingLogger`` that collects per-iteration metrics,
writes JSON-lines logs, and supports basic console output — without
depending on any external logging framework.

Usage::

    logger = TrainingLogger(log_dir="runs/exp_01")
    for it in range(num_iterations):
        ...
        logger.log(it, {"mean_reward": reward, "lr": lr})
    logger.close()
    summary = logger.summary()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np


class TrainingLogger:
    """Lightweight structured logger for MAPPO training runs.

    Parameters
    ----------
    log_dir:
        Directory for log files.  Created if absent.
        If ``None``, logging is in-memory only (no disk I/O).
    experiment_name:
        Human-readable name stored in the log header.
    console:
        If ``True``, print a summary line to stdout every ``print_every``
        iterations.
    print_every:
        Console print interval (iterations).
    """

    def __init__(
        self,
        log_dir: str | None = None,
        experiment_name: str = "mappo_run",
        console: bool = False,
        print_every: int = 10,
    ) -> None:
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.console = console
        self.print_every = max(1, print_every)

        self._entries: list[dict[str, Any]] = []
        self._start_time = time.monotonic()
        self._file: Any = None

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, "train_log.jsonl")
            self._file = open(path, "a")  # noqa: SIM115
            # Write header
            header = {
                "event": "start",
                "experiment": experiment_name,
                "timestamp": time.time(),
            }
            self._file.write(json.dumps(header) + "\n")
            self._file.flush()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log(self, iteration: int, metrics: dict[str, Any]) -> None:
        """Record metrics for one iteration."""
        entry = {
            "iteration": iteration,
            "wall_time": time.monotonic() - self._start_time,
            **metrics,
        }
        self._entries.append(entry)

        if self._file is not None:
            self._file.write(json.dumps(_serializable(entry)) + "\n")
            self._file.flush()

        if self.console and iteration % self.print_every == 0:
            reward = metrics.get("mean_reward", float("nan"))
            lr = metrics.get("lr", float("nan"))
            elapsed = entry["wall_time"]
            print(
                f"[{self.experiment_name}] it={iteration:>5d}  "
                f"reward={reward:+.4f}  lr={lr:.2e}  "
                f"elapsed={elapsed:.1f}s"
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_metric(self, key: str) -> list[float]:
        """Return a list of values for a specific metric across all logged iterations."""
        return [float(e[key]) for e in self._entries if key in e]

    def get_rewards(self) -> list[float]:
        """Shortcut for ``get_metric("mean_reward")``."""
        return self.get_metric("mean_reward")

    @property
    def entries(self) -> list[dict[str, Any]]:
        """Return all logged entries."""
        return list(self._entries)

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics from the logged training run."""
        rewards = self.get_rewards()
        wall_time = time.monotonic() - self._start_time

        if not rewards:
            return {
                "experiment": self.experiment_name,
                "num_iterations": 0,
                "wall_time_s": wall_time,
            }

        arr = np.array(rewards, dtype=float)
        window = min(10, len(arr))

        return {
            "experiment": self.experiment_name,
            "num_iterations": len(rewards),
            "wall_time_s": wall_time,
            "final_reward": float(arr[-1]),
            "best_reward": float(arr.max()),
            "mean_reward": float(arr.mean()),
            "std_reward": float(arr.std()),
            "smoothed_final": float(arr[-window:].mean()),
            "improvement": float(arr[-window:].mean() - arr[:window].mean()),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the log file (if any)."""
        if self._file is not None:
            footer = {
                "event": "end",
                "experiment": self.experiment_name,
                "timestamp": time.time(),
                "num_iterations": len(self._entries),
            }
            self._file.write(json.dumps(footer) + "\n")
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serializable(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
