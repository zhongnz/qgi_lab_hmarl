"""Curriculum learning scheduler for progressive environment complexity.

Starts with a simple environment configuration (few vessels, few ports,
short horizons) and gradually ramps parameters toward the target values
as training progresses.  This reduces the initial learning burden and
produces more stable reward curves.

Usage::

    curriculum = CurriculumScheduler(
        target_config={"num_vessels": 8, "num_ports": 5, "rollout_steps": 64},
        warmup_fraction=0.3,
    )
    for iteration in range(total_iters):
        env_cfg = curriculum.get_config(iteration, total_iters)
        # Rebuild or reconfigure the environment with env_cfg ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Parameters that support curriculum ramping (int-valued, ramped up)
_RAMPABLE_INT_KEYS = {
    "num_vessels",
    "num_ports",
    "rollout_steps",
    "docks_per_port",
}

# Parameters that support curriculum ramping (float-valued, ramped up)
_RAMPABLE_FLOAT_KEYS = {
    "emission_lambda",
    "sea_state_max",
    "weather_penalty_factor",
    "weather_shaping_weight",
}

# Parameters that support curriculum ramping (bool-valued, threshold at 0.5)
_RAMPABLE_BOOL_KEYS = {
    "weather_enabled",
}


@dataclass
class CurriculumStage:
    """A single stage in a multi-stage curriculum.

    Parameters
    ----------
    fraction:
        Fraction of total training at which this stage begins (0.0–1.0).
    config_overrides:
        Config values active during this stage.
    """

    fraction: float
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumScheduler:
    """Progressive curriculum scheduler for HMARL training.

    Two modes of operation:

    1. **Linear ramp** (default) — parameters are linearly interpolated
       from ``start_config`` to ``target_config`` over the first
       ``warmup_fraction`` of training.
    2. **Multi-stage** — explicit ``stages`` define the curriculum.
       When stages are provided the linear ramp is ignored.

    Parameters
    ----------
    target_config:
        The final (full-difficulty) environment configuration dict.
    start_config:
        The initial (easy) configuration overrides.  Keys not present
        here default to reasonable minimums (2 vessels, 2 ports, etc.).
    warmup_fraction:
        Fraction of total iterations over which to ramp from start to
        target (linear mode only).
    stages:
        Explicit curriculum stages.  If provided, overrides linear ramp.
    """

    target_config: dict[str, Any]
    start_config: dict[str, Any] = field(default_factory=dict)
    warmup_fraction: float = 0.3
    stages: list[CurriculumStage] | None = None

    def __post_init__(self) -> None:
        if self.warmup_fraction < 0.0 or self.warmup_fraction > 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1], got {self.warmup_fraction}")
        if self.stages is not None:
            fracs = [s.fraction for s in self.stages]
            if fracs != sorted(fracs):
                raise ValueError("stages must be sorted by ascending fraction")
            if any(f < 0.0 or f > 1.0 for f in fracs):
                raise ValueError("stage fractions must be in [0.0, 1.0]")

    # ------------------------------------------------------------------
    # Default easy configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _default_start() -> dict[str, Any]:
        """Minimal sensible starting config for curriculum warm-up."""
        return {
            "num_vessels": 2,
            "num_ports": 2,
            "docks_per_port": 2,
            "rollout_steps": 20,
        }

    # ------------------------------------------------------------------
    # Linear ramp mode
    # ------------------------------------------------------------------

    def _linear_config(self, progress: float) -> dict[str, Any]:
        """Compute interpolated config from linear curriculum ramp.

        ``progress`` is in [0.0, 1.0] representing fraction of training done.
        """
        start = {**self._default_start(), **self.start_config}
        target = dict(self.target_config)

        if self.warmup_fraction <= 0.0:
            return dict(target)

        alpha = min(progress / self.warmup_fraction, 1.0)

        result: dict[str, Any] = dict(target)
        for key in _RAMPABLE_INT_KEYS:
            if key in target:
                s = int(start.get(key, target[key]))
                t = int(target[key])
                result[key] = int(round(s + alpha * (t - s)))
                result[key] = max(result[key], 1)  # safety floor

        for key in _RAMPABLE_FLOAT_KEYS:
            if key in target:
                sf = float(start.get(key, target[key]))
                tf = float(target[key])
                result[key] = sf + alpha * (tf - sf)

        # Bool ramp: enable when alpha crosses 0.5
        for key in _RAMPABLE_BOOL_KEYS:
            if key in target:
                s_val = bool(start.get(key, False))
                t_val = bool(target[key])
                if s_val == t_val:
                    result[key] = t_val
                else:
                    # Ramp from False → True: enable at alpha ≥ 0.5
                    result[key] = alpha >= 0.5

        return result

    # ------------------------------------------------------------------
    # Multi-stage mode
    # ------------------------------------------------------------------

    def _staged_config(self, progress: float) -> dict[str, Any]:
        """Compute config from explicit curriculum stages."""
        assert self.stages is not None
        base = dict(self.target_config)
        active_overrides: dict[str, Any] = {}
        for stage in self.stages:
            if progress >= stage.fraction:
                active_overrides = dict(stage.config_overrides)
            else:
                break
        base.update(active_overrides)
        return base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self, iteration: int, total_iterations: int) -> dict[str, Any]:
        """Return the environment config for the given iteration.

        Parameters
        ----------
        iteration:
            Current training iteration (0-based).
        total_iterations:
            Total number of planned training iterations.

        Returns
        -------
        dict[str, Any]
            A config dict suitable for ``MaritimeEnv(config=...)``.
        """
        if total_iterations <= 0:
            return dict(self.target_config)
        progress = min(iteration / total_iterations, 1.0)
        if self.stages is not None:
            return self._staged_config(progress)
        return self._linear_config(progress)

    def get_progress(self, iteration: int, total_iterations: int) -> float:
        """Return curriculum progress in [0.0, 1.0]."""
        if total_iterations <= 0:
            return 1.0
        return min(iteration / total_iterations, 1.0)

    def is_at_target(self, iteration: int, total_iterations: int) -> bool:
        """Return True when the curriculum has fully ramped to the target."""
        progress = self.get_progress(iteration, total_iterations)
        if self.stages is not None:
            last_frac = self.stages[-1].fraction if self.stages else 0.0
            return progress >= last_frac
        return progress >= self.warmup_fraction


def make_curriculum_configs(
    target_config: dict[str, Any],
    total_iterations: int,
    warmup_fraction: float = 0.3,
    start_config: dict[str, Any] | None = None,
    sample_points: int = 5,
) -> list[tuple[int, dict[str, Any]]]:
    """Preview the curriculum by sampling configs at evenly spaced iterations.

    Returns a list of ``(iteration, config_dict)`` tuples useful for
    debugging and documentation.
    """
    scheduler = CurriculumScheduler(
        target_config=target_config,
        start_config=start_config or {},
        warmup_fraction=warmup_fraction,
    )
    points = np.linspace(0, total_iterations - 1, sample_points, dtype=int)
    return [
        (int(it), scheduler.get_config(int(it), total_iterations))
        for it in points
    ]
