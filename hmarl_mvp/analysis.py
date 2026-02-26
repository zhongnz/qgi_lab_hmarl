"""Analysis utilities for experiment post-processing and comparison.

Provides helpers for comparing MAPPO results against heuristic baselines,
computing ablation deltas, ranking hyperparameter configurations, and
generating structured experiment summaries.

Usage::

    from hmarl_mvp.analysis import (
        compare_to_baselines,
        rank_sweep_results,
        compute_ablation_deltas,
        summarize_experiment,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Aliases for mapping between MAPPOTrainer.evaluate() keys and
# run_experiment() DataFrame columns.
_MAPPO_KEY_ALIASES: dict[str, str] = {
    "avg_vessel_reward": "mean_vessel_reward",
    "avg_port_reward": "mean_port_reward",
    "coordinator_reward": "mean_coordinator_reward",
}

# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def compare_to_baselines(
    mappo_metrics: dict[str, float],
    baseline_results: dict[str, pd.DataFrame],
    metric_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Compare MAPPO evaluation metrics against heuristic baselines.

    Parameters
    ----------
    mappo_metrics:
        Dict from ``MAPPOTrainer.evaluate()`` or equivalent.
    baseline_results:
        ``{policy_name: dataframe}`` from ``run_experiment()`` runs.
    metric_keys:
        Which metric columns to compare.  Defaults to common reward metrics.

    Returns
    -------
    pd.DataFrame
        One row per policy (including ``"mappo"``), columns are metrics
        plus a ``"rank"`` column for overall performance ordering.
    """
    metric_keys = metric_keys or [
        "avg_vessel_reward",
        "avg_port_reward",
        "coordinator_reward",
        "total_reward",
    ]

    rows: list[dict[str, Any]] = []

    # MAPPO row
    mappo_row: dict[str, Any] = {"policy": "mappo"}
    for k in metric_keys:
        # Map evaluate() keys → run_experiment() keys for interop
        alt = _MAPPO_KEY_ALIASES.get(k, k)
        mappo_row[k] = mappo_metrics.get(k, mappo_metrics.get(alt, float("nan")))
    rows.append(mappo_row)

    # Baseline rows
    for policy_name, df in baseline_results.items():
        row: dict[str, Any] = {"policy": policy_name}
        for k in metric_keys:
            if k in df.columns:
                row[k] = float(df[k].mean())
            elif k == "total_reward":
                # Synthesize from component rewards if available
                total = 0.0
                for sub in ["mean_vessel_reward", "mean_port_reward", "mean_coordinator_reward"]:
                    if sub in df.columns:
                        total += float(df[sub].mean())
                row[k] = total
            else:
                row[k] = float("nan")
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Rank by total_reward (higher is better)
    if "total_reward" in result_df.columns:
        result_df["rank"] = result_df["total_reward"].rank(ascending=False).astype(int)
    else:
        result_df["rank"] = range(1, len(result_df) + 1)

    return result_df.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sweep result ranking
# ---------------------------------------------------------------------------


def rank_sweep_results(
    sweep_results: list[dict[str, Any]],
    sort_by: str = "mean_reward",
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank hyperparameter sweep configurations by a metric.

    Parameters
    ----------
    sweep_results:
        List of result dicts, each containing ``"config"`` (or individual
        param keys) and metric values.
    sort_by:
        Metric name to rank by.
    ascending:
        If True, lower is better (e.g. for loss metrics).

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with a ``"rank"`` column.
    """
    if not sweep_results:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for i, result in enumerate(sweep_results):
        row = {"sweep_idx": i}
        # Flatten config if present
        if "config" in result and isinstance(result["config"], dict):
            for k, v in result["config"].items():
                row[f"cfg_{k}"] = v
        # Copy all non-config keys
        for k, v in result.items():
            if k != "config":
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Ablation deltas
# ---------------------------------------------------------------------------


def compute_ablation_deltas(
    ablation_results: dict[str, dict[str, Any]],
    baseline_key: str = "baseline",
    metric_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Compute deltas between ablation variants and a baseline.

    Parameters
    ----------
    ablation_results:
        ``{variant_name: metrics_dict}`` — each value is a dict of metric
        names to floats.
    baseline_key:
        Which variant to use as the reference (default ``"baseline"``).
    metric_keys:
        Metrics to compare.  If None, uses all numeric keys from baseline.

    Returns
    -------
    pd.DataFrame
        One row per non-baseline variant.  Columns include the raw metric
        and a ``"delta_<metric>"`` showing the difference from baseline.
    """
    if baseline_key not in ablation_results:
        raise KeyError(f"baseline_key={baseline_key!r} not found in ablation_results")

    baseline = ablation_results[baseline_key]
    if metric_keys is None:
        metric_keys = [
            k for k, v in baseline.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

    rows: list[dict[str, Any]] = []
    for variant, metrics in ablation_results.items():
        if variant == baseline_key:
            continue
        row: dict[str, Any] = {"variant": variant}
        for k in metric_keys:
            val = float(metrics.get(k, float("nan")))
            base_val = float(baseline.get(k, float("nan")))
            row[k] = val
            row[f"delta_{k}"] = val - base_val
            if abs(base_val) > 1e-12:
                row[f"pct_{k}"] = ((val - base_val) / abs(base_val)) * 100.0
            else:
                row[f"pct_{k}"] = float("nan")
        rows.append(row)

    # Add baseline row for reference
    base_row: dict[str, Any] = {"variant": baseline_key}
    for k in metric_keys:
        base_row[k] = float(baseline.get(k, float("nan")))
        base_row[f"delta_{k}"] = 0.0
        base_row[f"pct_{k}"] = 0.0
    rows.insert(0, base_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training curve analysis
# ---------------------------------------------------------------------------


def compute_training_stats(
    reward_history: list[float],
    window: int = 10,
) -> dict[str, float]:
    """Compute summary statistics from a training reward curve.

    Parameters
    ----------
    reward_history:
        List of per-iteration mean rewards.
    window:
        Window size for smoothed metrics.

    Returns
    -------
    dict with keys: final_reward, best_reward, mean_reward, std_reward,
    smoothed_final (last ``window`` values), improvement (last - first window).
    """
    if not reward_history:
        return {
            "final_reward": float("nan"),
            "best_reward": float("nan"),
            "mean_reward": float("nan"),
            "std_reward": float("nan"),
            "smoothed_final": float("nan"),
            "improvement": float("nan"),
        }

    arr = np.array(reward_history, dtype=float)
    w = min(window, len(arr))
    smoothed_final = float(arr[-w:].mean())
    smoothed_first = float(arr[:w].mean())

    return {
        "final_reward": float(arr[-1]),
        "best_reward": float(arr.max()),
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "smoothed_final": smoothed_final,
        "improvement": smoothed_final - smoothed_first,
    }


# ---------------------------------------------------------------------------
# Experiment summary
# ---------------------------------------------------------------------------


def summarize_experiment(
    name: str,
    training_history: list[float] | None = None,
    eval_metrics: dict[str, float] | None = None,
    diagnostics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a structured experiment summary dict.

    Combines training curve statistics, evaluation metrics, diagnostics,
    and configuration into a single serialisable summary.
    """
    summary: dict[str, Any] = {"name": name}

    if training_history is not None:
        summary["training_stats"] = compute_training_stats(training_history)
        summary["num_iterations"] = len(training_history)

    if eval_metrics is not None:
        summary["eval"] = dict(eval_metrics)

    if diagnostics is not None:
        summary["diagnostics"] = dict(diagnostics)

    if config is not None:
        summary["config"] = dict(config)

    return summary


def format_comparison_table(
    experiments: list[dict[str, Any]],
    metric_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Format multiple experiment summaries into a comparison DataFrame.

    Parameters
    ----------
    experiments:
        List of dicts from ``summarize_experiment()``.
    metric_keys:
        Eval metrics to include as columns.  Defaults to all numeric eval keys.
    """
    if not experiments:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for exp in experiments:
        row: dict[str, Any] = {"name": exp.get("name", "?")}

        # Training stats
        ts = exp.get("training_stats", {})
        row["final_reward"] = ts.get("smoothed_final", float("nan"))
        row["improvement"] = ts.get("improvement", float("nan"))
        row["num_iterations"] = exp.get("num_iterations", 0)

        # Eval metrics
        ev = exp.get("eval", {})
        if metric_keys is None:
            for k, v in ev.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[f"eval_{k}"] = v
        else:
            for k in metric_keys:
                row[f"eval_{k}"] = ev.get(k, float("nan"))

        rows.append(row)

    return pd.DataFrame(rows)
