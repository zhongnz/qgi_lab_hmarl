"""Training report generation for MAPPO experiments.

Produces structured Markdown reports summarising training runs,
evaluation results, and hyperparameter configurations.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def generate_training_report(
    history: list[dict[str, Any]],
    eval_result: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    elapsed_seconds: float | None = None,
    title: str = "MAPPO Training Report",
) -> str:
    """Generate a Markdown training report from logged history.

    Parameters
    ----------
    history:
        Per-iteration log entries from ``MAPPOTrainer.train()``.
    eval_result:
        Multi-episode evaluation result from ``evaluate_episodes()``.
    config:
        Training/env configuration dict to record.
    elapsed_seconds:
        Wall-clock training time.
    title:
        Report title.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines: list[str] = []
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated: {timestamp}*")
    lines.append("")

    # ---- Configuration ----
    if config:
        lines.append("## Configuration")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        _flat = _flatten_config(config)
        for key, val in sorted(_flat.items()):
            lines.append(f"| `{key}` | `{val}` |")
        lines.append("")

    # ---- Training summary ----
    lines.append("## Training Summary")
    lines.append("")
    n_iters = len(history)
    lines.append(f"- **Iterations**: {n_iters}")
    if elapsed_seconds is not None:
        lines.append(f"- **Wall-clock time**: {_format_duration(elapsed_seconds)}")
        if n_iters > 0:
            lines.append(f"- **Time per iteration**: {elapsed_seconds / n_iters:.2f}s")

    if history:
        rewards = [e.get("mean_reward", 0.0) for e in history]
        lines.append(f"- **Final mean reward**: {rewards[-1]:.6f}")
        lines.append(f"- **Best mean reward**: {max(rewards):.6f}")
        lines.append(f"- **Worst mean reward**: {min(rewards):.6f}")
        lines.append(f"- **Reward std**: {float(np.std(rewards)):.6f}")

        # Improvement over training
        if n_iters >= 10:
            early = np.mean(rewards[:n_iters // 10])
            late = np.mean(rewards[-n_iters // 10:])
            delta = late - early
            lines.append(f"- **Early avg (first 10%)**: {early:.6f}")
            lines.append(f"- **Late avg (last 10%)**: {late:.6f}")
            lines.append(f"- **Improvement (late - early)**: {delta:+.6f}")

        # LR tracking
        if "lr" in history[-1]:
            lines.append(f"- **Final LR**: {history[-1]['lr']:.2e}")

        # Entropy coeff
        if "entropy_coeff" in history[-1]:
            lines.append(f"- **Final entropy coeff**: {history[-1]['entropy_coeff']:.4f}")
    lines.append("")

    # ---- Per-agent losses ----
    if history:
        lines.append("## Per-Agent Training Metrics (Final Iteration)")
        lines.append("")
        last = history[-1]
        agent_types = _extract_agent_types(last)
        if agent_types:
            lines.append("| Agent | Policy Loss | Value Loss | Entropy | Clip Frac | KL |")
            lines.append("|-------|------------|------------|---------|-----------|-----|")
            for at in agent_types:
                pl = last.get(f"{at}_policy_loss", "—")
                vl = last.get(f"{at}_value_loss", "—")
                ent = last.get(f"{at}_entropy", "—")
                cf = last.get(f"{at}_clip_frac", "—")
                kl = last.get(f"{at}_approx_kl", "—")
                lines.append(
                    f"| {at} | {_fmt(pl)} | {_fmt(vl)} | {_fmt(ent)} | {_fmt(cf)} | {_fmt(kl)} |"
                )
            lines.append("")

    # ---- Evaluation results ----
    if eval_result and "mean" in eval_result:
        lines.append("## Evaluation (Multi-Episode)")
        lines.append("")
        n_eps = len(eval_result.get("episodes", []))
        lines.append(f"- **Episodes**: {n_eps}")
        lines.append("")
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        for key in eval_result["mean"]:
            m = eval_result["mean"][key]
            s = eval_result["std"][key]
            mn = eval_result["min"][key]
            mx = eval_result["max"][key]
            lines.append(f"| {key} | {m:.4f} | {s:.4f} | {mn:.4f} | {mx:.4f} |")
        lines.append("")

    return "\n".join(lines)


def generate_sweep_report(
    sweep_df: Any,
    title: str = "Hyperparameter Sweep Report",
    sort_by: str = "eval_total_reward",
) -> str:
    """Generate a Markdown report from a sweep results DataFrame.

    Parameters
    ----------
    sweep_df:
        DataFrame from ``run_mappo_hyperparam_sweep()``.
    title:
        Report title.
    sort_by:
        Column to sort by (descending).

    Returns
    -------
    str
        Markdown-formatted report.
    """
    import pandas as pd

    df = pd.DataFrame(sweep_df) if not isinstance(sweep_df, pd.DataFrame) else sweep_df
    lines: list[str] = []
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated: {timestamp}*")
    lines.append("")
    lines.append(f"- **Configurations tested**: {len(df)}")

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    lines.append("")
    lines.append("## Results")
    lines.append("")

    # Build table
    cols = list(df.columns)
    lines.append("| Rank | " + " | ".join(cols) + " |")
    lines.append("|------|" + "|".join(["------"] * len(cols)) + "|")
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [_fmt(row[c]) for c in cols]
        lines.append(f"| {i + 1} | " + " | ".join(vals) + " |")

    lines.append("")

    # Best config
    if len(df) > 0:
        best = df.iloc[0]
        lines.append("## Best Configuration")
        lines.append("")
        for col in cols:
            lines.append(f"- **{col}**: {_fmt(best[col])}")
        lines.append("")

    return "\n".join(lines)


def generate_ablation_report(
    ablation_df: Any,
    title: str = "Ablation Study Report",
) -> str:
    """Generate a Markdown report from ablation results.

    Parameters
    ----------
    ablation_df:
        DataFrame from ``run_mappo_ablation()``.
    title:
        Report title.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    import pandas as pd

    df = pd.DataFrame(ablation_df) if not isinstance(ablation_df, pd.DataFrame) else ablation_df
    lines: list[str] = []
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated: {timestamp}*")
    lines.append("")
    lines.append(f"- **Variants tested**: {len(df)}")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["------"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = [_fmt(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    # Compute deltas from baseline if present
    if "ablation" in df.columns and "baseline" in df["ablation"].values:
        metric_cols = [c for c in cols if c not in ("ablation",) and df[c].dtype in ("float64", "int64")]
        if metric_cols:
            baseline_row = df[df["ablation"] == "baseline"].iloc[0]
            lines.append("## Deltas from Baseline")
            lines.append("")
            lines.append("| Variant | " + " | ".join(f"Δ {c}" for c in metric_cols) + " |")
            lines.append("|---------|" + "|".join(["------"] * len(metric_cols)) + "|")
            for _, row in df.iterrows():
                if row["ablation"] == "baseline":
                    continue
                deltas = []
                for c in metric_cols:
                    d = float(row[c]) - float(baseline_row[c])
                    deltas.append(f"{d:+.4f}")
                lines.append(f"| {row['ablation']} | " + " | ".join(deltas) + " |")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict into dot-separated keys."""
    flat: dict[str, Any] = {}
    for k, v in config.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_config(v, key))
        else:
            flat[key] = v
    return flat


def _format_duration(seconds: float) -> str:
    """Format seconds into a readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _extract_agent_types(entry: dict[str, Any]) -> list[str]:
    """Extract agent type names from a log entry's keys."""
    types: set[str] = set()
    for key in entry:
        if key.endswith("_policy_loss"):
            types.add(key.replace("_policy_loss", ""))
    return sorted(types)


def _fmt(val: Any) -> str:
    """Format a value for Markdown table display."""
    if isinstance(val, float):
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)
