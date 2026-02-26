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
