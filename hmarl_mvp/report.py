"""Training report generation for MAPPO experiments.

Produces structured Markdown reports summarising training runs,
evaluation results, and hyperparameter configurations.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

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
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
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
        if "joint_mean_reward" in history[-1]:
            joint_rewards = [e.get("joint_mean_reward", 0.0) for e in history]
            lines.append(f"- **Final joint mean reward**: {joint_rewards[-1]:.6f}")
        if "total_reward" in history[-1]:
            total_rewards = [e.get("total_reward", 0.0) for e in history]
            lines.append(f"- **Final total reward**: {total_rewards[-1]:.6f}")
        for agent in ("vessel", "port", "coordinator"):
            key = f"{agent}_mean_reward"
            if key in history[-1]:
                lines.append(f"- **Final {agent} mean reward**: {history[-1][key]:.6f}")

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

    # ---- Convergence analysis ----
    if history and len(history) >= 20:
        lines.append("## Convergence Analysis")
        lines.append("")
        conv = _convergence_analysis(history)
        lines.append(f"- **Plateau detected**: {'Yes' if conv['plateau_detected'] else 'No'}")
        if conv["plateau_detected"]:
            lines.append(f"- **Plateau start iteration**: {conv['plateau_start']}")
            lines.append(f"- **Suggested stopping point**: {conv['suggested_stop']}")
        lines.append(f"- **Final reward stability (last 10% std)**: {conv['final_std']:.6f}")
        lines.append(f"- **Convergence rate (early→late improvement)**: {conv['convergence_rate']:+.6f}")

        agent_types = _extract_agent_types(history[-1])
        if agent_types:
            lines.append("")
            lines.append("### Per-Agent Convergence")
            lines.append("")
            lines.append("| Agent | First-non-NaN iter | Final loss | Loss trend (last 20%) |")
            lines.append("|-------|--------------------|------------|----------------------|")
            for at in agent_types:
                losses = [e.get(f"{at}_policy_loss") for e in history]
                valid = [(i, v) for i, v in enumerate(losses) if v is not None]
                if valid:
                    first_i = valid[0][0]
                    final_v = valid[-1][1]
                    tail = [v for _, v in valid[-max(1, len(valid) // 5):]]
                    trend = float(np.mean(np.diff(tail))) if len(tail) >= 2 else 0.0
                    trend_str = "↓ improving" if trend < -1e-6 else ("→ stable" if abs(trend) < 1e-6 else "↑ diverging")
                    lines.append(f"| {at} | {first_i} | {_fmt(final_v)} | {trend_str} ({trend:+.2e}) |")
            lines.append("")

    # ---- Multi-seed statistical significance ----
    if eval_result and "seed_rewards" in eval_result:
        lines.append("## Multi-Seed Statistical Analysis")
        lines.append("")
        seed_data = eval_result["seed_rewards"]
        sig = _significance_analysis(seed_data)
        lines.append(f"- **Seeds**: {sig['n_seeds']}")
        lines.append(f"- **Mean reward**: {sig['mean']:.4f} ± {sig['ci95']:.4f} (95% CI)")
        lines.append(f"- **Coefficient of variation**: {sig['cv']:.4f}")
        lines.append(f"- **Shapiro-Wilk p-value**: {sig['shapiro_p']:.4f}")
        normality = "consistent with normal" if sig["shapiro_p"] > 0.05 else "non-normal (p < 0.05)"
        lines.append(f"- **Normality**: {normality}")
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


# ---------------------------------------------------------------------------
# Convergence & significance helpers
# ---------------------------------------------------------------------------


def _convergence_analysis(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyse reward convergence: plateau detection and convergence rate."""
    rewards = np.array([e.get("mean_reward", 0.0) for e in history], dtype=float)
    n = len(rewards)
    window = max(1, n // 10)

    early_mean = float(np.mean(rewards[:window]))
    late_mean = float(np.mean(rewards[-window:]))
    final_std = float(np.std(rewards[-window:]))
    convergence_rate = late_mean - early_mean

    # Plateau detection: sliding window std falls below threshold
    plateau_detected = False
    plateau_start = n
    suggested_stop = n
    if n >= 20:
        half_window = max(5, n // 20)
        threshold = max(final_std * 0.5, 1e-6)
        for i in range(half_window, n - half_window):
            seg_std = float(np.std(rewards[i : i + half_window]))
            if seg_std < threshold:
                plateau_detected = True
                plateau_start = i
                # Suggest stopping at plateau + 10% additional as safety buffer
                suggested_stop = min(i + max(half_window, n // 10), n)
                break

    return {
        "plateau_detected": plateau_detected,
        "plateau_start": plateau_start,
        "suggested_stop": suggested_stop,
        "final_std": final_std,
        "convergence_rate": convergence_rate,
    }


def _significance_analysis(seed_rewards: list[float]) -> dict[str, Any]:
    """Multi-seed statistical significance: mean, CI, CV, normality test."""
    arr = np.array(seed_rewards, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # 95% CI using t-distribution
    if n > 1:
        t_val = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci95 = t_val * std / np.sqrt(n)
    else:
        ci95 = 0.0

    cv = std / abs(mean) if abs(mean) > 1e-12 else 0.0

    # Shapiro-Wilk test (requires n >= 3)
    if n >= 3:
        _, shapiro_p = scipy_stats.shapiro(arr)
    else:
        shapiro_p = 1.0

    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci95": float(ci95),
        "cv": cv,
        "shapiro_p": float(shapiro_p),
    }
