"""Plot helpers for policy comparison and ablations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt


def plot_policy_comparison(
    results: dict[str, Any],
    out_path: str | None = None,
) -> None:
    """Reproduce the notebook-style 2x3 policy comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Policy Comparison - MVP Baselines (RQ2)", fontsize=14)

    metrics_to_plot = [
        ("avg_queue", "Avg Queue Length", "Queue"),
        ("dock_utilization", "Dock Utilization", "Utilization"),
        ("total_emissions_co2", "Cumulative CO2 (tons)", "CO2"),
        ("total_fuel_used", "Cumulative Fuel (tons)", "Fuel"),
        ("avg_vessel_reward", "Avg Vessel Reward", "Reward"),
        ("total_ops_cost_usd", "Total Ops Cost (USD)", "Cost ($)"),
    ]

    for ax, (col, title, ylabel) in zip(axes.flat, metrics_to_plot):
        for policy, df in results.items():
            ax.plot(df["t"], df[col], label=policy)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _save_or_show(fig, out_path)


def plot_horizon_sweep(
    horizon_results: dict[int, Any],
    out_path: str | None = None,
) -> None:
    """Plot forecast horizon ablation."""
    _plot_sweep(
        results=horizon_results,
        suptitle="Forecast Horizon Ablation (RQ3)",
        label_fn=lambda k: f"{k}h",
        out_path=out_path,
    )


def plot_noise_sweep(
    noise_results: dict[float, Any],
    out_path: str | None = None,
) -> None:
    """Plot forecast noise ablation."""
    _plot_sweep(
        results=noise_results,
        suptitle="Forecast Noise Ablation (RQ3)",
        label_fn=lambda k: f"sigma={k}",
        out_path=out_path,
    )


def plot_sharing_sweep(
    sharing_results: dict[str, Any],
    out_path: str | None = None,
) -> None:
    """Plot forecast sharing ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Forecast Sharing Ablation (RQ3)", fontsize=13)
    for label, df in sharing_results.items():
        axes[0].plot(df["t"], df["avg_queue"], label=label)
        axes[1].plot(df["t"], df["total_ops_cost_usd"], label=label)
    axes[0].set_title("Avg Queue")
    axes[1].set_title("Total Ops Cost ($)")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    _save_or_show(fig, out_path)


# -- Internal helpers --------------------------------------------------------

_SWEEP_METRICS = [
    ("avg_queue", "Avg Queue"),
    ("total_emissions_co2", "Cumulative CO2"),
    ("total_ops_cost_usd", "Total Ops Cost ($)"),
]


def _plot_sweep(
    results: dict[Any, Any],
    suptitle: str,
    label_fn: Callable[[Any], str],
    out_path: str | None = None,
) -> None:
    """Shared layout for 1×3 sweep ablation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(suptitle, fontsize=13)
    for ax, (col, title) in zip(axes, _SWEEP_METRICS):
        for key, df in results.items():
            ax.plot(df["t"], df[col], label=label_fn(key))
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    _save_or_show(fig, out_path)


def _save_or_show(fig: plt.Figure, out_path: str | None) -> None:
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# MAPPO training and comparison plots
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_log: Any,
    out_path: str | None = None,
) -> None:
    """Plot MAPPO training reward curve and value losses.

    Parameters
    ----------
    train_log:
        DataFrame with columns ``iteration``, ``mean_reward``,
        and optional ``*_value_loss`` columns.
    """
    import pandas as pd

    df = pd.DataFrame(train_log) if not isinstance(train_log, pd.DataFrame) else train_log

    has_losses = any(c.endswith("_value_loss") for c in df.columns)
    ncols = 2 if has_losses else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    fig.suptitle("MAPPO Training Progress", fontsize=13)

    # Reward curve
    ax = axes[0]
    ax.plot(df["iteration"], df["mean_reward"], color="steelblue", linewidth=1.5)
    if len(df) >= 10:
        window = max(5, len(df) // 10)
        smoothed = df["mean_reward"].rolling(window, min_periods=1).mean()
        ax.plot(df["iteration"], smoothed, color="darkblue", linewidth=2, label=f"MA({window})")
        ax.legend(fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Training Reward")

    # Value losses
    if has_losses:
        ax = axes[1]
        loss_cols = [c for c in df.columns if c.endswith("_value_loss")]
        for col in loss_cols:
            label = col.replace("_value_loss", "")
            ax.plot(df["iteration"], df[col], label=label, linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value Loss")
        ax.set_title("Critic Loss")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    _save_or_show(fig, out_path)


def plot_mappo_comparison(
    results: dict[str, Any],
    out_path: str | None = None,
) -> None:
    """Plot MAPPO evaluation against heuristic baselines.

    Parameters
    ----------
    results:
        Dict mapping policy name → per-step DataFrame, as returned by
        ``run_mappo_comparison()``.  The ``_train_log`` key is skipped.
    """
    # Filter out non-policy entries
    policy_results = {k: v for k, v in results.items() if not k.startswith("_")}
    if not policy_results:
        return

    metrics_to_plot = [
        ("avg_queue", "Avg Queue Length", "Queue"),
        ("dock_utilization", "Dock Utilization", "Utilization"),
        ("total_emissions_co2", "Cumulative CO2 (tons)", "CO2"),
        ("avg_vessel_reward", "Avg Vessel Reward", "Reward"),
        ("total_ops_cost_usd", "Total Ops Cost (USD)", "Cost ($)"),
        ("coordinator_reward", "Coordinator Reward", "Reward"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("MAPPO vs Heuristic Baselines", fontsize=14)

    for ax, (col, title, ylabel) in zip(axes.flat, metrics_to_plot):
        for policy, df in policy_results.items():
            if col in df.columns:
                style = {"linewidth": 2.5, "color": "crimson"} if policy == "mappo" else {}
                ax.plot(df["t"], df[col], label=policy, **style)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _save_or_show(fig, out_path)
