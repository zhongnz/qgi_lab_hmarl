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


# ---------------------------------------------------------------------------
# Hyperparameter sweep and ablation plots
# ---------------------------------------------------------------------------


def plot_sweep_heatmap(
    sweep_df: Any,
    x_param: str,
    y_param: str,
    metric: str = "total_reward",
    out_path: str | None = None,
) -> None:
    """Plot a 2-D heatmap of sweep results over two swept parameters.

    Parameters
    ----------
    sweep_df:
        DataFrame from ``run_mappo_hyperparam_sweep()``.
    x_param, y_param:
        Column names of the two swept parameters.
    metric:
        Metric column to plot as colour values.
    """
    import pandas as pd

    df = pd.DataFrame(sweep_df) if not isinstance(sweep_df, pd.DataFrame) else sweep_df
    if x_param not in df.columns or y_param not in df.columns:
        return
    if metric not in df.columns:
        return

    pivot = df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v}" for v in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v}" for v in pivot.index], fontsize=9)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"Sweep: {metric}")
    fig.colorbar(im, ax=ax, label=metric)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not _isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="white")

    plt.tight_layout()
    _save_or_show(fig, out_path)


def plot_ablation_bar(
    ablation_df: Any,
    metrics: list[str] | None = None,
    out_path: str | None = None,
) -> None:
    """Bar chart comparing ablation variants on key metrics.

    Parameters
    ----------
    ablation_df:
        DataFrame from ``run_mappo_ablation()``.
    metrics:
        Metric columns to plot. Defaults to reward metrics.
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(ablation_df) if not isinstance(ablation_df, pd.DataFrame) else ablation_df
    if "ablation" not in df.columns:
        return

    metrics = metrics or ["final_mean_reward", "best_mean_reward", "total_reward"]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    n_variants = len(df)
    n_metrics = len(metrics)
    x = np.arange(n_variants)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, n_variants * 1.2), 5))
    for i, metric in enumerate(metrics):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            df[metric].values,
            width,
            label=metric.replace("_", " ").title(),
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["ablation"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("Ablation Comparison")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, out_path)


def plot_training_dashboard(
    history: list[dict[str, Any]],
    out_path: str | None = None,
) -> None:
    """Multi-panel training dashboard showing reward, losses, KL, and entropy.

    Parameters
    ----------
    history:
        Per-iteration log entries from ``MAPPOTrainer.train()``.
    """
    import pandas as pd

    df = pd.DataFrame(history)
    iters = df["iteration"] if "iteration" in df.columns else range(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MAPPO Training Dashboard", fontsize=14)

    # Panel 1: Reward
    ax = axes[0, 0]
    if "mean_reward" in df.columns:
        ax.plot(iters, df["mean_reward"], color="steelblue", alpha=0.5, linewidth=1)
        if len(df) >= 10:
            window = max(5, len(df) // 10)
            smoothed = df["mean_reward"].rolling(window, min_periods=1).mean()
            ax.plot(iters, smoothed, color="darkblue", linewidth=2, label=f"MA({window})")
            ax.legend(fontsize=8)
    ax.set_title("Mean Reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")

    # Panel 2: Value losses
    ax = axes[0, 1]
    loss_cols = [c for c in df.columns if c.endswith("_value_loss")]
    for col in loss_cols:
        label = col.replace("_value_loss", "")
        ax.plot(iters, df[col], label=label, linewidth=1.2)
    ax.set_title("Critic Value Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    if loss_cols:
        ax.legend(fontsize=8)

    # Panel 3: Approximate KL
    ax = axes[1, 0]
    kl_cols = [c for c in df.columns if c.endswith("_approx_kl")]
    for col in kl_cols:
        label = col.replace("_approx_kl", "")
        ax.plot(iters, df[col], label=label, linewidth=1.2)
    ax.set_title("Approximate KL Divergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL")
    if kl_cols:
        ax.legend(fontsize=8)
        ax.axhline(y=0.02, color="red", linestyle="--", alpha=0.5, label="target")

    # Panel 4: Entropy
    ax = axes[1, 1]
    ent_cols = [c for c in df.columns if c.endswith("_entropy")]
    for col in ent_cols:
        label = col.replace("_entropy", "")
        ax.plot(iters, df[col], label=label, linewidth=1.2)
    if "entropy_coeff" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(iters, df["entropy_coeff"], color="gray", linestyle="--", linewidth=1, label="coeff")
        ax2.set_ylabel("Entropy Coeff", color="gray")
        ax2.legend(fontsize=7, loc="lower right")
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    if ent_cols:
        ax.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _save_or_show(fig, out_path)


def _isnan(val: Any) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False
