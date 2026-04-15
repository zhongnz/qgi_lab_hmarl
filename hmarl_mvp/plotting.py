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
    axes[0].set_xlabel("Step")
    axes[1].set_title("Total Ops Cost ($)")
    axes[1].set_xlabel("Step")
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
    """Plot MAPPO training reward and diagnostic curves.

    Parameters
    ----------
    train_log:
        DataFrame with columns ``iteration``, ``mean_reward``,
        optional ``joint_mean_reward``, per-agent reward columns,
        and optional critic/policy diagnostics.
    """
    import pandas as pd

    df = pd.DataFrame(train_log) if not isinstance(train_log, pd.DataFrame) else train_log
    if df.empty or "iteration" not in df.columns:
        return

    reward_cols = [
        ("vessel_mean_reward", "Vessel", "#1b9e77"),
        ("port_mean_reward", "Port", "#d95f02"),
        ("coordinator_mean_reward", "Coordinator", "#7570b3"),
    ]
    color_by_agent = {
        "vessel": "#1b9e77",
        "port": "#d95f02",
        "coordinator": "#7570b3",
    }
    iters = df["iteration"]
    window = max(5, len(df) // 10) if len(df) >= 10 else None

    def _smoothed(series: Any) -> Any:
        if window is None:
            return None
        return series.rolling(window, min_periods=1).mean()

    panel_builders: list[tuple[str, Any]] = []
    per_type_cols = [col for col, _, _ in reward_cols if col in df.columns]
    if "joint_mean_reward" in df.columns:
        joint_reward = df["joint_mean_reward"]
        joint_label = "Joint mean reward"
    elif per_type_cols:
        joint_reward = sum(df[col] for col in per_type_cols)
        joint_label = "Joint mean reward (derived)"
    else:
        joint_reward = None
        joint_label = "Joint mean reward"

    if joint_reward is not None or "mean_reward" in df.columns:
        def _plot_aggregate_reward(ax: Any) -> None:
            if joint_reward is not None:
                ax.plot(iters, joint_reward, color="steelblue", alpha=0.35, linewidth=1.0)
                smoothed = _smoothed(joint_reward)
                if smoothed is not None:
                    ax.plot(
                        iters,
                        smoothed,
                        color="darkblue",
                        linewidth=2.0,
                        label=f"{joint_label} MA({window})",
                    )
                else:
                    ax.plot(
                        iters,
                        joint_reward,
                        color="darkblue",
                        linewidth=1.8,
                        label=joint_label,
                    )
            if "mean_reward" in df.columns:
                legacy = df["mean_reward"]
                if joint_reward is None:
                    ax.plot(iters, legacy, color="steelblue", alpha=0.35, linewidth=1.0)
                    smoothed = _smoothed(legacy)
                    if smoothed is not None:
                        ax.plot(
                            iters,
                            smoothed,
                            color="darkblue",
                            linewidth=2.0,
                            label=f"mean_reward MA({window})",
                        )
                    else:
                        ax.plot(
                            iters,
                            legacy,
                            color="darkblue",
                            linewidth=1.8,
                            label="mean_reward",
                        )
                else:
                    ax.plot(
                        iters,
                        legacy,
                        color="gray",
                        linestyle="--",
                        linewidth=1.2,
                        label="Legacy mean_reward",
                    )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Reward")
            ax.set_title("Aggregate Reward")
            ax.legend(fontsize=8)

        panel_builders.append(("aggregate_reward", _plot_aggregate_reward))

    for col, label, color in reward_cols:
        if col not in df.columns:
            continue

        def _make_reward_panel(
            metric_col: str = col,
            panel_label: str = label,
            panel_color: str = color,
        ) -> Any:
            def _plot_reward(ax: Any) -> None:
                series = df[metric_col]
                ax.plot(iters, series, color=panel_color, alpha=0.35, linewidth=1.0)
                smoothed = _smoothed(series)
                if smoothed is not None:
                    ax.plot(iters, smoothed, color=panel_color, linewidth=2.0)
                else:
                    ax.plot(iters, series, color=panel_color, linewidth=1.8)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Reward")
                ax.set_title(f"{panel_label} Reward")

            return _plot_reward

        panel_builders.append((f"{col}_panel", _make_reward_panel()))

    has_value_loss = any(c.endswith("_value_loss") for c in df.columns)
    has_explained_var = any(c.endswith("_explained_variance") for c in df.columns)
    if has_value_loss or has_explained_var:
        def _plot_critic_diagnostics(ax: Any) -> None:
            lines = []
            labels = []
            ax2 = None
            for col in sorted(c for c in df.columns if c.endswith("_value_loss")):
                agent = col.replace("_value_loss", "")
                color = color_by_agent.get(agent)
                (line,) = ax.plot(
                    iters,
                    df[col],
                    color=color,
                    linewidth=1.5,
                    label=f"{agent} value loss",
                )
                lines.append(line)
                labels.append(f"{agent} value loss")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value Loss")
            if has_explained_var:
                ax2 = ax.twinx()
                for col in sorted(c for c in df.columns if c.endswith("_explained_variance")):
                    agent = col.replace("_explained_variance", "")
                    color = color_by_agent.get(agent)
                    (line,) = ax2.plot(
                        iters,
                        df[col],
                        color=color,
                        linestyle="--",
                        linewidth=1.3,
                        label=f"{agent} EV",
                    )
                    lines.append(line)
                    labels.append(f"{agent} EV")
                ax2.set_ylabel("Explained Variance")
                ax2.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.set_title("Critic Diagnostics")
            if lines:
                ax.legend(lines, labels, fontsize=8, loc="best")

        panel_builders.append(("critic_diagnostics", _plot_critic_diagnostics))

    has_entropy = any(c.endswith("_entropy") for c in df.columns)
    has_kl = any(c.endswith("_approx_kl") for c in df.columns)
    if has_entropy or has_kl:
        def _plot_policy_diagnostics(ax: Any) -> None:
            lines = []
            labels = []
            ax2 = None
            for col in sorted(c for c in df.columns if c.endswith("_entropy")):
                agent = col.replace("_entropy", "")
                color = color_by_agent.get(agent)
                (line,) = ax.plot(
                    iters,
                    df[col],
                    color=color,
                    linewidth=1.5,
                    label=f"{agent} entropy",
                )
                lines.append(line)
                labels.append(f"{agent} entropy")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Entropy")
            if has_kl:
                ax2 = ax.twinx()
                for col in sorted(c for c in df.columns if c.endswith("_approx_kl")):
                    agent = col.replace("_approx_kl", "")
                    color = color_by_agent.get(agent)
                    (line,) = ax2.plot(
                        iters,
                        df[col],
                        color=color,
                        linestyle="--",
                        linewidth=1.3,
                        label=f"{agent} KL",
                    )
                    lines.append(line)
                    labels.append(f"{agent} KL")
                ax2.axhline(0.02, color="red", linestyle=":", linewidth=0.9, alpha=0.6)
                ax2.set_ylabel("Approx KL")
            ax.set_title("Policy Diagnostics")
            if lines:
                ax.legend(lines, labels, fontsize=8, loc="best")

        panel_builders.append(("policy_diagnostics", _plot_policy_diagnostics))

    has_top1_prob = any(c.endswith("_top1_prob") for c in df.columns)
    has_entropy_gap = any(c.endswith("_entropy_gap_from_uniform") for c in df.columns)
    if has_top1_prob or has_entropy_gap:
        def _plot_policy_confidence(ax: Any) -> None:
            lines = []
            labels = []
            ax2 = None
            for col in sorted(c for c in df.columns if c.endswith("_top1_prob")):
                agent = col.replace("_top1_prob", "")
                color = color_by_agent.get(agent)
                (line,) = ax.plot(
                    iters,
                    df[col],
                    color=color,
                    linewidth=1.5,
                    label=f"{agent} top1",
                )
                lines.append(line)
                labels.append(f"{agent} top1")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Top-1 Probability")
            ax.set_ylim(bottom=0.0)
            if has_entropy_gap:
                ax2 = ax.twinx()
                for col in sorted(
                    c for c in df.columns if c.endswith("_entropy_gap_from_uniform")
                ):
                    agent = col.replace("_entropy_gap_from_uniform", "")
                    color = color_by_agent.get(agent)
                    (line,) = ax2.plot(
                        iters,
                        df[col],
                        color=color,
                        linestyle="--",
                        linewidth=1.3,
                        label=f"{agent} gap",
                    )
                    lines.append(line)
                    labels.append(f"{agent} gap")
                ax2.set_ylabel("Entropy Gap")
                ax2.set_ylim(bottom=0.0)
            ax.set_title("Policy Confidence")
            if lines:
                ax.legend(lines, labels, fontsize=8, loc="best")

        panel_builders.append(("policy_confidence", _plot_policy_confidence))

    n_panels = max(len(panel_builders), 1)
    ncols = 2 if n_panels > 1 else 1
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle("MAPPO Training Progress", fontsize=13)
    axes_flat = list(axes.flat)

    for ax, (_, builder) in zip(axes_flat, panel_builders):
        builder(ax)
    for ax in axes_flat[len(panel_builders):]:
        ax.set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
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


def plot_time_series_diagnostics(
    step_log: Any,
    out_path: str | None = None,
    time_col: str = "t",
    column_group: str = "all",
    title: str | None = None,
) -> None:
    """Plot varying numeric columns from a per-step diagnostics trace.

    Parameters
    ----------
    step_log:
        Per-step diagnostics DataFrame or record list.
    out_path:
        If provided, save the figure to this path.
    time_col:
        X-axis column (default ``"t"``).
    column_group:
        Which variable family to plot: ``"all"``, ``"aggregate"``,
        ``"vessel"``, ``"port"``, or ``"coordinator"``.
    title:
        Optional figure title override.
    """
    import math
    import re

    import pandas as pd

    df = pd.DataFrame(step_log) if not isinstance(step_log, pd.DataFrame) else step_log
    if df.empty or time_col not in df.columns:
        return

    valid_groups = {"all", "aggregate", "vessel", "port", "coordinator"}
    if column_group not in valid_groups:
        raise ValueError(
            f"unknown column_group={column_group!r}; expected one of {sorted(valid_groups)}"
        )

    entity_patterns = {
        "vessel": re.compile(r"^vessel_\d+_"),
        "port": re.compile(r"^port_\d+_"),
        "coordinator": re.compile(r"^coordinator_\d+_"),
    }

    def _matches_group(col: str) -> bool:
        if column_group == "all":
            return True
        if column_group == "aggregate":
            return not any(pattern.match(col) for pattern in entity_patterns.values())
        return bool(entity_patterns[column_group].match(col))

    x = df[time_col]
    numeric_cols: list[tuple[str, Any]] = []
    for col in df.columns:
        if col == time_col:
            continue
        if not _matches_group(col):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        numeric_cols.append((col, series))

    def _sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        col = item[0]
        if col.startswith("vessel_"):
            return (1, col)
        if col.startswith("port_"):
            return (2, col)
        if col.startswith("coordinator_"):
            return (3, col)
        return (0, col)

    numeric_cols.sort(key=_sort_key)

    figure_title = title or {
        "all": "Per-Step Diagnostics",
        "aggregate": "Per-Step Diagnostics - Aggregate",
        "vessel": "Per-Step Diagnostics - Vessels",
        "port": "Per-Step Diagnostics - Ports",
        "coordinator": "Per-Step Diagnostics - Coordinators",
    }[column_group]

    if not numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No varying numeric diagnostics", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(figure_title, fontsize=13)
        _save_or_show(fig, out_path)
        return

    nvars = len(numeric_cols)
    if nvars <= 4:
        ncols = 2
    elif nvars <= 15:
        ncols = 3
    elif nvars <= 40:
        ncols = 4
    else:
        ncols = 5
    nrows = int(math.ceil(nvars / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 2.6 * nrows),
        squeeze=False,
    )
    fig.suptitle(figure_title, fontsize=13)
    axes_flat = list(axes.flat)

    for ax, (col, series) in zip(axes_flat, numeric_cols):
        ax.plot(x, series, linewidth=1.4, color="steelblue")
        ax.set_title(col.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel(time_col)
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[len(numeric_cols):]:
        ax.set_visible(False)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
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
    ax.axhline(y=0.02, color="red", linestyle="--", alpha=0.5, label="target")
    if kl_cols:
        ax.legend(fontsize=8)

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


def plot_multi_seed_curves(
    multi_seed_result: dict[str, Any],
    metric: str = "mean_reward",
    title: str | None = None,
    out_path: str | None = None,
) -> None:
    """Plot learning curves with confidence bands across multiple seeds.

    Parameters
    ----------
    multi_seed_result:
        Output from ``train_multi_seed()`` containing ``histories``
        and ``seeds`` keys.
    metric:
        Per-iteration metric to plot (default ``"mean_reward"``).
    title:
        Optional plot title.
    out_path:
        If provided, save figure to this path.
    """
    import numpy as np

    histories = multi_seed_result["histories"]
    seeds = multi_seed_result.get("seeds", list(range(len(histories))))
    max_len = max(len(h) for h in histories) if histories else 0
    if max_len == 0:
        return

    # Build (num_seeds x max_iters) matrix
    mat = np.full((len(histories), max_len), np.nan, dtype=np.float64)
    for si, hist in enumerate(histories):
        for ti, entry in enumerate(hist):
            mat[si, ti] = entry.get(metric, np.nan)

    iters = np.arange(max_len)
    with np.errstate(all="ignore"):
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual seed traces (translucent)
    for si in range(len(histories)):
        valid = ~np.isnan(mat[si])
        ax.plot(
            iters[valid], mat[si][valid],
            alpha=0.25, linewidth=0.8, color="steelblue",
        )

    # Mean ± 1 std band
    valid = ~np.isnan(mean)
    ax.plot(iters[valid], mean[valid], color="darkblue", linewidth=2, label="mean")
    ax.fill_between(
        iters[valid],
        (mean - std)[valid],
        (mean + std)[valid],
        alpha=0.2, color="steelblue", label="± 1 std",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(fontsize=9)
    ax.set_title(title or f"Multi-Seed Training ({len(seeds)} seeds)")
    plt.tight_layout()
    _save_or_show(fig, out_path)


def plot_timing_breakdown(
    history: list[dict[str, Any]],
    out_path: str | None = None,
) -> None:
    """Stacked area chart of rollout vs update time per iteration.

    Parameters
    ----------
    history:
        Per-iteration log from ``MAPPOTrainer.train()`` with
        ``rollout_time`` and ``update_time`` keys.
    """
    import pandas as pd

    df = pd.DataFrame(history)
    if "rollout_time" not in df.columns or "update_time" not in df.columns:
        return
    iters = df["iteration"] if "iteration" in df.columns else range(len(df))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(
        iters,
        df["rollout_time"],
        df["update_time"],
        labels=["Rollout", "Update"],
        alpha=0.7,
        colors=["#4c72b0", "#dd8452"],
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (s)")
    ax.set_title("Training Time Breakdown")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, out_path)


def _isnan(val: Any) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Episode snapshot collection for animated replay
# ---------------------------------------------------------------------------


def collect_episode_snapshots(
    policy_type: str = "forecast",
    steps: int | None = None,
    seed: int = 42,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one episode and capture per-step snapshots for animated replay.

    Each snapshot captures the *full* multi-agent state needed to tell the
    coordination story:

    * **Vessel state** — position, fuel, emissions, schedule compliance,
      stall/pending flags, mission status, and the action the vessel chose.
    * **Port state** — queue composition (which vessels), dock occupancy,
      utilisation, and the port's admission decisions.
    * **Coordinator directives** — destination, departure window, emission
      budget — the three levers the coordinator pulls.
    * **Weather** — full route matrix plus per-edge fuel multiplier.
    * **Reward breakdown** — per-component for all three agent levels.
    * **Fleet KPIs** — on-time rate, queue imbalance, fleet emissions,
      stalled count, utilisation rate.
    * **Events** — structured log of dispatches, arrivals, stalls, refuels,
      slot accepts/rejects that occurred during this step.

    Parameters
    ----------
    policy_type:
        Heuristic policy to use (``"forecast"``, ``"reactive"``, etc.).
    steps:
        Number of simulation steps. Uses config default if ``None``.
    seed:
        Random seed for reproducibility.
    config:
        Optional config overrides.

    Returns
    -------
    list of snapshot dicts, one per step.
    """
    import numpy as np

    from .config import get_default_config
    from .dynamics import weather_fuel_multiplier
    from .env import MaritimeEnv

    cfg = get_default_config(**(config or {}))
    steps = cfg["rollout_steps"] if steps is None else steps
    cfg["rollout_steps"] = int(steps)
    env = MaritimeEnv(config=cfg, seed=seed)
    env.reset()

    distance_nm = env.distance_nm
    num_ports = len(env.ports)
    weather_penalty = float(cfg.get("weather_penalty_factor", 0.15))
    snapshots: list[dict[str, Any]] = []

    for step_idx in range(steps):
        # ── Capture pre-step state ──────────────────────────────────────

        # Vessel state — everything that matters for the coordination story
        vessel_data = []
        for v in env.vessels:
            leg_dist = (
                float(distance_nm[v.location, v.destination])
                if v.location != v.destination else 0.0
            )
            progress = (
                float(v.position_nm) / leg_dist
                if leg_dist > 0 and v.at_sea
                else (0.0 if v.at_sea else -1.0)
            )
            vessel_data.append({
                # Identity & position
                "vessel_id": v.vessel_id,
                "location": v.location,
                "destination": v.destination,
                "position_nm": float(v.position_nm),
                "leg_distance": leg_dist,
                "progress": float(min(progress, 1.0)) if progress >= 0 else -1.0,
                # Propulsion
                "speed": float(v.speed),
                "fuel": float(v.fuel),
                "initial_fuel": float(v.initial_fuel),
                "fuel_frac": float(v.fuel) / max(float(v.initial_fuel), 1e-6),
                "cumulative_fuel_used": float(v.cumulative_fuel_used),
                "emissions": float(v.emissions),
                # Status flags
                "at_sea": bool(v.at_sea),
                "stalled": bool(v.stalled),
                "pending_departure": bool(v.pending_departure),
                "depart_at_step": int(v.depart_at_step),
                "port_service_state": int(v.port_service_state),
                # Delays & schedule
                "delay_hours": float(v.delay_hours),
                "schedule_delay_hours": float(v.schedule_delay_hours),
                "last_schedule_delay_hours": float(v.last_schedule_delay_hours),
                "requested_arrival_time": float(v.requested_arrival_time),
                # Completion stats
                "completed_arrivals": int(v.completed_arrivals),
                "completed_scheduled_arrivals": int(v.completed_scheduled_arrivals),
                "on_time_arrivals": int(v.on_time_arrivals),
                # Mission state
                "mission_done": bool(v.mission_done),
                "mission_success": bool(v.mission_success),
                "mission_failed": bool(v.mission_failed),
            })

        # Port state — including which vessels are where
        port_data = []
        for p in env.ports:
            port_data.append({
                "port_id": p.port_id,
                "queue": p.queue,
                "docks": p.docks,
                "occupied": p.occupied,
                "available": max(p.docks - p.occupied, 0),
                "utilization": float(p.occupied) / max(p.docks, 1),
                "cumulative_wait_hours": float(p.cumulative_wait_hours),
                "vessels_served": p.vessels_served,
                "queued_vessel_ids": list(p.queued_vessel_ids),
                "servicing_vessel_ids": list(p.servicing_vessel_ids),
            })

        # Weather — matrix plus per-edge fuel multiplier
        weather_matrix = None
        fuel_multiplier_matrix = None
        if env._weather_enabled and env._weather is not None:
            weather_matrix = env._weather.copy()
            fuel_multiplier_matrix = np.zeros_like(weather_matrix)
            for i in range(num_ports):
                for j in range(num_ports):
                    fuel_multiplier_matrix[i, j] = weather_fuel_multiplier(
                        float(weather_matrix[i, j]), weather_penalty,
                    )

        # ── Take step ───────────────────────────────────────────────────
        actions = env.sample_stub_actions()
        coordinator_action = actions.get("coordinator", {})
        coordinator_actions_list = actions.get("coordinators", [coordinator_action])
        vessel_actions = actions.get("vessels", [])
        port_actions = actions.get("ports", [])
        _, rewards, done, info = env.step(actions)

        # ── Reward component breakdowns ─────────────────────────────────
        component_keys_vessel = [
            "fuel_cost", "delay_cost", "emission_cost", "transit_cost",
            "schedule_delay_cost", "arrival_bonus", "on_time_bonus",
            "weather_shaping_bonus",
        ]
        component_keys_port = [
            "wait_penalty", "idle_penalty", "accept_bonus",
            "reject_penalty", "service_bonus",
        ]
        component_keys_coord = [
            "fuel_penalty", "queue_penalty", "idle_penalty",
            "utilization_bonus", "delay_penalty", "schedule_delay_penalty",
            "emission_penalty", "accept_bonus", "reject_penalty",
            "throughput_bonus", "weather_shaping_bonus",
        ]

        vessel_rc: dict[str, float] = {}
        for comp_key in component_keys_vessel:
            vessel_rc[comp_key] = float(
                info.get(f"vessel_reward_{comp_key}_total", 0.0)
            )
        port_rc: dict[str, float] = {}
        for comp_key in component_keys_port:
            port_rc[comp_key] = float(
                info.get(f"port_reward_{comp_key}_total", 0.0)
            )
        coord_rc: dict[str, float] = {}
        for comp_key in component_keys_coord:
            coord_rc[comp_key] = float(
                info.get(f"coordinator_reward_{comp_key}_total", 0.0)
            )

        # ── Fleet-level KPIs ────────────────────────────────────────────
        queues = [p.queue for p in env.ports]
        queue_std = float(np.std(queues)) if len(queues) > 1 else 0.0

        fleet_kpis = {
            "on_time_rate": float(info.get("on_time_rate", 0.0)),
            "queue_imbalance_std": queue_std,
            "fleet_emissions": float(sum(v.emissions for v in env.vessels)),
            "fleet_fuel_used": float(sum(v.cumulative_fuel_used for v in env.vessels)),
            "stalled_count": int(sum(bool(v.stalled) for v in env.vessels)),
            "vessel_utilization_rate": float(info.get("vessel_utilization_rate", 0.0)),
            "active_vessel_count": int(info.get("active_vessel_count", 0)),
            "total_delay_hours": float(sum(v.delay_hours for v in env.vessels)),
            "total_schedule_delay_hours": float(
                sum(v.schedule_delay_hours for v in env.vessels)
            ),
        }

        # ── Build snapshot ──────────────────────────────────────────────
        snapshot = {
            "t": step_idx,
            "vessels": vessel_data,
            "ports": port_data,
            "weather": weather_matrix,
            "fuel_multiplier": fuel_multiplier_matrix,
            "distance_nm": distance_nm.copy(),
            "rewards": {
                "vessels": list(rewards["vessels"]),
                "ports": list(rewards["ports"]),
                "coordinator": float(rewards["coordinator"]),
            },
            "reward_components": {
                "vessel": vessel_rc,
                "port": port_rc,
                "coordinator": coord_rc,
            },
            "actions": {
                "coordinator": {
                    "dest_port": int(coordinator_action.get("dest_port", 0)),
                    "departure_window_hours": int(
                        coordinator_action.get("departure_window_hours", 0)
                    ),
                    "emission_budget": float(
                        coordinator_action.get("emission_budget", 0.0)
                    ),
                },
                "vessels": [
                    {
                        "target_speed": float(va.get("target_speed", 0.0)),
                        "request_arrival_slot": bool(va.get("request_arrival_slot", False)),
                        "requested_arrival_time": float(va.get("requested_arrival_time", 0.0)),
                    }
                    for va in vessel_actions
                ],
                "ports": [
                    {
                        "service_rate": int(pa.get("service_rate", 1)),
                        "accept_requests": int(pa.get("accept_requests", 0)),
                    }
                    for pa in port_actions
                ],
            },
            # Back-compat alias
            "coordinator_action": {
                "dest_port": int(coordinator_action.get("dest_port", 0)),
            },
            "info": {
                "step_fuel_used": float(info.get("step_fuel_used", 0.0)),
                "step_co2_emitted": float(info.get("step_co2_emitted", 0.0)),
                "step_vessels_served": float(info.get("step_vessels_served", 0.0)),
                "avg_queue": float(
                    info.get("port_metrics", {}).get("avg_queue", 0.0)
                ),
                "step_delay_hours": float(info.get("step_delay_hours", 0.0)),
                "step_schedule_delay_hours": float(
                    info.get("step_schedule_delay_hours", 0.0)
                ),
                "step_stall_hours": float(info.get("step_stall_hours", 0.0)),
                "step_on_time_arrivals": float(
                    info.get("step_on_time_arrivals", 0.0)
                ),
                "requests_submitted": float(info.get("requests_submitted", 0.0)),
                "requests_accepted": float(info.get("requests_accepted", 0.0)),
                "requests_rejected": float(info.get("requests_rejected", 0.0)),
            },
            "fleet_kpis": fleet_kpis,
            "events": list(info.get("events", [])),
            "done": done,
        }
        snapshots.append(snapshot)
        if done:
            break

    return snapshots


# ---------------------------------------------------------------------------
# Animated spatial replay
# ---------------------------------------------------------------------------


def _port_positions(num_ports: int) -> dict[int, tuple[float, float]]:
    """Compute port positions arranged in a circle."""
    import math
    positions: dict[int, tuple[float, float]] = {}
    for i in range(num_ports):
        angle = 2 * math.pi * i / num_ports - math.pi / 2
        positions[i] = (math.cos(angle), math.sin(angle))
    return positions


def _vessel_xy(
    vessel: dict[str, Any],
    port_pos: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """Compute vessel (x, y) position: at port or interpolated along edge."""
    if not vessel["at_sea"] or vessel["progress"] < 0:
        # Docked / queued / pending — at the port location
        loc = vessel["location"]
        px, py = port_pos[loc]
        # Slight offset so multiple docked vessels don't overlap perfectly
        offset = 0.03 * (vessel["vessel_id"] % 5 - 2)
        return (px + offset, py + offset)
    # In transit — linear interpolation along the edge
    src = vessel["location"]
    dst = vessel["destination"]
    sx, sy = port_pos[src]
    dx, dy = port_pos[dst]
    t = max(0.0, min(vessel["progress"], 1.0))
    return (sx + t * (dx - sx), sy + t * (dy - sy))


def _interpolate_snapshots(
    snapshots: list[dict[str, Any]],
    interp_factor: int,
) -> list[dict[str, Any]]:
    """Create interpolated sub-frames between consecutive snapshots.

    Numeric vessel/port fields are linearly interpolated.  Non-numeric,
    structural, and list fields are held constant from the *earlier*
    snapshot until the next key-frame.  The ``_source_step`` key records
    the original simulation step index.
    """
    import numpy as np

    # Fields that must NOT be interpolated (discrete state, identity, lists)
    _VESSEL_HOLD = {
        "vessel_id", "location", "destination", "at_sea", "stalled",
        "pending_departure", "depart_at_step", "port_service_state",
        "completed_arrivals", "completed_scheduled_arrivals",
        "on_time_arrivals", "mission_done", "mission_success",
        "mission_failed",
    }
    _PORT_HOLD = {
        "port_id", "docks", "vessels_served", "queued_vessel_ids",
        "servicing_vessel_ids",
    }

    if interp_factor <= 1 or len(snapshots) < 2:
        for s in snapshots:
            s["_source_step"] = s["t"]
        return snapshots

    interp: list[dict[str, Any]] = []
    for si in range(len(snapshots) - 1):
        s0 = snapshots[si]
        s1 = snapshots[si + 1]
        for k in range(interp_factor):
            alpha = k / interp_factor
            frame: dict[str, Any] = {
                "t": s0["t"],
                "_source_step": s0["t"],
                "done": s0["done"],
                "distance_nm": s0["distance_nm"],
            }
            # Pass through all non-interpolatable top-level keys
            for pass_key in (
                "coordinator_action", "actions", "info", "rewards",
                "reward_components", "fleet_kpis", "events",
                "fuel_multiplier",
            ):
                if pass_key in s0:
                    frame[pass_key] = s0[pass_key]

            # Interpolate vessels
            vessels_interp = []
            for v0, v1 in zip(s0["vessels"], s1["vessels"]):
                vi: dict[str, Any] = {}
                for key in v0:
                    if key in _VESSEL_HOLD or not isinstance(v0[key], (int, float)):
                        vi[key] = v0[key]
                    else:
                        vi[key] = float(v0[key]) + alpha * (
                            float(v1[key]) - float(v0[key])
                        )
                vessels_interp.append(vi)
            frame["vessels"] = vessels_interp

            # Interpolate ports
            ports_interp = []
            for p0, p1 in zip(s0["ports"], s1["ports"]):
                pi: dict[str, Any] = {}
                for key in p0:
                    if key in _PORT_HOLD or not isinstance(p0[key], (int, float)):
                        pi[key] = p0[key]
                    else:
                        pi[key] = float(p0[key]) + alpha * (
                            float(p1[key]) - float(p0[key])
                        )
                ports_interp.append(pi)
            frame["ports"] = ports_interp

            # Interpolate weather
            if (
                s0.get("weather") is not None
                and s1.get("weather") is not None
            ):
                frame["weather"] = (
                    s0["weather"] * (1 - alpha) + s1["weather"] * alpha
                )
            else:
                frame["weather"] = s0.get("weather")

            interp.append(frame)

    # Append the final snapshot as-is
    last = snapshots[-1].copy()
    last["_source_step"] = last["t"]
    interp.append(last)
    return interp


def _weather_to_grid(
    weather_matrix: Any,
    port_pos: dict[int, tuple[float, float]],
    resolution: int = 40,
) -> tuple[Any, Any, Any]:
    """Convert a port-to-port weather matrix into a spatial grid for contourf.

    Uses vectorised inverse-distance weighting from port positions to create
    a smooth 2-D field.
    """
    import numpy as np

    num_ports = weather_matrix.shape[0]
    xs = np.linspace(-1.5, 1.5, resolution)
    ys = np.linspace(-1.5, 1.5, resolution)
    xg, yg = np.meshgrid(xs, ys)

    port_sea = np.array([
        float(np.mean(weather_matrix[p, :])) for p in range(num_ports)
    ])
    port_xy = np.array([port_pos[p] for p in range(num_ports)])

    # Vectorised IDW: (resolution, resolution, num_ports)
    dx = port_xy[:, 0][np.newaxis, np.newaxis, :] - xg[:, :, np.newaxis]
    dy = port_xy[:, 1][np.newaxis, np.newaxis, :] - yg[:, :, np.newaxis]
    dists = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    weights = 1.0 / (dists ** 2)
    grid = np.sum(weights * port_sea[np.newaxis, np.newaxis, :], axis=2) / np.sum(
        weights, axis=2
    )
    return xg, yg, grid


def plot_animated_replay(
    snapshots: list[dict[str, Any]],
    out_path: str | None = None,
    fps: int = 3,
    figsize: tuple[float, float] = (18, 10),
    dpi: int = 120,
    interp_factor: int = 4,
    wake_length: int = 8,
) -> Any:
    """Create an animated replay of a multi-agent maritime episode.

    The animation tells the **coordination story** of the HMARL system:

    **Map panel (left):**
      Vessels as diamonds coloured by fuel level, with status badges
      showing stall (X), pending departure (clock), and on-time compliance.
      Ports as circles sized by queue depth with stacked queue/dock bars.
      Edges coloured by weather fuel penalty.  Coordinator arrow shows
      the recommended destination.  Trailing wakes show vessel paths.

    **Dashboard panel (right):**
      Four sub-panels track the key dynamics:

      - *Coordinator directives* — destination, departure window, emission
        budget, and slot request flow.
      - *Fleet KPIs* — on-time rate, queue imbalance, fleet emissions,
        utilisation over time.
      - *Reward sparklines* — cumulative per-level rewards.
      - *Step metrics* — fuel, speed, delays, weather.

    **Timeline (bottom):**
      Progress bar with event markers for arrivals and stalls.

    Parameters
    ----------
    snapshots:
        List of per-step snapshot dicts from ``collect_episode_snapshots()``.
    out_path:
        Save destination (``.gif``, ``.mp4``, etc.) or ``None`` to return
        the ``FuncAnimation`` object.
    fps:
        Base frames per second (before interpolation).
    figsize:
        Figure size in inches ``(width, height)``.
    dpi:
        Resolution for saved output.
    interp_factor:
        Sub-frames between simulation steps for smooth motion.
    wake_length:
        Trailing positions drawn behind each vessel.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import matplotlib
    import numpy as np
    from matplotlib import animation
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    if not snapshots:
        raise ValueError("snapshots list is empty")

    # --- Interpolate for smooth motion ---
    frames = _interpolate_snapshots(snapshots, interp_factor)
    effective_fps = fps * max(interp_factor, 1)

    num_ports = len(snapshots[0]["ports"])
    num_vessels = len(snapshots[0]["vessels"])
    port_pos = _port_positions(num_ports)
    has_weather = snapshots[0].get("weather") is not None
    has_actions = "actions" in snapshots[0]
    has_fleet_kpis = "fleet_kpis" in snapshots[0]

    # --- Pre-compute time series for sparklines ---
    cum_vessel_r: list[float] = []
    cum_port_r: list[float] = []
    cum_coord_r: list[float] = []
    ts_on_time: list[float] = []
    ts_queue_imbal: list[float] = []
    ts_emissions: list[float] = []
    ts_util: list[float] = []
    cv = cp = cc = 0.0
    for s in snapshots:
        rew = s["rewards"]
        cv += float(np.mean(rew["vessels"])) if rew["vessels"] else 0.0
        cp += float(np.mean(rew["ports"])) if rew["ports"] else 0.0
        cc += float(rew["coordinator"])
        cum_vessel_r.append(cv)
        cum_port_r.append(cp)
        cum_coord_r.append(cc)
        kpis = s.get("fleet_kpis", {})
        ts_on_time.append(float(kpis.get("on_time_rate", 0.0)))
        ts_queue_imbal.append(float(kpis.get("queue_imbalance_std", 0.0)))
        ts_emissions.append(float(kpis.get("fleet_emissions", 0.0)))
        ts_util.append(float(kpis.get("vessel_utilization_rate", 0.0)))

    # --- Pre-compute edge data ---
    dist_matrix = snapshots[0]["distance_nm"]
    edges: list[tuple[int, int]] = []
    for i in range(num_ports):
        for j in range(i + 1, num_ports):
            edges.append((i, j))

    # =====================================================================
    #  FIGURE LAYOUT
    #
    #  +------------------------------+-------------+
    #  |                              | Coordinator |
    #  |                              | Directives  |
    #  |          MAP                 |-------------|
    #  |                              | Fleet KPIs  |
    #  |                              |-------------|
    #  |                              | Rewards     |
    #  |                              |-------------|
    #  |                              | Metrics     |
    #  +------------------------------+-------------+
    #  |           TIMELINE / PROGRESS BAR          |
    #  +--------------------------------------------+
    # =====================================================================
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(
        2, 2, width_ratios=[3, 1.2], height_ratios=[20, 1],
        wspace=0.08, hspace=0.06,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_prog = fig.add_subplot(gs[1, :])

    # Right panel: 4 stacked sub-panels
    gs_right = gs[0, 1].subgridspec(4, 1, hspace=0.45)
    ax_coord = fig.add_subplot(gs_right[0])   # Coordinator directives
    ax_kpi = fig.add_subplot(gs_right[1])     # Fleet KPIs sparklines
    ax_reward = fig.add_subplot(gs_right[2])  # Reward sparklines
    ax_metric = fig.add_subplot(gs_right[3])  # Step metrics text

    # --- Map axis ---
    ax_map.set_xlim(-1.6, 1.6)
    ax_map.set_ylim(-1.6, 1.6)
    ax_map.set_aspect("equal")
    ax_map.set_facecolor("#f0f4f8")
    ax_map.axis("off")
    title_text = ax_map.set_title("", fontsize=13, fontweight="bold", pad=12)

    # --- Progress bar ---
    total_steps = len(snapshots)
    ax_prog.set_xlim(0, total_steps)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_facecolor("#ecf0f1")
    ax_prog.set_yticks([])
    ax_prog.set_xlabel("Simulation step", fontsize=8)
    ax_prog.tick_params(axis="x", labelsize=7)
    progress_bar = ax_prog.barh(
        0.5, 0, height=0.8, color="#3498db", alpha=0.7, align="center",
    )[0]
    # Event markers on timeline
    for si, s in enumerate(snapshots):
        for ev in s.get("events", []):
            if ev.get("event_type") == "vessel_arrived":
                ax_prog.axvline(si, color="#2ecc71", alpha=0.4, lw=0.8)
            elif ev.get("event_type") == "mission_failed":
                ax_prog.axvline(si, color="#e74c3c", alpha=0.6, lw=1.2)

    # --- Colormaps ---
    vessel_cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    port_cmap = matplotlib.colormaps.get_cmap("YlOrRd")
    weather_edge_cmap = matplotlib.colormaps.get_cmap("Reds")

    # =====================================================================
    #  MAP: STATIC ELEMENTS
    # =====================================================================

    # Port labels
    for pid, (px, py) in port_pos.items():
        ax_map.text(
            px, py + 0.18, f"P{pid}", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2c3e50", zorder=20,
        )

    # Edge lines — coloured by weather fuel penalty (updated per frame)
    edge_lines: list[Any] = []
    for i_port, j_port in edges:
        xi, yi = port_pos[i_port]
        xj, yj = port_pos[j_port]
        line, = ax_map.plot(
            [xi, xj], [yi, yj], color="#bdc3c7", linewidth=1.5,
            zorder=1, alpha=0.4, solid_capstyle="round",
        )
        edge_lines.append(line)

    # Weather edge overlays (thicker red for rough weather)
    weather_lines: list[Any] = []
    if has_weather:
        for _ in edges:
            line, = ax_map.plot(
                [], [], color="#e74c3c", linewidth=0, zorder=2, alpha=0.0,
            )
            weather_lines.append(line)

    # Weather heatmap placeholder
    weather_contour: list[Any] = [None]

    # Port scatter
    port_scatter = ax_map.scatter(
        [port_pos[i][0] for i in range(num_ports)],
        [port_pos[i][1] for i in range(num_ports)],
        s=[200] * num_ports, c=["#3498db"] * num_ports,
        edgecolors="#2c3e50", linewidths=2, zorder=10, alpha=0.9,
    )

    # Port annotation: queue/dock counts
    port_texts: list[Any] = []
    for pid, (px, py) in port_pos.items():
        txt = ax_map.text(
            px, py - 0.18, "", ha="center", va="center",
            fontsize=7, color="#2c3e50", zorder=20, family="monospace",
        )
        port_texts.append(txt)

    # Vessel scatter (diamonds)
    vessel_scatter = ax_map.scatter(
        [0] * num_vessels, [0] * num_vessels,
        s=[80] * num_vessels, c=["#2ecc71"] * num_vessels,
        edgecolors="white", linewidths=1.2, zorder=15, marker="D",
    )

    # Vessel labels — show ID + status
    vessel_labels: list[Any] = []
    for _ in range(num_vessels):
        txt = ax_map.text(
            0, 0, "", fontsize=5.5, ha="center", va="bottom",
            color="#2c3e50", zorder=16, family="monospace",
        )
        vessel_labels.append(txt)

    # Vessel status badges (stall marker, pending marker)
    vessel_badge_texts: list[Any] = []
    for _ in range(num_vessels):
        txt = ax_map.text(
            0, 0, "", fontsize=7, ha="center", va="top",
            color="#e74c3c", fontweight="bold", zorder=17,
        )
        vessel_badge_texts.append(txt)

    # Trailing wakes
    vessel_histories: list[list[tuple[float, float]]] = [
        [] for _ in range(num_vessels)
    ]
    wake_collections: list[Any] = []
    for _ in range(num_vessels):
        lc = LineCollection([], linewidths=[], colors=[], zorder=14, alpha=0.6)
        ax_map.add_collection(lc)
        wake_collections.append(lc)
    wake_cmap = matplotlib.colormaps.get_cmap("Purples")

    # Coordinator arrow (FancyArrowPatch for blit-compat)
    coord_arrow = FancyArrowPatch(
        (0, 0), (0, 0),
        arrowstyle="->,head_length=6,head_width=4",
        color="#9b59b6", linewidth=2.5, zorder=25, visible=False,
    )
    ax_map.add_patch(coord_arrow)
    coord_label = ax_map.text(
        0, 0, "", fontsize=7, ha="center", va="bottom",
        color="#9b59b6", fontweight="bold", zorder=26,
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ecc71",
               markersize=7, label="Vessel (green=full fuel)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=9, label="Port (size=queue)"),
        Line2D([0], [0], color="#9b59b6", linewidth=2, alpha=0.7,
               label="Coord. dest"),
    ]
    if has_weather:
        legend_elements.append(
            Line2D([0], [0], color="#e74c3c", linewidth=3, alpha=0.5,
                   label="Rough weather"),
        )
    ax_map.legend(
        handles=legend_elements, loc="lower left", fontsize=7,
        framealpha=0.8, edgecolor="#bdc3c7",
    )

    # =====================================================================
    #  DASHBOARD: COORDINATOR DIRECTIVES PANEL
    # =====================================================================
    ax_coord.set_facecolor("#faf9f7")
    ax_coord.set_xlim(0, 1)
    ax_coord.set_ylim(0, 1)
    ax_coord.axis("off")
    ax_coord.set_title("Coordinator Directives", fontsize=9, fontweight="bold",
                        color="#2c3e50", pad=4)
    coord_texts: dict[str, Any] = {}
    coord_items = [
        ("dest", "Dest port:", 0.85),
        ("dep_win", "Dep. window:", 0.65),
        ("em_budget", "CO2 budget:", 0.45),
        ("slots", "Slot req/acc/rej:", 0.25),
    ]
    for key, label, y in coord_items:
        ax_coord.text(0.02, y, label, fontsize=7.5, color="#7f8c8d",
                      va="center", family="monospace")
        coord_texts[key] = ax_coord.text(
            0.98, y, "--", fontsize=7.5, color="#2c3e50", va="center",
            ha="right", family="monospace", fontweight="bold",
        )

    # =====================================================================
    #  DASHBOARD: FLEET KPI SPARKLINES
    # =====================================================================
    ax_kpi.set_facecolor("#faf9f7")
    ax_kpi.set_title("Fleet KPIs", fontsize=9, fontweight="bold",
                      color="#2c3e50", pad=4)
    ax_kpi.tick_params(labelsize=5)
    ax_kpi.set_xlim(0, max(total_steps - 1, 1))
    ax_kpi_twin = ax_kpi.twinx()
    ax_kpi_twin.tick_params(labelsize=5)

    kpi_on_time, = ax_kpi.plot([], [], color="#27ae60", lw=1.2, label="On-time %")
    kpi_util, = ax_kpi.plot([], [], color="#2980b9", lw=1.2, label="Util %")
    kpi_q_imbal, = ax_kpi_twin.plot([], [], color="#e67e22", lw=1.0,
                                     ls="--", label="Q imbal.")
    ax_kpi.set_ylabel("Rate", fontsize=6)
    ax_kpi_twin.set_ylabel("Std", fontsize=6)
    # Combined legend
    kpi_lines = [kpi_on_time, kpi_util, kpi_q_imbal]
    ax_kpi.legend(kpi_lines, [l.get_label() for l in kpi_lines],
                  fontsize=5, loc="lower left", framealpha=0.7)
    if ts_on_time:
        ax_kpi.set_ylim(-0.05, 1.05)
        max_q = max(ts_queue_imbal) if ts_queue_imbal else 1.0
        ax_kpi_twin.set_ylim(0, max(max_q * 1.3, 0.5))

    # =====================================================================
    #  DASHBOARD: CUMULATIVE REWARD SPARKLINES
    # =====================================================================
    ax_reward.set_facecolor("#faf9f7")
    ax_reward.set_title("Cumulative Rewards", fontsize=9, fontweight="bold",
                         color="#2c3e50", pad=4)
    ax_reward.tick_params(labelsize=5)
    ax_reward.set_xlim(0, max(total_steps - 1, 1))
    spark_vessel, = ax_reward.plot([], [], color="#2ecc71", lw=1.2, label="Vessel")
    spark_port, = ax_reward.plot([], [], color="#3498db", lw=1.2, label="Port")
    spark_coord, = ax_reward.plot([], [], color="#9b59b6", lw=1.2, label="Coord")
    ax_reward.legend(fontsize=5, loc="upper left", framealpha=0.7)
    if cum_vessel_r:
        all_vals = cum_vessel_r + cum_port_r + cum_coord_r
        ymin, ymax = min(all_vals), max(all_vals)
        margin = max(abs(ymax - ymin) * 0.1, 0.1)
        ax_reward.set_ylim(ymin - margin, ymax + margin)

    # =====================================================================
    #  DASHBOARD: STEP METRICS TEXT
    # =====================================================================
    ax_metric.set_facecolor("#faf9f7")
    ax_metric.set_xlim(0, 1)
    ax_metric.set_ylim(0, 1)
    ax_metric.axis("off")
    ax_metric.set_title("Step Metrics", fontsize=9, fontweight="bold",
                         color="#2c3e50", pad=4)

    metric_texts: dict[str, Any] = {}
    metric_items = [
        ("step", "Step:", 0.92),
        ("fuel", "Avg fuel:", 0.78),
        ("speed", "Avg speed:", 0.64),
        ("at_sea", "At sea:", 0.50),
        ("delay", "Cum delay:", 0.36),
        ("emit", "Fleet CO2:", 0.22),
        ("weather", "Sea state:", 0.08),
    ]
    for key, label, y in metric_items:
        ax_metric.text(0.02, y, label, fontsize=7.5, color="#7f8c8d",
                       va="center", family="monospace")
        metric_texts[key] = ax_metric.text(
            0.98, y, "--", fontsize=7.5, color="#2c3e50", va="center",
            ha="right", family="monospace", fontweight="bold",
        )

    # =====================================================================
    #  UPDATE FUNCTION
    # =====================================================================
    def _update(frame_idx: int) -> list[Any]:
        snap = frames[frame_idx]
        vessels = snap["vessels"]
        ports = snap["ports"]
        weather = snap.get("weather")
        rew = snap["rewards"]
        source_step = snap.get("_source_step", snap["t"])
        actions = snap.get("actions", {})
        kpis = snap.get("fleet_kpis", {})
        info = snap.get("info", {})

        title_text.set_text(f"Maritime HMARL  |  Step {source_step}")

        # ── Weather heatmap ──────────────────────────────────────────
        if has_weather and weather is not None:
            if weather_contour[0] is not None:
                try:
                    weather_contour[0].remove()
                except (AttributeError, NotImplementedError):
                    for coll in getattr(weather_contour[0], "collections", []):
                        coll.remove()
                weather_contour[0] = None
            xg, yg, grid = _weather_to_grid(weather, port_pos, resolution=35)
            weather_contour[0] = ax_map.contourf(
                xg, yg, grid, levels=10, cmap="YlOrRd",
                alpha=0.12, zorder=0,
            )

        # ── Edges: colour by weather fuel penalty ────────────────────
        for idx, (i, j) in enumerate(edges):
            xi, yi = port_pos[i]
            xj, yj = port_pos[j]
            edge_lines[idx].set_data([xi, xj], [yi, yj])
            if has_weather and weather is not None:
                sea = float(weather[i, j])
                penalty = 1.0 + 0.15 * sea  # fuel multiplier
                norm_p = min((penalty - 1.0) / 0.45, 1.0)  # 0..1
                edge_lines[idx].set_color(
                    weather_edge_cmap(0.15 + 0.7 * norm_p)
                )
                edge_lines[idx].set_linewidth(1.0 + 3.0 * norm_p)
                edge_lines[idx].set_alpha(0.3 + 0.4 * norm_p)
            else:
                d = float(dist_matrix[i, j])
                max_d = float(np.max(dist_matrix)) or 1.0
                nd = d / max_d
                edge_lines[idx].set_color(f"#{int(180-60*nd):02x}{int(195-60*nd):02x}{int(199-40*nd):02x}")
                edge_lines[idx].set_linewidth(1.0 + 2.0 * nd)
                edge_lines[idx].set_alpha(0.4)

        # ── Weather edge overlays ────────────────────────────────────
        if has_weather and weather is not None:
            sea_max = float(np.max(weather)) if np.max(weather) > 0 else 1.0
            for idx, (i, j) in enumerate(edges):
                sea = float(weather[i, j])
                xi, yi = port_pos[i]
                xj, yj = port_pos[j]
                weather_lines[idx].set_data([xi, xj], [yi, yj])
                normalised = min(sea / max(sea_max, 1e-6), 1.0)
                weather_lines[idx].set_linewidth(normalised * 5.0)
                weather_lines[idx].set_alpha(normalised * 0.45)
                weather_lines[idx].set_color(
                    weather_edge_cmap(0.3 + 0.7 * normalised)
                )

        # ── Ports ────────────────────────────────────────────────────
        queue_vals = [p["queue"] for p in ports]
        util_vals = [p["utilization"] for p in ports]
        sizes = [200 + 120 * q for q in queue_vals]
        colors = [port_cmap(u * 0.8) for u in util_vals]
        port_scatter.set_sizes(sizes)
        port_scatter.set_facecolor(colors)

        for pid, p in enumerate(ports):
            q = int(round(p["queue"]))
            occ = int(round(p["occupied"]))
            docks = p["docks"]
            # Show queue, dock usage, and which vessels are servicing
            svc_ids = p.get("servicing_vessel_ids", [])
            svc_str = ",".join(
                f"V{vid}" for vid in svc_ids if vid >= 0
            ) if svc_ids else "-"
            port_texts[pid].set_text(
                f"Q:{q} D:{occ}/{docks} [{svc_str}]"
            )

        # ── Vessels: position, colour, badges ────────────────────────
        xs, ys = [], []
        fuel_colors = []
        edge_colors = []
        for vi, v in enumerate(vessels):
            vx, vy = _vessel_xy(v, port_pos)
            xs.append(vx)
            ys.append(vy)
            fuel_colors.append(vessel_cmap(v["fuel_frac"]))

            # Edge colour: red for stalled, orange for pending, white normal
            if v.get("stalled"):
                edge_colors.append("#e74c3c")
            elif v.get("pending_departure"):
                edge_colors.append("#f39c12")
            else:
                edge_colors.append("white")

            # Wake trail
            vessel_histories[vi].append((vx, vy))
            if len(vessel_histories[vi]) > wake_length:
                vessel_histories[vi] = vessel_histories[vi][-wake_length:]
            hist = vessel_histories[vi]
            if len(hist) >= 2:
                segments = [
                    [hist[k], hist[k + 1]] for k in range(len(hist) - 1)
                ]
                n_seg = len(segments)
                alphas = [0.1 + 0.5 * (k / max(n_seg - 1, 1)) for k in range(n_seg)]
                widths = [0.5 + 1.5 * (k / max(n_seg - 1, 1)) for k in range(n_seg)]
                clrs = [
                    (*wake_cmap(0.4 + 0.5 * (k / max(n_seg - 1, 1)))[:3], a)
                    for k, a in enumerate(alphas)
                ]
                wake_collections[vi].set_segments(segments)
                wake_collections[vi].set_linewidths(widths)
                wake_collections[vi].set_color(clrs)
            else:
                wake_collections[vi].set_segments([])

        vessel_scatter.set_offsets(list(zip(xs, ys)))
        vessel_scatter.set_facecolor(fuel_colors)
        vessel_scatter.set_edgecolor(edge_colors)

        # Vessel labels: ID + speed or status
        for vi, v in enumerate(vessels):
            vid = v["vessel_id"]
            parts = [f"V{vid}"]
            if v["at_sea"]:
                parts.append(f"{v['speed']:.0f}kt")
            on_time = v.get("on_time_arrivals", 0)
            sched = v.get("completed_scheduled_arrivals", 0)
            if sched > 0:
                parts.append(f"{on_time}/{sched}OT")
            vessel_labels[vi].set_position((xs[vi], ys[vi] + 0.07))
            vessel_labels[vi].set_text(" ".join(parts))

        # Status badges below vessel
        for vi, v in enumerate(vessels):
            badge = ""
            colour = "#e74c3c"
            if v.get("stalled"):
                badge = "STALL"
                colour = "#e74c3c"
            elif v.get("pending_departure"):
                dep_step = v.get("depart_at_step", 0)
                wait = max(0, dep_step - source_step)
                badge = f"HOLD {wait}t" if wait > 0 else "HOLD"
                colour = "#f39c12"
            elif v.get("mission_failed"):
                badge = "FAIL"
                colour = "#c0392b"
            elif v.get("mission_success"):
                badge = "DONE"
                colour = "#27ae60"
            vessel_badge_texts[vi].set_position((xs[vi], ys[vi] - 0.07))
            vessel_badge_texts[vi].set_text(badge)
            vessel_badge_texts[vi].set_color(colour)

        # ── Coordinator arrow + label ────────────────────────────────
        coord_act = actions.get("coordinator", snap.get("coordinator_action", {}))
        dest = int(coord_act.get("dest_port", 0))
        dep_win = int(coord_act.get("departure_window_hours", 0))
        dx, dy = port_pos[dest]
        coord_arrow.set_positions((dx, dy + 0.38), (dx, dy + 0.22))
        coord_arrow.set_visible(True)
        coord_label.set_position((dx, dy + 0.42))
        coord_label.set_text(f"P{dest}" + (f" +{dep_win}h" if dep_win else ""))

        # ── Progress bar ─────────────────────────────────────────────
        progress_bar.set_width(source_step + 1)

        # ── Dashboard: Coordinator directives ────────────────────────
        em_budget = float(coord_act.get("emission_budget", 0.0))
        coord_texts["dest"].set_text(f"P{dest}")
        coord_texts["dep_win"].set_text(
            f"{dep_win}h" if dep_win else "none"
        )
        coord_texts["em_budget"].set_text(
            f"{em_budget:.1f} t" if em_budget > 0 else "none"
        )
        req = int(info.get("requests_submitted", 0))
        acc = int(info.get("requests_accepted", 0))
        rej = int(info.get("requests_rejected", 0))
        coord_texts["slots"].set_text(f"{req}/{acc}/{rej}")

        # ── Dashboard: KPI sparklines ────────────────────────────────
        si = min(source_step, len(ts_on_time) - 1)
        x_data = list(range(si + 1))
        kpi_on_time.set_data(x_data, ts_on_time[: si + 1])
        kpi_util.set_data(x_data, ts_util[: si + 1])
        kpi_q_imbal.set_data(x_data, ts_queue_imbal[: si + 1])

        # ── Dashboard: Reward sparklines ─────────────────────────────
        spark_vessel.set_data(x_data, cum_vessel_r[: si + 1])
        spark_port.set_data(x_data, cum_port_r[: si + 1])
        spark_coord.set_data(x_data, cum_coord_r[: si + 1])

        # ── Dashboard: Step metrics ──────────────────────────────────
        metric_texts["step"].set_text(f"{source_step}/{total_steps}")
        avg_fuel = float(np.mean([v["fuel_frac"] for v in vessels]))
        avg_speed = float(np.mean([v["speed"] for v in vessels]))
        n_sea = sum(1 for v in vessels if v["at_sea"])
        n_stall = sum(1 for v in vessels if v.get("stalled"))
        metric_texts["fuel"].set_text(f"{avg_fuel:.0%}")
        metric_texts["speed"].set_text(f"{avg_speed:.1f} kt")
        sea_str = f"{n_sea}/{num_vessels}"
        if n_stall:
            sea_str += f" ({n_stall} stalled)"
        metric_texts["at_sea"].set_text(sea_str)
        total_delay = float(kpis.get("total_delay_hours", 0.0))
        metric_texts["delay"].set_text(f"{total_delay:.1f}h")
        fleet_co2 = float(kpis.get("fleet_emissions", 0.0))
        metric_texts["emit"].set_text(f"{fleet_co2:.1f} t")
        if has_weather and weather is not None:
            metric_texts["weather"].set_text(f"{float(np.mean(weather)):.2f}")
        else:
            metric_texts["weather"].set_text("N/A")

        # ── Collect updated artists ──────────────────────────────────
        updated: list[Any] = list(edge_lines)
        if has_weather:
            updated.extend(weather_lines)
        updated.extend([
            port_scatter, vessel_scatter, title_text, coord_arrow,
            coord_label, progress_bar,
            spark_vessel, spark_port, spark_coord,
            kpi_on_time, kpi_util, kpi_q_imbal,
        ])
        updated.extend(port_texts)
        updated.extend(vessel_labels)
        updated.extend(vessel_badge_texts)
        updated.extend(wake_collections)
        updated.extend(coord_texts.values())
        updated.extend(metric_texts.values())
        return updated

    anim = animation.FuncAnimation(
        fig, _update,
        frames=len(frames),
        interval=1000 // effective_fps,
        blit=not has_weather,  # contourf blocks blitting
        repeat=True,
    )

    if out_path:
        suffix = str(out_path).rsplit(".", 1)[-1].lower()
        if suffix == "gif":
            anim.save(out_path, writer="pillow", fps=effective_fps, dpi=dpi)
        elif suffix in ("mp4", "webm", "avi"):
            anim.save(out_path, writer="ffmpeg", fps=effective_fps, dpi=dpi)
        else:
            anim.save(out_path, fps=effective_fps, dpi=dpi)
        plt.close(fig)

    return anim


# ---------------------------------------------------------------------------
# Reward decomposition stacked-area charts
# ---------------------------------------------------------------------------


def plot_reward_decomposition(
    snapshots: list[dict[str, Any]],
    out_path: str | None = None,
) -> None:
    """Plot stacked-area reward decomposition for all three agent types.

    Shows how each reward component (fuel cost, delay cost, arrival bonus,
    queue penalty, etc.) contributes to the total reward signal over time.
    Positive components are stacked above zero, negative below, making it
    immediately visible which terms dominate the reward.

    Parameters
    ----------
    snapshots:
        List of per-step snapshot dicts from ``collect_episode_snapshots()``.
    out_path:
        If provided, save the figure to this path.
    """
    import numpy as np

    if not snapshots:
        return

    steps = np.array([s["t"] for s in snapshots])

    # --- Define component groups for each agent type ---
    vessel_costs = ["fuel_cost", "delay_cost", "emission_cost", "transit_cost", "schedule_delay_cost"]
    vessel_bonuses = ["arrival_bonus", "on_time_bonus", "weather_shaping_bonus"]
    port_costs = ["wait_penalty", "idle_penalty", "reject_penalty"]
    port_bonuses = ["accept_bonus", "service_bonus"]
    coord_costs = [
        "fuel_penalty", "queue_penalty", "idle_penalty", "delay_penalty",
        "schedule_delay_penalty", "emission_penalty", "reject_penalty",
    ]
    coord_bonuses = ["utilization_bonus", "accept_bonus", "throughput_bonus", "weather_shaping_bonus"]

    def _extract_series(component_key: str, agent_type: str) -> np.ndarray:
        values = []
        for snap in snapshots:
            comps = snap.get("reward_components", {}).get(agent_type, {})
            values.append(float(comps.get(component_key, 0.0)))
        return np.array(values)

    def _plot_decomposition(
        ax: Any,
        title: str,
        cost_keys: list[str],
        bonus_keys: list[str],
        agent_type: str,
        total_values: np.ndarray,
    ) -> None:
        # Costs (negative, stacked below zero)
        cost_labels = []
        cost_data = []
        for key in cost_keys:
            series = _extract_series(key, agent_type)
            if np.any(np.abs(series) > 1e-9):
                cost_labels.append(key.replace("_", " ").title())
                cost_data.append(series)

        # Bonuses (positive, stacked above zero)
        bonus_labels = []
        bonus_data = []
        for key in bonus_keys:
            series = _extract_series(key, agent_type)
            if np.any(np.abs(series) > 1e-9):
                bonus_labels.append(key.replace("_", " ").title())
                bonus_data.append(series)

        # Color palettes
        cost_colors = ["#e74c3c", "#c0392b", "#e67e22", "#d35400", "#f39c12", "#e84393", "#6c5ce7"]
        bonus_colors = ["#2ecc71", "#27ae60", "#00b894", "#00cec9"]

        # Stack costs below zero
        if cost_data:
            cost_stack = np.vstack(cost_data)
            ax.stackplot(
                steps, -cost_stack,
                labels=[f"-{l}" for l in cost_labels],
                colors=cost_colors[:len(cost_data)],
                alpha=0.75,
                baseline="zero",
            )

        # Stack bonuses above zero
        if bonus_data:
            bonus_stack = np.vstack(bonus_data)
            ax.stackplot(
                steps, bonus_stack,
                labels=bonus_labels,
                colors=bonus_colors[:len(bonus_data)],
                alpha=0.75,
                baseline="zero",
            )

        # Total reward line
        ax.plot(
            steps, total_values,
            color="black", linewidth=2.0, linestyle="-",
            label="Total", zorder=10,
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.8)
        ax.grid(True, alpha=0.15)

    # Total rewards per step
    vessel_totals = np.array([
        float(np.mean(s["rewards"]["vessels"])) if s["rewards"]["vessels"] else 0.0
        for s in snapshots
    ])
    port_totals = np.array([
        float(np.mean(s["rewards"]["ports"])) if s["rewards"]["ports"] else 0.0
        for s in snapshots
    ])
    coord_totals = np.array([
        float(s["rewards"]["coordinator"]) for s in snapshots
    ])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Reward Decomposition by Agent Type", fontsize=14, fontweight="bold")

    _plot_decomposition(
        axes[0], "Vessel Reward Decomposition",
        vessel_costs, vessel_bonuses, "vessel", vessel_totals,
    )
    _plot_decomposition(
        axes[1], "Port Reward Decomposition",
        port_costs, port_bonuses, "port", port_totals,
    )
    _plot_decomposition(
        axes[2], "Coordinator Reward Decomposition",
        coord_costs, coord_bonuses, "coordinator", coord_totals,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Publication-quality defaults
# ---------------------------------------------------------------------------

_PUB_RC = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
}


def set_publication_style() -> None:
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update(_PUB_RC)


# ---------------------------------------------------------------------------
# Gradient and critic diagnostic plots
# ---------------------------------------------------------------------------


def plot_gradient_diagnostics(
    train_log: Any,
    out_path: str | None = None,
) -> None:
    """Plot gradient norms and clip fractions per agent type over training.

    Parameters
    ----------
    train_log:
        DataFrame or list of dicts with ``{agent}_grad_norm`` and
        ``{agent}_clip_frac`` columns.
    """
    import pandas as pd

    df = pd.DataFrame(train_log) if not isinstance(train_log, pd.DataFrame) else train_log
    if df.empty or "iteration" not in df.columns:
        return

    agent_colors = {
        "vessel": "#1b9e77",
        "port": "#d95f02",
        "coordinator": "#7570b3",
    }
    iters = df["iteration"]
    grad_cols = sorted(c for c in df.columns if c.endswith("_grad_norm"))
    clip_cols = sorted(c for c in df.columns if c.endswith("_clip_frac"))

    if not grad_cols and not clip_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Policy Gradient Diagnostics", fontsize=13)

    # Gradient norms
    ax = axes[0]
    for col in grad_cols:
        agent = col.replace("_grad_norm", "")
        color = agent_colors.get(agent, None)
        ax.plot(iters, df[col], label=agent, color=color, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms by Agent Type")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="max_grad_norm")

    # Clip fractions
    ax = axes[1]
    for col in clip_cols:
        agent = col.replace("_clip_frac", "")
        color = agent_colors.get(agent, None)
        ax.plot(iters, df[col], label=agent, color=color, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Clip Fraction")
    ax.set_title("PPO Clip Fractions by Agent Type")
    ax.legend(fontsize=9)
    ax.axhspan(0.05, 0.25, alpha=0.08, color="green", label="healthy range")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    _save_or_show(fig, out_path)


def plot_explained_variance(
    train_log: Any,
    out_path: str | None = None,
) -> None:
    """Plot explained variance of critics over training iterations.

    Explained variance (EV) measures how well the value function predicts
    returns.  EV = 1 means perfect prediction; EV < 0 means the value
    function is worse than predicting the mean.

    Parameters
    ----------
    train_log:
        DataFrame or list of dicts with ``{agent}_explained_variance``
        columns.
    """
    import pandas as pd

    df = pd.DataFrame(train_log) if not isinstance(train_log, pd.DataFrame) else train_log
    if df.empty or "iteration" not in df.columns:
        return

    agent_colors = {
        "vessel": "#1b9e77",
        "port": "#d95f02",
        "coordinator": "#7570b3",
    }
    iters = df["iteration"]
    ev_cols = sorted(c for c in df.columns if c.endswith("_explained_variance"))
    if not ev_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ev_cols:
        agent = col.replace("_explained_variance", "")
        color = agent_colors.get(agent, None)
        ax.plot(iters, df[col], label=agent, color=color, linewidth=1.8)
    ax.axhline(0.0, color="red", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Critic Explained Variance (Value Function Quality)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, 1.1)
    plt.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Pareto frontier: fuel vs delay trade-off
# ---------------------------------------------------------------------------


def plot_pareto_frontier(
    results: dict[str, dict[str, float]],
    x_metric: str = "total_fuel_used",
    y_metric: str = "avg_delay_hours",
    out_path: str | None = None,
) -> None:
    """Scatter plot of policies on a fuel-vs-delay Pareto frontier.

    Parameters
    ----------
    results:
        Dict mapping policy name -> dict of final evaluation metrics.
    x_metric, y_metric:
        Metric keys for the x and y axes.
    """
    import numpy as np

    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = []
    ys = []
    labels = []
    for name, metrics in results.items():
        if x_metric in metrics and y_metric in metrics:
            xs.append(float(metrics[x_metric]))
            ys.append(float(metrics[y_metric]))
            labels.append(name)

    if not xs:
        plt.close(fig)
        return

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    # Compute Pareto front (lower is better for both)
    pareto_mask = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i != j and xs_arr[j] <= xs_arr[i] and ys_arr[j] <= ys_arr[i]:
                if xs_arr[j] < xs_arr[i] or ys_arr[j] < ys_arr[i]:
                    pareto_mask[i] = False
                    break

    # Plot all points
    ax.scatter(xs_arr[~pareto_mask], ys_arr[~pareto_mask],
               s=80, color="steelblue", alpha=0.6, zorder=5)
    ax.scatter(xs_arr[pareto_mask], ys_arr[pareto_mask],
               s=120, color="crimson", marker="*", zorder=10, label="Pareto front")

    # Connect Pareto front
    pareto_idx = np.where(pareto_mask)[0]
    if len(pareto_idx) > 1:
        order = pareto_idx[np.argsort(xs_arr[pareto_idx])]
        ax.plot(xs_arr[order], ys_arr[order], "r--", alpha=0.5, linewidth=1.2)

    # Label all points
    for i, label in enumerate(labels):
        ax.annotate(label, (xs_arr[i], ys_arr[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title("Policy Pareto Frontier")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Episode cost breakdown
# ---------------------------------------------------------------------------


def plot_episode_cost_breakdown(
    economic_metrics: dict[str, float],
    out_path: str | None = None,
) -> None:
    """Pie chart showing fuel/delay/carbon cost composition.

    Parameters
    ----------
    economic_metrics:
        Dict with keys ``fuel_cost_usd``, ``delay_cost_usd``,
        ``carbon_cost_usd`` from ``compute_economic_metrics()``.
    """
    labels = []
    sizes = []
    color_map = []

    for key, label, color in [
        ("fuel_cost_usd", "Fuel", "#4c72b0"),
        ("delay_cost_usd", "Delay", "#dd8452"),
        ("carbon_cost_usd", "Carbon", "#55a868"),
    ]:
        val = float(economic_metrics.get(key, 0.0))
        if val > 0:
            labels.append(label)
            sizes.append(val)
            color_map.append(color)

    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=color_map,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    total = sum(sizes)
    ax.set_title(
        f"Episode Cost Breakdown (Total: ${total:,.0f})",
        fontsize=13,
        fontweight="bold",
    )

    # Add value legend
    legend_labels = [f"{l}: ${v:,.0f}" for l, v in zip(labels, sizes)]
    ax.legend(
        wedges,
        legend_labels,
        title="Cost Components",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )

    plt.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Weather-conditioned reward scatter
# ---------------------------------------------------------------------------


def plot_weather_reward_scatter(
    step_log: Any,
    reward_col: str = "vessel_reward_total_total",
    out_path: str | None = None,
) -> None:
    """Scatter plot of reward vs mean sea state per step.

    Parameters
    ----------
    step_log:
        DataFrame with ``mean_sea_state`` and a reward column.
    """
    import pandas as pd

    df = pd.DataFrame(step_log) if not isinstance(step_log, pd.DataFrame) else step_log
    if "mean_sea_state" not in df.columns or reward_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        df["mean_sea_state"],
        df[reward_col],
        c=df.get("t", range(len(df))),
        cmap="viridis",
        alpha=0.6,
        s=20,
    )
    fig.colorbar(sc, ax=ax, label="Step")
    ax.set_xlabel("Mean Sea State")
    ax.set_ylabel(reward_col.replace("_", " ").title())
    ax.set_title("Weather-Conditioned Reward")
    plt.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Multi-seed comparison with error bars
# ---------------------------------------------------------------------------


def plot_multi_seed_final_bar(
    multi_seed_results: dict[str, dict[str, Any]],
    metric: str = "mean_reward",
    out_path: str | None = None,
) -> None:
    """Bar chart of final rewards across policies with error bars.

    Parameters
    ----------
    multi_seed_results:
        Dict mapping policy name -> multi-seed result dict (containing
        ``histories`` list of per-seed iteration logs).
    """
    import numpy as np

    names = []
    means = []
    stds = []
    ci95s = []

    for name, result in multi_seed_results.items():
        histories = result.get("histories", [])
        if not histories:
            continue
        finals = []
        for h in histories:
            if h:
                finals.append(float(h[-1].get(metric, 0.0)))
        if finals:
            arr = np.array(finals)
            names.append(name)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            ci95s.append(1.96 * float(arr.std()) / max(np.sqrt(len(arr)), 1))

    if not names:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=ci95s, capsize=5, color="steelblue", alpha=0.8)

    for i, (bar, m, ci) in enumerate(zip(bars, means, ci95s)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Final {metric.replace('_', ' ').title()} (95% CI)")
    plt.tight_layout()
    _save_or_show(fig, out_path)
