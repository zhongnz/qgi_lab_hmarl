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
                color = color_by_agent.get(agent, None)
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
                    color = color_by_agent.get(agent, None)
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
                color = color_by_agent.get(agent, None)
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
                    color = color_by_agent.get(agent, None)
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
    import pandas as pd
    import re

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
