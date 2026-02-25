"""Plot helpers for policy comparison and ablations."""

from __future__ import annotations

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
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_horizon_sweep(
    horizon_results: dict[int, Any],
    out_path: str | None = None,
) -> None:
    """Plot forecast horizon ablation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Forecast Horizon Ablation (RQ3)", fontsize=13)

    for ax, (col, title) in zip(
        axes,
        [
            ("avg_queue", "Avg Queue"),
            ("total_emissions_co2", "Cumulative CO2"),
            ("total_ops_cost_usd", "Total Ops Cost ($)"),
        ],
    ):
        for h, df in horizon_results.items():
            ax.plot(df["t"], df[col], label=f"{h}h")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_noise_sweep(
    noise_results: dict[float, Any],
    out_path: str | None = None,
) -> None:
    """Plot forecast noise ablation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Forecast Noise Ablation (RQ3)", fontsize=13)

    for ax, (col, title) in zip(
        axes,
        [
            ("avg_queue", "Avg Queue"),
            ("total_emissions_co2", "Cumulative CO2"),
            ("total_ops_cost_usd", "Total Ops Cost ($)"),
        ],
    ):
        for n, df in noise_results.items():
            ax.plot(df["t"], df[col], label=f"sigma={n}")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_sharing_sweep(
    sharing_results: dict[str, Any],
    out_path: str | None = None,
) -> None:
    """Plot forecast sharing ablation."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Forecast Sharing Ablation (RQ3)", fontsize=13)
    for label, df in sharing_results.items():
        ax[0].plot(df["t"], df["avg_queue"], label=label)
        ax[1].plot(df["t"], df["total_ops_cost_usd"], label=label)
    ax[0].set_title("Avg Queue")
    ax[1].set_title("Total Ops Cost ($)")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
