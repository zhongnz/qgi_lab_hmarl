#!/usr/bin/env python3
"""Generate publication-ready figures from experiment results.

Usage
-----
Run all experiments first, then::

    python scripts/generate_paper_figures.py [--runs-dir runs] [--out-dir figures]

The script reads CSV result files from ``runs/`` sub-directories and
produces PDF/PNG figures suitable for inclusion in a LaTeX paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hmarl_mvp.plotting import (
    plot_ablation_bar,
    plot_multi_seed_curves,
    plot_policy_comparison,
    plot_sweep_heatmap,
    plot_training_curves,
    plot_training_dashboard,
)

# ---------------------------------------------------------------------------
# Paper style defaults
# ---------------------------------------------------------------------------
PAPER_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _load_csv(path: str | Path) -> pd.DataFrame | None:
    """Load a CSV silently; return None if missing."""
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


def _save(fig: plt.Figure, out_dir: Path, name: str, formats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Save figure in multiple formats."""
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(str(path), format=fmt)
    plt.close(fig)
    print(f"  saved: {name}")


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_training_curves(runs_dir: Path, out_dir: Path) -> None:
    """Figure 1: MAPPO training curves (reward, loss, entropy over iterations)."""
    metrics = _load_csv(runs_dir / "baseline" / "metrics.csv")
    if metrics is None:
        metrics = _load_csv(runs_dir / "multi_seed" / "seed_42" / "metrics.csv")
    if metrics is None:
        print("  [skip] no baseline metrics.csv found")
        return
    fig = plot_training_curves(metrics, title="MAPPO Training Curves")
    _save(fig, out_dir, "fig1_training_curves")


def fig_policy_comparison(runs_dir: Path, out_dir: Path) -> None:
    """Figure 2: Bar chart comparing heuristic baselines vs MAPPO."""
    summary_path = runs_dir / "baseline" / "comparison_summary.csv"
    df = _load_csv(summary_path)
    if df is None:
        print("  [skip] no comparison_summary.csv found")
        return
    fig = plot_policy_comparison(df)
    _save(fig, out_dir, "fig2_policy_comparison")


def fig_multi_seed(runs_dir: Path, out_dir: Path) -> None:
    """Figure 3: Multi-seed training curves with mean and std bands."""
    ms_dir = runs_dir / "multi_seed"
    seed_dfs = []
    if ms_dir.exists():
        for seed_dir in sorted(ms_dir.iterdir()):
            csv = seed_dir / "metrics.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                df["seed"] = seed_dir.name
                seed_dfs.append(df)
    if not seed_dfs:
        print("  [skip] no multi_seed results found")
        return
    combined = pd.concat(seed_dfs, ignore_index=True)
    fig = plot_multi_seed_curves(combined, title="Multi-Seed MAPPO Training")
    _save(fig, out_dir, "fig3_multi_seed_curves")


def fig_parameter_sharing(runs_dir: Path, out_dir: Path) -> None:
    """Figure 4: Ablation bar chart — shared vs independent parameters."""
    shared = _load_csv(runs_dir / "multi_seed" / "summary.csv")
    independent = _load_csv(runs_dir / "no_sharing" / "summary.csv")
    if shared is None or independent is None:
        print("  [skip] need both multi_seed/summary.csv and no_sharing/summary.csv")
        return
    results = {"Shared Parameters": shared, "Independent": independent}
    fig = plot_ablation_bar(results, title="Parameter Sharing Ablation")
    _save(fig, out_dir, "fig4_parameter_sharing")


def fig_weather_impact(runs_dir: Path, out_dir: Path) -> None:
    """Figure 5: Weather curriculum training dashboard."""
    metrics = _load_csv(runs_dir / "weather_curriculum" / "metrics.csv")
    if metrics is None:
        print("  [skip] no weather_curriculum/metrics.csv found")
        return
    fig = plot_training_dashboard(metrics, title="Weather Curriculum Training")
    _save(fig, out_dir, "fig5_weather_dashboard")


def fig_hyperparam_heatmap(runs_dir: Path, out_dir: Path) -> None:
    """Figure 6: Hyperparameter sweep heatmap (LR vs entropy)."""
    sweep_csv = runs_dir / "hyperparam_sweep" / "sweep_results.csv"
    df = _load_csv(sweep_csv)
    if df is None:
        print("  [skip] no hyperparam_sweep/sweep_results.csv found")
        return
    fig = plot_sweep_heatmap(df, title="Hyperparameter Sweep")
    _save(fig, out_dir, "fig6_hyperparam_heatmap")


def fig_economic_comparison(runs_dir: Path, out_dir: Path) -> None:
    """Figure 7: Economic cost breakdown (stacked bar)."""
    econ_csv = runs_dir / "economic" / "cost_breakdown.csv"
    df = _load_csv(econ_csv)
    if df is None:
        print("  [skip] no economic/cost_breakdown.csv found")
        return

    cost_cols = [c for c in df.columns if c.endswith("_cost_usd")]
    if not cost_cols:
        print("  [skip] no cost columns in economic data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    policy_col = "policy" if "policy" in df.columns else df.columns[0]
    bottom = None
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    for idx, col in enumerate(cost_cols):
        label = col.replace("_cost_usd", "").replace("_", " ").title()
        vals = df[col].values
        ax.bar(df[policy_col], vals, bottom=bottom, label=label,
               color=colors[idx % len(colors)])
        bottom = vals if bottom is None else bottom + vals
    ax.set_ylabel("Cost (USD)")
    ax.set_title("Economic Cost Comparison (RQ4)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig7_economic_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURE_GENERATORS = [
    ("Fig 1: Training curves", fig_training_curves),
    ("Fig 2: Policy comparison", fig_policy_comparison),
    ("Fig 3: Multi-seed curves", fig_multi_seed),
    ("Fig 4: Parameter sharing", fig_parameter_sharing),
    ("Fig 5: Weather impact", fig_weather_impact),
    ("Fig 6: Hyperparam heatmap", fig_hyperparam_heatmap),
    ("Fig 7: Economic comparison", fig_economic_comparison),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment results.")
    parser.add_argument("--runs-dir", default="runs", help="Root directory of experiment results")
    parser.add_argument("--out-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--format", choices=["pdf", "png", "both"], default="both",
                        help="Output format (default: both)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PAPER_RC):
        print(f"Generating paper figures from {runs_dir} -> {out_dir}")
        for label, gen_fn in FIGURE_GENERATORS:
            print(f"\n{label}:")
            try:
                gen_fn(runs_dir, out_dir)
            except Exception as e:
                print(f"  [error] {e}")

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
