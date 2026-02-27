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
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from hmarl_mvp.plotting import (  # noqa: E402
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

_POLICY_NAMES = ("independent", "reactive", "forecast", "oracle", "mappo")


def _load_csv(path: str | Path) -> pd.DataFrame | None:
    """Load a CSV silently; return None if missing."""
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


def _formats(fmt: str) -> tuple[str, ...]:
    if fmt == "both":
        return ("pdf", "png")
    return (fmt,)


def _save_custom(fig: plt.Figure, out_dir: Path, name: str, formats: tuple[str, ...]) -> None:
    """Save a manually-constructed figure in multiple formats."""
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved: {name}")


def _call_plotter_multi(
    plotter: Any,
    *args: Any,
    out_dir: Path,
    name: str,
    formats: tuple[str, ...],
    **kwargs: Any,
) -> None:
    """Call a plotting helper once per format via its out_path parameter."""
    for ext in formats:
        out_path = out_dir / f"{name}.{ext}"
        plotter(*args, out_path=str(out_path), **kwargs)
    print(f"  saved: {name}")


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_policy_results(base_dir: Path, prefixes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load per-policy result CSVs from one directory."""
    results: dict[str, pd.DataFrame] = {}
    for policy in _POLICY_NAMES:
        candidates: list[Path] = []
        for prefix in prefixes:
            if prefix:
                candidates.append(base_dir / f"{prefix}_{policy}.csv")
            else:
                candidates.append(base_dir / f"{policy}.csv")
        if policy == "mappo":
            candidates.append(base_dir / "mappo.csv")
        path = _find_first_existing(candidates)
        if path is not None:
            results[policy] = pd.read_csv(path)
    return results


def _find_policy_results(runs_dir: Path) -> dict[str, pd.DataFrame]:
    """Find a usable policy-result dictionary for policy comparison plotting."""
    search = [
        (runs_dir / "full_run", ("mappo", "policy")),
        (runs_dir / "mappo_compare", ("",)),
        (runs_dir / "baselines", ("policy",)),
        (runs_dir / "baseline_refactor", ("policy",)),
    ]
    for base_dir, prefixes in search:
        if not base_dir.exists():
            continue
        results = _load_policy_results(base_dir, prefixes)
        if len(results) >= 2:
            return results
    return {}


def _coerce_policy_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize policy label column name for summary/economic tables."""
    if "policy" in df.columns:
        return df
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        return df.rename(columns={unnamed[0]: "policy"})
    first = df.columns[0]
    if first != "policy":
        return df.rename(columns={first: "policy"})
    return df


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def fig_training_curves(runs_dir: Path, out_dir: Path, formats: tuple[str, ...]) -> None:
    """Figure 1: MAPPO training curves (reward and critic loss)."""
    path = _find_first_existing(
        [
            runs_dir / "full_run" / "mappo__train_log.csv",
            runs_dir / "mappo_train" / "train_history.csv",
            runs_dir / "baseline" / "seed_42" / "metrics.csv",
            runs_dir / "multi_seed" / "seed_42" / "metrics.csv",
        ]
    )
    if path is None:
        print("  [skip] no training log CSV found")
        return
    df = pd.read_csv(path)
    if "iteration" not in df.columns or "mean_reward" not in df.columns:
        print(f"  [skip] training log missing required columns: {path}")
        return
    _call_plotter_multi(
        plot_training_curves,
        df,
        out_dir=out_dir,
        name="fig1_training_curves",
        formats=formats,
    )


def fig_policy_comparison(runs_dir: Path, out_dir: Path, formats: tuple[str, ...]) -> None:
    """Figure 2: Policy comparison (time-series across policies)."""
    results = _find_policy_results(runs_dir)
    if len(results) < 2:
        print("  [skip] no usable policy result files found")
        return
    _call_plotter_multi(
        plot_policy_comparison,
        results,
        out_dir=out_dir,
        name="fig2_policy_comparison",
        formats=formats,
    )


def fig_multi_seed(runs_dir: Path, out_dir: Path, formats: tuple[str, ...]) -> None:
    """Figure 3: Multi-seed learning curve with mean/std band."""
    ms_dir = runs_dir / "multi_seed"
    if not ms_dir.exists():
        print("  [skip] no multi_seed directory found")
        return

    histories: list[list[dict[str, Any]]] = []
    seeds: list[int] = []
    for seed_dir in sorted(ms_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        metrics_path = seed_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if "mean_reward" not in df.columns:
            continue
        histories.append(df.to_dict(orient="records"))
        try:
            seeds.append(int(seed_dir.name.replace("seed_", "")))
        except ValueError:
            seeds.append(len(seeds))

    if not histories:
        print("  [skip] no per-seed metrics.csv found")
        return

    multi_seed_result = {"histories": histories, "seeds": seeds}
    _call_plotter_multi(
        plot_multi_seed_curves,
        multi_seed_result,
        out_dir=out_dir,
        name="fig3_multi_seed_curves",
        formats=formats,
        metric="mean_reward",
    )


def fig_parameter_sharing(runs_dir: Path, out_dir: Path, formats: tuple[str, ...]) -> None:
    """Figure 4: Parameter-sharing ablation (shared vs no-sharing)."""
    shared_path = runs_dir / "multi_seed" / "summary.csv"
    no_share_path = runs_dir / "no_sharing" / "summary.csv"

    if shared_path.exists() and no_share_path.exists():
        shared = pd.read_csv(shared_path)
        no_share = pd.read_csv(no_share_path)
        rows: list[dict[str, Any]] = []

        def _mean_row(df: pd.DataFrame, label: str) -> dict[str, Any]:
            row: dict[str, Any] = {"ablation": label}
            for col in ("final_mean_reward", "best_mean_reward", "total_reward"):
                if col in df.columns:
                    row[col] = float(df[col].mean())
            return row

        rows.append(_mean_row(shared, "shared"))
        rows.append(_mean_row(no_share, "no_sharing"))
        ablation_df = pd.DataFrame(rows)
    else:
        ablation_path = _find_first_existing(
            [
                runs_dir / "mappo_ablation" / "ablation_results.csv",
                runs_dir / "ablation" / "ablation_results.csv",
            ]
        )
        if ablation_path is None:
            print("  [skip] no parameter-sharing ablation data found")
            return
        ablation_df = pd.read_csv(ablation_path)

    if "ablation" not in ablation_df.columns:
        print("  [skip] ablation data missing 'ablation' column")
        return

    _call_plotter_multi(
        plot_ablation_bar,
        ablation_df,
        out_dir=out_dir,
        name="fig4_parameter_sharing",
        formats=formats,
    )


def fig_weather_impact(runs_dir: Path, out_dir: Path, formats: tuple[str, ...]) -> None:
    """Figure 5: Weather-curriculum dashboard."""
    path = _find_first_existing(
        [
            runs_dir / "weather_curriculum" / "metrics.csv",
            runs_dir / "weather_curriculum" / "seed_42" / "metrics.csv",
        ]
    )
    if path is None:
        print("  [skip] no weather curriculum metrics found")
        return
    df = pd.read_csv(path)
    if "mean_reward" not in df.columns:
        print(f"  [skip] weather metrics missing mean_reward: {path}")
        return
    _call_plotter_multi(
        plot_training_dashboard,
        df.to_dict(orient="records"),
        out_dir=out_dir,
        name="fig5_weather_dashboard",
        formats=formats,
    )


def fig_hyperparam_heatmap(runs_dir: Path, out_dir: Path, formats: tuple[str, ...]) -> None:
    """Figure 6: Hyperparameter sweep heatmap (lr vs entropy_coeff)."""
    path = _find_first_existing(
        [
            runs_dir / "hyperparam_sweep" / "sweep_results.csv",
            runs_dir / "mappo_sweep" / "sweep_results.csv",
        ]
    )
    if path is None:
        print("  [skip] no sweep_results.csv found")
        return

    df = pd.read_csv(path)
    if "lr" not in df.columns or "entropy_coeff" not in df.columns:
        print("  [skip] sweep results missing lr/entropy_coeff")
        return

    metric = "total_reward"
    if metric not in df.columns:
        if "final_mean_reward" in df.columns:
            metric = "final_mean_reward"
        elif "best_mean_reward" in df.columns:
            metric = "best_mean_reward"
        else:
            print("  [skip] sweep results missing reward metric columns")
            return

    _call_plotter_multi(
        plot_sweep_heatmap,
        df,
        out_dir=out_dir,
        name="fig6_hyperparam_heatmap",
        formats=formats,
        x_param="lr",
        y_param="entropy_coeff",
        metric=metric,
    )


def fig_economic_comparison(runs_dir: Path, out_dir: Path, formats: tuple[str, ...]) -> None:
    """Figure 7: Economic cost breakdown (stacked bar by policy)."""
    df = _load_csv(runs_dir / "economic" / "cost_breakdown.csv")

    if df is None:
        summary_path = _find_first_existing(
            [
                runs_dir / "full_run" / "policy_summary.csv",
                runs_dir / "baselines" / "policy_summary.csv",
                runs_dir / "baseline_refactor" / "policy_summary.csv",
            ]
        )
        if summary_path is not None:
            df = pd.read_csv(summary_path)
            df = _coerce_policy_column(df)

    if df is None:
        policy_results = _find_policy_results(runs_dir)
        rows: list[dict[str, Any]] = []
        for policy, policy_df in policy_results.items():
            if policy_df.empty:
                continue
            last = policy_df.iloc[-1].to_dict()
            last["policy"] = policy
            rows.append(last)
        if rows:
            df = pd.DataFrame(rows)

    if df is None:
        print("  [skip] no economic data found")
        return

    df = _coerce_policy_column(df)
    cost_cols = [c for c in df.columns if c.endswith("_cost_usd")]
    if not cost_cols:
        print("  [skip] no *_cost_usd columns found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = None
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    for idx, col in enumerate(cost_cols):
        label = col.replace("_cost_usd", "").replace("_", " ").title()
        vals = df[col].to_numpy(dtype=float)
        ax.bar(
            df["policy"],
            vals,
            bottom=bottom,
            label=label,
            color=colors[idx % len(colors)],
        )
        bottom = vals if bottom is None else bottom + vals

    ax.set_ylabel("Cost (USD)")
    ax.set_title("Economic Cost Comparison (RQ4)")
    ax.legend()
    fig.tight_layout()
    _save_custom(fig, out_dir, "fig7_economic_comparison", formats)


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
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = _formats(args.format)

    with plt.rc_context(PAPER_RC):
        print(f"Generating paper figures from {runs_dir} -> {out_dir}")
        for label, gen_fn in FIGURE_GENERATORS:
            print(f"\n{label}:")
            try:
                gen_fn(runs_dir, out_dir, formats)
            except Exception as exc:  # pragma: no cover - defensive CLI guard
                print(f"  [error] {exc}")

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
