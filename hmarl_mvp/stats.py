"""Statistical evaluation utilities for rigorous method comparison.

Provides Welch's t-test, bootstrap confidence intervals, and
formatted comparison tables — the standard toolkit for reporting
RL experiment results in academic papers.

Usage::

    from hmarl_mvp.stats import compare_methods, bootstrap_ci

    # Compare two sets of per-seed final rewards
    result = compare_methods(rewards_a, rewards_b, names=("MAPPO", "Random"))
    print(result["summary"])

    # Bootstrap CI for a single metric
    lo, hi = bootstrap_ci(rewards, confidence=0.95)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def welch_t_test(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
) -> dict[str, float]:
    """Welch's unequal-variances t-test.

    Parameters
    ----------
    a, b:
        Samples from two independent groups.

    Returns
    -------
    dict with ``t_stat``, ``p_value``, ``df`` (Welch-Satterthwaite),
    ``mean_a``, ``mean_b``, ``diff`` (mean_a - mean_b),
    ``cohens_d`` (effect size).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n_a, n_b = len(a_arr), len(b_arr)

    if n_a < 2 or n_b < 2:
        return {
            "t_stat": 0.0, "p_value": 1.0, "df": 0.0,
            "mean_a": float(a_arr.mean()), "mean_b": float(b_arr.mean()),
            "diff": float(a_arr.mean() - b_arr.mean()), "cohens_d": 0.0,
        }

    mean_a, mean_b = a_arr.mean(), b_arr.mean()
    var_a, var_b = a_arr.var(ddof=1), b_arr.var(ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-15:
        return {
            "t_stat": 0.0, "p_value": 1.0,
            "df": float(n_a + n_b - 2),
            "mean_a": float(mean_a), "mean_b": float(mean_b),
            "diff": float(mean_a - mean_b), "cohens_d": 0.0,
        }

    t_stat = float((mean_a - mean_b) / se)

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = float(num / denom) if denom > 0 else float(n_a + n_b - 2)

    # p-value from t-distribution (two-sided)
    p_value = _t_cdf_two_sided(abs(t_stat), df)

    # Cohen's d (pooled std)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = float((mean_a - mean_b) / pooled_std) if pooled_std > 1e-15 else 0.0

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "df": df,
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "diff": float(mean_a - mean_b),
        "cohens_d": cohens_d,
    }


def bootstrap_ci(
    data: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    statistic: str = "mean",
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval.

    Parameters
    ----------
    data:
        1-D array of observations.
    confidence:
        Confidence level (e.g. 0.95 for 95% CI).
    n_bootstrap:
        Number of bootstrap resamples.
    statistic:
        ``"mean"`` or ``"median"``.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (lower, upper) bounds of the CI.
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    stat_fn = np.mean if statistic == "mean" else np.median

    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def compare_methods(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    names: tuple[str, str] = ("Method A", "Method B"),
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Full statistical comparison of two methods.

    Combines Welch's t-test with bootstrap CIs for both methods.

    Returns
    -------
    dict with:
        ``t_test`` — Welch's t-test results.
        ``ci_a``, ``ci_b`` — Bootstrap CIs for each method.
        ``significant`` — True if p < 0.05.
        ``summary`` — Human-readable markdown summary.
    """
    t_test = welch_t_test(a, b)
    ci_a = bootstrap_ci(a, confidence=confidence, n_bootstrap=n_bootstrap, seed=42)
    ci_b = bootstrap_ci(b, confidence=confidence, n_bootstrap=n_bootstrap, seed=43)

    significant = t_test["p_value"] < 0.05

    summary_lines = [
        f"## Statistical Comparison: {names[0]} vs {names[1]}",
        "",
        f"| Metric | {names[0]} | {names[1]} |",
        "|--------|" + "-" * (len(names[0]) + 2) + "|" + "-" * (len(names[1]) + 2) + "|",
        f"| Mean | {t_test['mean_a']:.4f} | {t_test['mean_b']:.4f} |",
        f"| 95% CI | [{ci_a[0]:.4f}, {ci_a[1]:.4f}] | [{ci_b[0]:.4f}, {ci_b[1]:.4f}] |",
        "",
        f"- **Difference**: {t_test['diff']:+.4f}",
        f"- **Welch's t**: t({t_test['df']:.1f}) = {t_test['t_stat']:.3f}, "
        f"p = {t_test['p_value']:.4f}",
        f"- **Cohen's d**: {t_test['cohens_d']:.3f}",
        f"- **Significant at α=0.05**: {'Yes' if significant else 'No'}",
    ]

    return {
        "t_test": t_test,
        "ci_a": ci_a,
        "ci_b": ci_b,
        "significant": significant,
        "summary": "\n".join(summary_lines),
    }


def multi_method_comparison(
    results: dict[str, list[float]],
    baseline_name: str | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare multiple methods against a baseline or each other.

    Parameters
    ----------
    results:
        Dict mapping method name → list of per-seed metric values.
    baseline_name:
        If provided, compare all methods against this baseline.
        If None, compare all pairs.
    confidence:
        Confidence level for bootstrap CIs.

    Returns
    -------
    dict with:
        ``per_method`` — per-method stats (mean, std, CI).
        ``comparisons`` — list of pairwise comparison dicts.
        ``summary`` — Formatted markdown table.
    """
    per_method: dict[str, dict[str, Any]] = {}
    for name, vals in results.items():
        arr = np.asarray(vals, dtype=np.float64)
        ci = bootstrap_ci(vals, confidence=confidence)
        per_method[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "n": len(arr),
        }

    comparisons: list[dict[str, Any]] = []
    names = list(results.keys())

    if baseline_name and baseline_name in results:
        for name in names:
            if name == baseline_name:
                continue
            cmp = compare_methods(
                results[name], results[baseline_name],
                names=(name, baseline_name),
                confidence=confidence,
            )
            comparisons.append({
                "method": name,
                "baseline": baseline_name,
                **cmp["t_test"],
                "significant": cmp["significant"],
            })
    else:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cmp = compare_methods(
                    results[names[i]], results[names[j]],
                    names=(names[i], names[j]),
                    confidence=confidence,
                )
                comparisons.append({
                    "method_a": names[i],
                    "method_b": names[j],
                    **cmp["t_test"],
                    "significant": cmp["significant"],
                })

    # Build summary table
    lines = [
        "## Method Comparison Summary",
        "",
        "| Method | Mean ± Std | 95% CI | N |",
        "|--------|-----------|--------|---|",
    ]
    for name, stats in per_method.items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} ± {stats['std']:.4f} | "
            f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] | {stats['n']} |"
        )

    if comparisons:
        lines.extend(["", "### Pairwise Comparisons", ""])
        for c in comparisons:
            a_name = c.get("method", c.get("method_a", "?"))
            b_name = c.get("baseline", c.get("method_b", "?"))
            sig = "✓" if c["significant"] else "✗"
            lines.append(
                f"- **{a_name}** vs **{b_name}**: "
                f"Δ={c['diff']:+.4f}, p={c['p_value']:.4f}, "
                f"d={c['cohens_d']:.3f} [{sig}]"
            )

    return {
        "per_method": per_method,
        "comparisons": comparisons,
        "summary": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Internal: t-distribution CDF approximation
# ---------------------------------------------------------------------------


def _t_cdf_two_sided(t_abs: float, df: float) -> float:
    """Approximate two-sided p-value from Student's t-distribution.

    Uses the regularised incomplete beta function relationship:
    ``p = I(df/(df+t²), df/2, 1/2)`` for the two-sided test.

    Falls back to scipy if available; otherwise uses a reasonable
    approximation.
    """
    try:
        from scipy import stats as sp_stats

        return float(2 * sp_stats.t.sf(t_abs, df))
    except ImportError:
        pass

    # Approximation: for large df, t → N(0,1)
    # For small df, use a conservative approximation
    x = df / (df + t_abs ** 2)
    # Regularised incomplete beta function approximation
    p = _approx_betainc(df / 2.0, 0.5, x)
    return float(max(0.0, min(1.0, p)))


def _approx_betainc(a: float, b: float, x: float) -> float:
    """Very rough regularised incomplete beta function.

    Sufficient for p-value ordering; for exact p-values, scipy
    is preferred (and imported automatically when available).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use continued fraction approximation (Lentz's method, 30 terms)
    # This gives reasonable accuracy for our use case
    from math import exp, lgamma

    log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
    prefix = exp(a * np.log(x) + b * np.log(1 - x) - log_beta) / a

    # Simple series expansion (sufficient for df > 2)
    result = 1.0
    term = 1.0
    for n in range(1, 200):
        term *= (n - b) * x / n
        coeff = term / (a + n)
        result += coeff
        if abs(coeff) < 1e-12:
            break

    return float(min(1.0, max(0.0, prefix * result)))
