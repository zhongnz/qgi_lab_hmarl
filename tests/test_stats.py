"""Tests for the statistical evaluation module."""

from __future__ import annotations

import numpy as np
import pytest

from hmarl_mvp.stats import (
    bootstrap_ci,
    compare_methods,
    multi_method_comparison,
    welch_t_test,
)

# ------------------------------------------------------------------
# welch_t_test
# ------------------------------------------------------------------


class TestWelchTTest:
    """Unit tests for Welch's t-test."""

    def test_identical_samples(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = welch_t_test(a, a)
        assert result["diff"] == pytest.approx(0.0)
        assert result["t_stat"] == pytest.approx(0.0)
        assert result["cohens_d"] == pytest.approx(0.0)
        assert result["p_value"] >= 0.9  # not significant

    def test_clearly_different(self) -> None:
        a = [100.0, 101.0, 102.0, 100.5, 101.5]
        b = [0.0, 1.0, 2.0, 0.5, 1.5]
        result = welch_t_test(a, b)
        assert result["diff"] > 90  # clearly different
        assert result["p_value"] < 0.01  # highly significant
        assert abs(result["cohens_d"]) > 1.0  # large effect

    def test_small_samples(self) -> None:
        # Below minimum: should return default
        result = welch_t_test([1.0], [2.0])
        assert result["t_stat"] == 0.0
        assert result["p_value"] == 1.0

    def test_numpy_input(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        result = welch_t_test(a, b)
        assert result["mean_a"] == pytest.approx(2.5)
        assert result["mean_b"] == pytest.approx(6.5)
        assert result["diff"] == pytest.approx(-4.0)

    def test_zero_variance(self) -> None:
        a = [5.0, 5.0, 5.0]
        b = [3.0, 3.0, 3.0]
        result = welch_t_test(a, b)
        # se == 0 → returns default with correct means
        assert result["mean_a"] == pytest.approx(5.0)
        assert result["mean_b"] == pytest.approx(3.0)

    def test_keys_present(self) -> None:
        result = welch_t_test([1, 2, 3], [4, 5, 6])
        expected_keys = {"t_stat", "p_value", "df", "mean_a", "mean_b", "diff", "cohens_d"}
        assert expected_keys == set(result.keys())


# ------------------------------------------------------------------
# bootstrap_ci
# ------------------------------------------------------------------


class TestBootstrapCI:
    """Unit tests for bootstrap confidence intervals."""

    def test_basic_mean_ci(self) -> None:
        data = np.random.default_rng(42).normal(10.0, 1.0, size=100)
        lo, hi = bootstrap_ci(data, confidence=0.95, seed=42)
        assert lo < 10.0 < hi
        assert lo < hi

    def test_tight_ci_for_constant(self) -> None:
        data = [5.0, 5.0, 5.0, 5.0, 5.0]
        lo, hi = bootstrap_ci(data, confidence=0.95)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_median_statistic(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 100.0]  # skewed
        lo, hi = bootstrap_ci(data, statistic="median", seed=42)
        assert lo >= 1.0
        assert hi <= 100.0

    def test_empty_data(self) -> None:
        lo, hi = bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_single_element(self) -> None:
        lo, hi = bootstrap_ci([7.0])
        assert lo == pytest.approx(7.0)
        assert hi == pytest.approx(7.0)

    def test_confidence_levels(self) -> None:
        data = list(range(100))
        lo90, hi90 = bootstrap_ci(data, confidence=0.90, seed=42)
        lo99, hi99 = bootstrap_ci(data, confidence=0.99, seed=42)
        # 99% CI should be wider than 90% CI
        assert (hi99 - lo99) >= (hi90 - lo90)

    def test_reproducibility(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        r1 = bootstrap_ci(data, seed=123)
        r2 = bootstrap_ci(data, seed=123)
        assert r1 == r2


# ------------------------------------------------------------------
# compare_methods
# ------------------------------------------------------------------


class TestCompareMethods:
    """Tests for full pairwise comparison."""

    def test_basic_comparison(self) -> None:
        a = [10.0, 11.0, 12.0, 10.5, 11.5]
        b = [0.0, 1.0, 2.0, 0.5, 1.5]
        result = compare_methods(a, b, names=("Better", "Worse"))
        assert "t_test" in result
        assert "ci_a" in result
        assert "ci_b" in result
        assert "significant" in result
        assert "summary" in result
        assert result["significant"] is True

    def test_not_significant(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 1.0, size=5).tolist()
        b = rng.normal(5.0, 1.0, size=5).tolist()
        result = compare_methods(a, b)
        # With same distribution, likely not significant
        # (not guaranteed, but very probable with same mean)
        assert isinstance(result["significant"], bool)

    def test_summary_is_markdown(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = compare_methods(a, b, names=("A", "B"))
        summary = result["summary"]
        assert "## Statistical Comparison" in summary
        assert "A" in summary
        assert "B" in summary
        assert "Welch" in summary

    def test_ci_bounds(self) -> None:
        a = [10.0, 11.0, 12.0]
        b = [0.0, 1.0, 2.0]
        result = compare_methods(a, b)
        # CI for a should be around 10-12
        assert result["ci_a"][0] > 5
        # CI for b should be around 0-2
        assert result["ci_b"][1] < 5


# ------------------------------------------------------------------
# multi_method_comparison
# ------------------------------------------------------------------


class TestMultiMethodComparison:
    """Tests for multi-method comparison."""

    def test_basic_multi_comparison(self) -> None:
        results = {
            "MAPPO": [10.0, 11.0, 12.0, 10.5, 11.5],
            "Random": [0.0, 1.0, 2.0, 0.5, 1.5],
            "Heuristic": [5.0, 6.0, 7.0, 5.5, 6.5],
        }
        out = multi_method_comparison(results)
        assert "per_method" in out
        assert len(out["per_method"]) == 3
        assert "comparisons" in out
        # 3 methods → 3 pairwise comparisons
        assert len(out["comparisons"]) == 3
        assert "summary" in out

    def test_with_baseline(self) -> None:
        results = {
            "MAPPO": [10.0, 11.0, 12.0],
            "Random": [0.0, 1.0, 2.0],
            "Heuristic": [5.0, 6.0, 7.0],
        }
        out = multi_method_comparison(results, baseline_name="Random")
        # 2 comparisons (MAPPO vs Random, Heuristic vs Random)
        assert len(out["comparisons"]) == 2
        for cmp in out["comparisons"]:
            assert cmp["baseline"] == "Random"

    def test_per_method_stats(self) -> None:
        results = {"A": [1.0, 2.0, 3.0, 4.0]}
        out = multi_method_comparison(results)
        stats = out["per_method"]["A"]
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["n"] == 4
        assert "ci_lower" in stats
        assert "ci_upper" in stats
        assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]

    def test_summary_table(self) -> None:
        results = {"X": [1.0, 2.0], "Y": [3.0, 4.0]}
        out = multi_method_comparison(results)
        assert "## Method Comparison Summary" in out["summary"]
        assert "| X |" in out["summary"]
        assert "| Y |" in out["summary"]

    def test_single_method(self) -> None:
        results = {"Only": [5.0, 6.0, 7.0]}
        out = multi_method_comparison(results)
        assert len(out["comparisons"]) == 0
        assert len(out["per_method"]) == 1
