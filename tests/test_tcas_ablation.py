"""Tests for TCAS ablation script."""

from __future__ import annotations

import numpy as np

from scripts.run_tcas_ablation import (
    arithmetic_mean,
    compute_all_components,
    geometric_mean,
    harmonic_mean,
    leave_one_out_ablation,
    minimum_agg,
    normalize_components,
)


class TestNormalizeComponents:
    def test_perfect_probe(self) -> None:
        c = normalize_components(1.0, 0.0, 0.5, 50.0)
        assert c["accessibility"] == 1.0
        assert c["linearity"] == 1.0
        assert c["causality"] == 0.5
        assert 0.0 < c["functional_load"] < 1.0

    def test_below_chance_accessibility(self) -> None:
        c = normalize_components(0.3, 0.0, 0.5, 50.0)
        assert c["accessibility"] == 0.0

    def test_zero_leace_drop(self) -> None:
        c = normalize_components(1.0, 0.0, 0.0, 0.0)
        assert c["causality"] == 0.0

    def test_large_downstream(self) -> None:
        c = normalize_components(1.0, 0.0, 0.5, 10000.0)
        assert c["functional_load"] > 0.99


class TestAggregations:
    def test_geometric_mean_basic(self) -> None:
        result = geometric_mean([4.0, 4.0, 4.0, 4.0])
        assert abs(result - 4.0) < 1e-10

    def test_geometric_mean_zero(self) -> None:
        assert geometric_mean([1.0, 0.0, 1.0]) == 0.0

    def test_arithmetic_mean(self) -> None:
        assert abs(arithmetic_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-10

    def test_harmonic_mean_basic(self) -> None:
        result = harmonic_mean([1.0, 1.0, 1.0])
        assert abs(result - 1.0) < 1e-10

    def test_harmonic_mean_zero(self) -> None:
        assert harmonic_mean([1.0, 0.0, 1.0]) == 0.0

    def test_min_agg(self) -> None:
        assert minimum_agg([0.3, 0.8, 0.5, 0.9]) == 0.3


class TestComputeAllComponents:
    def test_entry_count(self) -> None:
        entries = compute_all_components()
        # 7 models × 5 properties = 35 entries
        assert len(entries) == 35

    def test_entry_fields(self) -> None:
        entries = compute_all_components()
        for e in entries:
            assert "model" in e
            assert "property" in e
            assert "accessibility" in e
            assert "linearity" in e
            assert "causality" in e
            assert "functional_load" in e
            assert "tcas_geometric" in e

    def test_geometric_matches_manual(self) -> None:
        entries = compute_all_components()
        for e in entries:
            components = [e["accessibility"], e["linearity"], e["causality"], e["functional_load"]]
            expected = geometric_mean(components)
            assert abs(e["tcas_geometric"] - expected) < 1e-10

    def test_tcas_bounded(self) -> None:
        entries = compute_all_components()
        for e in entries:
            assert 0.0 <= e["tcas_geometric"] <= 1.0


class TestLeaveOneOut:
    def test_returns_all_components(self) -> None:
        entries = compute_all_components()
        results = leave_one_out_ablation(entries)
        assert set(results.keys()) == {
            "accessibility",
            "linearity",
            "causality",
            "functional_load",
        }

    def test_spearman_bounded(self) -> None:
        entries = compute_all_components()
        results = leave_one_out_ablation(entries)
        for _comp, vals in results.items():
            rho = vals["spearman_rho"]
            assert -1.0 <= rho <= 1.0 or np.isnan(rho)
