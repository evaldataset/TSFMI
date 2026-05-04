"""TCAS ablation study: validate the composite metric.

Analyzes TCAS = (A × L × C × F)^(1/4) through:
  1. Leave-one-out ablation — which component drives ranking changes?
  2. Aggregation comparison — geometric vs arithmetic vs harmonic mean
  3. Rank correlation — does TCAS agree with downstream importance?

This addresses the reviewer concern that TCAS is a heuristic without validation.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_tcas_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.utils.seed import seed_everything

# ---------------------------------------------------------------------------
# Hardcoded experiment results (authoritative source: run_neurips_enhancements.py)
# ---------------------------------------------------------------------------

PROBING: dict[str, dict[str, float]] = {
    "MOMENT": {
        "trend": 1.0,
        "stationarity": 1.0,
        "frequency": 1.0,
        "anomaly": 0.484,
        "change_point": 1.0,
    },
    "Chronos": {
        "trend": 1.0,
        "stationarity": 1.0,
        "frequency": 1.0,
        "anomaly": 0.665,
        "change_point": 1.0,
    },
    "PatchTST": {
        "trend": 1.0,
        "stationarity": 1.0,
        "frequency": 1.0,
        "anomaly": 0.475,
        "change_point": 1.0,
    },
    "GPT4TS": {
        "trend": 1.0,
        "stationarity": 1.0,
        "frequency": 1.0,
        "anomaly": 0.463,
        "change_point": 0.999,
    },
    "Timer": {
        "trend": 1.0,
        "stationarity": 1.0,
        "frequency": 1.0,
        "anomaly": 0.477,
        "change_point": 1.0,
    },
    "TimesFM": {
        "trend": 1.0,
        "stationarity": 1.0,
        "frequency": 1.0,
        "anomaly": 0.737,
        "change_point": 1.0,
    },
    "Moirai": {
        "trend": 1.0,
        "stationarity": 1.0,
        "frequency": 1.0,
        "anomaly": 0.657,
        "change_point": 1.0,
    },
}

SELECTIVITY: dict[str, dict[str, float]] = {
    "MOMENT": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": 0.002,
        "change_point": 0.0,
    },
    "Chronos": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": -0.01,
        "change_point": 0.0,
    },
    "PatchTST": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": 0.01,
        "change_point": 0.0,
    },
    "GPT4TS": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": -0.006,
        "change_point": 0.0,
    },
    "Timer": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
    "TimesFM": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": -0.11,
        "change_point": 0.0,
    },
    "Moirai": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": -0.09,
        "change_point": 0.0,
    },
}

LEACE_DROP: dict[str, dict[str, float]] = {
    "MOMENT": {
        "trend": 0.437,
        "stationarity": 0.535,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.469,
    },
    "Chronos": {
        "trend": 0.329,
        "stationarity": 0.538,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.567,
    },
    "PatchTST": {
        "trend": 0.245,
        "stationarity": 0.667,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.638,
    },
    "GPT4TS": {
        "trend": 0.260,
        "stationarity": 0.493,
        "frequency": 0.107,
        "anomaly": 0.0,
        "change_point": 0.505,
    },
    "Timer": {
        "trend": 0.335,
        "stationarity": 0.630,
        "frequency": 0.200,
        "anomaly": 0.04,
        "change_point": 0.635,
    },
    "TimesFM": {
        "trend": 0.370,
        "stationarity": 0.700,
        "frequency": 0.025,
        "anomaly": 0.155,
        "change_point": 0.530,
    },
    "Moirai": {
        "trend": 0.370,
        "stationarity": 0.700,
        "frequency": 0.175,
        "anomaly": 0.250,
        "change_point": 0.650,
    },
}

DOWNSTREAM: dict[str, dict[str, float]] = {
    "MOMENT": {
        "trend": 5.6,
        "stationarity": 18.0,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
    "Chronos": {
        "trend": 7808.0,
        "stationarity": 8.7,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
    "PatchTST": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
    "GPT4TS": {
        "trend": 37.5,
        "stationarity": 55.1,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
    "Timer": {
        "trend": 0.0,
        "stationarity": 0.0,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
    "TimesFM": {
        "trend": 413099.0,
        "stationarity": 5.0,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
    "Moirai": {
        "trend": 3.2,
        "stationarity": 10.9,
        "frequency": 0.0,
        "anomaly": 0.0,
        "change_point": 0.0,
    },
}

PROPERTIES = ["trend", "stationarity", "frequency", "anomaly", "change_point"]
MODEL_NAMES = list(PROBING.keys())


def normalize_components(
    probe_acc: float,
    selectivity: float,
    leace_drop: float,
    downstream_impact: float,
) -> dict[str, float]:
    """Normalize raw values to [0, 1] components for TCAS.

    Returns:
        Dict with keys: accessibility, linearity, causality, functional_load.
    """
    accessibility = max(0.0, min(1.0, (probe_acc - 0.5) / 0.5)) if probe_acc > 0.5 else 0.0
    linearity = max(0.0, 1.0 - min(abs(selectivity), 1.0))
    causality = max(0.0, min(1.0, leace_drop))
    functional_load = 1.0 / (1.0 + np.exp(-0.05 * (downstream_impact - 20)))
    return {
        "accessibility": accessibility,
        "linearity": linearity,
        "causality": causality,
        "functional_load": functional_load,
    }


def geometric_mean(values: list[float]) -> float:
    """Compute geometric mean, returning 0 if any value is 0."""
    if any(v == 0.0 for v in values):
        return 0.0
    product = 1.0
    for v in values:
        product *= v
    return float(product ** (1.0 / len(values)))


def arithmetic_mean(values: list[float]) -> float:
    """Compute arithmetic mean."""
    return float(np.mean(values))


def harmonic_mean(values: list[float]) -> float:
    """Compute harmonic mean, returning 0 if any value is 0."""
    if any(v == 0.0 for v in values):
        return 0.0
    return float(len(values) / sum(1.0 / v for v in values))


def minimum_agg(values: list[float]) -> float:
    """Min aggregation (bottleneck score)."""
    return float(min(values))


def compute_all_components() -> list[dict]:
    """Compute normalized TCAS components for all model-property pairs.

    Returns:
        List of dicts with model, property, 4 components, and raw values.
    """
    entries = []
    for model in MODEL_NAMES:
        for prop in PROPERTIES:
            raw = {
                "probe_acc": PROBING[model][prop],
                "selectivity": SELECTIVITY[model][prop],
                "leace_drop": LEACE_DROP[model][prop],
                "downstream": DOWNSTREAM[model][prop],
            }
            normed = normalize_components(
                raw["probe_acc"],
                raw["selectivity"],
                raw["leace_drop"],
                raw["downstream"],
            )
            components = list(normed.values())
            entries.append(
                {
                    "model": model,
                    "property": prop,
                    **normed,
                    "raw": raw,
                    "tcas_geometric": geometric_mean(components),
                    "tcas_arithmetic": arithmetic_mean(components),
                    "tcas_harmonic": harmonic_mean(components),
                    "tcas_min": minimum_agg(components),
                }
            )
    return entries


def leave_one_out_ablation(entries: list[dict]) -> dict[str, dict[str, float]]:
    """For each component, compute TCAS without it and measure ranking change.

    Returns:
        Dict with component name → {rank_change, spearman_vs_full}.
    """
    component_names = ["accessibility", "linearity", "causality", "functional_load"]

    full_scores = [e["tcas_geometric"] for e in entries]

    results: dict[str, dict[str, float]] = {}
    for exclude in component_names:
        ablated_scores = []
        for e in entries:
            remaining = [e[c] for c in component_names if c != exclude]
            ablated_scores.append(geometric_mean(remaining))

        # Rank correlation between full and ablated
        valid_mask = [
            f > 0 or a > 0
            for f, a in zip(full_scores, ablated_scores, strict=True)
        ]
        full_valid = [f for f, m in zip(full_scores, valid_mask, strict=True) if m]
        ablated_valid = [
            a for a, m in zip(ablated_scores, valid_mask, strict=True) if m
        ]

        if len(full_valid) >= 3:
            rho, p_val = spearmanr(full_valid, ablated_valid)
        else:
            rho, p_val = 0.0, 1.0

        # Count rank changes
        full_rank = np.argsort(np.argsort([-s for s in full_scores]))
        ablated_rank = np.argsort(np.argsort([-s for s in ablated_scores]))
        rank_changes = int(np.sum(full_rank != ablated_rank))

        results[exclude] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "rank_changes": rank_changes,
            "rank_change_fraction": rank_changes / len(entries),
        }

    return results


def aggregation_comparison(entries: list[dict]) -> dict[str, dict[str, float]]:
    """Compare rankings under different aggregation functions.

    Returns:
        Dict with method pairs → Spearman ρ.
    """
    methods = ["tcas_geometric", "tcas_arithmetic", "tcas_harmonic", "tcas_min"]
    comparisons: dict[str, dict[str, float]] = {}

    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            scores1 = [e[m1] for e in entries]
            scores2 = [e[m2] for e in entries]
            # Filter pairs where both are non-zero
            valid = [
                (s1, s2)
                for s1, s2 in zip(scores1, scores2, strict=True)
                if s1 > 0 or s2 > 0
            ]
            if len(valid) >= 3:
                rho, p_val = spearmanr([v[0] for v in valid], [v[1] for v in valid])
            else:
                rho, p_val = 0.0, 1.0

            comparisons[f"{m1}_vs_{m2}"] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val),
            }

    return comparisons


def rank_correlation_with_downstream(entries: list[dict]) -> dict[str, object]:
    """Correlate TCAS ranking with downstream impact ranking.

    Reports both:
    - Full correlation on all 35 entries (including zeros)
    - Filtered correlation on entries with nonzero downstream + TCAS

    Returns:
        Dict with full and filtered Spearman/Kendall results.
    """
    # Full correlation (all 35 entries, including zeros)
    all_tcas = [e["tcas_geometric"] for e in entries]
    all_downstream = [e["raw"]["downstream"] for e in entries]
    if len(entries) >= 3:
        full_rho, full_rho_p = spearmanr(all_tcas, all_downstream)
        full_tau, full_tau_p = kendalltau(all_tcas, all_downstream)
    else:
        full_rho, full_rho_p, full_tau, full_tau_p = 0.0, 1.0, 0.0, 1.0

    # Filtered (nonzero downstream AND nonzero TCAS)
    valid = [e for e in entries if e["raw"]["downstream"] > 0 and e["tcas_geometric"] > 0]
    if len(valid) >= 3:
        filt_tcas = [e["tcas_geometric"] for e in valid]
        filt_downstream = [e["raw"]["downstream"] for e in valid]
        filt_rho, filt_rho_p = spearmanr(filt_tcas, filt_downstream)
        filt_tau, filt_tau_p = kendalltau(filt_tcas, filt_downstream)
    else:
        filt_rho, filt_rho_p, filt_tau, filt_tau_p = float("nan"), 1.0, float("nan"), 1.0

    return {
        "full_n": len(entries),
        "full_spearman_rho": float(full_rho),
        "full_spearman_p": float(full_rho_p),
        "full_kendall_tau": float(full_tau),
        "full_kendall_p": float(full_tau_p),
        "filtered_n": len(valid),
        "filtered_spearman_rho": float(filt_rho),
        "filtered_spearman_p": float(filt_rho_p),
        "filtered_kendall_tau": float(filt_tau),
        "filtered_kendall_p": float(filt_tau_p),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="TCAS ablation study.")
    parser.add_argument("--output_dir", type=str, default="outputs/tcas_ablation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    """Run TCAS ablation study."""
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = compute_all_components()

    # 1. Leave-one-out
    loo_results = leave_one_out_ablation(entries)
    (output_dir / "leave_one_out.json").write_text(json.dumps(loo_results, indent=2))

    print("=== Leave-One-Out Ablation ===")
    print(f"{'Component':<20} {'Spearman ρ':>12} {'Rank Changes':>14} {'Fraction':>10}")
    print("-" * 58)
    for comp, vals in loo_results.items():
        print(
            f"{comp:<20} {vals['spearman_rho']:>12.4f} "
            f"{vals['rank_changes']:>14} {vals['rank_change_fraction']:>10.2%}"
        )

    # 2. Aggregation comparison
    agg_results = aggregation_comparison(entries)
    (output_dir / "aggregation_comparison.json").write_text(json.dumps(agg_results, indent=2))

    print("\n=== Aggregation Method Comparison ===")
    print(f"{'Pair':<45} {'Spearman ρ':>12}")
    print("-" * 58)
    for pair, vals in agg_results.items():
        print(f"{pair:<45} {vals['spearman_rho']:>12.4f}")

    # 3. Rank correlation with downstream
    rank_results = rank_correlation_with_downstream(entries)
    (output_dir / "rank_correlation.json").write_text(json.dumps(rank_results, indent=2))

    print("\n=== TCAS vs Downstream Impact Rank Correlation ===")
    fr = rank_results
    print(
        f"Full (n={fr['full_n']}): "
        f"ρ={fr['full_spearman_rho']:.4f} (p={fr['full_spearman_p']:.4f})  "
        f"τ={fr['full_kendall_tau']:.4f} (p={fr['full_kendall_p']:.4f})"
    )
    print(
        f"Filtered (n={fr['filtered_n']}): "
        f"ρ={fr['filtered_spearman_rho']:.4f} (p={fr['filtered_spearman_p']:.4f})  "
        f"τ={fr['filtered_kendall_tau']:.4f} (p={fr['filtered_kendall_p']:.4f})"
    )

    # 4. Save full component table
    full_table = []
    for e in entries:
        full_table.append(
            {
                "model": e["model"],
                "property": e["property"],
                "A": e["accessibility"],
                "L": e["linearity"],
                "C": e["causality"],
                "F": e["functional_load"],
                "TCAS_geo": e["tcas_geometric"],
                "TCAS_arith": e["tcas_arithmetic"],
                "TCAS_harm": e["tcas_harmonic"],
                "TCAS_min": e["tcas_min"],
            }
        )
    (output_dir / "full_component_table.json").write_text(json.dumps(full_table, indent=2))

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
