"""Summarize the PatchTST checkpoint replacement (legacy ETTh1 PatchTST -> PatchTST-FM-r1).

Compares, per canonical synthetic property, the committed ``patchtst_pretrained`` canonical
scores with the replacement ``patchtst_fm_meanpool`` scores (same canonical protocol:
60/20/20 split, val-only layer selection, 5 seeds, 1000-resample bootstrap 95% CI), plus:

* ``legacy_rerun``: the legacy checkpoint re-run on the *current* 1000-sample extraction,
  which shares dataset and splits with the replacement, so per-seed differences are paired.
* ``sensitivity_flat``: the replacement's raw (N, 32, 1024) activations flattened to
  32768-D, i.e. the same reduction family (identity/flatten) as ``patchtst_pretrained``.
* ``legacy_rerun_meanpool``: the legacy re-run mean-pooled over its 43 tokens, completing
  the model x view 2x2 so reduction effects are separable from model effects.
* the no-model baselines from ``outputs/canonical_baselines/all_results.json``.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/summarize_patchtst_replacement.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_canonical_benchmark import bootstrap_ci

PROPERTIES = ["trend", "seasonality", "frequency", "stationarity", "anomaly", "change_point"]
CANONICAL = Path("outputs/canonical")
REPLACEMENT = Path("outputs/patchtst_replacement")
BASELINES = Path("outputs/canonical_baselines/all_results.json")

OLD_NAME = "patchtst_pretrained"
NEW_NAME = "patchtst_fm_meanpool"

MODEL_FACTS: dict[str, dict[str, Any]] = {
    OLD_NAME: {
        "checkpoint": "ibm-granite/granite-timeseries-patchtst",
        "same_weights_as": "namctin/patchtst_etth1_forecast (config _name_or_path)",
        "training_data": "ETTh1 train split (supervised 96-step forecasting)",
        "architecture": "HF PatchTSTForPrediction, 3 layers, d_model 128, patch 12/stride 12, CLS",
        "canonical_view": "identity (flattened 43 x 128 = 5504-D)",
    },
    NEW_NAME: {
        "checkpoint": "ibm-granite/granite-timeseries-patchtst-fm-r1",
        "training_data": (
            "GIFT-Eval-Pretrain (132 datasets) + 10M KernelSynth + 4M 'clean' TSMixup "
            "excluding GIFT-Eval eval datasets"
        ),
        "architecture": (
            "PatchTST-FM (tsfm_public PatchTSTFMForPrediction): 20 pre-norm Transformer "
            "blocks, d_model 1024, 16 heads, patch 16/stride 16, 257,895,552 params"
        ),
        "hooked_layers": "backbone.blocks.0 .. backbone.blocks.19",
        "canonical_view": "meanpool over the 32 context patches (1024-D)",
    },
}


def _load(path: Path) -> dict[str, Any] | None:
    """Load a canonical_results.json, or None when absent."""
    return json.loads(path.read_text()) if path.exists() else None


def _cell(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce a canonical_results.json to the fields reported in the summary."""
    if result is None:
        return None
    scores = result["test_score_per_seed"]
    return {
        "test_mean": result["test_mean"],
        "test_ci95": [result["test_ci95_low"], result["test_ci95_high"]],
        "test_score_per_seed": scores,
        "best_layer_per_seed": result["best_layer_per_seed"],
        # n_test = 200 for 1000 samples; committed legacy cells are on a 1/1000 grid.
        "scores_on_1_over_200_grid": bool(
            all(abs(s * 200 - round(s * 200)) < 1e-6 for s in scores)
        ),
    }


def _paired_diff(new: dict[str, Any] | None, ref: dict[str, Any] | None) -> dict[str, Any] | None:
    """Per-seed paired difference new - ref with the canonical bootstrap CI."""
    if new is None or ref is None:
        return None
    diffs = list(np.subtract(new["test_score_per_seed"], ref["test_score_per_seed"]))
    mean, lo, hi = bootstrap_ci([float(d) for d in diffs])
    return {"mean": mean, "ci95": [lo, hi], "per_seed": [float(d) for d in diffs]}


def _ci_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool | None:
    """Whether two cells' 95% CIs overlap."""
    if a is None or b is None:
        return None
    return bool(a["test_ci95"][0] <= b["test_ci95"][1] and b["test_ci95"][0] <= a["test_ci95"][1])


def _fmt(cell: dict[str, Any] | None) -> str:
    """Format a cell as 'mean [lo, hi]'."""
    if cell is None:
        return "n/a"
    lo, hi = cell["test_ci95"]
    return f"{cell['test_mean']:.4f} [{lo:.4f}, {hi:.4f}]"


def main() -> None:
    """Write outputs/patchtst_replacement/results.json."""
    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else []
    rows: dict[str, Any] = {}
    for prop in PROPERTIES:
        old = _cell(_load(CANONICAL / f"{OLD_NAME}_{prop}" / "canonical_results.json"))
        new = _cell(_load(CANONICAL / f"{NEW_NAME}_{prop}" / "canonical_results.json"))
        rerun = _cell(_load(REPLACEMENT / "legacy_rerun" / prop / "canonical_results.json"))
        flat = _cell(_load(REPLACEMENT / "sensitivity_flat" / prop / "canonical_results.json"))
        rerun_mp = _cell(
            _load(REPLACEMENT / "legacy_rerun_meanpool" / prop / "canonical_results.json")
        )
        base = {
            b["baseline"]: {
                "test_mean": b["test_mean"],
                "test_ci95": [b["test_ci95_low"], b["test_ci95_high"]],
            }
            for b in baselines
            if b.get("property") == prop
        }
        metric = "R2" if prop == "seasonality" else "accuracy"
        rows[prop] = {
            "metric": metric,
            "old_committed": old,
            "new_canonical": new,
            "legacy_rerun_same_protocol": rerun,
            "new_sensitivity_flat": flat,
            "legacy_rerun_meanpool": rerun_mp,
            "delta_new_minus_old_committed": (
                None if old is None or new is None else new["test_mean"] - old["test_mean"]
            ),
            "ci_overlap_new_vs_old_committed": _ci_overlap(new, old),
            "paired_new_minus_legacy_rerun": _paired_diff(new, rerun),
            "paired_same_view_meanpool_new_minus_legacy": _paired_diff(new, rerun_mp),
            "paired_same_view_flat_new_minus_legacy": _paired_diff(flat, rerun),
            "baselines": base,
        }

    summary = {
        "description": (
            "PatchTST checkpoint replacement: committed patchtst_pretrained canonical cells vs "
            "PatchTST-FM-r1 (patchtst_fm_meanpool). Canonical protocol: 60/20/20 split, "
            "val-only best-layer selection, seeds 0-4, 1000-resample bootstrap 95% CI "
            "(scripts/run_canonical_benchmark.py)."
        ),
        "models": MODEL_FACTS,
        "caveats": [
            "Committed patchtst_pretrained per-seed scores lie on a 1/1000 grid (n_test=1000, "
            "i.e. a 5000-sample extraction) while the current pipeline and the new cells use "
            "1000 samples (n_test=200). legacy_rerun_same_protocol re-derives the legacy cells "
            "on the identical 1000-sample data/splits so the paired difference is clean.",
            "new_canonical uses mean-pool (as Timer/TimesFM/Moirai); the legacy cells use the "
            "flattened identity view. new_sensitivity_flat removes that reduction confound.",
        ],
        "properties": rows,
    }
    out = REPLACEMENT / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")
    for prop, row in rows.items():
        print(
            f"{prop:13s} old={_fmt(row['old_committed'])}"
            f" rerun={_fmt(row['legacy_rerun_same_protocol'])}"
            f" rerun_mp={_fmt(row['legacy_rerun_meanpool'])}"
            f" new={_fmt(row['new_canonical'])} flat={_fmt(row['new_sensitivity_flat'])}"
        )


if __name__ == "__main__":
    main()
