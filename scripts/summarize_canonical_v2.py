"""Summarise outputs/canonical_v2 and compare it with the committed canonical numbers.

Reads:
    <v2_root>/<model>_<property>/canonical_results.json      (new, one protocol)
    <v2_root>/baselines/all_results.json                      (new baselines)
    outputs/canonical/<old_dir>_<property>/canonical_results.json   (committed models)
    outputs/canonical_baselines/all_results.json              (committed baselines)

Writes:
    <v2_root>/summary.json    every cell: new and old mean / 95% CI / per-seed, delta, flags
    <v2_root>/comparison.md   the same as Markdown tables (pasted into README.md)

A cell is flagged ``outside_old_ci`` when the new test mean lies outside the committed 95% CI.

Usage:
    PYTHONPATH=. python scripts/summarize_canonical_v2.py --v2_root outputs/canonical_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROPS: list[tuple[str, str]] = [
    ("trend", "acc"),
    ("seasonality", "R2"),
    ("frequency", "acc"),
    ("stationarity", "acc"),
    ("anomaly", "acc"),
    ("change_point", "acc"),
]

# new model key -> (committed outputs/canonical dir prefix, committed recipe, v2 view)
MODELS: dict[str, tuple[str, str, str]] = {
    "moment": (
        "moment_pca512",
        "n=5000; PCA-512 fit on all 5000 samples (incl. val/test) before splitting",
        "flattened full layer; PCA-512 fit on train split per seed (in pipeline)",
    ),
    "gpt4ts": (
        "gpt4ts_pca512",
        "n=5000; PCA-512 fit on all 5000 samples (incl. val/test) before splitting",
        "flattened full layer; PCA-512 fit on train split per seed (in pipeline)",
    ),
    "chronos": ("chronos", "n=5000; mean-pool over 33 tokens", "mean-pool over tokens"),
    "timer": ("timer_meanpool", "n=1000; mean-pool over patches", "mean-pool over patches"),
    "timesfm": ("timesfm_meanpool", "n=1000; mean-pool over patches", "mean-pool over patches"),
    "moirai": ("moirai_meanpool", "n=1000; mean-pool over patches", "mean-pool over patches"),
    "patchtst_fm": (
        "patchtst_fm_meanpool",
        "n=1000; mean-pool over patches (outputs/patchtst_replacement)",
        "mean-pool over patches",
    ),
}
BASELINES = ["hand_crafted", "raw_signal", "random_projection"]
DISPLAY = {
    "moment": "MOMENT",
    "gpt4ts": "GPT4TS",
    "chronos": "Chronos-Bolt",
    "timer": "Timer",
    "timesfm": "TimesFM",
    "moirai": "Moirai",
    "patchtst_fm": "PatchTST-FM",
    "hand_crafted": "Hand-crafted (8-D)",
    "raw_signal": "Raw signal",
    "random_projection": "Random proj. (256-D)",
}


def _stats(r: dict[str, Any]) -> dict[str, Any]:
    per_seed = r.get("test_score_per_seed", r.get("per_seed_test"))
    out = {
        "mean": r["test_mean"],
        "ci95": [r["test_ci95_low"], r["test_ci95_high"]],
        "per_seed": per_seed,
    }
    if "best_layer_per_seed" in r:
        out["best_layer_per_seed"] = r["best_layer_per_seed"]
    return out


def _compare(new: dict[str, Any], old: dict[str, Any] | None) -> dict[str, Any]:
    if old is None:
        return {"old": None, "delta": None, "outside_old_ci": None, "ci_overlap": None}
    lo, hi = old["ci95"]
    nlo, nhi = new["ci95"]
    return {
        "old": old,
        "delta": round(new["mean"] - old["mean"], 4),
        "outside_old_ci": not (lo <= new["mean"] <= hi),
        "ci_overlap": not (nhi < lo or nlo > hi),
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def build(v2_root: Path, old_root: Path, old_bl_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    missing: list[str] = []
    for model, (old_dir, old_recipe, view) in MODELS.items():
        for prop, metric in PROPS:
            new_r = _load(v2_root / f"{model}_{prop}" / "canonical_results.json")
            if new_r is None:
                missing.append(f"{model}_{prop}")
                continue
            old_r = _load(old_root / f"{old_dir}_{prop}" / "canonical_results.json")
            new = _stats(new_r)
            old = _stats(old_r) if old_r is not None else None
            cells.append(
                {
                    "kind": "model",
                    "name": model,
                    "property": prop,
                    "metric": metric,
                    "new": new,
                    "new_view": view,
                    "old_source": f"{old_root}/{old_dir}_{prop}",
                    "old_recipe": old_recipe,
                    **_compare(new, old),
                }
            )

    new_bl = _load(v2_root / "baselines" / "all_results.json") or []
    old_bl = _load(old_bl_root / "all_results.json") or []
    old_bl_idx = {(r["property"], r["baseline"]): r for r in old_bl}
    for r in new_bl:
        o = old_bl_idx.get((r["property"], r["baseline"]))
        new = _stats(r)
        old = _stats(o) if o is not None else None
        metric = dict(PROPS)[r["property"]]
        cells.append(
            {
                "kind": "baseline",
                "name": r["baseline"],
                "property": r["property"],
                "metric": metric,
                "new": new,
                "old_source": f"{old_bl_root}/all_results.json",
                "old_recipe": "n=1000, data seed 42 (same data and splits)",
                **_compare(new, old),
            }
        )

    # Anomaly headline: every TSFM against the hand-crafted baseline, old vs new.
    hc = next(
        (
            c
            for c in cells
            if c["kind"] == "baseline"
            and c["name"] == "hand_crafted"
            and c["property"] == "anomaly"
        ),
        None,
    )
    headline = []
    if hc is not None:
        for c in cells:
            if c["kind"] != "model" or c["property"] != "anomaly":
                continue
            headline.append(
                {
                    "model": c["name"],
                    "old_gap_vs_hc": (
                        round(c["old"]["mean"] - hc["old"]["mean"], 4) if c["old"] else None
                    ),
                    "new_gap_vs_hc": round(c["new"]["mean"] - hc["new"]["mean"], 4),
                    "new_ci_below_hc_ci": c["new"]["ci95"][1] < hc["new"]["ci95"][0],
                }
            )

    flagged = [f"{c['name']}/{c['property']}" for c in cells if c["outside_old_ci"] is True]
    return {
        "protocol": {
            "data": "6 synthetic properties, n=1000, seq_len=512, data seed 42",
            "split": "60/20/20 train/val/test, split seeds 0-4",
            "selection": "best layer on validation only; test score reported",
            "ci": "1000-resample bootstrap over the 5 seed scores, 95%",
            "estimator": "StandardScaler + LogisticRegression(lbfgs, max_iter=1000) / Ridge(1.0)",
            "views": {m: v[2] for m, v in MODELS.items()},
        },
        "cells": cells,
        "missing_cells": missing,
        "flagged_outside_old_ci": flagged,
        "anomaly_vs_hand_crafted": headline,
    }


def _fmt(s: dict[str, Any] | None) -> str:
    if s is None:
        return "n/a"
    return f"{s['mean']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"


def to_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for kind, title in (("model", "TSFMs"), ("baseline", "No-model baselines")):
        lines += [
            f"### {title}",
            "",
            "| Model | Property | Old (committed) mean [95% CI] | New (v2) mean [95% CI] "
            "| Δ new − old | Outside old CI |",
            "|---|---|---|---|---|---|",
        ]
        for c in summary["cells"]:
            if c["kind"] != kind:
                continue
            flag = "**YES**" if c["outside_old_ci"] else ("no" if c["old"] else "n/a")
            delta = f"{c['delta']:+.4f}" if c["delta"] is not None else "n/a"
            lines.append(
                f"| {DISPLAY.get(c['name'], c['name'])} | {c['property']} ({c['metric']}) | "
                f"{_fmt(c['old'])} | {_fmt(c['new'])} | {delta} | {flag} |"
            )
        lines.append("")
    lines += [
        "### Anomaly: each TSFM minus the hand-crafted baseline (accuracy)",
        "",
        "| Model | Old gap | New gap | New TSFM CI entirely below new HC CI |",
        "|---|---|---|---|",
    ]
    for h in summary["anomaly_vs_hand_crafted"]:
        old = f"{h['old_gap_vs_hc']:+.4f}" if h["old_gap_vs_hc"] is not None else "n/a"
        lines.append(
            f"| {DISPLAY[h['model']]} | {old} | {h['new_gap_vs_hc']:+.4f} | "
            f"{'yes' if h['new_ci_below_hc_ci'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Summarise canonical v2 vs committed canonical")
    p.add_argument("--v2_root", default="outputs/canonical_v2")
    p.add_argument("--old_root", default="outputs/canonical")
    p.add_argument("--old_baselines_root", default="outputs/canonical_baselines")
    args = p.parse_args(argv)
    v2_root = Path(args.v2_root)
    summary = build(v2_root, Path(args.old_root), Path(args.old_baselines_root))
    (v2_root / "summary.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary)
    (v2_root / "comparison.md").write_text(md)
    print(md)
    if summary["missing_cells"]:
        print(f"MISSING cells: {summary['missing_cells']}")
    print(f"Flagged (new mean outside old 95% CI): {summary['flagged_outside_old_ci']}")


if __name__ == "__main__":
    main()
