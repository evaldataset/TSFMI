"""Single-feature probes of the 8-D hand-crafted vector on all six canonical properties.

For every property and every one of the eight hand-crafted features (and the full vector), the
canonical probe is fit on that feature alone: n = 1000, data seed 42, 60/20/20 split, split seeds
0-4, StandardScaler + LogisticRegression (accuracy) or Ridge (R^2, seasonality), bootstrap 95% CI.
This is the per-property ablation that shows which feature carries which property.

Usage:
    PYTHONPATH=. python scripts/run_per_feature_all_properties.py \
        --out outputs/per_feature_all_properties
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.run_canonical_baselines import DATASETS, hand_crafted_features
from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, three_way_split, train_and_score

FEATURES = (
    "slope",
    "residual_std",
    "mean",
    "std",
    "kurtosis",
    "argmax_fft",
    "spectral_entropy",
    "diff_std",
)
NUM_SAMPLES, SEQ_LEN, DATA_SEED = 1000, 512, 42


def score(X, y, task: str) -> dict:
    per_seed = []
    for seed in SEEDS:
        X_tr, y_tr, _, _, X_te, y_te = three_way_split(X, y, seed)
        per_seed.append(train_and_score(X_tr, y_tr, X_te, y_te, task, seed))
    mean, lo, hi = bootstrap_ci(per_seed)
    return {
        "mean": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "per_seed": [round(v, 4) for v in per_seed],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/per_feature_all_properties"))
    args = ap.parse_args()
    results = {}
    for prop, (task, gen) in DATASETS.items():
        ds = gen(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
        H = hand_crafted_features(ds.sequences)
        row = {"task_type": task, "full": score(H, ds.labels, task)}
        for i, name in enumerate(FEATURES):
            row[name] = score(H[:, [i]], ds.labels, task)
        results[prop] = row
        print(
            f"{prop:<13} full={row['full']['mean']:.3f} "
            + " ".join(f"{f}={row[f]['mean']:.3f}" for f in FEATURES),
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "protocol": "canonical: n=1000, data seed 42, 60/20/20, split seeds 0-4, "
                "StandardScaler + LogisticRegression / Ridge(alpha=1), bootstrap 95% CI",
                "features": list(FEATURES),
                "results": results,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
