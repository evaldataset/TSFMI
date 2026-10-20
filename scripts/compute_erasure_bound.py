"""How much label information can survive linear erasure of a concept? An exact in-sample bound.

LEACE makes every linear function of the erased representation Z' uncorrelated with the concept S
on the data it is fit on. Write the label as y = yhat + e, with yhat the least-squares projection
of y on (1, S) and e the residual. For any linear score u = w'Z', Cov(u, yhat) = 0, so
|corr(u, y)| = |Cov(u, e)| / (sd(u) sd(y)) <= sd(e) / sd(y) = sqrt(1 - R^2(y ~ S)).
Erasing a concept therefore caps the linear correlation of any probe with the label at
sqrt(1 - R^2), whatever the representation encodes. This script reports R^2 and the cap on the
training split of every split seed, for the concepts erased in the paper, and the label proxy's.

Usage:
    PYTHONPATH=. python scripts/compute_erasure_bound.py --out outputs/erasure_bound
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.run_canonical_baselines import hand_crafted_features
from scripts.run_canonical_benchmark import SEEDS, three_way_split
from scripts.run_realistic_anomaly import realistic_anomaly_dataset
from scripts.run_statistic_erasure import DATA_SEED, NUM_SAMPLES, SEQ_LEN, label_proxy
from scripts.run_statistic_recoverability import targets
from src.datasets.synthetic import generate_anomaly_dataset


def load_task(task: str) -> tuple[NDArray, NDArray]:
    if task == "realistic":
        x, y = realistic_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, DATA_SEED)
    else:
        ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
        x, y = ds.sequences, ds.labels
    return np.asarray(x), np.asarray(y).astype(np.int64)


def concept_matrix(x: NDArray, concept: str) -> NDArray:
    if concept == "statistics":
        t = targets(x)
        return np.stack([t["llr"], t["max_abs"], t["kurtosis"]], axis=1)
    return hand_crafted_features(x)


def r_squared(y: NDArray, S: NDArray) -> float:
    X = np.column_stack([np.ones(len(y)), S])
    coef, *_ = np.linalg.lstsq(X, y.astype(np.float64), rcond=None)
    resid = y - X @ coef
    return float(1.0 - resid.var() / y.var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/erasure_bound"))
    args = ap.parse_args()
    out: dict = {}
    for task, concept in (
        ("canonical", "statistics"),
        ("canonical", "handcrafted"),
        ("realistic", "handcrafted"),
        ("realistic", "statistics"),
    ):
        seqs, y = load_task(task)
        S = concept_matrix(seqs, concept)
        rows = []
        for seed in SEEDS:
            S_tr, y_tr, *_ = three_way_split(S, y, seed)
            r2 = r_squared(y_tr, S_tr)
            zs_tr = (S_tr - S_tr.mean(0)) / (S_tr.std(0) + 1e-12)
            proxy, _, _ = label_proxy(zs_tr, y_tr, seed)
            rows.append(
                {
                    "seed": seed,
                    "r2": r2,
                    "cap": float(np.sqrt(max(0.0, 1.0 - r2))),
                    "proxy_r2": r_squared(y_tr, proxy),
                }
            )
        out[f"{task}/{concept}"] = {
            "per_seed": rows,
            "r2_mean": float(np.mean([r["r2"] for r in rows])),
            "cap_mean": float(np.mean([r["cap"] for r in rows])),
            "proxy_r2_mean": float(np.mean([r["proxy_r2"] for r in rows])),
        }
        summary = {k: round(v, 3) for k, v in out[f"{task}/{concept}"].items() if k != "per_seed"}
        print(task, concept, summary)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
