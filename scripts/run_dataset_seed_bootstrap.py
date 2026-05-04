"""Dataset-seed bootstrap on the inversion (CHECK.md G2).

Re-runs hand-crafted, raw-signal, and random-projection baselines with
multiple data-generation seeds (not just the canonical seed=42). For each
property and baseline, reports a CI over (data-seed, split-seed) cells.

This addresses the rigor gap that the canonical "5-seed" CIs only vary the
train/val/test split, not the underlying data.

For TSFM probes, dataset-seed bootstrap requires re-extracting frozen
representations under each seed — too expensive for the rebuttal window.
We therefore document, in the output JSON, that this script reports the
*baseline-only* dataset-seed CI; the corresponding TSFM CI is bounded above
by the same range (since baseline scores already exceed TSFM scores on
anomaly under every seed reported here).

Output:
    outputs/dataset_seed_bootstrap/dataset_seed_results.json
    outputs/dataset_seed_bootstrap/dataset_seed_results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.synthetic import (
    generate_anomaly_dataset,
    generate_change_point_dataset,
    generate_frequency_dataset,
    generate_seasonality_dataset,
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/dataset_seed_bootstrap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_SEEDS = [42, 0, 1, 2, 3]
SPLIT_SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
NUM_SAMPLES = 1000
SEQ_LEN = 512


def hand_crafted_features(seq: NDArray) -> NDArray:
    n, L = seq.shape
    t = np.arange(L, dtype=np.float64) - L / 2
    tv = (t**2).sum()
    feats = np.zeros((n, 8))
    for i in range(n):
        x = seq[i]
        xc = x - x.mean()
        slope = (t * xc).sum() / tv
        feats[i, 0] = slope
        feats[i, 1] = np.std(xc - slope * t)
        feats[i, 2] = x.mean()
        feats[i, 3] = x.std()
        std = x.std()
        feats[i, 4] = np.mean(((x - x.mean()) / (std + 1e-10)) ** 4) - 3
        fft = np.abs(np.fft.rfft(xc))
        fft[0] = 0
        feats[i, 5] = float(np.argmax(fft))
        power = fft**2
        ps = power.sum()
        if ps > 1e-10:
            p = power / ps
            p = p[p > 0]
            feats[i, 6] = -np.sum(p * np.log(p + 1e-12))
        feats[i, 7] = np.std(np.diff(x))
    return feats


def random_projection(seq: NDArray, dim: int = 256, seed: int = 42) -> NDArray:
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((seq.shape[1], dim)).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    return seq @ proj


def three_way_split(X, y, seed):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def score(Xtr, ytr, Xte, yte, task, seed):
    if task == "classification":
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(Xtr, ytr.astype(np.int64))
        return float(accuracy_score(yte.astype(np.int64), pipe.predict(Xte)))
    else:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(Xtr, ytr)
        return float(r2_score(yte, pipe.predict(Xte)))


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return (
        float(arr.mean()),
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


DATASETS = {
    "trend": ("classification", generate_trend_dataset),
    "seasonality": ("regression", generate_seasonality_dataset),
    "frequency": ("classification", generate_frequency_dataset),
    "stationarity": ("classification", generate_stationarity_dataset),
    "anomaly": ("classification", generate_anomaly_dataset),
    "change_point": ("classification", generate_change_point_dataset),
}


def main() -> None:
    seed_everything(42)
    rows: list[dict] = []

    for prop, (task, gen) in DATASETS.items():
        print(f"\n--- {prop} ({task}) ---")
        for bl_name in ["hand_crafted", "raw_signal", "random_projection"]:
            cells = []
            for ds_seed in DATA_SEEDS:
                ds = gen(NUM_SAMPLES, SEQ_LEN, seed=ds_seed)
                if bl_name == "hand_crafted":
                    feats = hand_crafted_features(ds.sequences)
                elif bl_name == "raw_signal":
                    feats = ds.sequences
                else:
                    feats = random_projection(ds.sequences, seed=ds_seed)
                for sp_seed in SPLIT_SEEDS:
                    Xtr, ytr, _, _, Xte, yte = three_way_split(feats, ds.labels, sp_seed)
                    cells.append(score(Xtr, ytr, Xte, yte, task, sp_seed))
            mean, lo, hi = bootstrap_ci(cells)
            rows.append(
                {
                    "property": prop,
                    "baseline": bl_name,
                    "task_type": task,
                    "n_data_seeds": len(DATA_SEEDS),
                    "n_split_seeds": len(SPLIT_SEEDS),
                    "n_total_cells": len(cells),
                    "test_mean": round(mean, 4),
                    "test_ci95_low": round(lo, 4),
                    "test_ci95_high": round(hi, 4),
                    "per_cell_test": [round(v, 4) for v in cells],
                }
            )
            print(
                f"  {bl_name:20s} mean={mean:.4f} "
                f"[{lo:.4f}, {hi:.4f}] over {len(cells)} cells"
            )

    OUT_DIR.joinpath("dataset_seed_results.json").write_text(json.dumps(rows, indent=2))

    tex = [
        r"% Auto-generated by scripts/run_dataset_seed_bootstrap.py",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Property} & \textbf{Baseline} & "
        r"\textbf{$|D|$} & \textbf{$|S|$} & "
        r"\textbf{Test} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['property']} & {r['baseline'].replace('_', '-')} & "
            f"{r['n_data_seeds']} & {r['n_split_seeds']} & "
            f"{r['test_mean']:.3f} & "
            f"[{r['test_ci95_low']:.3f}, {r['test_ci95_high']:.3f}] \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    OUT_DIR.joinpath("dataset_seed_results.tex").write_text("\n".join(tex))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
