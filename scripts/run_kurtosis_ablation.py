"""Causal-adjacent necessity test (D2): erase the kurtosis-encoding direction from a
TSFM's representation and measure the drop in anomaly-probe accuracy.

If anomaly probing relies on linearly-encoded kurtosis, removing that single direction
should collapse anomaly accuracy toward chance. Complements the dose-response (which is
correlational) with a per-model intervention.

Method (single-concept linear erasure):
  1. best anomaly layer selected on a val split;
  2. fit Ridge(kurtosis ~ X) on train -> unit direction w-hat;
  3. erase: X' = X - (X w-hat) w-hat^T  (remove the kurtosis-predictive component);
  4. compare anomaly LogisticRegression accuracy on X vs X'.

Requires reps + labels from scripts/extract_representations.py (synthetic_anomaly).

Usage:  PYTHONPATH=. python scripts/run_kurtosis_ablation.py
Output: outputs/kurtosis_ablation/results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.synthetic import generate_anomaly_dataset

PCA_DIM = 64  # reduce before fitting the kurtosis direction so it generalizes (avoids overfit)
OUT_DIR = Path("outputs/kurtosis_ablation")
NUM, LEN, SEED = 1000, 512, 42


def _layer_files(model: str) -> list[str]:
    hits = [f for f in glob.glob(f"outputs/representations/{model}*/synthetic_anomaly/*.pt")
            if "label" not in f.lower()]

    def idx(f: str) -> int:
        m = re.findall(r"(\d+)", Path(f).stem)
        return int(m[-1]) if m else 0

    return sorted(hits, key=idx)


def _load_pooled(f: str) -> NDArray[np.float64]:
    X = torch.load(f, map_location="cpu").float()
    while X.ndim > 2:  # pool any patch/token/channel axes -> (N, D)
        X = X.mean(dim=1)
    return X.numpy().astype(np.float64)


def _split(n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = int(n * 0.6)
    nva = int(n * 0.2)
    return idx[:ntr], idx[ntr : ntr + nva], idx[ntr + nva :]


def _acc(Xtr, ytr, Xte, yte) -> float:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    pipe.fit(Xtr, ytr)
    return float(accuracy_score(yte, pipe.predict(Xte)))


def _pca_reduce(X, tr):
    ncomp = min(PCA_DIM, X.shape[1], len(tr) - 1)
    pca = PCA(n_components=ncomp, random_state=SEED).fit(X[tr])
    return pca.transform(X)


def run_model(model: str, y: NDArray, kurt: NDArray) -> dict:
    tr, va, te = _split(len(y))
    # select best anomaly layer (in PCA space) on the val split
    best = None
    for f in _layer_files(model):
        Xp = _pca_reduce(_load_pooled(f), tr)
        base_va = _acc(Xp[tr], y[tr], Xp[va], y[va])
        if best is None or base_va > best[1]:
            best = (f, base_va, Xp)
    f, _, Xp = best
    base = _acc(Xp[tr], y[tr], Xp[te], y[te])

    # generalizable kurtosis direction (RidgeCV on train), reported by TEST R^2
    reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xp[tr], kurt[tr])
    kr2_test = float(r2_score(kurt[te], reg.predict(Xp[te])))
    w = reg.coef_.astype(np.float64)
    wh = w / (np.linalg.norm(w) + 1e-12)
    # Erase the COVARIANCE direction Sigma @ w-hat, not the raw coefficient w-hat.
    # Under correlated features the two differ (~60 deg here); removing Sigma @ w-hat
    # is the rank-1 subspace LEACE erases for a scalar target, and it fully removes the
    # linear kurtosis information in a single (correctly chosen) direction.
    sigma = np.cov(Xp[tr].T)
    d = sigma @ wh
    dh = d / (np.linalg.norm(d) + 1e-12)
    Xer = Xp - (Xp @ dh)[:, None] * dh[None, :]
    erased = _acc(Xer[tr], y[tr], Xer[te], y[te])
    return {
        "model": model,
        "best_layer": Path(f).stem,
        "kurtosis_test_r2": round(kr2_test, 4),
        "anomaly_acc_baseline": round(base, 4),
        "anomaly_acc_kurtosis_erased": round(erased, 4),
        "drop": round(base - erased, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["moment", "chronos", "timesfm", "gpt4ts", "timer", "moirai"])
    args = ap.parse_args()
    ds = generate_anomaly_dataset(NUM, LEN, seed=SEED)
    y = ds.labels.astype(np.int64)
    sig = ds.sequences
    std = sig.std(1, keepdims=True) + 1e-10
    kurt = np.mean(((sig - sig.mean(1, keepdims=True)) / std) ** 4, axis=1) - 3.0

    rows = []
    for m in args.models:
        try:
            r = run_model(m, y, kurt)
        except Exception as e:  # pragma: no cover
            r = {"model": m, "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        print(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"saved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
