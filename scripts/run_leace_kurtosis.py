"""Rigorous necessity test (D2): LEACE-erase the full linear kurtosis subspace from a
TSFM's representation, then measure the drop in anomaly-probe accuracy.

Unlike single-direction removal (run_kurtosis_ablation.py), LEACE (Belrose et al. 2023)
guarantees that no linear predictor can recover kurtosis after erasure, while changing
the representation minimally. If anomaly probing relies on linearly-accessible kurtosis,
LEACE erasure should collapse anomaly accuracy toward chance.

We report: kurtosis test-R^2 before vs after erasure (confirming the erasure), and
anomaly accuracy before vs after.

Usage:  PYTHONPATH=. python scripts/run_leace_kurtosis.py
Output: outputs/leace_kurtosis/results.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from scripts.run_kurtosis_ablation import _acc, _layer_files, _load_pooled, _split
from src.datasets.synthetic import generate_anomaly_dataset

OUT_DIR = Path("outputs/leace_kurtosis")
NUM, LEN, SEED = 1000, 512, 42


def _kurt_r2(Xtr, ktr, Xte, kte) -> float:
    reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xtr, ktr)
    return float(r2_score(kte, reg.predict(Xte)))


def _leace_erase(Xtr: NDArray, ktr: NDArray, Xall: NDArray) -> NDArray:
    """Fit LEACE on train (continuous kurtosis) and erase all rows."""
    ce = importlib.import_module("concept_erasure")
    z = torch.tensor(ktr, dtype=torch.float32).unsqueeze(-1)
    fitter = ce.LeaceFitter.fit(torch.tensor(Xtr, dtype=torch.float32), z)
    eraser = fitter.eraser
    return eraser(torch.tensor(Xall, dtype=torch.float32)).numpy()


def run_model(model: str, y: NDArray, kurt: NDArray) -> dict:
    tr, va, te = _split(len(y))
    best = None
    for f in _layer_files(model):
        X = _load_pooled(f)
        base_va = _acc(X[tr], y[tr], X[va], y[va])
        if best is None or base_va > best[1]:
            best = (f, base_va, X)
    f, _, X = best
    base = _acc(X[tr], y[tr], X[te], y[te])
    kr2_before = _kurt_r2(X[tr], kurt[tr], X[te], kurt[te])

    Xer = _leace_erase(X[tr], kurt[tr], X)
    kr2_after = _kurt_r2(Xer[tr], kurt[tr], Xer[te], kurt[te])
    erased = _acc(Xer[tr], y[tr], Xer[te], y[te])
    return {
        "model": model,
        "best_layer": Path(f).stem,
        "kurtosis_r2_before": round(kr2_before, 4),
        "kurtosis_r2_after_leace": round(kr2_after, 4),
        "anomaly_acc_baseline": round(base, 4),
        "anomaly_acc_leace_erased": round(erased, 4),
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
