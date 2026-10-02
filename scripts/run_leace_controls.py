"""Specificity controls for the kurtosis LEACE necessity test (addresses reviewer F4/F5).

For each model's best anomaly layer, LEACE-erase four concepts and compare the drop in
anomaly-probe accuracy:
  - kurtosis        (the claimed statistic)
  - max|x|          (the true near-sufficient statistic; corr with kurtosis ~0.92)
  - signal mean     (placebo: a real but anomaly-irrelevant statistic)
  - random target   (rank-matched random direction; the amnesic-probing control)

A causal claim specific to kurtosis requires: drop(kurtosis) >> drop(placebo)=drop(random),
AND drop(kurtosis) not dominated by drop(max|x|). Reports all four so the reader can judge.

Usage:  PYTHONPATH=. python scripts/run_leace_controls.py
Output: outputs/leace_controls/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_kurtosis_ablation import _acc, _layer_files, _load_pooled, _split
from scripts.run_leace_kurtosis import _leace_erase
from src.datasets.synthetic import generate_anomaly_dataset

OUT_DIR = Path("outputs/leace_controls")
NUM, LEN, SEED = 1000, 512, 42


def _concepts(sig: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    std = sig.std(1, keepdims=True) + 1e-10
    z = (sig - sig.mean(1, keepdims=True)) / std
    return {
        "kurtosis": np.mean(z**4, axis=1) - 3.0,
        "max_abs": np.abs(sig).max(axis=1),
        "signal_mean": sig.mean(axis=1),  # placebo: real but anomaly-irrelevant
        "random": rng.normal(size=len(sig)),  # rank-matched random direction
    }


def run_model(model: str, y: np.ndarray, concepts: dict) -> dict:
    tr, va, te = _split(len(y))
    best = None
    for f in _layer_files(model):
        X = _load_pooled(f)
        acc_va = _acc(X[tr], y[tr], X[va], y[va])
        if best is None or acc_va > best[1]:
            best = (f, acc_va, X)
    f, _, X = best
    base = _acc(X[tr], y[tr], X[te], y[te])
    row = {"model": model, "best_layer": Path(f).stem, "anomaly_acc_baseline": round(base, 4)}
    for name, c in concepts.items():
        Xer = _leace_erase(X[tr], c[tr], X)
        erased = _acc(Xer[tr], y[tr], Xer[te], y[te])
        row[f"drop_{name}"] = round(base - erased, 4)
    return row


def main() -> None:
    ds = generate_anomaly_dataset(NUM, LEN, seed=SEED)
    y = ds.labels.astype(np.int64)
    rng = np.random.default_rng(SEED)
    concepts = _concepts(ds.sequences, rng)
    models = ["moment", "chronos", "timesfm", "gpt4ts", "timer", "moirai", "patchtst_pretrained"]
    rows = []
    for m in models:
        try:
            r = run_model(m, y, concepts)
        except Exception as e:  # pragma: no cover
            r = {"model": m, "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        print(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"saved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
