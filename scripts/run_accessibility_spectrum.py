"""Step 1 of the transfer-function program: measure the linear ACCESSIBILITY SPECTRUM
of frozen TSFM representations over a principled functional basis (Hermite moments).

For Gaussian-ish inputs, the natural degree-k functional is the empirical probabilist-
Hermite moment  f_k(x) = mean_t He_k(x_t)  (He_1=x, He_2=x^2-1, ..., He_6). Kurtosis is
~degree 4; a lone tail spike loads on high degrees. We measure, per model, how well a
regularized linear probe recovers f_k from the representation (held-out test R^2). The
HYPOTHESIS (H5) is that accessibility decays with Hermite degree k, and that the decay
is a property of the pretraining objective — not of any single task.

This replaces the (false) "kurtosis sufficiency" framing with a spectrum: the anomaly
blind spot becomes the high-degree tail of a measurable transfer function.

To avoid the anomaly-optimal-layer circularity (reviewer F8), we use a FIXED a-priori
layer (middle of the stack) and also report the max over layers for reference.

Usage:  PYTHONPATH=. python scripts/run_accessibility_spectrum.py
Output: outputs/accessibility_spectrum/results.json
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermeval
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from scripts.run_kurtosis_ablation import _split
from src.datasets.synthetic import generate_anomaly_dataset

OUT_DIR = Path("outputs/accessibility_spectrum")
NUM, LEN, SEED = 1000, 512, 42
DEGREES = (1, 2, 3, 4, 5, 6)
MODELS = ("moment", "chronos", "timesfm", "gpt4ts", "timer", "moirai", "patchtst_pretrained")


def hermite_moment(sig: np.ndarray, k: int) -> np.ndarray:
    """Empirical probabilist-Hermite moment of degree k: mean_t He_k(x_t) per window."""
    coef = np.zeros(k + 1)
    coef[k] = 1.0
    he = hermeval(sig, coef)  # He_k evaluated elementwise, shape (N, L)
    return he.mean(axis=1)


def bayes_ceiling(sig: np.ndarray, labels: np.ndarray) -> float:
    """Exact Bayes accuracy: LLR = log mean_t cosh(5 x_t) - 12.5, oracle threshold."""
    llr = np.log(np.cosh(5.0 * sig).mean(axis=1) + 1e-30) - 12.5
    thr = np.unique(llr)
    best = 0.0
    for t in thr:
        acc = max(((llr >= t) == labels).mean(), ((llr < t) == labels).mean())
        best = max(best, acc)
    return float(best)


def _layers(model: str) -> list[str]:
    hits = [f for f in glob.glob(f"outputs/representations/{model}*/synthetic_anomaly/*.pt")
            if "label" not in f.lower()]
    return sorted(hits, key=lambda f: int(re.findall(r"\d+", Path(f).stem)[-1] or 0))


def _pooled(f: str) -> np.ndarray:
    X = torch.load(f, map_location="cpu").float()
    while X.ndim > 2:
        X = X.mean(dim=1)
    return X.numpy().astype(np.float64)


def accessibility(X: np.ndarray, target: np.ndarray, tr, te) -> float:
    ncomp = min(64, X.shape[1], len(tr) - 1)
    Xp = PCA(n_components=ncomp, random_state=SEED).fit(X[tr]).transform(X)
    reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xp[tr], target[tr])
    return float(r2_score(target[te], reg.predict(Xp[te])))


def main() -> None:
    ds = generate_anomaly_dataset(NUM, LEN, seed=SEED)
    sig, y = ds.sequences, ds.labels.astype(np.int64)
    tr, _, te = _split(NUM)
    targets = {k: hermite_moment(sig, k) for k in DEGREES}
    bayes = bayes_ceiling(sig, y)

    rows = []
    for m in MODELS:
        layers = _layers(m)
        if not layers:
            continue
        mid_f = layers[len(layers) // 2]  # fixed a-priori layer (anti-circular)
        Xmid = _pooled(mid_f)
        spec_mid, spec_max = {}, {}
        for k in DEGREES:
            spec_mid[k] = round(accessibility(Xmid, targets[k], tr, te), 4)
            spec_max[k] = round(max(accessibility(_pooled(f), targets[k], tr, te)
                                    for f in layers), 4)
        rows.append({"model": m, "mid_layer": Path(mid_f).stem, "n_layers": len(layers),
                     "A_mid_by_degree": spec_mid, "A_max_by_degree": spec_max})
        print(f"{m:20} mid-layer A(k): " + "  ".join(f"H{k}={spec_mid[k]:+.2f}" for k in DEGREES))

    # decay check: mean accessibility per degree across models (mid layer)
    mean_by_deg = {k: round(float(np.mean([r["A_mid_by_degree"][k] for r in rows])), 4)
                   for k in DEGREES}
    print(f"\nBayes ceiling (exact) = {bayes:.4f}")
    print("mean A(k) across models (mid layer):", mean_by_deg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(
        {"bayes_ceiling": round(bayes, 4), "mean_A_by_degree_midlayer": mean_by_deg,
         "per_model": rows, "note": "A(k)=held-out ridge R^2 recovering empirical Hermite-k "
         "moment from representation; H5 predicts decay in k."}, indent=2))
    print(f"saved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
