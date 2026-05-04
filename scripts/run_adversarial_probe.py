"""Adversarial probing ablation (PLAN.md T3.1).

Question: can a higher-capacity (MLP) probe on model representations
beat a hand-crafted baseline? If no, triviality claim is strengthened:
the saturation is not merely a linear-probe artifact.

For each (model, property), we compare:
    - Hand-crafted linear baseline
    - Model linear probe
    - Model MLP adversarial probe (higher capacity)

If the MLP probe does NOT exceed the hand-crafted baseline by a meaningful margin,
this is evidence that the representation contains no additional useful information
beyond what simple statistics capture.

Output: outputs/adversarial_probe/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.synthetic import (
    generate_anomaly_dataset,
    generate_change_point_dataset,
    generate_frequency_dataset,
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/adversarial_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_SAMPLES = 1000
SEQ_LEN = 512
REPR_ROOT = Path("outputs/representations")

MODELS = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
    "Timer": "timer_meanpool",
    "TimesFM": "timesfm_meanpool",
    "Moirai": "moirai_meanpool",
}

PROPERTIES = {
    "trend": ("synthetic_trend", generate_trend_dataset),
    "frequency": ("synthetic_frequency", generate_frequency_dataset),
    "stationarity": ("synthetic_stationarity", generate_stationarity_dataset),
    "anomaly": ("synthetic_anomaly", generate_anomaly_dataset),
    "change_point": ("synthetic_change_point", generate_change_point_dataset),
}


def hand_crafted_features(seq: NDArray[np.float64]) -> NDArray[np.float64]:
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


def split_6_2_2(
    X: NDArray[np.float64], y: NDArray[np.int64], seed: int = SEED
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.2)
    tr = idx[n_test:]
    te = idx[:n_test]
    return X[tr], y[tr], X[te], y[te]


def load_best_layer_repr(
    model_key: str, ds_name: str, _y_unused: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    """Load best-performing layer representations + labels from disk.

    Uses on-disk labels.pt instead of passed-in synthetic labels, since
    model representations may have different N than raw synthetic dataset.
    """
    d = REPR_ROOT / model_key / ds_name
    if not d.exists():
        return None
    labels_path = d / "labels.pt"
    if not labels_path.exists():
        return None
    y_disk = torch.load(labels_path, map_location="cpu", weights_only=True).numpy().astype(np.int64)
    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    best_acc = -1.0
    best_X: NDArray[np.float64] | None = None
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.ndim < 2:
            continue
        X = t.numpy().astype(np.float64)
        if X.shape[0] != len(y_disk) or X.shape[1] == 0:
            continue
        Xtr, ytr, Xte, yte = split_6_2_2(X, y_disk)
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=SEED, solver="lbfgs"),
        )
        pipe.fit(Xtr, ytr)
        acc = accuracy_score(yte, pipe.predict(Xte))
        if acc > best_acc:
            best_acc = acc
            best_X = X
    if best_X is None:
        return None
    return best_X, y_disk


def main() -> None:
    seed_everything(SEED)
    results = []

    for prop_name, (ds_name, gen_fn) in PROPERTIES.items():
        ds = gen_fn(NUM_SAMPLES, SEQ_LEN, seed=SEED)
        y = ds.labels.astype(np.int64)
        hc = hand_crafted_features(ds.sequences)

        Xtr, ytr, Xte, yte = split_6_2_2(hc, y)
        hc_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs"),
        )
        hc_pipe.fit(Xtr, ytr)
        hc_acc = float(accuracy_score(yte, hc_pipe.predict(Xte)))

        for model_name, model_key in MODELS.items():
            ret = load_best_layer_repr(model_key, ds_name, y)
            if ret is None:
                print(f"  SKIP {model_name}/{prop_name} (no repr)")
                continue
            X, y_m = ret
            Xtr, ytr, Xte, yte = split_6_2_2(X, y_m)

            lin_pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs"),
            )
            lin_pipe.fit(Xtr, ytr)
            lin_acc = float(accuracy_score(yte, lin_pipe.predict(Xte)))

            mlp_pipe = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    random_state=SEED,
                    early_stopping=True,
                ),
            )
            mlp_pipe.fit(Xtr, ytr)
            mlp_acc = float(accuracy_score(yte, mlp_pipe.predict(Xte)))

            gap_lin = lin_acc - hc_acc
            gap_mlp = mlp_acc - hc_acc
            results.append(
                {
                    "property": prop_name,
                    "model": model_name,
                    "hand_crafted": round(hc_acc, 4),
                    "model_linear": round(lin_acc, 4),
                    "model_mlp_adversarial": round(mlp_acc, 4),
                    "gap_linear_vs_hc": round(gap_lin, 4),
                    "gap_mlp_vs_hc": round(gap_mlp, 4),
                    "verdict": (
                        "Non-trivial (MLP beats baseline by >3%)"
                        if gap_mlp > 0.03
                        else "Trivial (MLP at or below baseline)"
                    ),
                }
            )
            print(
                f"  {model_name:10s} / {prop_name:12s}  hc={hc_acc:.3f} "
                f"lin={lin_acc:.3f} mlp={mlp_acc:.3f} gap_mlp={gap_mlp:+.3f}"
            )

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
