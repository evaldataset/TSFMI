"""Anomaly mechanism analysis (S6).

Explains *why* hand-crafted features outperform every TSFM on anomaly probing.

Strategy:
1. Per-feature predictive power: probe each of the 8 HC features individually
   for anomaly. Identify which features carry the signal.
2. Cross-probe: probe TSFM representations specifically for those signal-
   carrying HC features (regression). If TSFM representations cannot recover
   the discriminating HC features, the anomaly inversion has a clear
   mechanistic explanation: TSFMs do not encode the specific statistics
   that distinguish anomalous windows.

Output: outputs/anomaly_mechanism/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.synthetic import generate_anomaly_dataset
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/anomaly_mechanism")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")

SEED = 42
NUM_SAMPLES = 1000
SEQ_LEN = 512
SEEDS = [0, 1, 2, 3, 4]

FEATURE_NAMES = [
    "slope",  # 0
    "residual_std",  # 1
    "mean",  # 2
    "std",  # 3
    "kurtosis",  # 4
    "argmax_fft",  # 5
    "spectral_entropy",  # 6
    "diff_std",  # 7
]

MODELS = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
    "Timer": "timer_meanpool",
    "TimesFM": "timesfm_meanpool",
    "Moirai": "moirai_meanpool",
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


def split(X: NDArray, y: NDArray, seed: int) -> tuple[NDArray, ...]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def cls_acc(X: NDArray, y: NDArray, seed: int) -> float:
    Xtr, ytr, _, _, Xte, yte = split(X, y, seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    )
    clf.fit(Xtr, ytr.astype(np.int64))
    return float(accuracy_score(yte.astype(np.int64), clf.predict(Xte)))


def reg_r2(X: NDArray, y: NDArray, seed: int) -> float:
    Xtr, ytr, _, _, Xte, yte = split(X, y, seed)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(Xtr, ytr)
    return float(r2_score(yte, reg.predict(Xte)))


def find_best_anomaly_layer(
    model_key: str,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Return (best_layer_repr, on_disk_labels, on_disk_sequences)."""
    d = REPR_ROOT / model_key / "synthetic_anomaly"
    if not d.exists():
        return None
    labels_path = d / "labels.pt"
    if not labels_path.exists():
        return None
    y_disk = (
        torch.load(labels_path, map_location="cpu", weights_only=True)
        .numpy()
        .astype(np.int64)
    )
    n_disk = len(y_disk)
    # Regenerate sequences with the same N as on-disk representations so that
    # HC features computed below match the model's input distribution.
    seq_disk = generate_anomaly_dataset(n_disk, SEQ_LEN, seed=SEED).sequences

    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    best_acc = -1.0
    best_X = None
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.ndim < 2 or t.shape[1] == 0 or t.shape[0] != n_disk:
            continue
        X = t.numpy().astype(np.float64)
        acc = cls_acc(X, y_disk, seed=0)
        if acc > best_acc:
            best_acc = acc
            best_X = X
    if best_X is None:
        return None
    return best_X, y_disk, seq_disk


def main() -> None:
    seed_everything(SEED)
    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)
    sequences = ds.sequences
    y = ds.labels.astype(np.int64)
    hc = hand_crafted_features(sequences)

    # Step 1: per-feature anomaly accuracy
    print("\n=== Step 1: Per-feature anomaly classification accuracy ===")
    per_feature = []
    for fi, fname in enumerate(FEATURE_NAMES):
        X = hc[:, fi : fi + 1]  # 1-D feature
        accs = [cls_acc(X, y, s) for s in SEEDS]
        per_feature.append(
            {
                "feature_idx": fi,
                "feature_name": fname,
                "anomaly_acc_mean": round(float(np.mean(accs)), 4),
                "anomaly_acc_std": round(float(np.std(accs)), 4),
            }
        )
        print(f"  feat[{fi}] {fname:20s} acc = {np.mean(accs):.4f}")

    # Identify top-3 discriminative features
    sorted_features = sorted(per_feature, key=lambda r: -r["anomaly_acc_mean"])
    top_k = sorted_features[:3]
    print("\nTop-3 anomaly-discriminative HC features:")
    for r in top_k:
        print(f"  feat[{r['feature_idx']}] {r['feature_name']}: {r['anomaly_acc_mean']:.3f}")

    # Step 2: can each TSFM regress the top-3 HC features from its own anomaly representations?
    print("\n=== Step 2: TSFM -> HC feature regression on anomaly data ===")
    print("If TSFMs cannot recover the discriminative HC features, that explains")
    print("the anomaly inversion mechanistically.\n")

    cross_results = []
    for model_name, model_key in MODELS.items():
        ret = find_best_anomaly_layer(model_key)
        if ret is None:
            print(f"  SKIP {model_name}: no representations")
            continue
        X, y_m, seq_m = ret
        # HC features computed on the same N as the model's representations
        hc_m = hand_crafted_features(seq_m)
        for r in top_k:
            fi = r["feature_idx"]
            fname = r["feature_name"]
            target = hc_m[:, fi]
            r2_scores = [reg_r2(X, target, s) for s in SEEDS]
            mean_r2 = float(np.mean(r2_scores))
            cross_results.append(
                {
                    "model": model_name,
                    "target_feature": fname,
                    "feature_idx": fi,
                    "regression_r2_mean": round(mean_r2, 4),
                    "regression_r2_std": round(float(np.std(r2_scores)), 4),
                    "anomaly_classification_acc": r["anomaly_acc_mean"],
                }
            )
            print(
                f"  {model_name:8s} -> {fname:20s} "
                f"R^2 = {mean_r2:.3f}  "
                f"(anomaly_acc this feature alone: {r['anomaly_acc_mean']:.3f})"
            )

    out = {"per_feature_anomaly_acc": per_feature, "tsfm_to_hc_regression": cross_results}
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
