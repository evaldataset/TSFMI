"""Multi-capacity adversarial probe sweep (S2).

Tests whether HIGHER probe capacity can rescue TSFMs on the saturated/inverted
properties. Probes tested per (model, property):
    - LogisticRegression (linear, baseline)
    - MLP with hidden_dim in {64, 128, 256, 512, 1024}
    - RandomForest (n_estimators=200, max_depth=20)
    - XGBoost (n_estimators=200, max_depth=8)

If NO probe at any capacity exceeds the hand-crafted baseline, the
linear-probe-artifact attack is fully neutralized.

Output: outputs/multicapacity_probe/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore[import-untyped]

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.datasets.synthetic import (
    generate_anomaly_dataset,
    generate_change_point_dataset,
    generate_frequency_dataset,
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/multicapacity_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")

SEED = 42
NUM_SAMPLES = 1000
SEQ_LEN = 512
SEEDS = [0, 1, 2]  # 3 seeds (capacity sweep is the variable, not seed)

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
    "anomaly": ("synthetic_anomaly", generate_anomaly_dataset),  # priority: inversion
    "trend": ("synthetic_trend", generate_trend_dataset),
    "frequency": ("synthetic_frequency", generate_frequency_dataset),
    "stationarity": ("synthetic_stationarity", generate_stationarity_dataset),
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


def split(X: NDArray, y: NDArray, seed: int) -> tuple[NDArray, ...]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]


def score_classifier(name: str, X: NDArray, y: NDArray, seeds: list[int]) -> dict:
    accs = []
    for seed in seeds:
        Xtr, ytr, Xte, yte = split(X, y.astype(np.int64), seed)
        if name == "linear":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
            )
        elif name.startswith("mlp_"):
            hidden = int(name.split("_")[1])
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(hidden,),
                    max_iter=300,
                    random_state=seed,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            )
        elif name == "mlp_2x512":
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(512, 256),
                    max_iter=300,
                    random_state=seed,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            )
        elif name == "rf":
            clf = make_pipeline(
                StandardScaler(),
                RandomForestClassifier(
                    n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1
                ),
            )
        elif name == "xgb":
            if not HAS_XGB:
                return {"name": name, "mean": float("nan"), "std": float("nan"), "skipped": True}
            clf = make_pipeline(
                StandardScaler(),
                XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=seed,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
            )
        else:
            raise ValueError(f"Unknown probe: {name}")
        clf.fit(Xtr, ytr)
        accs.append(float(accuracy_score(yte, clf.predict(Xte))))
    return {
        "name": name,
        "mean": round(float(np.mean(accs)), 4),
        "std": round(float(np.std(accs)), 4),
        "per_seed": [round(a, 4) for a in accs],
    }


def find_best_layer(
    model_key: str, ds_name: str, gen_fn, seed: int
) -> tuple[NDArray, NDArray] | None:
    """Return (best_layer_repr, on_disk_labels) using on-disk N, regenerating
    matching synthetic data so HC features computed downstream use same N.
    """
    d = REPR_ROOT / model_key / ds_name
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
    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    best_acc = -1.0
    best_X: NDArray | None = None
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.ndim < 2 or t.shape[1] == 0 or t.shape[0] != len(y_disk):
            continue
        X = t.numpy().astype(np.float64)
        Xtr, ytr, Xte, yte = split(X, y_disk, seed)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=seed, solver="lbfgs"),
        )
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        if acc > best_acc:
            best_acc = acc
            best_X = X
    if best_X is None:
        return None
    return best_X, y_disk


PROBES = ["linear", "mlp_64", "mlp_128", "mlp_256", "mlp_512", "mlp_1024", "rf"]
if HAS_XGB:
    PROBES.append("xgb")


def main() -> None:
    seed_everything(SEED)
    rows = []
    for prop_name, (ds_name, gen_fn) in PROPERTIES.items():
        # HC baseline at the canonical N=NUM_SAMPLES
        ds_baseline = gen_fn(NUM_SAMPLES, SEQ_LEN, seed=SEED)
        y_baseline = ds_baseline.labels.astype(np.int64)
        hc_baseline = hand_crafted_features(ds_baseline.sequences)
        hc_res = score_classifier("linear", hc_baseline, y_baseline, SEEDS)
        rows.append(
            {
                "property": prop_name,
                "model": "HAND_CRAFTED",
                "probe": "linear",
                "mean": hc_res["mean"],
                "std": hc_res["std"],
                "per_seed": hc_res["per_seed"],
            }
        )
        print(f"[{prop_name:12s}] HC linear: {hc_res['mean']:.3f}")

        for model_name, model_key in MODELS.items():
            ret = find_best_layer(model_key, ds_name, gen_fn, seed=0)
            if ret is None:
                print(f"  SKIP {model_name}/{prop_name}")
                continue
            X, y_m = ret
            for probe in PROBES:
                try:
                    r = score_classifier(probe, X, y_m, SEEDS)
                except Exception as e:
                    print(f"  ERR {model_name}/{prop_name}/{probe}: {e}")
                    r = {"name": probe, "mean": float("nan"), "std": float("nan"), "per_seed": []}
                rows.append(
                    {
                        "property": prop_name,
                        "model": model_name,
                        "probe": probe,
                        "mean": r["mean"],
                        "std": r["std"],
                        "per_seed": r.get("per_seed", []),
                    }
                )
                print(f"  {model_name:8s} {probe:10s} {r['mean']:.3f}")

    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
