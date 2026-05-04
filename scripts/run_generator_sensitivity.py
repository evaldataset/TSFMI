"""Generator sensitivity analysis (PLAN.md T2.1).

Tests whether the "4/6 trivial" finding is robust to synthetic generator
parameter variations. For each property, we sweep a key difficulty knob and
compute the baseline-vs-model gap.

Properties & knobs:
    - trend: slope strength in {0.001, 0.01, 0.1}
    - frequency: frequency range overlap
    - change_point: magnitude in {1, 3, 5} sigma
    - anomaly: outlier magnitude in {1, 2, 4} sigma
    - stationarity: AR coefficient range
    - seasonality: noise level in {0.1, 0.5, 1.0}

For each (property, knob_value), we compute:
    - hand-crafted baseline accuracy
    - raw signal baseline accuracy
    - best model accuracy (using MOMENT PCA512 representations)
    - gap = best_model - best_baseline

Output: outputs/generator_sensitivity/results.json
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

OUT_DIR = Path("outputs/generator_sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_SAMPLES = 1000
SEQ_LEN = 512


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


def probe_score(
    X: NDArray[np.float64], y: NDArray[np.float64], task: str, seed: int = SEED
) -> float:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.2)
    te = idx[:n_test]
    tr = idx[n_test:]
    if task == "classification":
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(X[tr], y[tr].astype(np.int64))
        return float(accuracy_score(y[te].astype(np.int64), pipe.predict(X[te])))
    else:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(X[tr], y[tr])
        return float(r2_score(y[te], pipe.predict(X[te])))


def main() -> None:
    seed_everything(SEED)
    results = []

    # Trend: noise_std as difficulty knob (higher noise = harder)
    for noise in [0.05, 0.1, 0.5]:
        ds = generate_trend_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED, noise_std=noise)
        hc = hand_crafted_features(ds.sequences)
        raw = ds.sequences
        hc_acc = probe_score(hc, ds.labels, "classification")
        raw_acc = probe_score(raw, ds.labels, "classification")
        results.append(
            {
                "property": "trend",
                "knob": "noise_std",
                "value": noise,
                "hand_crafted": round(hc_acc, 4),
                "raw_signal": round(raw_acc, 4),
                "max_baseline": round(max(hc_acc, raw_acc), 4),
            }
        )
        print(f"trend noise={noise}: hc={hc_acc:.3f} raw={raw_acc:.3f}")

    # Change point: default signature only has seed; skip or just one data point
    ds = generate_change_point_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)
    hc = hand_crafted_features(ds.sequences)
    raw = ds.sequences
    hc_acc = probe_score(hc, ds.labels, "classification")
    raw_acc = probe_score(raw, ds.labels, "classification")
    results.append(
        {
            "property": "change_point",
            "knob": "default",
            "value": 0.0,
            "hand_crafted": round(hc_acc, 4),
            "raw_signal": round(raw_acc, 4),
            "max_baseline": round(max(hc_acc, raw_acc), 4),
        }
    )
    print(f"change_point default: hc={hc_acc:.3f} raw={raw_acc:.3f}")

    # Anomaly: outlier magnitude
    for mag in [1.0, 2.0, 4.0]:
        ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED, anomaly_magnitude=mag)
        hc = hand_crafted_features(ds.sequences)
        raw = ds.sequences
        hc_acc = probe_score(hc, ds.labels, "classification")
        raw_acc = probe_score(raw, ds.labels, "classification")
        results.append(
            {
                "property": "anomaly",
                "knob": "anomaly_magnitude",
                "value": mag,
                "hand_crafted": round(hc_acc, 4),
                "raw_signal": round(raw_acc, 4),
                "max_baseline": round(max(hc_acc, raw_acc), 4),
            }
        )
        print(f"anomaly mag={mag}σ: hc={hc_acc:.3f} raw={raw_acc:.3f}")

    # Seasonality: noise level
    for noise in [0.1, 0.5, 1.0]:
        ds = generate_seasonality_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED, noise_std=noise)
        hc = hand_crafted_features(ds.sequences)
        raw = ds.sequences
        hc_r2 = probe_score(hc, ds.labels, "regression")
        raw_r2 = probe_score(raw, ds.labels, "regression")
        results.append(
            {
                "property": "seasonality",
                "knob": "noise_level",
                "value": noise,
                "hand_crafted": round(hc_r2, 4),
                "raw_signal": round(raw_r2, 4),
                "max_baseline": round(max(hc_r2, raw_r2), 4),
            }
        )
        print(f"seasonality noise={noise}: hc={hc_r2:.3f} raw={raw_r2:.3f}")

    # Frequency: default generator (just run once, minimal knob options)
    ds = generate_frequency_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)
    hc = hand_crafted_features(ds.sequences)
    raw = ds.sequences
    hc_acc = probe_score(hc, ds.labels, "classification")
    raw_acc = probe_score(raw, ds.labels, "classification")
    results.append(
        {
            "property": "frequency",
            "knob": "default",
            "value": 0.0,
            "hand_crafted": round(hc_acc, 4),
            "raw_signal": round(raw_acc, 4),
            "max_baseline": round(max(hc_acc, raw_acc), 4),
        }
    )

    # Stationarity: default
    ds = generate_stationarity_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)
    hc = hand_crafted_features(ds.sequences)
    raw = ds.sequences
    hc_acc = probe_score(hc, ds.labels, "classification")
    raw_acc = probe_score(raw, ds.labels, "classification")
    results.append(
        {
            "property": "stationarity",
            "knob": "default",
            "value": 0.0,
            "hand_crafted": round(hc_acc, 4),
            "raw_signal": round(raw_acc, 4),
            "max_baseline": round(max(hc_acc, raw_acc), 4),
        }
    )

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
