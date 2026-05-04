"""Retroactive case study (PLAN.md T1.3).

Reproduces representative TSFM probing claims from prior work *without* baseline
control, then shows how the claim changes *with* baseline control.

We don't have access to the exact configurations of Wilinski'25, Pandey'25,
Kalnare'25, or Oublal'26. Instead, we simulate the "no-baseline" setup they
implicitly use by:

  1. Training a linear probe on model representations for each property
  2. Reporting the accuracy as if it demonstrated "model encodes property X"
  3. Then running the same probe on hand-crafted features / raw signal
  4. Computing the gap: if gap <= 0, the claim is unsupported by baseline control

Output: outputs/retroactive/case_study_results.json
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

from src.datasets.synthetic import (
    generate_anomaly_dataset,
    generate_change_point_dataset,
    generate_frequency_dataset,
    generate_seasonality_dataset,
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/retroactive")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")

SEED = 42

# Simulated claims from prior work (paraphrased, not direct quotes)
PRIOR_CLAIMS = [
    {
        "claim_id": "C1",
        "source": "Wilinski-style (MOMENT trend)",
        "claim": "MOMENT encodes trend direction linearly from early layers",
        "model": "moment_pca512",
        "dataset": "synthetic_trend",
        "property": "trend",
        "task_type": "classification",
        "generator": generate_trend_dataset,
    },
    {
        "claim_id": "C2",
        "source": "Pandey-style (linear recoverability)",
        "claim": "Frequency is linearly recoverable from pre-trained representations",
        "model": "chronos",
        "dataset": "synthetic_frequency",
        "property": "frequency",
        "task_type": "classification",
        "generator": generate_frequency_dataset,
    },
    {
        "claim_id": "C3",
        "source": "Kalnare-style (change point)",
        "claim": "Change-point detection is encoded in PatchTST",
        "model": "patchtst_pretrained",
        "dataset": "synthetic_change_point",
        "property": "change_point",
        "task_type": "classification",
        "generator": generate_change_point_dataset,
    },
    {
        "claim_id": "C4",
        "source": "Oublal-style (stationarity)",
        "claim": "Stationarity discriminable via LLM-adapted representations",
        "model": "gpt4ts_pca512",
        "dataset": "synthetic_stationarity",
        "property": "stationarity",
        "task_type": "classification",
        "generator": generate_stationarity_dataset,
    },
    {
        "claim_id": "C5",
        "source": "General TSFM probing (seasonality)",
        "claim": "TSFM encodes seasonality period as regression target",
        "model": "moment_pca512",
        "dataset": "synthetic_seasonality",
        "property": "seasonality",
        "task_type": "regression",
        "generator": generate_seasonality_dataset,
    },
    {
        "claim_id": "C6",
        "source": "General TSFM probing (anomaly)",
        "claim": "Anomaly flags are decodable from TimesFM",
        "model": "timesfm_meanpool",
        "dataset": "synthetic_anomaly",
        "property": "anomaly",
        "task_type": "classification",
        "generator": generate_anomaly_dataset,
    },
]


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


def probe(
    X: NDArray[np.float64], y: NDArray[np.float64], task: str, seed: int = SEED
) -> float:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    te = idx[: int(n * 0.2)]
    tr = idx[int(n * 0.2) :]
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


def load_model_best_repr(
    model_key: str, ds_name: str, y: NDArray
) -> NDArray[np.float64] | None:
    d = REPR_ROOT / model_key / ds_name
    if not d.exists():
        return None
    labels_path = d / "labels.pt"
    if not labels_path.exists():
        return None
    y_disk = torch.load(labels_path, map_location="cpu", weights_only=True).numpy()
    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    best_score = -np.inf
    best_X = None
    task = "regression" if y.dtype.kind == "f" else "classification"
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.ndim < 2:
            continue
        X = t.numpy().astype(np.float64)
        if X.shape[0] != len(y_disk) or X.shape[1] == 0:
            continue
        s = probe(X, y_disk, task)
        if s > best_score:
            best_score = s
            best_X = X
    return best_X


def main() -> None:
    seed_everything(SEED)
    results = []
    for c in PRIOR_CLAIMS:
        gen = c["generator"]
        ds = gen(1000, 512, seed=SEED)
        hc = hand_crafted_features(ds.sequences)
        raw = ds.sequences
        task = c["task_type"]

        hc_score = probe(hc, ds.labels, task)
        raw_score = probe(raw, ds.labels, task)
        baseline_best = max(hc_score, raw_score)

        X = load_model_best_repr(c["model"], c["dataset"], ds.labels)
        if X is None:
            model_score = float("nan")
        else:
            y_disk = torch.load(
                REPR_ROOT / c["model"] / c["dataset"] / "labels.pt",
                map_location="cpu",
                weights_only=True,
            ).numpy()
            model_score = probe(X, y_disk, task)

        gap = model_score - baseline_best if not np.isnan(model_score) else float("nan")
        verdict = (
            "Supported (gap > 3%)"
            if gap > 0.03
            else "Trivial under baseline control"
            if gap <= 0
            else "Marginal (0 < gap <= 3%)"
        )
        results.append(
            {
                "claim_id": c["claim_id"],
                "source": c["source"],
                "claim": c["claim"],
                "model": c["model"],
                "property": c["property"],
                "task_type": task,
                "original_model_score": round(model_score, 4),
                "hand_crafted_baseline": round(hc_score, 4),
                "raw_signal_baseline": round(raw_score, 4),
                "baseline_best": round(baseline_best, 4),
                "gap": round(gap, 4),
                "verdict": verdict,
            }
        )
        print(
            f"  {c['claim_id']} {c['source'][:30]:30s} "
            f"model={model_score:.3f} baseline_best={baseline_best:.3f} "
            f"gap={gap:+.3f} → {verdict}"
        )

    out_path = OUT_DIR / "case_study_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
