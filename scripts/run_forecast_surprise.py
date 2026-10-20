"""Unsupervised anomaly scores: TSFM forecast surprise versus classical detectors (H8).

A forecasting model is used as a detector by asking how surprised it is by what follows a context.
For every test window of the resolution study, forecasts with horizon h are issued at origins 32,
32 + h, 32 + 2h, ... (covering steps 32-511), h in {1, 8, 64}; h = 1 is the one-step forecast that
the AR(16) detector also uses. The window's score is the largest robustly standardised residual
|r_t - median(r)| / (1.4826 MAD(r)). Classical scores on the same windows:

- max |z|: largest absolute z-score of the window;
- first difference: the same standardised maximum on x_t - x_{t-1} (the naive forecast);
- AR(16): one-step residuals of a least-squares AR(16) fit to the window;
- matrix profile discord: the largest z-normalised nearest-neighbour distance of any subsequence
  (Yeh et al., 2016), subsequence length 16, 32 (primary), 64.

All scores are label-free; ROC-AUC is computed against the window label, overall and per anomaly
kind (each kind's anomalous windows against all normal windows), with paired bootstrap intervals for
differences. The supervised probes of the resolution study are listed for reference.

Usage:
    PYTHONPATH=. python scripts/run_forecast_surprise.py --dataset realistic \
        --out outputs/forecast_surprise
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from scripts.run_resolution_study import load_data, split_indices

FIRST_ORIGIN = 32
HORIZONS = (1, 8, 64)
N_BOOT = 1000


def robust_max(r: NDArray) -> NDArray[np.float64]:
    """Row-wise max |r - median| / (1.4826 MAD), NaNs ignored."""
    med = np.nanmedian(r, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(r - med), axis=1, keepdims=True)
    return np.nanmax(np.abs(r - med), axis=1) / np.maximum(mad[:, 0], 1e-8)


def forecast_residuals(x: NDArray, predict, horizon: int) -> NDArray[np.float64]:
    """Residuals x_t - forecast_t on steps FIRST_ORIGIN.., one forecast every `horizon` steps."""
    r = np.full(x.shape, np.nan)
    for o in range(FIRST_ORIGIN, x.shape[1], horizon):
        h = min(horizon, x.shape[1] - o)
        r[:, o : o + h] = x[:, o : o + h] - predict(x[:, :o], h)
    return r


def timesfm_predictor(device: str, batch: int = 512):
    import torch
    from transformers import TimesFmModelForPrediction

    model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch").to(device)
    model.eval()

    def predict(ctx: NDArray, h: int) -> NDArray:
        out = []
        for s in range(0, len(ctx), batch):
            past = torch.tensor(ctx[s : s + batch], dtype=torch.float32, device=device)
            with torch.no_grad():
                res = model(
                    past_values=list(past),
                    freq=[0] * len(past),
                    forecast_context_len=int(-(-ctx.shape[1] // 32) * 32),
                )
            out.append(res.mean_predictions[:, :h].float().cpu().numpy())
        return np.concatenate(out)

    return predict


def chronos_predictor(device: str, batch: int = 1024):
    import torch
    from chronos import ChronosBoltPipeline

    pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device)

    def predict(ctx: NDArray, h: int) -> NDArray:
        out = []
        for s in range(0, len(ctx), batch):
            q, _ = pipe.predict_quantiles(
                torch.tensor(ctx[s : s + batch], dtype=torch.float32),
                prediction_length=h,
                quantile_levels=[0.5],
            )
            out.append(q[..., 0].float().cpu().numpy())
        return np.concatenate(out)

    return predict


def ar_residuals(x: NDArray, p: int = 16) -> NDArray[np.float64]:
    r = np.full(x.shape, np.nan)
    for i, row in enumerate(x):
        lagged = np.stack([row[p - k - 1 : len(row) - k - 1] for k in range(p)], axis=1)
        design = np.column_stack([lagged, np.ones(len(lagged))])
        coef, *_ = np.linalg.lstsq(design, row[p:], rcond=None)
        r[i, p:] = row[p:] - design @ coef
    return r


def mp_discord(x: NDArray, m: int) -> NDArray[np.float64]:
    import stumpy

    return np.array([float(np.max(stumpy.stump(row.astype(np.float64), m)[:, 0])) for row in x])


def auc_table(scores: dict[str, NDArray], y: NDArray, kinds: NDArray | None) -> dict:
    out = {}
    for name, s in scores.items():
        row = {"auc": float(roc_auc_score(y, s))}
        if kinds is not None:
            for k in (0, 1, 2, 3):
                mask = (kinds == k) | (kinds == -1)
                row[f"auc_kind{k}"] = float(roc_auc_score(y[mask], s[mask]))
        out[name] = row
    return out


def paired_auc_ci(a: NDArray, b: NDArray, y: NDArray, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    return {
        "diff": d,
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="realistic")
    ap.add_argument("--out", type=Path, default=Path("outputs/forecast_surprise"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    d = load_data(args.dataset)
    te = split_indices(len(d["y"]))["test"]
    x, y = d["x"][te].astype(np.float64), d["y"][te].astype(np.int64)
    kinds = d["kinds"][te] if "kinds" in d else None
    scores: dict[str, NDArray] = {}
    seconds: dict[str, float] = {}

    from scripts.run_resolution_study import STORE

    cache = STORE / "forecast_surprise" / f"{args.dataset}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cached = dict(np.load(cache)) if cache.exists() else {}

    def timed(name: str, fn) -> None:
        t0 = time.time()
        scores[name] = cached[name] if name in cached else np.asarray(fn(), dtype=np.float64)
        seconds[name] = round(time.time() - t0, 1)
        np.savez(cache, **scores)
        print(f"{name}: AUC {roc_auc_score(y, scores[name]):.4f} ({seconds[name]}s)", flush=True)

    from joblib import Parallel, delayed

    timed(
        "max_abs_z", lambda: np.abs((x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)).max(1)
    )
    timed("first_difference", lambda: robust_max(np.diff(x, axis=1)))
    timed("ar16_residual", lambda: robust_max(ar_residuals(x)))
    for m in (16, 32, 64):
        timed(
            f"matrix_profile_m{m}",
            lambda m=m: np.concatenate(
                Parallel(n_jobs=args.jobs)(
                    delayed(mp_discord)(x[s : s + 250], m) for s in range(0, len(x), 250)
                )
            ),
        )
    predictors = {"timesfm": timesfm_predictor, "chronos_bolt": chronos_predictor}
    for model, predict in predictors.items():
        for h in HORIZONS:
            timed(
                f"{model}_surprise_h{h}",
                lambda p=predict, h=h: robust_max(forecast_residuals(x, p(args.device), h)),
            )

    # Supervised references from the resolution study (same test windows).
    supervised = {}
    for rep, label in (
        ("model__timesfm-pretrained", "probe_timesfm"),
        ("model__patchtst_fm-pretrained", "probe_patchtst_fm"),
        ("ctrl__rocket", "probe_rocket"),
        ("ctrl__hand_crafted", "probe_hand_crafted"),
    ):
        p = STORE / "probes" / args.dataset / f"{rep}.npz"
        if p.exists():
            with np.load(p) as z:
                for key in z.files:
                    if key.startswith("p_test_"):
                        n = key.removeprefix("p_test_")
                        supervised[f"{label}_n{n}"] = z[key].astype(np.float64)
    table = auc_table(scores, y, kinds)
    sup_table = auc_table(supervised, y, kinds)
    best_classical = max((k for k in scores if "_surprise" not in k), key=lambda k: table[k]["auc"])
    diffs = {
        f"{f}-{best_classical}": paired_auc_ci(scores[f], scores[best_classical], y)
        for f in scores
        if "_surprise" in f
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.dataset}.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "n_test": int(len(y)),
                "first_origin": FIRST_ORIGIN,
                "horizons": list(HORIZONS),
                "unsupervised": table,
                "supervised_reference": sup_table,
                "best_classical": best_classical,
                "paired_differences": diffs,
                "seconds": seconds,
            },
            indent=1,
        )
    )
    print(json.dumps(diffs, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
