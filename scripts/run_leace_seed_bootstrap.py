"""Seed-bootstrap LEACE for the confirmatory 4 models x 2 properties (PLAN.md I9).

Addresses the "single-run LEACE" rigor concern by running LEACE with multiple
seeds on a canonical subset. Uses the canonical pipeline (train-only fit via
three-way split) and reports bootstrap 95% CI on the per-seed accuracy drop.

Subset:
    Models: MOMENT-PCA512, Chronos, PatchTST, GPT4TS (the encoder-side set)
    Properties: stationarity, change_point (the two where LEACE is most informative)

Output:
    outputs/leace_bootstrap/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from concept_erasure import LeaceFitter
except ImportError as err:  # pragma: no cover - fail fast when dep missing
    raise ImportError(
        "concept-erasure package required; pip install concept-erasure"
    ) from err

from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/leace_bootstrap")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")

MODELS = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
}
PROPERTIES = ["stationarity", "change_point"]
SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000


def three_way_split(X: NDArray, y: NDArray, seed: int) -> tuple[NDArray, ...]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def find_best_layer(model_key: str, ds_name: str, y: NDArray, seed: int) -> Path | None:
    d = REPR_ROOT / model_key / ds_name
    if not d.exists():
        return None
    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    best_acc = -1.0
    best_f = None
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.ndim != 2 or t.shape[1] == 0:
            continue
        X = t.numpy().astype(np.float64)
        if X.shape[0] != len(y):
            continue
        Xtr, ytr, Xva, yva, *_ = three_way_split(X, y, seed)
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(Xtr, ytr.astype(np.int64))
        acc = accuracy_score(yva.astype(np.int64), pipe.predict(Xva))
        if acc > best_acc:
            best_acc = acc
            best_f = lf
    return best_f


def leace_drop(X: NDArray, y: NDArray, seed: int) -> tuple[float, float]:
    """Return (pre_drop_acc, post_drop_acc) on held-out test."""
    y_int = y.astype(np.int64)
    Xtr, ytr, Xva, yva, Xte, yte = three_way_split(X, y_int, seed)
    # Pre-drop accuracy
    pipe_pre = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs"),
    )
    pipe_pre.fit(Xtr, ytr)
    pre = float(accuracy_score(yte, pipe_pre.predict(Xte)))

    # Fit LEACE on training split only
    n_classes = int(np.unique(ytr).size)
    Y_onehot = np.zeros((len(ytr), n_classes), dtype=np.float32)
    Y_onehot[np.arange(len(ytr)), ytr] = 1.0
    fitter = LeaceFitter(x_dim=X.shape[1], z_dim=n_classes, dtype=torch.float32)
    fitter.update(x=torch.tensor(Xtr, dtype=torch.float32), z=torch.tensor(Y_onehot))
    eraser = fitter.eraser

    def erase(a: NDArray) -> NDArray:
        return eraser(torch.tensor(a, dtype=torch.float32)).numpy()

    Xtr_e = erase(Xtr)
    Xte_e = erase(Xte)
    pipe_post = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs"),
    )
    pipe_post.fit(Xtr_e, ytr)
    post = float(accuracy_score(yte, pipe_post.predict(Xte_e)))
    return pre, post


def bootstrap_ci(values: list[float], n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    seed_everything(42)
    all_rows = []

    for model_name, model_key in MODELS.items():
        for prop in PROPERTIES:
            ds_name = f"synthetic_{prop}"
            labels_path = REPR_ROOT / model_key / ds_name / "labels.pt"
            if not labels_path.exists():
                print(f"  SKIP {model_name}/{prop}: no labels")
                continue
            y_all = (
                torch.load(labels_path, map_location="cpu", weights_only=True)
                .numpy()
                .astype(np.int64)
            )
            # Find best layer on seed=0 (fixed, to avoid layer-selection leakage across seeds)
            best_lf = find_best_layer(model_key, ds_name, y_all, seed=0)
            if best_lf is None:
                print(f"  SKIP {model_name}/{prop}: no suitable layer")
                continue
            t = torch.load(best_lf, map_location="cpu", weights_only=True)
            if t.ndim > 2:
                t = t.reshape(t.shape[0], -1)
            X = t.numpy().astype(np.float64)

            pre_list: list[float] = []
            post_list: list[float] = []
            drop_list: list[float] = []
            for seed in SEEDS:
                pre, post = leace_drop(X, y_all, seed)
                pre_list.append(pre)
                post_list.append(post)
                drop_list.append(pre - post)

            pre_m, pre_lo, pre_hi = bootstrap_ci(pre_list)
            post_m, post_lo, post_hi = bootstrap_ci(post_list)
            drop_m, drop_lo, drop_hi = bootstrap_ci(drop_list)

            row = {
                "model": model_name,
                "property": prop,
                "layer": best_lf.stem,
                "seeds": SEEDS,
                "pre_per_seed": [round(v, 4) for v in pre_list],
                "post_per_seed": [round(v, 4) for v in post_list],
                "drop_per_seed": [round(v, 4) for v in drop_list],
                "drop_mean": round(drop_m, 4),
                "drop_ci95_low": round(drop_lo, 4),
                "drop_ci95_high": round(drop_hi, 4),
                "pre_mean": round(pre_m, 4),
                "post_mean": round(post_m, 4),
            }
            all_rows.append(row)
            print(
                f"  {model_name:10s} / {prop:15s}  pre={pre_m:.3f} post={post_m:.3f} "
                f"drop={drop_m:.3f} [{drop_lo:.3f}, {drop_hi:.3f}]"
            )

    (OUT_DIR / "results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nSaved to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
