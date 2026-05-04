"""Seed-bootstrap cross-property LEACE for MOMENT and GPT4TS (S7).

Replaces the single-run cross-property entanglement matrix with 5-seed
bootstrap CIs on the (erase A, evaluate B) accuracy drop. Uses the same
canonical 60/20/20 protocol as the main paper.

Models: MOMENT-PCA512, GPT4TS-PCA512 (the two models in main-text
        entanglement tables)
Concepts: trend, frequency, stationarity, change_point
Off-diagonal pairs only (4*3 = 12 pairs per model)

Output:
    outputs/cross_leace_bootstrap/results.json
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
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "concept-erasure required; pip install concept-erasure"
    ) from err

from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/cross_leace_bootstrap")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")

MODELS = {
    "MOMENT": "moment_pca512",
    "GPT4TS": "gpt4ts_pca512",
}
CONCEPTS = ["trend", "frequency", "stationarity", "change_point"]
SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000


def three_way_split(X: NDArray, y: NDArray, seed: int) -> tuple[NDArray, ...]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te, va, tr = (
        idx[:n_test],
        idx[n_test : n_test + n_val],
        idx[n_test + n_val :],
    )
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def find_best_layer_for(
    model_key: str, concept: str, seed: int = 0
) -> tuple[NDArray, NDArray] | None:
    d = REPR_ROOT / model_key / f"synthetic_{concept}"
    if not d.exists():
        return None
    labels_path = d / "labels.pt"
    if not labels_path.exists():
        return None
    y = (
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
        if t.ndim != 2 or t.shape[1] == 0 or t.shape[0] != len(y):
            continue
        X = t.numpy().astype(np.float64)
        Xtr, ytr, Xva, yva, *_ = three_way_split(X, y, seed)
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(Xtr, ytr)
        acc = accuracy_score(yva, pipe.predict(Xva))
        if acc > best_acc:
            best_acc = acc
            best_X = X
    if best_X is None:
        return None
    return best_X, y


def fit_leace(Xtr: NDArray, ytr: NDArray) -> object:
    n_classes = int(np.unique(ytr).size)
    Yh = np.zeros((len(ytr), n_classes), dtype=np.float32)
    Yh[np.arange(len(ytr)), ytr] = 1.0
    fitter = LeaceFitter(x_dim=Xtr.shape[1], z_dim=n_classes, dtype=torch.float32)
    fitter.update(x=torch.tensor(Xtr, dtype=torch.float32), z=torch.tensor(Yh))
    return fitter.eraser


def cross_drop(
    X_a: NDArray, y_a: NDArray, X_b: NDArray, y_b: NDArray, seed: int
) -> tuple[float, float]:
    """Erase concept A on X_a, evaluate B classifier on X_b before/after.

    X_a and X_b are different concept datasets but should share N (the same
    underlying representations from a model trained on the same protocol). If
    N differs, we trim to common length so the indices align deterministically.
    """
    n = min(len(X_a), len(X_b))
    X_a, y_a, X_b, y_b = X_a[:n], y_a[:n], X_b[:n], y_b[:n]
    Xa_tr, ya_tr, _, _, _, _ = three_way_split(X_a, y_a, seed)
    Xb_tr, yb_tr, _, _, Xb_te, yb_te = three_way_split(X_b, y_b, seed)
    # Pre-erasure: train B classifier on B representations
    pipe_pre = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, random_state=seed, solver="lbfgs"),
    )
    pipe_pre.fit(Xb_tr, yb_tr)
    pre = float(accuracy_score(yb_te, pipe_pre.predict(Xb_te)))
    # Fit eraser on concept-A training split
    eraser = fit_leace(Xa_tr, ya_tr)

    def er(a: NDArray) -> NDArray:
        return eraser(torch.tensor(a, dtype=torch.float32)).numpy()

    Xb_tr_e = er(Xb_tr)
    Xb_te_e = er(Xb_te)
    pipe_post = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, random_state=seed, solver="lbfgs"),
    )
    pipe_post.fit(Xb_tr_e, yb_tr)
    post = float(accuracy_score(yb_te, pipe_post.predict(Xb_te_e)))
    return pre, post


def bootstrap_ci(values: list[float]) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return (
        float(arr.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def main() -> None:
    seed_everything(42)
    rows = []
    for model_name, model_key in MODELS.items():
        # Pre-load best layers per concept
        layers: dict[str, tuple[NDArray, NDArray]] = {}
        for c in CONCEPTS:
            ret = find_best_layer_for(model_key, c, seed=0)
            if ret is not None:
                layers[c] = ret
                print(f"  [{model_name}] {c}: layer shape={ret[0].shape}")
            else:
                print(f"  [{model_name}] {c}: SKIP")

        for erase_concept in CONCEPTS:
            if erase_concept not in layers:
                continue
            X_a, y_a = layers[erase_concept]
            for eval_concept in CONCEPTS:
                if eval_concept == erase_concept or eval_concept not in layers:
                    continue
                X_b, y_b = layers[eval_concept]
                pre_list: list[float] = []
                post_list: list[float] = []
                drop_list: list[float] = []
                for seed in SEEDS:
                    pre, post = cross_drop(X_a, y_a, X_b, y_b, seed)
                    pre_list.append(pre)
                    post_list.append(post)
                    drop_list.append(pre - post)
                pre_m, _, _ = bootstrap_ci(pre_list)
                post_m, _, _ = bootstrap_ci(post_list)
                drop_m, drop_lo, drop_hi = bootstrap_ci(drop_list)
                rows.append(
                    {
                        "model": model_name,
                        "erase": erase_concept,
                        "eval": eval_concept,
                        "pre_mean": round(pre_m, 4),
                        "post_mean": round(post_m, 4),
                        "drop_mean": round(drop_m, 4),
                        "drop_ci95_low": round(drop_lo, 4),
                        "drop_ci95_high": round(drop_hi, 4),
                        "pre_per_seed": [round(v, 4) for v in pre_list],
                        "post_per_seed": [round(v, 4) for v in post_list],
                    }
                )
                print(
                    f"  {model_name:8s} erase={erase_concept:13s} -> eval={eval_concept:13s} "
                    f"drop={drop_m:.3f} [{drop_lo:.3f},{drop_hi:.3f}]"
                )

    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {OUT_DIR}/results.json ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
