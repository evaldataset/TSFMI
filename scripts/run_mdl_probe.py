"""Voita-Titov MDL probe on canonical representations (CHECK.md G5).

Implements online (prequential) MDL probing for a logistic-regression probe.
We split the training set into K=10 increasing-size chunks, refit the probe
after each chunk, and accumulate the negative log-likelihood of the next
chunk under the previous probe -- equivalent to a description length over
the test labels given the representation.

Lower MDL = the representation linearly encodes the property more compactly.
We report MDL alongside the standard linear probe accuracy on the canonical
test split, for the four confirmatory encoder-side models on
(anomaly, change_point, frequency).

Output: outputs/mdl_probe/results.json
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

from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/mdl_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPR_ROOT = Path("outputs/representations")

MODELS = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
}
PROPERTIES = ["anomaly", "change_point", "frequency"]
SEEDS = [0, 1, 2, 3, 4]
CHUNKS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]


def three_way_split(X, y, seed):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def best_layer_path(model_key: str, prop: str, y: NDArray, seed: int) -> Path | None:
    d = REPR_ROOT / model_key / f"synthetic_{prop}"
    if not d.exists():
        return None
    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    best_acc = -1.0
    best = None
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
            LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(Xtr, ytr.astype(np.int64))
        acc = accuracy_score(yva.astype(np.int64), pipe.predict(Xva))
        if acc > best_acc:
            best_acc = acc
            best = lf
    return best


def online_mdl(
    X_tr: NDArray, y_tr: NDArray, X_te: NDArray, y_te: NDArray, n_classes: int, seed: int
) -> tuple[float, float]:
    """Returns (online_codelength_bits, test_acc).

    Prequential / online MDL: process the training data in K increasing chunks,
    compute the predictive log-loss on each chunk under the model fit on
    the previous prefix. Accumulate to get total description length.
    """
    n = len(X_tr)
    # Uniform start prior: the first chunk has no prior model, so we charge
    # log2(n_classes) bits per label.
    code_bits = 0.0
    prev_end = 0
    pipe: object | None = None
    chunk_sizes = [int(round(c * n)) for c in CHUNKS]
    chunk_sizes = sorted(set([1, *chunk_sizes]))
    for end in chunk_sizes:
        if end > n:
            end = n
        if end <= prev_end:
            continue
        new_X = X_tr[prev_end:end]
        new_y = y_tr[prev_end:end].astype(np.int64)
        if pipe is None:
            # First chunk: uniform prior -> log2(n_classes) per sample.
            code_bits += float(len(new_y)) * np.log2(n_classes)
        else:
            log_proba = pipe.predict_log_proba(new_X)  # type: ignore[attr-defined]
            picks = log_proba[np.arange(len(new_y)), new_y]
            # Convert ln to log2.
            code_bits += float(-(picks / np.log(2)).sum())
        # Refit on prefix.
        prefix_X = X_tr[:end]
        prefix_y = y_tr[:end].astype(np.int64)
        if len(np.unique(prefix_y)) < 2:
            prev_end = end
            continue
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(prefix_X, prefix_y)
        prev_end = end

    # Test accuracy with full-train probe.
    acc = float(accuracy_score(y_te.astype(np.int64), pipe.predict(X_te))) if pipe else float("nan")
    return code_bits, acc


def main() -> None:
    seed_everything(42)
    rows = []
    for prop in PROPERTIES:
        for model_name, model_key in MODELS.items():
            d = REPR_ROOT / model_key / f"synthetic_{prop}"
            labels_path = d / "labels.pt"
            if not labels_path.exists():
                continue
            y_all = (
                torch.load(labels_path, map_location="cpu", weights_only=True)
                .numpy()
                .astype(np.int64)
            )
            n_classes = int(np.unique(y_all).size)
            best_lf = best_layer_path(model_key, prop, y_all, seed=0)
            if best_lf is None:
                continue
            t = torch.load(best_lf, map_location="cpu", weights_only=True)
            if t.ndim > 2:
                t = t.reshape(t.shape[0], -1)
            X = t.numpy().astype(np.float64)
            mdl_per_seed = []
            acc_per_seed = []
            for s in SEEDS:
                Xtr, ytr, _, _, Xte, yte = three_way_split(X, y_all, s)
                mdl, acc = online_mdl(Xtr, ytr, Xte, yte, n_classes, s)
                mdl_per_seed.append(mdl)
                acc_per_seed.append(acc)
            row = {
                "model": model_name,
                "property": prop,
                "layer": best_lf.stem,
                "n_classes": n_classes,
                "uniform_codelength_bits": float(len(X) * 0.6 * np.log2(n_classes)),
                "online_mdl_bits_per_seed": [round(v, 2) for v in mdl_per_seed],
                "online_mdl_bits_mean": round(float(np.mean(mdl_per_seed)), 2),
                "test_acc_per_seed": [round(v, 4) for v in acc_per_seed],
                "test_acc_mean": round(float(np.mean(acc_per_seed)), 4),
            }
            rows.append(row)
            print(
                f"  {model_name:10s} / {prop:14s}  "
                f"MDL={row['online_mdl_bits_mean']:.1f} bits "
                f"(uniform={row['uniform_codelength_bits']:.1f}); "
                f"acc={row['test_acc_mean']:.3f}"
            )

    OUT_DIR.joinpath("results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
