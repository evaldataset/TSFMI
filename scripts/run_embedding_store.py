"""Do stored TSFM embeddings beat raw-series distances? A 1-NN benchmark on the UCR 2018 archive.

A data system that stores an embedding per series instead of the series itself is betting that
nearest neighbours in embedding space are better neighbours. This script tests that bet the way
Paparrizos et al. (SIGMOD 2020) compared distance measures: 1-NN classification accuracy on every
fixed-length dataset of the UCR 2018 archive, official train/test splits, with

- ED: Euclidean distance on the z-normalised series at native length;
- DTW: dynamic time warping with a 10% Sakoe-Chiba window, native length;
- ROCKET-style features (the study's non-model control), standardised, Euclidean;
- each TSFM's mean-pooled layer embedding, cosine distance. Series are z-normalised and linearly
  resized to 512 steps (as in the UCR probes); the layer is chosen by leave-one-out 1-NN accuracy
  on the training split only.

A linear read-out (StandardScaler + logistic regression on the same layer) is reported as well;
storage per series follows from each representation's dimension. ``--init random`` repeats the
model part with every weight re-initialised (models_random.json), the architecture-only control.

Usage:
    PYTHONPATH=. python scripts/run_embedding_store.py --ucr-root /path/to/UCRArchive_2018 \
        --out outputs/embedding_store
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MODELS = ("timesfm", "moment", "patchtst_fm", "gpt4ts", "chronos", "moirai", "timer")
CTX = 512
MODEL_SEED = 42


def load_ucr(root: Path, name: str) -> tuple[NDArray, NDArray, NDArray, NDArray] | None:
    """Official split of one UCR 2018 dataset; None for variable-length (NaN-padded) datasets."""
    if not (root / name / f"{name}_TRAIN.tsv").exists():  # e.g. the archive's adjusted-data folder
        return None
    tr = np.loadtxt(root / name / f"{name}_TRAIN.tsv", delimiter="\t")
    te = np.loadtxt(root / name / f"{name}_TEST.tsv", delimiter="\t")
    if np.isnan(tr).any() or np.isnan(te).any():
        return None
    classes = {c: i for i, c in enumerate(sorted(set(tr[:, 0]) | set(te[:, 0])))}
    y_tr = np.array([classes[c] for c in tr[:, 0]])
    y_te = np.array([classes[c] for c in te[:, 0]])
    return tr[:, 1:], y_tr, te[:, 1:], y_te


def znorm(x: NDArray) -> NDArray[np.float64]:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(sd > 1e-8, sd, 1.0)


def resize(x: NDArray, length: int = CTX) -> NDArray[np.float64]:
    grid = np.linspace(0.0, 1.0, length)
    src = np.linspace(0.0, 1.0, x.shape[1])
    return np.stack([np.interp(grid, src, row) for row in x])


def nn_accuracy(d_te_tr: NDArray, y_tr: NDArray, y_te: NDArray) -> float:
    return float(np.mean(y_tr[np.argmin(d_te_tr, axis=1)] == y_te))


def loo_accuracy(d_tr_tr: NDArray, y_tr: NDArray) -> float:
    d = d_tr_tr.copy()
    np.fill_diagonal(d, np.inf)
    return float(np.mean(y_tr[np.argmin(d, axis=1)] == y_tr))


def euclidean(a: NDArray, b: NDArray) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(d, 0.0))


def cosine(a: NDArray, b: NDArray) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 1.0 - a @ b.T


def linear_accuracy(F_tr: NDArray, y_tr: NDArray, F_te: NDArray, y_te: NDArray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_tr)) < 2:
        return float(np.mean(y_te == y_tr[0]))
    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=0, solver="lbfgs")
    )
    pipe.fit(F_tr, y_tr)
    return float(np.mean(pipe.predict(F_te) == y_te))


def select_and_score(
    layers: dict[str, NDArray], y_tr: NDArray, y_te: NDArray, n_tr: int
) -> dict[str, object]:
    """1-NN and linear scores of a layered embedding; the layer is chosen by train LOO 1-NN."""
    best_loo, best_layer = -1.0, None
    for ln, F in layers.items():
        acc = loo_accuracy(cosine(F[:n_tr], F[:n_tr]), y_tr)
        if acc > best_loo:
            best_loo, best_layer = acc, ln
    F = layers[best_layer]
    return {
        "nn_acc": nn_accuracy(cosine(F[n_tr:], F[:n_tr]), y_tr, y_te),
        "nn_layer": best_layer,
        "nn_loo_train": best_loo,
        "linear_acc": linear_accuracy(F[:n_tr], y_tr, F[n_tr:], y_te),
        "dim": int(F.shape[1]),
    }


def embed_all_layers(wrapper, x: NDArray, batch_size: int = 64) -> dict[str, NDArray]:
    import torch

    from src.extractors.hook_manager import HookManager

    names = wrapper.get_layer_names()
    out: dict[str, list[NDArray]] = {n: [] for n in names}
    device = next(wrapper.model.parameters()).device
    with HookManager(wrapper.model, names) as hm, torch.inference_mode():
        for s in range(0, len(x), batch_size):
            wrapper.forward(torch.tensor(x[s : s + batch_size], dtype=torch.float32, device=device))
            for ln, a in hm.get_activations().items():
                a = a.float()
                a = a.reshape(a.shape[0], -1, a.shape[-1]) if a.ndim != 2 else a[:, None, :]
                out[ln].append(a.mean(dim=1).cpu().numpy().astype(np.float32))
            hm.clear()
    return {ln: np.concatenate(v) for ln, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ucr-root", type=Path, default=Path(os.environ.get("TSFMI_UCR2018", "")))
    ap.add_argument("--out", type=Path, default=Path("outputs/embedding_store"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--max-series", type=int, default=30_000)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--part", choices=("classical", "models"), required=True)
    ap.add_argument("--init", choices=("pretrained", "random"), default="pretrained")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.datasets or sorted(p.name for p in args.ucr_root.iterdir() if p.is_dir())
    suffix = "_random" if args.part == "models" and args.init == "random" else ""
    path = args.out / f"{args.part}{suffix}.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    if args.part == "models":
        import torch

        from scripts.extract_representations import load_model_wrapper
        from src.utils.seed import seed_everything

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in args.models:
            seed_everything(MODEL_SEED)
            wrapper = load_model_wrapper(model, None, device)
            if args.init == "random":
                from src.models.random_init import reinitialize_

                reinitialize_(wrapper, seed=MODEL_SEED)
            for name in names:
                if model in results.get(name, {}):
                    continue
                d = load_ucr(args.ucr_root, name)
                if d is None or len(d[1]) + len(d[3]) > args.max_series:
                    continue
                x_tr, y_tr, x_te, y_te = d
                t0 = time.time()
                x = resize(znorm(np.concatenate([x_tr, x_te])))
                layers = embed_all_layers(wrapper, x)
                t_embed = time.time() - t0
                row = select_and_score(layers, y_tr, y_te, len(y_tr))
                row["embed_seconds"] = round(t_embed, 2)
                results.setdefault(name, {})[model] = row
                print(f"[{model} {name}] nn={row['nn_acc']:.3f} lin={row['linear_acc']:.3f}")
                path.write_text(json.dumps(results, indent=1))
            del wrapper
            torch.cuda.empty_cache()
        return 0

    from aeon.distances import dtw_pairwise_distance

    from scripts.run_resolution_study import rocket_parallel

    for name in names:
        if name in results:
            continue
        d = load_ucr(args.ucr_root, name)
        if d is None or len(d[1]) + len(d[3]) > args.max_series:
            continue
        x_tr, y_tr, x_te, y_te = (np.asarray(v) for v in d)
        z_tr, z_te = znorm(x_tr), znorm(x_te)
        row: dict[str, object] = {
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "length": int(x_tr.shape[1]),
            "n_classes": int(len(np.unique(np.concatenate([y_tr, y_te])))),
        }
        t0 = time.time()
        row["ed_nn_acc"] = nn_accuracy(euclidean(z_te, z_tr), y_tr, y_te)
        row["ed_seconds"] = round(time.time() - t0, 3)
        t0 = time.time()
        d_dtw = dtw_pairwise_distance(z_te, z_tr, window=0.1, n_jobs=args.jobs)
        row["dtw_nn_acc"] = nn_accuracy(d_dtw, y_tr, y_te)
        row["dtw_seconds"] = round(time.time() - t0, 3)
        row["raw_linear_acc"] = linear_accuracy(z_tr, y_tr, z_te, y_te)
        r = rocket_parallel(resize(znorm(np.concatenate([x_tr, x_te]))), args.jobs)
        mu, sd = r[: len(y_tr)].mean(0), r[: len(y_tr)].std(0) + 1e-8
        rs = (r - mu) / sd
        row["rocket_nn_acc"] = nn_accuracy(euclidean(rs[len(y_tr) :], rs[: len(y_tr)]), y_tr, y_te)
        row["rocket_linear_acc"] = linear_accuracy(r[: len(y_tr)], y_tr, r[len(y_tr) :], y_te)
        results[name] = row
        print(
            f"[{name}] ed={row['ed_nn_acc']:.3f} dtw={row['dtw_nn_acc']:.3f} "
            f"rocket_nn={row['rocket_nn_acc']:.3f} rocket_lin={row['rocket_linear_acc']:.3f}",
            flush=True,
        )
        path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
