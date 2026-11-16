"""UCR 2018 k-shot label efficiency with layer and C chosen by cross-validation inside the shot.

scripts/run_ucr_kshot.py fixes the layer (chosen on the synthetic realistic task) and C = 1. Here,
for k >= 2, each representation's layer and C are chosen by stratified cross-validation on the
k-shot training series alone (min(k, 5) folds, accuracy, ties to the fixed default), so no extra
labels are used; k = 1 keeps the defaults. Candidates: up to eight evenly spaced blocks of each
model, always including the default layer (Mantis-8M: all six blocks, default = last).
C in {0.01, 0.1, 1, 10}. The union (hand-crafted + catch22 + MiniRocket fit on the shot) gets the
same C search; union + model uses the layer chosen for the model alone and searches C again.
MiniRocket is fit once on the k-shot series (an unsupervised transform, not refit inside folds).
Datasets, k grid, draws, shots, and the official test split are those of run_ucr_kshot.py.

Parts: extract --model M (GPU) -> evaluate (CPU).
Arrays go to $UCR_KSHOT_CV_STORE (default <resolution store>/ucr_kshot_cv); results to
outputs/ucr_kshot/results_cv.json.
Usage:
    TSFMI_UCR2018=... TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_ucr_kshot_cv.py \
        --part extract --model mantis
    ... run_ucr_kshot_cv.py --part evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts import run_ucr_kshot as base

OUT = Path("outputs/ucr_kshot")
STORE = Path(
    os.environ.get(
        "UCR_KSHOT_CV_STORE",
        Path(os.environ.get("TSFMI_RESOLUTION_STORE", "outputs/resolution_store")) / "ucr_kshot_cv",
    )
)
MODELS = ("timesfm", "patchtst_fm", "moment", "chronos", "mantis")
CS = (0.01, 0.1, 1.0, 10.0)
MAX_LAYERS = 8


def default_layer(model: str) -> str:
    if model == "mantis":
        return "vit_unit_transformer_taps_5"
    from scripts.run_resolution_study import selected_layer

    return selected_layer(model)


def candidates(names: list[str], default: str) -> list[str]:
    keys = [n.replace(".", "_") for n in names]
    idx = np.unique(np.linspace(0, len(keys) - 1, min(MAX_LAYERS, len(keys))).round().astype(int))
    out = [keys[i] for i in idx]
    return out if default in out else sorted([*out, default], key=keys.index)


def cmd_extract(args: argparse.Namespace) -> None:
    import torch

    from scripts.extract_representations import load_model_wrapper
    from scripts.run_embedding_store import embed_all_layers, load_ucr, resize, znorm
    from src.utils.seed import seed_everything

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = load_model_wrapper(args.model, None, device)
    cand = candidates(wrapper.get_layer_names(), default_layer(args.model))
    for ds in base.datasets():
        out = STORE / args.model / ds
        if (out / "done").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        x_tr, _, x_te, _ = load_ucr(base.UCR, ds)
        x = resize(znorm(np.concatenate([x_tr, x_te]))).astype(np.float32)
        layers = embed_all_layers(wrapper, x)
        for ln, a in layers.items():
            if ln.replace(".", "_") in cand:
                np.save(out / f"{ln.replace('.', '_')}.npy", a.astype(np.float32))
        (out / "meta.json").write_text(
            json.dumps({"candidates": cand, "default": default_layer(args.model)})
        )
        (out / "done").write_text("ok")
        print("extracted", ds, args.model, flush=True)


def lr_acc(F_tr: NDArray, y_tr: NDArray, F_te: NDArray, y_te: NDArray, c: float) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    clf.fit(F_tr, y_tr)
    return float(np.mean(clf.predict(F_te) == y_te))


def cv_score(F: NDArray, y: NDArray, c: float, k: int, seed: int) -> float:
    from sklearn.model_selection import StratifiedKFold

    folds = StratifiedKFold(n_splits=min(k, 5), shuffle=True, random_state=seed)
    return float(np.mean([lr_acc(F[a], y[a], F[b], y[b], c) for a, b in folds.split(F, y)]))


def select(options: dict, y: NDArray, k: int, seed: int, default: tuple) -> tuple:
    """Pick the (option, C) with the best CV accuracy; ties (within 1e-9) keep the default."""
    best_key, best = default, cv_score(options[default[0]], y, default[1], k, seed)
    for name, F in options.items():
        for c in CS:
            if (name, c) == default:
                continue
            s = cv_score(F, y, c, k, seed)
            if s > best + 1e-9:
                best_key, best = (name, c), s
    return best_key


def eval_ds(ds: str) -> list[dict]:
    from threadpoolctl import threadpool_limits

    from scripts.run_canonical_baselines import hand_crafted_features
    from scripts.run_embedding_store import load_ucr, znorm
    from scripts.run_minirocket_controls import minirocket
    from scripts.run_strong_controls_ext import catch22

    x_tr, y_tr, x_te, y_te = load_ucr(base.UCR, ds)
    z = np.concatenate([znorm(x_tr), znorm(x_te)])
    n_tr = len(y_tr)
    hc = hand_crafted_features(z).astype(np.float32)
    c22 = catch22(z, 2)
    reps, defaults = {}, {}
    for m in MODELS:
        meta = json.loads((STORE / m / ds / "meta.json").read_text())
        reps[m] = {ln: np.load(STORE / m / ds / f"{ln}.npy") for ln in meta["candidates"]}
        defaults[m] = meta["default"]
    rows = []
    with threadpool_limits(limits=2):
        for k in base.KS:
            for d in range(base.DRAWS):
                rng = np.random.default_rng(1000 * k + d)  # same shots as T1c
                tr = np.concatenate(
                    [
                        rng.choice(np.flatnonzero(y_tr == c), size=k, replace=False)
                        for c in np.unique(y_tr)
                    ]
                )
                ytr = y_tr[tr]
                mr = minirocket(z[tr], z, 2)
                union = np.concatenate([hc, c22, mr], axis=1)
                row = {"dataset": ds, "k": k, "draw": d}
                if k == 1:
                    cu = 1.0
                else:
                    _, cu = select({"u": union[tr]}, ytr, k, d, ("u", 1.0))
                row["union"] = lr_acc(union[tr], ytr, union[n_tr:], y_te, cu)
                row["union_C"] = cu
                for m in MODELS:
                    if k == 1:
                        ln, c = defaults[m], 1.0
                    else:
                        ln, c = select(
                            {n: F[tr] for n, F in reps[m].items()}, ytr, k, d, (defaults[m], 1.0)
                        )
                    Z = reps[m][ln]
                    row[m] = lr_acc(Z[tr], ytr, Z[n_tr:], y_te, c)
                    row[f"{m}_layer"], row[f"{m}_C"] = ln, c
                    J = np.concatenate([union, Z], axis=1)
                    cj = 1.0 if k == 1 else select({"j": J[tr]}, ytr, k, d, ("j", 1.0))[1]
                    row[f"union+{m}"] = lr_acc(J[tr], ytr, J[n_tr:], y_te, cj)
                rows.append(row)
    print(ds, "done", flush=True)
    return rows


def cmd_evaluate(args: argparse.Namespace) -> None:
    from joblib import Parallel, delayed

    OUT.mkdir(parents=True, exist_ok=True)
    names = base.datasets()
    rows = Parallel(n_jobs=args.jobs)(delayed(eval_ds)(ds) for ds in names)
    (OUT / "results_cv.json").write_text(
        json.dumps({"datasets": names, "rows": [r for rs in rows for r in rs]}, indent=1)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("extract", "evaluate"), required=True)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    {"extract": cmd_extract, "evaluate": cmd_evaluate}[args.part](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
