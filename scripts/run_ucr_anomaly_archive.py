"""Window-level anomaly probing on real data: the UCR Time Series Anomaly Archive (Wu & Keogh).

Each of the 250 series has an anomaly-free training prefix and one labelled anomaly in the rest.
From the part after the prefix we cut windows of length 512: positive windows contain at least half
of the anomaly (at most 8 per series, evenly spaced), negative windows do not overlap it (as many as
positives, sampled with a fixed seed). Splits are by series (60/20/20 of the series, five split
seeds), so no series contributes windows to both training and test.

Compared under the canonical protocol (StandardScaler + logistic regression, layer chosen on
validation per split; every window z-normalised): seven frozen TSFMs (mean-pooled layers), the raw
window, the hand-crafted
vector, ROCKET-style and tsfresh (efficient) features, their union ("all controls"), and the
conditional probe [all controls, Z] for every model, all regularisation tuned on validation for
the conditional comparison. Label-free scores (max |z|, first difference, AR(16) residual,
matrix-profile discord) are evaluated by ROC-AUC on the same test windows.

Usage:
    PYTHONPATH=. python scripts/run_ucr_anomaly_archive.py --archive <.../UCR_Anomaly_FullData> \
        --out outputs/ucr_anomaly_archive
    Add --transform reverse|negate (with a matching --out) for the pretraining-overlap check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

L = 512
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
MODELS = ("moment", "chronos", "gpt4ts", "timer", "timesfm", "moirai", "patchtst_fm")
NAME = re.compile(r"(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")
STORE = Path(os.environ.get("TSFMI_RESOLUTION_STORE", "outputs/resolution_store")) / "ucr_anomaly"
# Pretraining-overlap check: a series the model saw during pretraining is no longer the same series
# once reversed in time or negated, while most hand-crafted statistics are unchanged. Windows are
# already z-normalised, so both transforms keep them z-normalised.
TRANSFORMS = {
    "none": lambda x: x,
    "reverse": lambda x: np.ascontiguousarray(x[:, ::-1]),
    "negate": lambda x: -x,
}


def load_series(path: Path) -> NDArray[np.float64]:
    return np.array(path.read_text().split(), dtype=np.float64)


def build_windows(archive: Path) -> dict[str, NDArray]:
    rng = np.random.default_rng(0)
    xs, ys, groups = [], [], []
    for f in sorted(archive.glob("*.txt")):
        m = NAME.search(f.name)
        if not m:
            continue
        sid, train_end, a0, a1 = int(m.group(1)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        x = load_series(f)
        need = max(1, (min(a1 - a0 + 1, L) + 1) // 2)
        starts = np.arange(train_end, len(x) - L + 1)
        if len(starts) == 0:
            continue
        overlap = np.clip(np.minimum(starts + L, a1 + 1) - np.maximum(starts, a0), 0, None)
        pos = starts[overlap >= need]
        neg = starts[overlap == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos = pos[np.linspace(0, len(pos) - 1, min(MAX_POS, len(pos))).round().astype(int)]
        neg = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
        for s, lab in [(p, 1) for p in pos] + [(q, 0) for q in neg]:
            xs.append(x[s : s + L])
            ys.append(lab)
            groups.append(sid)
    # The 250 series come from different sources in arbitrary units; every representation (models
    # and controls alike) sees each window z-normalised, as in the UCR 1-NN benchmark.
    xw = np.asarray(xs)
    xw = (xw - xw.mean(1, keepdims=True)) / np.maximum(xw.std(1, keepdims=True), 1e-8)
    return {"x": xw, "y": np.asarray(ys), "group": np.asarray(groups)}


def group_split(groups: NDArray, seed: int) -> dict[str, NDArray]:
    ids = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(ids)
    n_te = n_va = len(ids) // 5
    te, va = set(perm[:n_te]), set(perm[n_te : n_te + n_va])
    idx = np.arange(len(groups))
    is_te = np.isin(groups, list(te))
    is_va = np.isin(groups, list(va))
    return {"test": idx[is_te], "val": idx[is_va], "train": idx[~is_te & ~is_va]}


def extract(model: str, x: NDArray, out: Path) -> None:
    import torch

    from scripts.extract_representations import load_model_wrapper
    from scripts.run_embedding_store import embed_all_layers
    from src.utils.seed import seed_everything

    if (out / "done").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = load_model_wrapper(model, None, device)
    layers = embed_all_layers(wrapper, x.astype(np.float32))
    for ln, a in layers.items():
        np.save(out / f"{ln.replace('.', '_')}.npy", a.astype(np.float32))  # float16 overflows
    json.dump(
        {"layer_order": [ln.replace(".", "_") for ln in wrapper.get_layer_names()]},
        open(out / "meta.json", "w"),
    )
    (out / "done").write_text("ok")


def fit_lr(Xtr: NDArray, ytr: NDArray, Xs: list[NDArray], c: float = 1.0) -> list[NDArray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=0, solver="lbfgs")
    )
    pipe.fit(Xtr, ytr)
    return [pipe.predict_proba(X)[:, 1] for X in Xs]


def tuned(X: NDArray, y: NDArray, sp: dict) -> NDArray:
    from sklearn.metrics import log_loss

    best = (np.inf, None)
    for c in (0.001, 0.01, 0.1, 1.0):
        p_va, p_te = fit_lr(X[sp["train"]], y[sp["train"]], [X[sp["val"]], X[sp["test"]]], c)
        loss = log_loss(y[sp["val"]], p_va, labels=[0, 1])
        if loss < best[0]:
            best = (loss, p_te)
    return best[1]


def main() -> int:
    from sklearn.metrics import roc_auc_score

    from scripts.run_canonical_baselines import hand_crafted_features
    from scripts.run_forecast_surprise import ar_residuals, mp_discord, robust_max
    from scripts.run_resolution_study import rocket_parallel, tsfresh_features
    from src.metrics.paired_tests import paired_logloss_difference

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path, help="archive root; needed only to build windows")
    ap.add_argument("--out", type=Path, default=Path("outputs/ucr_anomaly_archive"))
    ap.add_argument("--part", choices=("data", "extract", "evaluate"), required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--transform", choices=TRANSFORMS, default="none")
    args = ap.parse_args()
    store = STORE if args.transform == "none" else STORE.parent / f"ucr_anomaly_{args.transform}"
    store.mkdir(parents=True, exist_ok=True)
    data_path = store / "windows.npz"

    if args.part == "data":
        if args.transform == "none":
            if args.archive is None:
                ap.error("--part data needs --archive")
            d = build_windows(args.archive)
        else:
            with np.load(STORE / "windows.npz") as z:
                d = {k: z[k] for k in z.files}
            d["x"] = TRANSFORMS[args.transform](d["x"])
        np.savez(data_path, **d)
        x = d["x"]
        np.save(store / "hand_crafted.npy", hand_crafted_features(x).astype(np.float32))
        np.save(store / "rocket.npy", rocket_parallel(x, args.jobs))
        np.save(store / "tsfresh_efficient.npy", tsfresh_features(x, args.jobs))
        print(
            f"{len(d['y'])} windows from {len(np.unique(d['group']))} series, "
            f"positive rate {d['y'].mean():.3f}",
            flush=True,
        )
        return 0

    with np.load(data_path) as z:
        x, y, groups = z["x"], z["y"].astype(np.int64), z["group"]
    if args.part == "extract":
        for m in [args.model] if args.model else MODELS:
            extract(m, x, store / "repr" / m)
            print(f"extracted {m}", flush=True)
        return 0

    ctrl = {k: np.load(store / f"{k}.npy") for k in ("hand_crafted", "rocket", "tsfresh_efficient")}
    ctrl["raw_signal"] = x.astype(np.float32)
    C_all = np.concatenate([ctrl["hand_crafted"], ctrl["rocket"], ctrl["tsfresh_efficient"]], 1)
    res: dict = {
        "n_windows": int(len(y)),
        "n_series": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "per_seed": {},
    }
    from scripts.run_minirocket_controls import minirocket
    from scripts.run_resolution_study import compare

    for seed in SEEDS:
        sp = group_split(groups, seed)
        y_te = y[sp["test"]]
        row: dict = {"n_test": int(len(y_te))}
        # MiniRocket is fit on this split's training windows and joins the control union.
        mr = minirocket(x[sp["train"]], x, args.jobs)
        p_mr = tuned(mr, y, sp)
        row["ctrl:minirocket"] = {"acc": float(np.mean((p_mr > 0.5) == y_te))}
        C_plus = np.concatenate([C_all, mr], 1)
        p_cplus = tuned(C_plus, y, sp)
        row["ctrl:all_plus_minirocket"] = {"acc": float(np.mean((p_cplus > 0.5) == y_te))}

        def acc(p: NDArray, yt: NDArray = y_te) -> float:
            return float(np.mean((p > 0.5) == yt))

        for k, F in ctrl.items():
            (p,) = fit_lr(F[sp["train"]], y[sp["train"]], [F[sp["test"]]])
            row[f"ctrl:{k}"] = {"acc": acc(p), "auc": float(roc_auc_score(y_te, p))}
        p_c = tuned(C_all, y, sp)
        row["ctrl:all_controls"] = {"acc": acc(p_c), "auc": float(roc_auc_score(y_te, p_c))}
        for m in MODELS:
            rdir = store / "repr" / m
            order = json.loads((rdir / "meta.json").read_text())["layer_order"]
            best = (-1.0, None, None)
            for ln in order:
                Z = np.load(rdir / f"{ln}.npy").astype(np.float32)
                p_va, p_te = fit_lr(Z[sp["train"]], y[sp["train"]], [Z[sp["val"]], Z[sp["test"]]])
                va = float(np.mean((p_va > 0.5) == y[sp["val"]]))
                if va > best[0]:
                    best = (va, ln, p_te)
            Z = np.load(rdir / f"{best[1]}.npy").astype(np.float32)
            p_j = tuned(np.concatenate([C_all, Z], 1), y, sp)
            p_jplus = tuned(np.concatenate([C_plus, Z], 1), y, sp)
            row[f"model:{m}"] = {
                "given_all_plus_minirocket": compare(p_jplus, p_cplus, y_te),
                "vs_all_plus_minirocket": compare(best[2], p_cplus, y_te),
                "layer": best[1],
                "acc": acc(best[2]),
                "auc": float(roc_auc_score(y_te, best[2])),
                "joint_acc": acc(p_j),
                "gain_acc": acc(p_j) - acc(p_c),
                "gain_bits": paired_logloss_difference(p_j, p_c, y_te),
            }
        xt = x[sp["test"]]
        scores = {
            "max_abs_z": np.abs(
                (xt - xt.mean(1, keepdims=True)) / (xt.std(1, keepdims=True) + 1e-9)
            ).max(1),
            "first_difference": robust_max(np.diff(xt, axis=1)),
            "ar16_residual": robust_max(ar_residuals(xt)),
            "matrix_profile_m64": mp_discord(xt, 64),
        }
        row["label_free_auc"] = {k: float(roc_auc_score(y_te, v)) for k, v in scores.items()}
        res["per_seed"][str(seed)] = row
        print(
            f"[seed {seed}] all_controls={row['ctrl:all_controls']['acc']:.3f} "
            + " ".join(f"{m}={row[f'model:{m}']['acc']:.3f}" for m in MODELS),
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
