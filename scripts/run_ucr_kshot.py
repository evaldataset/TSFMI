"""Label efficiency of frozen TSFM embeddings on UCR 2018 classification, k-shot.

Datasets: fixed-length UCR 2018 datasets from the paper's 1-NN benchmark (outputs/embedding_store/
classical.json) with at most 10 classes, at least 20 training series in every class, and at most
3,000 test series; the first 20 in alphabetical order. Representations of the z-normalised series:
TimesFM 2.0, PatchTST-FM, MOMENT, Chronos-Bolt (series resized to 512, mean-pooled at the layer
chosen on the realistic task — a label-free choice, so no extra labels select layers); controls on
native length: hand-crafted, catch22, MiniRocket (fit on the k-shot training series), their union.
k in {1, 2, 5, 10, 20} training series per class (five draws); test on the official test split.
Probe: standardised logistic regression, C = 1 (no validation data at k-shot).

Parts: extract --model M (GPU) -> evaluate (CPU).
Arrays go to $UCR_KSHOT_STORE (default outputs/resolution_store/ucr_kshot); results to
outputs/ucr_kshot/results.json.
Usage:
    TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_ucr_kshot.py --part extract \
        --model timesfm
    ... run_ucr_kshot.py --part evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/ucr_kshot")
STORE = Path(
    os.environ.get(
        "UCR_KSHOT_STORE",
        Path(os.environ.get("TSFMI_RESOLUTION_STORE", "outputs/resolution_store")) / "ucr_kshot",
    )
)
UCR = Path(os.environ.get("TSFMI_UCR2018", "data/UCRArchive_2018"))
MODELS = ("timesfm", "patchtst_fm", "moment", "chronos")
KS = (1, 2, 5, 10, 20)
DRAWS = 5


def datasets() -> list[str]:
    from scripts.run_embedding_store import load_ucr

    classical = json.loads(Path("outputs/embedding_store/classical.json").read_text())
    keep = []
    for name in sorted(classical):
        r = classical[name]
        if r["n_classes"] > 10 or r["n_test"] > 3000:
            continue
        d = load_ucr(UCR, name)
        if d is None:
            continue
        _, y_tr, _, _ = d
        if np.bincount(y_tr.astype(int)).min() >= 20:
            keep.append(name)
        if len(keep) == 20:
            break
    return keep


def cmd_extract(args: argparse.Namespace) -> None:
    import torch

    from scripts.extract_representations import load_model_wrapper
    from scripts.run_embedding_store import load_ucr, resize, znorm
    from scripts.run_resolution_study import selected_layer
    from src.extractors.hook_manager import HookManager
    from src.utils.seed import seed_everything

    STORE.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = load_model_wrapper(args.model, None, device)
    layer = selected_layer(args.model)
    name = next(n for n in wrapper.get_layer_names() if n.replace(".", "_") == layer)
    for ds in datasets():
        path = STORE / f"{ds}_{args.model}.npy"
        if path.exists():
            continue
        x_tr, _, x_te, _ = load_ucr(UCR, ds)
        x = resize(znorm(np.concatenate([x_tr, x_te]))).astype(np.float32)
        out = []
        with HookManager(wrapper.model, [name]) as hm, torch.inference_mode():
            for s in range(0, len(x), args.batch):
                wrapper.forward(
                    torch.tensor(x[s : s + args.batch], dtype=torch.float32, device=device)
                )
                a = hm.get_activations()[name].float()
                out.append(a.reshape(a.shape[0], -1, a.shape[-1]).mean(1).cpu().numpy())
                hm.clear()
        np.save(path, np.concatenate(out).astype(np.float32))
        print("extracted", ds, args.model, flush=True)


def eval_ds(ds: str) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    from scripts.run_canonical_baselines import hand_crafted_features
    from scripts.run_embedding_store import load_ucr, znorm
    from scripts.run_minirocket_controls import minirocket
    from scripts.run_strong_controls_ext import catch22

    x_tr, y_tr, x_te, y_te = load_ucr(UCR, ds)
    z = np.concatenate([znorm(x_tr), znorm(x_te)])
    n_tr = len(y_tr)
    hc = hand_crafted_features(z).astype(np.float32)
    c22 = catch22(z, 2)
    reps = {m: np.load(STORE / f"{ds}_{m}.npy") for m in MODELS}
    rows = []

    def acc(F: np.ndarray, tr: np.ndarray) -> float:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        clf.fit(F[tr], y_tr[tr])
        return float(np.mean(clf.predict(F[n_tr:]) == y_te))

    with threadpool_limits(limits=2):
        for k in KS:
            for d in range(DRAWS):
                rng = np.random.default_rng(1000 * k + d)
                tr = np.concatenate(
                    [
                        rng.choice(np.flatnonzero(y_tr == c), size=k, replace=False)
                        for c in np.unique(y_tr)
                    ]
                )
                mr = minirocket(z[tr], z, 2)
                union = np.concatenate([hc, c22, mr], axis=1)
                row = {
                    "dataset": ds,
                    "k": k,
                    "draw": d,
                    "minirocket": acc(mr, tr),
                    "catch22": acc(c22, tr),
                    "union": acc(union, tr),
                }
                for m, Z in reps.items():
                    row[m] = acc(Z, tr)
                    row[f"union+{m}"] = acc(np.concatenate([union, Z], axis=1), tr)
                rows.append(row)
    print(ds, "done", flush=True)
    return rows


def cmd_evaluate(args: argparse.Namespace) -> None:
    from joblib import Parallel, delayed

    OUT.mkdir(parents=True, exist_ok=True)
    names = datasets()
    rows = Parallel(n_jobs=args.jobs)(delayed(eval_ds)(ds) for ds in names)
    (OUT / "results.json").write_text(
        json.dumps({"datasets": names, "rows": [r for rs in rows for r in rs]}, indent=1)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("extract", "evaluate", "list"), required=True)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    if args.part == "list":
        print(datasets())
        return 0
    {"extract": cmd_extract, "evaluate": cmd_evaluate}[args.part](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
