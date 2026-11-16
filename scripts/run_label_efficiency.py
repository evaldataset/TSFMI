"""Label efficiency of frozen TSFM embeddings on the UCR Anomaly Archive.

Extends P3 to (i) all seven models (linear probe on the validation-selected mean-pooled layer),
(ii) the time-reversed and negated window stores (pretraining-overlap sensitivity), and (iii) the
conditional probe union+model for four models. Within each of the paper's five series-level splits
the training series are subsampled to n in {5, 10, 20, 40, 80, all} (three draws each); validation
and test series are fixed. MiniRocket is fit on the drawn training windows. Label-equivalence is
estimated by scripts/summarize_label_efficiency.py.

Usage:
    TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_label_efficiency.py --store none
    (--store negate / reverse for the transformed windows; run run_ucr_anomaly_archive.py first)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/label_efficiency")
SIZES = (5, 10, 20, 40, 80, None)
DRAWS = 3
MODELS = ("moment", "chronos", "gpt4ts", "timer", "timesfm", "moirai", "patchtst_fm", "mantis")
JOINT = ("patchtst_fm", "moment", "timesfm", "chronos", "mantis")


def store_path(name: str) -> Path:
    from scripts.run_ucr_anomaly_archive import STORE

    if name.startswith("tsbad"):
        return STORE.parent / name  # TSB-AD-U windows (scripts/run_tsbad_windows.py)
    return STORE if name == "none" else STORE.parent / f"ucr_anomaly_{name}"


def run(
    store_name: str, split_seed: int, n_series: int | None, draw: int, models: tuple = MODELS
) -> dict:
    from threadpoolctl import threadpool_limits

    from scripts.run_minirocket_controls import minirocket
    from scripts.run_ucr_anomaly_archive import fit_lr, group_split, tuned

    store = store_path(store_name)
    with np.load(store / "windows.npz") as z:
        x, y, groups = z["x"], z["y"].astype(np.int64), z["group"]
    sp = group_split(groups, split_seed)
    tr_series = np.unique(groups[sp["train"]])
    if n_series is not None:
        rng = np.random.default_rng(1000 * split_seed + 10 * draw + n_series)  # same draws as P3
        keep = rng.choice(tr_series, size=min(n_series, len(tr_series)), replace=False)
        sp = {**sp, "train": sp["train"][np.isin(groups[sp["train"]], keep)]}
    y_te = y[sp["test"]]

    def acc(p: np.ndarray) -> float:
        return float(np.mean((p > 0.5) == y_te))

    row = {
        "store": store_name,
        "split": split_seed,
        "n_series": n_series or int(len(tr_series)),
        "draw": draw,
        "n_train_windows": int(len(sp["train"])),
    }
    with threadpool_limits(limits=4):
        mr = minirocket(x[sp["train"]], x, 4)
        ctrl = np.concatenate(
            [np.load(store / f"{k}.npy") for k in ("hand_crafted", "rocket", "tsfresh_efficient")]
            + [mr],
            axis=1,
        )
        row["minirocket"] = acc(tuned(mr, y, sp))
        row["strong_union"] = acc(tuned(ctrl, y, sp))
        for m in models:
            rdir = store / "repr" / m
            order = json.loads((rdir / "meta.json").read_text())["layer_order"]
            best = (-1.0, None)
            for ln in order:
                Z = np.load(rdir / f"{ln}.npy").astype(np.float32)
                (p_va,) = fit_lr(Z[sp["train"]], y[sp["train"]], [Z[sp["val"]]])
                va = float(np.mean((p_va > 0.5) == y[sp["val"]]))
                if va > best[0]:
                    best = (va, ln)
            Z = np.load(rdir / f"{best[1]}.npy").astype(np.float32)
            (p_z,) = fit_lr(Z[sp["train"]], y[sp["train"]], [Z[sp["test"]]])
            row[m] = acc(p_z)
            row[f"{m}_layer"] = best[1]
            if m in JOINT:
                row[f"union+{m}"] = acc(tuned(np.concatenate([ctrl, Z], 1), y, sp))
    print(json.dumps({k: v for k, v in row.items() if not k.endswith("_layer")}), flush=True)
    return row


def main() -> int:
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--store", choices=("none", "negate", "reverse", "tsbad", "tsbad_negate"), required=True
    )
    ap.add_argument(
        "--models", nargs="*", default=list(MODELS), help="subset; merged into the file"
    )
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [
        (args.store, s, n, d) for s in range(5) for n in SIZES for d in (range(DRAWS) if n else [0])
    ]
    rows = Parallel(n_jobs=args.jobs)(delayed(run)(*t, tuple(args.models)) for t in tasks)
    path = OUT / f"{args.store}.json"
    if path.exists() and set(args.models) != set(MODELS):  # add these models' columns
        old = {(r["split"], r["n_series"], r["draw"]): r for r in json.loads(path.read_text())}
        for r in rows:
            key = (r["split"], r["n_series"], r["draw"])
            new = {
                k: v
                for k, v in r.items()
                if any(k in (m, f"{m}_layer", f"union+{m}") for m in args.models)
            }
            old[key] = {**old[key], **new} if key in old else r  # keep committed controls
        rows = list(old.values())
    path.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
