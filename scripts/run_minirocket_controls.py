"""MiniRocket (Dempster et al., 2021) as an additional non-model control.

MiniRocket's bias values are quantiles of convolution outputs on training data, so the transform is
fit on the training split only and applied to validation and test windows. Probe: StandardScaler +
logistic regression with C tuned on validation log-loss (as for the conditional probes).

Parts:
- realistic: the large-sample realistic task (seed 1001, 30,000 / 10,000 / 10,000), MiniRocket alone
  and together with all other feature controls, paired against the all-controls probe;
- ucr_anomaly: the UCR Anomaly Archive windows with the five series-level split seeds.

Usage:
    TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_minirocket_controls.py \
        --part realistic --out outputs/minirocket
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

N_KERNELS = 10_000


def minirocket(x_fit: NDArray, x_all: NDArray, jobs: int) -> NDArray[np.float32]:
    from aeon.transformations.collection.convolution_based import MiniRocket

    m = MiniRocket(n_kernels=N_KERNELS, random_state=0, n_jobs=jobs)
    m.fit(x_fit[:, None, :].astype(np.float32))
    parts = [
        m.transform(x_all[s : s + 5000, None, :].astype(np.float32))
        for s in range(0, len(x_all), 5000)
    ]
    return np.concatenate(parts).astype(np.float32)


def main() -> int:
    from scripts.run_resolution_study import (
        STORE,
        compare,
        fit_predict_tuned,
        load_data,
        split_indices,
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("realistic", "ucr_anomaly"), required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/minirocket"))
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    if args.part == "realistic":
        d = load_data("realistic")
        x, y = d["x"], d["y"].astype(np.int64)
        sp = split_indices(len(y))
        F = minirocket(x[sp["train"]], x, args.jobs)
        base = STORE / "controls" / "realistic"
        C = np.concatenate(
            [np.load(base / f"{k}.npy") for k in ("hand_crafted", "rocket", "tsfresh_efficient")], 1
        )
        y_te = y[sp["test"]]
        n_tr = len(sp["train"])
        _, _, p_mr = fit_predict_tuned(F, y, sp, n_tr)
        _, _, p_c = fit_predict_tuned(C, y, sp, n_tr)
        _, _, p_cm = fit_predict_tuned(np.concatenate([C, F], 1), y, sp, n_tr)
        results["realistic"] = {
            "n_features": int(F.shape[1]),
            "minirocket_acc": float(np.mean((p_mr > 0.5) == y_te)),
            "all_controls_acc": float(np.mean((p_c > 0.5) == y_te)),
            "all_controls_plus_minirocket_acc": float(np.mean((p_cm > 0.5) == y_te)),
            "minirocket_vs_all_controls": compare(p_mr, p_c, y_te),
            "minirocket_given_controls": compare(p_cm, p_c, y_te),
        }
        np.save(STORE / "controls" / "realistic" / "minirocket.npy", F.astype(np.float16))
        print(json.dumps({k: v for k, v in results["realistic"].items() if "acc" in k}), flush=True)

    else:
        from scripts.run_ucr_anomaly_archive import SEEDS, group_split
        from scripts.run_ucr_anomaly_archive import STORE as USTORE

        with np.load(USTORE / "windows.npz") as z:
            x, y, groups = z["x"], z["y"].astype(np.int64), z["group"]
        rows = {}
        for seed in SEEDS:
            sp = group_split(groups, seed)
            F = minirocket(x[sp["train"]], x, args.jobs)
            _, _, p = fit_predict_tuned(F, y, sp, len(sp["train"]))
            rows[str(seed)] = float(np.mean((p > 0.5) == y[sp["test"]]))
            print(f"[ucr_anomaly seed {seed}] minirocket={rows[str(seed)]:.4f}", flush=True)
        results["ucr_anomaly"] = {"per_seed_acc": rows}

    path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
