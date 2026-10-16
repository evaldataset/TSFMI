"""Strong non-model controls on every task where models are compared with controls.

The ROCKET-style transform and tsfresh features were originally run only on the canonical tasks and
(ROCKET) on the hard variants. This script runs them, with the raw-signal and hand-crafted controls,
on the remaining tasks under the protocol each task uses for the models:

- realistic anomaly generator, data seeds 42, 43, 44: canonical 60/20/20 splits, split seeds 0-4,
  StandardScaler + logistic regression, test accuracy;
- hard variants (tsfresh minimal/efficient, which were missing there): same protocol;
- canonical tasks (`--parts canonical`): tsfresh minimal/efficient with per-seed scores;
- six UCR datasets: official train/test split, every series linearly resized to 512 as for the
  models, StandardScaler + logistic regression fit on the training split, test ROC-AUC. The
  hand-crafted vector is reported on the resized input and, for comparison, on the native length.

Usage:
    PYTHONPATH=. python scripts/run_strong_controls_ext.py --out outputs/strong_controls_ext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_canonical_baselines import DATASETS, HARD_DATASETS, hand_crafted_features
from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, three_way_split, train_and_score
from scripts.run_realistic_anomaly import realistic_anomaly_dataset
from scripts.run_rocket_baseline import N_KERNELS, random_kernels, rocket_features
from scripts.run_strong_baselines import extract_tsfresh
from scripts.run_ucr_tsfm_probe import resize_to
from src.datasets.ucr_anomaly import load_ucr_anomaly
from src.metrics.tsad_metrics import window_threshold_free_metrics

UCR = ("ECG200", "Wafer", "Earthquakes", "FordA", "FordB", "Strawberry")
NUM_SAMPLES, SEQ_LEN = 1000, 512
KERNEL_SEED = 42


def control_features(x: NDArray, kernels) -> dict[str, NDArray]:
    feats = {
        "raw_signal": x,
        "hand_crafted": hand_crafted_features(x),
        "rocket": rocket_features(x, kernels),
    }
    for name, minimal in (("tsfresh_minimal", True), ("tsfresh_efficient", False)):
        f = extract_tsfresh(x, minimal=minimal)
        if f is None:
            raise SystemExit("tsfresh is required (pip install tsfresh)")
        feats[name] = f
    return feats


def split_scores(F: NDArray, y: NDArray, task: str = "classification") -> dict:
    scores = []
    for seed in SEEDS:
        X_tr, y_tr, _, _, X_te, y_te = three_way_split(F, y, seed)
        scores.append(train_and_score(X_tr, y_tr, X_te, y_te, task, seed))
    mean, lo, hi = bootstrap_ci(scores)
    return {
        "mean": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "per_seed": [round(s, 4) for s in scores],
        "dim": int(F.shape[1]),
    }


def ucr_auc(F_tr: NDArray, y_tr: NDArray, F_te: NDArray, y_te: NDArray) -> float:
    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    )
    pipe.fit(F_tr, y_tr)
    return round(
        float(window_threshold_free_metrics(pipe.predict_proba(F_te)[:, 1], y_te)["roc_auc"]), 4
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/strong_controls_ext"))
    ap.add_argument("--parts", nargs="*", default=["realistic", "hard", "ucr"])
    args = ap.parse_args()
    kernels = random_kernels(seed=KERNEL_SEED, num_kernels=N_KERNELS)
    out: dict[str, dict] = {"n_kernels": N_KERNELS, "kernel_seed": KERNEL_SEED}

    if "canonical" in args.parts:  # tsfresh on the canonical tasks, with per-seed scores
        out["canonical"] = {}
        for prop, (task, gen) in DATASETS.items():
            d = gen(NUM_SAMPLES, SEQ_LEN, seed=42)
            out["canonical"][prop] = {}
            for name, minimal in (("tsfresh_minimal", True), ("tsfresh_efficient", False)):
                f = extract_tsfresh(d.sequences, minimal=minimal)
                out["canonical"][prop][name] = split_scores(f, d.labels, task)
            print(
                f"[canonical {prop}] "
                + " ".join(f"{k}={v['mean']:.3f}" for k, v in out["canonical"][prop].items()),
                flush=True,
            )

    if "realistic" in args.parts:
        out["realistic"] = {}
        for ds in (42, 43, 44):
            x, y = realistic_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, ds)
            x, y = np.asarray(x), np.asarray(y).astype(np.int64)
            out["realistic"][str(ds)] = {
                k: split_scores(F, y) for k, F in control_features(x, kernels).items()
            }
            print(
                f"[realistic {ds}] "
                + " ".join(f"{k}={v['mean']:.3f}" for k, v in out["realistic"][str(ds)].items()),
                flush=True,
            )

    if "hard" in args.parts:
        out["hard"] = {}
        for prop, (_task, gen) in HARD_DATASETS.items():
            d = gen(NUM_SAMPLES, SEQ_LEN, seed=42)
            feats = control_features(d.sequences, kernels)
            out["hard"][prop] = {
                k: split_scores(feats[k], d.labels.astype(np.int64))
                for k in ("tsfresh_minimal", "tsfresh_efficient", "rocket")
            }
            print(
                f"[hard {prop}] "
                + " ".join(f"{k}={v['mean']:.3f}" for k, v in out["hard"][prop].items()),
                flush=True,
            )

    if "ucr" in args.parts:
        out["ucr"] = {}
        for name in UCR:
            t = load_ucr_anomaly(name)
            n_tr = len(t.train_labels)
            native = hand_crafted_features(np.concatenate([t.train_sequences, t.test_sequences]))
            allx = resize_to(np.concatenate([t.train_sequences, t.test_sequences]))
            feats = control_features(allx, kernels)
            feats["hand_crafted_native"] = native
            row = {
                k: ucr_auc(F[:n_tr], t.train_labels, F[n_tr:], t.test_labels)
                for k, F in feats.items()
            }
            row["n_train"], row["n_test"] = int(n_tr), int(len(t.test_labels))
            row["positive_rate_train"] = round(float(t.train_labels.mean()), 4)
            out["ucr"][name] = row
            print(f"[ucr {name}] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    # Parts run separately are merged into the existing file rather than replacing it.
    path = args.out / "results.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    commands = merged.get("commands", [])
    merged.update(out)
    merged["commands"] = [*commands, " ".join(sys.argv)]
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
