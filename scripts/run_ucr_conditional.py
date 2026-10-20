"""Conditional probing on the six binary UCR datasets: does a model add to the strongest control?

For every dataset and model, the model's representation is built exactly as in
run_ucr_tsfm_probe.py (series resized to 512, mean-pooled layers, PCA-512 fit on the training split
where wider, layer chosen by validation ROC-AUC on a split of the training data). The probe is then
refit on [control, Z] and compared with the probe on the control alone (Z | control) and on Z alone
(control | Z), on the official test split, each probe with its own C chosen by log-loss on the inner
validation split from {0.001, 0.01, 0.1, 1}: ROC-AUC differences with paired bootstrap intervals and
held-out log-loss differences in bits (conditional V-information; Hewitt et al., 2021). Controls:
the ROCKET-style transform, tsfresh (efficient), and their union with the hand-crafted vector.

Usage:
    PYTHONPATH=. python scripts/run_ucr_conditional.py --out outputs/ucr_conditional
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_canonical_baselines import hand_crafted_features
from scripts.run_realistic_anomaly_tsfm import load_wrapper
from scripts.run_resolution_study import tsfresh_features
from scripts.run_rocket_baseline import N_KERNELS, random_kernels, rocket_features
from scripts.run_ucr_tsfm_probe import SEED, extract_pooled, resize_to
from src.datasets.ucr_anomaly import load_ucr_anomaly
from src.metrics.paired_tests import paired_logloss_difference
from src.utils.seed import seed_everything

UCR = ("ECG200", "Wafer", "Earthquakes", "FordA", "FordB", "Strawberry")
MODELS = ("moment", "chronos", "gpt4ts", "timer", "timesfm", "moirai", "patchtst_fm")
N_BOOT = 2000


C_GRID = (0.001, 0.01, 0.1, 1.0)


def probe(X_tr: NDArray, y_tr: NDArray, X_te: NDArray, c: float = 1.0) -> NDArray:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=1000, random_state=SEED, solver="lbfgs"),
    )
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def inner_split(n_tr: int) -> tuple[NDArray, NDArray]:
    idx = np.random.default_rng(SEED).permutation(n_tr)
    n_val = max(10, int(n_tr * 0.25))
    return idx[n_val:], idx[:n_val]


def tuned_probe(X: NDArray, y_tr: NDArray, n_tr: int) -> tuple[NDArray, float]:
    """Test P(y=1) with C chosen by log-loss on the inner validation split of the training data."""
    from sklearn.metrics import log_loss

    vtr, vva = inner_split(n_tr)
    X_tr = X[:n_tr]
    losses = [
        log_loss(y_tr[vva], probe(X_tr[vtr], y_tr[vtr], X_tr[vva], c), labels=[0, 1])
        for c in C_GRID
    ]
    c = C_GRID[int(np.argmin(losses))]
    return probe(X_tr, y_tr, X[n_tr:], c), c


def model_features(model: str, allseq: NDArray, y_tr: NDArray, n_tr: int) -> tuple[NDArray, str]:
    """Selected-layer features (train rows, then test rows), built as in tsfm_score."""
    seed_everything(SEED)
    wrapper = load_wrapper(model)
    wrapper.load()
    wrapper.freeze()
    feats = extract_pooled(wrapper, allseq)
    vtr, vva = inner_split(n_tr)
    best, best_auc, best_X = None, -np.inf, None
    for ln, X in feats.items():
        X_tr, X_te = X[:n_tr], X[n_tr:]
        if X_tr.shape[1] > 512:
            pca = PCA(n_components=min(512, len(vtr) - 1), random_state=SEED).fit(X_tr[vtr])
            X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)
        auc = roc_auc_score(y_tr[vva], probe(X_tr[vtr], y_tr[vtr], X_tr[vva]))
        if auc > best_auc:
            best, best_auc, best_X = ln, auc, np.concatenate([X_tr, X_te])
    return best_X, best


def auc_diff(p_a: NDArray, p_b: NDArray, y: NDArray) -> dict[str, float]:
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    d = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return {
        "diff": float(d),
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/ucr_conditional"))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    kernels = random_kernels(seed=42, num_kernels=N_KERNELS)
    for name in UCR:
        t = load_ucr_anomaly(name)
        n_tr = len(t.train_labels)
        y_tr, y_te = t.train_labels, t.test_labels
        raw = np.concatenate([t.train_sequences, t.test_sequences])
        allseq = resize_to(raw)
        ctrl = {
            "rocket": rocket_features(allseq, kernels),
            "tsfresh_efficient": tsfresh_features(allseq, n_jobs=16),
        }
        ctrl["all_controls"] = np.concatenate(
            [ctrl["rocket"], ctrl["tsfresh_efficient"], hand_crafted_features(allseq)], axis=1
        )
        row = results.setdefault(name, {"n_train": int(n_tr), "n_test": int(len(y_te))})
        p_ctrl, c_ctrl = {}, {}
        for k, C in ctrl.items():
            p_ctrl[k], c_ctrl[k] = tuned_probe(C, y_tr, n_tr)
        row["controls"] = {k: float(roc_auc_score(y_te, p)) for k, p in p_ctrl.items()}
        row["controls_C"] = c_ctrl
        for model in args.models:
            if model in row:
                continue
            Z, layer = model_features(model, allseq, y_tr, n_tr)
            p_z, c_z = tuned_probe(Z, y_tr, n_tr)
            entry = {"layer": layer, "auc": float(roc_auc_score(y_te, p_z)), "C": c_z}
            for k, C in ctrl.items():
                p_j, c_j = tuned_probe(np.concatenate([C, Z], axis=1), y_tr, n_tr)
                entry[k] = {
                    "C": c_j,
                    "auc_joint": float(roc_auc_score(y_te, p_j)),
                    "z_given_control": auc_diff(p_j, p_ctrl[k], y_te),
                    "control_given_z": auc_diff(p_j, p_z, y_te),
                    "bits_z_given_control": paired_logloss_difference(p_j, p_ctrl[k], y_te),
                    "bits_control_given_z": paired_logloss_difference(p_j, p_z, y_te),
                }
            row[model] = entry
            g = entry["all_controls"]["z_given_control"]
            print(
                f"[{name} {model}] Z={entry['auc']:.3f} all={row['controls']['all_controls']:.3f}"
                f" joint={entry['all_controls']['auc_joint']:.3f}"
                f" gain={g['diff']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]",
                flush=True,
            )
            path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
