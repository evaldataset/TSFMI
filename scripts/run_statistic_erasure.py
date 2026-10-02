"""Does anomaly accessibility in TSFM representations run through the sufficient statistics?

For the canonical anomaly task, the Bayes statistic (LLR), max|x| and excess kurtosis nearly
determine the label. This script removes all linear information about those three statistics
from each representation with LEACE (Belrose et al., 2023) and re-probes the anomaly label.

Protocol (canonical v2): split seeds 0-4 and 60/20/20 splits, the layer that canonical v2 selected
on validation for each seed, StandardScaler (then PCA-512 for MOMENT and GPT4TS) fit on the training
split, LEACE fit on the training split only and applied to both splits, then the canonical
logistic-regression probe. Conditions:
    none         no erasure (reproduces the canonical v2 test score)
    statistics   erase z = standardized (LLR, max|x|, kurtosis)
    random       erase a 3-D Gaussian z independent of the data (negative control)
    label        erase the anomaly label itself (positive control; should reach chance)
For each erasure the held-out R^2 of a ridge regression from the erased representation to the three
statistics is recorded to confirm that the linear information is gone.

With ``--concept handcrafted`` the erased variable is instead the standardized 8-D hand-crafted
vector (and the random control is 8-D), which asks whether a representation carries anomaly
information beyond the hand-crafted statistics; ``--task realistic`` runs on the realistic generator
with the layers selected in outputs/realistic_anomaly_v2.

Usage:
    PYTHONPATH=. python scripts/run_statistic_erasure.py --out outputs/statistic_erasure
    PYTHONPATH=. python scripts/run_statistic_erasure.py --task realistic --concept handcrafted \
        --out outputs/statistic_erasure/realistic_handcrafted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from concept_erasure import LeaceFitter
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from scripts.run_anomaly_diagnostics import MODELS, PCA_MODELS, canonical_dir, load_layer
from scripts.run_canonical_baselines import hand_crafted_features
from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, three_way_split
from scripts.run_realistic_anomaly import realistic_anomaly_dataset
from scripts.run_statistic_recoverability import targets
from src.datasets.synthetic import generate_anomaly_dataset

NUM_SAMPLES, SEQ_LEN, DATA_SEED = 1000, 512, 42
CONDITIONS = ("none", "statistics", "random", "label")  # "statistics" = the erased concept
CONCEPT_NAMES = {
    "statistics": ("llr", "max_abs", "kurtosis"),
    "handcrafted": (
        "slope",
        "residual_std",
        "mean",
        "std",
        "kurtosis",
        "argmax_fft",
        "spectral_entropy",
        "diff_std",
    ),
}


def erase(X_tr: NDArray, X_te: NDArray, z_tr: NDArray) -> tuple[NDArray, NDArray]:
    fitter = LeaceFitter.fit(
        torch.tensor(X_tr, dtype=torch.float64), torch.tensor(z_tr, dtype=torch.float64)
    )
    eraser = fitter.eraser
    return (
        eraser(torch.tensor(X_tr, dtype=torch.float64)).numpy(),
        eraser(torch.tensor(X_te, dtype=torch.float64)).numpy(),
    )


def run_row(X: NDArray, y: NDArray, S: NDArray, seed: int, pca: int | None) -> dict:
    X_tr, y_tr, _, _, X_te, y_te = three_way_split(X, y, seed)
    S_tr, _, _, _, S_te, _ = three_way_split(S, y, seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    if pca is not None and X_tr.shape[1] > pca:
        p = PCA(n_components=pca, random_state=seed).fit(X_tr)
        X_tr, X_te = p.transform(X_tr), p.transform(X_te)
    s_mu, s_sd = S_tr.mean(0), S_tr.std(0) + 1e-12
    zs_tr, zs_te = (S_tr - s_mu) / s_sd, (S_te - s_mu) / s_sd
    rng = np.random.default_rng(1000 + seed)
    out = {}
    for cond in CONDITIONS:
        match cond:
            case "none":
                A_tr, A_te = X_tr, X_te
            case "statistics":
                A_tr, A_te = erase(X_tr, X_te, zs_tr)
            case "random":
                A_tr, A_te = erase(X_tr, X_te, rng.standard_normal((len(X_tr), S.shape[1])))
            case "label":
                A_tr, A_te = erase(X_tr, X_te, y_tr.reshape(-1, 1).astype(np.float64))
        clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        acc = float((clf.fit(A_tr, y_tr).predict(A_te) == y_te).mean())
        r2 = r2_score(
            zs_te, Ridge(alpha=1.0).fit(A_tr, zs_tr).predict(A_te), multioutput="raw_values"
        )
        out[cond] = {"acc": acc, "stat_r2": [float(v) for v in r2]}
    return out


def summarize(per_seed: list[dict], names: tuple[str, ...]) -> dict:
    res = {}
    for cond in CONDITIONS:
        accs = [s[cond]["acc"] for s in per_seed]
        mean, lo, hi = bootstrap_ci(accs)
        r2 = np.mean([s[cond]["stat_r2"] for s in per_seed], axis=0)
        res[cond] = {
            "acc_mean": round(mean, 4),
            "acc_ci95": [round(lo, 4), round(hi, 4)],
            "acc_per_seed": [round(a, 4) for a in accs],
            "stat_r2_mean": {k: round(float(v), 4) for k, v in zip(names, r2, strict=True)},
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("outputs/statistic_erasure"))
    ap.add_argument("--task", choices=("canonical", "realistic"), default="canonical")
    ap.add_argument("--concept", choices=tuple(CONCEPT_NAMES), default="statistics")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.task == "realistic":
        x, y = realistic_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, DATA_SEED)
    else:
        ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
        x, y = ds.sequences, ds.labels
    y = np.asarray(y).astype(np.int64)
    hc = hand_crafted_features(x)
    if args.concept == "statistics":
        t = targets(x)
        S = np.stack([t["llr"], t["max_abs"], t["kurtosis"]], axis=1)
    else:
        S = hc
    names = CONCEPT_NAMES[args.concept]

    rows: dict[str, dict] = {}
    for m in MODELS:
        res = json.loads((canonical_dir(root, args.task, m) / "canonical_results.json").read_text())
        repr_dir = Path(res["representations_dir"])
        labels = torch.load(repr_dir / "labels.pt", map_location="cpu", weights_only=True).numpy()
        if not np.array_equal(labels, y):
            raise SystemExit(f"{m}: stored labels differ from the regenerated dataset")
        per_seed = []
        for seed, layer in zip(res["seeds"], res["best_layer_per_seed"], strict=True):
            per_seed.append(run_row(load_layer(repr_dir, layer), y, S, seed, PCA_MODELS.get(m)))
        rows[m] = summarize(per_seed, names)
        print(
            f"{m:<12} " + " ".join(f"{c}={rows[m][c]['acc_mean']:.3f}" for c in CONDITIONS),
            flush=True,
        )
    rows["hand_crafted"] = summarize([run_row(hc, y, S, seed, None) for seed in SEEDS], names)
    print(
        "hand_crafted "
        + " ".join(f"{c}={rows['hand_crafted'][c]['acc_mean']:.3f}" for c in CONDITIONS),
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "protocol": __doc__.split("Usage:")[0].strip(),
                "task": args.task,
                "concept": args.concept,
                "concept_variables": list(names),
                "conditions": list(CONDITIONS),
                "rows": rows,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
