"""Probe-capacity sweep on the anomaly tasks under the canonical v2 protocol.

For every model and the hand-crafted control, probes of increasing capacity -- logistic regression
(the canonical probe), one-hidden-layer MLPs with 64-1024 units, and a random forest -- are fit on
the canonical 60/20/20 splits (split seeds 0-4). For every probe the layer is selected separately on
the validation split and the test score of that layer is reported, so a nonlinear probe is never
scored on a layer chosen for the linear one. MOMENT and GPT4TS go through StandardScaler + PCA-512
fit on the training rows, as in the canonical pipeline; the other models use their mean-pooled view.

Usage:
    PYTHONPATH=. python scripts/run_capacity_probe_v2.py --out outputs/capacity_probe_v2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from scripts.run_anomaly_diagnostics import MODELS, PCA_MODELS, canonical_dir
from scripts.run_canonical_baselines import hand_crafted_features
from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, three_way_split
from scripts.run_realistic_anomaly import realistic_anomaly_dataset
from src.datasets.synthetic import generate_anomaly_dataset

PROBES = ("linear", "mlp_64", "mlp_128", "mlp_256", "mlp_512", "mlp_1024", "rf")
NUM_SAMPLES, SEQ_LEN, DATA_SEED = 1000, 512, 42


def make_probe(name: str, seed: int):
    if name == "linear":
        return LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    if name.startswith("mlp_"):
        return MLPClassifier(
            hidden_layer_sizes=(int(name.split("_")[1]),),
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=10,
        )
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=20, random_state=seed, n_jobs=8)
    raise ValueError(name)


def layer_files(d: Path) -> list[Path]:
    """Layer tensors in depth order (numeric suffix), not string order."""
    files = [f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata")]
    return sorted(files, key=lambda f: int(re.search(r"(\d+)$", f.stem).group(1)))


def features(X: NDArray, seed: int, y: NDArray, pca: int | None):
    X_tr, y_tr, X_va, y_va, X_te, y_te = three_way_split(X, y, seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(a) for a in (X_tr, X_va, X_te))
    if pca is not None and X_tr.shape[1] > pca:
        p = PCA(n_components=pca, random_state=seed).fit(X_tr)
        X_tr, X_va, X_te = (p.transform(a) for a in (X_tr, X_va, X_te))
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def load_layer(f: Path) -> NDArray:
    t = torch.load(f, map_location="cpu", weights_only=True)
    return t.reshape(t.shape[0], -1).numpy().astype(np.float32)


def sweep(sources: list, y: NDArray, pca: int | None) -> dict:
    """Per probe: per seed, the test score of the layer with the best validation score.

    ``sources`` are callables returning one layer's features, so that only one layer is held in
    memory at a time (MOMENT's flattened per-token layers are 65,536-dimensional).
    """
    best = {(s, p): (-np.inf, float("nan"), -1) for s in SEEDS for p in PROBES}
    for li, get in enumerate(sources):
        X = get()
        for seed in SEEDS:
            X_tr, y_tr, X_va, y_va, X_te, y_te = features(X, seed, y, pca)
            for p in PROBES:
                clf = make_probe(p, seed).fit(X_tr, y_tr)
                val = float((clf.predict(X_va) == y_va).mean())
                if val > best[(seed, p)][0]:
                    best[(seed, p)] = (val, float((clf.predict(X_te) == y_te).mean()), li)
        del X
    out = {}
    for p in PROBES:
        tests = [best[(s, p)][1] for s in SEEDS]
        mean, lo, hi = bootstrap_ci(tests)
        out[p] = {
            "mean": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "per_seed": [round(v, 4) for v in tests],
            "layer_index": [best[(s, p)][2] for s in SEEDS],
        }
    return out


def load_task(task: str) -> tuple[NDArray, NDArray]:
    if task == "realistic":
        x, y = realistic_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, DATA_SEED)
        return np.asarray(x), np.asarray(y).astype(np.int64)
    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
    return ds.sequences, ds.labels.astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("outputs/capacity_probe_v2"))
    ap.add_argument("--tasks", nargs="*", default=["canonical", "realistic"])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    root = args.root.resolve()
    results: dict[str, dict] = {}
    for task in args.tasks:
        x, y = load_task(task)
        rows = {}
        for m in args.models:
            res = json.loads((canonical_dir(root, task, m) / "canonical_results.json").read_text())
            d = Path(res["representations_dir"])
            stored = torch.load(d / "labels.pt", map_location="cpu", weights_only=True).numpy()
            if not np.array_equal(stored, y):
                raise SystemExit(f"{task}/{m}: stored labels differ from the regenerated dataset")
            files = layer_files(d)
            if not files:
                raise SystemExit(f"{task}/{m}: no layer tensors in {d}")
            rows[m] = sweep([lambda f=f: load_layer(f) for f in files], y, PCA_MODELS.get(m))
            print(
                f"[{task}] {m:<12} " + " ".join(f"{p}={rows[m][p]['mean']:.3f}" for p in PROBES),
                flush=True,
            )
        hc = hand_crafted_features(x)
        rows["hand_crafted"] = sweep([lambda hc=hc: hc], y, None)
        print(
            f"[{task}] hand_crafted "
            + " ".join(f"{p}={rows['hand_crafted'][p]['mean']:.3f}" for p in PROBES),
            flush=True,
        )
        results[task] = rows
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "protocol": __doc__.split("Usage:")[0].strip(),
                "probes": list(PROBES),
                "results": results,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
