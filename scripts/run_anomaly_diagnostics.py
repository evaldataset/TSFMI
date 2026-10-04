"""Two follow-up diagnostics of the anomaly probes, on stored canonical-v2 representations.

1. ``per_type``: on the realistic generator, the recall of each probe for each of the four anomaly
   kinds (point, level shift, variance change, contextual) and its specificity on normal windows.
2. ``learning_curve``: test accuracy as a function of the number of labelled training windows, on
   the canonical and the realistic anomaly tasks.

Both reuse the canonical v2 protocol unchanged: the split function, the probe pipeline (PCA-512
inside the pipeline for MOMENT and GPT4TS), split seeds 0-4, and, for every model and seed, the
layer that the canonical v2 run selected on validation. For the learning curve the layer is the one
selected with the full training split; only the probe is refit on the subsample. Non-model controls
(raw signal, hand-crafted 8-D) go through the same code.

Usage:
    PYTHONPATH=. python scripts/run_anomaly_diagnostics.py --out outputs/anomaly_diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from scripts.run_canonical_baselines import hand_crafted_features
from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, build_pipeline, three_way_split
from scripts.run_realistic_anomaly import realistic_anomaly_dataset
from src.datasets.synthetic import generate_anomaly_dataset

MODELS = ("moment", "chronos", "timer", "timesfm", "moirai", "gpt4ts", "patchtst_fm")
PCA_MODELS = {"moment": 512, "gpt4ts": 512}
KINDS = {0: "point", 1: "level_shift", 2: "variance", 3: "contextual"}
TRAIN_SIZES = (50, 100, 200, 400, 600)
NUM_SAMPLES, SEQ_LEN, DATA_SEED = 1000, 512, 42
RESULT_DIRS = {
    "canonical": ("outputs/canonical_v2", "{m}_anomaly"),
    "realistic": ("outputs/realistic_anomaly_v2", "{m}_anomaly_realistic"),
}


def canonical_dir(root: Path, task: str, model: str) -> Path:
    base, pattern = RESULT_DIRS[task]
    cands = [
        d
        for d in (root / base).iterdir()
        if d.name
        in (
            pattern.format(m=model),
            pattern.format(m=f"{model}_pca512"),
            pattern.format(m=f"{model}_meanpool"),
        )
    ]
    if len(cands) != 1:
        raise SystemExit(f"{task}/{model}: expected one result dir, found {cands}")
    return cands[0]


def load_layer(repr_dir: Path, layer: str) -> NDArray[np.float64]:
    t = torch.load(repr_dir / f"{layer}.pt", map_location="cpu", weights_only=True)
    return t.reshape(t.shape[0], -1).numpy().astype(np.float64)


def data(task: str) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64] | None]:
    if task == "realistic":
        return realistic_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, DATA_SEED, return_kinds=True)
    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
    return ds.sequences, ds.labels, None


def feature_sources(root: Path, task: str, x: NDArray[np.float64], y: NDArray[np.int64]):
    """Yield (row, per-seed feature matrix getter, pca components) for models and controls."""
    for m in MODELS:
        res = json.loads((canonical_dir(root, task, m) / "canonical_results.json").read_text())
        repr_dir = Path(res["representations_dir"])
        labels = torch.load(repr_dir / "labels.pt", map_location="cpu", weights_only=True).numpy()
        if not np.array_equal(labels, y):
            raise SystemExit(f"{task}/{m}: stored labels differ from the regenerated dataset")
        layers = dict(zip(res["seeds"], res["best_layer_per_seed"], strict=True))
        cache: dict[str, NDArray[np.float64]] = {}

        def getter(
            seed: int, repr_dir: Path = repr_dir, layers: dict = layers, cache: dict = cache
        ) -> NDArray[np.float64]:
            name = layers[seed]
            if name not in cache:
                cache.clear()
                cache[name] = load_layer(repr_dir, name)
            return cache[name]

        yield m, getter, PCA_MODELS.get(m)
    hc = hand_crafted_features(x)
    yield "hand_crafted", lambda seed: hc, None
    yield "raw_signal", lambda seed: x, None


def fit_predict(X_tr, y_tr, X_te, seed: int, pca: int | None) -> NDArray[np.int64]:
    if pca is not None:
        pca = min(pca, len(X_tr))
    pipe = build_pipeline("classification", seed, X_tr.shape[1], pca)
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def per_type(root: Path) -> dict:
    x, y, kinds = data("realistic")
    idx = np.arange(len(y), dtype=np.float64)
    out = {}
    for row, get, pca in feature_sources(root, "realistic", x, y):
        hits = {k: [0, 0] for k in [*KINDS.values(), "normal"]}
        acc = []
        for seed in SEEDS:
            X = get(seed)
            X_tr, y_tr, _, _, X_te, y_te = three_way_split(X, y, seed)
            _, _, _, _, i_te, _ = three_way_split(idx, y, seed)
            k_te = kinds[i_te.astype(np.int64)]
            pred = fit_predict(X_tr, y_tr, X_te, seed, pca)
            acc.append(float((pred == y_te).mean()))
            for code, name in [*KINDS.items(), (-1, "normal")]:
                sel = k_te == code
                correct = pred[sel] == (0 if code == -1 else 1)
                hits[name][0] += int(correct.sum())
                hits[name][1] += int(sel.sum())
        out[row] = {
            "accuracy_mean": round(float(np.mean(acc)), 4),
            "recall": {k: round(h / n, 4) for k, (h, n) in hits.items()},
            "counts": {k: n for k, (_, n) in hits.items()},
        }
        r = out[row]["recall"]
        print(
            f"[per_type] {row:<13} acc={out[row]['accuracy_mean']:.3f} "
            + " ".join(f"{k}={v:.2f}" for k, v in r.items()),
            flush=True,
        )
    return out


def learning_curve(root: Path, task: str) -> dict:
    x, y, _ = data(task)
    out = {}
    for row, get, pca in feature_sources(root, task, x, y):
        out[row] = {}
        for n_train in TRAIN_SIZES:
            scores = []
            for seed in SEEDS:
                X_tr, y_tr, _, _, X_te, y_te = three_way_split(get(seed), y, seed)
                sub = np.random.default_rng(seed).permutation(len(y_tr))[:n_train]
                pred = fit_predict(X_tr[sub], y_tr[sub], X_te, seed, pca)
                scores.append(float((pred == y_te).mean()))
            mean, lo, hi = bootstrap_ci(scores)
            out[row][str(n_train)] = {
                "mean": round(mean, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "per_seed": [round(s, 4) for s in scores],
            }
        print(
            f"[curve/{task}] {row:<13} "
            + " ".join(f"{n}:{out[row][str(n)]['mean']:.3f}" for n in TRAIN_SIZES),
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("outputs/anomaly_diagnostics"))
    args = ap.parse_args()
    root = args.root.resolve()
    result = {
        "protocol": "canonical v2 splits (60/20/20, seeds 0-4), probe pipeline and per-seed "
        "validation-selected layers reused from outputs/canonical_v2 and "
        "outputs/realistic_anomaly_v2; learning curve refits only the probe on a "
        "seeded subsample of the training split",
        "kinds": KINDS,
        "train_sizes": list(TRAIN_SIZES),
        "per_type_realistic": per_type(root),
        "learning_curve": {t: learning_curve(root, t) for t in ("canonical", "realistic")},
        "command": " ".join(sys.argv),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
