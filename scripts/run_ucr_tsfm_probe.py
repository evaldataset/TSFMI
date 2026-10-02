"""Real UCR anomaly probing with TSFMs vs non-model baselines (Phase-1 / D1).

Extends run_real_anomaly_baselines.py by adding frozen-TSFM probes on the real UCR
anomaly tasks, under the official leak-free train/test split. UCR series are resized
to length 512 (linear interpolation) to match the wrappers' context. Layer selection
uses a validation split carved from TRAIN only (no test leakage); the reported score
is best-layer test ROC-AUC.

Usage:
    PYTHONPATH=. python scripts/run_ucr_tsfm_probe.py --models moment chronos
Output:
    outputs/ucr_tsfm_probe/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_per_feature_anomaly import hand_crafted_features
from scripts.run_realistic_anomaly_tsfm import DEVICE, load_wrapper
from src.datasets.ucr_anomaly import ANOMALY_DATASETS, load_ucr_anomaly
from src.extractors.hook_manager import HookManager
from src.metrics.tsad_metrics import window_threshold_free_metrics

OUT_DIR = Path("outputs/ucr_tsfm_probe")
SEED = 42
CTX = 512


def resize_to(seqs: NDArray, length: int = CTX) -> NDArray[np.float64]:
    """Linear-interpolate each series to a fixed length."""
    x_new = np.linspace(0.0, 1.0, length)
    out = np.zeros((len(seqs), length), dtype=np.float64)
    for i, s in enumerate(seqs):
        x_old = np.linspace(0.0, 1.0, len(s))
        out[i] = np.interp(x_new, x_old, s)
    return out


def extract_pooled(wrapper, seqs: NDArray) -> dict[str, NDArray]:
    """Return {layer_name: (N, D)} token-mean-pooled activations for the sequences."""
    layers = wrapper.get_layer_names()
    acc: dict[str, list[NDArray]] = {ln: [] for ln in layers}
    t = torch.tensor(seqs, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    with HookManager(wrapper.model, layers) as hm:
        for s in range(0, t.shape[0], 32):
            with torch.no_grad():
                wrapper.forward(t[s : s + 32])
            a = hm.get_activations()
            for ln in layers:
                x = a[ln]
                if x.ndim == 4:
                    x = x.mean(dim=2)
                if x.ndim == 3:
                    x = x.mean(dim=1)
                acc[ln].append(x.detach().cpu().numpy())
    return {ln: np.concatenate(v, axis=0).astype(np.float64) for ln, v in acc.items()}


def _probe(Xtr, ytr, Xte):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs"),
    )
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


def tsfm_score(model: str, task) -> float:
    wrapper = load_wrapper(model)
    wrapper.load(device=DEVICE)
    wrapper.freeze()
    n_tr = task.train_sequences.shape[0]
    allseq = resize_to(np.concatenate([task.train_sequences, task.test_sequences], 0))
    feats = extract_pooled(wrapper, allseq)

    ytr, yte = task.train_labels, task.test_labels
    # inner val split from TRAIN only for leak-free layer selection
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n_tr)
    n_val = max(10, int(n_tr * 0.25))
    vtr, vva = idx[n_val:], idx[:n_val]

    best_layer, best_val = None, -np.inf
    cache: dict[str, tuple[NDArray, NDArray]] = {}
    for ln, X in feats.items():
        Xtr_all, Xte = X[:n_tr], X[n_tr:]
        if Xtr_all.shape[1] > 512:  # PCA fit on train only
            ncomp = min(512, Xtr_all.shape[1], len(vtr) - 1)
            pca = PCA(n_components=ncomp, random_state=SEED).fit(Xtr_all[vtr])
            Xtr_all, Xte = pca.transform(Xtr_all), pca.transform(Xte)
        cache[ln] = (Xtr_all, Xte)
        proba_va = _probe(Xtr_all[vtr], ytr[vtr], Xtr_all[vva])
        m = window_threshold_free_metrics(proba_va, ytr[vva])
        if not np.isnan(m["roc_auc"]) and m["roc_auc"] > best_val:
            best_val, best_layer = m["roc_auc"], ln

    Xtr_all, Xte = cache[best_layer]
    proba_te = _probe(Xtr_all, ytr, Xte)
    return float(window_threshold_free_metrics(proba_te, yte)["roc_auc"])


def baseline_score(kind: str, task) -> float:
    if kind == "hand_crafted":
        Xtr = hand_crafted_features(task.train_sequences)
        Xte = hand_crafted_features(task.test_sequences)
    else:  # raw_signal on resized series
        Xtr = resize_to(task.train_sequences)
        Xte = resize_to(task.test_sequences)
    proba = _probe(Xtr, task.train_labels, Xte)
    return float(window_threshold_free_metrics(proba, task.test_labels)["roc_auc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["moment", "chronos"])
    ap.add_argument("--datasets", nargs="+", default=list(ANOMALY_DATASETS))
    ap.add_argument("--out", type=Path, default=OUT_DIR,
                    help="Output directory; results.json there is overwritten.")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.datasets:
        task = load_ucr_anomaly(name)
        row = {"dataset": name, "hand_crafted": round(baseline_score("hand_crafted", task), 4),
               "raw_signal": round(baseline_score("raw_signal", task), 4)}
        for m in args.models:
            try:
                row[m] = round(tsfm_score(m, task), 4)
            except Exception as e:  # pragma: no cover
                row[m] = f"ERR {type(e).__name__}"
        rows.append(row)
        print(name, row)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"saved -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
