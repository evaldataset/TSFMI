"""D2 real run: train a TopK SAE on MOMENT anomaly representations and test whether a
kurtosis feature emerges (encoded-but-entangled vs. non-encoded).

Requires representations extracted via:
    scripts/extract_representations.py --model moment --dataset synthetic_anomaly --layers 11,17,23

Usage:
    PYTHONPATH=. python scripts/run_sae_moment_anomaly.py --layer 17
Output:
    outputs/sae_moment_anomaly/results.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from scripts.prototype_sae_anomaly import best_kurtosis_feature, train_sae
from src.datasets.synthetic import generate_anomaly_dataset

OUT_DIR = Path("outputs/sae_moment_anomaly")
NUM_SAMPLES, SEQ_LEN, SEED = 1000, 512, 42


def _find_layer_file(layer: int) -> str:
    pats = [
        f"outputs/representations/moment*/synthetic_anomaly/*{layer}*.pt",
        f"outputs/representations/moment*/synthetic_anomaly/layer_{layer}.pt",
    ]
    for p in pats:
        hits = [f for f in glob.glob(p) if "label" not in f.lower()]
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f"no rep file for layer {layer}; run extraction first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--d-hidden", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    rep_file = _find_layer_file(args.layer)
    X = torch.load(rep_file, map_location="cpu").float()
    if X.ndim == 3:  # (N, tokens, D) -> mean-pool tokens
        X = X.mean(dim=1)
    print(f"loaded {rep_file}  shape={tuple(X.shape)}")

    # Regenerate the exact anomaly signals (same seed/order) to get per-sample kurtosis.
    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)
    sig = ds.sequences
    std = sig.std(axis=1, keepdims=True) + 1e-10
    kurtosis = np.mean(((sig - sig.mean(1, keepdims=True)) / std) ** 4, axis=1) - 3.0
    assert X.shape[0] == len(kurtosis), (X.shape, len(kurtosis))

    _, feats = train_sae(X, d_hidden=args.d_hidden, k=args.k, epochs=400)
    idx, r = best_kurtosis_feature(feats, kurtosis)
    verdict = "encoded-but-entangled" if r > 0.5 else "weak/absent (evidence for non-encoded)"
    result = {
        "model": "moment",
        "layer": args.layer,
        "rep_shape": list(X.shape),
        "sae": {"d_hidden": args.d_hidden, "k": args.k},
        "best_kurtosis_feature_idx": int(idx),
        "abs_corr_kurtosis": round(float(r), 4),
        "verdict": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
