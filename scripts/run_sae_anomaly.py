"""D2 real run (general): train a TopK SAE on any TSFM's anomaly representations and
test whether a kurtosis feature emerges. Used to check the dose-response prediction:
models with higher linear kurtosis-R^2 should yield stronger SAE kurtosis features.

Requires representations extracted via:
    scripts/extract_representations.py --model <m> --dataset synthetic_anomaly --layers all

Usage:
    PYTHONPATH=. python scripts/run_sae_anomaly.py --model chronos
    PYTHONPATH=. python scripts/run_sae_anomaly.py --model timesfm --layer-frac 0.75
Output:
    outputs/sae_anomaly/<model>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

from scripts.prototype_sae_anomaly import best_kurtosis_feature, train_sae
from src.datasets.synthetic import generate_anomaly_dataset

OUT_DIR = Path("outputs/sae_anomaly")
NUM_SAMPLES, SEQ_LEN, SEED = 1000, 512, 42


def _layer_files(model: str) -> list[str]:
    hits = glob.glob(f"outputs/representations/{model}*/synthetic_anomaly/*.pt")
    hits = [f for f in hits if "label" not in f.lower()]
    if not hits:
        raise FileNotFoundError(f"no rep files for model={model}; run extraction first")

    def idx(f: str) -> int:
        m = re.findall(r"(\d+)", Path(f).stem)
        return int(m[-1]) if m else 0

    return sorted(hits, key=idx)


def _kurtosis() -> np.ndarray:
    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)
    sig = ds.sequences
    std = sig.std(axis=1, keepdims=True) + 1e-10
    return np.mean(((sig - sig.mean(1, keepdims=True)) / std) ** 4, axis=1) - 3.0


def _run_layer(rep_file: str, kurtosis: np.ndarray, d_hidden: int, k: int) -> float:
    X = torch.load(rep_file, map_location="cpu").float()
    while X.ndim > 2:  # pool any patch/token/channel axes -> (N, D)
        X = X.mean(dim=1)
    _, feats = train_sae(X, d_hidden=d_hidden, k=k, epochs=400)
    _, r = best_kurtosis_feature(feats, kurtosis)
    return float(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    files = _layer_files(args.model)
    kurtosis = _kurtosis()
    per_layer = {}
    for f in files:
        stem = Path(f).stem
        X0 = torch.load(f, map_location="cpu")
        assert X0.shape[0] == len(kurtosis), (f, X0.shape, len(kurtosis))
        r = _run_layer(f, kurtosis, args.d_hidden, args.k)
        per_layer[stem] = round(r, 4)
        print(f"  {args.model:10} {stem:24} |corr(kurtosis)|={r:.3f}")

    best = max(per_layer.values())
    verdict = "encoded-but-entangled" if best > 0.5 else "weak/absent (evidence for non-encoded)"
    result = {
        "model": args.model,
        "sae": {"d_hidden": args.d_hidden, "k": args.k},
        "per_layer_abs_corr_kurtosis": per_layer,
        "best_abs_corr_kurtosis": round(best, 4),
        "verdict": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.model}.json").write_text(json.dumps(result, indent=2))
    print(f"  -> best={best:.3f}  {verdict}")


if __name__ == "__main__":
    main()
