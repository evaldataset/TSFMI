"""SAE robustness sweep (D2): does the "kurtosis is weakly encoded" conclusion hold
across SAE hyperparameters? For each model's best-kurtosis layer, sweep dictionary
width and sparsity and report the maximum kurtosis-feature correlation over the grid.

If even the best config stays below ~0.5 for a model, weak/entangled encoding is robust
(not a hyperparameter artifact). Uses the prototype TopK SAE (not the SAELens library);
the grid mirrors common SAE width/sparsity choices.

Usage:  PYTHONPATH=. python scripts/run_sae_sweep.py
Output: outputs/sae_sweep/results.json
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

from scripts.prototype_sae_anomaly import best_kurtosis_feature, train_sae
from src.datasets.synthetic import generate_anomaly_dataset

OUT_DIR = Path("outputs/sae_sweep")
NUM, LEN, SEED = 1000, 512, 42
WIDTHS = (512, 1024, 2048)
KS = (8, 32)
MODELS = ("moment", "chronos", "timesfm", "gpt4ts", "timer", "moirai")


def _kurtosis() -> np.ndarray:
    ds = generate_anomaly_dataset(NUM, LEN, seed=SEED)
    sig = ds.sequences
    std = sig.std(1, keepdims=True) + 1e-10
    return np.mean(((sig - sig.mean(1, keepdims=True)) / std) ** 4, axis=1) - 3.0


def _best_layer_file(model: str) -> str:
    """Pick the layer that gave the highest kurtosis |r| in the default SAE run."""
    j = Path(f"outputs/sae_anomaly/{model}.json")
    files = {re.sub(r"\D", "", Path(f).stem) or Path(f).stem: f
             for f in glob.glob(f"outputs/representations/{model}*/synthetic_anomaly/*.pt")
             if "label" not in f.lower()}
    if j.exists():
        per = json.loads(j.read_text())["per_layer_abs_corr_kurtosis"]
        best_stem = max(per, key=per.get)
        key = re.sub(r"\D", "", best_stem) or best_stem
        if key in files:
            return files[key]
    return sorted(files.values())[len(files) // 2]  # fallback: middle layer


def main() -> None:
    kurt = _kurtosis()
    rows = []
    for m in MODELS:
        f = _best_layer_file(m)
        X = torch.load(f, map_location="cpu").float()
        while X.ndim > 2:  # pool any patch/token/channel axes -> (N, D)
            X = X.mean(dim=1)
        grid = {}
        best = (-1.0, None)
        for w in WIDTHS:
            for k in KS:
                _, feats = train_sae(X, d_hidden=w, k=k, epochs=400)
                _, r = best_kurtosis_feature(feats, kurt)
                grid[f"w{w}_k{k}"] = round(float(r), 4)
                if r > best[0]:
                    best = (float(r), f"w{w}_k{k}")
        row = {"model": m, "layer": Path(f).stem, "grid": grid,
               "max_abs_corr": round(best[0], 4), "best_config": best[1],
               "crosses_0.5": bool(best[0] > 0.5)}
        rows.append(row)
        print(f"{m:9} layer={Path(f).stem:20} max|r|={best[0]:.3f} @ {best[1]} "
              f"{'(>0.5)' if best[0] > 0.5 else ''}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    n_cross = sum(r["crosses_0.5"] for r in rows)
    print(f"\nmodels crossing |r|>0.5 in any config: {n_cross}/{len(rows)}")
    print(f"saved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
