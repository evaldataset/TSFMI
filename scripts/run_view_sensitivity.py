"""How much does the extraction view change the anomaly result? (MOMENT and GPT4TS)

The canonical v2 protocol probes MOMENT and GPT4TS through a PCA-512 of the flattened per-token
activations, fit inside each seed's training fold, because only these two models keep per-token
tensors. This script re-probes the same stored tensors under three token-pooling views -- mean over
tokens, max over tokens, and the last token -- with the unchanged protocol: 60/20/20 splits, split
seeds 0-4, layer selected on validation per seed, StandardScaler + LogisticRegression, bootstrap CI.
The PCA view is read from the canonical v2 results, not recomputed.

Usage:
    PYTHONPATH=. python scripts/run_view_sensitivity.py --out outputs/view_sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from scripts.run_anomaly_diagnostics import canonical_dir
from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, three_way_split, train_and_score

MODELS = ("moment", "gpt4ts")
TASKS = {"canonical": "synthetic_anomaly", "realistic": "synthetic_anomaly_realistic"}
VIEWS = {
    "mean": lambda t: t.mean(dim=1),
    "max": lambda t: t.amax(dim=1),
    "last": lambda t: t[:, -1, :],
}


def pooled_layers(d: Path, view: str) -> list[tuple[str, NDArray[np.float64]]]:
    files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    out = []
    for f in files:
        t = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        if t.ndim != 3:
            raise SystemExit(f"{f}: expected (N, tokens, d), got {tuple(t.shape)}")
        out.append((f.stem, VIEWS[view](t.float()).numpy().astype(np.float64)))
    return out


def probe(layers: list[tuple[str, NDArray]], y: NDArray) -> dict:
    tests, chosen = [], []
    for seed in SEEDS:
        best_val, best_test, best_layer = -np.inf, float("nan"), ""
        for name, X in layers:
            X_tr, y_tr, X_va, y_va, X_te, y_te = three_way_split(X, y, seed)
            val = train_and_score(X_tr, y_tr, X_va, y_va, "classification", seed)
            if val > best_val:
                best_val, best_layer = val, name
                best_test = train_and_score(X_tr, y_tr, X_te, y_te, "classification", seed)
        tests.append(best_test)
        chosen.append(best_layer)
    mean, lo, hi = bootstrap_ci(tests)
    return {
        "test_mean": round(mean, 4),
        "test_ci95": [round(lo, 4), round(hi, 4)],
        "per_seed": [round(v, 4) for v in tests],
        "best_layer_per_seed": chosen,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("outputs/view_sensitivity"))
    args = ap.parse_args()
    root = args.root.resolve()
    rows: dict[str, dict] = {}
    for task, dataset in TASKS.items():
        for m in MODELS:
            res = json.loads((canonical_dir(root, task, m) / "canonical_results.json").read_text())
            d = Path(res["representations_dir"])
            if d.name != dataset:
                raise SystemExit(f"{task}/{m}: unexpected representations dir {d}")
            y = torch.load(d / "labels.pt", map_location="cpu", weights_only=True).numpy()
            row = {
                "pca512_canonical": {
                    "test_mean": res["test_mean"],
                    "test_ci95": [res["test_ci95_low"], res["test_ci95_high"]],
                }
            }
            for view in VIEWS:
                row[view] = probe(pooled_layers(d, view), y)
            rows[f"{task}/{m}"] = row
            print(
                f"{task:<9} {m:<7} " + " ".join(f"{v}={row[v]['test_mean']:.3f}" for v in row),
                flush=True,
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "protocol": __doc__.split("Usage:")[0].strip(),
                "rows": rows,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
