"""Linear recoverability of anomaly-task statistics from TSFM representations (canonical v2).

For the canonical anomaly generator the Bayes-optimal statistic is the likelihood ratio
LLR(x) = log mean_t cosh(5 x_t) - 12.5; max|x| is nearly as good and excess kurtosis is a weaker
proxy (reports/bayes_ceiling_check in the private revision notes; outputs/per_feature_anomaly).
This script asks how well a ridge probe recovers each statistic from each model's representation,
under the canonical v2 protocol: n = 1000, 60/20/20 split, seeds 0-4, layer selected on the
validation R^2 of the target, test R^2 reported, PCA-512 inside the pipeline for MOMENT and GPT4TS.
It then correlates recoverability with each model's canonical v2 anomaly accuracy.

Usage:
    PYTHONPATH=. python scripts/run_statistic_recoverability.py \
        --repr-root outputs/representations_v2 \
        --canonical-root outputs/canonical_v2 --out outputs/statistic_recoverability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import kurtosis, spearmanr

from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, three_way_split, train_and_score
from src.datasets.synthetic import generate_anomaly_dataset

# model key -> (representation subdirectory, PCA components inside the pipeline or None)
MODELS: dict[str, tuple[str, int | None]] = {
    "moment": ("raw/moment", 512),
    "gpt4ts": ("raw/gpt4ts", 512),
    "chronos": ("chronos_meanpool", None),
    "timer": ("timer_meanpool", None),
    "timesfm": ("timesfm_meanpool", None),
    "moirai": ("moirai_meanpool", None),
    "patchtst_fm": ("patchtst_fm_meanpool", None),
}
NUM_SAMPLES, SEQ_LEN, DATA_SEED, SPIKE = 1000, 512, 42, 5.0


def targets(x: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Per-window statistics of the input: Bayes LLR, max |x|, excess kurtosis."""
    return {
        "llr": np.log(np.cosh(SPIKE * x).mean(axis=1)) - SPIKE**2 / 2,
        "max_abs": np.abs(x).max(axis=1),
        "kurtosis": kurtosis(x, axis=1, fisher=True),
    }


def load_layers(d: Path) -> list[tuple[str, NDArray[np.float64]]]:
    files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    out = []
    for f in files:
        t = torch.load(f, map_location="cpu", weights_only=True)
        t = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
        out.append((f.stem, t.numpy().astype(np.float64)))
    return out


def recoverability(
    layers: list[tuple[str, NDArray[np.float64]]], y: NDArray[np.float64], pca: int | None
) -> dict:
    per_seed, best_layers = [], []
    for seed in SEEDS:
        best_val, best_test, best_layer = -np.inf, float("nan"), ""
        for name, X in layers:
            X_tr, y_tr, X_va, y_va, X_te, y_te = three_way_split(X, y, seed)
            val = train_and_score(X_tr, y_tr, X_va, y_va, "regression", seed, pca_components=pca)
            if val > best_val:
                best_val = val
                best_test = train_and_score(
                    X_tr, y_tr, X_te, y_te, "regression", seed, pca_components=pca
                )
                best_layer = name
        per_seed.append(best_test)
        best_layers.append(best_layer)
    mean, lo, hi = bootstrap_ci(per_seed)
    return {"test_r2_mean": round(mean, 4), "test_r2_ci95": [round(lo, 4), round(hi, 4)],
            "per_seed": [round(v, 4) for v in per_seed], "best_layer_per_seed": best_layers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repr-root", type=Path, required=True)
    ap.add_argument("--canonical-root", type=Path, default=Path("outputs/canonical_v2"))
    ap.add_argument("--out", type=Path, default=Path("outputs/statistic_recoverability"))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--combine", nargs="*", type=Path, default=None,
                    help="Merge per-model results.json files (from separate --models runs) and "
                         "compute the cross-model correlation instead of running probes.")
    args = ap.parse_args()

    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
    stats = targets(ds.sequences)
    rows: dict[str, dict] = {}
    commands: list[str] = []
    for f in args.combine or []:
        part = json.loads(f.read_text())
        rows.update(part["models"])
        commands.append(part["command"])
    for m in [] if args.combine else args.models:
        sub, pca = MODELS[m]
        d = args.repr_root / sub / "synthetic_anomaly"
        labels = torch.load(d / "labels.pt", map_location="cpu", weights_only=True).numpy()
        if not np.array_equal(labels, ds.labels):
            raise SystemExit(f"{m}: stored labels do not match the regenerated dataset order")
        layers = load_layers(d)
        canon = json.loads((args.canonical_root / f"{m}_anomaly" / "canonical_results.json")
                           .read_text())
        rows[m] = {"anomaly_accuracy": canon["test_mean"], "n_layers": len(layers),
                   "pca_in_pipeline": pca}
        for name, y in stats.items():
            rows[m][name] = recoverability(layers, y, pca)
            print(f"{m:<12} {name:<9} R2={rows[m][name]['test_r2_mean']:+.3f}", flush=True)

    corr = {}
    for name in stats:
        x = [rows[m][name]["test_r2_mean"] for m in rows]
        acc = [rows[m]["anomaly_accuracy"] for m in rows]
        rho, p = spearmanr(x, acc)
        corr[name] = {"spearman_rho": round(float(rho), 4), "p_value": round(float(p), 4),
                      "n_models": len(rows)}
        print(f"spearman(R2[{name}], anomaly acc) = {rho:+.3f} (p={p:.3f}, n={len(rows)})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({
        "protocol": "canonical v2: n=1000, seed 42, 60/20/20, split seeds 0-4, layer selected on "
                    "validation R^2 of the target, test R^2, Ridge(alpha=1) after StandardScaler; "
                    "PCA-512 inside the pipeline for moment/gpt4ts",
        "targets": {"llr": "log mean_t cosh(5 x_t) - 12.5 (Bayes statistic)",
                    "max_abs": "max_t |x_t|", "kurtosis": "excess kurtosis (Fisher)"},
        "command": " ".join(sys.argv),
        "merged_from": commands,
        "models": rows, "spearman_with_anomaly_accuracy": corr,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
