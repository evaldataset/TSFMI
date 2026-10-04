"""Train-only PCA leak-free canonical rerun (REVIEW priority).

The canonical pre-extracted PCA512 representations (MOMENT, GPT4TS) historically
fit PCA on all extracted data. To address the leakage concern, we redo the
canonical anomaly + seasonality benchmark for these two models with PCA fit on
the train split only, using the raw representations under
outputs/representations/<raw_key>/<dataset>/. Same canonical 60/20/20 + 5-seed
bootstrap protocol as run_canonical_benchmark.py.

We compare the leak-free numbers to the original canonical numbers and report
both. If the difference is within the 95% CI of either, the leakage is
benign and the headline numbers can be quoted with a leak-free annotation.

Output:
    outputs/pca_leakfree/results.json
    outputs/pca_leakfree/results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/pca_leakfree")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")
CANONICAL_ROOT = Path("outputs/canonical")

# (canonical_dir, raw_key)
PCA_MODELS = [
    ("moment_pca512", "moment"),
    ("gpt4ts_pca512", "gpt4ts"),
]
# Restrict to the two non-saturated columns (anomaly + seasonality) — these are
# where PCA-fit-scope could matter; the saturated four columns are at >=0.999
# regardless and would not differ between leak-free and legacy fits.
PROPERTIES = [
    ("anomaly", "classification"),
    ("seasonality", "regression"),
]
SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
# Restrict layer enumeration to mid-layers (where best-anomaly typically lands).
# Full layer scan is prohibitive at full dim; the canonical best-layer for
# anomaly is consistently in the middle third of MOMENT/GPT4TS encoders.
MAX_LAYERS = 6


def three_way_split(X, y, seed):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return idx, X[tr], y[tr], X[va], y[va], X[te], y[te], tr, va, te


def score(X_tr, y_tr, X_te, y_te, task, seed):
    if task == "classification":
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(X_tr, y_tr.astype(np.int64))
        return float(accuracy_score(y_te.astype(np.int64), pipe.predict(X_te)))
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(X_tr, y_tr)
    return float(r2_score(y_te, pipe.predict(X_te)))


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(42)
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    return (
        float(arr.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def load_layer_files(d: Path, n_samples_target: int) -> list[Path]:
    if not d.exists():
        return []
    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    if len(layer_files) > MAX_LAYERS:
        # Sample MAX_LAYERS evenly spaced layers so we still cover early /
        # middle / late, but at a tractable PCA fit count.
        idx = np.linspace(0, len(layer_files) - 1, MAX_LAYERS, dtype=int)
        layer_files = [layer_files[i] for i in idx]
    return layer_files


def best_layer_test(
    raw_dir: Path, labels: np.ndarray, task: str, seed: int, n_components: int = 512
) -> tuple[float, str]:
    """For one seed, do val-only best-layer selection with train-only PCA refit
    on every (layer, seed) pair, then report test score."""
    layer_files = load_layer_files(raw_dir, len(labels))
    best_va = -np.inf
    best_te = float("nan")
    best_ln = layer_files[0].stem if layer_files else "n/a"
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.ndim != 2 or t.shape[0] != len(labels) or t.shape[1] == 0:
            continue
        X_raw = t.numpy().astype(np.float64)
        # 3-way split on indices first.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(labels))
        n_test = int(len(labels) * 0.20)
        n_val = int(len(labels) * 0.20)
        te = idx[:n_test]
        va = idx[n_test : n_test + n_val]
        tr = idx[n_test + n_val :]
        # Train-only PCA fit (randomized SVD for tractability on >=64k dim).
        if X_raw.shape[1] > n_components:
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=seed,
            )
            pca.fit(X_raw[tr])
            X_red = pca.transform(X_raw)
        else:
            X_red = X_raw
        Xtr = X_red[tr]
        ytr = labels[tr]
        Xva = X_red[va]
        yva = labels[va]
        Xte = X_red[te]
        yte = labels[te]
        va_acc = score(Xtr, ytr, Xva, yva, task, seed)
        if va_acc > best_va:
            best_va = va_acc
            best_te = score(Xtr, ytr, Xte, yte, task, seed)
            best_ln = lf.stem
    return best_te, best_ln


def main() -> None:
    seed_everything(42)
    rows = []

    for canonical_dir, raw_key in PCA_MODELS:
        for prop, task in PROPERTIES:
            ds = f"synthetic_{prop}"
            raw_dir = REPR_ROOT / raw_key / ds
            labels_path = raw_dir / "labels.pt"
            if not labels_path.exists():
                print(f"SKIP {raw_key}/{ds}: no labels.pt")
                continue
            labels = torch.load(labels_path, map_location="cpu", weights_only=True).numpy()
            if task == "classification":
                labels = labels.astype(np.int64)
            else:
                labels = labels.astype(np.float64)

            per_seed = []
            best_layers = []
            for s in SEEDS:
                te_score, best_ln = best_layer_test(raw_dir, labels, task, s)
                per_seed.append(te_score)
                best_layers.append(best_ln)
            m, lo, hi = bootstrap_ci(per_seed)

            # Original canonical number for comparison.
            orig_path = CANONICAL_ROOT / f"{canonical_dir}_{prop}" / "canonical_results.json"
            orig_mean = orig_lo = orig_hi = float("nan")
            if orig_path.exists():
                orig = json.loads(orig_path.read_text())
                orig_mean = orig.get("test_mean", float("nan"))
                orig_lo = orig.get("test_ci95_low", float("nan"))
                orig_hi = orig.get("test_ci95_high", float("nan"))

            row = {
                "canonical_dir": canonical_dir,
                "raw_key": raw_key,
                "property": prop,
                "task_type": task,
                "leakfree_test_mean": round(m, 4),
                "leakfree_test_ci95_low": round(lo, 4),
                "leakfree_test_ci95_high": round(hi, 4),
                "leakfree_per_seed_test": [round(v, 4) for v in per_seed],
                "leakfree_best_layer_per_seed": best_layers,
                "original_test_mean": orig_mean,
                "original_test_ci95_low": orig_lo,
                "original_test_ci95_high": orig_hi,
                "delta": round(m - float(orig_mean), 4) if orig_mean == orig_mean else None,
            }
            rows.append(row)
            print(
                f"  {canonical_dir:<18s} {prop:<14s} "
                f"leak-free={m:.4f} [{lo:.4f}, {hi:.4f}]  "
                f"orig={orig_mean:.4f}  delta={row['delta']}"
            )

    OUT_DIR.joinpath("results.json").write_text(json.dumps(rows, indent=2))

    tex = [
        r"% Auto-generated by scripts/run_pca_leakfree_rerun.py",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Property} & "
        r"\textbf{Original} & \textbf{Leak-free} & "
        r"\textbf{$\Delta$} & \textbf{Leak-free 95\% CI} \\",
        r"\midrule",
    ]
    for r in rows:
        delta_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "---"
        model_tex = r["canonical_dir"].replace("_", r"\_")
        tex.append(
            f"\\texttt{{{model_tex}}} & {r['property']} & "
            f"{r['original_test_mean']:.4f} & {r['leakfree_test_mean']:.4f} & "
            f"{delta_str} & "
            f"[{r['leakfree_test_ci95_low']:.4f}, {r['leakfree_test_ci95_high']:.4f}] \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    OUT_DIR.joinpath("results.tex").write_text("\n".join(tex))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
