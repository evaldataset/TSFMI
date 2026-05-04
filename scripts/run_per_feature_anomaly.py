"""Per-feature attribution for the anomaly inversion (CHECK.md G1).

Trains the canonical sklearn LogisticRegression on each individual hand-crafted
feature, plus on each "leave-one-out" subset, plus on a small grid that mirrors
the headline 8-D vector. The goal is to refute the reviewer line "the inversion
is just kurtosis" by quantifying *which* HC features carry the signal.

Reuses the same 60/20/20 + 5-seed + bootstrap protocol as
scripts/run_canonical_baselines.py.

Output:
    outputs/per_feature_anomaly/per_feature_results.json
    outputs/per_feature_anomaly/per_feature_results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.synthetic import generate_anomaly_dataset
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/per_feature_anomaly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
NUM_SAMPLES = 1000
SEQ_LEN = 512

FEATURE_NAMES = [
    "slope",
    "residual_std",
    "mean",
    "std",
    "kurtosis",
    "argmax_fft",
    "spectral_entropy",
    "diff_std",
]


def hand_crafted_features(seq: NDArray[np.float64]) -> NDArray[np.float64]:
    """Same 8-D HC vector used by run_canonical_baselines."""
    n, L = seq.shape
    t = np.arange(L, dtype=np.float64) - L / 2
    tv = (t**2).sum()
    feats = np.zeros((n, 8))
    for i in range(n):
        x = seq[i]
        xc = x - x.mean()
        slope = (t * xc).sum() / tv
        feats[i, 0] = slope
        feats[i, 1] = np.std(xc - slope * t)
        feats[i, 2] = x.mean()
        feats[i, 3] = x.std()
        std = x.std()
        feats[i, 4] = np.mean(((x - x.mean()) / (std + 1e-10)) ** 4) - 3
        fft = np.abs(np.fft.rfft(xc))
        fft[0] = 0
        feats[i, 5] = float(np.argmax(fft))
        power = fft**2
        ps = power.sum()
        if ps > 1e-10:
            p = power / ps
            p = p[p > 0]
            feats[i, 6] = -np.sum(p * np.log(p + 1e-12))
        feats[i, 7] = np.std(np.diff(x))
    return feats


def max_abs_feature(seq: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.abs(seq).max(axis=1, keepdims=True)


def three_way_split(
    X: NDArray[np.float64], y: NDArray[np.int64], seed: int
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def score_classifier(
    X_tr: NDArray, y_tr: NDArray, X_te: NDArray, y_te: NDArray, seed: int
) -> float:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    )
    pipe.fit(X_tr, y_tr.astype(np.int64))
    return float(accuracy_score(y_te.astype(np.int64), pipe.predict(X_te)))


def bootstrap_ci(
    values: list[float], n_bootstrap: int = N_BOOTSTRAP, ci: float = 0.95
) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return (
        float(arr.mean()),
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


def evaluate_subset(
    features: NDArray, labels: NDArray, name: str
) -> dict:
    per_seed_test: list[float] = []
    for seed in SEEDS:
        Xtr, ytr, _, _, Xte, yte = three_way_split(features, labels, seed)
        per_seed_test.append(score_classifier(Xtr, ytr, Xte, yte, seed))
    mean, lo, hi = bootstrap_ci(per_seed_test)
    return {
        "subset": name,
        "n_features": int(features.shape[1]),
        "test_mean": round(mean, 4),
        "test_ci95_low": round(lo, 4),
        "test_ci95_high": round(hi, 4),
        "per_seed_test": [round(v, 4) for v in per_seed_test],
    }


def main() -> None:
    seed_everything(42)
    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=42)
    sequences = ds.sequences
    labels = ds.labels.astype(np.int64)

    feats_full = hand_crafted_features(sequences)
    max_abs = max_abs_feature(sequences)

    rows: list[dict] = []

    # Per-feature: each HC feature on its own.
    for j, fname in enumerate(FEATURE_NAMES):
        rows.append(evaluate_subset(feats_full[:, j : j + 1], labels, f"only_{fname}"))

    # Single-feature trivial controls.
    rows.append(evaluate_subset(max_abs, labels, "max_abs_only"))

    # Leave-one-out: drop each feature individually.
    for j, fname in enumerate(FEATURE_NAMES):
        keep = [k for k in range(8) if k != j]
        rows.append(evaluate_subset(feats_full[:, keep], labels, f"drop_{fname}"))

    # Two-feature ablations of the obvious culprits.
    rows.append(
        evaluate_subset(
            feats_full[:, [4, 7]], labels, "kurtosis+diff_std"
        )
    )
    rows.append(
        evaluate_subset(
            feats_full[:, [3, 4]], labels, "std+kurtosis"
        )
    )

    # Full 8-D reference.
    rows.append(evaluate_subset(feats_full, labels, "full_8D"))

    # Drop kurtosis AND max-related: the "no obvious anomaly statistic" control.
    rows.append(
        evaluate_subset(
            feats_full[:, [0, 1, 2, 5, 6, 7]],
            labels,
            "drop_kurtosis_and_std",
        )
    )

    out_json = OUT_DIR / "per_feature_results.json"
    out_json.write_text(json.dumps(rows, indent=2))

    # LaTeX summary
    tex = [
        r"% Auto-generated by scripts/run_per_feature_anomaly.py",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Subset} & \textbf{$|F|$} & \textbf{Test acc} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"\\texttt{{{r['subset']}}} & {r['n_features']} & "
            f"{r['test_mean']:.3f} & "
            f"[{r['test_ci95_low']:.3f}, {r['test_ci95_high']:.3f}] \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "per_feature_results.tex").write_text("\n".join(tex))

    print(f"\n{'Subset':<32s} {'|F|':>5s} {'mean':>8s} {'95% CI':>22s}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['subset']:<32s} {r['n_features']:>5d} "
            f"{r['test_mean']:>8.3f} "
            f"[{r['test_ci95_low']:.3f}, {r['test_ci95_high']:.3f}]"
        )
    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    main()
