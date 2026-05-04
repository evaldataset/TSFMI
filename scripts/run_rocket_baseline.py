"""ROCKET / MiniROCKET baseline on anomaly + frequency_hard (CHECK.md G3).

Self-contained MiniROCKET-style random-convolutional baseline. We avoid an extra
dependency by implementing a small fixed-kernel random-convolutional feature
transform: K random kernels with PPV (proportion of positive values) pooling,
which mimics MiniROCKET's headline transform.

Same canonical 60/20/20 + 5-seed + bootstrap protocol as
run_canonical_baselines.py. Compares against the hand-crafted baseline.

Output:
    outputs/rocket_baselines/rocket_results.json
    outputs/rocket_baselines/rocket_results.tex
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

from src.datasets.synthetic import (
    generate_anomaly_dataset,
    generate_anomaly_hard_dataset,
    generate_change_point_hard_dataset,
    generate_frequency_dataset,
    generate_frequency_hard_dataset,
    generate_stationarity_hard_dataset,
    generate_trend_hard_dataset,
)
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/rocket_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
NUM_SAMPLES = 1000
SEQ_LEN = 512
N_KERNELS = 1024


def random_kernels(
    seed: int, num_kernels: int, kernel_len_choices=(7, 9, 11)
) -> list[tuple[NDArray, int, int]]:
    """Generate random kernels (weights, dilation, padding)."""
    rng = np.random.default_rng(seed)
    kernels = []
    for _ in range(num_kernels):
        klen = int(rng.choice(kernel_len_choices))
        weights = rng.standard_normal(klen)
        weights -= weights.mean()
        max_dil = int(np.log2((SEQ_LEN - 1) / (klen - 1) + 1)) if klen > 1 else 0
        dil = int(2 ** rng.integers(0, max_dil + 1))
        pad = (klen - 1) * dil // 2
        kernels.append((weights, dil, pad))
    return kernels


def apply_kernel(
    seq: NDArray, weights: NDArray, dilation: int, padding: int
) -> NDArray[np.float64]:
    """1D convolution with dilation, returns conv output."""
    n, L = seq.shape
    klen = len(weights)
    eff = (klen - 1) * dilation + 1
    out_len = L + 2 * padding - eff + 1
    if out_len <= 0:
        return np.zeros((n, 1))
    padded = np.pad(seq, ((0, 0), (padding, padding)))
    out = np.zeros((n, out_len), dtype=np.float64)
    for k in range(klen):
        out += weights[k] * padded[:, k * dilation : k * dilation + out_len]
    return out


def rocket_features(seq: NDArray, kernels: list) -> NDArray[np.float64]:
    """ROCKET-style features: per-kernel PPV (proportion of positive values) +
    max — 2 features per kernel, same as MiniROCKET's headline."""
    n = seq.shape[0]
    feats = np.zeros((n, len(kernels) * 2), dtype=np.float64)
    for i, (w, dil, pad) in enumerate(kernels):
        out = apply_kernel(seq, w, dil, pad)
        feats[:, 2 * i] = (out > 0).mean(axis=1)
        feats[:, 2 * i + 1] = out.max(axis=1)
    return feats


def three_way_split(X, y, seed):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def score_lr(Xtr, ytr, Xte, yte, seed):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    )
    pipe.fit(Xtr, ytr.astype(np.int64))
    return float(accuracy_score(yte.astype(np.int64), pipe.predict(Xte)))


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95):
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


DATASETS = {
    "anomaly": generate_anomaly_dataset,
    "anomaly_hard": generate_anomaly_hard_dataset,
    "frequency": generate_frequency_dataset,
    "frequency_hard": generate_frequency_hard_dataset,
    "trend_hard": generate_trend_hard_dataset,
    "change_point_hard": generate_change_point_hard_dataset,
    "stationarity_hard": generate_stationarity_hard_dataset,
}


def main() -> None:
    seed_everything(42)
    rows = []
    kernels = random_kernels(seed=42, num_kernels=N_KERNELS)
    print(f"Built {N_KERNELS} ROCKET-like kernels.")

    for ds_name, gen_fn in DATASETS.items():
        print(f"\n--- {ds_name} ---")
        ds = gen_fn(NUM_SAMPLES, SEQ_LEN, seed=42)
        X = rocket_features(ds.sequences, kernels)
        per_seed = []
        for seed in SEEDS:
            Xtr, ytr, _, _, Xte, yte = three_way_split(X, ds.labels, seed)
            per_seed.append(score_lr(Xtr, ytr, Xte, yte, seed))
        mean, lo, hi = bootstrap_ci(per_seed)
        rows.append(
            {
                "dataset": ds_name,
                "n_kernels": N_KERNELS,
                "test_mean": round(mean, 4),
                "test_ci95_low": round(lo, 4),
                "test_ci95_high": round(hi, 4),
                "per_seed_test": [round(v, 4) for v in per_seed],
            }
        )
        print(f"  {ds_name:20s} test={mean:.4f} [{lo:.4f}, {hi:.4f}]")

    OUT_DIR.joinpath("rocket_results.json").write_text(json.dumps(rows, indent=2))

    tex = [
        r"% Auto-generated by scripts/run_rocket_baseline.py",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{ROCKET test acc} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['dataset']} & {r['test_mean']:.3f} & "
            f"[{r['test_ci95_low']:.3f}, {r['test_ci95_high']:.3f}] \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    OUT_DIR.joinpath("rocket_results.tex").write_text("\n".join(tex))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
