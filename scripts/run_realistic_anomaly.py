"""Realistic anomaly generator + canonical baseline comparison (CHECK.md G6).

Builds an anomaly task that does NOT trivially decompose into kurtosis/max:
  - subtle magnitudes (1.5--3 sigma instead of 5);
  - multiple anomaly types (point, level shift, variance change, contextual);
  - structured background (seasonality + AR(1) drift);
  - some "ambiguous" series where 0 anomalies are present but background
    happens to spike >2 sigma a few times -- a baseline that uses max(|x|)
    will mis-classify these.

Reports the same canonical 60/20/20 + 5-seed bootstrap protocol on three
baselines (HC 8-D, raw signal, random projection 256). The TSFM-side number
remains the previously-extracted canonical anomaly value, since re-extracting
all 7 models on a new dataset is out of the rebuttal window. We document this
upper-bound interpretation in the output JSON.

Output:
    outputs/realistic_anomaly/realistic_anomaly_results.json
    outputs/realistic_anomaly/realistic_anomaly_results.tex
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

from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/realistic_anomaly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
NUM_SAMPLES = 1000
SEQ_LEN = 512


def make_background(rng: np.random.Generator, t: NDArray) -> NDArray:
    """Seasonality + AR(1) drift + small Gaussian noise."""
    period = int(rng.choice([16, 32, 48, 64, 96, 128]))
    amp = rng.uniform(0.5, 1.8)
    phase = rng.uniform(0, 2 * np.pi)
    season = amp * np.sin(2 * np.pi * t / period + phase)
    # AR(1)
    rho = rng.uniform(0.6, 0.95)
    eps = rng.normal(0, 0.3, size=len(t))
    ar = np.zeros_like(t, dtype=np.float64)
    for i in range(1, len(t)):
        ar[i] = rho * ar[i - 1] + eps[i]
    return season + ar


def realistic_anomaly_dataset(num_samples: int, seq_len: int, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len, dtype=np.float64)
    sequences: list[NDArray] = []
    labels: list[int] = []
    half = num_samples // 2
    for _ in range(half):
        # Class 0: ambiguous-normal: realistic background, NO injected anomaly,
        # but background may itself show extreme transient excursions.
        sequences.append(make_background(rng, t))
        labels.append(0)
    for _ in range(num_samples - half):
        # Class 1: subtle anomaly of one of 4 types injected on the realistic background.
        y = make_background(rng, t)
        kind = int(rng.integers(4))
        sigma = float(np.std(y))
        if kind == 0:
            # Point anomaly with magnitude in [1.5, 3] sigma (subtle).
            pos = int(rng.integers(seq_len))
            y[pos] += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 3.0) * sigma
        elif kind == 1:
            # Level shift over 5--40 steps with magnitude 1--2 sigma.
            n = int(rng.integers(5, 41))
            start = int(rng.integers(0, seq_len - n))
            y[start : start + n] += (
                rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 2.0) * sigma
            )
        elif kind == 2:
            # Variance change in a contiguous segment.
            n = int(rng.integers(20, 80))
            start = int(rng.integers(0, seq_len - n))
            y[start : start + n] *= rng.uniform(1.5, 2.5)
        else:
            # Contextual outlier: replace a small segment by phase-shifted seasonality.
            n = int(rng.integers(10, 30))
            start = int(rng.integers(0, seq_len - n))
            y[start : start + n] = rng.uniform(0.5, 1.5) * np.sin(
                2 * np.pi * t[start : start + n] / float(rng.choice([8, 12, 24]))
                + rng.uniform(0, 2 * np.pi)
            )
        sequences.append(y)
        labels.append(1)
    perm = rng.permutation(num_samples)
    seqs = np.asarray(sequences, dtype=np.float64)[perm]
    labs = np.asarray(labels, dtype=np.int64)[perm]
    return seqs, labs


def hand_crafted_features(seq: NDArray) -> NDArray:
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


def random_projection(seq, dim=256, seed=42):
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((seq.shape[1], dim)).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    return seq @ proj


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


def main() -> None:
    seed_everything(42)
    sequences, labels = realistic_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=42)

    baselines = {
        "hand_crafted": hand_crafted_features(sequences),
        "raw_signal": sequences,
        "random_projection": random_projection(sequences, seed=42),
    }

    rows = []
    for bl, feats in baselines.items():
        per_seed = []
        for s in SEEDS:
            Xtr, ytr, _, _, Xte, yte = three_way_split(feats, labels, s)
            per_seed.append(score_lr(Xtr, ytr, Xte, yte, s))
        m, lo, hi = bootstrap_ci(per_seed)
        rows.append(
            {
                "baseline": bl,
                "test_mean": round(m, 4),
                "test_ci95_low": round(lo, 4),
                "test_ci95_high": round(hi, 4),
                "per_seed_test": [round(v, 4) for v in per_seed],
            }
        )
        print(f"  {bl:20s} test={m:.4f} [{lo:.4f}, {hi:.4f}]")

    out = {
        "task": "realistic_anomaly",
        "design": (
            "Mixed-type anomaly (point, level-shift, variance, contextual) on "
            "structured background (seasonality + AR(1) + noise); subtle "
            "magnitudes (1.5--3 sigma)."
        ),
        "n_samples": NUM_SAMPLES,
        "seq_len": SEQ_LEN,
        "rows": rows,
        "note": (
            "TSFM probes are not re-run here; the corresponding canonical "
            "anomaly TSFM scores are read from outputs/canonical/summary.json. "
            "If HC accuracy on this realistic task drops below the canonical "
            "TSFM ceiling (0.753), the inversion narrows; if HC still beats "
            "TSFM, the inversion holds under the harder construction."
        ),
    }
    OUT_DIR.joinpath("realistic_anomaly_results.json").write_text(json.dumps(out, indent=2))

    tex = [
        r"% Auto-generated by scripts/run_realistic_anomaly.py",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Baseline} & \textbf{Test acc} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['baseline'].replace('_', '-')} & {r['test_mean']:.3f} & "
            f"[{r['test_ci95_low']:.3f}, {r['test_ci95_high']:.3f}] \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    OUT_DIR.joinpath("realistic_anomaly_results.tex").write_text("\n".join(tex))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
