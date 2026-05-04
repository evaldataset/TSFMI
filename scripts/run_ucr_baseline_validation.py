"""UCR cross-domain validation of the baseline-controlled triviality finding (S4).

Applies our canonical non-model baselines (8-D hand-crafted, raw signal,
random projection, tsfresh-minimal) to a curated subset of UCR Archive
datasets that are most analogous to our anomaly / change-point / trend
synthetic concepts. Uses each UCR dataset's published train/test split
(no resplitting) to remain comparable with downstream-task TSFM numbers
in the literature.

Datasets included (anomaly-, change-, motion-relevant):
    - ECG200            (myocardial infarction binary anomaly)
    - ECGFiveDays       (5-day pulse anomaly)
    - Earthquakes       (binary anomaly: quake vs. no-quake)
    - Wafer             (manufacturing defect binary anomaly)
    - GunPoint          (motion: gun-point vs. point only)
    - TwoLeadECG        (ECG variant)

If hand-crafted / raw / tsfresh-minimal baselines saturate or come close
to published TSFM probing numbers on these datasets, the triviality
finding extends from synthetic to real-world.

Output: outputs/ucr_validation/results.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("outputs/ucr_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "ECG200",
    "ECGFiveDays",
    "Earthquakes",
    "Wafer",
    "GunPoint",
    "TwoLeadECG",
]

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000


def load_ucr_via_aeon(name: str) -> tuple[NDArray, NDArray, NDArray, NDArray] | None:
    """Load (X_train, y_train, X_test, y_test) via aeon-toolkit. Returns None on failure."""
    try:
        from aeon.datasets import load_classification
    except ImportError:
        print("  aeon-toolkit not installed; pip install aeon")
        return None
    try:
        X_tr, y_tr = load_classification(name=name, split="train")
        X_te, y_te = load_classification(name=name, split="test")
    except Exception as e:
        print(f"  aeon load failed for {name}: {e}")
        return None
    # X is shape (N, n_channels, L). For univariate UCR n_channels==1.
    if X_tr.ndim == 3 and X_tr.shape[1] == 1:
        X_tr = X_tr[:, 0, :]
        X_te = X_te[:, 0, :]
    elif X_tr.ndim != 2:
        print(f"  unsupported shape {X_tr.shape} for {name}")
        return None
    # Map string labels to int
    classes = np.unique(np.concatenate([y_tr, y_te]))
    cmap = {c: i for i, c in enumerate(classes)}
    y_tr_int = np.array([cmap[v] for v in y_tr], dtype=np.int64)
    y_te_int = np.array([cmap[v] for v in y_te], dtype=np.int64)
    return X_tr.astype(np.float64), y_tr_int, X_te.astype(np.float64), y_te_int


def load_ts(path: Path) -> tuple[NDArray, NDArray]:
    """Load a .ts (sktime/aeon) file.

    Falls back to .tsv for the older UCR format which is much simpler:
    each line is "<class>\\t<v1>\\t<v2>...<vL>".
    """
    if path.suffix == ".ts":
        # Minimal .ts parser — handles the univariate equal-length case used by UCR.
        rows: list[list[float]] = []
        labels: list[float] = []
        in_data = False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("@"):
                    if line.lower().startswith("@data"):
                        in_data = True
                    continue
                if not in_data:
                    continue
                # Format: "v1,v2,...,vL:label"
                if ":" in line:
                    series_str, lbl = line.rsplit(":", 1)
                else:
                    parts = line.split(",")
                    lbl = parts[-1]
                    series_str = ",".join(parts[:-1])
                rows.append([float(v) for v in series_str.split(",")])
                labels.append(float(lbl))
        return np.array(rows, dtype=np.float64), np.array(labels)
    elif path.suffix == ".tsv":
        arr = np.loadtxt(path, delimiter="\t")
        return arr[:, 1:].astype(np.float64), arr[:, 0]
    else:
        raise ValueError(path)


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


def random_projection(seq: NDArray, dim: int = 256, seed: int = 42) -> NDArray:
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((seq.shape[1], dim)).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    return seq @ proj


def tsfresh_minimal(seq: NDArray) -> NDArray | None:
    try:
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError:
        return None
    rows = []
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            rows.append({"id": i, "time": j, "value": float(seq[i, j])})
    df = pd.DataFrame(rows)
    feats = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=4,
    )
    feats = feats.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return feats.to_numpy()


def probe(
    Xtr: NDArray, ytr: NDArray, Xte: NDArray, yte: NDArray, seeds: list[int]
) -> tuple[float, float, float]:
    """Train LogisticRegression, return (mean, ci_low, ci_high)."""
    accs = []
    for seed in seeds:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(Xtr, ytr.astype(np.int64))
        accs.append(float(accuracy_score(yte.astype(np.int64), pipe.predict(Xte))))
    rng = np.random.default_rng(42)
    arr = np.array(accs)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    rows = []
    for name in DATASETS:
        loaded = load_ucr_via_aeon(name)
        if loaded is None:
            print(f"[{name}] SKIP: aeon load failed")
            continue
        Xtr_raw, ytr, Xte_raw, yte = loaded
        if Xtr_raw.shape[1] != Xte_raw.shape[1]:
            print(f"[{name}] SKIP: train/test length mismatch")
            continue
        n_classes = int(np.unique(ytr).size)
        print(
            f"[{name}] train={Xtr_raw.shape} test={Xte_raw.shape} classes={n_classes}"
        )

        # 1. raw signal
        m, lo, hi = probe(Xtr_raw, ytr, Xte_raw, yte, SEEDS)
        rows.append({"dataset": name, "baseline": "raw_signal", "mean": round(m, 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                     "n_train": int(len(ytr)), "n_test": int(len(yte)),
                     "seq_len": int(Xtr_raw.shape[1]), "n_classes": n_classes})
        print(f"  raw_signal       {m:.4f} [{lo:.4f}, {hi:.4f}]")

        # 2. hand-crafted 8-D
        Xtr_hc = hand_crafted_features(Xtr_raw)
        Xte_hc = hand_crafted_features(Xte_raw)
        m, lo, hi = probe(Xtr_hc, ytr, Xte_hc, yte, SEEDS)
        rows.append({"dataset": name, "baseline": "hand_crafted_8d", "mean": round(m, 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                     "n_train": int(len(ytr)), "n_test": int(len(yte)),
                     "seq_len": int(Xtr_raw.shape[1]), "n_classes": n_classes})
        print(f"  hand_crafted_8d  {m:.4f} [{lo:.4f}, {hi:.4f}]")

        # 3. random projection
        Xtr_rp = random_projection(Xtr_raw)
        Xte_rp = random_projection(Xte_raw)
        m, lo, hi = probe(Xtr_rp, ytr, Xte_rp, yte, SEEDS)
        rows.append({"dataset": name, "baseline": "random_projection", "mean": round(m, 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                     "n_train": int(len(ytr)), "n_test": int(len(yte)),
                     "seq_len": int(Xtr_raw.shape[1]), "n_classes": n_classes})
        print(f"  random_projection {m:.4f} [{lo:.4f}, {hi:.4f}]")

        # 4. tsfresh minimal (10 features)
        Xtr_tsf = tsfresh_minimal(Xtr_raw)
        Xte_tsf = tsfresh_minimal(Xte_raw)
        if Xtr_tsf is not None and Xte_tsf is not None:
            m, lo, hi = probe(Xtr_tsf, ytr, Xte_tsf, yte, SEEDS)
            rows.append({"dataset": name, "baseline": "tsfresh_minimal_10", "mean": round(m, 4),
                         "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                         "n_train": int(len(ytr)), "n_test": int(len(yte)),
                         "seq_len": int(Xtr_raw.shape[1]), "n_classes": n_classes})
            print(f"  tsfresh_minimal  {m:.4f} [{lo:.4f}, {hi:.4f}]")

    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} rows to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
