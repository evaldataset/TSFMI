"""Window-level anomaly data from TSB-AD-U (Liu & Paparrizos, NeurIPS 2024 D&B), for replication.

Builds the same window task as scripts/run_ucr_anomaly_archive.py from the TSB-AD-U univariate
collection, excluding sources that are synthetic (TODS, MGAB), mixed real/synthetic (YAHOO), or
already used (UCR, i.e. the UCR Anomaly Archive). Per series: the part after the anomaly-free
training prefix (``tr_<n>`` in the file name) is cut into windows of 512 steps; positives overlap a
labelled anomaly segment by at least half of min(segment length, 512) (at most eight per series,
evenly spaced), negatives overlap no anomaly (as many as positives, seeded sample). Every window is
z-normalised; the series is the group for series-level splits. The store mirrors the UCR store
(windows.npz, hand_crafted.npy, rocket.npy, tsfresh_efficient.npy, repr/<model>/), plus a negated
copy for the transformation-sensitivity check.

Usage:
    TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_tsbad_windows.py --part data \
        --root <TSB-AD-U directory>
    ... run_tsbad_windows.py --part extract --model patchtst_fm  (each model; --negate: copy)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

L = 512
MAX_POS = 8
EXCLUDE = {"UCR", "TODS", "MGAB", "YAHOO"}
NAME = re.compile(r"^(\d+)_([A-Za-z0-9-]+)_id_(\d+)_(.+)_tr_(\d+)_1st_(\d+)\.csv$")
BASE = Path(os.environ.get("TSFMI_RESOLUTION_STORE", "outputs/resolution_store"))


def store(negate: bool = False) -> Path:
    return BASE / ("tsbad_negate" if negate else "tsbad")


def segments(label: NDArray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of contiguous runs of label == 1."""
    d = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def build(root: Path) -> dict[str, NDArray]:
    import pandas as pd

    rng = np.random.default_rng(0)
    xs, ys, groups, sources = [], [], [], []
    for f in sorted(root.glob("*.csv")):
        m = NAME.match(f.name)
        if not m or m.group(2) in EXCLUDE:
            continue
        df = pd.read_csv(f)
        x = df["Data"].to_numpy(np.float64)
        lab = df["Label"].to_numpy()
        train_end = int(m.group(5))
        starts = np.arange(train_end, len(x) - L + 1)
        segs = [(a, b) for a, b in segments(lab) if b >= train_end]
        if len(starts) == 0 or not segs:
            continue
        overlap_any = np.zeros(len(starts))
        pos_mask = np.zeros(len(starts), bool)
        for a, b in segs:
            ov = np.clip(np.minimum(starts + L, b + 1) - np.maximum(starts, a), 0, None)
            overlap_any += ov
            pos_mask |= ov >= max(1, (min(b - a + 1, L) + 1) // 2)
        pos, neg = starts[pos_mask], starts[overlap_any == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos = pos[np.linspace(0, len(pos) - 1, min(MAX_POS, len(pos))).round().astype(int)]
        neg = rng.choice(neg, size=min(len(pos), len(neg)), replace=False)
        for s, y in [(p, 1) for p in pos] + [(q, 0) for q in neg]:
            w = x[s : s + L]
            xs.append((w - w.mean()) / max(w.std(), 1e-8))
            ys.append(y)
            groups.append(int(m.group(1)))
            sources.append(m.group(2))
    return {
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "group": np.asarray(groups),
        "source": np.asarray(sources),
    }


def main() -> int:
    from scripts.run_canonical_baselines import hand_crafted_features
    from scripts.run_resolution_study import rocket_parallel, tsfresh_features
    from scripts.run_ucr_anomaly_archive import extract

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("data", "extract"), required=True)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--negate", action="store_true")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    if args.part == "data":
        d = build(args.root)
        for neg in (False, True):
            out = store(neg)
            out.mkdir(parents=True, exist_ok=True)
            x = -d["x"] if neg else d["x"]
            np.savez(out / "windows.npz", x=x, y=d["y"], group=d["group"], source=d["source"])
            np.save(out / "hand_crafted.npy", hand_crafted_features(x).astype(np.float32))
            np.save(out / "rocket.npy", rocket_parallel(x, args.jobs))
            np.save(out / "tsfresh_efficient.npy", tsfresh_features(x, args.jobs))
        src, cnt = np.unique(d["source"], return_counts=True)
        per_source = dict(zip(src.tolist(), cnt.tolist(), strict=True))
        print(
            f"{len(d['y'])} windows, {len(np.unique(d['group']))} series, positive rate "
            f"{d['y'].mean():.3f}; windows per source: {per_source}",
            flush=True,
        )
        return 0
    with np.load(store(args.negate) / "windows.npz") as z:
        x = z["x"]
    extract(args.model, x, store(args.negate) / "repr" / args.model)
    print("extracted", args.model, "negate" if args.negate else "", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
