"""Bayes-optimal accuracy of the canonical anomaly task under the canonical protocol.

The canonical anomaly generator draws x ~ N(0, I_512) and, for the positive class, adds one spike
of +-5 at a uniform position. Its exact likelihood ratio is LLR(x) = log mean_t cosh(5 x_t) - 12.5,
and the Bayes rule (equal priors) predicts an anomaly iff LLR > 0. The rule needs no fitting; it is
scored on the same test splits (60/20/20, split seeds 0-4) as every probe, with the same bootstrap
CI.

Usage:
    PYTHONPATH=. python scripts/run_bayes_ceiling.py --out outputs/bayes_ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.run_canonical_benchmark import SEEDS, bootstrap_ci, three_way_split
from src.datasets.synthetic import generate_anomaly_dataset

NUM_SAMPLES, SEQ_LEN, DATA_SEED, SPIKE = 1000, 512, 42, 5.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/bayes_ceiling"))
    args = ap.parse_args()
    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
    llr = np.log(np.cosh(SPIKE * ds.sequences).mean(axis=1)) - SPIKE**2 / 2
    accs = []
    for seed in SEEDS:
        _, _, _, _, s_te, y_te = three_way_split(llr, ds.labels, seed)
        accs.append(float(((s_te > 0).astype(int) == y_te).mean()))
    mean, lo, hi = bootstrap_ci(accs)
    print(f"Bayes rule (LLR > 0): {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "generator": "normal: x ~ N(0, I_512); anomaly: same plus one spike of +-5 at "
                "a uniform position (generate_anomaly_dataset(1000, 512, seed=42))",
                "statistic": "LLR(x) = log mean_t cosh(5 x_t) - 12.5; rule: anomaly iff LLR > 0",
                "bayes_rule": {
                    "mean": round(mean, 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "per_seed": [round(a, 4) for a in accs],
                },
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
