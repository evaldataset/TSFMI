"""Compare a freshly reproduced canonical v2 cell with the committed result.

Usage (make reproduce-one-cell runs this):
    PYTHONPATH=. python scripts/compare_one_cell.py --model chronos \
        --prop synthetic_anomaly:anomaly:classification
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prop", required=True, help="dataset:property:task, as PROPS_OVERRIDE")
    ap.add_argument("--fresh", type=Path, default=Path("outputs/one_cell"))
    ap.add_argument("--committed", type=Path, default=Path("outputs/canonical_v2"))
    ap.add_argument("--tol", type=float, default=0.005, help="largest accepted |difference|")
    args = ap.parse_args()
    prop = args.prop.split(":")[1]
    cell = f"{args.model}_{prop}/canonical_results.json"
    fresh = json.loads((args.fresh / cell).read_text())
    committed = json.loads((args.committed / cell).read_text())
    diff = fresh["test_mean"] - committed["test_mean"]
    print(
        f"{args.model}/{prop}: reproduced {fresh['test_mean']:.4f}, committed "
        f"{committed['test_mean']:.4f}, difference {diff:+.4f} (tolerance {args.tol})"
    )
    print(f"  per seed reproduced {fresh['test_score_per_seed']}")
    print(f"  per seed committed  {committed['test_score_per_seed']}")
    return 0 if abs(diff) <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
