"""Label-efficiency summaries: label-equivalence ratios with bootstrap intervals.

Label-equivalence at budget n: the strong-union learning curve (mean accuracy per budget, linear in
log budget between grid points, made monotone) is inverted at the accuracy of a representation at n;
ratio = n' / n. If the representation exceeds the union's best budget the ratio is right-censored.
Uncertainty: bootstrap over the five series-level splits (UCR Anomaly Archive and TSB-AD-U windows)
or over datasets (k-shot classification), 2,000 draws, percentile 95% interval.

Usage:
    PYTHONPATH=. python scripts/summarize_label_efficiency.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = ("moment", "chronos", "gpt4ts", "timer", "timesfm", "moirai", "patchtst_fm", "mantis")
SHAPES = {
    "Fish",
    "HandOutlines",
    "Herring",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
}


def invert(curve_n: np.ndarray, curve_acc: np.ndarray, target: float) -> tuple[float, bool]:
    """Budget at which the (monotonised) curve reaches target; (value, censored)."""
    acc = np.maximum.accumulate(curve_acc)
    if target <= acc[0]:
        return float(curve_n[0]), False
    if target > acc[-1]:
        return float(curve_n[-1]), True
    j = int(np.searchsorted(acc, target))
    x0, x1 = np.log(curve_n[j - 1]), np.log(curve_n[j])
    a0, a1 = acc[j - 1], acc[j]
    t = 0.0 if a1 == a0 else (target - a0) / (a1 - a0)
    return float(np.exp(x0 + t * (x1 - x0))), False


def label_equiv(
    rows: list[dict],
    unit: str,
    budget_key: str,
    union_key: str,
    model_keys: list[str],
    n_boot: int = 2000,
) -> dict:
    budgets = sorted({r[budget_key] for r in rows})
    units = sorted({r[unit] for r in rows})
    cell = defaultdict(list)
    for r in rows:
        cell[(r[unit], r[budget_key])].append(r)

    def curves(sample: list) -> dict:
        out = {}
        for key in [union_key, *model_keys]:
            out[key] = np.array(
                [np.mean([r[key] for u in sample for r in cell[(u, b)]]) for b in budgets]
            )
        return out

    def ratios(c: dict) -> dict:
        res = {}
        for m in model_keys:
            res[m] = []
            for i, b in enumerate(budgets):
                n_eq, cens = invert(np.array(budgets, float), c[union_key], c[m][i])
                res[m].append((n_eq / b, cens))
        return res

    point = ratios(curves(units))
    rng = np.random.default_rng(0)
    boots = defaultdict(lambda: defaultdict(list))
    for _ in range(n_boot):
        sample = list(rng.choice(units, size=len(units), replace=True))
        for m, rs in ratios(curves(sample)).items():
            for i, (v, _) in enumerate(rs):
                boots[m][i].append(v)
    out = {
        "budgets": budgets,
        "union_curve": curves(units)[union_key].round(4).tolist(),
        "models": {},
    }
    for m in model_keys:
        out["models"][m] = [
            {
                "budget": b,
                "ratio": round(point[m][i][0], 3),
                "censored": point[m][i][1],
                "lo": round(float(np.quantile(boots[m][i], 0.025)), 3),
                "hi": round(float(np.quantile(boots[m][i], 0.975)), 3),
                "acc": round(float(curves(units)[m][i]), 4),
            }
            for i, b in enumerate(budgets)
        ]
    return out


def main() -> None:
    out: dict = {"anomaly": {}, "classification": {}}
    for st in ("none", "negate", "reverse", "tsbad", "tsbad_negate"):
        f = Path(f"outputs/label_efficiency/{st}.json")
        if f.exists():
            rows = json.loads(f.read_text())
            keys = [k for k in rows[0] if k in MODELS or k.startswith("union+")]
            out["anomaly"][st] = label_equiv(rows, "split", "n_series", "strong_union", keys)
    f = Path("outputs/ucr_kshot/results.json")
    if f.exists():
        d = json.loads(f.read_text())
        rows = d["rows"]
        keys = [k for k in rows[0] if k in MODELS or k.startswith("union+")]
        out["classification"]["all"] = label_equiv(rows, "dataset", "k", "union", keys)
        ts = [r for r in rows if r["dataset"] not in SHAPES]
        out["classification"]["time_series_only"] = label_equiv(ts, "dataset", "k", "union", keys)
        out["classification"]["n_datasets"] = len(d["datasets"])
        out["classification"]["n_time_series"] = len({r["dataset"] for r in ts})
    Path("outputs/label_efficiency/summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
