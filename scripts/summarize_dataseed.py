"""Collect the anomaly results across generator (data) seeds 42, 43, 44.

Seed 42 is the canonical dataset (outputs/canonical_v2, outputs/realistic_anomaly_v2); seeds 43
and 44 come from ``DATA_SEED=<s> scripts/run_canonical_v2.sh`` with ``PROPS_OVERRIDE`` set to the
canonical and realistic anomaly datasets (outputs/dataseed_v2/seed<s>). The raw-signal and
hand-crafted controls are recomputed here for every data seed with the canonical split function
and probe.

Usage:
    PYTHONPATH=. python scripts/summarize_dataseed.py --out outputs/dataseed_v2/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.run_anomaly_diagnostics import MODELS, canonical_dir
from scripts.run_canonical_baselines import hand_crafted_features
from scripts.run_canonical_benchmark import SEEDS, three_way_split, train_and_score
from scripts.run_realistic_anomaly import realistic_anomaly_dataset
from src.datasets.synthetic import generate_anomaly_dataset

DATA_SEEDS = (42, 43, 44)
TASKS = {"canonical": "anomaly", "realistic": "anomaly_realistic"}


def model_result(root: Path, task: str, model: str, data_seed: int) -> float:
    if data_seed == 42:
        d = canonical_dir(root, task, model)
    else:
        base = root / f"outputs/dataseed_v2/seed{data_seed}"
        cands = [
            p
            for p in base.iterdir()
            if p.name
            in (
                f"{model}_{TASKS[task]}",
                f"{model}_pca512_{TASKS[task]}",
                f"{model}_meanpool_{TASKS[task]}",
            )
        ]
        if len(cands) != 1:
            raise SystemExit(f"{task}/{model}/seed{data_seed}: expected one dir, found {cands}")
        d = cands[0]
    return float(json.loads((d / "canonical_results.json").read_text())["test_mean"])


def control_result(task: str, control: str, data_seed: int) -> float:
    if task == "realistic":
        x, y = realistic_anomaly_dataset(1000, 512, data_seed)
    else:
        ds = generate_anomaly_dataset(1000, 512, seed=data_seed)
        x, y = ds.sequences, ds.labels
    X = hand_crafted_features(x) if control == "hand_crafted" else x
    scores = []
    for seed in SEEDS:
        X_tr, y_tr, _, _, X_te, y_te = three_way_split(X, y, seed)
        scores.append(train_and_score(X_tr, y_tr, X_te, y_te, "classification", seed))
    return round(float(np.mean(scores)), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("outputs/dataseed_v2/summary.json"))
    args = ap.parse_args()
    root = args.root.resolve()
    tasks: dict[str, dict] = {}
    claims: dict[str, dict] = {}
    for task in TASKS:
        rows: dict[str, dict] = {}
        for row in [*MODELS, "hand_crafted", "raw_signal"]:
            vals = {
                s: (
                    model_result(root, task, row, s)
                    if row in MODELS
                    else control_result(task, row, s)
                )
                for s in DATA_SEEDS
            }
            rows[row] = {
                "per_data_seed": vals,
                "range": round(max(vals.values()) - min(vals.values()), 4),
            }
            print(f"{task:<9} {row:<13} " + " ".join(f"{s}:{v:.3f}" for s, v in vals.items()))
        hc = rows["hand_crafted"]["per_data_seed"]
        above = {m: sum(rows[m]["per_data_seed"][s] > hc[s] for s in DATA_SEEDS) for m in MODELS}
        claims[task] = {
            "nseeds": len(DATA_SEEDS),
            "nmodels_below_hc_every_seed": sum(v == 0 for v in above.values()),
            "nmodels_above_hc_every_seed": sum(v == len(DATA_SEEDS) for v in above.values()),
            "timesfm_seeds_above_hc": above["timesfm"],
            "max_model_range": max(rows[m]["range"] for m in MODELS),
        }
        tasks[task] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "data_seeds": list(DATA_SEEDS),
                "tasks": tasks,
                "claims": claims,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    print(json.dumps(claims, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
