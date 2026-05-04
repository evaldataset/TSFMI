#!/usr/bin/env bash
# Small-footprint smoke reproduction for reviewers (CHECK.md G7 / smoke target).
#
# Reproduces *one* (model, property) cell from a pre-extracted representation
# directory in <10 minutes on a CPU. Verifies the canonical 60/20/20 protocol,
# the canonical baselines, and the per-feature anomaly attribution.
#
# Usage:
#   make smoke
# or:
#   bash scripts/run_smoke.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then VENV_PY="python"; fi

echo "=== Smoke reproduction ==="
echo "Working dir: $ROOT"
echo "Python:     $($VENV_PY --version)"

# Stage 1: canonical baselines on synthetic anomaly (CPU only).
mkdir -p outputs/canonical_baselines
PYTHONPATH=. "$VENV_PY" -c "
from scripts.run_canonical_baselines import (
    DATASETS, NUM_SAMPLES, SEQ_LEN, SEEDS,
    hand_crafted_features, random_projection,
    three_way_split, score, bootstrap_ci,
)
from src.datasets.synthetic import generate_anomaly_dataset
ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=42)
feats = hand_crafted_features(ds.sequences)
per_seed = [
    score(*three_way_split(feats, ds.labels, s)[:2],
          *three_way_split(feats, ds.labels, s)[4:6],
          'classification', s)
    for s in SEEDS
]
m, lo, hi = bootstrap_ci(per_seed)
print(f'  HC anomaly  test={m:.4f} [{lo:.4f}, {hi:.4f}]  per-seed={per_seed}')
"

# Stage 2: per-feature anomaly attribution.
PYTHONPATH=. "$VENV_PY" scripts/run_per_feature_anomaly.py | tail -20

# Stage 3: pytest sanity.
"$VENV_PY" -m pytest tests/ -q -x -m "not slow" --tb=line | tail -5

echo ""
echo "=== Smoke reproduction complete ==="
