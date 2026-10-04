#!/usr/bin/env bash
# Single-cell reproduction (make reproduce-one-cell).
#
# Reproduces ONE (canonical_model, property) cell of the canonical 60/20/20 +
# 5-seed + bootstrap protocol end-to-end:
#   1) extract raw representations for the chosen model/dataset
#   2) reduce to the standardized canonical view (PCA512 / mean-pool / identity)
#   3) run the canonical benchmark probe and emit canonical_results.json
#   4) print the resulting test_mean / 95% CI alongside the matched HC baseline
#
# Defaults: PatchTST-Pre on synthetic_anomaly (CPU-feasible, ~10-20 min on a CPU,
# <2 min on GPU). Override via:
#   ONLY_MODEL=chronos ONLY_DATASET=synthetic_anomaly bash scripts/run_reproduce_one_cell.sh
#
# Output:
#   outputs/canonical/<canonical_model>_<property>/canonical_results.json
#   outputs/canonical_baselines/<property>/canonical_results.json (if missing)

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then VENV_PY="python"; fi

ONLY_MODEL="${ONLY_MODEL:-patchtst_pretrained}"
ONLY_DATASET="${ONLY_DATASET:-synthetic_anomaly}"

# Map dataset -> property + task type.
case "$ONLY_DATASET" in
    synthetic_trend)         PROP="trend";        TASK="classification";;
    synthetic_seasonality)   PROP="seasonality";  TASK="regression";;
    synthetic_frequency)     PROP="frequency";    TASK="classification";;
    synthetic_stationarity)  PROP="stationarity"; TASK="classification";;
    synthetic_anomaly)       PROP="anomaly";      TASK="classification";;
    synthetic_change_point)  PROP="change_point"; TASK="classification";;
    *) echo "ERROR: unknown dataset $ONLY_DATASET" >&2; exit 2;;
esac

echo "=== Reproduce-one-cell: $ONLY_MODEL / $ONLY_DATASET ($TASK) ==="

# Step 1: extraction + reduction.
ONLY_MODEL="$ONLY_MODEL" ONLY_DATASET="$ONLY_DATASET" \
    bash scripts/extract_all_canonical.sh

# Step 2: canonical probe.
out_dir="outputs/canonical/${ONLY_MODEL}_${PROP}"
mkdir -p "$out_dir"
PYTHONPATH=. "$VENV_PY" scripts/run_canonical_benchmark.py \
    --representations_dir "outputs/representations/${ONLY_MODEL}/${ONLY_DATASET}" \
    --property "$PROP" \
    --task_type "$TASK" \
    --output_dir "$out_dir"

# Step 3: ensure baseline-control row exists (cheap: CPU only).
if [[ ! -f "outputs/canonical_baselines/${PROP}/canonical_results.json" ]]; then
    PYTHONPATH=. "$VENV_PY" scripts/run_canonical_baselines.py
fi

# Step 4: report.
"$VENV_PY" -c "
import json
m = json.load(open('outputs/canonical/${ONLY_MODEL}_${PROP}/canonical_results.json'))
b = json.load(open('outputs/canonical_baselines/${PROP}/canonical_results.json'))
print()
print('Canonical model probe: ${ONLY_MODEL} on ${PROP}')
print(f\"  test mean = {m['test_mean']:.4f}  95% CI = [{m['test_ci95_low']:.4f}, {m['test_ci95_high']:.4f}]\")
print('Matched non-model baselines (same protocol):')
for r in b:
    print(f\"  {r['baseline']:<20s}  test mean = {r['test_mean']:.4f}  95% CI = [{r['test_ci95_low']:.4f}, {r['test_ci95_high']:.4f}]\")
"

echo ""
echo "=== Reproduce-one-cell complete ==="
