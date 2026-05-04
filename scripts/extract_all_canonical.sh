#!/usr/bin/env bash
# Full extraction pipeline for the 7 confirmatory models x 6 synthetic properties.
#
# Idempotent: skips any (model, dataset) whose representation directory already
# contains layer .pt files. Distributes work across GPUs 0/1/2 round-robin.
#
# Usage:
#   make extract-representations
# or:
#   bash scripts/extract_all_canonical.sh

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then VENV_PY="python"; fi
REPR_ROOT="outputs/representations"
mkdir -p "$REPR_ROOT"

MODELS=(
    "moment_pca512:moment"
    "chronos:chronos"
    "patchtst_pretrained:patchtst_pretrained"
    "gpt4ts_pca512:gpt4ts"
    "timer_meanpool:timer"
    "timesfm_meanpool:timesfm"
    "moirai_meanpool:moirai"
)
DATASETS=(
    "synthetic_trend"
    "synthetic_seasonality"
    "synthetic_frequency"
    "synthetic_stationarity"
    "synthetic_anomaly"
    "synthetic_change_point"
)

i=0
for model_pair in "${MODELS[@]}"; do
    model_dir="${model_pair%%:*}"
    model_arg="${model_pair##*:}"
    for ds in "${DATASETS[@]}"; do
        out="$REPR_ROOT/$model_dir/$ds"
        if [[ -d "$out" ]] && ls "$out"/*.pt &>/dev/null; then
            echo "SKIP: $out already populated"
            continue
        fi
        gpu=$(( i % 3 ))
        i=$(( i + 1 ))
        echo "EXTRACT: $model_arg $ds -> $out (GPU $gpu)"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. "$VENV_PY" \
            scripts/extract_representations.py \
            --model "$model_arg" \
            --dataset "$ds" \
            --layers all \
            --output_dir "$REPR_ROOT" \
            2>&1 | tail -5 || true
    done
done

echo ""
echo "=== Extraction complete ==="
du -sh "$REPR_ROOT"
