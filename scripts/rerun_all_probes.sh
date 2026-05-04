#!/bin/bash
# Re-run all probe training with proper train/val/test 3-way split.
# Usage: bash scripts/rerun_all_probes.sh <GPU_ID> <MODEL_LIST>
# Example: bash scripts/rerun_all_probes.sh 0 "moment_pca512 chronos"

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID> <MODEL_LIST>}"
MODEL_LIST="${2:?Usage: $0 <GPU_ID> <MODEL_LIST>}"

REPR_ROOT="outputs/representations"
OUT_ROOT="outputs/probes_v2"
PROPERTIES="synthetic_trend synthetic_seasonality synthetic_frequency synthetic_stationarity synthetic_anomaly synthetic_change_point"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
PYTHON="PYTHONPATH=. .venv/bin/python"

echo "=== GPU $GPU_ID: Re-running probes for: $MODEL_LIST ==="
echo "=== Output: $OUT_ROOT ==="
echo "=== Split: 80/10/10 (train/val/test) ==="
echo ""

for MODEL in $MODEL_LIST; do
    for PROP in $PROPERTIES; do
        REPR_DIR="$REPR_ROOT/$MODEL/$PROP"
        if [ ! -d "$REPR_DIR" ]; then
            echo "SKIP: $REPR_DIR not found"
            continue
        fi

        # Linear probe
        OUT_DIR="$OUT_ROOT/${MODEL}_${PROP}_linear"
        echo "[GPU $GPU_ID] Training linear probe: $MODEL / $PROP"
        eval $PYTHON scripts/train_probe.py \
            --representations_dir "$REPR_DIR" \
            --probe_type linear \
            --val_split 0.1 \
            --test_split 0.1 \
            --epochs 100 \
            --output_dir "$OUT_DIR" 2>&1 | tail -1

        # MLP control probe
        OUT_DIR="$OUT_ROOT/${MODEL}_${PROP}_mlp_control"
        echo "[GPU $GPU_ID] Training MLP control probe: $MODEL / $PROP"
        eval $PYTHON scripts/train_probe.py \
            --representations_dir "$REPR_DIR" \
            --probe_type mlp_control \
            --val_split 0.1 \
            --test_split 0.1 \
            --epochs 100 \
            --output_dir "$OUT_DIR" 2>&1 | tail -1
    done
done

echo ""
echo "=== GPU $GPU_ID: All probes complete ==="
