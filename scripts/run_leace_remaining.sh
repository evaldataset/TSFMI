#!/bin/bash
# LEACE erasure for moment_pca512, chronos, patchtst_pretrained
# These models use probe dirs WITHOUT 'synthetic_' prefix
set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
PYTHON=".venv/bin/python"

echo "============================================"
echo "  LEACE Erasure: Remaining Models"
echo "============================================"

MODELS=("moment_pca512" "chronos" "patchtst_pretrained")
PROPS=("trend" "seasonality" "frequency" "stationarity" "anomaly" "change_point")

for model in "${MODELS[@]}"; do
    for prop in "${PROPS[@]}"; do
        REPR_DIR="outputs/representations/${model}/synthetic_${prop}"
        PROBE_DIR="outputs/probes/${model}_${prop}_linear"
        OUTPUT_DIR="outputs/leace/${model}_${prop}"

        if [ ! -d "$REPR_DIR" ]; then
            echo "SKIP: ${model}/${prop} (no repr dir: $REPR_DIR)"
            continue
        fi
        if [ ! -d "$PROBE_DIR" ]; then
            echo "SKIP: ${model}/${prop} (no probe dir: $PROBE_DIR)"
            continue
        fi
        if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/leace_results.json" ]; then
            echo "SKIP (exists): $OUTPUT_DIR"
            continue
        fi

        echo "--- LEACE: ${model}/${prop} ---"
        $PYTHON scripts/run_leace_erasure.py \
            --representations_dir $REPR_DIR \
            --probe_dir $PROBE_DIR \
            --output_dir $OUTPUT_DIR
    done
done

echo ""
echo "=== LEACE Remaining Models Complete ==="
