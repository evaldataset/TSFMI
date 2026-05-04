#!/bin/bash
# Multi-seed probing for key experiments (seeds 42, 123, 456)
# Only re-trains probes — representations are already extracted.
set -e

export PYTHONPATH=.
PYTHON=".venv/bin/python"
DEVICE="cuda:1"

MODELS=("moment_pca512" "chronos" "patchtst_pretrained" "gpt4ts_pca512")
PROPS=("synthetic_trend" "synthetic_frequency" "synthetic_stationarity" "synthetic_anomaly" "synthetic_change_point")
SEEDS=(123 456)

for SEED in "${SEEDS[@]}"; do
    echo "========== SEED=$SEED =========="
    for MODEL in "${MODELS[@]}"; do
        for PROP in "${PROPS[@]}"; do
            REPR_DIR="outputs/representations/${MODEL}/${PROP}"
            OUT_DIR="outputs/probes_seed${SEED}/${MODEL}_${PROP}_linear"
            if [ ! -d "$REPR_DIR" ]; then
                echo "SKIP $MODEL/$PROP (no repr)"
                continue
            fi
            if [ -d "$OUT_DIR" ]; then
                echo "SKIP $MODEL/$PROP seed=$SEED (exists)"
                continue
            fi
            echo "Training $MODEL/$PROP seed=$SEED"
            $PYTHON scripts/train_probe.py \
                --representations_dir "$REPR_DIR" \
                --probe_type linear \
                --output_dir "$OUT_DIR" \
                --seed "$SEED" \
                --device "$DEVICE" \
                --epochs 100 \
                --verbose 2>&1 | tail -1
        done
    done
done

echo "Multi-seed probing complete."
