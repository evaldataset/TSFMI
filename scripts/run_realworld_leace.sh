#!/bin/bash
# Real-world LEACE for ETTh1 and Weather (trend and stationarity)
# Requires: trained probes + extracted representations for real-world datasets
set -e

export PYTHONPATH=.
PYTHON=".venv/bin/python"
DEVICE="cpu"

MODELS=("moment_pca512" "chronos" "patchtst_pretrained" "gpt4ts_pca512")
DATASETS=("etth1" "weather")
PROPS=("trend" "stationarity")

for MODEL in "${MODELS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for PROP in "${PROPS[@]}"; do
            REPR_DIR="outputs/representations/${MODEL}/${DS}_${PROP}"
            PROBE_DIR="outputs/probes/${MODEL}_${DS}_${PROP}_linear"
            OUT_DIR="outputs/leace_realworld/${MODEL}_${DS}_${PROP}"

            if [ ! -d "$REPR_DIR" ]; then
                echo "SKIP $MODEL/$DS/$PROP (no repr)"
                continue
            fi
            if [ ! -d "$PROBE_DIR" ]; then
                echo "Training probe: $MODEL/$DS/$PROP"
                $PYTHON scripts/train_probe.py \
                    --representations_dir "$REPR_DIR" \
                    --probe_type linear \
                    --output_dir "$PROBE_DIR" \
                    --device "$DEVICE" \
                    --epochs 100 2>&1 | tail -1
            fi
            if [ -d "$OUT_DIR" ]; then
                echo "SKIP $MODEL/$DS/$PROP LEACE (exists)"
                continue
            fi

            echo "Running LEACE: $MODEL/$DS/$PROP"
            $PYTHON scripts/run_leace_erasure.py \
                --representations_dir "$REPR_DIR" \
                --probe_dir "$PROBE_DIR" \
                --output_dir "$OUT_DIR" \
                --device "$DEVICE" \
                2>&1 | tail -3
        done
    done
done

echo "Real-world LEACE complete."
