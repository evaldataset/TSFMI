#!/bin/bash
# Extract synthetic representations for new model wrappers (autoformer, timesnet, fedformer)
# Properties: trend, seasonality, frequency, stationarity, anomaly, change_point

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3

MODELS=("autoformer" "timesnet" "fedformer")
DATASETS=("synthetic_trend" "synthetic_seasonality" "synthetic_frequency" "synthetic_stationarity" "synthetic_anomaly" "synthetic_change_point")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "===== $model / $dataset ====="
        if [ -f "outputs/representations/$model/$dataset/metadata.json" ]; then
            echo "  SKIP (exists)"
            continue
        fi
        python scripts/extract_representations.py \
            --model "$model" \
            --dataset "$dataset" \
            --output_dir outputs/representations/ \
            --num_samples 1000 \
            --batch_size 32 \
            2>&1 | tail -3
        echo "  DONE"
    done
done

echo "===== NEW MODEL EXTRACTION COMPLETE ====="
