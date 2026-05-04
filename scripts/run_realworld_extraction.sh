#!/bin/bash
# Extract real-world representations for moirai, timesfm, timer
# Datasets: etth1, weather, electricity, traffic, exchange_rate
# Properties: trend, stationarity, seasonality, seasonality_binary, change_point

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3

MODELS=("moirai" "timesfm" "timer")
DATASETS=("etth1_trend" "etth1_stationarity" "etth1_seasonality" "etth1_seasonality_binary" "etth1_change_point"
           "weather_trend" "weather_stationarity" "weather_seasonality" "weather_seasonality_binary" "weather_change_point"
           "electricity_trend" "electricity_stationarity" "electricity_seasonality" "electricity_seasonality_binary" "electricity_change_point"
           "traffic_trend" "traffic_stationarity" "traffic_seasonality" "traffic_seasonality_binary" "traffic_change_point"
           "exchange_rate_trend" "exchange_rate_stationarity" "exchange_rate_seasonality" "exchange_rate_seasonality_binary" "exchange_rate_change_point")

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

echo "===== ALL EXTRACTION COMPLETE ====="
