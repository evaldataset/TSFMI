#!/bin/bash
# Train linear + MLP control probes on moirai/timesfm/timer real-world representations
# Expects extraction outputs from run_realworld_extraction.sh

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
PROBE_TYPES=("linear")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        repr_dir="outputs/representations/$model/$dataset"
        if [ ! -f "$repr_dir/metadata.json" ]; then
            echo "SKIP $model/$dataset (no representations)"
            continue
        fi
        for probe_type in "${PROBE_TYPES[@]}"; do
            probe_dir="outputs/probes/${model}_${dataset}_${probe_type}"
            if [ -f "$probe_dir/summary.json" ]; then
                echo "SKIP $model/$dataset/$probe_type (exists)"
                continue
            fi
            echo "===== PROBE: $model / $dataset / $probe_type ====="
            python scripts/train_probe.py \
                --representations_dir "$repr_dir" \
                --output_dir "$probe_dir" \
                --probe_type "$probe_type" \
                --device cuda \
                2>&1 | tail -2
            echo "  DONE"
        done
    done
done

echo "===== ALL PROBE TRAINING COMPLETE ====="
