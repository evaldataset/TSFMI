#!/bin/bash
# GPT4TS probe training + evaluation (extraction already done)
set -e
export PYTHONPATH=.
MODEL="gpt4ts"
PYTHON=".venv/bin/python"

ALL_DATASETS=(
    "synthetic_trend"
    "synthetic_seasonality"
    "synthetic_frequency"
    "synthetic_stationarity"
    "synthetic_anomaly"
    "synthetic_change_point"
    "synthetic_trend_hard"
    "synthetic_frequency_hard"
    "synthetic_anomaly_hard"
    "etth1_trend"
    "etth1_stationarity"
    "etth1_seasonality"
    "etth1_change_point"
    "weather_trend"
    "weather_stationarity"
    "weather_seasonality"
    "weather_change_point"
)

SYNTHETIC_ONLY=(
    "synthetic_trend"
    "synthetic_seasonality"
    "synthetic_frequency"
    "synthetic_stationarity"
    "synthetic_anomaly"
    "synthetic_change_point"
    "synthetic_trend_hard"
    "synthetic_frequency_hard"
    "synthetic_anomaly_hard"
)

echo "=== GPT4TS Probe Training + Evaluation ==="

# Train linear + MLP probes
for ds in "${ALL_DATASETS[@]}"; do
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    if [ ! -d "$REPR_DIR" ]; then
        echo "SKIP: $REPR_DIR not found"
        continue
    fi

    echo "--- Training linear: ${MODEL}/${ds} ---"
    $PYTHON scripts/train_probe.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/probes/${MODEL}_${ds}_linear" \
        --probe_type linear --epochs 100 --learning_rate 0.001

    echo "--- Training MLP control: ${MODEL}/${ds} ---"
    $PYTHON scripts/train_probe.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/probes/${MODEL}_${ds}_mlp_control" \
        --probe_type mlp_control --epochs 100 --learning_rate 0.001
done

# Evaluate with selectivity
for ds in "${ALL_DATASETS[@]}"; do
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    if [ ! -d "$REPR_DIR" ]; then continue; fi

    echo "--- Evaluating: ${MODEL}/${ds} ---"
    $PYTHON scripts/evaluate_probe.py \
        --representations_dir $REPR_DIR \
        --probe_dir "outputs/probes/${MODEL}_${ds}_linear" \
        --output_dir "outputs/eval/${MODEL}_${ds}" \
        --control_probe_dir "outputs/probes/${MODEL}_${ds}_mlp_control"
done

# CKA heatmaps (synthetic only)
for ds in "${SYNTHETIC_ONLY[@]}"; do
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    PROP="${ds#synthetic_}"
    if [ ! -d "$REPR_DIR" ]; then continue; fi

    echo "--- CKA: ${MODEL}/${PROP} ---"
    $PYTHON scripts/compute_cka_heatmap.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/cka/${MODEL}_${PROP}" \
        --max_samples 2000
done

echo "=== GPT4TS Pipeline Complete ==="
