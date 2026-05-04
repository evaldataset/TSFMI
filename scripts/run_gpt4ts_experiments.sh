#!/bin/bash
# Full experiment pipeline for GPT4TS (frozen GPT-2 backbone)
# Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/run_gpt4ts_experiments.sh

set -e
export PYTHONPATH=.

MODEL="gpt4ts"
PYTHON=".venv/bin/python"

SYNTHETIC_DATASETS=(
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

REAL_DATASETS=(
    "etth1_trend"
    "etth1_stationarity"
    "etth1_seasonality"
    "etth1_change_point"
    "weather_trend"
    "weather_stationarity"
    "weather_seasonality"
    "weather_change_point"
)

echo "=== GPT4TS Full Pipeline ==="

# Step 1: Extract representations (synthetic)
for ds in "${SYNTHETIC_DATASETS[@]}"; do
    echo "--- Extracting: ${MODEL} / ${ds} ---"
    $PYTHON scripts/extract_representations.py \
        --model $MODEL --dataset $ds \
        --num_samples 5000 --output_dir outputs/representations/
done

# Step 2: Extract representations (real-world)
for ds in "${REAL_DATASETS[@]}"; do
    STRIDE=256
    if [[ "$ds" == etth1_* ]]; then
        STRIDE=64
    fi
    echo "--- Extracting: ${MODEL} / ${ds} (stride=${STRIDE}) ---"
    $PYTHON scripts/extract_representations.py \
        --model $MODEL --dataset $ds \
        --num_samples 5000 --stride $STRIDE \
        --output_dir outputs/representations/
done

# Step 3: Train linear + MLP probes (synthetic)
for ds in "${SYNTHETIC_DATASETS[@]}"; do
    PROP="${ds#synthetic_}"
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    
    echo "--- Training linear probe: ${MODEL} / ${ds} ---"
    $PYTHON scripts/train_probe.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/probes/${MODEL}_${ds}_linear" \
        --probe_type linear --epochs 100 --learning_rate 0.001

    echo "--- Training MLP control probe: ${MODEL} / ${ds} ---"
    $PYTHON scripts/train_probe.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/probes/${MODEL}_${ds}_mlp_control" \
        --probe_type mlp_control --epochs 100 --learning_rate 0.001
done

# Step 4: Train probes (real-world)
for ds in "${REAL_DATASETS[@]}"; do
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    
    echo "--- Training linear probe: ${MODEL} / ${ds} ---"
    $PYTHON scripts/train_probe.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/probes/${MODEL}_${ds}_linear" \
        --probe_type linear --epochs 100 --learning_rate 0.001

    echo "--- Training MLP control probe: ${MODEL} / ${ds} ---"
    $PYTHON scripts/train_probe.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/probes/${MODEL}_${ds}_mlp_control" \
        --probe_type mlp_control --epochs 100 --learning_rate 0.001
done

# Step 5: Evaluate probes with selectivity (synthetic)
for ds in "${SYNTHETIC_DATASETS[@]}"; do
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    
    echo "--- Evaluating: ${MODEL} / ${ds} ---"
    $PYTHON scripts/evaluate_probe.py \
        --representations_dir $REPR_DIR \
        --probe_dir "outputs/probes/${MODEL}_${ds}_linear" \
        --output_dir "outputs/eval/${MODEL}_${ds}" \
        --control_probe_dir "outputs/probes/${MODEL}_${ds}_mlp_control"
done

# Step 6: Evaluate probes (real-world)
for ds in "${REAL_DATASETS[@]}"; do
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    
    echo "--- Evaluating: ${MODEL} / ${ds} ---"
    $PYTHON scripts/evaluate_probe.py \
        --representations_dir $REPR_DIR \
        --probe_dir "outputs/probes/${MODEL}_${ds}_linear" \
        --output_dir "outputs/eval/${MODEL}_${ds}" \
        --control_probe_dir "outputs/probes/${MODEL}_${ds}_mlp_control"
done

# Step 7: CKA heatmaps (synthetic only)
for ds in "${SYNTHETIC_DATASETS[@]}"; do
    REPR_DIR="outputs/representations/${MODEL}/${ds}"
    PROP="${ds#synthetic_}"
    
    echo "--- CKA heatmap: ${MODEL} / ${ds} ---"
    $PYTHON scripts/compute_cka_heatmap.py \
        --representations_dir $REPR_DIR \
        --output_dir "outputs/cka/${MODEL}_${PROP}" \
        --max_samples 2000
done

echo "=== GPT4TS Pipeline Complete ==="
