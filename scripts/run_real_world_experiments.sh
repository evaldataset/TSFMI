#!/usr/bin/env bash
# run_real_world_experiments.sh — Full extract→train→eval pipeline for real-world datasets.
#
# Runs MOMENT (with PCA), Chronos-Bolt, and PatchTST-Pre on ETTh1 and Weather
# datasets for 4 temporal properties each.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_real_world_experiments.sh

set -euo pipefail

PYTHON="PYTHONPATH=. .venv/bin/python"
SEED=42
EPOCHS=100
BATCH_SIZE=256
LR=0.001
SEQ_LEN=512

# Real-world datasets
DATASETS=(
    "etth1_trend"
    "etth1_stationarity"
    "etth1_seasonality"
    "etth1_change_point"
    "weather_trend"
    "weather_stationarity"
    "weather_seasonality"
    "weather_change_point"
)

# Models to run: model_name:output_name:num_samples:stride
# ETTh1 has ~265 windows at stride=64, Weather has ~1641 at stride=256
# We use stride=64 for ETTh1, stride=256 for Weather, cap at 1000 samples
MODELS=(
    "moment:moment_pca512"
    "chronos:chronos"
    "patchtst_pretrained:patchtst_pretrained"
)

for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS=: read -r MODEL_NAME OUTPUT_NAME <<< "${MODEL_ENTRY}"

    for DATASET in "${DATASETS[@]}"; do
        # Determine stride based on dataset
        if [[ "${DATASET}" == etth1_* ]]; then
            STRIDE=64
            NUM_SAMPLES=265
        else
            STRIDE=256
            NUM_SAMPLES=1000
        fi

        # Parse property from dataset name
        PROPERTY="${DATASET}"

        REPR_DIR="outputs/representations/${OUTPUT_NAME}/${DATASET}"
        PROBE_LINEAR_DIR="outputs/probes/${OUTPUT_NAME}_${PROPERTY}_linear"
        PROBE_MLP_DIR="outputs/probes/${OUTPUT_NAME}_${PROPERTY}_mlp_control"
        EVAL_DIR="outputs/eval/${OUTPUT_NAME}_${PROPERTY}"

        echo "============================================"
        echo "  Model:    ${MODEL_NAME} (${OUTPUT_NAME})"
        echo "  Dataset:  ${DATASET}"
        echo "  Samples:  ${NUM_SAMPLES}, Stride: ${STRIDE}"
        echo "============================================"

        # Step 1: Extract representations
        # For MOMENT, extract to moment/ then PCA to moment_pca512/
        if [[ "${OUTPUT_NAME}" == "moment_pca512" ]]; then
            REPR_DIR_RAW="outputs/representations/moment/${DATASET}"
            REPR_DIR="outputs/representations/moment_pca512/${DATASET}"
            # Extract raw MOMENT representations
            if [ -d "${REPR_DIR_RAW}" ] && [ -f "${REPR_DIR_RAW}/labels.pt" ]; then
                echo ">>> Step 1: Raw MOMENT representations exist, skipping"
            else
                echo ">>> Step 1: Extract raw MOMENT representations"
                eval ${PYTHON} scripts/extract_representations.py \
                    --model "${MODEL_NAME}" \
                    --dataset "${DATASET}" \
                    --layers all \
                    --output_dir outputs/representations/ \
                    --num_samples "${NUM_SAMPLES}" \
                    --seq_len "${SEQ_LEN}" \
                    --stride "${STRIDE}" \
                    --batch_size 64 \
                    --seed "${SEED}"
            fi
            # PCA reduction
            if [ -d "${REPR_DIR}" ] && [ "$(find ${REPR_DIR} -name '*.pt' ! -name 'labels.pt' 2>/dev/null | head -1)" ]; then
                echo ">>> Step 1b: PCA512 representations already exist, skipping"
            else
                echo ">>> Step 1b: PCA reduction to 512D"
                eval ${PYTHON} scripts/reduce_representations.py \
                    --input_dir "${REPR_DIR_RAW}" \
                    --output_dir "${REPR_DIR}" \
                    --n_components 512
            fi
        else
            # Non-MOMENT models: extract directly
            if [ -d "${REPR_DIR}" ] && [ -f "${REPR_DIR}/labels.pt" ]; then
                echo ">>> Step 1: Representations already exist, skipping"
            else
                echo ">>> Step 1: Extract representations"
                eval ${PYTHON} scripts/extract_representations.py \
                    --model "${MODEL_NAME}" \
                    --dataset "${DATASET}" \
                    --layers all \
                    --output_dir outputs/representations/ \
                    --num_samples "${NUM_SAMPLES}" \
                    --seq_len "${SEQ_LEN}" \
                    --stride "${STRIDE}" \
                    --batch_size 64 \
                    --seed "${SEED}"
            fi
        fi

        # Step 2a: Train linear probes
        if [ -d "${PROBE_LINEAR_DIR}" ] && [ -f "${PROBE_LINEAR_DIR}/summary.json" ]; then
            echo ">>> Step 2a: Linear probes already exist, skipping"
        else
            echo ">>> Step 2a: Train linear probes"
            eval ${PYTHON} scripts/train_probe.py \
                --representations_dir "${REPR_DIR}" \
                --probe_type linear \
                --output_dir "${PROBE_LINEAR_DIR}" \
                --learning_rate "${LR}" \
                --epochs "${EPOCHS}" \
                --batch_size "${BATCH_SIZE}" \
                --seed "${SEED}"
        fi

        # Step 2b: Train MLP control probes
        if [ -d "${PROBE_MLP_DIR}" ] && [ -f "${PROBE_MLP_DIR}/summary.json" ]; then
            echo ">>> Step 2b: MLP control probes already exist, skipping"
        else
            echo ">>> Step 2b: Train MLP control probes"
            eval ${PYTHON} scripts/train_probe.py \
                --representations_dir "${REPR_DIR}" \
                --probe_type mlp_control \
                --output_dir "${PROBE_MLP_DIR}" \
                --learning_rate "${LR}" \
                --epochs "${EPOCHS}" \
                --batch_size "${BATCH_SIZE}" \
                --seed "${SEED}"
        fi

        # Step 3: Evaluate with selectivity
        if [ -d "${EVAL_DIR}" ] && [ -f "${EVAL_DIR}/layer_metrics.json" ]; then
            echo ">>> Step 3: Evaluation already exists, skipping"
        else
            echo ">>> Step 3: Evaluate probes"
            eval ${PYTHON} scripts/evaluate_probe.py \
                --probe_dir "${PROBE_LINEAR_DIR}" \
                --representations_dir "${REPR_DIR}" \
                --output_dir "${EVAL_DIR}" \
                --control_probe_dir "${PROBE_MLP_DIR}" \
                --metrics accuracy f1 r2 selectivity \
                --plot
        fi

        echo ""
    done
done

echo "============================================"
echo "  All real-world experiments complete!"
echo "============================================"
