#!/usr/bin/env bash
# run_experiment.sh — Run the full extract→train→evaluate pipeline for one model × one dataset.
#
# Usage:
#   ./scripts/run_experiment.sh <model> <dataset> [num_samples] [seq_len]
#
# Example:
#   ./scripts/run_experiment.sh itransformer synthetic_trend 5000 96

set -euo pipefail

MODEL="${1:?Usage: $0 <model> <dataset> [num_samples] [seq_len]}"
DATASET="${2:?Usage: $0 <model> <dataset> [num_samples] [seq_len]}"
NUM_SAMPLES="${3:-5000}"
SEQ_LEN="${4:-96}"

SEED=42
EPOCHS=100
BATCH_SIZE=256
LR=0.001
PROBE_TYPE="linear"

# Derive directory names
PROPERTY="${DATASET#synthetic_}"  # strip "synthetic_" prefix
REPR_DIR="outputs/representations/${MODEL}/${DATASET}"
PROBE_DIR="outputs/probes/${MODEL}_${PROPERTY}_${PROBE_TYPE}"
EVAL_DIR="outputs/eval/${MODEL}_${PROPERTY}"

PYTHON="PYTHONPATH=. .venv/bin/python"

echo "============================================"
echo "  Model:    ${MODEL}"
echo "  Dataset:  ${DATASET}"
echo "  Property: ${PROPERTY}"
echo "  Samples:  ${NUM_SAMPLES}, Seq len: ${SEQ_LEN}"
echo "============================================"

# Step 1: Extract representations
echo ""
echo ">>> Step 1/3: Extract representations"
eval ${PYTHON} scripts/extract_representations.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --layers all \
    --output_dir outputs/representations/ \
    --num_samples "${NUM_SAMPLES}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size 64 \
    --seed "${SEED}"

# Step 2: Train linear probes
echo ""
echo ">>> Step 2/3: Train probes"
eval ${PYTHON} scripts/train_probe.py \
    --representations_dir "${REPR_DIR}" \
    --probe_type "${PROBE_TYPE}" \
    --output_dir "${PROBE_DIR}" \
    --learning_rate "${LR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}"

# Step 3: Evaluate
echo ""
echo ">>> Step 3/3: Evaluate probes"
eval ${PYTHON} scripts/evaluate_probe.py \
    --probe_dir "${PROBE_DIR}" \
    --representations_dir "${REPR_DIR}" \
    --output_dir "${EVAL_DIR}" \
    --metrics accuracy f1 r2 \
    --plot

echo ""
echo ">>> Done! Results in ${EVAL_DIR}/"
echo "============================================"
