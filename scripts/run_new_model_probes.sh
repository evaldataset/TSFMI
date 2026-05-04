#!/bin/bash
# Train linear + MLP control probes on new model wrappers (autoformer/timesnet/fedformer)
# Expects extraction outputs from run_new_model_extraction.sh

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3

MODELS=("autoformer" "timesnet" "fedformer")
DATASETS=("synthetic_trend" "synthetic_seasonality" "synthetic_frequency" "synthetic_stationarity" "synthetic_anomaly" "synthetic_change_point")
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
                --epochs 50 \
                --verbose \
                2>&1 | tail -5
            echo "  DONE"
        done
    done
done

echo "===== ALL PROBE TRAINING COMPLETE ====="
