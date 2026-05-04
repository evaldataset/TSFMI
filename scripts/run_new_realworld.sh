#!/bin/bash
# Pipeline for 3 new real-world datasets (Electricity, Traffic, Exchange-Rate)
# Runs extraction → PCA reduction → probe training → evaluation for all pre-trained models
set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
PYTHON=".venv/bin/python"

echo "============================================"
echo "  New Real-World Datasets Pipeline"
echo "============================================"

# --- Configuration ---
# Models that need raw extraction (all pre-trained ones)
EXTRACT_MODELS=("moment" "patchtst_pretrained" "chronos" "gpt4ts")

# Models that need PCA reduction before probing
PCA_MODELS=("moment" "gpt4ts")
# Models that can be probed directly (small enough representations)
DIRECT_MODELS=("patchtst_pretrained" "chronos")

# Probe model keys (after PCA for large models)
ALL_PROBE_MODELS=("moment_pca512" "patchtst_pretrained" "chronos" "gpt4ts_pca512")

# New datasets with appropriate strides
declare -A STRIDES
STRIDES["electricity"]=256   # 140256 rows → ~546 windows
STRIDES["traffic"]=64        # 17544 rows → ~267 windows
STRIDES["exchange_rate"]=32  # 7588 rows → ~222 windows

NEW_DATASETS=("electricity" "traffic" "exchange_rate")
PROPERTIES=("trend" "stationarity" "seasonality" "change_point")

# ============================================
# Part 1: Extract representations from all models
# ============================================
echo ""
echo "=== Part 1: Extract Representations ==="

for dataset in "${NEW_DATASETS[@]}"; do
    stride=${STRIDES[$dataset]}
    for prop in "${PROPERTIES[@]}"; do
        ds_name="${dataset}_${prop}"
        for model in "${EXTRACT_MODELS[@]}"; do
            OUTPUT_DIR="outputs/representations/${model}/${ds_name}"
            if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/labels.pt" ]; then
                echo "SKIP extract (exists): ${model}/${ds_name}"
                continue
            fi
            echo "--- Extract: ${model} / ${ds_name} (stride=${stride}) ---"
            $PYTHON scripts/extract_representations.py \
                --model $model \
                --dataset $ds_name \
                --layers all \
                --output_dir outputs/representations/ \
                --num_samples 5000 \
                --seq_len 512 \
                --batch_size 32 \
                --stride $stride \
                --seed 42
        done
    done
done

echo "=== Part 1 Complete ==="

# ============================================
# Part 2: PCA reduction for large models (MOMENT, GPT4TS)
# ============================================
echo ""
echo "=== Part 2: PCA Reduction ==="

for model in "${PCA_MODELS[@]}"; do
    pca_model="${model}_pca512"
    for dataset in "${NEW_DATASETS[@]}"; do
        for prop in "${PROPERTIES[@]}"; do
            ds_name="${dataset}_${prop}"
            INPUT_DIR="outputs/representations/${model}/${ds_name}"
            OUTPUT_DIR="outputs/representations/${pca_model}/${ds_name}"
            if [ ! -d "$INPUT_DIR" ]; then
                echo "SKIP PCA: ${INPUT_DIR} not found"
                continue
            fi
            if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/labels.pt" ]; then
                echo "SKIP PCA (exists): ${OUTPUT_DIR}"
                continue
            fi
            echo "--- PCA reduce: ${pca_model}/${ds_name} ---"
            $PYTHON scripts/reduce_representations.py \
                --input_dir $INPUT_DIR \
                --output_dir $OUTPUT_DIR \
                --n_components 512
        done
    done
done

echo "=== Part 2 Complete ==="

# ============================================
# Part 3: Train linear + MLP probes
# ============================================
echo ""
echo "=== Part 3: Train Probes ==="

for probe_model in "${ALL_PROBE_MODELS[@]}"; do
    for dataset in "${NEW_DATASETS[@]}"; do
        for prop in "${PROPERTIES[@]}"; do
            ds_name="${dataset}_${prop}"
            REPR_DIR="outputs/representations/${probe_model}/${ds_name}"
            if [ ! -d "$REPR_DIR" ]; then
                echo "SKIP probe: ${REPR_DIR} not found"
                continue
            fi

            LINEAR_DIR="outputs/probes/${probe_model}_${ds_name}_linear"
            MLP_DIR="outputs/probes/${probe_model}_${ds_name}_mlp_control"

            if [ -d "$LINEAR_DIR" ] && ls "$LINEAR_DIR"/*/probe.pt 1>/dev/null 2>&1; then
                echo "SKIP linear probe (exists): ${LINEAR_DIR}"
            else
                echo "--- Training linear: ${probe_model}/${ds_name} ---"
                $PYTHON scripts/train_probe.py \
                    --representations_dir $REPR_DIR \
                    --output_dir "$LINEAR_DIR" \
                    --probe_type linear --epochs 100 --learning_rate 0.001
            fi

            if [ -d "$MLP_DIR" ] && ls "$MLP_DIR"/*/probe.pt 1>/dev/null 2>&1; then
                echo "SKIP MLP probe (exists): ${MLP_DIR}"
            else
                echo "--- Training MLP control: ${probe_model}/${ds_name} ---"
                $PYTHON scripts/train_probe.py \
                    --representations_dir $REPR_DIR \
                    --output_dir "$MLP_DIR" \
                    --probe_type mlp_control --epochs 100 --learning_rate 0.001
            fi
        done
    done
done

echo "=== Part 3 Complete ==="

# ============================================
# Part 4: Evaluate with selectivity
# ============================================
echo ""
echo "=== Part 4: Evaluate Probes ==="

for probe_model in "${ALL_PROBE_MODELS[@]}"; do
    for dataset in "${NEW_DATASETS[@]}"; do
        for prop in "${PROPERTIES[@]}"; do
            ds_name="${dataset}_${prop}"
            REPR_DIR="outputs/representations/${probe_model}/${ds_name}"
            if [ ! -d "$REPR_DIR" ]; then continue; fi

            EVAL_DIR="outputs/eval/${probe_model}_${ds_name}"
            if [ -d "$EVAL_DIR" ] && [ -f "$EVAL_DIR/layer_metrics.json" ]; then
                echo "SKIP eval (exists): ${EVAL_DIR}"
                continue
            fi

            echo "--- Eval: ${probe_model}/${ds_name} ---"
            $PYTHON scripts/evaluate_probe.py \
                --representations_dir $REPR_DIR \
                --probe_dir "outputs/probes/${probe_model}_${ds_name}_linear" \
                --output_dir "$EVAL_DIR" \
                --control_probe_dir "outputs/probes/${probe_model}_${ds_name}_mlp_control"
        done
    done
done

echo "=== Part 4 Complete ==="

# ============================================
# Part 5: Re-aggregate all results
# ============================================
echo ""
echo "=== Part 5: Aggregate Results ==="
$PYTHON scripts/aggregate_results.py --eval_dir outputs/eval/ --output_dir outputs/summary/

echo ""
echo "============================================"
echo "  New Real-World Datasets Pipeline Complete!"
echo "============================================"
