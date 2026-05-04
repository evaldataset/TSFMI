#!/bin/bash
# Phase 4: Complete pipeline for GPT4TS + LEACE + Cross-model CKA
set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2
PYTHON=".venv/bin/python"

echo "============================================"
echo "  Phase 4: Full Pipeline"
echo "============================================"

# ============================================
# Part A: GPT4TS PCA reduction + probes
# ============================================

ALL_SYNTHETIC=(
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

ALL_REAL=(
    "etth1_trend"
    "etth1_stationarity"
    "etth1_seasonality"
    "etth1_change_point"
    "weather_trend"
    "weather_stationarity"
    "weather_seasonality"
    "weather_change_point"
)

MODEL="gpt4ts"
PCA_MODEL="gpt4ts_pca512"

echo "=== Part A: GPT4TS PCA Reduction + Probes ==="

# A1: PCA reduce all GPT4TS representations
for ds in "${ALL_SYNTHETIC[@]}" "${ALL_REAL[@]}"; do
    INPUT_DIR="outputs/representations/${MODEL}/${ds}"
    OUTPUT_DIR="outputs/representations/${PCA_MODEL}/${ds}"
    if [ ! -d "$INPUT_DIR" ]; then
        echo "SKIP PCA: $INPUT_DIR not found"
        continue
    fi
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/labels.pt" ]; then
        echo "SKIP PCA (exists): $OUTPUT_DIR"
        continue
    fi
    echo "--- PCA reduce: ${ds} ---"
    $PYTHON scripts/reduce_representations.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --n_components 512
done

# A2: Train linear + MLP probes on PCA-reduced
for ds in "${ALL_SYNTHETIC[@]}" "${ALL_REAL[@]}"; do
    REPR_DIR="outputs/representations/${PCA_MODEL}/${ds}"
    if [ ! -d "$REPR_DIR" ]; then
        echo "SKIP probe: $REPR_DIR not found"
        continue
    fi
    
    LINEAR_DIR="outputs/probes/${PCA_MODEL}_${ds}_linear"
    MLP_DIR="outputs/probes/${PCA_MODEL}_${ds}_mlp_control"
    
    if [ -d "$LINEAR_DIR" ] && ls "$LINEAR_DIR"/*/probe.pt 1>/dev/null 2>&1; then
        echo "SKIP linear probe (exists): $LINEAR_DIR"
    else
        echo "--- Training linear: ${PCA_MODEL}/${ds} ---"
        $PYTHON scripts/train_probe.py \
            --representations_dir $REPR_DIR \
            --output_dir "$LINEAR_DIR" \
            --probe_type linear --epochs 100 --learning_rate 0.001
    fi
    
    if [ -d "$MLP_DIR" ] && ls "$MLP_DIR"/*/probe.pt 1>/dev/null 2>&1; then
        echo "SKIP MLP probe (exists): $MLP_DIR"
    else
        echo "--- Training MLP control: ${PCA_MODEL}/${ds} ---"
        $PYTHON scripts/train_probe.py \
            --representations_dir $REPR_DIR \
            --output_dir "$MLP_DIR" \
            --probe_type mlp_control --epochs 100 --learning_rate 0.001
    fi
done

# A3: Evaluate with selectivity
for ds in "${ALL_SYNTHETIC[@]}" "${ALL_REAL[@]}"; do
    REPR_DIR="outputs/representations/${PCA_MODEL}/${ds}"
    if [ ! -d "$REPR_DIR" ]; then continue; fi
    
    EVAL_DIR="outputs/eval/${PCA_MODEL}_${ds}"
    if [ -d "$EVAL_DIR" ] && [ -f "$EVAL_DIR/layer_metrics.json" ]; then
        echo "SKIP eval (exists): $EVAL_DIR"
        continue
    fi
    
    echo "--- Eval: ${PCA_MODEL}/${ds} ---"
    $PYTHON scripts/evaluate_probe.py \
        --representations_dir $REPR_DIR \
        --probe_dir "outputs/probes/${PCA_MODEL}_${ds}_linear" \
        --output_dir "$EVAL_DIR" \
        --control_probe_dir "outputs/probes/${PCA_MODEL}_${ds}_mlp_control"
done

# A4: CKA heatmaps (synthetic)
for ds in "${ALL_SYNTHETIC[@]}"; do
    REPR_DIR="outputs/representations/${PCA_MODEL}/${ds}"
    PROP="${ds#synthetic_}"
    if [ ! -d "$REPR_DIR" ]; then continue; fi
    
    CKA_DIR="outputs/cka/${PCA_MODEL}_${PROP}"
    if [ -d "$CKA_DIR" ] && [ -f "$CKA_DIR/cka_matrix.json" ]; then
        echo "SKIP CKA (exists): $CKA_DIR"
        continue
    fi
    
    echo "--- CKA: ${PCA_MODEL}/${PROP} ---"
    $PYTHON scripts/compute_cka_heatmap.py \
        --representations_dir $REPR_DIR \
        --output_dir "$CKA_DIR" \
        --max_samples 2000
done

echo "=== Part A Complete ==="

# ============================================
# Part B: LEACE Concept Erasure (all pre-trained models)
# ============================================
echo ""
echo "=== Part B: LEACE Concept Erasure ==="

LEACE_MODELS=("moment_pca512" "chronos" "patchtst_pretrained" "gpt4ts_pca512")
LEACE_PROPS=("synthetic_trend" "synthetic_seasonality" "synthetic_frequency" "synthetic_stationarity" "synthetic_anomaly" "synthetic_change_point")

for model in "${LEACE_MODELS[@]}"; do
    for ds in "${LEACE_PROPS[@]}"; do
        REPR_DIR="outputs/representations/${model}/${ds}"
        PROBE_DIR="outputs/probes/${model}_${ds}_linear"
        PROP="${ds#synthetic_}"
        OUTPUT_DIR="outputs/leace/${model}_${PROP}"
        
        if [ ! -d "$REPR_DIR" ] || [ ! -d "$PROBE_DIR" ]; then
            echo "SKIP LEACE: ${model}/${ds} (missing repr or probe)"
            continue
        fi
        if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/leace_results.json" ]; then
            echo "SKIP LEACE (exists): $OUTPUT_DIR"
            continue
        fi
        
        echo "--- LEACE: ${model}/${PROP} ---"
        $PYTHON scripts/run_leace_erasure.py \
            --representations_dir $REPR_DIR \
            --probe_dir $PROBE_DIR \
            --output_dir $OUTPUT_DIR
    done
done

echo "=== Part B Complete ==="

# ============================================
# Part C: Cross-model CKA
# ============================================
echo ""
echo "=== Part C: Cross-model CKA ==="

# Pairs to compare (on synthetic_trend)
CROSS_PAIRS=(
    "moment_pca512:chronos:synthetic_trend"
    "moment_pca512:patchtst_pretrained:synthetic_trend"
    "moment_pca512:gpt4ts_pca512:synthetic_trend"
    "chronos:patchtst_pretrained:synthetic_trend"
    "chronos:gpt4ts_pca512:synthetic_trend"
    "patchtst_pretrained:gpt4ts_pca512:synthetic_trend"
)

for pair in "${CROSS_PAIRS[@]}"; do
    IFS=':' read -r model_a model_b dataset <<< "$pair"
    
    REPR_A="outputs/representations/${model_a}/${dataset}"
    REPR_B="outputs/representations/${model_b}/${dataset}"
    OUTPUT_DIR="outputs/cka/${model_a}_vs_${model_b}_${dataset#synthetic_}"
    
    if [ ! -d "$REPR_A" ] || [ ! -d "$REPR_B" ]; then
        echo "SKIP cross-CKA: ${model_a} vs ${model_b} (missing repr)"
        continue
    fi
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/cross_model_cka_matrix.json" ]; then
        echo "SKIP cross-CKA (exists): $OUTPUT_DIR"
        continue
    fi
    
    echo "--- Cross-CKA: ${model_a} vs ${model_b} (${dataset}) ---"
    $PYTHON scripts/compute_cross_model_cka.py \
        --repr_dir_a $REPR_A \
        --repr_dir_b $REPR_B \
        --output_dir $OUTPUT_DIR \
        --model_a_name "$model_a" \
        --model_b_name "$model_b" \
        --max_samples 2000
done

echo "=== Part C Complete ==="

# ============================================
# Part D: Re-aggregate all results
# ============================================
echo ""
echo "=== Part D: Aggregate Results ==="
$PYTHON scripts/aggregate_results.py --eval_dir outputs/eval/ --output_dir outputs/summary/

echo ""
echo "============================================"
echo "  Phase 4 Pipeline Complete!"
echo "============================================"
