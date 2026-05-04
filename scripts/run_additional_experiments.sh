#!/bin/bash
# Master script for all additional experiments (Phase 5)
# Run with: bash scripts/run_additional_experiments.sh
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
PY=".venv/bin/python"

echo "=============================================="
echo "TSFMI Additional Experiments — Phase 5"
echo "=============================================="

# =====================================================
# P1.3: Paper Figure Generation (no GPU needed)
# =====================================================
echo ""
echo ">>> [P1.3] Generating paper figures..."
$PY scripts/plot_paper_figures.py
echo ">>> [P1.3] Done"

# =====================================================
# P2.3: Hard Synthetic — stationarity_hard, change_point_hard
# Extract + Probe + Evaluate for all 4 pre-trained models
# =====================================================
MODELS="moment patchtst_pretrained chronos gpt4ts"
NEW_HARD_DATASETS="synthetic_stationarity_hard synthetic_change_point_hard"

for model in $MODELS; do
    for dataset in $NEW_HARD_DATASETS; do
        prop=$(echo $dataset | sed 's/synthetic_//')
        echo ""
        echo ">>> [P2.3] Extract: $model / $dataset"
        $PY scripts/extract_representations.py \
            --model $model --dataset $dataset \
            --num_samples 5000 --output_dir outputs/representations/

        # PCA reduction for moment and gpt4ts
        if [ "$model" = "moment" ] || [ "$model" = "gpt4ts" ]; then
            repr_key="${model}_pca512"
            echo ">>> [P2.3] PCA reduce: $model -> $repr_key"
            $PY scripts/reduce_representations.py \
                --input_dir outputs/representations/${model}/${dataset}/ \
                --output_dir outputs/representations/${repr_key}/${dataset}/ \
                --n_components 512
        else
            repr_key=$model
        fi

        echo ">>> [P2.3] Train probe: $repr_key / $dataset"
        $PY scripts/train_probe.py \
            --representations_dir outputs/representations/${repr_key}/${dataset}/ \
            --output_dir outputs/probes/${repr_key}_${dataset}_linear/ \
            --probe_type linear --epochs 100

        echo ">>> [P2.3] Evaluate: $repr_key / $dataset"
        $PY scripts/evaluate_probe.py \
            --probe_dir outputs/probes/${repr_key}_${dataset}_linear/ \
            --representations_dir outputs/representations/${repr_key}/${dataset}/ \
            --output_dir outputs/eval/${repr_key}_${dataset}/
    done
done
echo ">>> [P2.3] Hard synthetic complete"

# =====================================================
# P1.1: Seasonality Binary — Real-world datasets
# =====================================================
REAL_DATASETS="etth1 weather electricity traffic exchange_rate"

for model in $MODELS; do
    for real_ds in $REAL_DATASETS; do
        dataset="${real_ds}_seasonality_binary"
        echo ""
        echo ">>> [P1.1] Extract: $model / $dataset"

        stride=256
        if [ "$real_ds" = "etth1" ]; then stride=64; fi

        $PY scripts/extract_representations.py \
            --model $model --dataset $dataset \
            --num_samples 5000 --stride $stride \
            --output_dir outputs/representations/

        if [ "$model" = "moment" ] || [ "$model" = "gpt4ts" ]; then
            repr_key="${model}_pca512"
            $PY scripts/reduce_representations.py \
                --input_dir outputs/representations/${model}/${dataset}/ \
                --output_dir outputs/representations/${repr_key}/${dataset}/ \
                --n_components 512
        else
            repr_key=$model
        fi

        echo ">>> [P1.1] Train probe: $repr_key / $dataset"
        $PY scripts/train_probe.py \
            --representations_dir outputs/representations/${repr_key}/${dataset}/ \
            --output_dir outputs/probes/${repr_key}_${dataset}_linear/ \
            --probe_type linear --epochs 100

        echo ">>> [P1.1] Evaluate: $repr_key / $dataset"
        $PY scripts/evaluate_probe.py \
            --probe_dir outputs/probes/${repr_key}_${dataset}_linear/ \
            --representations_dir outputs/representations/${repr_key}/${dataset}/ \
            --output_dir outputs/eval/${repr_key}_${dataset}/
    done
done
echo ">>> [P1.1] Seasonality binary complete"

# =====================================================
# P1.2: Full-D Probing (MOMENT and GPT4TS without PCA)
# =====================================================
FULLD_PROPS="synthetic_seasonality synthetic_anomaly"

for model in moment gpt4ts; do
    for dataset in $FULLD_PROPS; do
        echo ""
        echo ">>> [P1.2] Full-D Ridge probe: $model / $dataset"
        $PY scripts/run_fulld_probe.py \
            --representations_dir outputs/representations/${model}/${dataset}/ \
            --output_dir outputs/fulld_probes/${model}_${dataset}/ \
            --alpha 1.0 --max_samples 2000
    done
done
echo ">>> [P1.2] Full-D probing complete"

# =====================================================
# P2.1: Cross-Property LEACE
# =====================================================
CROSS_MODELS="moment_pca512 gpt4ts_pca512"
CROSS_PAIRS="trend:stationarity trend:change_point stationarity:change_point frequency:trend frequency:stationarity stationarity:trend change_point:trend change_point:stationarity"

for model in $CROSS_MODELS; do
    for pair in $CROSS_PAIRS; do
        erase_prop=$(echo $pair | cut -d: -f1)
        eval_prop=$(echo $pair | cut -d: -f2)
        echo ""
        echo ">>> [P2.1] Cross-LEACE: $model erase=$erase_prop eval=$eval_prop"
        $PY scripts/run_cross_property_leace.py \
            --model $model \
            --erase_property $erase_prop \
            --eval_property $eval_prop \
            --output_dir outputs/cross_leace/${model}/ \
            || echo ">>> WARNING: Cross-LEACE failed for $model $erase_prop->$eval_prop"
    done
done
echo ">>> [P2.1] Cross-property LEACE complete"

# =====================================================
# P2.2: Layer-wise LEACE (already in existing script, just more combos)
# Already done in Phase 4 for best layers. The existing script does ALL layers.
# =====================================================
echo ">>> [P2.2] Layer-wise LEACE already covered by existing runs"

# =====================================================
# P3.1: Attention Analysis
# =====================================================
ATTN_MODELS="patchtst_pretrained chronos"
ATTN_DATASETS="synthetic_trend synthetic_seasonality"

for model in $ATTN_MODELS; do
    for dataset in $ATTN_DATASETS; do
        echo ""
        echo ">>> [P3.1] Attention: $model / $dataset"
        $PY scripts/extract_attention.py \
            --model $model --dataset $dataset \
            --output_dir outputs/attention/${model}_$(echo $dataset | sed 's/synthetic_//')/ \
            --num_samples 500 \
            || echo ">>> WARNING: Attention extraction failed for $model $dataset"
    done
done
echo ">>> [P3.1] Attention analysis complete"

# =====================================================
# P3.2: Structural Probe
# =====================================================
STRUCT_MODELS="moment chronos patchtst_pretrained gpt4ts"

for model in $STRUCT_MODELS; do
    echo ""
    echo ">>> [P3.2] Structural probe: $model"
    # Use raw (non-PCA) representations to preserve patch dimension
    $PY scripts/run_structural_probe.py \
        --representations_dir outputs/representations/${model}/synthetic_trend/ \
        --output_dir outputs/structural_probe/${model}_trend/ \
        --num_samples 100 \
        || echo ">>> WARNING: Structural probe failed for $model"
done
echo ">>> [P3.2] Structural probe complete"

# =====================================================
# Re-aggregate all results
# =====================================================
echo ""
echo ">>> Re-aggregating all results..."
$PY scripts/aggregate_results.py \
    --eval_dir outputs/eval \
    --output_dir outputs/summary

echo ""
echo "=============================================="
echo "ALL ADDITIONAL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "New outputs:"
echo "  outputs/paper_figures/       — Publication figures"
echo "  outputs/fulld_probes/        — Full-dimensional probing results"
echo "  outputs/cross_leace/         — Cross-property LEACE interaction"
echo "  outputs/attention/           — Attention pattern analysis"
echo "  outputs/structural_probe/    — Structural probe results"
echo "  outputs/representations/*/synthetic_stationarity_hard/"
echo "  outputs/representations/*/synthetic_change_point_hard/"
echo "  outputs/representations/*/*_seasonality_binary/"
