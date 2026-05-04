#!/bin/bash
# GPT4TS intervention experiments (LDA steering vectors)
# Run AFTER phase4 pipeline completes (needs gpt4ts_pca512 probes)
set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
PYTHON=".venv/bin/python"

echo "============================================"
echo "  GPT4TS Intervention Experiments"
echo "============================================"

MODEL="gpt4ts_pca512"
PROPERTIES=(
    "synthetic_trend:trend:classification"
    "synthetic_seasonality:seasonality:regression"
    "synthetic_frequency:frequency:classification"
    "synthetic_stationarity:stationarity:classification"
    "synthetic_anomaly:anomaly:classification"
    "synthetic_change_point:change_point:classification"
    "synthetic_trend_hard:trend_hard:classification"
    "synthetic_frequency_hard:frequency_hard:classification"
    "synthetic_anomaly_hard:anomaly_hard:classification"
)

for entry in "${PROPERTIES[@]}"; do
    IFS=':' read -r dataset prop task_type <<< "$entry"
    REPR_DIR="outputs/representations/${MODEL}/${dataset}"
    PROBE_DIR="outputs/probes/${MODEL}_${dataset}_linear"
    OUTPUT_DIR="outputs/interventions/${MODEL}_${prop}"

    if [ ! -d "$REPR_DIR" ] || [ ! -d "$PROBE_DIR" ]; then
        echo "SKIP: ${MODEL}/${prop} (missing repr or probe dir)"
        continue
    fi
    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/intervention_results.json" ]; then
        echo "SKIP (exists): ${OUTPUT_DIR}"
        continue
    fi

    echo "--- Intervention: ${MODEL}/${prop} ---"
    $PYTHON scripts/run_intervention.py \
        --representations_dir $REPR_DIR \
        --probe_dir $PROBE_DIR \
        --output_dir $OUTPUT_DIR
done

echo ""
echo "=== GPT4TS Interventions Complete ==="
