#!/usr/bin/env bash
# Run canonical benchmark across all 7 models x 5 classification properties + seasonality regression.
# Uses GPU 0,1,2 in parallel (though sklearn is CPU-bound, this distributes torch.load I/O).
#
# Usage:
#   bash scripts/run_canonical_all.sh

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

REPR_ROOT="outputs/representations"
OUT_ROOT="outputs/canonical"
mkdir -p "$OUT_ROOT"

MODELS=(
    "moment_pca512"
    "chronos"
    "patchtst_pretrained"
    "gpt4ts_pca512"
    "timer_meanpool"
    "timesfm_meanpool"
    "moirai_meanpool"
)

CLASS_PROPS=(
    "synthetic_trend:trend:classification"
    "synthetic_frequency:frequency:classification"
    "synthetic_stationarity:stationarity:classification"
    "synthetic_anomaly:anomaly:classification"
    "synthetic_change_point:change_point:classification"
)
REG_PROPS=(
    "synthetic_seasonality:seasonality:regression"
)

# Distribute model index across GPUs 0,1,2 (round-robin)
run_one() {
    local model=$1
    local ds=$2
    local prop=$3
    local task=$4
    local gpu=$5
    local out="$OUT_ROOT/${model}_${prop}"
    local repr="$REPR_ROOT/$model/$ds"
    if [[ ! -d "$repr" ]]; then
        echo "SKIP: $repr not found"
        return
    fi
    if [[ -f "$out/canonical_results.json" ]]; then
        echo "SKIP: $out/canonical_results.json already exists"
        return
    fi
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. .venv/bin/python scripts/run_canonical_benchmark.py \
        --representations_dir "$repr" \
        --property "$prop" \
        --task_type "$task" \
        --output_dir "$out" 2>&1 | tail -5
}

export -f run_one
export REPR_ROOT OUT_ROOT

idx=0
pids=()
for model in "${MODELS[@]}"; do
    for spec in "${CLASS_PROPS[@]}" "${REG_PROPS[@]}"; do
        IFS=':' read -r ds prop task <<< "$spec"
        gpu=$((idx % 3))
        echo "[GPU$gpu] $model / $prop / $task"
        run_one "$model" "$ds" "$prop" "$task" "$gpu" &
        pids+=($!)
        idx=$((idx+1))
        # Limit concurrency to 6 at a time (2 per GPU)
        if [[ ${#pids[@]} -ge 6 ]]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
done
wait
echo "All canonical benchmark runs complete."
