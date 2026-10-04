#!/usr/bin/env bash
# Canonical benchmark v2: ONE protocol, ONE environment, for every model and baseline.
#
# Fixes F16 (the committed outputs/canonical table mixes recipes: PCA-512 fit on all 5000
# samples for MOMENT/GPT4TS, n=5000 for Chronos, n=1000 elsewhere). Here every cell uses:
#   - data: 6 canonical synthetic properties, n=1000, seq_len=512, data seed 42
#   - split: 60/20/20 train/val/test, split seeds 0-4, best layer chosen on val only,
#     test score reported, 1000-resample bootstrap 95% CI (scripts/run_canonical_benchmark.py)
#   - views: mean-pool over patches/tokens (per-sample, leak-free) for chronos, timer, timesfm,
#     moirai, patchtst_fm; for moment and gpt4ts the full flattened layer is passed and PCA-512 is
#     fit inside the per-seed pipeline on the TRAINING split only (--pca-in-pipeline 512)
#   - baselines: raw signal, 8-D hand-crafted, 256-D random projection on the same data
#
# Stages (select with STAGES, default all): env extract pool bench baselines summary
#
# Usage (from the TSFMI root, after creating .venv; see outputs/canonical_v2/README.md):
#   make reproduce-canonical-v2
#   REPR_ROOT=/big/disk/representations_v2 GPUS="1 2 3" bash scripts/run_canonical_v2.sh
#
# Environment knobs:
#   REPR_ROOT   where tensors go (default outputs/representations_v2; gitignored)
#   OUT_ROOT    result JSONs (default outputs/canonical_v2)
#   DATA_SEED   seed of the synthetic generators (default 42, the canonical dataset)
#   GPUS        space-separated GPU ids for extraction workers (default "0")
#   BATCH_SIZE  forward-pass batch size (default 32)
#   BENCH_JOBS  concurrent CPU benchmark jobs (default 6); BENCH_THREADS BLAS threads each (8)
#   ONLY_MODELS space-separated subset of model keys to process (default: all 7)
#   KEEP_RAW    1 keeps raw (N, P, D) tensors of mean-pooled models; 0 (default) deletes them
#               once the pooled view is written and verified (moment/gpt4ts raw is always kept,
#               it is the benchmark input)
#
# Idempotent: finished extractions, pooled views and result JSONs are skipped.
# Failures are not swallowed: any failed step exits non-zero.

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="python"; fi
REPR_ROOT="${REPR_ROOT:-outputs/representations_v2}"
OUT_ROOT="${OUT_ROOT:-outputs/canonical_v2}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BENCH_JOBS="${BENCH_JOBS:-6}"
BENCH_THREADS="${BENCH_THREADS:-8}"
KEEP_RAW="${KEEP_RAW:-0}"
STAGES="${STAGES:-env extract pool bench baselines summary}"
NUM_SAMPLES=1000
SEQ_LEN=512
DATA_SEED="${DATA_SEED:-42}"
PCA_DIM=512

RAW_ROOT="$REPR_ROOT/raw"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$RAW_ROOT" "$OUT_ROOT" "$LOG_DIR"

# model_key:view  (view = pca_in_pipeline | meanpool). Ordered heaviest first so the
# per-GPU queues are balanced.
MODELS=(
    "timesfm:meanpool"
    "moment:pca_in_pipeline"
    "patchtst_fm:meanpool"
    "gpt4ts:pca_in_pipeline"
    "chronos:meanpool"
    "moirai:meanpool"
    "timer:meanpool"
)
if [[ -n "${ONLY_MODELS:-}" ]]; then
    _keep=()
    for spec in "${MODELS[@]}"; do
        [[ " $ONLY_MODELS " == *" ${spec%%:*} "* ]] && _keep+=("$spec")
    done
    MODELS=("${_keep[@]}")
fi
PROPS=(
    "synthetic_trend:trend:classification"
    "synthetic_seasonality:seasonality:regression"
    "synthetic_frequency:frequency:classification"
    "synthetic_stationarity:stationarity:classification"
    "synthetic_anomaly:anomaly:classification"
    "synthetic_change_point:change_point:classification"
)

# PROPS_OVERRIDE="dataset:property:task ..." runs the same pipeline on other datasets, e.g. the
# realistic anomaly generator: "synthetic_anomaly_realistic:anomaly_realistic:classification".
if [[ -n "${PROPS_OVERRIDE:-}" ]]; then
    read -r -a PROPS <<< "$PROPS_OVERRIDE"
fi

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
n_layers() { find "$1" -maxdepth 1 -name '*.pt' ! -name labels.pt 2>/dev/null | wc -l; }
has_layers() { [[ -d "$1" && "$(n_layers "$1")" -gt 0 ]]; }

# ---------------------------------------------------------------- env
if has_stage env; then
    echo "=== [env] recording environment -> $OUT_ROOT/environment.txt"
    "$PY" -m pip freeze 2>/dev/null > "$OUT_ROOT/environment.txt"
    "$PY" - > "$OUT_ROOT/environment_runtime.json" <<'EOF'
import json, platform, sys
import numpy, pandas, scipy, sklearn, statsmodels, torch, transformers
info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "statsmodels": statsmodels.__version__,
    "transformers": transformers.__version__,
}
print(json.dumps(info, indent=2))
EOF
    cat "$OUT_ROOT/environment_runtime.json"
fi

# ---------------------------------------------------------------- extract + pool
pooled_dir() { echo "$REPR_ROOT/${1}_meanpool/$2"; }

pool_one() {
    local model=$1 ds=$2
    local raw="$RAW_ROOT/$model/$ds" dst
    dst=$(pooled_dir "$model" "$ds")
    if has_layers "$dst" && [[ -f "$dst/meanpool_metadata.json" ]]; then
        echo "SKIP[pool]: $dst"
    else
        PYTHONPATH=. "$PY" scripts/meanpool_representations.py --input_dir "$raw" --output_dir "$dst"
    fi
    if [[ "$KEEP_RAW" == "0" && -d "$raw" ]] && has_layers "$raw"; then
        # Delete raw layers only when every raw layer has a pooled counterpart.
        if [[ "$(n_layers "$raw")" == "$(n_layers "$dst")" ]]; then
            find "$raw" -maxdepth 1 -name '*.pt' ! -name labels.pt -delete
            echo "RAW-DELETED: $raw (pooled view verified: $(n_layers "$dst") layers)"
        else
            echo "ERROR: layer count mismatch $raw vs $dst; raw kept" >&2
            return 4
        fi
    fi
}

extract_one() {
    local model=$1 view=$2 ds=$3 gpu=$4
    local raw="$RAW_ROOT/$model/$ds"
    local done_marker="$raw/.extract_done"
    if [[ -f "$done_marker" ]]; then
        echo "SKIP[extract]: $raw"
    else
        echo "EXTRACT: $model $ds (GPU $gpu)"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. "$PY" scripts/extract_representations.py \
            --model "$model" --dataset "$ds" --layers all \
            --num_samples "$NUM_SAMPLES" --seq_len "$SEQ_LEN" --seed "$DATA_SEED" \
            --batch_size "$BATCH_SIZE" --output_dir "$RAW_ROOT"
        touch "$done_marker"
    fi
    if [[ "$view" == "meanpool" ]] && has_stage pool; then
        pool_one "$model" "$ds"
    fi
}

if has_stage extract || has_stage pool; then
    read -r -a GPU_ARR <<< "$GPUS"
    n_gpu=${#GPU_ARR[@]}
    declare -a QUEUES
    for ((g = 0; g < n_gpu; g++)); do QUEUES[g]=""; done
    i=0
    for spec in "${MODELS[@]}"; do
        g=$((i % n_gpu))
        QUEUES[g]+="$spec "
        i=$((i + 1))
    done
    pids=()
    for ((g = 0; g < n_gpu; g++)); do
        gpu=${GPU_ARR[g]}
        (
            set -euo pipefail
            for spec in ${QUEUES[g]}; do
                model=${spec%%:*}
                view=${spec##*:}
                for p in "${PROPS[@]}"; do
                    ds=${p%%:*}
                    if has_stage extract; then
                        extract_one "$model" "$view" "$ds" "$gpu"
                    elif [[ "$view" == "meanpool" ]]; then
                        pool_one "$model" "$ds"
                    fi
                done
            done
        ) > "$LOG_DIR/extract_gpu${gpu}.log" 2>&1 &
        pids+=($!)
        echo "=== [extract] GPU $gpu queue: ${QUEUES[g]} (log: $LOG_DIR/extract_gpu${gpu}.log)"
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    if [[ $fail -ne 0 ]]; then
        echo "ERROR: extraction failed; see $LOG_DIR/extract_gpu*.log" >&2
        exit 3
    fi
fi

# ---------------------------------------------------------------- bench
bench_one() {
    local model=$1 view=$2 ds=$3 prop=$4 task=$5
    local out="$OUT_ROOT/${model}_${prop}"
    local repr extra=()
    if [[ -f "$out/canonical_results.json" ]]; then
        echo "SKIP[bench]: $out"
        return 0
    fi
    if [[ "$view" == "pca_in_pipeline" ]]; then
        repr="$RAW_ROOT/$model/$ds"
        extra=(--pca-in-pipeline "$PCA_DIM")
    else
        repr=$(pooled_dir "$model" "$ds")
    fi
    if ! has_layers "$repr"; then
        echo "ERROR: no representations in $repr" >&2
        return 5
    fi
    mkdir -p "$out"
    OMP_NUM_THREADS=$BENCH_THREADS OPENBLAS_NUM_THREADS=$BENCH_THREADS \
        MKL_NUM_THREADS=$BENCH_THREADS CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        "$PY" scripts/run_canonical_benchmark.py \
        --representations_dir "$repr" --property "$prop" --task_type "$task" \
        --output_dir "$out" "${extra[@]}" > "$LOG_DIR/bench_${model}_${prop}.log" 2>&1
    tail -1 "$LOG_DIR/bench_${model}_${prop}.log" | sed "s|^|[$model/$prop] |"
    grep "test mean" "$LOG_DIR/bench_${model}_${prop}.log" | sed "s|^|[$model/$prop] |"
}

if has_stage bench; then
    echo "=== [bench] $((${#MODELS[@]} * ${#PROPS[@]})) cells, $BENCH_JOBS concurrent"
    pids=()
    fail=0
    for spec in "${MODELS[@]}"; do
        model=${spec%%:*}
        view=${spec##*:}
        for p in "${PROPS[@]}"; do
            IFS=':' read -r ds prop task <<< "$p"
            bench_one "$model" "$view" "$ds" "$prop" "$task" &
            pids+=($!)
            if [[ ${#pids[@]} -ge $BENCH_JOBS ]]; then
                wait "${pids[0]}" || fail=1
                pids=("${pids[@]:1}")
            fi
        done
    done
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    if [[ $fail -ne 0 ]]; then
        echo "ERROR: a benchmark cell failed; see $LOG_DIR/bench_*.log" >&2
        exit 6
    fi
fi

# ---------------------------------------------------------------- baselines
if has_stage baselines; then
    echo "=== [baselines] raw / hand-crafted / random projection -> $OUT_ROOT/baselines"
    PYTHONPATH=. "$PY" scripts/run_canonical_baselines.py --output_dir "$OUT_ROOT/baselines" \
        --num_samples "$NUM_SAMPLES" --seq_len "$SEQ_LEN" --data_seed "$DATA_SEED" \
        | tee "$LOG_DIR/baselines.log"
fi

# ---------------------------------------------------------------- summary
if has_stage summary; then
    echo "=== [summary] -> $OUT_ROOT/summary.json, $OUT_ROOT/comparison.md"
    PYTHONPATH=. "$PY" scripts/summarize_canonical_v2.py --v2_root "$OUT_ROOT"
fi

echo "=== canonical v2 complete ==="
