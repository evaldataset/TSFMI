#!/usr/bin/env bash
# Full extraction pipeline for the 7 confirmatory models x 6 synthetic properties.
#
# This orchestrator runs the *two-step* canonical extraction:
#   1) raw extraction:        outputs/representations/<raw_key>/<dataset>/
#   2) standardized view:     outputs/representations/<canonical_dir>/<dataset>/
#      - PCA512 (train-only)  for moment_pca512, gpt4ts_pca512
#      - mean-pool over patch for timer_meanpool, timesfm_meanpool, moirai_meanpool
#      - identity copy        for chronos, patchtst_pretrained
#
# This is the script that `make reproduce-all-from-scratch` requires; the
# downstream `scripts/run_canonical_all.sh` reads from the canonical_dir paths
# and silently skips any (model, dataset) whose canonical_dir is empty.
#
# Idempotent: any (model, dataset) whose canonical output already contains
# layer .pt files is skipped. Distributes work across GPUs 0/1/2 round-robin.
# Failures are NOT swallowed: the script exits non-zero on the first failed cell.
#
# Usage:
#   make extract-representations
#   bash scripts/extract_all_canonical.sh

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then VENV_PY="python"; fi
REPR_ROOT="outputs/representations"
mkdir -p "$REPR_ROOT"

# Each entry: canonical_dir:raw_key:reduction
#   reduction = pca512 | meanpool | identity
MODELS=(
    "moment_pca512:moment:pca512"
    "chronos:chronos:identity"
    "patchtst_pretrained:patchtst_pretrained:identity"
    "gpt4ts_pca512:gpt4ts:pca512"
    "timer_meanpool:timer:meanpool"
    "timesfm_meanpool:timesfm:meanpool"
    "moirai_meanpool:moirai:meanpool"
)
DATASETS=(
    "synthetic_trend"
    "synthetic_seasonality"
    "synthetic_frequency"
    "synthetic_stationarity"
    "synthetic_anomaly"
    "synthetic_change_point"
)

# Optional restriction to a subset (used by make reproduce-one-cell).
ONLY_MODEL="${ONLY_MODEL:-}"
ONLY_DATASET="${ONLY_DATASET:-}"

extract_one() {
    local canonical_dir=$1
    local raw_key=$2
    local reduction=$3
    local ds=$4
    local gpu=$5

    local raw_out="$REPR_ROOT/$raw_key/$ds"
    local canonical_out="$REPR_ROOT/$canonical_dir/$ds"

    if [[ -d "$canonical_out" ]] && ls "$canonical_out"/*.pt &>/dev/null; then
        echo "SKIP: $canonical_out already populated"
        return 0
    fi

    # Step 1: raw extraction (idempotent on its own output).
    if [[ ! -d "$raw_out" ]] || ! ls "$raw_out"/*.pt &>/dev/null; then
        echo "EXTRACT[raw]: $raw_key $ds -> $raw_out (GPU $gpu)"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. "$VENV_PY" \
            scripts/extract_representations.py \
            --model "$raw_key" \
            --dataset "$ds" \
            --layers all \
            --output_dir "$REPR_ROOT"
    else
        echo "REUSE[raw]:   $raw_out"
    fi

    # Step 2: produce the canonical view.
    case "$reduction" in
        identity)
            if [[ "$canonical_dir" != "$raw_key" ]]; then
                echo "COPY:        $raw_out -> $canonical_out"
                mkdir -p "$canonical_out"
                cp -r "$raw_out"/. "$canonical_out"/
            fi
            ;;
        pca512)
            echo "REDUCE[pca]: train-only PCA512 -> $canonical_out"
            PYTHONPATH=. "$VENV_PY" scripts/reduce_representations.py \
                --input_dir "$raw_out" \
                --output_dir "$canonical_out" \
                --n_components 512 \
                --train_ratio 0.6
            ;;
        meanpool)
            echo "REDUCE[mp]:  mean-pool over patch -> $canonical_out"
            PYTHONPATH=. "$VENV_PY" scripts/meanpool_representations.py \
                --input_dir "$raw_out" \
                --output_dir "$canonical_out"
            ;;
        *)
            echo "ERROR: unknown reduction '$reduction'" >&2
            exit 2
            ;;
    esac

    # Sanity check: the canonical output must now have at least one layer .pt.
    if ! ls "$canonical_out"/*.pt &>/dev/null; then
        echo "ERROR: canonical output $canonical_out is still empty after reduction" >&2
        exit 3
    fi
}

i=0
for model_pair in "${MODELS[@]}"; do
    canonical_dir="${model_pair%%:*}"
    rest="${model_pair#*:}"
    raw_key="${rest%%:*}"
    reduction="${rest##*:}"

    if [[ -n "$ONLY_MODEL" && "$canonical_dir" != "$ONLY_MODEL" ]]; then
        continue
    fi

    for ds in "${DATASETS[@]}"; do
        if [[ -n "$ONLY_DATASET" && "$ds" != "$ONLY_DATASET" ]]; then
            continue
        fi
        gpu=$(( i % 3 ))
        i=$(( i + 1 ))
        extract_one "$canonical_dir" "$raw_key" "$reduction" "$ds" "$gpu"
    done
done

echo ""
echo "=== Extraction complete ==="
du -sh "$REPR_ROOT"
