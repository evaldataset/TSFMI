#!/bin/bash
# Extract hard variant representations for Timer, TimesFM, Moirai
# Then mean-pool 3D representations to 2D for probing compatibility.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/extract_hard_variants_batch.sh

set -euo pipefail

PYTHONPATH=.
export PYTHONPATH

PYTHON=".venv/bin/python"
DEVICE="${DEVICE:-cuda:1}"

# Validate device string
if [[ ! "$DEVICE" =~ ^(cuda(:[0-9]+)?|cpu)$ ]]; then
    echo "ERROR: Invalid DEVICE='$DEVICE'. Expected 'cpu' or 'cuda[:N]'."
    exit 1
fi

HARD_DATASETS="synthetic_trend_hard synthetic_frequency_hard synthetic_stationarity_hard synthetic_anomaly_hard synthetic_change_point_hard"

echo "=== Extracting hard variant representations ==="
echo "Device: $DEVICE"
echo ""

for model in timer timesfm moirai; do
    for dataset in $HARD_DATASETS; do
        outdir="outputs/representations/${model}/${dataset}"
        if [ -d "$outdir" ] && [ "$(ls "$outdir"/*.pt 2>/dev/null | wc -l)" -gt 1 ]; then
            echo "[SKIP] $model / $dataset (already exists)"
            continue
        fi
        echo "[EXTRACT] $model / $dataset"
        $PYTHON scripts/extract_representations.py \
            --model "$model" \
            --dataset "$dataset" \
            --layers all \
            --output_dir outputs/representations/ \
            --device "$DEVICE" || echo "[WARN] Failed: $model / $dataset"
    done
done

echo ""
echo "=== Mean-pooling 3D representations ==="

$PYTHON -c "
import shutil
import json
from pathlib import Path
import torch

models = ['timer', 'timesfm', 'moirai']
hard_datasets = [
    'synthetic_trend_hard', 'synthetic_frequency_hard',
    'synthetic_stationarity_hard', 'synthetic_anomaly_hard',
    'synthetic_change_point_hard',
]

for model in models:
    for dataset in hard_datasets:
        src_dir = Path(f'outputs/representations/{model}/{dataset}')
        dst_dir = Path(f'outputs/representations/{model}_meanpool/{dataset}')

        if not src_dir.exists():
            print(f'  SKIP {model}_meanpool/{dataset}: source not found')
            continue

        if dst_dir.exists() and any(dst_dir.glob('*.pt')):
            pt_count = len(list(dst_dir.glob('*.pt')))
            if pt_count > 1:
                print(f'  SKIP {model}_meanpool/{dataset}: already exists ({pt_count} files)')
                continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy labels
        labels_src = src_dir / 'labels.pt'
        if labels_src.exists():
            shutil.copy2(labels_src, dst_dir / 'labels.pt')

        # Copy metadata
        meta_src = src_dir / 'metadata.json'
        if meta_src.exists():
            meta = json.loads(meta_src.read_text())
            meta['mean_pooled'] = True
            (dst_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))

        # Mean-pool each layer
        for pt_file in sorted(src_dir.glob('*.pt')):
            if pt_file.stem == 'labels':
                continue
            t = torch.load(pt_file, map_location='cpu', weights_only=True)
            if t.ndim == 3:
                t = t.mean(dim=1)
            torch.save(t, dst_dir / pt_file.name)
            print(f'  {model}_meanpool/{dataset}/{pt_file.stem}: {t.shape}')

print('Done.')
"

echo ""
echo "=== Running hard variant benchmark ==="
$PYTHON scripts/run_hard_variant_benchmark.py

echo ""
echo "=== Complete ==="
