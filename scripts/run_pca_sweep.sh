#!/bin/bash
# PCA dimension sweep for MOMENT and GPT4TS seasonality
# Tests PCA at {64, 128, 256, 512} dimensions
set -e

export PYTHONPATH=.
PYTHON=".venv/bin/python"
DEVICE="cuda:1"

MODELS=("moment" "gpt4ts")
PCA_DIMS=(64 128 256)

for MODEL in "${MODELS[@]}"; do
    for DIM in "${PCA_DIMS[@]}"; do
        REPR_DIR="outputs/representations/${MODEL}/synthetic_seasonality"
        OUT_DIR="outputs/probes_pca_sweep/${MODEL}_pca${DIM}_seasonality_linear"
        
        if [ ! -d "$REPR_DIR" ]; then
            echo "SKIP $MODEL (no full-D repr)"
            continue
        fi
        if [ -d "$OUT_DIR" ]; then
            echo "SKIP $MODEL PCA${DIM} (exists)"
            continue
        fi
        
        echo "Extracting PCA${DIM} representations for $MODEL seasonality..."
        $PYTHON -c "
import torch
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

repr_dir = Path('$REPR_DIR')
out_dir = Path('outputs/representations/${MODEL}_pca${DIM}/synthetic_seasonality')
out_dir.mkdir(parents=True, exist_ok=True)

# Copy labels
labels = torch.load(repr_dir / 'labels.pt', weights_only=True)
torch.save(labels, out_dir / 'labels.pt')

# PCA each layer
for pt_file in sorted(repr_dir.glob('*.pt')):
    if pt_file.stem == 'labels':
        continue
    X = torch.load(pt_file, weights_only=True).numpy()
    orig_shape = X.shape
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    n_comp = min($DIM, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    torch.save(torch.tensor(X_pca, dtype=torch.float32), out_dir / pt_file.name)
    var = pca.explained_variance_ratio_.sum()
    print(f'  {pt_file.stem}: {orig_shape} -> {X_pca.shape}, var={var:.3f}')
"
        
        echo "Training probe for $MODEL PCA${DIM}..."
        $PYTHON scripts/train_probe.py \
            --representations_dir "outputs/representations/${MODEL}_pca${DIM}/synthetic_seasonality" \
            --probe_type linear \
            --output_dir "$OUT_DIR" \
            --device "$DEVICE" \
            --epochs 100 \
            --verbose 2>&1 | tail -1
    done
done

echo "PCA sweep complete."
