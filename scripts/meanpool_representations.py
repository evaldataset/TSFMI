"""Mean-pool reducer for multi-patch representation tensors.

Loads per-layer .pt files of shape (N, P, D) and writes (N, D) tensors after
mean-pooling over the patch dimension P. Used for Timer / TimesFM / Moirai whose
encoder outputs are inherently patched and where mean-pool is the standardized
view referenced by the canonical TSFMI protocol.

Usage:
    PYTHONPATH=. python scripts/meanpool_representations.py \
        --input_dir outputs/representations/timer/synthetic_anomaly \
        --output_dir outputs/representations/timer_meanpool/synthetic_anomaly
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mean-pool over patch dim for canonical TSFMI views.")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Carry labels and metadata through unchanged.
    for keep in ("labels.pt", "metadata.json"):
        s = src / keep
        if s.exists():
            shutil.copy2(s, dst / keep)

    layer_files = sorted(f for f in src.glob("*.pt") if f.stem not in ("labels", "metadata"))
    pooled = []
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim == 3:
            t_mp = t.mean(dim=1)
        elif t.ndim == 2:
            t_mp = t  # already (N, D)
        else:
            # Higher-rank: collapse all middle dims by mean.
            t_mp = t.reshape(t.shape[0], -1, t.shape[-1]).mean(dim=1)
        torch.save(t_mp.contiguous(), dst / lf.name)
        pooled.append({"layer": lf.stem, "in_shape": list(t.shape), "out_shape": list(t_mp.shape)})

    meta_path = dst / "meanpool_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "input_dir": str(src),
                "output_dir": str(dst),
                "n_layers": len(pooled),
                "reduction": "mean_pool_over_patch_dim",
                "layers": pooled,
            },
            indent=2,
        )
    )
    print(f"Mean-pooled {len(pooled)} layers from {src} -> {dst}")


if __name__ == "__main__":
    main()
