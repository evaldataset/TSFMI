"""Measure the cost of all-layer extraction per model: throughput and peak GPU memory.

For each model, the frozen wrapper runs forward passes over windows of length 512 with forward
hooks on every probed layer, exactly as in extraction, and the hooked activations are copied to
host memory. One warm-up batch is excluded. Timing uses ``torch.cuda.synchronize`` around the
measured loop. Because the GPU may be shared with other jobs, the script also records the GPU
utilization reported by NVML before the run, so that contended measurements can be recognized.

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/measure_extraction_throughput.py \
        --out outputs/extraction_throughput
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_realistic_anomaly_tsfm import load_wrapper
from src.datasets.synthetic import generate_anomaly_dataset
from src.extractors.hook_manager import HookManager
from src.utils.device import resolve_device

MODELS = ("moment", "chronos", "timer", "timesfm", "moirai", "gpt4ts", "patchtst_fm")


def gpu_utilization() -> int | None:
    """GPU utilization (%) of the visible device from nvidia-smi, or None if unavailable."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return int(out.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def measure(model: str, x: torch.Tensor, batch: int, device: torch.device) -> dict:
    wrapper = load_wrapper(model)
    wrapper.load(device=device)
    wrapper.freeze()
    layers = wrapper.get_layer_names()
    torch.cuda.reset_peak_memory_stats(device)

    def run(chunk: torch.Tensor, w=wrapper) -> int:
        with HookManager(w.model, layers) as hm, torch.no_grad():
            w.forward(chunk)
            acts = hm.get_activations()
            return sum(a.detach().cpu().numel() * 4 for a in acts.values())

    run(x[:batch].to(device))  # warm-up
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    host_bytes = 0
    for s in range(batch, x.shape[0], batch):
        host_bytes += run(x[s : s + batch].to(device))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    n = x.shape[0] - batch
    out = {
        "n_layers": len(layers),
        "windows": n,
        "seconds": round(elapsed, 3),
        "windows_per_second": round(n / elapsed, 1),
        "activation_bytes_per_window": int(host_bytes / n),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
    }
    del wrapper
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/extraction_throughput"))
    ap.add_argument("--windows", type=int, default=544)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    device = resolve_device("cuda")
    ds = generate_anomaly_dataset(args.windows, 512, seed=42)
    x = torch.tensor(np.asarray(ds.sequences), dtype=torch.float32).unsqueeze(-1)
    rows = {}
    for m in args.models:
        util = gpu_utilization()
        rows[m] = {"gpu_utilization_before_percent": util, **measure(m, x, args.batch, device)}
        print(f"{m:<12} {rows[m]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "batch": args.batch,
                "seq_len": 512,
                "models": rows,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
