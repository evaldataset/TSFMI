"""Stage-separated runtime of the representations compared in the paper.

Every stage is timed on its own, after a warm-up, as the median of three repetitions:

- model loading (once per model, reported separately);
- representation extraction for N windows of length 512 (TSFM mean-pooled embedding of the
  selected layer, ROCKET-style, tsfresh efficient, hand-crafted), N in {1k, 5k, 20k};
- 1-NN query latency against a stored collection of N series: cosine on embeddings, Euclidean on
  z-normalised series (ED), and DTW with a 10% window on a 100-query sample;
- label-free detection on 1,000 windows: TimesFM one-step forecast surprise vs. AR(16) residuals.

The machine is shared; GPU utilisation by other jobs is recorded with every GPU measurement, so GPU
times are upper bounds.

Usage:
    PYTHONPATH=. python scripts/run_runtime_benchmark.py --out outputs/runtime_benchmark
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SIZES = (1000, 5000, 20000)
MODELS = ("timesfm", "patchtst_fm", "moirai", "chronos")
REPEATS = 3


def gpu_util() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return [int(v) for v in out.split()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def timed(fn, repeats: int = REPEATS, warm=None) -> float:
    (warm or fn)()  # warm-up (a cheaper call can be passed for expensive stages)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def windows(n: int) -> NDArray[np.float64]:
    from scripts.run_realistic_anomaly import realistic_anomaly_dataset

    x, _ = realistic_anomaly_dataset(n, 512, 7)
    return np.asarray(x, dtype=np.float64)


def main() -> int:
    import torch

    from scripts.extract_representations import load_model_wrapper
    from scripts.run_canonical_baselines import hand_crafted_features
    from scripts.run_forecast_surprise import ar_residuals, forecast_residuals, timesfm_predictor
    from scripts.run_resolution_study import rocket_parallel, selected_layer, tsfresh_features
    from src.extractors.hook_manager import HookManager

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/runtime_benchmark"))
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res: dict = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu_jobs": args.jobs,
        "repeats": REPEATS,
        "load_seconds": {},
        "extract_seconds": {},
        "query_ms": {},
        "detect_seconds": {},
        "gpu_util_samples": [],
    }
    data = {n: windows(n) for n in SIZES}

    for model in MODELS:
        t0 = time.perf_counter()
        wrapper = load_model_wrapper(model, None, device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        res["load_seconds"][model] = time.perf_counter() - t0
        layer = selected_layer(model)
        name = next(nm for nm in wrapper.get_layer_names() if nm.replace(".", "_") == layer)

        def embed(x: NDArray, w=wrapper, nm=name) -> NDArray:
            out = []
            with HookManager(w.model, [nm]) as hm, torch.inference_mode():
                for s in range(0, len(x), 128):
                    w.forward(torch.tensor(x[s : s + 128], dtype=torch.float32, device=device))
                    a = hm.get_activations()[nm].float()
                    a = a.reshape(a.shape[0], -1, a.shape[-1]).mean(dim=1)
                    out.append(a.cpu().numpy())
                    hm.clear()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return np.concatenate(out)

        res["extract_seconds"][model] = {}
        for n in SIZES:
            res["gpu_util_samples"].append(gpu_util())
            res["extract_seconds"][model][str(n)] = timed(lambda x=data[n]: embed(x))
            print(
                f"[extract {model} n={n}] {res['extract_seconds'][model][str(n)]:.2f}s", flush=True
            )
        emb = embed(data[SIZES[-1]])
        res["embedding_dim"] = res.get("embedding_dim", {}) | {model: int(emb.shape[1])}
        del wrapper
        torch.cuda.empty_cache()

    for name, fn in (
        ("hand_crafted", hand_crafted_features),
        ("rocket", lambda x: rocket_parallel(x, args.jobs)),
        ("tsfresh_efficient", lambda x: tsfresh_features(x, args.jobs)),
    ):
        res["extract_seconds"][name] = {}
        for n in SIZES:
            reps = 1 if name == "tsfresh_efficient" and n == SIZES[-1] else REPEATS
            res["extract_seconds"][name][str(n)] = timed(lambda x=data[n], f=fn: f(x), reps)
            print(f"[extract {name} n={n}] {res['extract_seconds'][name][str(n)]:.2f}s", flush=True)

    from aeon.distances import dtw_pairwise_distance

    rng = np.random.default_rng(0)
    for n in SIZES:
        coll = data[n]
        z = (coll - coll.mean(1, keepdims=True)) / coll.std(1, keepdims=True)
        q = z[rng.choice(n, 100, replace=False)]
        emb = rng.standard_normal((n, 1280)).astype(np.float32)  # TimesFM width; values irrelevant
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        eq = emb[:100]
        res["query_ms"][str(n)] = {
            "cosine_1280d": 1000 * timed(lambda e=emb, a=eq: np.argmax(a @ e.T, axis=1)) / 100,
            "ed_512": 1000
            * timed(lambda zz=z, qq=q: np.argmin((zz**2).sum(1)[None] - 2 * qq @ zz.T, axis=1))
            / 100,
            "dtw_512_w10": 1000
            * timed(
                lambda zz=z, qq=q: dtw_pairwise_distance(qq[:10], zz, window=0.1, n_jobs=args.jobs),
                1,
            )
            / 10,
        }
        print(f"[query n={n}] {res['query_ms'][str(n)]}", flush=True)

    x = data[SIZES[0]]
    res["gpu_util_samples"].append(gpu_util())
    predict = timesfm_predictor(str(device))
    res["detect_seconds"]["timesfm_onestep_1000"] = timed(
        lambda: forecast_residuals(x, predict, 1),
        1,
        warm=lambda: forecast_residuals(x[:8], predict, 64),
    )
    res["detect_seconds"]["ar16_1000"] = timed(lambda: ar_residuals(x))
    print(f"[detect] {res['detect_seconds']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
