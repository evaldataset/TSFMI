"""Hewitt & Liang (2019) control-task selectivity for the canonical TSFMI benchmark.

Selectivity (Hewitt & Liang, EMNLP 2019, Sec. 1 / Fig. 2) is

    selectivity = task score - control-task score,

where both scores come from the SAME probe architecture with the SAME
hyperparameters on the SAME representation. A control task (H&L Sec. 2) has
*structure* -- the output for a token is a deterministic function of its word
type -- and *randomness* -- the output for each type is sampled independently
at random (H&L Sec. 2.1 fn. 2: from the empirical label distribution of the
real task). It can therefore only be solved by memorising type identity.

Time-series adaptation (time series have no recurring word types):
    * type = k-means cluster id of the RAW input window, computed on a fixed,
      model-independent descriptor: the per-window z-normalised series
      block-mean downsampled to 64 points. k-means is fit on the full dataset
      with a fixed seed; k in {20, 50}.
    * control label of a window = label of its cluster, one label per cluster
      drawn (fixed seed) from the empirical distribution of the real task's
      labels (classification) or targets (regression, scored with R^2).

Everything else is the canonical protocol of ``run_canonical_benchmark.py``:
60/20/20 split with split seeds 0-4, StandardScaler + LogisticRegression
(classification) / Ridge (regression), the canonical per-seed best layer read
from ``outputs/canonical/<dir>_<prop>/canonical_results.json`` and the
canonical reduction. The same selectivity is computed for the non-model
baselines of ``run_canonical_baselines.py`` (raw signal, 8-D hand-crafted,
256-D random projection).

Sanity gate: the task score of every (model, property) cell must reproduce the
canonical test score (mean inside the canonical 95% CI, +/- 1e-3). Cells that
fail are recorded with ``status="sanity_failed"`` and no control/selectivity
is computed for them.

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/run_hl_selectivity.py \
        --models moment chronos gpt4ts timer timesfm moirai \
        --properties trend seasonality frequency stationarity anomaly change_point \
        --ks 20 50 --extract

    # A model that is not in MODEL_SPECS (e.g. a PatchTST replacement):
    ... --models mymodel --model_spec mymodel:mymodel_meanpool:mymodel_raw_key:meanpool:1000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from scripts.run_canonical_baselines import hand_crafted_features, random_projection
from scripts.run_canonical_benchmark import bootstrap_ci, three_way_split, train_and_score
from src.datasets.synthetic import (
    SyntheticDataset,
    generate_anomaly_dataset,
    generate_change_point_dataset,
    generate_frequency_dataset,
    generate_seasonality_dataset,
    generate_stationarity_dataset,
    generate_trend_dataset,
)

SEEDS = [0, 1, 2, 3, 4]
SEQ_LEN = 512
DATA_SEED = 42  # dataset seed used by extract_representations.py / run_canonical_baselines.py
DESCRIPTOR_LEN = 64
KMEANS_SEED = 0
CONTROL_SEED = 0
SANITY_TOL = 1e-3
PCA_SEED = 42  # reduce_representations.py default
PCA_COMPONENTS = 512

DEFAULT_STORE = Path("outputs/representations_selectivity")
DEFAULT_OUTPUT = Path("outputs/hl_selectivity/results.json")
CANONICAL_ROOT = Path("outputs/canonical")
CANONICAL_BASELINE_ROOT = Path("outputs/canonical_baselines")

PROPERTIES: dict[str, tuple[str, str]] = {
    "trend": ("synthetic_trend", "classification"),
    "seasonality": ("synthetic_seasonality", "regression"),
    "frequency": ("synthetic_frequency", "classification"),
    "stationarity": ("synthetic_stationarity", "classification"),
    "anomaly": ("synthetic_anomaly", "classification"),
    "change_point": ("synthetic_change_point", "classification"),
}
GENERATORS = {
    "trend": generate_trend_dataset,
    "seasonality": generate_seasonality_dataset,
    "frequency": generate_frequency_dataset,
    "stationarity": generate_stationarity_dataset,
    "anomaly": generate_anomaly_dataset,
    "change_point": generate_change_point_dataset,
}
BASELINES = ("raw_signal", "hand_crafted", "random_projection")


@dataclass(frozen=True)
class ModelSpec:
    """How one canonical model row was produced.

    Attributes:
        name: CLI name of the model row.
        canonical_dir: Prefix of ``outputs/canonical/<canonical_dir>_<prop>/``.
        raw_key: ``--model`` key of ``scripts/extract_representations.py``.
        reduction: ``pca512_all`` (PCA fit on all samples), ``pca512_train``
            (PCA fit on the first 60% of samples), ``meanpool`` or ``identity``.
        num_samples: ``--num_samples`` used for the canonical extraction.
    """

    name: str
    canonical_dir: str
    raw_key: str
    reduction: str
    num_samples: int


# Reductions / sample counts are the ones that REPRODUCE the committed canonical
# numbers (verified per seed), which differ from scripts/extract_all_canonical.sh
# for moment / gpt4ts (PCA fit on all samples, n=5000) and chronos (mean-pool, n=5000).
MODEL_SPECS: dict[str, ModelSpec] = {
    "moment": ModelSpec("moment", "moment_pca512", "moment", "pca512_all", 5000),
    "gpt4ts": ModelSpec("gpt4ts", "gpt4ts_pca512", "gpt4ts", "pca512_all", 5000),
    "chronos": ModelSpec("chronos", "chronos", "chronos", "meanpool", 5000),
    "timer": ModelSpec("timer", "timer_meanpool", "timer", "meanpool", 1000),
    "timesfm": ModelSpec("timesfm", "timesfm_meanpool", "timesfm", "meanpool", 1000),
    "moirai": ModelSpec("moirai", "moirai_meanpool", "moirai", "meanpool", 1000),
}

# Canonical v2 (outputs/canonical_v2, scripts/run_canonical_v2.sh): n = 1000 for every model,
# mean-pooled views stored as <raw_key>_meanpool/, and for MOMENT / GPT4TS the full layer with
# PCA-512 fit inside the probe pipeline on each seed's training rows.
MODEL_SPECS_V2: dict[str, ModelSpec] = {
    "moment": ModelSpec("moment", "moment", "moment", "pca512_pipeline", 1000),
    "gpt4ts": ModelSpec("gpt4ts", "gpt4ts", "gpt4ts", "pca512_pipeline", 1000),
    "chronos": ModelSpec("chronos", "chronos", "chronos", "v2_meanpool", 1000),
    "timer": ModelSpec("timer", "timer", "timer", "v2_meanpool", 1000),
    "timesfm": ModelSpec("timesfm", "timesfm", "timesfm", "v2_meanpool", 1000),
    "moirai": ModelSpec("moirai", "moirai", "moirai", "v2_meanpool", 1000),
    "patchtst_fm": ModelSpec("patchtst_fm", "patchtst_fm", "patchtst_fm", "v2_meanpool", 1000),
}
V2_PCA_DIM = 512  # matches scripts/run_canonical_v2.sh PCA_DIM
V2_STORE = Path("outputs/representations_v2")
V2_CANONICAL_ROOT = Path("outputs/canonical_v2")
V2_CANONICAL_BASELINE_ROOT = Path("outputs/canonical_v2/baselines")


def pca_for(spec: ModelSpec) -> int | None:
    """PCA components fit inside the probe pipeline, or None if the view is pre-reduced."""
    return V2_PCA_DIM if spec.reduction == "pca512_pipeline" else None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--protocol",
        choices=["v1", "v2"],
        default="v1",
        help="v1: reproduce the committed outputs/canonical recipes; v2: outputs/canonical_v2.",
    )
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--properties", nargs="*", default=list(PROPERTIES))
    p.add_argument("--baselines", nargs="*", default=list(BASELINES))
    p.add_argument(
        "--baseline_num_samples",
        nargs="*",
        type=int,
        default=[1000],
        help="Dataset sizes for baseline rows (1000 = canonical_baselines).",
    )
    p.add_argument("--ks", nargs="*", type=int, default=[20, 50])
    p.add_argument(
        "--model_spec",
        action="append",
        default=[],
        help="Extra model spec name:canonical_dir:raw_key:reduction:num_samples.",
    )
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--extract", action="store_true", help="Extract missing layers on GPU.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_threads", type=int, default=16)
    p.add_argument(
        "--negative_control",
        action="store_true",
        help="Also run the k = n_samples raw-signal negative control.",
    )
    p.add_argument(
        "--no_sanity_gate",
        action="store_true",
        help="Compute control scores even when the task score misses canonical.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Control task
# --------------------------------------------------------------------------- #
def window_descriptor(sequences: NDArray[np.float64]) -> NDArray[np.float64]:
    """Model-independent type descriptor: z-normalised window, block-mean to 64 pts.

    Args:
        sequences: Raw windows of shape (N, L) with L divisible by 64.

    Returns:
        Descriptor array of shape (N, 64).
    """
    n, length = sequences.shape
    if length % DESCRIPTOR_LEN != 0:
        raise ValueError(f"seq_len {length} not divisible by {DESCRIPTOR_LEN}")
    mu = sequences.mean(axis=1, keepdims=True)
    sd = sequences.std(axis=1, keepdims=True)
    z = (sequences - mu) / (sd + 1e-8)
    return z.reshape(n, DESCRIPTOR_LEN, length // DESCRIPTOR_LEN).mean(axis=2)


def assign_types(sequences: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Assign each window a 'type' = k-means cluster id of its raw descriptor.

    ``k >= N`` is the degenerate negative control: every window is its own type.

    Args:
        sequences: Raw windows (N, L).
        k: Number of clusters.

    Returns:
        Integer type ids of shape (N,).
    """
    n = sequences.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)
    km = KMeans(n_clusters=k, n_init=10, random_state=KMEANS_SEED)
    return km.fit_predict(window_descriptor(sequences)).astype(np.int64)


def control_labels(
    types: NDArray[np.int64], y: NDArray[np.float64], seed: int = CONTROL_SEED
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw one random output per type from the empirical label distribution.

    Args:
        types: Type id per window (N,).
        y: Real-task labels / targets (N,), defining the empirical distribution.
        seed: Fixed control-task seed.

    Returns:
        Tuple of (per-window control labels (N,), per-type behaviour (n_types,)).
    """
    n_types = int(types.max()) + 1
    rng = np.random.default_rng(seed)
    behaviour = rng.choice(y, size=n_types, replace=True)
    return behaviour[types], behaviour


def probe_score(
    x_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    x_te: NDArray[np.float64],
    y_te: NDArray[np.float64],
    task_type: str,
    seed: int,
    pca: int | None = None,
) -> tuple[float, bool]:
    """Canonical probe score, with a constant fallback for single-class train labels.

    Args:
        x_tr: Train features.
        y_tr: Train labels.
        x_te: Test features.
        y_te: Test labels.
        task_type: ``classification`` or ``regression``.
        seed: Split seed (also the LogisticRegression random_state, as canonical).

    Returns:
        Tuple of (score, used_constant_fallback).
    """
    if task_type == "classification" and len(np.unique(y_tr.astype(np.int64))) < 2:
        return float(np.mean(y_te.astype(np.int64) == int(y_tr[0]))), True
    return train_and_score(x_tr, y_tr, x_te, y_te, task_type, seed, pca_components=pca), False


def summarize(values: list[float]) -> dict[str, object]:
    """Mean and repo-convention bootstrap 95% CI over split seeds.

    Args:
        values: One value per split seed.

    Returns:
        Dict with mean, ci95_low, ci95_high and per_seed values.
    """
    mean, lo, hi = bootstrap_ci(values)
    return {
        "mean": round(mean, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "per_seed": [round(v, 4) for v in values],
    }


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #
def layer_index(stem: str) -> int:
    """Hook index of a saved layer stem (``encoder_block_17`` -> 17)."""
    m = re.search(r"_(\d+)$", stem)
    if m is None:
        raise ValueError(f"Cannot parse layer index from {stem!r}")
    return int(m.group(1))


def extract_layers(
    spec: ModelSpec, dataset: str, stems: list[str], store: Path, batch_size: int
) -> None:
    """Run ``scripts/extract_representations.py`` for the missing layers only.

    Args:
        spec: Model spec.
        dataset: Dataset name (e.g. ``synthetic_anomaly``).
        stems: Layer file stems needed.
        store: Store root; raw tensors land in ``store/raw/<raw_key>/<dataset>/``.
        batch_size: Extraction batch size.

    Raises:
        RuntimeError: If extraction fails or does not produce the expected files.
    """
    raw_dir = store / "raw" / spec.raw_key / dataset
    missing = [s for s in sorted(set(stems)) if not (raw_dir / f"{s}.pt").exists()]
    if not missing:
        return
    idx = ",".join(str(layer_index(s)) for s in missing)
    cmd = [
        sys.executable,
        "scripts/extract_representations.py",
        "--model",
        spec.raw_key,
        "--dataset",
        dataset,
        "--layers",
        idx,
        "--num_samples",
        str(spec.num_samples),
        "--batch_size",
        str(batch_size),
        "--output_dir",
        str(store / "raw"),
    ]
    print(f"  EXTRACT {spec.raw_key}/{dataset} layers={missing}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"extraction failed:\n{proc.stderr[-3000:]}")
    still = [s for s in missing if not (raw_dir / f"{s}.pt").exists()]
    if still:
        raise RuntimeError(f"extraction did not produce {still} (layer index mapping?)")


def reduce_layer(raw: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply the canonical reduction to one raw layer tensor.

    Mirrors ``reduce_representations.py`` (PCA512, random_state=42) and
    ``meanpool_representations.py`` (mean over the patch axis).

    Args:
        raw: Raw activation tensor (N, ...).
        reduction: One of ``pca512_all``, ``pca512_train``, ``meanpool``, ``identity``.

    Returns:
        Reduced tensor.
    """
    if reduction == "identity":
        return raw
    if reduction == "meanpool":
        if raw.ndim == 3:
            return raw.mean(dim=1).contiguous()
        if raw.ndim == 2:
            return raw
        return raw.reshape(raw.shape[0], -1, raw.shape[-1]).mean(dim=1).contiguous()
    if reduction in ("pca512_all", "pca512_train"):
        flat = raw.view(raw.shape[0], -1).numpy().astype(np.float32)
        if flat.shape[1] <= PCA_COMPONENTS:
            return raw
        n_comp = min(PCA_COMPONENTS, *flat.shape)
        pca = PCA(n_components=n_comp, random_state=PCA_SEED)
        n_train = int(flat.shape[0] * 0.6)
        if reduction == "pca512_train" and n_train >= n_comp:
            pca.fit(flat[:n_train])
            return torch.from_numpy(pca.transform(flat))
        return torch.from_numpy(pca.fit_transform(flat))
    raise ValueError(f"unknown reduction {reduction!r}")


def load_reduced(
    spec: ModelSpec, dataset: str, stems: list[str], store: Path
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64]]:
    """Load (reducing and caching if needed) the canonical view of each layer.

    Args:
        spec: Model spec.
        dataset: Dataset name.
        stems: Layer stems.
        store: Store root.

    Returns:
        Tuple of ({stem: (N, D) float64 features}, labels).
    """
    raw_dir = store / "raw" / spec.raw_key / dataset
    if spec.reduction in ("pca512_pipeline", "v2_meanpool"):
        src = raw_dir if spec.reduction == "pca512_pipeline" else (
            store / f"{spec.raw_key}_meanpool" / dataset)
        feats_v2: dict[str, NDArray[np.float64]] = {}
        for s in sorted(set(stems)):
            t = torch.load(src / f"{s}.pt", map_location="cpu", weights_only=True)
            feats_v2[s] = t.reshape(t.shape[0], -1).numpy().astype(np.float64)
        lab = torch.load(src / "labels.pt", map_location="cpu", weights_only=True).numpy()
        return feats_v2, lab
    red_dir = store / spec.canonical_dir / dataset
    red_dir.mkdir(parents=True, exist_ok=True)
    feats: dict[str, NDArray[np.float64]] = {}
    for s in sorted(set(stems)):
        red_path = red_dir / f"{s}.pt"
        if not red_path.exists():
            raw = torch.load(raw_dir / f"{s}.pt", map_location="cpu", weights_only=True)
            torch.save(reduce_layer(raw, spec.reduction), red_path)
            del raw
        t = torch.load(red_path, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        feats[s] = t.numpy().astype(np.float64)
    labels = torch.load(raw_dir / "labels.pt", map_location="cpu", weights_only=True).numpy()
    return feats, labels


# --------------------------------------------------------------------------- #
# Cell evaluation
# --------------------------------------------------------------------------- #
def evaluate_cell(
    feats_per_seed: list[NDArray[np.float64]],
    y: NDArray[np.float64],
    task_type: str,
    types_by_k: dict[int, NDArray[np.int64]],
    sanity_ref: dict[str, object] | None,
    gate: bool,
    pca: int | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Task score + control score for every k on one (row, property) cell.

    Args:
        feats_per_seed: Feature matrix used for each split seed (per-seed best layer).
        y: Real-task labels.
        task_type: ``classification`` or ``regression``.
        types_by_k: Type assignment per k.
        sanity_ref: Canonical reference (test_mean / ci) or None.
        gate: Whether a failed sanity check suppresses the control computation.

    Returns:
        Tuple of (task summary incl. sanity verdict, list of per-k records).
    """
    task_scores = []
    for seed, x in zip(SEEDS, feats_per_seed, strict=True):
        xtr, ytr, _, _, xte, yte = three_way_split(x, y, seed)
        task_scores.append(probe_score(xtr, ytr, xte, yte, task_type, seed, pca)[0])
    task = summarize(task_scores)

    sanity: dict[str, object] = {"reference": sanity_ref, "pass": None}
    if sanity_ref is not None:
        lo = float(sanity_ref["test_ci95_low"]) - SANITY_TOL
        hi = float(sanity_ref["test_ci95_high"]) + SANITY_TOL
        ok = lo <= float(task["mean"]) <= hi
        sanity["pass"] = ok
        sanity["delta_mean"] = round(float(task["mean"]) - float(sanity_ref["test_mean"]), 4)
    task["sanity"] = sanity
    if gate and sanity["pass"] is False:
        return task, []

    records = []
    for k, types in types_by_k.items():
        yc, behaviour = control_labels(types, y)
        ctrl, sel, seen, maj, fallback = [], [], [], [], 0
        for seed, x, t_s in zip(SEEDS, feats_per_seed, task_scores, strict=True):
            xtr, yctr, _, _, xte, ycte = three_way_split(x, yc, seed)
            ttr, _, _, _, tte, _ = three_way_split(types, yc, seed)
            c, fb = probe_score(xtr, yctr, xte, ycte, task_type, seed, pca)
            fallback += int(fb)
            ctrl.append(c)
            sel.append(t_s - c)
            seen.append(float(np.isin(tte, ttr).mean()))
            if task_type == "classification":
                vals, cnt = np.unique(yctr.astype(np.int64), return_counts=True)
                maj.append(float(np.mean(ycte.astype(np.int64) == vals[np.argmax(cnt)])))
        rec: dict[str, object] = {
            "k": k,
            "n_types": int(types.max()) + 1,
            "control": summarize(ctrl),
            "selectivity": summarize(sel),
            "frac_test_types_seen_in_train": round(float(np.mean(seen)), 4),
            "constant_fallback_seeds": fallback,
        }
        if task_type == "classification":
            rec["control_majority_chance"] = summarize(maj)
            rec["control_label_counts_per_type"] = np.bincount(behaviour.astype(np.int64)).tolist()
        records.append(rec)
    return task, records


def canonical_model_ref(spec: ModelSpec, prop: str) -> dict[str, object]:
    """Load the committed canonical result for a (model, property) cell."""
    path = CANONICAL_ROOT / f"{spec.canonical_dir}_{prop}" / "canonical_results.json"
    return json.loads(path.read_text())


def canonical_baseline_ref(prop: str, baseline: str) -> dict[str, object] | None:
    """Load the committed canonical baseline result, if any."""
    path = CANONICAL_BASELINE_ROOT / prop / "canonical_results.json"
    if not path.exists():
        return None
    for r in json.loads(path.read_text()):
        if r["baseline"] == baseline:
            return r
    return None


def regenerate(prop: str, n: int) -> SyntheticDataset:
    """Regenerate the exact synthetic dataset used by extraction / baselines."""
    return GENERATORS[prop](n, SEQ_LEN, seed=DATA_SEED)


def baseline_features(name: str, seqs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Baseline feature matrix via the canonical baseline functions."""
    if name == "raw_signal":
        return seqs
    if name == "hand_crafted":
        return hand_crafted_features(seqs)
    if name == "random_projection":
        return random_projection(seqs)
    raise ValueError(name)


def main() -> None:
    """Run the H&L selectivity grid and write results JSON."""
    args = parse_args()
    t0 = time.time()
    global CANONICAL_ROOT, CANONICAL_BASELINE_ROOT
    if args.protocol == "v2":
        specs = dict(MODEL_SPECS_V2)
        CANONICAL_ROOT, CANONICAL_BASELINE_ROOT = V2_CANONICAL_ROOT, V2_CANONICAL_BASELINE_ROOT
        if args.store == DEFAULT_STORE:
            args.store = V2_STORE
        if args.output == DEFAULT_OUTPUT:
            args.output = Path("outputs/hl_selectivity_v2/results.json")
        if args.extract:
            raise SystemExit("--extract is not supported with --protocol v2 (use run_canonical_v2)")
    else:
        specs = dict(MODEL_SPECS)
    if args.models is None:
        args.models = list(specs)
    for s in args.model_spec:
        name, cdir, raw, red, n = s.split(":")
        specs[name] = ModelSpec(name, cdir, raw, red, int(n))
    command = (
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} PYTHONPATH=. "
        + " ".join(shlex.quote(a) for a in [".venv/bin/python", *sys.argv])
    )
    gate = not args.no_sanity_gate
    results: list[dict[str, object]] = []
    negative: list[dict[str, object]] = []
    type_cache: dict[tuple[str, int], tuple[NDArray[np.float64], dict[int, NDArray[np.int64]]]] = {}

    def types_for(prop: str, n: int) -> tuple[NDArray[np.float64], dict[int, NDArray[np.int64]]]:
        key = (prop, n)
        if key not in type_cache:
            ds = regenerate(prop, n)
            type_cache[key] = (ds.sequences, {k: assign_types(ds.sequences, k) for k in args.ks})
        return type_cache[key]

    with threadpool_limits(args.n_threads):
        # ---------------- models ----------------
        for m in args.models:
            spec = specs[m]
            for prop in args.properties:
                dataset, task_type = PROPERTIES[prop]
                tc = time.time()
                ref = canonical_model_ref(spec, prop)
                stems = list(ref["best_layer_per_seed"])
                if args.extract:
                    extract_layers(spec, dataset, stems, args.store, args.batch_size)
                feats, labels = load_reduced(spec, dataset, stems, args.store)
                seqs, types_by_k = types_for(prop, spec.num_samples)
                ds_labels = regenerate(prop, spec.num_samples).labels
                if not np.array_equal(ds_labels, labels):
                    raise RuntimeError(f"{m}/{prop}: extracted labels != regenerated labels")
                del seqs
                task, recs = evaluate_cell(
                    [feats[s] for s in stems], labels, task_type, types_by_k, ref, gate,
                    pca_for(spec),
                )
                base = {
                    "row": m,
                    "row_kind": "model",
                    "canonical_dir": spec.canonical_dir,
                    "property": prop,
                    "task_type": task_type,
                    "num_samples": spec.num_samples,
                    "best_layer_per_seed": stems,
                    "reduction": spec.reduction,
                    "task": task,
                    "status": "ok" if recs else "sanity_failed",
                    "command": command,
                }
                for r in recs or [{"k": k} for k in args.ks]:
                    results.append({**base, **r})
                print(
                    f"[{m:8s} {prop:13s}] task={task['mean']} "
                    f"canon={ref['test_mean']} sanity={task['sanity']['pass']} "
                    + " ".join(
                        f"k={r['k']}:ctl={r['control']['mean']},sel={r['selectivity']['mean']}"
                        for r in recs
                    )
                    + f" ({time.time() - tc:.0f}s)",
                    flush=True,
                )
                del feats

        # ---------------- baselines ----------------
        for n in args.baseline_num_samples:
            for prop in args.properties:
                _, task_type = PROPERTIES[prop]
                seqs, types_by_k = types_for(prop, n)
                y = regenerate(prop, n).labels
                for bl in args.baselines:
                    x = baseline_features(bl, seqs)
                    ref = canonical_baseline_ref(prop, bl) if n == 1000 else None
                    task, recs = evaluate_cell(
                        [x] * len(SEEDS), y, task_type, types_by_k, ref, gate
                    )
                    base = {
                        "row": bl,
                        "row_kind": "baseline",
                        "property": prop,
                        "task_type": task_type,
                        "num_samples": n,
                        "best_layer_per_seed": None,
                        "reduction": "none",
                        "task": task,
                        "status": "ok" if recs else "sanity_failed",
                        "command": command,
                    }
                    for r in recs or [{"k": k} for k in args.ks]:
                        results.append({**base, **r})
                    print(
                        f"[{bl:17s} n={n} {prop:13s}] task={task['mean']} "
                        f"sanity={task['sanity']['pass']} "
                        + " ".join(
                            f"k={r['k']}:ctl={r['control']['mean']},sel={r['selectivity']['mean']}"
                            for r in recs
                        ),
                        flush=True,
                    )

        # ---------------- negative control: k = n (type = sample) ----------------
        if args.negative_control:
            n = 1000
            for prop in args.properties:
                _, task_type = PROPERTIES[prop]
                ds = regenerate(prop, n)
                types = assign_types(ds.sequences, n)
                task, recs = evaluate_cell(
                    [ds.sequences] * len(SEEDS),
                    ds.labels,
                    task_type,
                    {n: types},
                    canonical_baseline_ref(prop, "raw_signal"),
                    gate,
                )
                for r in recs:
                    negative.append(
                        {
                            "row": "raw_signal",
                            "property": prop,
                            "task_type": task_type,
                            "num_samples": n,
                            "task": task,
                            **r,
                            "command": command,
                        }
                    )
                    print(
                        f"[NEG raw k=n {prop:13s}] ctl={r['control']['mean']} "
                        f"chance={r.get('control_majority_chance', {}).get('mean', 'R2~0')} "
                        f"seen={r['frac_test_types_seen_in_train']}",
                        flush=True,
                    )

    # ---------------- merge + write ----------------
    out: dict[str, object] = {"meta": {}, "results": [], "negative_control": []}
    if args.output.exists():
        out = json.loads(args.output.read_text())

    def key(r: dict[str, object]) -> tuple[object, ...]:
        return (r["row"], r["property"], r["k"], r["num_samples"])

    new_keys = {key(r) for r in results}
    out["results"] = [r for r in out["results"] if key(r) not in new_keys] + results
    if negative:
        neg_keys = {key(r) for r in negative}
        out["negative_control"] = [
            r for r in out["negative_control"] if key(r) not in neg_keys
        ] + negative
    out["meta"] = {
        "reference": "Hewitt & Liang (2019), Designing and Interpreting Probes with "
        "Control Tasks, EMNLP-IJCNLP, Sec. 1-3",
        "selectivity": "task score - control-task score, same probe/hparams/layer/split",
        "type_definition": f"k-means (n_init=10, random_state={KMEANS_SEED}) on per-window "
        f"z-normalised raw series block-mean downsampled to {DESCRIPTOR_LEN} points, "
        "fit on the full dataset",
        "control_labels": f"one draw per type from the empirical label/target "
        f"distribution, numpy default_rng({CONTROL_SEED}); fixed across split seeds",
        "probe": "StandardScaler + LogisticRegression(max_iter=1000, lbfgs, "
        "random_state=split seed) / Ridge(alpha=1.0) via run_canonical_benchmark.train_and_score",
        "splits": "run_canonical_benchmark.three_way_split, 60/20/20, seeds 0-4; test scores",
        "ci": "run_canonical_benchmark.bootstrap_ci (1000 resamples of 5 seeds, seed 42)",
        "sanity_gate": f"task mean within canonical 95% CI +/- {SANITY_TOL}",
        "store": str(args.store),
        "model_specs": {k: vars(v) for k, v in specs.items()},
        "last_command": command,
        "last_runtime_sec": round(time.time() - t0, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Saved {args.output} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
