"""MOMENT-Large Fisher LDR vs. raw-signal / hand-crafted controls (Wiliński et al., ICML 2025).

Wiliński et al. ("Exploring Representations and Interventions in Time Series Foundation
Models", ICML 2025) localise concepts in MOMENT-1-large with Fisher's Linear Discriminant
Ratio (LDR) per (layer, patch) and publish only *min-max-scaled* heatmaps (Fig. 7 / Fig. 14).
``scripts/run_wilinski_reproduction.py`` computed LDR for raw-signal and hand-crafted (HC)
controls but could not compare against MOMENT because the unscaled MOMENT values were never
published. This script closes that gap by re-running *their* pipeline end to end:

1. Data: their ``steertool.dataset_generator.generate_datasets`` on their YAML configs
   (``seed_everything(42)`` once, configs visited in ``os.listdir`` order, as in their code).
2. Activations: their ``steertool.moment.get_activations_MOMENT`` (``AutonLab/MOMENT-1-large``,
   reconstruction head, forward hook on ``encoder.block[i].layer[-1]`` of all 24 blocks,
   all-ones ``input_mask``; MOMENT's internal RevIN is the only input normalisation).
   The only shim is a cache around ``MOMENTPipeline.from_pretrained`` so the checkpoint is
   loaded once instead of once per call.
3. LDR: their ``steertool.separability.compute_linear_separability`` (sklearn LDA fitted and
   evaluated in-sample, Fisher ratio of the 1-D LDA projection), per (layer, patch) and on the
   patch-mean per layer (the red curve of their figures), followed by their global min-max
   scaling.

The controls (raw 512-D signal, RevIN-normalised signal = MOMENT's actual input, 8-D HC
features, and 8-D raw patches) are evaluated with the *same* LDR function on the *same*
series. Because the in-sample LDR is not comparable across feature spaces of different
dimensionality when n is small, every space is additionally scored after PCA to ``--pca-dim``
components (fit on the pooled two-class data) and with a held-out (cross-validated) LDR.

Example:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/run_wilinski_moment_ldr.py \
        --wilinski-repo /path/to/representations-in-tsfms
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from scripts.run_wilinski_reproduction import hand_crafted_features  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

LdrFn = Callable[[int, int, NDArray[np.floating], NDArray[np.floating], int], tuple[Any, ...]]

# (pair id, dataset 1, dataset 2, their --type flag, role)
PAIRS: list[tuple[str, str, str, str, str]] = [
    ("constant_vs_sine", "none_constant", "sine_constant", "constant-sine", "primary"),
    ("increasing_vs_decreasing", "none_increasing", "none_decreasing", "trend", "primary"),
    ("high_vs_low_freq", "sine_freq_high", "sine_freq_low", "periodicity", "primary"),
    # Extra pairs only used to check the scaled heatmaps against their Fig. 14 (ii)-(iv).
    ("constant_vs_increasing", "none_constant", "none_increasing", "constant-sine", "figure"),
    ("sine_vs_increasing", "sine_constant", "sine_increasing", "trend", "figure"),
    ("sine_vs_decreasing", "sine_constant", "sine_decreasing", "trend", "figure"),
]
PATCH_LEN = 8


# --------------------------------------------------------------------------------------
# Wiliński repo plumbing
# --------------------------------------------------------------------------------------
class _CachedPipelineFactory:
    """Stand-in for ``MOMENTPipeline`` that memoises ``from_pretrained``.

    Their ``get_activations_MOMENT`` reloads the checkpoint on every call; the cached object
    is the very same ``MOMENTPipeline`` instance, so the forward path is unchanged.
    """

    def __init__(self, real_cls: Any) -> None:
        """Store the real ``MOMENTPipeline`` class.

        Args:
            real_cls: ``momentfm.MOMENTPipeline``.
        """
        self._real_cls = real_cls
        self._cache: dict[str, Any] = {}

    def from_pretrained(self, name: str, model_kwargs: dict[str, Any]) -> Any:
        """Return a cached pipeline, loading it on first use.

        Args:
            name: HuggingFace checkpoint id.
            model_kwargs: Keyword arguments forwarded to ``MOMENTPipeline.from_pretrained``.

        Returns:
            The loaded ``MOMENTPipeline``.
        """
        if name not in self._cache:
            self._cache[name] = self._real_cls.from_pretrained(
                name, model_kwargs=copy.deepcopy(model_kwargs)
            )
        return self._cache[name]

    def loaded(self) -> list[Any]:
        """Return all pipelines loaded so far."""
        return list(self._cache.values())


def import_wilinski(repo: Path) -> dict[str, Any]:
    """Import the needed functions from the cloned Wiliński repository.

    Args:
        repo: Root of the ``representations-in-tsfms`` clone.

    Returns:
        Mapping with their ``generate_datasets``, ``load_dataset``, ``get_activations_MOMENT``,
        ``compute_linear_separability``, ``plot_linear_separability`` and the model factory.
    """
    steering = repo / "steering"
    if not (steering / "steertool" / "separability.py").exists():
        raise FileNotFoundError(f"{steering}/steertool/separability.py not found")
    sys.path.insert(0, str(steering))
    # steertool/__init__.py eagerly imports its Chronos module (nnsight, chronos-forecasting),
    # which the MOMENT separability path never touches. Stub them only if they are missing.
    import importlib.util
    import types

    for mod in ("nnsight", "chronos"):
        if importlib.util.find_spec(mod) is None:
            stub = types.ModuleType(mod)
            stub.NNsight = stub.ChronosPipeline = None  # type: ignore[attr-defined]
            sys.modules[mod] = stub
    import steertool.moment as st_moment
    import steertool.separability as st_sep
    from joblib.externals.cloudpickle import register_pickle_by_value
    from steertool.dataset_generator import generate_datasets
    from steertool.separability import compute_linear_separability, plot_linear_separability
    from steertool.utils import load_dataset

    # Ship their LDR function to joblib workers by value (workers cannot import steertool).
    register_pickle_by_value(st_sep)
    factory = _CachedPipelineFactory(st_moment.MOMENTPipeline)
    st_moment.MOMENTPipeline = factory
    return {
        "generate_datasets": generate_datasets,
        "load_dataset": load_dataset,
        "get_activations_MOMENT": st_moment.get_activations_MOMENT,
        "compute_linear_separability": compute_linear_separability,
        "plot_linear_separability": plot_linear_separability,
        "factory": factory,
    }


def git_commit(repo: Path) -> str:
    """Return the HEAD commit hash of ``repo``.

    Args:
        repo: Git working tree.

    Returns:
        40-character commit hash.
    """
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def generate_their_datasets(
    wil: dict[str, Any], repo: Path, work_dir: Path, seed: int
) -> tuple[dict[str, Path], list[str]]:
    """Run their dataset generator exactly as ``steertool.cli generate`` does.

    Args:
        wil: Output of :func:`import_wilinski`.
        repo: Wiliński repository root.
        work_dir: Scratch directory used as the generator's working directory.
        seed: Seed passed to their ``seed_everything`` (their CLI default is 42).

    Returns:
        Mapping dataset name -> parquet path, and the config visiting order.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "datasets").mkdir(exist_ok=True)
    link = work_dir / "configs"
    if not link.exists():
        link.symlink_to(repo / "steering" / "configs", target_is_directory=True)
    order = [f for f in os.listdir(link) if f != "config.yaml"]
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        wil["generate_datasets"](config_dir="configs", random_seed=seed)
    finally:
        os.chdir(cwd)
    paths = {p.stem: p for p in (work_dir / "datasets").glob("*.parquet")}
    return paths, order


# --------------------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------------------
def moment_activations(
    wil: dict[str, Any], series: torch.Tensor, device: str, batch_size: int
) -> NDArray[np.float32]:
    """Extract MOMENT activations through their ``get_activations_MOMENT``.

    Args:
        wil: Output of :func:`import_wilinski`.
        series: Float tensor ``(n, 1, 512)`` as returned by their ``load_dataset``.
        device: Device string.
        batch_size: Chunk size (their code feeds the whole set in a single batch).

    Returns:
        Array ``(n_layers, n, n_patches, d_model)``.
    """
    chunks = []
    for start in range(0, series.shape[0], batch_size):
        acts = wil["get_activations_MOMENT"](series[start : start + batch_size], device=device)
        chunks.append(acts.float().cpu().numpy())
    return np.concatenate(chunks, axis=1)


def revin_input(series: NDArray[np.float64]) -> NDArray[np.float32]:
    """Apply MOMENT's own RevIN (all-ones mask) to reproduce what the encoder actually sees.

    Args:
        series: ``(n, length)`` raw series.

    Returns:
        ``(n, length)`` float32 normalised series, computed with ``momentfm``'s RevIN module.
    """
    from momentfm.models.layers.revin import RevIN

    x = torch.tensor(series[:, None, :], dtype=torch.float32)
    mask = torch.ones(x.shape[0], x.shape[-1])
    out = RevIN(num_features=1, affine=False)(x=x, mask=mask, mode="norm")
    out = torch.nan_to_num(out, nan=0, posinf=0, neginf=0)
    return out[:, 0, :].numpy()


# --------------------------------------------------------------------------------------
# LDR variants
# --------------------------------------------------------------------------------------
def their_ldr(ldr_fn: LdrFn, a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Call their ``compute_linear_separability`` verbatim and return the Fisher score.

    Args:
        ldr_fn: Their ``compute_linear_separability``.
        a: Class-one features ``(n, d)``.
        b: Class-other features ``(n, d)``.

    Returns:
        LDR (may be ``inf``/``nan`` when the within-class variance of the projection is 0).
    """
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return float(ldr_fn(0, 0, a, b, len(a))[2])
        except (ValueError, np.linalg.LinAlgError):
            return float("nan")


def pca_ldr(ldr_fn: LdrFn, a: NDArray[np.floating], b: NDArray[np.floating], k: int) -> float:
    """Their LDR after PCA to ``k`` dims (PCA fit on the pooled two-class data).

    Args:
        ldr_fn: Their ``compute_linear_separability``.
        a: Class-one features ``(n, d)``.
        b: Class-other features ``(n, d)``.
        k: Target dimensionality.

    Returns:
        LDR in the ``k``-dim PCA space.
    """
    x = np.concatenate([a, b]).astype(np.float64)
    k_eff = min(k, x.shape[1], x.shape[0] - 1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        z = PCA(n_components=k_eff, random_state=0).fit_transform(x)
    return their_ldr(ldr_fn, z[: len(a)], z[len(a) :])


def cv_ldr(
    a: NDArray[np.floating], b: NDArray[np.floating], folds: int, seed: int
) -> tuple[float, float, float]:
    """Held-out Fisher ratio: fit LDA on training folds, score the 1-D test projections.

    Args:
        a: Class-one features ``(n, d)``.
        b: Class-other features ``(n, d)``.
        folds: Number of stratified folds.
        seed: Fold shuffling seed.

    Returns:
        (mean held-out LDR, mean held-out LDA accuracy, mean held-out accuracy of the
        difference-of-class-means direction thresholded at the midpoint). The last one is
        immune to sklearn's rank truncation, which can discard a discriminative direction
        whose within-class variance is exactly zero.
    """
    x = np.concatenate([a, b]).astype(np.float64)
    y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    ldrs, accs, dm_accs = [], [], []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for tr, te in skf.split(x, y):
            try:
                lda = LinearDiscriminantAnalysis().fit(x[tr], y[tr])
            except (ValueError, np.linalg.LinAlgError):
                return float("nan"), float("nan"), float("nan")
            p = lda.transform(x[te]).ravel()
            p1, p0 = p[y[te] == 1], p[y[te] == 0]
            ldrs.append(float((p1.mean() - p0.mean()) ** 2 / (p1.var() + p0.var())))
            accs.append(float(lda.score(x[te], y[te])))
            m1, m0 = x[tr][y[tr] == 1].mean(axis=0), x[tr][y[tr] == 0].mean(axis=0)
            proj = (x[te] - (m1 + m0) / 2) @ (m1 - m0)
            dm_accs.append(float(((proj > 0) == (y[te] == 1)).mean()))
    return float(np.mean(ldrs)), float(np.mean(accs)), float(np.mean(dm_accs))


def within_between(a: NDArray[np.floating], b: NDArray[np.floating]) -> dict[str, float]:
    """Degeneracy diagnostic: total within-class variance vs squared mean distance.

    Args:
        a: Class-one features ``(n, d)``.
        b: Class-other features ``(n, d)``.

    Returns:
        Dict with ``trace_within`` (sum of per-feature within-class variances, both classes),
        ``mean_dist_sq`` and their ratio.
    """
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    tw = float(a64.var(axis=0).sum() + b64.var(axis=0).sum())
    md = float(((a64.mean(axis=0) - b64.mean(axis=0)) ** 2).sum())
    return {"trace_within": tw, "mean_dist_sq": md, "within_over_between": tw / md if md else 0}


def minmax(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Their global min-max scaling (``compute_and_plot_separability``), verbatim formula.

    Args:
        x: Array of LDR values.

    Returns:
        Scaled array (``nan`` everywhere if ``x`` contains ``inf``, as in their code).
    """
    with np.errstate(all="ignore"):
        return (x - x.min()) / (x.max() - x.min())


def score_space(
    ldr_fn: LdrFn,
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    pca_dim: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    """All LDR variants for one feature space.

    Args:
        ldr_fn: Their ``compute_linear_separability``.
        a: Class-one features ``(n, d)``.
        b: Class-other features ``(n, d)``.
        pca_dim: PCA target dimensionality for the dimension-matched variant.
        folds: CV folds for the held-out variant.
        seed: CV seed.

    Returns:
        Dict with native, PCA and held-out LDRs plus diagnostics.
    """
    cv, acc, dm_acc = cv_ldr(a, b, folds, seed)
    return {
        "dim": int(a.shape[1]),
        "ldr": their_ldr(ldr_fn, a, b),
        f"ldr_pca{pca_dim}": pca_ldr(ldr_fn, a, b, pca_dim),
        "ldr_heldout": cv,
        "lda_heldout_acc": acc,
        "diffmean_heldout_acc": dm_acc,
        **within_between(a, b),
    }


def patch_heatmap(
    ldr_fn: LdrFn, a: NDArray[np.float32], b: NDArray[np.float32], n_jobs: int
) -> NDArray[np.float64]:
    """Their (layer, patch) LDR loop, unscaled.

    Args:
        ldr_fn: Their ``compute_linear_separability``.
        a: Activations ``(L, n, P, D)`` for class one.
        b: Activations ``(L, n, P, D)`` for class other.
        n_jobs: joblib workers.

    Returns:
        ``(L, P)`` unscaled LDR.
    """
    n_layers, _, n_patches, _ = a.shape
    cells = list(product(range(n_layers), range(n_patches)))
    vals = Parallel(n_jobs=n_jobs)(
        delayed(their_ldr)(ldr_fn, a[layer, :, patch, :], b[layer, :, patch, :])
        for layer, patch in cells
    )
    out = np.zeros((n_layers, n_patches))
    for (layer, patch), v in zip(cells, vals, strict=True):
        out[layer, patch] = v
    return out


# --------------------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------------------
def jnum(x: float) -> float | str:
    """JSON-safe float (``inf``/``nan`` become strings), 6 significant digits."""
    if x is None:
        return "nan"
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(f"{x:.6g}")


def jtree(obj: Any) -> Any:
    """Recursively convert numpy types / non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {k: jtree(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jtree(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jtree(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return jnum(float(obj))
    return obj


def sha(x: NDArray[np.floating]) -> str:
    """Short SHA-256 of an array's bytes."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def analyse_moment(
    wil: dict[str, Any],
    acts_a: NDArray[np.float32],
    acts_b: NDArray[np.float32],
    args: argparse.Namespace,
    plot_path: Path | None,
    keep_heatmaps: bool,
) -> dict[str, Any]:
    """Their per-(layer, patch) and patch-mean LDR on MOMENT activations, plus variants.

    Args:
        wil: Output of :func:`import_wilinski`.
        acts_a: MOMENT activations ``(L, n, P, D)`` of dataset 1.
        acts_b: MOMENT activations ``(L, n, P, D)`` of dataset 2.
        args: CLI arguments.
        plot_path: Where to save their heatmap plot (``None`` to skip).
        keep_heatmaps: Store the full unscaled / scaled ``(L, P)`` heatmaps in the result.

    Returns:
        Result dict with a summary, per-layer rows and (optionally) heatmaps.
    """
    ldr_fn = wil["compute_linear_separability"]
    k, folds, seed = args.pca_dim, args.cv_folds, args.seed
    n_layers = acts_a.shape[0]

    heat = patch_heatmap(ldr_fn, acts_a, acts_b, args.n_jobs)
    pooled_a, pooled_b = acts_a.mean(axis=2), acts_b.mean(axis=2)
    per_layer_raw = Parallel(n_jobs=min(args.n_jobs, n_layers))(
        delayed(score_space)(ldr_fn, pooled_a[layer], pooled_b[layer], k, folds, seed)
        for layer in range(n_layers)
    )
    pooled = np.array([r["ldr"] for r in per_layer_raw])
    heat_scaled, pooled_scaled = minmax(heat), minmax(pooled)

    finite_heat = np.where(np.isfinite(heat), heat, -np.inf)
    best_cell = np.unravel_index(int(np.argmax(finite_heat)), heat.shape)
    best_cell_scores = score_space(
        ldr_fn,
        acts_a[best_cell[0], :, best_cell[1], :],
        acts_b[best_cell[0], :, best_cell[1], :],
        k,
        folds,
        seed,
    )

    per_layer = []
    for layer in range(n_layers):
        row = heat[layer]
        fin = np.where(np.isfinite(row), row, -np.inf)
        per_layer.append(
            {
                "layer": layer,
                "pooled_ldr": per_layer_raw[layer]["ldr"],
                "pooled_ldr_scaled": pooled_scaled[layer],
                f"pooled_ldr_pca{k}": per_layer_raw[layer][f"ldr_pca{k}"],
                "pooled_ldr_heldout": per_layer_raw[layer]["ldr_heldout"],
                "pooled_lda_heldout_acc": per_layer_raw[layer]["lda_heldout_acc"],
                "pooled_diffmean_heldout_acc": per_layer_raw[layer]["diffmean_heldout_acc"],
                "pooled_within_over_between": per_layer_raw[layer]["within_over_between"],
                "patch_ldr_max": float(row[int(np.argmax(fin))]),
                "patch_ldr_argmax": int(np.argmax(fin)),
                "patch_ldr_median": float(np.nanmedian(row)),
                "patch_ldr_n_nonfinite": int((~np.isfinite(row)).sum()),
            }
        )

    def pick(key: str) -> dict[str, Any]:
        vals = np.array([r[key] for r in per_layer_raw], dtype=np.float64)
        fin = np.where(np.isfinite(vals), vals, -np.inf)
        i = int(np.argmax(fin)) if np.isfinite(vals).any() else int(np.argmax(vals))
        return {"layer": i, "value": float(vals[i]), "n_nonfinite": int((~np.isfinite(vals)).sum())}

    finite_vals = np.where(np.isfinite(heat), heat, np.nan)
    summary = {
        "best_pooled_layer_ldr": pick("ldr"),
        f"best_pooled_layer_ldr_pca{k}": pick(f"ldr_pca{k}"),
        "best_pooled_layer_ldr_heldout": pick("ldr_heldout"),
        "best_pooled_layer_lda_heldout_acc": pick("lda_heldout_acc"),
        "best_pooled_layer_diffmean_heldout_acc": pick("diffmean_heldout_acc"),
        "best_patch_cell": {
            "layer": int(best_cell[0]),
            "patch": int(best_cell[1]),
            "ldr": float(heat[best_cell]),
            f"ldr_pca{k}": best_cell_scores[f"ldr_pca{k}"],
            "ldr_heldout": best_cell_scores["ldr_heldout"],
            "lda_heldout_acc": best_cell_scores["lda_heldout_acc"],
        },
        "heatmap_n_nonfinite": int((~np.isfinite(heat)).sum()),
        "heatmap_min": float(np.nanmin(finite_vals)),
        "heatmap_median": float(np.nanmedian(finite_vals)),
        "heatmap_max_finite": float(np.nanmax(finite_vals)),
        "pooled_scaled_argmax_layer": int(np.nanargmax(pooled_scaled))
        if np.isfinite(pooled_scaled).any()
        else None,
        "pooled_scaled_curve": pooled_scaled,
        "heatmap_scaled_layer_mean": np.nanmean(heat_scaled, axis=1)
        if np.isfinite(heat_scaled).any()
        else None,
    }

    if plot_path is not None:
        import matplotlib.pyplot as plt

        wil["plot_linear_separability"](
            heat_scaled, pooled_scaled, n_layers, output_file=str(plot_path)
        )
        plt.close("all")

    out: dict[str, Any] = {"summary": summary, "per_layer": per_layer}
    if keep_heatmaps:
        out["heatmap_unscaled"] = heat
        out["heatmap_scaled"] = heat_scaled
    return out


def analyse_controls(
    wil: dict[str, Any],
    raw_a: NDArray[np.float64],
    raw_b: NDArray[np.float64],
    n_patches: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Score the no-model controls with their LDR on the series that were fed to MOMENT.

    Args:
        wil: Output of :func:`import_wilinski`.
        raw_a: Raw series ``(n, 512)`` of dataset 1.
        raw_b: Raw series ``(n, 512)`` of dataset 2.
        n_patches: Number of MOMENT patches (for the per-patch raw control).
        args: CLI arguments.

    Returns:
        Dict keyed by control name.
    """
    ldr_fn = wil["compute_linear_separability"]
    k, folds, seed = args.pca_dim, args.cv_folds, args.seed
    rev_a, rev_b = revin_input(raw_a), revin_input(raw_b)
    controls: dict[str, Any] = {
        "raw_signal": score_space(ldr_fn, raw_a, raw_b, k, folds, seed),
        "raw_revin": score_space(ldr_fn, rev_a, rev_b, k, folds, seed),
        "hand_crafted": score_space(
            ldr_fn, hand_crafted_features(raw_a), hand_crafted_features(raw_b), k, folds, seed
        ),
    }
    for name, (xa, xb) in {"raw_patch": (raw_a, raw_b), "raw_revin_patch": (rev_a, rev_b)}.items():
        vals = np.array(
            [
                their_ldr(
                    ldr_fn,
                    xa[:, p * PATCH_LEN : (p + 1) * PATCH_LEN],
                    xb[:, p * PATCH_LEN : (p + 1) * PATCH_LEN],
                )
                for p in range(n_patches)
            ]
        )
        fin = np.where(np.isfinite(vals), vals, -np.inf)
        controls[name] = {
            "dim": PATCH_LEN,
            "per_patch_ldr": vals,
            "max_ldr": float(vals[int(np.argmax(fin))]),
            "argmax_patch": int(np.argmax(fin)),
            "median_ldr": float(np.nanmedian(vals)),
            "n_nonfinite": int((~np.isfinite(vals)).sum()),
        }
    return controls


def build_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--wilinski-repo", type=Path, required=True, help="Clone of their repo")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wilinski_moment_ldr"))
    p.add_argument("--work-dir", type=Path, default=None, help="Where their parquet files go")
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 512])
    p.add_argument("--data-seed", type=int, default=42, help="Their generate --seed (default 42)")
    p.add_argument(
        "--dropout-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="torch seeds for the faithful (train-mode, dropout on) extraction",
    )
    p.add_argument("--no-eval-mode", action="store_true", help="Skip the eval-mode variant")
    p.add_argument("--seed", type=int, default=0, help="Seed for CV folds / PCA")
    p.add_argument("--pca-dim", type=int, default=8)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    repo = args.wilinski_repo.resolve()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or (repo / "steering" / "_tsfmi_work")).resolve()
    t0 = time.time()

    wil = import_wilinski(repo)
    commit = git_commit(repo)
    paths, order = generate_their_datasets(wil, repo, work_dir, args.data_seed)
    needed = sorted({d for _, d1, d2, _, _ in PAIRS for d in (d1, d2)})
    sizes = sorted(args.sample_sizes)
    n_small, n_max = sizes[0], sizes[-1]

    raw: dict[str, NDArray[np.float64]] = {}
    tens: dict[str, torch.Tensor] = {}
    for name in needed:
        tens[name] = wil["load_dataset"](str(paths[name]), type="torch", device=str(device))
        raw[name] = wil["load_dataset"](str(paths[name]), type="numpy")[:, 0, :].astype(np.float64)

    # Load once through the cache with their exact arguments, then inspect.
    model = wil["factory"].from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction", "device": device},
    )
    model.init()
    model.to(device)
    loaded_training_mode = bool(model.training)

    runs: list[tuple[str, int]] = [("train_dropout", s) for s in args.dropout_seeds]
    if not args.no_eval_mode:
        runs.append(("eval", 0))
    primary_runs = {f"train_dropout_seed{args.dropout_seeds[0]}", "eval"}

    results: dict[str, Any] = {}
    for pair_id, d1, d2, their_type, role in PAIRS:
        results[pair_id] = {"dataset1": d1, "dataset2": d2, "their_type": their_type, "role": role}
        for n in sizes:
            results[pair_id][f"n{n}"] = {
                "n_per_class": n,
                "series_sha256_16": {"dataset1": sha(raw[d1][:n]), "dataset2": sha(raw[d2][:n])},
                "controls": analyse_controls(wil, raw[d1][:n], raw[d2][:n], 64, args),
                "moment": {},
            }

    batch_consistency: dict[str, float] = {}
    for mode, run_seed in runs:
        run_id = "eval" if mode == "eval" else f"train_dropout_seed{run_seed}"
        model.train(mode != "eval")
        acts: dict[tuple[str, int], NDArray[np.float32]] = {}
        for di, name in enumerate(needed):
            for ni, n in enumerate(sizes):
                torch.manual_seed(run_seed * 1000 + di * 10 + ni)
                # Their extract_activations: dataset[:num_samples], one call. For n <= batch
                # size this is literally their single-batch call.
                acts[(name, n)] = moment_activations(
                    wil, tens[name][:n], str(device), n if n <= args.batch_size else args.batch_size
                )
            if mode == "eval" and n_small != n_max:
                diff = np.abs(acts[(name, n_small)] - acts[(name, n_max)][:, :n_small]).max()
                batch_consistency[name] = float(diff)
        print(f"[{run_id}] activations extracted ({time.time() - t0:.0f}s)", flush=True)

        for pair_id, d1, d2, _, _ in PAIRS:
            for n in sizes:
                primary = run_id in primary_runs
                plot = (
                    out_dir / f"heatmap_{pair_id}_n{n}_{run_id}.png"
                    if primary and not args.no_plots
                    else None
                )
                res = analyse_moment(wil, acts[(d1, n)], acts[(d2, n)], args, plot, primary)
                results[pair_id][f"n{n}"]["moment"][run_id] = res
                c = results[pair_id][f"n{n}"]["controls"]
                bp, bc = res["summary"]["best_pooled_layer_ldr"], res["summary"]["best_patch_cell"]
                print(
                    f"[{run_id} {pair_id} n={n}] MOMENT pooled best L{bp['layer']}="
                    f"{bp['value']:.4g} cell max={bc['ldr']:.4g}"
                    f" | raw={c['raw_signal']['ldr']:.4g} revin={c['raw_revin']['ldr']:.4g}"
                    f" hc={c['hand_crafted']['ldr']:.4g} rawpatch={c['raw_patch']['max_ldr']:.4g}",
                    flush=True,
                )
        del acts

    cfg = model.config
    model_info = {
        "checkpoint": "AutonLab/MOMENT-1-large",
        "task_name": str(model.task_name),
        "training_mode_after_their_load": loaded_training_mode,
        "dropout_p": float(model.encoder.config.dropout_rate),
        "mask_ratio": float(model.mask_generator.mask_ratio),
        "revin_eps": float(model.normalizer.eps),
        "revin_affine": bool(model.normalizer.affine),
        "d_model": int(cfg.d_model),
        "n_layers": len(model.encoder.block),
        "patch_len": int(model.patch_len),
        "hook": "encoder.block[i].layer[-1] forward output (residual stream after FFN)",
        "input_mask": "all ones (their create_prediction_mask_MOMENT(b, 512, 512))",
        "eval_mode_max_abs_diff_first_n_small_single_batch_vs_chunked_n_max": batch_consistency,
        "runs": {
            "train_dropout_seed*": "faithful: model left in the train() state their loading "
            "path produces (safetensors branch of PyTorchModelHubMixin never calls eval()); "
            "T5 dropout active under torch.no_grad",
            "eval": "sensitivity: same pipeline with model.eval() (dropout off, deterministic)",
        },
    }

    env_prefix = f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} PYTHONPATH=."
    payload = {
        "description": (
            "MOMENT-1-large Fisher LDR (Wilinski et al. ICML 2025 pipeline) vs raw-signal and "
            "hand-crafted controls on identical series. LDR = their "
            "steertool.separability.compute_linear_separability (in-sample sklearn LDA, Fisher "
            "ratio of the 1-D projection). *_scaled = their global min-max scaling."
        ),
        "wilinski_repo": "https://github.com/moment-timeseries-foundation-model/representations-in-tsfms",
        "wilinski_commit": commit,
        "command": f"{env_prefix} {shlex.join([sys.executable, *sys.argv])}",
        "data": {
            "generator": "steertool.dataset_generator.generate_datasets (their code, unmodified)",
            "data_seed": args.data_seed,
            "config_order_os_listdir": order,
            "n_series_per_dataset": int(raw[needed[0]].shape[0]),
            "length": int(raw[needed[0]].shape[1]),
            "series_sha256_16_full": {k: sha(v) for k, v in raw.items()},
        },
        "model": model_info,
        "settings": {
            "sample_sizes": sizes,
            "dropout_seeds": args.dropout_seeds,
            "pca_dim": args.pca_dim,
            "cv_folds": args.cv_folds,
            "cv_seed": args.seed,
            "batch_size": args.batch_size,
            "device": str(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "momentfm": __import__("importlib.metadata").metadata.version("momentfm"),
        },
        "pairs": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (out_dir / "results.json").write_text(json.dumps(jtree(payload), indent=1))
    print(f"Saved {out_dir / 'results.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
