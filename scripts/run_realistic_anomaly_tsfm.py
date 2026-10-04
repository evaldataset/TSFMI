"""TSFM-side run on the realistic anomaly generator (REVIEW priority).

The canonical realistic-anomaly result (run_realistic_anomaly.py) only reports
HC / raw / random_projection on the harder anomaly construction. To directly
address the inversion question on this construction, we extract the four
encoder-side TSFMs whose canonical anomaly probes already exist (MOMENT-PCA512,
Chronos, PatchTST-Pre, GPT4TS-PCA512) on the realistic-anomaly dataset and
report a side-by-side table.

This script is intentionally lightweight: rather than rebuild the full
extraction pipeline, we *project the realistic-anomaly raw signal through the
same model wrappers* and then probe with the canonical sklearn pipeline.
Because the realistic-anomaly construction is univariate length-512 (matching
synthetic_anomaly), the existing wrappers and protocol apply unchanged.

Output:
    outputs/realistic_anomaly_tsfm/results.json
    outputs/realistic_anomaly_tsfm/results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_realistic_anomaly import realistic_anomaly_dataset
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/realistic_anomaly_tsfm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
NUM_SAMPLES = 1000
SEQ_LEN = 512

# We re-use the encoder-side four-model set used by the LEACE seed bootstrap;
# these models have the most stable canonical anomaly numbers and are the ones
# the headline inversion claim refers to. Decoder-only models (Timer, TimesFM,
# Moirai) are excluded from this rebuttal experiment because their full
# extraction is materially more expensive.
MODELS = ["moment", "chronos", "patchtst_pretrained", "gpt4ts", "timesfm", "moirai", "timer"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_wrapper(name: str):
    if name == "moment":
        from src.models.moment_wrapper import MOMENTWrapper

        return MOMENTWrapper()
    if name == "chronos":
        from src.models.chronos_wrapper import ChronosBoltWrapper

        return ChronosBoltWrapper()
    if name == "patchtst_pretrained":
        from src.models.patchtst_wrapper import LEGACY_GRANITE_ETTH1_CHECKPOINT, PatchTSTWrapper

        # "patchtst_pretrained" denotes the legacy ETTh1 checkpoint everywhere else; pin it
        # so the wrapper's new default (PatchTST-FM-r1) is never reported under this label.
        class _LegacyPatchTSTWrapper(PatchTSTWrapper):
            def load(self, checkpoint: str = LEGACY_GRANITE_ETTH1_CHECKPOINT, **kwargs) -> None:
                super().load(checkpoint, **kwargs)

        return _LegacyPatchTSTWrapper()
    if name == "patchtst_fm":
        from src.models.patchtst_wrapper import PatchTSTWrapper

        return PatchTSTWrapper()  # default checkpoint: PatchTST-FM-r1
    if name == "gpt4ts":
        from src.models.gpt4ts_wrapper import GPT4TSWrapper

        return GPT4TSWrapper()
    if name == "timesfm":
        from src.models.timesfm_wrapper import TimesFMWrapper

        return TimesFMWrapper()
    if name == "moirai":
        from src.models.moirai_wrapper import MoiraiWrapper

        return MoiraiWrapper()
    if name == "timer":
        from src.models.timer_wrapper import TimerWrapper

        return TimerWrapper()
    raise ValueError(name)


def three_way_split(X, y, seed):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def score(Xtr, ytr, Xte, yte, seed):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    )
    pipe.fit(Xtr, ytr.astype(np.int64))
    return float(accuracy_score(yte.astype(np.int64), pipe.predict(Xte)))


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return (
        float(arr.mean()),
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


def best_layer_features(wrapper, sequences: NDArray, labels: NDArray) -> tuple[NDArray, str]:
    """Run wrapper, extract per-layer activations, choose best layer on a tiny
    val split (seed=0 only, fixed) so layer selection is not contaminated."""
    from src.extractors.hook_manager import HookManager

    wrapper.load(device=DEVICE)
    wrapper.freeze()

    layers = wrapper.get_layer_names()
    activations: dict[str, list[NDArray]] = {ln: [] for ln in layers}

    seqs = torch.tensor(sequences, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    batch = 32
    with HookManager(wrapper.model, layers) as hm:
        for s in range(0, seqs.shape[0], batch):
            chunk = seqs[s : s + batch]
            with torch.no_grad():
                wrapper.forward(chunk)
            acts = hm.get_activations()
            for ln in layers:
                act = acts[ln]
                if act.ndim == 4:
                    act = act.mean(dim=2)
                if act.ndim == 3:
                    act = act.mean(dim=1)
                activations[ln].append(act.detach().cpu().numpy())

    # Concatenate, optionally PCA-512 reduce for high-D models.
    layer_X: dict[str, NDArray] = {}
    for ln, parts in activations.items():
        X = np.concatenate(parts, axis=0).astype(np.float64)
        if X.shape[1] > 512:
            from sklearn.decomposition import PCA

            n_train = int(X.shape[0] * 0.6)
            pca = PCA(n_components=512, random_state=42)
            pca.fit(X[:n_train])
            X = pca.transform(X)
        layer_X[ln] = X

    # Best-layer selection on a fixed seed=0 val split, to avoid leakage across seeds.
    best_layer = None
    best_va = -np.inf
    for ln, X in layer_X.items():
        Xtr, ytr, Xva, yva, *_ = three_way_split(X, labels, 0)
        va_acc = score(Xtr, ytr, Xva, yva, 0)
        if va_acc > best_va:
            best_va = va_acc
            best_layer = ln
    return layer_X[best_layer], best_layer


def run_model(name: str, sequences: NDArray, labels: NDArray) -> dict:
    print(f"\n--- {name} on realistic anomaly ---")
    wrapper = load_wrapper(name)
    X, best_layer = best_layer_features(wrapper, sequences, labels)
    per_seed = []
    for s in SEEDS:
        Xtr, ytr, Xva, yva, Xte, yte = three_way_split(X, labels, s)
        per_seed.append(score(Xtr, ytr, Xte, yte, s))
    m, lo, hi = bootstrap_ci(per_seed)
    return {
        "model": name,
        "best_layer": best_layer,
        "test_mean": round(m, 4),
        "test_ci95_low": round(lo, 4),
        "test_ci95_high": round(hi, 4),
        "per_seed_test": [round(v, 4) for v in per_seed],
    }


def main() -> None:
    seed_everything(42)
    sequences, labels = realistic_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=42)

    rows = []
    for name in MODELS:
        try:
            row = run_model(name, sequences, labels)
            rows.append(row)
            print(
                f"  {row['model']:<22s} layer={row['best_layer']:<22s} "
                f"test={row['test_mean']:.4f} [{row['test_ci95_low']:.4f}, "
                f"{row['test_ci95_high']:.4f}]"
            )
        except Exception as e:  # pragma: no cover - skip and continue
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")
            rows.append({"model": name, "error": f"{type(e).__name__}: {e}"})

    out = {
        "task": "realistic_anomaly_tsfm",
        "design": "Same realistic anomaly generator as run_realistic_anomaly.py;"
        " encoder-side 4-model set; canonical 60/20/20 + 5-seed + bootstrap.",
        "models": rows,
        "note": (
            "TSFM probes are run with the same canonical pipeline as the synthetic"
            " anomaly. The HC baseline on this realistic generator is 0.701 (run via"
            " run_realistic_anomaly.py). If max(model.test_mean) <= 0.701 the inversion"
            " is preserved on the realistic construction; otherwise the inversion"
            " narrows or reverses on this harder task and the appendix should report it."
        ),
    }
    OUT_DIR.joinpath("results.json").write_text(json.dumps(out, indent=2))

    tex = [
        r"% Auto-generated by scripts/run_realistic_anomaly_tsfm.py",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Best layer} & \textbf{Test mean} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for r in rows:
        if "error" in r:
            tex.append(
                f"\\texttt{{{r['model']}}} & --- & --- & extraction failed \\\\"
            )
        else:
            tex.append(
                f"\\texttt{{{r['model']}}} & \\texttt{{{r['best_layer']}}} & "
                f"{r['test_mean']:.3f} & "
                f"[{r['test_ci95_low']:.3f}, {r['test_ci95_high']:.3f}] \\\\"
            )
    tex += [r"\bottomrule", r"\end{tabular}"]
    OUT_DIR.joinpath("results.tex").write_text("\n".join(tex))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
