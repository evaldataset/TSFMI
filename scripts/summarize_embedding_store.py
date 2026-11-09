"""Summarise the 1-NN embedding-store benchmark (outputs/embedding_store/{classical,models}.json).

For every dataset with all representations: per-representation mean accuracy, win/tie/loss counts
against ED and DTW, two-sided Wilcoxon signed-rank tests over datasets with Holm correction across
the seven models, mean ranks, and storage per series. "Best TSFM" picks, per dataset, the model with
the highest leave-one-out 1-NN accuracy on the training split, so it uses no test labels.

Usage:
    PYTHONPATH=. python scripts/summarize_embedding_store.py --dir outputs/embedding_store
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

MODELS = ("timesfm", "moment", "patchtst_fm", "gpt4ts", "chronos", "moirai", "timer")
TIE = 1e-9


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def tost_mean(d: np.ndarray, margin: float, alpha: float = 0.05) -> dict[str, float | bool]:
    """Two one-sided t tests that the mean per-dataset difference lies within +-margin."""
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n))
    mean = float(d.mean())
    t_crit = float(stats.t.ppf(1 - alpha, n - 1))
    p_lo = float(stats.t.sf((mean + margin) / se, n - 1))
    p_hi = float(stats.t.cdf((mean - margin) / se, n - 1))
    return {
        "ci90": [mean - t_crit * se, mean + t_crit * se],
        "p": max(p_lo, p_hi),
        "equivalent": bool(max(p_lo, p_hi) < alpha),
    }


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired comparison over datasets: wins/ties/losses, Wilcoxon difference test, and TOST
    equivalence of the mean accuracy difference within 0.02 and 0.01 (a non-significant Wilcoxon
    test is not evidence of equivalence)."""
    d = a - b
    wins, losses = int(np.sum(d > TIE)), int(np.sum(d < -TIE))
    nz = d[np.abs(d) > TIE]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    return {
        "wins": wins,
        "ties": int(len(d) - wins - losses),
        "losses": losses,
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "p_wilcoxon": p,
        "tost_0.02": tost_mean(d, 0.02),
        "tost_0.01": tost_mean(d, 0.01),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("outputs/embedding_store"))
    args = ap.parse_args()
    classical = json.loads((args.dir / "classical.json").read_text())
    models = json.loads((args.dir / "models.json").read_text())
    names = sorted(n for n in classical if all(m in models.get(n, {}) for m in MODELS))

    acc: dict[str, np.ndarray] = {
        "ed": np.array([classical[n]["ed_nn_acc"] for n in names]),
        "dtw": np.array([classical[n]["dtw_nn_acc"] for n in names]),
        "rocket": np.array([classical[n]["rocket_nn_acc"] for n in names]),
    }
    for m in MODELS:
        acc[m] = np.array([models[n][m]["nn_acc"] for n in names])
    best_by_loo = [max(MODELS, key=lambda m, n=n: models[n][m]["nn_loo_train"]) for n in names]
    acc["best_tsfm"] = np.array(
        [models[n][m]["nn_acc"] for n, m in zip(names, best_by_loo, strict=True)]
    )
    rpath = args.dir / "models_random.json"
    random_models = json.loads(rpath.read_text()) if rpath.exists() else {}
    if all(all(m in random_models.get(n, {}) for m in MODELS) for n in names):
        for m in MODELS:
            acc[f"{m}_random"] = np.array([random_models[n][m]["nn_acc"] for n in names])
    lin: dict[str, np.ndarray] = {
        "raw": np.array([classical[n]["raw_linear_acc"] for n in names]),
        "rocket": np.array([classical[n]["rocket_linear_acc"] for n in names]),
    }
    for m in MODELS:
        lin[m] = np.array([models[n][m]["linear_acc"] for n in names])
    spath = args.dir / "strong.json"  # written by run_embedding_store.py --part strong
    strong = json.loads(spath.read_text()) if spath.exists() else {}
    refs = ["ed", "dtw", "rocket"]
    if strong:
        missing = [n for n in names if n not in strong]
        if missing:
            raise SystemExit(f"strong.json lacks {len(missing)} datasets, e.g. {missing[:3]}")
        for key in ("minirocket", "catch22"):
            acc[key] = np.array([strong[n][f"{key}_nn_acc"] for n in names])
            lin[key] = np.array([strong[n][f"{key}_linear_acc"] for n in names])
        refs.append("minirocket")
    rpath = args.dir / "strong_resized.json"  # MiniRocket on the models' 512-step input
    resized = json.loads(rpath.read_text()) if rpath.exists() else {}
    if resized and all(n in resized for n in names):
        acc["minirocket512"] = np.array([resized[n]["minirocket_nn_acc"] for n in names])
        lin["minirocket512"] = np.array([resized[n]["minirocket_linear_acc"] for n in names])

    out: dict = {"n_datasets": len(names), "datasets": names, "nn": {}, "linear": {}}
    for k, v in acc.items():
        out["nn"][k] = {"mean_acc": float(v.mean())}
        if k not in ("ed", "dtw"):
            out["nn"][k]["vs_ed"] = compare(v, acc["ed"])
            out["nn"][k]["vs_dtw"] = compare(v, acc["dtw"])
        if k not in ("ed", "dtw", "rocket"):
            out["nn"][k]["vs_rocket"] = compare(v, acc["rocket"])
        if "minirocket" in refs and k not in ("ed", "dtw", "rocket", "minirocket"):
            out["nn"][k]["vs_minirocket"] = compare(v, acc["minirocket"])
        if "minirocket512" in acc and k in MODELS:
            out["nn"][k]["vs_minirocket512"] = compare(v, acc["minirocket512"])
    out["nn"]["dtw"]["vs_ed"] = compare(acc["dtw"], acc["ed"])
    for m in MODELS:
        if f"{m}_random" in acc:
            out["nn"][m]["vs_random_init"] = compare(acc[m], acc[f"{m}_random"])
    for ref in refs:
        adj = holm({m: out["nn"][m][f"vs_{ref}"]["p_wilcoxon"] for m in MODELS})
        for m in MODELS:
            out["nn"][m][f"vs_{ref}"]["p_holm"] = adj[m]
    ranked = (*refs, *(k for k in ("minirocket512", "catch22") if k in acc), *MODELS)
    table = np.stack([acc[k] for k in ranked], axis=1)
    ranks = stats.rankdata(-table, axis=1).mean(axis=0)
    out["nn_mean_rank"] = dict(zip(ranked, map(float, ranks), strict=True))
    out["nn"]["n_tsfm_beats_dtw"] = int(sum(acc[m].mean() > acc["dtw"].mean() for m in MODELS))
    for k, v in lin.items():
        out["linear"][k] = {"mean_acc": float(v.mean())}
        if k != "rocket":
            out["linear"][k]["vs_rocket"] = compare(v, lin["rocket"])
        if "minirocket" in lin and k not in ("rocket", "minirocket", "minirocket512"):
            out["linear"][k]["vs_minirocket"] = compare(v, lin["minirocket"])
        if "minirocket512" in lin and k in MODELS:
            out["linear"][k]["vs_minirocket512"] = compare(v, lin["minirocket512"])
    for ref in ("rocket", "minirocket") if "minirocket" in lin else ("rocket",):
        adj = holm({m: out["linear"][m][f"vs_{ref}"]["p_wilcoxon"] for m in MODELS})
        for m in MODELS:
            out["linear"][m][f"vs_{ref}"]["p_holm"] = adj[m]
    lengths = np.array([classical[n]["length"] for n in names])
    out["storage_bytes"] = {
        "raw_mean": float(4 * lengths.mean()),
        "raw_median": float(4 * np.median(lengths)),
        "rocket": 4 * 2048,
        **{m: 4 * int(models[names[0]][m]["dim"]) for m in MODELS},
    }
    out["seconds"] = {
        "ed": float(sum(classical[n]["ed_seconds"] for n in names)),
        "dtw": float(sum(classical[n]["dtw_seconds"] for n in names)),
        **{m: float(sum(models[n][m]["embed_seconds"] for n in names)) for m in MODELS},
    }
    out["best_tsfm_choice"] = {m: best_by_loo.count(m) for m in MODELS}
    (args.dir / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "datasets"}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
