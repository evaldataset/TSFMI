"""Generate the ICDE manuscript figures from TSFMI result files.

Like numbers.tex, every figure is drawn from outputs/ and regenerated before each build, so a
figure cannot disagree with the tables.

Usage (from the repository root; `make icde-numbers` runs this):
    PYTHONPATH=. python tools/make_figures.py --out outputs/paper_numbers/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [
    ("moment", "MOMENT"),
    ("gpt4ts", "GPT-2 (rand. emb.)"),
    ("timer", "Timer"),
    ("chronos", "Chronos-Bolt"),
    ("moirai", "Moirai 2.0"),
    ("patchtst_fm", "PatchTST-FM"),
    ("timesfm", "TimesFM 2.0"),
]
PROPS = [
    ("trend", "Trend"),
    ("seasonality", "Seas."),
    ("frequency", "Freq."),
    ("stationarity", "Stat."),
    ("anomaly", "Anom."),
    ("change_point", "CP"),
]
MODEL_COLOR, HC_COLOR, BAYES_COLOR, RAW_COLOR = "#4C72B0", "#C44E52", "#55A868", "#8C8C8C"

plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "serif",
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load(p: Path) -> object:
    return json.loads(p.read_text())


def anomaly_figure(root: Path, out: Path) -> None:
    """Canonical vs realistic anomaly accuracy per model with non-model references.

    Intervals are the corrected resampled t-intervals used for every number in the paper.
    """
    from make_numbers import corrected_t  # same directory

    def ci(per_seed: list[float]) -> tuple[float, float, float]:
        mean, lo, hi, _ = corrected_t(per_seed)
        return mean, lo, hi

    canon = {
        m: load(root / f"outputs/canonical_v2/{m}_anomaly/canonical_results.json")
        for m, _ in MODELS
    }
    real = {
        m: load(root / f"outputs/realistic_anomaly_v2/{m}_anomaly_realistic/canonical_results.json")
        for m, _ in MODELS
    }
    base = {
        r["baseline"]: r
        for r in load(root / "outputs/canonical_v2/baselines/all_results.json")
        if r["property"] == "anomaly"
    }
    rbase = {
        r["baseline"]: r
        for r in load(root / "outputs/realistic_anomaly/realistic_anomaly_results.json")["rows"]
    }
    bayes = load(root / "outputs/bayes_ceiling/results.json")["bayes_rule"]
    rocket = {
        "canonical": next(
            r["test_mean"]
            for r in load(root / "outputs/rocket_baselines/rocket_results.json")
            if r["dataset"] == "anomaly"
        ),
        "realistic": load(root / "outputs/strong_controls_ext/results.json")["realistic"]["42"][
            "rocket"
        ]["mean"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    panels = [
        (axes[0], canon, base, "Canonical (single spike)", "canonical"),
        (axes[1], real, rbase, "Realistic (mixed type)", "realistic"),
    ]
    y = np.arange(len(MODELS))
    for ax, res, b, title, task in panels:
        stats = [ci(res[m]["test_score_per_seed"]) for m, _ in MODELS]
        means = np.array([t[0] for t in stats])
        lo = means - np.array([t[1] for t in stats])
        hi = np.array([t[2] for t in stats]) - means
        ax.barh(
            y,
            means,
            xerr=[lo, hi],
            color=MODEL_COLOR,
            height=0.6,
            error_kw={"elinewidth": 0.7, "capsize": 1.5},
        )
        hc_mean, hc_lo, hc_hi = ci(b["hand_crafted"]["per_seed_test"])
        ax.axvspan(hc_lo, hc_hi, color=HC_COLOR, alpha=0.15, lw=0)
        ax.axvline(hc_mean, color=HC_COLOR, lw=1.1, label="hand-crafted (8-D)")
        ax.axvline(rocket[task], color="#8172B2", lw=1.1, ls="-.", label="ROCKET-style")
        raw = b["raw_signal"]["test_mean"]
        ax.axvline(raw, color=RAW_COLOR, lw=1.0, ls=":", label="raw signal")
        if task == "canonical":
            ax.axvline(bayes["mean"], color=BAYES_COLOR, lw=1.1, ls="--", label="Bayes-optimal")
        ax.set_xlim(0.45, 0.95)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("test accuracy")
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(y, [name for _, name in MODELS])
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    labels = [lab for ax in axes for lab in ax.get_legend_handles_labels()[1]]
    uniq = dict(zip(labels, handles, strict=True))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.55, -0.06),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "anomaly_canonical_vs_realistic.pdf", bbox_inches="tight")
    plt.close(fig)


def selectivity_figure(root: Path, out: Path) -> None:
    """Control-task selectivity heatmaps for k=20 and k=50 input clusters."""
    rows = load(root / "outputs/hl_selectivity_v2/results.json")["results"]
    order = [m for m, _ in MODELS] + ["raw_signal", "random_projection", "hand_crafted"]
    names = [n for _, n in MODELS] + ["Raw signal", "Random proj.", "Hand-crafted"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, k in zip(axes, (20, 50), strict=True):
        table: dict[str, dict[str, float]] = {}
        for r in rows:
            if r.get("k") != k or "selectivity" not in r:
                continue
            if r["row_kind"] != "model" and r["num_samples"] != 1000:
                continue
            table.setdefault(r["row"], {})[r["property"]] = r["selectivity"]["mean"]
        data = np.array([[table[r][p] for p, _ in PROPS] for r in order])
        im = ax.imshow(np.clip(data, -0.4, 1.2), cmap="RdBu_r", vmin=-0.4, vmax=1.2, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(data[i, j]) > 0.8 else "black",
                )
        ax.axhline(len(MODELS) - 0.5, color="black", lw=0.8)
        ax.set_xticks(range(len(PROPS)), [n for _, n in PROPS], rotation=35, ha="right")
        ax.set_title(f"$k={k}$ input clusters", fontsize=8)
    axes[0].set_yticks(range(len(names)), names)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="selectivity")
    fig.savefig(out / "selectivity_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def task_examples_figure(root: Path, out: Path) -> None:
    """One window of each of two classes for every canonical, hard, and realistic task."""
    import sys

    sys.path.insert(0, str(root))
    import src.datasets.synthetic as syn  # noqa: PLC0415
    from scripts.run_realistic_anomaly import realistic_anomaly_dataset  # noqa: PLC0415

    def gen(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name == "realistic":
            x, y = realistic_anomaly_dataset(1000, 512, seed=42)
            return np.asarray(x), np.asarray(y)
        d = getattr(syn, f"generate_{name}_dataset")(1000, 512, seed=42)
        return d.sequences, d.labels

    panels = [
        [
            ("trend", "Trend"),
            ("seasonality", "Seasonality"),
            ("frequency", "Frequency"),
            ("stationarity", "Stationarity"),
            ("anomaly", "Anomaly"),
            ("change_point", "Change pt."),
        ],
        [
            ("trend_hard", "Trend (hard)"),
            ("frequency_hard", "Freq. (hard)"),
            ("stationarity_hard", "Stat. (hard)"),
            ("anomaly_hard", "Anomaly (hard)"),
            ("change_point_hard", "Change pt. (hard)"),
            ("realistic", "Anomaly (realistic)"),
        ],
    ]
    fig, axes = plt.subplots(2, 6, figsize=(7.0, 2.2))
    for row, names in zip(axes, panels, strict=True):
        for ax, (name, title) in zip(row, names, strict=True):
            x, y = gen(name)
            lo, hi = (np.argmin(y), np.argmax(y))
            ax.plot(x[lo], lw=0.5, color=RAW_COLOR, label="class min")
            ax.plot(x[hi], lw=0.5, color=HC_COLOR, alpha=0.85, label="class max")
            ax.set_title(title, fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    fig.savefig(out / "task_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def learning_curve_figure(root: Path, out: Path) -> None:
    """Anomaly test accuracy vs number of labelled training windows (canonical, realistic)."""
    f = root / "outputs/anomaly_diagnostics/results.json"
    if not f.exists():
        print("note: anomaly_diagnostics missing; learning-curve figure skipped")
        return
    d = load(f)
    sizes = d["train_sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, (task, title) in zip(
        axes,
        (("canonical", "Canonical (single spike)"), ("realistic", "Realistic (mixed type)")),
        strict=True,
    ):
        rows = d["learning_curve"][task]
        for i, (m, name) in enumerate(MODELS):
            ys = [rows[m][str(n)]["mean"] for n in sizes]
            ax.plot(sizes, ys, marker="o", ms=2.5, lw=0.9, color=cmap(i), label=name)
        for key, name, color, ls in (
            ("hand_crafted", "Hand-crafted", HC_COLOR, "--"),
            ("raw_signal", "Raw signal", RAW_COLOR, ":"),
        ):
            ys = [rows[key][str(n)]["mean"] for n in sizes]
            ax.plot(sizes, ys, color=color, ls=ls, lw=1.4, marker="s", ms=2.5, label=name)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("labelled training windows")
        ax.set_title(title, fontsize=8)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("test accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12)
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "learning_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tsfmi", type=Path, default=Path("."), help="TSFMI repository root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.tsfmi.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    anomaly_figure(root, args.out)
    selectivity_figure(root, args.out)
    task_examples_figure(root, args.out)
    learning_curve_figure(root, args.out)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
