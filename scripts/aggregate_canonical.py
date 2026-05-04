"""Aggregate canonical benchmark results into a single summary table.

Reads outputs/canonical/<model>_<property>/canonical_results.json for all
7 models x 6 properties and produces a markdown + latex summary.
"""

from __future__ import annotations

import json
from pathlib import Path

MODELS = [
    "moment_pca512",
    "chronos",
    "patchtst_pretrained",
    "gpt4ts_pca512",
    "timer_meanpool",
    "timesfm_meanpool",
    "moirai_meanpool",
]
PROPERTIES = ["trend", "seasonality", "frequency", "stationarity", "anomaly", "change_point"]
MODEL_DISPLAY = {
    "moment_pca512": "MOMENT",
    "chronos": "Chronos",
    "patchtst_pretrained": "PatchTST",
    "gpt4ts_pca512": "GPT4TS",
    "timer_meanpool": "Timer",
    "timesfm_meanpool": "TimesFM",
    "moirai_meanpool": "Moirai",
}

OUT = Path("outputs/canonical/summary.json")
LATEX_OUT = Path("outputs/canonical/summary_table.tex")


def main() -> None:
    summary: dict = {}
    for m in MODELS:
        summary[m] = {}
        for p in PROPERTIES:
            path = Path(f"outputs/canonical/{m}_{p}/canonical_results.json")
            if not path.exists():
                summary[m][p] = None
                continue
            data = json.loads(path.read_text())
            summary[m][p] = {
                "mean": data["test_mean"],
                "ci_low": data["test_ci95_low"],
                "ci_high": data["test_ci95_high"],
            }

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    # LaTeX table
    lines = [
        r"% Canonical rerun: train/val/test 60/20/20, 5 seeds, matched estimator",
        r"\begin{tabular}{l" + "c" * len(PROPERTIES) + "}",
        r"\toprule",
        "Model & " + " & ".join(p.replace("_", " ").title() for p in PROPERTIES) + r" \\",
        r"\midrule",
    ]
    for m in MODELS:
        row = [MODEL_DISPLAY[m]]
        for p in PROPERTIES:
            v = summary[m][p]
            if v is None:
                row.append("---")
            else:
                row.append(f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    LATEX_OUT.write_text("\n".join(lines))

    # Print markdown preview
    print("\n| Model | " + " | ".join(p[:8] for p in PROPERTIES) + " |")
    print("|" + "---|" * (len(PROPERTIES) + 1))
    for m in MODELS:
        row = [MODEL_DISPLAY[m]]
        for p in PROPERTIES:
            v = summary[m][p]
            row.append("---" if v is None else f"{v['mean']:.3f}")
        print("| " + " | ".join(row) + " |")

    print(f"\nSaved JSON to {OUT}")
    print(f"Saved LaTeX to {LATEX_OUT}")


if __name__ == "__main__":
    main()
