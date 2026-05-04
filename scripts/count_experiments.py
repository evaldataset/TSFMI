"""Generate an auditable experiment-count manifest (PLAN.md T2.1).

Replaces the prose "2,500+ experiments" claim with a machine-countable
breakdown over the actual output directories. Output:
    outputs/paper/experiment_manifest.json
    outputs/paper/experiment_manifest.tex   (LaTeX table snippet)

Design:
- Each row of the manifest counts one *type* of experiment artifact.
- Runs counted as unique (model, property, dataset, layer) where layer
  metadata is available, else unique (model, property, dataset).
- Multi-seed repetitions are NOT counted (counted once at the layer level).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("outputs")
OUT_JSON = Path("outputs/paper/experiment_manifest.json")
OUT_TEX = Path("outputs/paper/latex/experiment_manifest.tex")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def count_layer_files(
    root: Path,
    layer_prefix: tuple[str, ...] = ("encoder_", "transformer_", "model_"),
) -> int:
    """Count .pt layer files (excluding labels/metadata) under a root."""
    if not root.exists():
        return 0
    n = 0
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.glob("*.pt"):
            if f.stem in ("labels", "metadata"):
                continue
            n += 1
    return n


def count_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for d in root.iterdir() if d.is_dir())


def count_json(root: Path, name: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for f in root.rglob(name))


def main() -> None:
    manifest: list[dict[str, object]] = []

    # 1. Layer-level probe runs (legacy PyTorch). Each metrics.json is one
    #    unique (model, property, dataset, layer) probe training run.
    probes_root = ROOT / "probes"
    layer_level_probes = len(list(probes_root.rglob("metrics.json"))) if probes_root.exists() else 0
    manifest.append(
        {
            "category": "Layer-level PyTorch probe runs",
            "description": (
                "One metrics.json per unique (model, property, dataset, layer) probe run"
            ),
            "count": layer_level_probes,
            "source_glob": "outputs/probes/**/metrics.json",
        }
    )
    probe_dirs = count_dirs(probes_root)
    manifest.append(
        {
            "category": "Probe-run aggregate directories",
            "description": (
                "One directory per (model, property, probe-type) training sweep; "
                "layer probes live inside"
            ),
            "count": probe_dirs,
            "source_glob": "outputs/probes/*/",
        }
    )

    # 2. Canonical sklearn runs (7 models x 6 properties)
    canonical_runs = count_dirs(ROOT / "canonical")
    manifest.append(
        {
            "category": "Canonical model runs (60/20/20, 5 seeds)",
            "description": "One (model, property) directory per canonical benchmark run",
            "count": canonical_runs,
            "source_glob": "outputs/canonical/*/canonical_results.json",
        }
    )

    # 3. Canonical baselines (3 baselines x 6 properties)
    canonical_baseline_runs = count_dirs(ROOT / "canonical_baselines")
    manifest.append(
        {
            "category": "Canonical baseline runs (60/20/20, 5 seeds)",
            "description": "Per-property baseline dirs (each contains HC, raw, random_projection)",
            "count": canonical_baseline_runs,
            "source_glob": "outputs/canonical_baselines/*/canonical_results.json",
        }
    )

    # 4. LEACE layer-level erasures (one layer per entry inside each JSON)
    leace_runs = count_dirs(ROOT / "leace")
    leace_layers = 0
    if (ROOT / "leace").exists():
        for jf in (ROOT / "leace").rglob("leace_results.json"):
            try:
                leace_layers += len(json.loads(jf.read_text()))
            except Exception:
                pass
    manifest.append(
        {
            "category": "LEACE layer-level erasures",
            "description": "One entry per (model, property, layer) linear-erasure evaluation",
            "count": leace_layers,
            "source_glob": "outputs/leace/**/leace_results.json (per-layer entries)",
        }
    )
    manifest.append(
        {
            "category": "LEACE run aggregate directories",
            "description": "One directory per (model, property) LEACE sweep",
            "count": leace_runs,
            "source_glob": "outputs/leace/*/",
        }
    )

    # 5. CKA pairwise artifacts
    cka_runs = count_dirs(ROOT / "cka")
    manifest.append(
        {
            "category": "CKA similarity artifacts",
            "description": "Cross-model and within-model CKA matrices",
            "count": cka_runs,
            "source_glob": "outputs/cka/*/",
        }
    )

    # 6. Intervention / steering artifacts
    interventions_runs = count_dirs(ROOT / "interventions")
    manifest.append(
        {
            "category": "Intervention / steering runs",
            "description": "LDA steering and random-direction control artifacts",
            "count": interventions_runs,
            "source_glob": "outputs/interventions/*/",
        }
    )

    # 7. Cross-property LEACE
    cross_leace_runs = count_dirs(ROOT / "cross_leace")
    manifest.append(
        {
            "category": "Cross-property LEACE entanglement runs",
            "description": "One directory per (model, erase, eval) concept pair",
            "count": cross_leace_runs,
            "source_glob": "outputs/cross_leace/*/",
        }
    )

    # 8. Hard-variant benchmark
    hard_entries = 0
    hard_table = ROOT / "hard_variant_benchmark/easy_vs_hard_table.json"
    if hard_table.exists():
        hard_entries = len(json.loads(hard_table.read_text()))
    manifest.append(
        {
            "category": "Hard-variant comparisons",
            "description": "Easy vs.\\ hard per (model, property) rows",
            "count": hard_entries,
            "source_glob": "outputs/hard_variant_benchmark/easy_vs_hard_table.json",
        }
    )

    # 9. Underlying (model, property, layer) probe files (the most granular unit)
    layer_probe_total = count_layer_files(ROOT / "representations")
    # Note: this counts extracted representation tensors, not probe runs; each
    # tensor can be probed once per (protocol, seed).
    manifest.append(
        {
            "category": "Extracted layer tensors (representations on disk)",
            "description": (
                "Per (model, dataset, layer) frozen representation files; "
                "each is the input unit to one probe"
            ),
            "count": layer_probe_total,
            "source_glob": "outputs/representations/<model>/<dataset>/<layer>.pt",
        }
    )

    total = sum(int(row["count"]) for row in manifest)

    # Write JSON
    OUT_JSON.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "total_experiment_artifacts": total,
                "notes": (
                    "Counts are unique experiment artifacts on disk. Multi-seed and "
                    "multi-bootstrap resamples are NOT counted separately; they live "
                    "inside the per-artifact JSON. The 'total' number represents the "
                    "union across all categories and therefore exceeds the headline "
                    "'probe-run' count that the paper footnote uses."
                ),
            },
            indent=2,
        )
    )

    # Write LaTeX snippet. Use \scriptsize + tabularx + \seqsplit so that the
    # longest source globs (50+ chars) break inside the column at the smaller
    # font size and never overflow the textwidth. \tabcolsep is also reduced
    # to give the X column maximum room.
    tex = [
        r"% Auto-generated by scripts/count_experiments.py -- do not edit by hand",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.5cm}r>{\raggedright\arraybackslash}X@{}}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Count} & \textbf{Source glob} \\",
        r"\midrule",
    ]
    for row in manifest:
        cat = str(row["category"]).replace("_", r"\_")
        glob = str(row["source_glob"]).replace("_", r"\_")
        tex.append(f"{cat} & {row['count']} & \\texttt{{\\seqsplit{{{glob}}}}} \\\\")
    tex.extend(
        [
            r"\midrule",
            rf"\textbf{{Total experiment artifacts}} & \textbf{{{total}}} & \textemdash \\",
            r"\bottomrule",
            r"\end{tabularx}",
            r"}",
        ]
    )
    OUT_TEX.write_text("\n".join(tex))

    # Print summary
    print(f"\n{'Category':<55s} {'Count':>8s}")
    print("-" * 64)
    for row in manifest:
        print(f"{str(row['category']):<55s} {int(row['count']):>8d}")
    print("-" * 64)
    print(f"{'TOTAL':<55s} {total:>8d}")
    print(f"\nSaved to: {OUT_JSON}")
    print(f"          {OUT_TEX}")


if __name__ == "__main__":
    main()
