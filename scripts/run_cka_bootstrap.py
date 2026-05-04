"""Compute cross-model CKA with bootstrap confidence intervals.

Adds 95% CIs to the 7x7 CKA similarity matrix for synthetic trend
representations, addressing Reviewer E's concern about point estimates.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_cka_bootstrap.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.metrics.probing_metrics import compute_cka_with_ci
from src.utils.seed import seed_everything

MODELS: dict[str, str] = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
    "Timer": "timer_meanpool",
    "TimesFM": "timesfm_meanpool",
    "Moirai": "moirai_meanpool",
}

DATASET = "synthetic_trend"
MAX_SAMPLES = 1000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CKA with bootstrap CIs.")
    parser.add_argument("--repr_root", type=str, default="outputs/representations")
    parser.add_argument("--output_dir", type=str, default="outputs/cka_bootstrap")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_best_layer(repr_dir: Path, max_samples: int) -> torch.Tensor | None:
    """Load the last layer representation (typically best for CKA)."""
    layer_files = sorted(
        [f for f in repr_dir.glob("*.pt") if f.stem != "labels"],
    )
    if not layer_files:
        return None
    # Use last layer (deepest representation)
    t = torch.load(layer_files[-1], map_location="cpu", weights_only=True)
    if t.ndim >= 3:
        t = t.view(t.shape[0], -1)
    if t.shape[0] > max_samples:
        t = t[:max_samples]
    return t.float()


def main() -> None:
    """Compute 7x7 CKA with bootstrap CIs."""
    args = parse_args()
    seed_everything(args.seed)
    repr_root = Path(args.repr_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(MODELS.keys())
    representations: dict[str, torch.Tensor] = {}

    for name, key in MODELS.items():
        rdir = repr_root / key / DATASET
        t = load_best_layer(rdir, MAX_SAMPLES)
        if t is not None:
            representations[name] = t
            print(f"Loaded {name}: {t.shape}")
        else:
            print(f"SKIP {name}: not found")

    results: dict[str, dict[str, dict[str, float]]] = {}

    for i, m1 in enumerate(model_names):
        results[m1] = {}
        for j, m2 in enumerate(model_names):
            if m1 not in representations or m2 not in representations:
                results[m1][m2] = {"cka": 0.0, "ci_low": 0.0, "ci_high": 0.0}
                continue
            X = representations[m1]
            Y = representations[m2]
            # Align sample counts
            n_samples = min(X.shape[0], Y.shape[0])
            cka, ci_low, ci_high = compute_cka_with_ci(
                X[:n_samples], Y[:n_samples],
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            results[m1][m2] = {
                "cka": round(cka, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
            }
            if i <= j:
                print(
                    f"  {m1} vs {m2}: CKA={cka:.4f} "
                    f"[{ci_low:.4f}, {ci_high:.4f}]"
                )

    # Save
    out_path = output_dir / "cka_bootstrap_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")

    # Print table
    print("\n" + "=" * 80)
    print("CKA with 95% Bootstrap CI")
    print("=" * 80)
    header = f"{'':>10}"
    for m in model_names:
        header += f" {m:>12}"
    print(header)
    for m1 in model_names:
        row = f"{m1:>10}"
        for m2 in model_names:
            r = results.get(m1, {}).get(m2, {})
            cka = r.get("cka", 0.0)
            row += f" {cka:>12.3f}"
        print(row)


if __name__ == "__main__":
    main()
