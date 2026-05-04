"""Fast PCA dimension sweep for seasonality probing.

Uses sklearn's randomized SVD (default for large matrices) and properly
copies metadata to ensure regression probes are trained for seasonality.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_pca_sweep_fast.py --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from sklearn.decomposition import PCA

from src.probes.linear_probe import LinearProbe
from src.probes.probe_trainer import ProbeTrainer, ProbeTrainerConfig
from src.utils.seed import seed_everything

MODELS = {
    "moment": {"full_d_dir": "moment", "pca512_dir": "moment_pca512"},
    "gpt4ts": {"full_d_dir": "gpt4ts", "pca512_dir": "gpt4ts_pca512"},
}
PCA_DIMS = [64, 128, 256]
DATASET = "synthetic_seasonality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast PCA sweep for seasonality R²")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--repr_root", type=str, default="outputs/representations")
    parser.add_argument("--output_root", type=str, default="outputs/probes_pca_sweep")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _layer_sort_key(name: str) -> tuple[int, str]:
    parts = name.replace(".", "_").split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part), name
    return 0, name


def extract_pca(
    full_d_dir: Path,
    out_dir: Path,
    n_components: int,
) -> None:
    """Extract PCA representations using randomized SVD."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy labels
    labels_src = full_d_dir / "labels.pt"
    labels_dst = out_dir / "labels.pt"
    if not labels_dst.exists():
        shutil.copy2(labels_src, labels_dst)

    # Copy and update metadata
    meta_src = full_d_dir / "metadata.json"
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
        meta["pca_n_components"] = n_components
        meta["pca_source_dir"] = str(full_d_dir)
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # PCA each layer
    layer_files = sorted(
        [f for f in full_d_dir.glob("*.pt") if f.stem not in ("labels",)],
        key=lambda p: _layer_sort_key(p.stem),
    )

    for pt_file in layer_files:
        out_path = out_dir / pt_file.name
        if out_path.exists():
            continue

        repr_array = torch.load(pt_file, map_location="cpu", weights_only=True).numpy()
        orig_shape = repr_array.shape
        if repr_array.ndim > 2:
            repr_array = repr_array.reshape(repr_array.shape[0], -1)

        n_comp = min(n_components, repr_array.shape[0], repr_array.shape[1])
        # svd_solver='randomized' is much faster for high-dimensional data
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
        repr_pca = pca.fit_transform(repr_array)
        variance = pca.explained_variance_ratio_.sum()

        torch.save(torch.tensor(repr_pca, dtype=torch.float32), out_path)
        print(f"  {pt_file.stem}: {orig_shape} -> {repr_pca.shape}, var={variance:.4f}")


def train_regression_probe(
    repr_dir: Path,
    output_dir: Path,
    device: str,
) -> dict[str, float]:
    """Train regression probe on all layers, return best R²."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = torch.load(repr_dir / "labels.pt", map_location="cpu", weights_only=True).float()

    layer_files = sorted(
        [f for f in repr_dir.glob("*.pt") if f.stem not in ("labels",)],
        key=lambda p: _layer_sort_key(p.stem),
    )

    results: dict[str, float] = {}

    for layer_file in layer_files:
        reps = torch.load(layer_file, map_location="cpu", weights_only=True).float()
        if reps.ndim > 2:
            reps = reps.reshape(reps.shape[0], -1)

        probe = LinearProbe(input_dim=reps.shape[1], output_dim=1)
        trainer = ProbeTrainer(
            probe,
            ProbeTrainerConfig(
                learning_rate=0.001,
                num_epochs=100,
                batch_size=256,
                probe_type="regression",
                val_split=0.1,
                split_seed=42,
                device=device,
                verbose=False,
            ),
        )

        _, metrics = trainer.train(reps, labels)
        r2 = metrics.get("val_r2", 0.0)
        results[layer_file.stem] = float(r2)

    # Save summary
    summary = {
        "probe_type": "linear",
        "label_type": "regression",
        "layers": {k: {"val_r2": v} for k, v in results.items()},
        "best_r2": max(results.values()) if results else 0.0,
        "best_layer": max(results.items(), key=lambda item: item[1])[0] if results else "",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return results


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    repr_root = Path(args.repr_root)
    output_root = Path(args.output_root)

    all_results: dict[str, dict[int, float]] = {}

    for model_name, dirs in MODELS.items():
        full_d_dir = repr_root / dirs["full_d_dir"] / DATASET
        if not full_d_dir.exists():
            print(f"SKIP {model_name}: {full_d_dir} not found")
            continue

        model_results: dict[int, float] = {}

        for dim in PCA_DIMS:
            pca_dir_name = f"{dirs['full_d_dir']}_pca{dim}"
            pca_repr_dir = repr_root / pca_dir_name / DATASET
            probe_out = output_root / f"{model_name}_pca{dim}_seasonality_linear"

            if probe_out.exists() and (probe_out / "summary.json").exists():
                summary = json.loads((probe_out / "summary.json").read_text())
                best_r2 = summary.get("best_r2", 0.0)
                print(f"[{model_name} PCA{dim}] CACHED: best R²={best_r2:.4f}")
                model_results[dim] = best_r2
                continue

            # Extract PCA
            print(f"[{model_name} PCA{dim}] Extracting PCA representations...")
            extract_pca(full_d_dir, pca_repr_dir, dim)

            # Train probe
            print(f"[{model_name} PCA{dim}] Training regression probe...")
            layer_r2s = train_regression_probe(pca_repr_dir, probe_out, args.device)
            best_r2 = max(layer_r2s.values()) if layer_r2s else 0.0
            print(f"[{model_name} PCA{dim}] Best R²={best_r2:.4f}")
            model_results[dim] = best_r2

        all_results[model_name] = model_results

    # Also include existing PCA512 results
    for model_name, dirs in MODELS.items():
        pca512_eval_dir = Path("outputs/eval") / f"{dirs['pca512_dir']}_synthetic_seasonality"
        if pca512_eval_dir.exists():
            metrics_path = pca512_eval_dir / "layer_metrics.json"
            if metrics_path.exists():
                data = json.loads(metrics_path.read_text())
                best_r2 = max(float(row.get("val_r2", float("-inf"))) for row in data)
                if model_name not in all_results:
                    all_results[model_name] = {}
                all_results[model_name][512] = best_r2

    # Print summary
    print("\n=== PCA Sweep Summary ===")
    for model_name, dims in all_results.items():
        print(f"\n{model_name}:")
        for dim in sorted(dims.keys()):
            print(f"  PCA{dim}: R² = {dims[dim]:.4f}")

    # Save
    summary_path = output_root / "pca_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        model: {str(d): r2 for d, r2 in sorted(dims.items())} for model, dims in all_results.items()
    }
    summary_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
