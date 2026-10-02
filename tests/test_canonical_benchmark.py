"""Tests for scripts/run_canonical_benchmark.py, in particular --pca-in-pipeline.

The option exists to remove a train/test leak (F16): PCA must be fit on the training rows of
each split seed only, never on validation or test rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts import run_canonical_benchmark as rcb

N_SAMPLES = 60
N_PCA = 5


class _RecordingPCA(PCA):
    """PCA that records every matrix it is fit on."""

    fitted_inputs: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _RecordingPCA:
        _RecordingPCA.fitted_inputs.append(np.array(X, copy=True))
        return super().fit(X, y)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        _RecordingPCA.fitted_inputs.append(np.array(X, copy=True))
        return super().fit_transform(X, y)


@pytest.fixture()
def tiny_repr_dir(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray], np.ndarray]:
    """Two random layers shaped (N, patches, dim), flattening to 24 features, plus labels."""
    gen = torch.Generator().manual_seed(0)
    layers = {
        "layer_0": torch.randn(N_SAMPLES, 3, 8, generator=gen),
        "layer_1": torch.randn(N_SAMPLES, 3, 8, generator=gen),
    }
    labels = torch.arange(N_SAMPLES) % 2
    for name, t in layers.items():
        torch.save(t, tmp_path / f"{name}.pt")
    torch.save(labels, tmp_path / "labels.pt")
    flat = {k: v.reshape(N_SAMPLES, -1).numpy().astype(np.float64) for k, v in layers.items()}
    return tmp_path, flat, labels.numpy()


def test_build_pipeline_default_has_no_pca() -> None:
    pipe = rcb.build_pipeline("classification", seed=0, n_features=100)
    assert [type(s).__name__ for _, s in pipe.steps] == ["StandardScaler", "LogisticRegression"]


def test_build_pipeline_inserts_pca_after_scaler() -> None:
    pipe = rcb.build_pipeline("regression", seed=3, n_features=100, pca_components=10)
    assert [type(s).__name__ for _, s in pipe.steps] == ["StandardScaler", "PCA", "Ridge"]
    assert pipe.steps[1][1].n_components == 10
    # A layer that is already small enough is not reduced.
    small = rcb.build_pipeline("regression", seed=3, n_features=10, pca_components=10)
    assert [type(s).__name__ for _, s in small.steps] == ["StandardScaler", "Ridge"]


def test_pca_is_fit_on_training_rows_only(
    tiny_repr_dir: tuple[Path, dict[str, np.ndarray], np.ndarray],
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repr_dir, flat, _ = tiny_repr_dir
    out_dir = tmp_path_factory.mktemp("out")
    _RecordingPCA.fitted_inputs = []
    monkeypatch.setattr(rcb, "PCA", _RecordingPCA)

    rcb.main(
        [
            "--representations_dir",
            str(repr_dir),
            "--output_dir",
            str(out_dir),
            "--property",
            "toy",
            "--task_type",
            "classification",
            "--pca-in-pipeline",
            str(N_PCA),
        ]
    )

    fitted = _RecordingPCA.fitted_inputs
    assert fitted, "PCA was never fit"
    n_train = N_SAMPLES - 2 * int(N_SAMPLES * rcb.TEST_RATIO)
    assert all(f.shape == (n_train, 24) for f in fitted)

    # Every PCA fit must equal the train-split-standardised rows of some (seed, layer), and
    # every (seed, layer) must have been fit, i.e. PCA never saw validation or test rows.
    expected = []
    for seed in rcb.SEEDS:
        n = N_SAMPLES
        idx = np.random.default_rng(seed).permutation(n)
        train_idx = idx[int(n * rcb.TEST_RATIO) + int(n * rcb.VAL_RATIO) :]
        for X in flat.values():
            expected.append(StandardScaler().fit_transform(X[train_idx]))
    for f in fitted:
        assert any(np.allclose(f, e) for e in expected)
    for e in expected:
        assert any(np.allclose(f, e) for f in fitted)

    result = json.loads((out_dir / "canonical_results.json").read_text())
    assert result["protocol"]["pca_in_pipeline"]["n_components"] == N_PCA
    assert len(result["test_score_per_seed"]) == len(rcb.SEEDS)


def test_default_path_does_not_use_pca(
    tiny_repr_dir: tuple[Path, dict[str, np.ndarray], np.ndarray],
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repr_dir, _, _ = tiny_repr_dir
    out_dir = tmp_path_factory.mktemp("out_default")
    _RecordingPCA.fitted_inputs = []
    monkeypatch.setattr(rcb, "PCA", _RecordingPCA)
    rcb.main(
        [
            "--representations_dir",
            str(repr_dir),
            "--output_dir",
            str(out_dir),
            "--property",
            "toy",
            "--task_type",
            "regression",
        ]
    )
    assert _RecordingPCA.fitted_inputs == []
    result = json.loads((out_dir / "canonical_results.json").read_text())
    assert "pca_in_pipeline" not in result["protocol"]
