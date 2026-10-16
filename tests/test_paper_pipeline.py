"""Tests for the code paths that produce the manuscript's numbers.

Covered: layer selection and tie-breaking in the canonical benchmark, the corrected t-interval
used for every split-seed interval, the TimesFM context handling that removes left padding, the
LEACE erasure controls, the control-task construction, the capacity sweep's layer order, and a
golden check that the canonical baselines reproduce the committed scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_canonical_benchmark as rcb

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- layer selection
def test_first_layer_wins_ties_and_layer_10_does_not_precede_layer_2(tmp_path: Path) -> None:
    """With identical layers, the first block (layer_0) must be selected for every seed."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(60, 6, generator=gen)
    labels = (x[:, 0] > 0).long()
    for i in (0, 1, 2, 10):
        torch.save(x.clone(), tmp_path / f"layer_{i}.pt")
    torch.save(labels, tmp_path / "labels.pt")
    out = tmp_path / "out"
    rcb.main(
        [
            "--representations_dir",
            str(tmp_path),
            "--output_dir",
            str(out),
            "--property",
            "toy",
            "--task_type",
            "classification",
        ]
    )
    res = json.loads((out / "canonical_results.json").read_text())
    assert set(res["best_layer_per_seed"]) == {"layer_0"}


def test_strictly_better_deeper_layer_is_selected(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(1)
    y = torch.arange(80) % 2
    noise = torch.randn(80, 4, generator=gen)
    torch.save(noise, tmp_path / "layer_0.pt")
    informative = noise.clone()
    informative[:, 0] = y.float() * 4 - 2
    torch.save(informative, tmp_path / "layer_1.pt")
    torch.save(y, tmp_path / "labels.pt")
    out = tmp_path / "out"
    rcb.main(
        [
            "--representations_dir",
            str(tmp_path),
            "--output_dir",
            str(out),
            "--property",
            "toy",
            "--task_type",
            "classification",
        ]
    )
    res = json.loads((out / "canonical_results.json").read_text())
    assert set(res["best_layer_per_seed"]) == {"layer_1"}
    assert res["test_mean"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- corrected interval
def _make_numbers():
    sys.path.insert(0, str(ROOT / "tools"))
    import make_numbers  # noqa: PLC0415

    return make_numbers


def test_corrected_t_matches_hand_computation() -> None:
    from scipy import stats

    mn = _make_numbers()
    x = [0.70, 0.72, 0.75, 0.73, 0.71]
    mean, lo, hi, p = mn.corrected_t(x)
    var = np.var(x, ddof=1) * (1 / 5 + 200 / 600)
    half = stats.t.ppf(0.975, 4) * var**0.5
    assert mean == pytest.approx(np.mean(x))
    assert (lo, hi) == pytest.approx((np.mean(x) - half, np.mean(x) + half))
    assert p < 1e-3  # far from zero
    # The correction is wider than the naive standard error of the mean.
    naive = stats.t.ppf(0.975, 4) * (np.var(x, ddof=1) / 5) ** 0.5
    assert hi - lo > 2 * naive


def test_corrected_t_zero_variance_and_numbers_rules() -> None:
    mn = _make_numbers()
    assert mn.corrected_t([1.0] * 5)[:3] == (1.0, 1.0, 1.0)
    n = mn.Numbers()
    n.put_ci("a", 0.0, 0.0, 0.0, "src", [0.70, 0.72, 0.75, 0.73, 0.71])
    assert float(n.values["a/lo"]) < 0.72 < float(n.values["a/hi"])
    with pytest.raises(ValueError):
        n.put("nan", float("nan"), "src")
    with pytest.raises(KeyError):
        n.put("a/mean", 0.5, "src")


# --------------------------------------------------------------------------- TimesFM padding
class _FakeTimesFM:
    """Stands in for TimesFmModelForPrediction: records the context length it is called with."""

    def __init__(self, fail: bool = False) -> None:
        self.config = SimpleNamespace(patch_length=32)
        self.context_len = 2048
        self.seen: list[int] = []
        self.fail = fail

    def __call__(self, past_values: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        self.seen.append(self.context_len)
        if self.fail:
            raise RuntimeError("boom")
        return past_values


def _wrapper_with(fake: _FakeTimesFM):
    from src.models.timesfm_wrapper import TimesFMWrapper

    w = TimesFMWrapper()
    w._prediction_model = fake
    w._model = torch.nn.Identity()
    w._device = torch.device("cpu")
    return w


@pytest.mark.parametrize(("length", "expected"), [(512, 512), (500, 512), (4096, 2048)])
def test_timesfm_runs_at_input_length_and_restores_context(length: int, expected: int) -> None:
    fake = _FakeTimesFM()
    _wrapper_with(fake).forward(torch.zeros(2, length))
    assert fake.seen == [expected]
    assert fake.context_len == 2048


def test_timesfm_restores_context_on_error() -> None:
    fake = _FakeTimesFM(fail=True)
    with pytest.raises(RuntimeError):
        _wrapper_with(fake).forward(torch.zeros(1, 512))
    assert fake.context_len == 2048


# --------------------------------------------------------------------------- erasure controls
def _toy_erasure_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], 300)
    stat = y * 2.0 + rng.normal(0, 0.6, len(y))  # a statistic that predicts the label
    X = np.column_stack([stat, rng.normal(size=(len(y), 60))])  # high-dimensional, as models are
    return X, y, stat[:, None]


def test_label_erasure_reaches_chance_and_random_erasure_does_not_hurt() -> None:
    from scripts.run_statistic_erasure import run_row

    X, y, S = _toy_erasure_data()
    r = run_row(X, y, S, seed=0, pca=None)
    assert r["none"]["acc"] > 0.85
    assert r["label"]["acc"] < 0.6
    assert r["random"]["acc"] > 0.8
    assert r["statistics"]["acc"] < 0.6
    # the label proxy is calibrated to the concept's own predictiveness
    assert abs(r["labelproxy"]["proxy_train_acc"] - r["labelproxy"]["concept_train_acc"]) < 0.03


# --------------------------------------------------------------------------- control task
def test_control_task_types_and_labels() -> None:
    from scripts.run_hl_selectivity import assign_types, control_labels

    rng = np.random.default_rng(0)
    seqs = rng.normal(size=(40, 128))
    assert np.array_equal(assign_types(seqs, 40), np.arange(40))
    types = assign_types(seqs, 5)
    assert types.shape == (40,) and len(np.unique(types)) <= 5
    y = np.tile([0.0, 1.0], 20)
    labels, per_type = control_labels(types, y)
    # every window of a type gets that type's label
    for t in np.unique(types):
        assert len(np.unique(labels[types == t])) == 1
    assert set(np.unique(labels)) <= {0.0, 1.0}


# --------------------------------------------------------------------------- capacity sweep
def test_capacity_sweep_orders_layers_by_depth(tmp_path: Path) -> None:
    from scripts.run_capacity_probe_v2 import layer_files

    for i in (0, 1, 2, 10, 11):
        (tmp_path / f"layers_{i}.pt").write_bytes(b"")
    (tmp_path / "labels.pt").write_bytes(b"")
    assert [f.stem for f in layer_files(tmp_path)] == [
        "layers_0",
        "layers_1",
        "layers_2",
        "layers_10",
        "layers_11",
    ]


# --------------------------------------------------------------------------- golden check
def test_canonical_hand_crafted_anomaly_matches_committed_scores() -> None:
    from scripts.run_canonical_baselines import hand_crafted_features
    from src.datasets.synthetic import generate_anomaly_dataset

    committed = next(
        r
        for r in json.loads((ROOT / "outputs/canonical_v2/baselines/all_results.json").read_text())
        if r["baseline"] == "hand_crafted" and r["property"] == "anomaly"
    )
    ds = generate_anomaly_dataset(1000, 512, seed=42)
    X = hand_crafted_features(ds.sequences)
    scores = []
    for seed in rcb.SEEDS:
        X_tr, y_tr, _, _, X_te, y_te = rcb.three_way_split(X, ds.labels, seed)
        scores.append(rcb.train_and_score(X_tr, y_tr, X_te, y_te, "classification", seed))
    assert scores == pytest.approx(committed["per_seed_test"], abs=1e-9)
