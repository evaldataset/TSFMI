"""Tests for the resolution-study primitives: paired tests, surrogates, re-initialisation, generator
scaling, and the 1-NN store helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.datasets.surrogates import iaaft, spectral_error
from src.metrics.paired_tests import (
    label_entropy_bits,
    mcnemar,
    mcnemar_sample_size,
    paired_logloss_difference,
    tost,
    vinfo_bits,
)


# --------------------------------------------------------------------------- paired tests
def test_mcnemar_counts_and_symmetry() -> None:
    a = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    b = np.array([1, 0, 0, 0, 1, 1], dtype=bool)
    r = mcnemar(a, b)
    assert (r["n_a_only"], r["n_b_only"]) == (2, 1)
    assert r["p_value"] == pytest.approx(1.0)
    assert mcnemar(b, a)["p_value"] == pytest.approx(r["p_value"])
    big_a = np.r_[np.ones(60, bool), np.zeros(10, bool)]
    big_b = np.r_[np.zeros(60, bool), np.ones(10, bool)]
    assert mcnemar(big_a, big_b)["p_value"] < 1e-8


def test_tost_declares_equivalence_only_with_power() -> None:
    rng = np.random.default_rng(0)
    u = rng.random(20_000)
    both = rng.random(20_000) < 0.8
    a = np.where(u < 0.05, True, np.where(u < 0.10, False, both))  # 5% a-only, 5% b-only
    b = np.where(u < 0.05, False, np.where(u < 0.10, True, both))
    assert tost(a, b, margin=0.02)["equivalent"]
    assert not tost(a[:200], b[:200], margin=0.02)["equivalent"]  # too few windows
    worse = a & (rng.random(20_000) > 0.1)  # ~8 points lower
    assert not tost(a, worse, margin=0.02)["equivalent"]


def test_sample_size_matches_formula() -> None:
    # (1.96 + 0.8416)^2 * 0.1 / 0.01^2 = 7849
    assert mcnemar_sample_size(0.01, 0.1) == 7849
    assert mcnemar_sample_size(0.02, 0.1) == 1963


def test_vinfo_and_conditional_loss() -> None:
    y = np.tile([0, 1], 500)
    assert label_entropy_bits(y) == pytest.approx(1.0)
    perfect = np.where(y == 1, 1 - 1e-9, 1e-9)
    constant = np.full(len(y), 0.5)
    assert vinfo_bits(perfect, y, y) == pytest.approx(1.0, abs=1e-6)
    assert vinfo_bits(constant, y, y) == pytest.approx(0.0, abs=1e-9)
    d = paired_logloss_difference(perfect, constant, y)
    assert d["bits"] == pytest.approx(1.0, abs=1e-6) and d["lo"] > 0.99


# --------------------------------------------------------------------------- surrogates
def test_iaaft_keeps_marginal_and_spectrum_and_moves_events() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(256)
    x = np.sin(2 * np.pi * t / 32)[None, :] + 0.3 * rng.normal(size=(20, 256))
    x[:, 100] += 6.0  # a spike at a fixed position
    s = iaaft(x, n_iter=100, seed=0)
    assert np.array_equal(np.sort(s, axis=1), np.sort(x, axis=1))
    assert np.median(spectral_error(x, s)) < 0.05
    assert np.mean(np.argmax(s, axis=1) == 100) < 0.5  # the event is no longer pinned in time


# --------------------------------------------------------------------------- re-initialisation
class _ToyWrapper:
    """Minimal stand-in exposing the attributes reinitialize_ reads."""

    def __init__(self) -> None:
        self.model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
        self._patch_embedding = nn.Linear(2, 4)
        self.scale = None


class _BareParam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.ones(3, 3) * 5)
        self.norm_weight = nn.Parameter(torch.full((3,), 2.0))


def test_reinitialize_changes_every_matrix_and_is_seeded() -> None:
    from src.models.random_init import reinitialize_

    torch.manual_seed(0)
    w = _ToyWrapper()
    w.model.append(_BareParam())
    with torch.no_grad():  # make every weight "pretrained": none at its default value
        for p in w.model.parameters():
            p.add_(torch.randn_like(p))
    before = [p.detach().clone() for p in w.model.parameters()]
    counts = reinitialize_(w, seed=3)
    after = list(w.model.parameters())
    assert all(not torch.equal(a, b) for a, b in zip(after, before, strict=True) if a.ndim >= 2)
    assert torch.all(w.model[3].norm_weight == 1.0)
    assert counts["n_reset_by_fallback"] == 2
    w2 = _ToyWrapper()
    w2.model.append(_BareParam())
    reinitialize_(w2, seed=3)
    assert torch.equal(w2.model[0].weight, w.model[0].weight)


# --------------------------------------------------------------------------- generator scaling
def test_realistic_generator_scale_one_is_unchanged_and_scale_zero_removes_offsets() -> None:
    from scripts.run_realistic_anomaly import realistic_anomaly_dataset

    x, y, k = realistic_anomaly_dataset(400, 128, 7, return_kinds=True)
    x1, y1 = realistic_anomaly_dataset(400, 128, 7, magnitude_scale=1.0)
    assert np.array_equal(x, x1) and np.array_equal(y, y1)
    x0, _ = realistic_anomaly_dataset(400, 128, 7, magnitude_scale=0.0)
    assert np.array_equal(x0[y == 0], x[y == 0])  # normal windows never change
    for kind in (0, 1, 2, 3):  # scale 0 changes every anomalous kind relative to scale 1
        assert not np.allclose(x0[k == kind], x[k == kind])


# --------------------------------------------------------------------------- 1-NN store
def test_nn_helpers() -> None:
    from scripts.run_embedding_store import cosine, euclidean, loo_accuracy, nn_accuracy, znorm

    a = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    assert euclidean(a, a)[0, 1] == pytest.approx(10.0)
    assert cosine(np.array([[1.0, 0.0]]), np.array([[2.0, 0.0], [0.0, 3.0]]))[0].tolist() == (
        pytest.approx([0.0, 1.0])
    )
    y = np.array([0, 1, 0])
    assert loo_accuracy(euclidean(a, a), y) == pytest.approx(2 / 3)
    assert nn_accuracy(euclidean(a[:1], a[1:]), y[1:], y[:1]) == 1.0  # nearest is (0, 1)
    z = znorm(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert z[0].std() == pytest.approx(1.0) and np.all(z[1] == 0)
