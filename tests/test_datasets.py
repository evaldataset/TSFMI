"""Tests for synthetic time series generators."""

import numpy as np
import pytest

from src.datasets.synthetic import (
    generate_anomaly_dataset,
    generate_anomaly_hard_dataset,
    generate_change_point_dataset,
    generate_change_point_hard_dataset,
    generate_frequency_dataset,
    generate_frequency_hard_dataset,
    generate_seasonality_dataset,
    generate_stationarity_dataset,
    generate_stationarity_hard_dataset,
    generate_trend_dataset,
    generate_trend_hard_dataset,
)


def test_trend_shape():
    ds = generate_trend_dataset(num_samples=300, seq_len=64)
    assert ds.sequences.shape == (300, 64)
    assert ds.labels.shape == (300,)
    assert ds.label_type == "classification"
    assert ds.property_name == "trend"


def test_trend_labels_balanced():
    ds = generate_trend_dataset(num_samples=300, seq_len=64)
    unique, counts = np.unique(ds.labels, return_counts=True)
    assert set(unique.tolist()) == {0, 1, 2}
    # 300 / 3 = 100 per class
    assert all(c >= 90 for c in counts)


def test_trend_metadata():
    ds = generate_trend_dataset(num_samples=300, seq_len=64, noise_std=0.5, seed=7)
    assert ds.metadata["num_samples"] == 300
    assert ds.metadata["seq_len"] == 64
    assert ds.metadata["noise_std"] == 0.5
    assert ds.metadata["seed"] == 7


def test_seasonality_shape():
    ds = generate_seasonality_dataset(num_samples=100, seq_len=64)
    assert ds.sequences.shape == (100, 64)
    assert ds.labels.shape == (100,)
    assert ds.label_type == "regression"
    assert ds.property_name == "seasonality"


def test_seasonality_labels_valid():
    ds = generate_seasonality_dataset(num_samples=100, seq_len=64, periods=[8, 16, 32])
    assert all(label in {8.0, 16.0, 32.0} for label in ds.labels.tolist())


def test_seasonality_default_periods():
    ds = generate_seasonality_dataset(num_samples=100, seq_len=64)
    assert all(label in {8.0, 16.0, 32.0, 64.0} for label in ds.labels.tolist())


def test_frequency_shape():
    ds = generate_frequency_dataset(num_samples=100, seq_len=64, freq_bins=4)
    assert ds.sequences.shape == (100, 64)
    assert ds.labels.shape == (100,)
    assert ds.label_type == "classification"
    assert ds.property_name == "frequency"


def test_frequency_labels_valid():
    ds = generate_frequency_dataset(num_samples=100, seq_len=64, freq_bins=4)
    assert set(ds.labels.tolist()).issubset({0, 1, 2, 3})


def test_stationarity_binary():
    ds = generate_stationarity_dataset(num_samples=100, seq_len=64)
    assert ds.sequences.shape == (100, 64)
    assert ds.label_type == "classification"
    assert ds.property_name == "stationarity"
    assert set(ds.labels.tolist()).issubset({0, 1})


def test_stationarity_balanced():
    ds = generate_stationarity_dataset(num_samples=100, seq_len=64)
    unique, counts = np.unique(ds.labels, return_counts=True)
    assert set(unique.tolist()) == {0, 1}
    assert all(c >= 45 for c in counts)  # ~50 each


def test_anomaly_binary():
    ds = generate_anomaly_dataset(num_samples=100, seq_len=64)
    assert ds.sequences.shape == (100, 64)
    assert ds.label_type == "classification"
    assert ds.property_name == "anomaly"
    assert set(ds.labels.tolist()).issubset({0, 1})


def test_anomaly_fraction():
    ds = generate_anomaly_dataset(num_samples=100, seq_len=64, anomaly_fraction=0.3)
    num_anomaly = (ds.labels == 1).sum()
    assert num_anomaly == 30


def test_change_point_binary():
    ds = generate_change_point_dataset(num_samples=100, seq_len=64)
    assert ds.sequences.shape == (100, 64)
    assert ds.label_type == "classification"
    assert ds.property_name == "change_point"
    assert set(ds.labels.tolist()).issubset({0, 1})


def test_change_point_metadata():
    ds = generate_change_point_dataset(num_samples=100, seq_len=64)
    assert ds.metadata["change_point_position"] == 32  # seq_len // 2


def test_reproducibility():
    ds1 = generate_trend_dataset(num_samples=50, seq_len=32, seed=99)
    ds2 = generate_trend_dataset(num_samples=50, seq_len=32, seed=99)
    np.testing.assert_array_equal(ds1.sequences, ds2.sequences)
    np.testing.assert_array_equal(ds1.labels, ds2.labels)


def test_different_seeds_differ():
    ds1 = generate_trend_dataset(num_samples=50, seq_len=32, seed=1)
    ds2 = generate_trend_dataset(num_samples=50, seq_len=32, seed=2)
    assert not np.array_equal(ds1.sequences, ds2.sequences)


def test_reproducibility_seasonality():
    ds1 = generate_seasonality_dataset(num_samples=50, seq_len=32, seed=99)
    ds2 = generate_seasonality_dataset(num_samples=50, seq_len=32, seed=99)
    np.testing.assert_array_equal(ds1.sequences, ds2.sequences)
    np.testing.assert_array_equal(ds1.labels, ds2.labels)


@pytest.mark.parametrize(
    ("generator", "kwargs", "label_type", "property_name"),
    [
        (
            generate_trend_hard_dataset,
            {"num_samples": 90, "seq_len": 64, "seed": 7},
            "classification",
            "trend",
        ),
        (
            generate_frequency_hard_dataset,
            {"num_samples": 100, "seq_len": 64, "freq_bins": 5, "seed": 7},
            "classification",
            "frequency",
        ),
        (
            generate_anomaly_hard_dataset,
            {"num_samples": 100, "seq_len": 64, "anomaly_fraction": 0.3, "seed": 7},
            "classification",
            "anomaly",
        ),
        (
            generate_stationarity_hard_dataset,
            {"num_samples": 100, "seq_len": 64, "seed": 7},
            "classification",
            "stationarity",
        ),
        (
            generate_change_point_hard_dataset,
            {"num_samples": 100, "seq_len": 64, "seed": 7},
            "classification",
            "change_point",
        ),
    ],
)
def test_hard_generators_shape(
    generator,
    kwargs,
    label_type,
    property_name,
):
    """Verify hard generators return expected shapes and metadata tags.

    Args:
        generator: Hard dataset generator function under test.
        kwargs: Keyword arguments for generator invocation.
        label_type: Expected dataset label type string.
        property_name: Expected temporal property name.
    """
    ds = generator(**kwargs)
    assert ds.sequences.shape == (kwargs["num_samples"], kwargs["seq_len"])
    assert ds.labels.shape == (kwargs["num_samples"],)
    assert ds.label_type == label_type
    assert ds.property_name == property_name


def test_trend_hard_labels_valid() -> None:
    """Verify hard trend labels are valid 3-way classes."""
    ds = generate_trend_hard_dataset(num_samples=90, seq_len=64, seed=11)
    assert set(ds.labels.tolist()).issubset({0, 1, 2})


def test_frequency_hard_labels_valid() -> None:
    """Verify hard frequency labels stay within configured bin range."""
    ds = generate_frequency_hard_dataset(num_samples=120, seq_len=64, freq_bins=6, seed=11)
    assert set(ds.labels.tolist()).issubset({0, 1, 2, 3, 4, 5})


def test_anomaly_hard_labels_valid() -> None:
    """Verify hard anomaly labels are binary with exact anomaly fraction."""
    ds = generate_anomaly_hard_dataset(
        num_samples=100,
        seq_len=64,
        anomaly_fraction=0.25,
        seed=11,
    )
    assert set(ds.labels.tolist()).issubset({0, 1})
    assert int((ds.labels == 1).sum()) == 25


def test_stationarity_hard_labels_valid() -> None:
    """Verify hard stationarity labels are binary."""
    ds = generate_stationarity_hard_dataset(num_samples=100, seq_len=64, seed=11)
    assert set(ds.labels.tolist()).issubset({0, 1})


def test_change_point_hard_labels_valid() -> None:
    """Verify hard change-point labels are binary."""
    ds = generate_change_point_hard_dataset(num_samples=100, seq_len=64, seed=11)
    assert set(ds.labels.tolist()).issubset({0, 1})


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    [
        (generate_trend_hard_dataset, {"num_samples": 50, "seq_len": 32, "seed": 99}),
        (
            generate_frequency_hard_dataset,
            {"num_samples": 50, "seq_len": 32, "freq_bins": 4, "seed": 99},
        ),
        (
            generate_anomaly_hard_dataset,
            {"num_samples": 50, "seq_len": 64, "anomaly_fraction": 0.4, "seed": 99},
        ),
        (
            generate_stationarity_hard_dataset,
            {"num_samples": 50, "seq_len": 32, "seed": 99},
        ),
        (
            generate_change_point_hard_dataset,
            {"num_samples": 50, "seq_len": 32, "seed": 99},
        ),
    ],
)
def test_hard_generators_reproducibility(generator, kwargs):
    """Verify hard generators are deterministic under fixed seeds.

    Args:
        generator: Hard dataset generator function under test.
        kwargs: Keyword arguments including a fixed seed.
    """
    ds1 = generator(**kwargs)
    ds2 = generator(**kwargs)
    np.testing.assert_array_equal(ds1.sequences, ds2.sequences)
    np.testing.assert_array_equal(ds1.labels, ds2.labels)
