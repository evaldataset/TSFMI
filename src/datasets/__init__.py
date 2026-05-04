"""Synthetic and real time series datasets for probing experiments."""

from src.datasets.synthetic import (
    SyntheticDataset,
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

__all__ = [
    "SyntheticDataset",
    "generate_anomaly_dataset",
    "generate_anomaly_hard_dataset",
    "generate_change_point_dataset",
    "generate_change_point_hard_dataset",
    "generate_frequency_dataset",
    "generate_frequency_hard_dataset",
    "generate_seasonality_dataset",
    "generate_stationarity_dataset",
    "generate_stationarity_hard_dataset",
    "generate_trend_dataset",
    "generate_trend_hard_dataset",
]
