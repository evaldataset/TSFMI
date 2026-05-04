from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.device import resolve_device
from src.utils.logging import ExperimentLogger
from src.utils.seed import seed_everything


def test_resolve_device_cuda_string() -> None:
    assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_cpu_string() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_none_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None) == torch.device("cpu")


def test_resolve_device_cuda_index() -> None:
    assert resolve_device("cuda:0") == torch.device("cuda:0")


def test_seed_everything_reproducibility() -> None:
    seed_everything(42)
    torch_values_1 = torch.randn(5)
    numpy_values_1 = np.random.rand(5)
    random_values_1 = [random.random() for _ in range(5)]

    seed_everything(42)
    torch_values_2 = torch.randn(5)
    numpy_values_2 = np.random.rand(5)
    random_values_2 = [random.random() for _ in range(5)]

    assert torch.equal(torch_values_1, torch_values_2)
    assert np.array_equal(numpy_values_1, numpy_values_2)
    assert random_values_1 == random_values_2


def test_seed_everything_sets_pythonhashseed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    seed_everything(123)
    assert "PYTHONHASHSEED" in os.environ
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_experiment_logger_creates_output_directory(tmp_path: Path) -> None:
    logger = ExperimentLogger("exp", tmp_path)
    assert logger.output_dir.exists()
    assert logger.output_dir.is_dir()


def test_experiment_logger_log_config_writes_valid_json(tmp_path: Path) -> None:
    logger = ExperimentLogger("exp", tmp_path)
    config: dict[str, object] = {"model": "linear", "epochs": 10, "nested": {"lr": 1e-3}}

    logger.log_config(config)

    config_path = logger.output_dir / "config.json"
    assert config_path.exists()
    loaded = json.loads(config_path.read_text())
    assert loaded == config


def test_experiment_logger_log_metrics_appends_jsonl_records(tmp_path: Path) -> None:
    logger = ExperimentLogger("exp", tmp_path)

    logger.log_metrics({"accuracy": 0.9}, step=1)
    logger.log_metrics({"accuracy": 0.95}, step=2)

    metrics_path = logger.output_dir / "metrics.jsonl"
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert "timestamp" in first
    assert "timestamp" in second
    assert first["step"] == 1
    assert second["step"] == 2
    assert first["metrics"]["accuracy"] == pytest.approx(0.9)
    assert second["metrics"]["accuracy"] == pytest.approx(0.95)


def test_experiment_logger_log_message_writes_timestamped_log(tmp_path: Path) -> None:
    logger = ExperimentLogger("exp", tmp_path, verbose=False)
    logger.log_message("hello world")

    log_path = logger.output_dir / "run.log"
    line = log_path.read_text().strip()
    assert line.startswith("[")
    assert "] hello world" in line


def test_experiment_logger_save_probe_writes_pt_file(tmp_path: Path) -> None:
    logger = ExperimentLogger("exp", tmp_path)
    probe_state: dict[str, object] = {"weight": torch.tensor([1.0, 2.0])}

    saved_path = logger.save_probe(probe_state, "probe.pt")

    assert saved_path.suffix == ".pt"
    assert saved_path.exists()
    loaded = torch.load(saved_path)
    assert isinstance(loaded, dict)
    assert isinstance(loaded["weight"], torch.Tensor)
    assert isinstance(probe_state["weight"], torch.Tensor)
    assert torch.equal(loaded["weight"], probe_state["weight"])


def test_experiment_logger_verbose_false_suppresses_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = ExperimentLogger("exp", tmp_path, verbose=False)
    logger.log_message("do not print")

    captured = capsys.readouterr()
    assert captured.out == ""
