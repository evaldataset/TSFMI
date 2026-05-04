"""Structured experiment logger for probing experiments."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


class ExperimentLogger:
    """Logger that writes config, metrics, and messages to structured files.

    Creates an experiment directory under output_dir and writes:
    - config.json: experiment configuration
    - metrics.jsonl: one JSON object per line with metrics + timestamp
    - run.log: human-readable log messages

    Args:
        experiment_name: Name of the experiment (used as subdirectory).
        output_dir: Root output directory.
        verbose: Whether to also print log messages to stdout.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Path | str,
        *,
        verbose: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_config(self, config: dict[str, object]) -> None:
        """Write experiment configuration as JSON.

        Args:
            config: Configuration dictionary to serialize.
        """
        config_path = self.output_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")

    def log_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int | None = None,
    ) -> None:
        """Append metrics as a JSONL record with timestamp.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional training step or epoch number.
        """
        record: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        if step is not None:
            record["step"] = step

        metrics_path = self.output_dir / "metrics.jsonl"
        with metrics_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_message(self, message: str) -> None:
        """Write a log message to run.log and optionally stdout.

        Args:
            message: Message string to log.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}\n"

        log_path = self.output_dir / "run.log"
        with log_path.open("a") as f:
            f.write(line)

        if self.verbose:
            sys.stdout.write(line)
            sys.stdout.flush()

    def save_probe(self, probe_state: dict[str, object], filename: str) -> Path:
        """Save a probe checkpoint via torch.save.

        Args:
            probe_state: State dict or any serializable object.
            filename: Filename for the checkpoint (e.g. "probe_layer3.pt").

        Returns:
            Path to the saved checkpoint file.
        """
        save_path = self.output_dir / filename
        torch.save(probe_state, save_path)
        return save_path
