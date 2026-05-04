"""Unified training loop for LinearProbe and MLPControlProbe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics.probing_metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


@dataclass
class ProbeTrainerConfig:
    """Configuration for ProbeTrainer.

    Attributes:
        learning_rate: Adam optimizer learning rate.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size for DataLoader.
        probe_type: "classification" or "regression" — determines loss function.
        val_split: Fraction of data held out for validation (0 = no validation).
        split_seed: Random seed used for deterministic train/validation shuffling.
        device: Training device.
        save_dir: Where to save checkpoints. None = no saving.
        verbose: Print per-epoch metrics.
    """

    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 256
    probe_type: str = "classification"  # "classification" | "regression"
    val_split: float = 0.2
    test_split: float = 0.0
    split_seed: int = 42
    device: str = "cpu"
    save_dir: str | None = None
    verbose: bool = True


class ProbeTrainer:
    """Unified training loop for LinearProbe and MLPControlProbe.

    Trains a probe on frozen representations with Adam optimizer.
    Logs per-epoch train loss and validation metrics to stdout (if verbose).

    Args:
        probe: Probe module to train (LinearProbe or MLPControlProbe).
        config: ProbeTrainerConfig with training hyperparameters.
    """

    def __init__(self, probe: nn.Module, config: ProbeTrainerConfig) -> None:
        self.probe = probe
        self.config = config
        self.device = torch.device(config.device)
        self.probe.to(self.device)

    def train(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[nn.Module, dict[str, float]]:
        """Train probe on frozen representations.

        Args:
            representations: Shape (N, hidden_dim) — frozen, not requiring grad.
            labels: Shape (N,) — int64 for classification, float for regression.

        Returns:
            Tuple of (trained probe, final metrics dict).
            Metrics keys:
                - train_loss (final epoch)
                - val_accuracy, val_f1_macro, val_f1_weighted (classification)
                - val_r2, val_mae, val_mse (regression)

        Raises:
            ValueError: If representations.ndim != 2.
        """
        if representations.ndim != 2:
            raise ValueError(
                f"Expected 2D representations (N, hidden_dim), got shape {representations.shape}"
            )

        representations = representations.to(self.device)
        labels = labels.to(self.device)

        # Split train/val/test
        train_repr, train_labels, val_repr, val_labels, test_repr, test_labels = (
            self._split_data(representations, labels)
        )

        # DataLoader for training (shuffle=True)
        train_dataset = TensorDataset(train_repr, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Loss and optimizer
        is_classification = self.config.probe_type == "classification"
        criterion: nn.Module = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=self.config.learning_rate)

        # Training loop
        final_metrics: dict[str, float] = {}
        num_epochs = self.config.num_epochs

        for epoch in range(num_epochs):
            self.probe.train()
            total_loss = 0.0

            for batch_repr, batch_labels in train_loader:
                optimizer.zero_grad()
                output = self.probe(batch_repr)
                if is_classification:
                    loss = criterion(output, batch_labels)
                else:
                    loss = criterion(output.squeeze(-1), batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluate on validation set periodically
            if (
                val_repr is not None
                and val_labels is not None
                and (epoch % 10 == 0 or epoch == num_epochs - 1)
            ):
                metrics = self._evaluate(val_repr, val_labels, is_classification)
                metrics["train_loss"] = avg_loss
                final_metrics = metrics

                if self.config.verbose:
                    self._log_epoch(epoch, num_epochs, metrics, is_classification)
            else:
                final_metrics["train_loss"] = avg_loss

        # Evaluate on held-out test set if available
        if test_repr is not None and test_labels is not None:
            test_metrics = self._evaluate(test_repr, test_labels, is_classification)
            for k, v in test_metrics.items():
                final_metrics[k.replace("val_", "test_")] = v

        # Save checkpoint
        if self.config.save_dir is not None:
            self._save_checkpoint()

        return self.probe, final_metrics

    def _split_data(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Split data into train, validation, and optionally test sets.

        Uses a deterministic shuffled split based on `split_seed`.

        Returns:
            Tuple of (train_repr, train_labels, val_repr, val_labels,
            test_repr, test_labels). val/test can be None if their split is 0.
        """
        n = representations.shape[0]
        test_size = int(n * self.config.test_split)
        val_size = int(n * self.config.val_split)

        if val_size == 0 and test_size == 0:
            return representations, labels, None, None, None, None

        generator = torch.Generator(device=representations.device)
        generator.manual_seed(self.config.split_seed)
        indices = torch.randperm(n, generator=generator, device=representations.device)

        # Split: [train | val | test]
        train_end = n - val_size - test_size
        val_end = n - test_size

        train_idx = indices[:train_end]
        train_repr = representations[train_idx]
        train_labels = labels[train_idx]

        val_repr: torch.Tensor | None = None
        val_labels: torch.Tensor | None = None
        if val_size > 0:
            val_idx = indices[train_end:val_end]
            val_repr = representations[val_idx]
            val_labels = labels[val_idx]

        test_repr: torch.Tensor | None = None
        test_labels: torch.Tensor | None = None
        if test_size > 0:
            test_idx = indices[val_end:]
            test_repr = representations[test_idx]
            test_labels = labels[test_idx]

        return train_repr, train_labels, val_repr, val_labels, test_repr, test_labels

    @torch.no_grad()
    def _evaluate(
        self,
        val_repr: torch.Tensor,
        val_labels: torch.Tensor,
        is_classification: bool,
    ) -> dict[str, float]:
        """Evaluate probe on validation set.

        Args:
            val_repr: Validation representations of shape (N_val, hidden_dim).
            val_labels: Validation labels of shape (N_val,).
            is_classification: Whether this is a classification task.

        Returns:
            Dict of validation metrics with "val_" prefix.
        """
        self.probe.eval()
        val_output = self.probe(val_repr)

        if is_classification:
            raw = compute_classification_metrics(val_output, val_labels)
            return {f"val_{k}": v for k, v in raw.items()}
        else:
            raw = compute_regression_metrics(val_output, val_labels)
            return {f"val_{k}": v for k, v in raw.items()}

    def _log_epoch(
        self,
        epoch: int,
        num_epochs: int,
        metrics: dict[str, float],
        is_classification: bool,
    ) -> None:
        """Print epoch training progress."""
        loss_str = f"loss={metrics['train_loss']:.4f}"
        if is_classification:
            val_str = (
                f"val_acc={metrics.get('val_accuracy', 0.0):.4f} "
                f"val_f1={metrics.get('val_f1_macro', 0.0):.4f}"
            )
        else:
            val_str = (
                f"val_r2={metrics.get('val_r2', 0.0):.4f} val_mae={metrics.get('val_mae', 0.0):.4f}"
            )
        print(f"Epoch {epoch + 1}/{num_epochs}: {loss_str} | {val_str}")

    def _save_checkpoint(self) -> None:
        """Save probe state dict to save_dir."""
        assert self.config.save_dir is not None
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "probe.pt"
        torch.save(self.probe.state_dict(), path)
