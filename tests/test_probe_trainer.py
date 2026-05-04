"""Tests for ProbeTrainer data splitting behavior."""

import torch

from src.probes.linear_probe import LinearProbe
from src.probes.probe_trainer import ProbeTrainer, ProbeTrainerConfig


def test_split_data_is_deterministic_for_same_seed() -> None:
    representations = torch.arange(20, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([0] * 10 + [1] * 10, dtype=torch.int64)

    config = ProbeTrainerConfig(val_split=0.2, split_seed=7, verbose=False)
    trainer_a = ProbeTrainer(LinearProbe(input_dim=1, output_dim=2), config)
    trainer_b = ProbeTrainer(LinearProbe(input_dim=1, output_dim=2), config)

    split_a = trainer_a._split_data(representations, labels)
    split_b = trainer_b._split_data(representations, labels)

    assert split_a[2] is not None
    assert split_a[3] is not None
    assert split_b[2] is not None
    assert split_b[3] is not None

    assert torch.equal(split_a[0], split_b[0])
    assert torch.equal(split_a[1], split_b[1])
    assert torch.equal(split_a[2], split_b[2])
    assert torch.equal(split_a[3], split_b[3])


def test_split_data_shuffles_instead_of_taking_last_block() -> None:
    representations = torch.arange(30, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10, dtype=torch.int64)

    config = ProbeTrainerConfig(val_split=0.2, split_seed=3, verbose=False)
    trainer = ProbeTrainer(LinearProbe(input_dim=1, output_dim=3), config)

    _train_repr, train_labels, _val_repr, val_labels, _test_repr, _test_labels = (
        trainer._split_data(representations, labels)
    )

    assert torch.unique(train_labels).numel() > 1
    assert val_labels is not None
    assert torch.unique(val_labels).numel() > 1


def test_train_classification_reduces_loss() -> None:
    """Verify classification training lowers loss on a tiny separable dataset.

    Args:
        None.
    """
    torch.manual_seed(123)
    representations = torch.randn(80, 6)
    labels = (representations[:, 0] + 0.5 * representations[:, 1] > 0).long()

    probe = LinearProbe(input_dim=6, output_dim=2)
    logits_before = probe(representations)
    initial_loss = torch.nn.CrossEntropyLoss()(logits_before, labels).item()

    config = ProbeTrainerConfig(
        learning_rate=5e-2,
        num_epochs=10,
        batch_size=16,
        probe_type="classification",
        val_split=0.2,
        split_seed=5,
        device="cpu",
        verbose=False,
    )
    trainer = ProbeTrainer(probe, config)
    trained_probe, metrics = trainer.train(representations, labels)

    logits_after = trained_probe(representations)
    final_loss = torch.nn.CrossEntropyLoss()(logits_after, labels).item()

    assert "train_loss" in metrics
    assert final_loss < initial_loss


def test_train_regression_mode() -> None:
    """Verify regression mode reports validation R2 metrics.

    Args:
        None.
    """
    torch.manual_seed(321)
    representations = torch.randn(100, 4)
    target_weights = torch.tensor([1.5, -0.7, 0.25, 0.9])
    labels = representations @ target_weights

    probe = LinearProbe(input_dim=4, output_dim=1)
    config = ProbeTrainerConfig(
        learning_rate=1e-2,
        num_epochs=20,
        batch_size=20,
        probe_type="regression",
        val_split=0.2,
        split_seed=9,
        device="cpu",
        verbose=False,
    )
    trainer = ProbeTrainer(probe, config)
    _trained_probe, metrics = trainer.train(representations, labels)

    assert "train_loss" in metrics
    assert "val_r2" in metrics
    assert "val_mae" in metrics
    assert "val_mse" in metrics


def test_train_no_val_split() -> None:
    """Verify training with val_split=0 omits validation metrics.

    Args:
        None.
    """
    torch.manual_seed(456)
    representations = torch.randn(60, 5)
    labels = torch.randint(0, 2, (60,))

    config = ProbeTrainerConfig(
        learning_rate=1e-2,
        num_epochs=5,
        batch_size=15,
        probe_type="classification",
        val_split=0.0,
        split_seed=4,
        device="cpu",
        verbose=False,
    )
    trainer = ProbeTrainer(LinearProbe(input_dim=5, output_dim=2), config)
    _trained_probe, metrics = trainer.train(representations, labels)

    assert "train_loss" in metrics
    assert all(not key.startswith("val_") for key in metrics)
