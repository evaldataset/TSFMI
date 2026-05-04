from __future__ import annotations

import torch

from src.probes.linear_probe import LinearProbe


def test_linear_probe_smoke_training_reduces_loss() -> None:
    torch.manual_seed(7)
    representations = torch.randn(100, 64)
    target_direction = torch.randn(64)
    labels = (representations @ target_direction > 0).long()

    probe = LinearProbe(input_dim=64, output_dim=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.2)

    with torch.no_grad():
        initial_logits = probe(representations)
        initial_loss = criterion(initial_logits, labels).item()

    for _ in range(200):
        optimizer.zero_grad()
        logits = probe(representations)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits = probe(representations)
        final_loss = criterion(final_logits, labels).item()

    assert final_logits.shape == (100, 2)
    assert final_loss < initial_loss
