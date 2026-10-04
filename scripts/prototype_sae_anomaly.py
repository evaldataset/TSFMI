"""D2 prototype: does a kurtosis feature emerge in a TSFM's representation?

Trains a minimal TopK Sparse Autoencoder on a frozen representation matrix and asks
whether any learned feature aligns with the per-sample excess kurtosis of the input
signal. This operationalizes the "linearly inaccessible vs. not encoded" question:

  - a high-|corr| kurtosis feature  => "encoded but entangled" (present, non-linear)
  - no such feature                 => evidence for "not encoded"

Usage:
    # Self-test the pipeline on synthetic data (no models / no disk needed):
    PYTHONPATH=. python scripts/prototype_sae_anomaly.py --self-test

    # Real run (once representations exist), e.g.:
    PYTHONPATH=. python scripts/prototype_sae_anomaly.py \
        --reps outputs/representations/moment_pca512/synthetic_anomaly/layer_17.pt \
        --signals outputs/representations/moment_pca512/synthetic_anomaly/signals.pt

The self-test builds a representation whose signal carries a kurtosis component and
verifies the SAE recovers a feature correlated with kurtosis (sanity check for the
attribution logic before spending GPU/disk on real extractions).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


class TopKSAE(nn.Module):
    """Minimal TopK sparse autoencoder (encoder + tied-free decoder + pre-bias)."""

    def __init__(self, d_in: int, d_hidden: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.enc(x - self.pre_bias))
        topv, topi = z.topk(self.k, dim=-1)
        f = torch.zeros_like(z).scatter_(-1, topi, topv)
        return f

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.dec(f) + self.pre_bias, f


def train_sae(
    X: torch.Tensor, d_hidden: int, k: int, epochs: int = 200, lr: float = 1e-3
) -> tuple[TopKSAE, NDArray[np.float64]]:
    """Train a TopK SAE on X (N, D); return the model and per-sample features (N, H)."""
    d_in = X.shape[1]
    sae = TopKSAE(d_in, d_hidden, k)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for _ in range(epochs):
        opt.zero_grad()
        recon, _ = sae(Xn)
        loss = ((recon - Xn) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        feats = sae.encode(Xn).cpu().numpy()
    return sae, feats


def best_kurtosis_feature(
    feats: NDArray[np.float64], kurtosis: NDArray[np.float64]
) -> tuple[int, float]:
    """Return (feature_index, |Pearson r|) of the SAE feature most aligned with kurtosis."""
    k = (kurtosis - kurtosis.mean()) / (kurtosis.std() + 1e-9)
    best_idx, best_r = -1, 0.0
    for j in range(feats.shape[1]):
        fj = feats[:, j]
        if fj.std() < 1e-9:
            continue
        r = float(np.corrcoef(fj, k)[0, 1])
        if abs(r) > abs(best_r):
            best_idx, best_r = j, r
    return best_idx, abs(best_r)


def _self_test() -> None:
    """Synthetic check: a representation carrying a kurtosis component should yield an
    SAE feature correlated with kurtosis. Verifies the attribution pipeline end-to-end."""
    rng = np.random.default_rng(0)
    n = 2000
    kurtosis = rng.gamma(2.0, 1.0, size=n)  # per-sample kurtosis-like signal
    kz = (kurtosis - kurtosis.mean()) / kurtosis.std()
    # 16-dim representation: dim 0 encodes kurtosis (entangled with a rotation), rest noise
    base = rng.normal(size=(n, 16))
    base[:, 0] = kz + 0.1 * rng.normal(size=n)
    rot, _ = np.linalg.qr(rng.normal(size=(16, 16)))  # entangle across dims
    X = torch.tensor((base @ rot).astype(np.float32))
    _, feats = train_sae(X, d_hidden=64, k=4, epochs=300)
    idx, r = best_kurtosis_feature(feats, kurtosis)
    print(f"[self-test] best kurtosis-aligned SAE feature: idx={idx}, |r|={r:.3f}")
    assert r > 0.5, f"pipeline failed to recover kurtosis feature (|r|={r:.3f})"
    print("[self-test] PASS — SAE recovers an entangled kurtosis feature.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--reps", type=str, default=None, help="path to layer rep tensor (N,D)")
    ap.add_argument("--signals", type=str, default=None, help="path to raw signals (N,L)")
    ap.add_argument("--d-hidden", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    if args.self_test or args.reps is None:
        _self_test()
        return

    X = torch.load(args.reps).float()
    signals = torch.load(args.signals).numpy()
    std = signals.std(axis=1, keepdims=True) + 1e-10
    kurtosis = np.mean(((signals - signals.mean(1, keepdims=True)) / std) ** 4, axis=1) - 3.0
    _, feats = train_sae(X, d_hidden=args.d_hidden, k=args.k)
    idx, r = best_kurtosis_feature(feats, kurtosis)
    verdict = (
        "encoded-but-entangled" if r > 0.5 else "weak/absent (evidence for non-encoded)"
    )
    print(f"best kurtosis-aligned SAE feature idx={idx}, |r|={r:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
