"""H10 — scale sweep: does model capacity build HIGH-DEGREE accessibility?

Completes the two-factor decomposition. H9 showed the objective does not shape the spectrum
and instance-norm only kills the low end; the high-degree half (the anomaly-relevant band) was
absent at 1M params. Hypothesis H10: high-degree accessibility EMERGES with scale, given a corpus
that actually contains high-degree structure.

Design: fix objective = masked reconstruction (must represent the full signal incl. tails);
fix a HEAVY-TAILED corpus (AR+seasonal + Student-t noise + occasional spikes); sweep model width;
measure A(H3-6) vs params. Random-init baseline at each size attributes any rise to training.

Usage:  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/run_scale_sweep.py
Output: outputs/scale_sweep/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.polynomial.hermite_e import hermeval
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from src.datasets.synthetic import generate_anomaly_dataset
from src.utils.seed import seed_everything

OUT = Path("outputs/scale_sweep")
L, PATCH = 256, 16
NPATCH = L // PATCH
STEPS, BS, LR, CORPUS_N = 2000, 256, 3e-4, 30000
SIZES = [(64, 2), (128, 3), (256, 4), (512, 4)]  # (d_model, layers)
DEG = (1, 2, 3, 4, 5, 6)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_corpus_heavy(n: int, length: int, seed: int) -> torch.Tensor:
    """AR(1)+seasonal + Student-t noise + occasional spikes -> genuine high-degree structure."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.3, 0.95, size=n)[:, None]
    freq = rng.uniform(0.01, 0.15, size=n)[:, None]
    amp = rng.uniform(0.0, 2.0, size=n)[:, None]
    t = np.arange(length)[None, :]
    seasonal = amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi, size=n)[:, None])
    noise = rng.standard_t(df=3, size=(n, length))  # heavy-tailed innovations
    ar = np.zeros((n, length))
    for i in range(1, length):
        ar[:, i] = phi[:, 0] * ar[:, i - 1] + noise[:, i]
    x = ar + seasonal
    spike_mask = rng.random((n, length)) < 0.01  # rare large spikes
    x = x + spike_mask * rng.normal(0, 6.0, size=(n, length))
    x = (x - x.mean()) / (x.std() + 1e-8)
    return torch.tensor(x, dtype=torch.float32)


class Enc(nn.Module):
    def __init__(self, d: int, layers: int) -> None:
        super().__init__()
        self.d = d
        self.embed = nn.Linear(PATCH, d)
        self.pos = nn.Parameter(torch.randn(1, NPATCH, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        lyr = nn.TransformerEncoderLayer(d, 4, d * 4, batch_first=True, activation="gelu",
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(lyr, layers)

    def patchify(self, x):
        return x.unfold(1, PATCH, PATCH)

    def forward(self, x, mask=None):
        h = self.embed(self.patchify(x)) + self.pos
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_tok, h)
        return self.tr(h)


def train_reconstruct(d: int, layers: int, corpus: torch.Tensor) -> Enc:
    enc = Enc(d, layers).to(DEVICE)
    head = nn.Linear(d, PATCH).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR)
    g = torch.Generator().manual_seed(0)
    enc.train()
    for _ in range(STEPS):
        idx = torch.randint(0, corpus.shape[0], (BS,), generator=g)
        x = corpus[idx].to(DEVICE)
        patches = enc.patchify(x)
        m = torch.rand(BS, NPATCH, device=DEVICE) < 0.5
        pred = head(enc(x, mask=m))
        loss = F.mse_loss(pred[m], patches[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc.eval()


def hermite_moment(sig, k):
    c = np.zeros(k + 1)
    c[k] = 1.0
    return hermeval(sig, c).mean(axis=1)


def spectrum(enc: Enc, sig: np.ndarray) -> dict:
    x = torch.tensor(sig, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        H = np.concatenate([enc(x[s:s + 256]).mean(1).cpu().numpy()
                            for s in range(0, len(sig), 256)]).astype(np.float64)
    rng = np.random.default_rng(42)
    n = len(sig)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.6 * n)], idx[int(0.8 * n) :]
    Xp = PCA(n_components=min(64, H.shape[1]), random_state=42).fit(H[tr]).transform(H)
    out = {}
    for k in DEG:
        f = hermite_moment(sig, k)
        out[k] = round(float(r2_score(
            f[te], RidgeCV(alphas=[0.1, 1, 10, 100]).fit(Xp[tr], f[tr]).predict(Xp[te]))), 4)
    return out


def main() -> None:
    seed_everything(0)
    corpus = make_corpus_heavy(CORPUS_N, L, seed=1)
    probe = generate_anomaly_dataset(1000, L, seed=42).sequences
    rows = []
    for d, layers in SIZES:
        params = sum(p.numel() for p in Enc(d, layers).parameters())
        rnd = spectrum(Enc(d, layers).to(DEVICE).eval(), probe)
        trn = spectrum(train_reconstruct(d, layers, corpus), probe)
        hi = lambda s: round((s[3] + s[4] + s[5] + s[6]) / 4, 3)  # noqa: E731
        lo = lambda s: round((s[1] + s[2]) / 2, 3)  # noqa: E731
        rows.append({"d_model": d, "layers": layers, "params": params,
                     "random_high": hi(rnd), "trained_high": hi(trn),
                     "trained_low": lo(trn), "trained_A_by_degree": trn})
        print(f"d={d:<4} L={layers} ({params/1e6:.2f}M)  A(H3-6): random={hi(rnd):+.2f} "
              f"trained={hi(trn):+.2f} | trained low={lo(trn):+.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"corpus": "heavy-tailed AR+seasonal+spikes", "objective": "masked reconstruction",
         "sweep": rows,
         "hypothesis": "H10: trained_high A(H3-6) increases with params."}, indent=2))
    print(f"\nsaved -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
