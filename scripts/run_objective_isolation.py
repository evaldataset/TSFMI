"""H9 — objective-isolation experiment (the causal heart of the transfer-function program).

Hold architecture, corpus, size, and optimizer FIXED; vary ONLY the pretraining objective.
Then measure each model's linear accessibility spectrum A(k) (held-out ridge R^2 recovering
the empirical Hermite-k moment of the input). This isolates whether the pretraining OBJECTIVE
causally shapes which input functionals survive linearly in the representation.

Arms (all from an identical encoder init):
  - random    : untrained (architecture-only baseline)
  - forecast  : causal next-patch prediction (canonical TSFM objective)
  - reconstruct: masked-patch reconstruction (MOMENT-style)
  - contrastive: NT-Xent on two augmented views (TS2Vec-style)

To avoid confounding with input normalization (H7), we do NOT per-window normalize; the corpus
is globally standardized AR(1)+seasonal+noise (natural predictive structure, known statistics).

Usage:  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/run_objective_isolation.py
Output: outputs/objective_isolation/results.json
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

OUT = Path("outputs/objective_isolation")
L, PATCH, DMODEL, LAYERS, HEADS = 256, 16, 128, 3, 4
NPATCH = L // PATCH
STEPS, BS, LR = 3000, 256, 3e-4
CORPUS_N = 40000
DEG = (1, 2, 3, 4, 5, 6)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_corpus(n: int, length: int, seed: int) -> torch.Tensor:
    """AR(1)+seasonal+noise windows, globally standardized. Natural predictive structure."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.3, 0.95, size=n)[:, None]
    freq = rng.uniform(0.01, 0.15, size=n)[:, None]
    amp = rng.uniform(0.0, 2.0, size=n)[:, None]
    t = np.arange(length)[None, :]
    seasonal = amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi, size=n)[:, None])
    noise = rng.normal(size=(n, length))
    ar = np.zeros((n, length))
    for i in range(1, length):
        ar[:, i] = phi[:, 0] * ar[:, i - 1] + noise[:, i]
    x = ar + seasonal
    x = (x - x.mean()) / (x.std() + 1e-8)
    return torch.tensor(x, dtype=torch.float32)


class PatchEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(PATCH, DMODEL)
        self.pos = nn.Parameter(torch.randn(1, NPATCH, DMODEL) * 0.02)
        self.mask_tok = nn.Parameter(torch.randn(1, 1, DMODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(DMODEL, HEADS, DMODEL * 4, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(layer, LAYERS)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(1, PATCH, PATCH)  # (B, NPATCH, PATCH)

    def forward(self, x: torch.Tensor, mask=None, causal=False) -> torch.Tensor:
        h = self.embed(self.patchify(x)) + self.pos
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_tok, h)
        attn = None
        if causal:
            attn = torch.triu(torch.ones(NPATCH, NPATCH, device=x.device) * float("-inf"), 1)
        return self.tr(h, mask=attn)  # (B, NPATCH, DMODEL)


def train_arm(objective: str, corpus: torch.Tensor, init_state: dict) -> PatchEncoder:
    enc = PatchEncoder().to(DEVICE)
    enc.load_state_dict(init_state)  # identical init across arms
    if objective == "random":
        return enc.eval()
    fore_head = nn.Linear(DMODEL, PATCH).to(DEVICE)
    proj = nn.Sequential(nn.Linear(DMODEL, DMODEL), nn.GELU(), nn.Linear(DMODEL, 64)).to(DEVICE)
    params = list(enc.parameters()) + list(fore_head.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    n = corpus.shape[0]
    g = torch.Generator().manual_seed(0)
    enc.train()
    for _ in range(STEPS):
        idx = torch.randint(0, n, (BS,), generator=g)
        x = corpus[idx].to(DEVICE)
        patches = enc.patchify(x)  # (B, NPATCH, PATCH)
        if objective == "forecast":
            tok = enc(x, causal=True)
            pred = fore_head(tok[:, :-1])  # predict next patch
            loss = F.mse_loss(pred, patches[:, 1:])
        elif objective == "reconstruct":
            m = torch.rand(BS, NPATCH, device=DEVICE) < 0.5
            tok = enc(x, mask=m)
            pred = fore_head(tok)
            loss = F.mse_loss(pred[m], patches[m])
        elif objective == "contrastive":
            def aug(z):
                return z + 0.1 * torch.randn_like(z) * z.std()
            z1 = proj(enc(aug(x)).mean(1))
            z2 = proj(enc(aug(x)).mean(1))
            z = F.normalize(torch.cat([z1, z2]), dim=1)
            sim = z @ z.T / 0.1
            sim.fill_diagonal_(float("-inf"))
            tgt = torch.arange(BS, device=DEVICE)
            tgt = torch.cat([tgt + BS, tgt])
            loss = F.cross_entropy(sim, tgt)
        else:
            raise ValueError(objective)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc.eval()


def hermite_moment(sig: np.ndarray, k: int) -> np.ndarray:
    c = np.zeros(k + 1)
    c[k] = 1.0
    return hermeval(sig, c).mean(axis=1)


def spectrum(enc: PatchEncoder, sig: np.ndarray) -> dict:
    x = torch.tensor(sig, dtype=torch.float32, device=DEVICE)
    feats = []
    with torch.no_grad():
        for s in range(0, len(sig), 256):
            feats.append(enc(x[s:s + 256]).mean(1).cpu().numpy())
    H = np.concatenate(feats).astype(np.float64)
    rng = np.random.default_rng(42)
    n = len(sig)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.6 * n)], idx[int(0.8 * n) :]
    Xp = PCA(n_components=min(64, H.shape[1]), random_state=42).fit(H[tr]).transform(H)
    out = {}
    for k in DEG:
        f = hermite_moment(sig, k)
        reg = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(Xp[tr], f[tr])
        out[k] = round(float(r2_score(f[te], reg.predict(Xp[te]))), 4)
    return out


def main() -> None:
    seed_everything(0)
    corpus = make_corpus(CORPUS_N, L, seed=1)
    probe_sig = generate_anomaly_dataset(1000, L, seed=42).sequences  # same probe as the TSFMs
    init_state = {k: v.clone() for k, v in PatchEncoder().to(DEVICE).state_dict().items()}
    rows = []
    for obj in ["random", "forecast", "reconstruct", "contrastive"]:
        enc = train_arm(obj, corpus, init_state)
        spec = spectrum(enc, probe_sig)
        lo = (spec[1] + spec[2]) / 2
        hi = (spec[3] + spec[4] + spec[5] + spec[6]) / 4
        rows.append({"objective": obj, "A_by_degree": spec,
                     "A_low_H12": round(lo, 3), "A_high_H36": round(hi, 3)})
        print(f"{obj:12} A(k): " + " ".join(f"H{k}={spec[k]:+.2f}" for k in DEG)
              + f"  | low={lo:+.2f} high={hi:+.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"config": {"L": L, "patch": PATCH, "d": DMODEL, "layers": LAYERS, "steps": STEPS,
                    "corpus_n": CORPUS_N}, "arms": rows,
         "hypothesis": "H9: objective causally shapes A(k); forecast suppresses non-predictive "
         "high-degree functionals vs reconstruct/random."}, indent=2))
    print(f"\nsaved -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
