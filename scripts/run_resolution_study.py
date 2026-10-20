"""Resolution study: large-sample confirmatory comparisons of TSFM probes and non-model controls.

The canonical protocol scores 200 test windows per split seed, so one window is 0.005 and
differences of about 0.01 cannot be resolved. This study regenerates the anomaly tasks at
n = 50,000 (30,000 / 10,000 / 10,000 train / validation / test, one fixed split) and asks, with
paired tests on 10,000 test windows:

- H1 (resolution): does the best model differ from the strongest control, and if not, are the two
  equivalent within a pre-specified margin (TOST, margins 0.02 and 0.01)?
- H2 (pretraining): does a re-initialised copy of each model (same architecture, tokenisation and
  scaling) score as well as the pretrained model?
- H3 (conditional information): does a probe on [control, Z] beat a probe on the control alone
  (conditional probing; Hewitt et al., 2021), and vice versa?
- H4 (views): do max, final-token, or attention pooling of the token sequence recover information
  that mean pooling loses?
- H5 (signal strength): does the model-control difference change sign as the anomaly magnitude of
  the realistic generator is scaled from 0.25 to 2?
- H6 (surrogates): how much accuracy does each representation keep on IAAFT surrogates, which keep
  every window's value distribution and amplitude spectrum and destroy local structure?

Datasets (generator seeds differ from the canonical seed 42, so nothing here was seen before):
    realistic         run_realistic_anomaly.realistic_anomaly_dataset, n=50,000, seed 1001
    canonical         synthetic.generate_anomaly_dataset, n=50,000, seed 1001
    realistic_iaaft   IAAFT surrogate of every `realistic` window (50 iterations, seed 1001)
    snr_<s>           realistic generator, n=20,000, seed 2001, magnitude_scale s; every scale
                      shares backgrounds, anomaly kinds and positions

Subcommands (run in this order; each is idempotent):
    data      write the datasets to STORE/data
    extract   mean-pooled activations of every layer (or of --layers) of one (model, init, dataset),
              float16; --tokens also keeps the token sequence of those layers
    controls  raw signal, 8-D hand-crafted, ROCKET-style, tsfresh (efficient) features; for
              `canonical` also the Bayes log-likelihood ratio and max|x|
    probe     StandardScaler + logistic regression per layer at n_train in TRAIN_SIZES, layer chosen
              on validation accuracy, test probabilities kept
    views     H4 on the token sequences
    cond      H3 conditional probes at the protocol's fixed C = 1 (the joint probe overfits)
    cond-tuned  H3 with C chosen on validation log-loss for control, model and joint probes
    analyze   paired tests -> outputs/resolution_study/results.json

Large arrays go to STORE ($TSFMI_RESOLUTION_STORE, default outputs/resolution_store, gitignored);
the JSON results go to outputs/resolution_study/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

OUT = Path("outputs/resolution_study")
STORE = Path(os.environ.get("TSFMI_RESOLUTION_STORE", "outputs/resolution_store"))
MODELS = ("timesfm", "moment", "patchtst_fm", "gpt4ts", "chronos", "moirai", "timer")
INITS = ("pretrained", "random")
MODEL_SEED = 42
SEQ_LEN = 512
SNR_SCALES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
CONTROLS = ("raw_signal", "hand_crafted", "rocket", "tsfresh_efficient")
CANONICAL_EXTRA = ("bayes_llr", "max_abs")
MARGINS = (0.02, 0.01)


# ----------------------------------------------------------------------------------------- data
def dataset_names() -> list[str]:
    return ["realistic", "canonical", "realistic_iaaft", *[f"snr_{s}" for s in SNR_SCALES]]


def make_dataset(name: str) -> dict[str, NDArray]:
    from scripts.run_realistic_anomaly import realistic_anomaly_dataset
    from src.datasets.surrogates import iaaft
    from src.datasets.synthetic import generate_anomaly_dataset

    if name == "realistic":
        x, y, k = realistic_anomaly_dataset(50_000, SEQ_LEN, 1001, return_kinds=True)
        return {"x": x, "y": y, "kinds": k}
    if name == "canonical":
        d = generate_anomaly_dataset(50_000, SEQ_LEN, seed=1001)
        return {"x": d.sequences, "y": d.labels.astype(np.int64)}
    if name == "realistic_iaaft":
        src = load_data("realistic")
        return {"x": iaaft(src["x"], n_iter=50, seed=1001), "y": src["y"], "kinds": src["kinds"]}
    if name.startswith("snr_"):
        s = float(name.removeprefix("snr_"))
        x, y, k = realistic_anomaly_dataset(
            20_000, SEQ_LEN, 2001, return_kinds=True, magnitude_scale=s
        )
        return {"x": x, "y": y, "kinds": k}
    raise ValueError(name)


def load_data(name: str) -> dict[str, NDArray]:
    with np.load(STORE / "data" / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def split_indices(n: int) -> dict[str, NDArray]:
    """Fixed 60/20/20 split (seed 0): test first, then validation, then train."""
    idx = np.random.default_rng(0).permutation(n)
    n_te = n // 5
    return {"test": idx[:n_te], "val": idx[n_te : 2 * n_te], "train": idx[2 * n_te :]}


def train_sizes(n: int) -> list[int]:
    n_train = len(split_indices(n)["train"])
    return [s for s in (600, 3000) if s < n_train] + [n_train]


def cmd_data(_: argparse.Namespace) -> None:
    (STORE / "data").mkdir(parents=True, exist_ok=True)
    for name in dataset_names():
        path = STORE / "data" / f"{name}.npz"
        if path.exists():
            print(f"SKIP {path}")
            continue
        d = make_dataset(name)
        np.savez(path, **d)
        print(f"wrote {path}: x {d['x'].shape}, positive rate {d['y'].mean():.3f}", flush=True)


# ------------------------------------------------------------------------------------ extraction
def rep_dir(model: str, init: str, ds: str) -> Path:
    return STORE / "repr" / f"{model}-{init}" / ds


def safe(layer: str) -> str:
    return layer.replace(".", "_").replace("/", "_")


def cmd_extract(args: argparse.Namespace) -> None:
    import torch

    from scripts.extract_representations import load_model_wrapper
    from src.extractors.hook_manager import HookManager
    from src.models.random_init import reinitialize_
    from src.utils.device import resolve_device
    from src.utils.seed import seed_everything

    out = rep_dir(args.model, args.init, args.dataset)
    done = out / "meta.json"
    if done.exists() and not args.layers:
        print(f"SKIP {out}")
        return
    device = resolve_device(args.device)
    seed_everything(MODEL_SEED)
    wrapper = load_model_wrapper(args.model, None, device, seq_len=SEQ_LEN)
    counts = reinitialize_(wrapper, seed=MODEL_SEED) if args.init == "random" else None
    names = wrapper.get_layer_names()
    if args.layers:
        missing = set(args.layers) - {safe(n) for n in names}
        if missing:
            raise SystemExit(f"unknown layers {missing}")
        names = [n for n in names if safe(n) in args.layers]
        if all((out / f"{safe(n)}.npy").exists() for n in names) and not args.tokens:
            print(f"SKIP {out} {args.layers}")
            return
    x = load_data(args.dataset)["x"].astype(np.float32)
    n = len(x)
    out.mkdir(parents=True, exist_ok=True)
    pooled: dict[str, np.memmap] = {}
    tokens: dict[str, np.memmap] = {}
    t0 = time.time()
    with HookManager(wrapper.model, names) as hm, torch.inference_mode():
        for s in range(0, n, args.batch_size):
            wrapper.forward(torch.from_numpy(x[s : s + args.batch_size]).to(device))
            acts = hm.get_activations()
            for ln in names:
                a = acts[ln].float()
                a = a.reshape(a.shape[0], -1, a.shape[-1]) if a.ndim != 2 else a[:, None, :]
                if ln not in pooled:
                    pooled[ln] = np.lib.format.open_memmap(
                        out / f"{safe(ln)}.npy", "w+", np.float16, (n, a.shape[-1])
                    )
                    if args.tokens:
                        tdir = STORE / "tokens" / f"{args.model}-{args.init}" / args.dataset
                        tdir.mkdir(parents=True, exist_ok=True)
                        tokens[ln] = np.lib.format.open_memmap(
                            tdir / f"{safe(ln)}.npy", "w+", np.float16, (n, *a.shape[1:])
                        )
                pooled[ln][s : s + len(a)] = a.mean(dim=1).cpu().numpy().astype(np.float16)
                if args.tokens:
                    tokens[ln][s : s + len(a)] = a.cpu().numpy().astype(np.float16)
            hm.clear()
            if (s // args.batch_size) % 200 == 0:
                print(f"  {args.model}/{args.init}/{args.dataset} {s + len(a)}/{n}", flush=True)
    for m in [*pooled.values(), *tokens.values()]:
        m.flush()
    meta = json.loads(done.read_text()) if done.exists() else {}
    meta.update(
        {
            "model": args.model,
            "init": args.init,
            "dataset": args.dataset,
            "model_seed": MODEL_SEED,
            "reinit_counts": counts,
            "layers": sorted({*meta.get("layers", []), *(safe(n) for n in names)}),
            "layer_order": [safe(n) for n in wrapper.get_layer_names()],
            "n": n,
            "seconds": round(time.time() - t0, 1),
        }
    )
    done.write_text(json.dumps(meta, indent=2))
    print(f"done {out} in {time.time() - t0:.0f}s", flush=True)


# -------------------------------------------------------------------------------------- controls
def tsfresh_features(x: NDArray, n_jobs: int, chunk: int = 5000) -> NDArray[np.float32]:
    """tsfresh EfficientFCParameters features, NaN/inf set to 0 (as run_strong_baselines)."""
    import pandas as pd
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters

    parts = []
    columns = None
    for s in range(0, len(x), chunk):
        xs = x[s : s + chunk]
        n, length = xs.shape
        df = pd.DataFrame(
            {
                "id": np.repeat(np.arange(n), length),
                "time": np.tile(np.arange(length), n),
                "value": xs.ravel().astype(np.float64),
            }
        )
        f = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=EfficientFCParameters(),
            disable_progressbar=True,
            n_jobs=n_jobs,
        )
        columns = f.columns if columns is None else columns
        parts.append(f[columns].to_numpy())
        print(f"  tsfresh {s + n}/{len(x)}", flush=True)
    out = np.concatenate(parts)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def rocket_parallel(x: NDArray, n_jobs: int, chunk: int = 1000) -> NDArray[np.float32]:
    from joblib import Parallel, delayed

    from scripts.run_rocket_baseline import N_KERNELS, random_kernels, rocket_features

    kernels = random_kernels(seed=42, num_kernels=N_KERNELS)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(rocket_features)(x[s : s + chunk].astype(np.float64), kernels)
        for s in range(0, len(x), chunk)
    )
    return np.concatenate(parts).astype(np.float32)


def control_dir(ds: str) -> Path:
    return STORE / "controls" / ds


def cmd_controls(args: argparse.Namespace) -> None:
    from scripts.run_canonical_baselines import hand_crafted_features

    out = control_dir(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(args.dataset)
    x = d["x"].astype(np.float64)
    makers = {
        "raw_signal": lambda: x.astype(np.float32),
        "hand_crafted": lambda: hand_crafted_features(x).astype(np.float32),
        "rocket": lambda: rocket_parallel(x, args.jobs),
        "tsfresh_efficient": lambda: tsfresh_features(x, args.jobs),
    }
    if args.dataset == "canonical":
        makers["bayes_llr"] = lambda: (np.log(np.cosh(5.0 * x).mean(axis=1)) - 12.5)[:, None]
        makers["max_abs"] = lambda: np.abs(x).max(axis=1, keepdims=True)
    for name, make in makers.items():
        path = out / f"{name}.npy"
        if path.exists():
            print(f"SKIP {path}")
            continue
        t0 = time.time()
        f = np.asarray(make(), dtype=np.float32)
        np.save(path, f)
        print(f"wrote {path} {f.shape} in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------- probes
def fit_predict(
    X: NDArray, y: NDArray, sp: dict[str, NDArray], n_train: int
) -> tuple[float, NDArray, NDArray]:
    """Fit on the first n_train training windows; return val accuracy and val/test P(y=1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = sp["train"][:n_train]
    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=0, solver="lbfgs")
    )
    pipe.fit(np.asarray(X[tr], dtype=np.float32), y[tr])
    p_va = pipe.predict_proba(np.asarray(X[sp["val"]], dtype=np.float32))[:, 1]
    p_te = pipe.predict_proba(np.asarray(X[sp["test"]], dtype=np.float32))[:, 1]
    acc_va = float(np.mean((p_va > 0.5) == y[sp["val"]]))
    return acc_va, p_va, p_te


def rep_layers(rep: str, ds: str) -> list[tuple[str, Path]]:
    """[(layer, path)] of a representation in depth order. rep = 'model:init' or 'ctrl:name'."""
    kind, name = rep.split(":")
    if kind == "ctrl":
        return [(name, control_dir(ds) / f"{name}.npy")]
    model, init = name.rsplit("-", 1) if "-" in name else (name, "pretrained")
    d = rep_dir(model, init, ds)
    meta = json.loads((d / "meta.json").read_text())
    return [(ln, d / f"{ln}.npy") for ln in meta["layer_order"] if (d / f"{ln}.npy").exists()]


def probe_path(ds: str, rep: str) -> Path:
    return STORE / "probes" / ds / f"{rep.replace(':', '__')}.npz"


def _layer_job(path: Path, y: NDArray, sp: dict, n_train: int) -> tuple[float, NDArray, NDArray]:
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=4):
        return fit_predict(np.load(path, mmap_mode="r"), y, sp, n_train)


def run_probe(ds: str, rep: str, jobs: int) -> dict:
    """Layer selection on validation accuracy at every training size; keeps test P(y=1)."""
    from joblib import Parallel, delayed

    path = probe_path(ds, rep)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["summary"]))
    d = load_data(ds)
    y = d["y"].astype(np.int64)
    sp = split_indices(len(y))
    layers = rep_layers(rep, ds)
    summary: dict = {"rep": rep, "dataset": ds, "sizes": {}}
    arrays: dict[str, NDArray] = {}
    for n_train in train_sizes(len(y)):
        res = Parallel(n_jobs=min(jobs, len(layers)))(
            delayed(_layer_job)(p, y, sp, n_train) for _, p in layers
        )
        val = [r[0] for r in res]
        best = int(np.argmax(val))  # first layer wins ties
        p_te = res[best][2]
        acc_te = float(np.mean((p_te > 0.5) == y[sp["test"]]))
        summary["sizes"][str(n_train)] = {
            "layer": layers[best][0],
            "val_acc": val[best],
            "test_acc": acc_te,
            "val_acc_by_layer": dict(zip([ln for ln, _ in layers], val, strict=True)),
        }
        arrays[f"p_test_{n_train}"] = p_te.astype(np.float32)
        arrays[f"p_val_{n_train}"] = res[best][1].astype(np.float32)
        print(f"[{ds} {rep} n={n_train}] {layers[best][0]} val={val[best]:.4f} test={acc_te:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, summary=json.dumps(summary), **arrays)
    return summary


def model_reps(inits: tuple[str, ...] = INITS) -> list[str]:
    return [f"model:{m}-{i}" for i in inits for m in MODELS]


def control_reps(ds: str) -> list[str]:
    extra = CANONICAL_EXTRA if ds == "canonical" else ()
    return [f"ctrl:{c}" for c in (*CONTROLS, *extra)]


def available_reps(ds: str) -> list[str]:
    reps = [r for r in control_reps(ds) if (control_dir(ds) / f"{r[5:]}.npy").exists()]
    for r in model_reps():
        model, init = r[6:].rsplit("-", 1)
        if (rep_dir(model, init, ds) / "meta.json").exists():
            reps.append(r)
    return reps


def cmd_probe(args: argparse.Namespace) -> None:
    for ds in args.datasets:
        reps = args.reps or available_reps(ds)
        for rep in reps:
            run_probe(ds, rep, args.jobs)


def selected_layer(model: str) -> str:
    """Layer chosen on validation for the pretrained model on `realistic` at the largest n."""
    s = run_probe("realistic", f"model:{model}-pretrained", 1)
    return s["sizes"][max(s["sizes"], key=int)]["layer"]


def cmd_layers(_: argparse.Namespace) -> None:
    """Print 'model layer' lines for the shell driver of the second extraction pass."""
    for m in MODELS:
        print(m, selected_layer(m))


# ----------------------------------------------------------------------------------------- views
def attention_probe(
    tok: NDArray,
    y: NDArray,
    sp: dict[str, NDArray],
    heads: int = 4,
    seed: int = 0,
    return_pooled: bool = False,
) -> tuple[float, NDArray] | tuple[float, NDArray, NDArray]:
    """Multi-query attention pooling over tokens + linear read-out, early-stopped on validation.

    Tokens are standardised per dimension with training statistics. Returns validation accuracy and
    test P(y=1); with ``return_pooled`` also the pooled vectors (N, heads * d) of every window,
    computed with the attention weights learned on the training split.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = sp["train"]
    sample = np.asarray(tok[tr[:5000]], dtype=np.float32)
    mu = torch.tensor(sample.mean(axis=(0, 1)), device=dev)
    sd = torch.tensor(sample.std(axis=(0, 1)) + 1e-6, device=dev)
    data = {k: torch.tensor(np.asarray(tok[v]), device=dev) for k, v in sp.items()}
    labels = {k: torch.tensor(y[v], dtype=torch.float32, device=dev) for k, v in sp.items()}
    dim = tok.shape[-1]

    class Pool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Parameter(torch.randn(heads, dim) * 0.01)
            self.out = nn.Linear(heads * dim, 1)

        def pool(self, t: torch.Tensor) -> torch.Tensor:
            z = (t.float() - mu) / sd
            a = torch.softmax(torch.einsum("npd,hd->nhp", z, self.q) / dim**0.5, dim=-1)
            return torch.einsum("nhp,npd->nhd", a, z).flatten(1)

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return self.out(self.pool(t)).squeeze(-1)

    net = Pool().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    def evaluate(split: str) -> tuple[float, torch.Tensor]:
        with torch.no_grad():
            logits = torch.cat([net(c) for c in data[split].split(2048)])
        return float(loss_fn(logits, labels[split])), torch.sigmoid(logits)

    best, best_state, patience = np.inf, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _epoch in range(100):
        net.train()
        for b in torch.randperm(len(tr), generator=g).split(256):
            opt.zero_grad()
            loss_fn(net(data["train"][b.to(dev)]), labels["train"][b.to(dev)]).backward()
            opt.step()
        net.eval()
        loss, _ = evaluate("val")
        if loss < best - 1e-4:
            best, patience = loss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is None:
        raise RuntimeError("attention probe: validation loss never finite")
    net.load_state_dict(best_state)
    _, p_va = evaluate("val")
    _, p_te = evaluate("test")
    acc_va = float(((p_va > 0.5).float() == labels["val"]).float().mean())
    if not return_pooled:
        return acc_va, p_te.cpu().numpy()
    pooled = np.zeros((len(y), heads * dim), dtype=np.float32)
    with torch.no_grad():
        for k, idx in sp.items():
            pooled[idx] = torch.cat([net.pool(c) for c in data[k].split(2048)]).cpu().numpy()
    return acc_va, p_te.cpu().numpy(), pooled


def cmd_views(args: argparse.Namespace) -> None:
    """H4: mean / max / final-token / attention pooling of the selected layer's tokens."""
    ds = "realistic"
    y = load_data(ds)["y"].astype(np.int64)
    sp = split_indices(len(y))
    n_train = len(sp["train"])
    out_path = STORE / "probes" / ds / "views.npz"
    store = dict(np.load(out_path)) if out_path.exists() else {}
    summary = json.loads(str(store.pop("summary"))) if "summary" in store else {}
    # Re-initialised models are read at the pretrained model's selected layer, so the comparison
    # changes only the weights.
    views = (
        ("mean", "max", "last", "attention") if args.init == "pretrained" else ("mean", "attention")
    )
    for m in args.models:
        layer = selected_layer(m)
        tok = np.load(STORE / "tokens" / f"{m}-{args.init}" / ds / f"{layer}.npy", mmap_mode="r")
        for view in views:
            key = f"{m}/{view}" if args.init == "pretrained" else f"{m}-random/{view}"
            if key in summary:
                continue
            t0 = time.time()
            if view == "attention":
                acc_va, p_te = attention_probe(tok, y, sp)
            else:
                reduce = {
                    "mean": lambda c: c.astype(np.float32).mean(axis=1),
                    "max": lambda c: c.astype(np.float32).max(axis=1),
                    "last": lambda c: c[:, -1].astype(np.float32),
                }[view]
                X = np.concatenate([reduce(tok[s : s + 2000]) for s in range(0, len(y), 2000)])
                acc_va, _, p_te = fit_predict(X, y, sp, n_train)
            acc_te = float(np.mean((p_te > 0.5) == y[sp["test"]]))
            summary[key] = {
                "layer": layer,
                "val_acc": acc_va,
                "test_acc": acc_te,
                "tokens": int(tok.shape[1]),
            }
            store[f"p_test/{key}"] = np.asarray(p_te, dtype=np.float32)
            print(f"[views {key}] val={acc_va:.4f} test={acc_te:.4f} ({time.time() - t0:.0f}s)")
            np.savez(out_path, summary=json.dumps(summary), **store)


def mlp_probe(
    X: NDArray, y: NDArray, sp: dict[str, NDArray], hidden: int = 512, seed: int = 0
) -> tuple[float, NDArray]:
    """One-hidden-layer MLP (dropout 0.2, Adam), early-stopped on validation loss."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    mu, sd = X[sp["train"]].mean(0), X[sp["train"]].std(0) + 1e-6
    data = {k: torch.tensor((X[v] - mu) / sd, device=dev) for k, v in sp.items()}
    labels = {k: torch.tensor(y[v], dtype=torch.float32, device=dev) for k, v in sp.items()}
    net = nn.Sequential(
        nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1)
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, patience = np.inf, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _epoch in range(200):
        net.train()
        for b in torch.randperm(len(sp["train"]), generator=g).split(256):
            b = b.to(dev)
            opt.zero_grad()
            loss_fn(net(data["train"][b]).squeeze(-1), labels["train"][b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            loss = float(loss_fn(net(data["val"]).squeeze(-1), labels["val"]))
        if loss < best - 1e-4:
            best, patience = loss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        p_va = torch.sigmoid(net(data["val"]).squeeze(-1))
        p_te = torch.sigmoid(net(data["test"]).squeeze(-1)).cpu().numpy()
    return float(((p_va > 0.5).float() == labels["val"]).float().mean()), p_te


def cmd_capacity_controls(_: argparse.Namespace) -> None:
    """Capacity-matched controls for the attention view: an MLP on all non-model controls, and the
    same attention probe on raw-signal patch tokens (patch lengths 16 and 32)."""
    ds = "realistic"
    d = load_data(ds)
    y = d["y"].astype(np.int64)
    sp = split_indices(len(y))
    out_path = STORE / "probes" / ds / "capacity_controls.npz"
    store = dict(np.load(out_path)) if out_path.exists() else {}
    summary = json.loads(str(store.pop("summary"))) if "summary" in store else {}
    jobs = {
        "mlp_all_controls": lambda: mlp_probe(rep_matrix(ds, "ctrl:all_controls"), y, sp),
        "mlp_rocket": lambda: mlp_probe(rep_matrix(ds, "ctrl:rocket"), y, sp),
    }
    for plen in (16, 32):
        x = d["x"].astype(np.float32)
        tokens = x.reshape(len(x), -1, plen)
        tokens = (tokens - tokens.mean(axis=(1, 2), keepdims=True)) / (
            tokens.std(axis=(1, 2), keepdims=True) + 1e-6
        )
        jobs[f"attention_raw_patch{plen}"] = lambda t=tokens: attention_probe(t, y, sp)
    for name, run in jobs.items():
        if name in summary:
            continue
        acc_va, p_te = run()
        summary[name] = {
            "val_acc": acc_va,
            "test_acc": float(np.mean((p_te > 0.5) == y[sp["test"]])),
        }
        store[f"p_test/{name}"] = np.asarray(p_te, dtype=np.float32)
        print(
            f"[capacity {name}] val={acc_va:.4f} test={summary[name]['test_acc']:.4f}", flush=True
        )
        np.savez(out_path, summary=json.dumps(summary), **store)


def crossfit_pooled(
    tok: NDArray, y: NDArray, sp: dict[str, NDArray], folds: int = 5
) -> tuple[float, NDArray, NDArray]:
    """Attention-pooled vectors usable as features by a second-stage probe.

    Training rows get out-of-fold vectors: the training split is cut into ``folds`` parts and each
    part is pooled by an attention net trained on the other parts (early-stopped on validation).
    Validation and test rows are pooled by the net trained on the whole training split, whose
    validation accuracy and test P(y=1) are also returned. No row's own label shapes its vector.
    """
    acc_va, p_te, pooled = attention_probe(tok, y, sp, return_pooled=True)
    tr = sp["train"]
    parts = np.array_split(np.random.default_rng(0).permutation(len(tr)), folds)
    for k, held in enumerate(parts):
        fit = np.setdiff1d(np.arange(len(tr)), held)
        sub = {"train": tr[fit], "val": sp["val"], "test": tr[held]}
        _, _, pk = attention_probe(tok, y, sub, seed=k + 1, return_pooled=True)
        pooled[tr[held]] = pk[tr[held]]
    return acc_va, p_te, pooled


def cmd_views_cond(args: argparse.Namespace) -> None:
    """H4 + H3: does an attention-pooled model view add to the union of the feature controls?

    The attention probe's pooled vectors (cross-fitted on the training split, see
    ``crossfit_pooled``) are features for a logistic probe on [feature controls, pooled], compared
    with the feature-control probe and with a probe on the pooled vectors alone, every logistic
    probe with C tuned on validation log-loss. ``--init random`` reads the re-initialised model's
    tokens at the pretrained model's selected layer.
    """
    ds = "realistic"
    y = load_data(ds)["y"].astype(np.int64)
    sp = split_indices(len(y))
    suffix = "" if args.init == "pretrained" else "_random"
    out_path = STORE / "probes" / ds / f"views_conditional_cf{suffix}.npz"
    store = dict(np.load(out_path)) if out_path.exists() else {}
    summary = json.loads(str(store.pop("summary"))) if "summary" in store else {}
    C_all = rep_matrix(ds, "ctrl:all_controls")
    for m in args.models:
        key = m if args.init == "pretrained" else f"{m}-random"
        if key in summary:
            continue
        layer = selected_layer(m)
        tok = np.load(STORE / "tokens" / f"{m}-{args.init}" / ds / f"{layer}.npy", mmap_mode="r")
        acc_va, p_att, pooled = crossfit_pooled(tok, y, sp)
        c_p, acc_p, p_pool = fit_predict_tuned(pooled, y, sp, len(sp["train"]))
        c_j, acc_j, p_joint = fit_predict_tuned(
            np.concatenate([C_all, pooled], axis=1), y, sp, len(sp["train"])
        )
        summary[key] = {"layer": layer, "val_attention": acc_va, "C_pooled": c_p, "C_joint": c_j}
        store[f"p_test/{key}/attention"] = p_att.astype(np.float32)
        store[f"p_test/{key}/pooled_linear"] = p_pool.astype(np.float32)
        store[f"p_test/{key}/joint_all"] = p_joint.astype(np.float32)
        print(f"[views-cond {key}] attention val={acc_va:.4f} joint val={acc_j:.4f}", flush=True)
        np.savez(out_path, summary=json.dumps(summary), **store)


# ------------------------------------------------------------------------------------ conditional
def rep_matrix(ds: str, rep: str, layer: str | None = None) -> NDArray:
    """Feature matrix of a representation at its selected layer (largest training size)."""
    kind, name = rep.split(":")
    if kind == "ctrl":
        if name == "all_controls":
            return np.concatenate(
                [np.load(control_dir(ds) / f"{c}.npy") for c in CONTROLS if c != "raw_signal"],
                axis=1,
            )
        return np.load(control_dir(ds) / f"{name}.npy")
    if layer is None:
        s = run_probe(ds, rep, 1)
        layer = s["sizes"][max(s["sizes"], key=int)]["layer"]
    model, init = name.rsplit("-", 1)
    return np.load(rep_dir(model, init, ds) / f"{layer}.npy")


def _cond_job(
    ds: str, rep: str, ctrl: str, y: NDArray, sp: dict[str, NDArray]
) -> tuple[str, float, NDArray]:
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=8):
        X = np.concatenate(
            [rep_matrix(ds, f"ctrl:{ctrl}"), rep_matrix(ds, rep).astype(np.float32)], axis=1
        )
        acc_va, _, p_te = fit_predict(X, y, sp, len(sp["train"]))
    return f"{rep}|{ctrl}", acc_va, p_te


C_GRID = (0.001, 0.01, 0.1, 1.0)


def fit_predict_tuned(
    X: NDArray, y: NDArray, sp: dict[str, NDArray], n_train: int
) -> tuple[float, float, NDArray]:
    """Logistic regression with C chosen from C_GRID by validation log-loss (warm-started path).

    Returns the chosen C, the validation accuracy, and test P(y=1). Used for conditional probing,
    where a fixed C lets the high-dimensional joint probe overfit and understate what it adds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler

    tr = sp["train"][:n_train]
    scaler = StandardScaler().fit(np.asarray(X[tr], dtype=np.float32))
    X_tr = scaler.transform(np.asarray(X[tr], dtype=np.float32))
    X_va = scaler.transform(np.asarray(X[sp["val"]], dtype=np.float32))
    X_te = scaler.transform(np.asarray(X[sp["test"]], dtype=np.float32))
    lr = LogisticRegression(max_iter=2000, random_state=0, solver="lbfgs", warm_start=True)
    best = (np.inf, None, None, None)
    for c in C_GRID:
        lr.set_params(C=c).fit(X_tr, y[tr])
        p_va = lr.predict_proba(X_va)[:, 1]
        loss = log_loss(y[sp["val"]], p_va, labels=[0, 1])
        if loss < best[0]:
            acc_va = float(np.mean((p_va > 0.5) == y[sp["val"]]))
            best = (loss, c, acc_va, lr.predict_proba(X_te)[:, 1])
    return best[1], best[2], best[3]


def _tuned_job(ds: str, key: str, y: NDArray, sp: dict[str, NDArray]) -> tuple[str, dict, NDArray]:
    """key: 'C|<ctrl>', 'Z|<rep>' or 'J|<rep>|<ctrl>'."""
    from threadpoolctl import threadpool_limits

    parts = key.split("|")
    with threadpool_limits(limits=6):
        if parts[0] == "C":
            X = rep_matrix(ds, f"ctrl:{parts[1]}")
        elif parts[0] == "Z":
            X = rep_matrix(ds, parts[1]).astype(np.float32)
        else:
            X = np.concatenate(
                [rep_matrix(ds, f"ctrl:{parts[2]}"), rep_matrix(ds, parts[1]).astype(np.float32)],
                axis=1,
            )
        c, acc_va, p_te = fit_predict_tuned(X, y, sp, len(sp["train"]))
    return key, {"C": c, "val_acc": acc_va, "dim": int(X.shape[1])}, p_te


def cmd_cond(args: argparse.Namespace) -> None:
    """H3: probes on [control, Z] for every model representation and control set."""
    from joblib import Parallel, delayed

    for ds in args.datasets:
        y = load_data(ds)["y"].astype(np.int64)
        sp = split_indices(len(y))
        n_train = len(sp["train"])
        out_path = STORE / "probes" / ds / "conditional.npz"
        store = dict(np.load(out_path)) if out_path.exists() else {}
        summary = json.loads(str(store.pop("summary"))) if "summary" in store else {}
        ctrls = ["hand_crafted", "rocket", "tsfresh_efficient", "all_controls"]
        reps = [r for r in available_reps(ds) if r.startswith("model:")]
        todo = [(r, c) for r in reps for c in ctrls if f"{r}|{c}" not in summary]

        if "ctrl:all_controls" not in summary:
            acc_va, _, p_te = fit_predict(rep_matrix(ds, "ctrl:all_controls"), y, sp, n_train)
            summary["ctrl:all_controls"] = {"val_acc": acc_va}
            store["p_test/ctrl:all_controls"] = p_te.astype(np.float32)
        jobs = Parallel(n_jobs=args.jobs)(delayed(_cond_job)(ds, r, c, y, sp) for r, c in todo)
        for key, acc_va, p_te in jobs:
            summary[key] = {"val_acc": acc_va}
            store[f"p_test/{key}"] = p_te.astype(np.float32)
            print(f"[cond {ds} {key}] val={acc_va:.4f}", flush=True)
        np.savez(out_path, summary=json.dumps(summary), **store)


def cmd_cond_tuned(args: argparse.Namespace) -> None:
    """H3 with C tuned on validation log-loss for the control, model and joint probes alike."""
    from joblib import Parallel, delayed

    ctrls = ["hand_crafted", "rocket", "tsfresh_efficient", "all_controls"]
    for ds in args.datasets:
        y = load_data(ds)["y"].astype(np.int64)
        sp = split_indices(len(y))
        out_path = STORE / "probes" / ds / "conditional_tuned.npz"
        store = dict(np.load(out_path)) if out_path.exists() else {}
        summary = json.loads(str(store.pop("summary"))) if "summary" in store else {}
        reps = [r for r in available_reps(ds) if r.startswith("model:")]
        if args.reps:
            reps = [r for r in reps if r in args.reps]
        keys = [f"C|{c}" for c in ctrls] + [f"Z|{r}" for r in reps]
        keys += [f"J|{r}|{c}" for r in reps for c in ctrls]
        todo = [k for k in keys if k not in summary]
        for key, info, p_te in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(_tuned_job)(ds, k, y, sp) for k in todo
        ):
            summary[key] = info
            store[f"p_test/{key}"] = p_te.astype(np.float32)
            print(f"[cond-tuned {ds} {key}] C={info['C']} val={info['val_acc']:.4f}", flush=True)
            np.savez(out_path, summary=json.dumps(summary), **store)


# --------------------------------------------------------------------------------------- analyze
def _test_probs(ds: str, rep: str, n_train: int | None = None) -> tuple[NDArray, dict]:
    s = run_probe(ds, rep, 1)
    key = str(n_train) if n_train is not None else max(s["sizes"], key=int)
    with np.load(probe_path(ds, rep)) as z:
        return z[f"p_test_{key}"], s["sizes"][key]


def compare(p_a: NDArray, p_b: NDArray, y: NDArray) -> dict:
    from src.metrics.paired_tests import mcnemar, paired_logloss_difference, tost

    ca, cb = (p_a > 0.5) == y, (p_b > 0.5) == y
    out = {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()), "mcnemar": mcnemar(ca, cb)}
    for m in MARGINS:
        out[f"tost_{m}"] = tost(ca, cb, margin=m)
    out["logloss_bits"] = paired_logloss_difference(p_a, p_b, y)
    return out


def analyze_dataset(ds: str) -> dict:
    from src.metrics.paired_tests import mcnemar_sample_size, vinfo_bits

    d = load_data(ds)
    y = d["y"].astype(np.int64)
    sp = split_indices(len(y))
    y_te, y_tr = y[sp["test"]], y[sp["train"]]
    reps = available_reps(ds)
    table = {}
    for rep in reps:
        s = run_probe(ds, rep, 1)
        p, _ = _test_probs(ds, rep)
        table[rep] = {
            "sizes": {
                k: {kk: v[kk] for kk in ("layer", "val_acc", "test_acc")}
                for k, v in s["sizes"].items()
            },
            "vinfo_bits": vinfo_bits(p, y_te, y_tr),
        }
    pretrained = [r for r in reps if r.endswith("-pretrained")]
    ctrls = [r for r in reps if r.startswith("ctrl:")]
    n_max = max(table[reps[0]]["sizes"], key=int)

    def best(group: list[str]) -> str:  # chosen on validation accuracy, never on test
        return max(group, key=lambda r: table[r]["sizes"][n_max]["val_acc"])

    res: dict = {"n": int(len(y)), "n_test": int(len(y_te)), "table": table}
    if pretrained and ctrls:
        bm, bc = best(pretrained), best(ctrls)
        p_m, _ = _test_probs(ds, bm)
        p_c, _ = _test_probs(ds, bc)
        res["best_model"], res["best_control"] = bm, bc
        n_min = min(table[reps[0]]["sizes"], key=int)
        bm0 = max(pretrained, key=lambda r: table[r]["sizes"][n_min]["val_acc"])
        bc0 = max(ctrls, key=lambda r: table[r]["sizes"][n_min]["val_acc"])
        res["at_min_train"] = {
            "n_train": int(n_min),
            "best_model": bm0,
            "best_control": bc0,
            "compare": compare(
                _test_probs(ds, bm0, int(n_min))[0], _test_probs(ds, bc0, int(n_min))[0], y_te
            ),
        }
        res["best_model_vs_best_control"] = compare(p_m, p_c, y_te)
        disc = float(np.mean((p_m > 0.5) != (p_c > 0.5)))
        res["discordance"] = disc
        res["n_needed"] = {
            str(dd): mcnemar_sample_size(dd, max(disc, 1e-3)) for dd in (0.005, 0.01, 0.02)
        }
        res["models_vs_best_control"] = {
            r: compare(_test_probs(ds, r)[0], p_c, y_te) for r in reps if r.startswith("model:")
        }
        res["pretrained_vs_random"] = {}
        for m in MODELS:
            a, b = f"model:{m}-pretrained", f"model:{m}-random"
            if a in table and b in table:
                res["pretrained_vs_random"][m] = compare(
                    _test_probs(ds, a)[0], _test_probs(ds, b)[0], y_te
                )
    tpath = STORE / "probes" / ds / "conditional_tuned.npz"
    if tpath.exists():
        with np.load(tpath) as z:
            summary = json.loads(str(z["summary"]))
            probs = {k.removeprefix("p_test/"): z[k] for k in z.files if k.startswith("p_test/")}
        cond = {}
        for key in summary:
            if not key.startswith("J|"):
                continue
            _, rep, ctrl = key.split("|")
            if f"C|{ctrl}" not in probs or f"Z|{rep}" not in probs:
                continue
            cond[f"{rep}|{ctrl}"] = {
                "z_given_control": compare(probs[key], probs[f"C|{ctrl}"], y_te),
                "control_given_z": compare(probs[key], probs[f"Z|{rep}"], y_te),
                "C": {
                    "joint": summary[key]["C"],
                    "control": summary[f"C|{ctrl}"]["C"],
                    "z": summary[f"Z|{rep}"]["C"],
                },
            }
        res["conditional_tuned"] = cond
        res["controls_tuned"] = {
            k[2:]: float(np.mean((v > 0.5) == y_te)) for k, v in probs.items() if k.startswith("C|")
        }
    cpath = STORE / "probes" / ds / "conditional.npz"
    if cpath.exists():
        with np.load(cpath) as z:
            summary = json.loads(str(z["summary"]))
            probs = {k.removeprefix("p_test/"): z[k] for k in z.files if k.startswith("p_test/")}
        cond = {}
        for key in summary:
            if "|" not in key:
                continue
            rep, ctrl = key.split("|")
            p_joint = probs[key]
            p_ctrl = probs["ctrl:all_controls"] if ctrl == "all_controls" else None
            if p_ctrl is None:
                p_ctrl, _ = _test_probs(ds, f"ctrl:{ctrl}")
            p_rep, _ = _test_probs(ds, rep)
            cond[key] = {
                "z_given_control": compare(p_joint, p_ctrl, y_te),  # I(Z -> Y | C)
                "control_given_z": compare(p_joint, p_rep, y_te),  # I(C -> Y | Z)
                "val_acc_joint": summary[key]["val_acc"],
            }
        res["conditional"] = cond
        res["all_controls"] = {
            "test_acc": float(np.mean((probs["ctrl:all_controls"] > 0.5) == y_te)),
            "vinfo_bits": vinfo_bits(probs["ctrl:all_controls"], y_te, y_tr),
        }
    if "kinds" in d:
        k_te = d["kinds"][sp["test"]]
        res["recall_by_kind"] = {}
        for rep in reps:
            p, _ = _test_probs(ds, rep)
            res["recall_by_kind"][rep] = {
                str(k): float(np.mean(p[k_te == k] > 0.5)) for k in (0, 1, 2, 3)
            } | {"normal_specificity": float(np.mean(p[k_te == -1] <= 0.5))}
    return res


def cmd_analyze(_: argparse.Namespace) -> None:
    results: dict = {
        "protocol": {
            "split": "fixed 60/20/20 (permutation seed 0)",
            "probe": "StandardScaler + LogisticRegression(lbfgs, C=1, max_iter=2000)",
            "layer_selection": "validation accuracy at each training size, first layer on ties",
            "margins": list(MARGINS),
            "model_seed": MODEL_SEED,
        }
    }
    for ds in dataset_names():
        if (STORE / "probes" / ds).exists():
            results[ds] = analyze_dataset(ds)
            print(f"analyzed {ds}", flush=True)
    # H6: the same test windows, original vs IAAFT surrogate
    if "realistic" in results and "realistic_iaaft" in results:
        y_te = load_data("realistic")["y"][split_indices(50_000)["test"]]
        results["surrogate_drop"] = {
            rep: compare(
                _test_probs("realistic", rep)[0], _test_probs("realistic_iaaft", rep)[0], y_te
            )
            for rep in results["realistic_iaaft"]["table"]
            if rep in results["realistic"]["table"]
        }
    # H4
    vpath = STORE / "probes" / "realistic" / "views.npz"
    if vpath.exists():
        with np.load(vpath) as z:
            summary = json.loads(str(z["summary"]))
            y_te = load_data("realistic")["y"][split_indices(50_000)["test"]]
            best_c = results["realistic"]["best_control"]
            p_c, _ = _test_probs("realistic", best_c)
            results["views"] = {
                k: v | {"vs_best_control": compare(z[f"p_test/{k}"], p_c, y_te)}
                for k, v in summary.items()
            }
    ccpath = STORE / "probes" / "realistic" / "capacity_controls.npz"
    if ccpath.exists():
        y_te = load_data("realistic")["y"][split_indices(50_000)["test"]]
        with np.load(ccpath) as z:
            summary = json.loads(str(z["summary"]))
            results["capacity_controls"] = {
                k: v | {"p_test_key": f"p_test/{k}"} for k, v in summary.items()
            }
            if "views" in results:
                for key in results["views"]:
                    if key.endswith("/attention") and "mlp_all_controls" in summary:
                        results["views"][key]["vs_mlp_all_controls"] = compare(
                            np.load(vpath)[f"p_test/{key}"],
                            z["p_test/mlp_all_controls"],
                            y_te,
                        )
    tpath = STORE / "probes" / "realistic" / "conditional_tuned.npz"
    nocf = STORE / "probes" / "realistic" / "views_conditional.npz"  # before cross-fitting
    if nocf.exists() and tpath.exists():
        y_te = load_data("realistic")["y"][split_indices(50_000)["test"]]
        with np.load(nocf) as z, np.load(tpath) as t:
            summary = json.loads(str(z["summary"]))
            results["views_conditional_nocrossfit"] = {
                m: compare(z[f"p_test/{m}/joint_all"], t["p_test/C|all_controls"], y_te)
                for m in summary
            }
    for suffix in ("", "_random"):
        vcpath = STORE / "probes" / "realistic" / f"views_conditional_cf{suffix}.npz"
        if not (vcpath.exists() and tpath.exists()):
            continue
        y_te = load_data("realistic")["y"][split_indices(50_000)["test"]]
        with np.load(vcpath) as z, np.load(tpath) as t:
            summary = json.loads(str(z["summary"]))
            p_all = t["p_test/C|all_controls"]
            vc = results.setdefault("views_conditional", {})
            for m in summary:
                vc[m] = {
                    "attention_vs_all_controls": compare(z[f"p_test/{m}/attention"], p_all, y_te),
                    "pooled_given_controls": compare(z[f"p_test/{m}/joint_all"], p_all, y_te),
                    "controls_given_pooled": compare(
                        z[f"p_test/{m}/joint_all"], z[f"p_test/{m}/pooled_linear"], y_te
                    ),
                    **summary[m],
                }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'results.json'}")


# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True, choices=MODELS)
    e.add_argument("--init", default="pretrained", choices=INITS)
    e.add_argument("--dataset", required=True)
    e.add_argument("--layers", nargs="*", default=None)
    e.add_argument("--tokens", action="store_true")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--device", default=None)
    c = sub.add_parser("controls")
    c.add_argument("--dataset", required=True)
    c.add_argument("--jobs", type=int, default=32)
    p = sub.add_parser("probe")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--reps", nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=16)
    sub.add_parser("layers")
    v = sub.add_parser("views")
    v.add_argument("--models", nargs="+", default=list(MODELS))
    v.add_argument("--init", default="pretrained", choices=INITS)
    sub.add_parser("capacity-controls")
    vc = sub.add_parser("views-cond")
    vc.add_argument("--models", nargs="+", default=list(MODELS))
    vc.add_argument("--init", default="pretrained", choices=INITS)
    cd = sub.add_parser("cond")
    cd.add_argument("--datasets", nargs="+", required=True)
    cd.add_argument("--jobs", type=int, default=8)
    ct = sub.add_parser("cond-tuned")
    ct.add_argument("--datasets", nargs="+", required=True)
    ct.add_argument("--reps", nargs="*", default=None)
    ct.add_argument("--jobs", type=int, default=8)
    sub.add_parser("analyze")
    args = ap.parse_args(argv)
    {
        "data": cmd_data,
        "extract": cmd_extract,
        "controls": cmd_controls,
        "probe": cmd_probe,
        "layers": cmd_layers,
        "views": cmd_views,
        "cond": cmd_cond,
        "cond-tuned": cmd_cond_tuned,
        "views-cond": cmd_views_cond,
        "capacity-controls": cmd_capacity_controls,
        "analyze": cmd_analyze,
    }[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
