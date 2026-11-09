"""Lower bound on the Bayes accuracy of the realistic anomaly task, and the headroom it implies.

The realistic generator (scripts/run_realistic_anomaly.py) mixes four anomaly kinds with random
position, length and a magnitude that depends on the window, so its likelihood ratio has no closed
form. Any classifier's accuracy is a lower bound on the Bayes accuracy a*, so a specialist trained
on far more windows than the probes see bounds the headroom of the control union C from below:
a* - a(C) >= a(specialist) - a(C). The share of that headroom a representation closes is then at
most gain / (a(specialist) - a(C)).

The specialist is a 1-D ResNet on the raw window and its first difference, trained on fresh windows
from data seeds 2001-2020 (50,000 each, disjoint from the evaluation seeds), with early stopping on
the validation split of seed 1001. Training sizes 30,000 (the probes' size) to 1,000,000 give a
learning curve. It is scored on the test splits of seeds 1001-1003 (the replication's windows) and
compared with the tuned all-controls probe on the same windows.

Usage:
    TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_bayes_headroom.py data
    ... run_bayes_headroom.py train --n 1000000
    ... run_bayes_headroom.py summarize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.run_replication import SEEDS, controls, dataset
from scripts.run_resolution_study import STORE, compare, fit_predict_tuned, split_indices

TRAIN_SEEDS = tuple(range(2001, 2021))
CHUNK = 50_000
SIZES = (30_000, 100_000, 300_000, 1_000_000)
OUT = Path("outputs/bayes_headroom")
DIR = STORE / "bayes_headroom"


def chunk_path(seed: int) -> Path:
    return DIR / f"train_{seed}.npz"


def cmd_data(args: argparse.Namespace) -> None:
    from joblib import Parallel, delayed

    from scripts.run_realistic_anomaly import realistic_anomaly_dataset

    DIR.mkdir(parents=True, exist_ok=True)

    def make(seed: int) -> None:
        if chunk_path(seed).exists():
            return
        x, y = realistic_anomaly_dataset(CHUNK, 512, seed)
        np.savez(chunk_path(seed), x=x.astype(np.float32), y=y.astype(np.int8))

    Parallel(n_jobs=args.jobs)(delayed(make)(s) for s in TRAIN_SEEDS)
    print(f"wrote {len(TRAIN_SEEDS)} chunks of {CHUNK} windows to {DIR}", flush=True)


def load_train(n: int) -> tuple[NDArray, NDArray]:
    xs, ys = [], []
    for s in TRAIN_SEEDS[: -(-n // CHUNK)]:
        with np.load(chunk_path(s)) as z:
            xs.append(z["x"])
            ys.append(z["y"])
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n].astype(np.float32)


def build_net(width: int = 64):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c_in: int, c_out: int, k: int, stride: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(c_in, c_out, k, stride, k // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Conv1d(c_out, c_out, k, 1, k // 2, bias=False),
                nn.BatchNorm1d(c_out),
            )
            self.skip = (
                nn.Identity()
                if c_in == c_out and stride == 1
                else nn.Sequential(
                    nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out)
                )
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.body(x) + self.skip(x))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(2, width, 7, 1, 3, bias=False), nn.BatchNorm1d(width), nn.ReLU()
            )
            self.blocks = nn.Sequential(
                Block(width, width, 7, 1),
                Block(width, 2 * width, 5, 2),
                Block(2 * width, 2 * width, 3, 2),
            )
            self.head = nn.Linear(4 * width, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.blocks(self.stem(x))
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(-1)

    return Net()


def to_input(x: NDArray, mu: float, sd: float):
    import torch

    x = (x - mu) / sd
    d = np.diff(x, axis=1, prepend=x[:, :1])
    return torch.from_numpy(np.stack([x, d], 1).astype(np.float32))


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from torch import nn

    out_path = DIR / f"specialist_{args.n}.json"
    if out_path.exists():
        print(f"SKIP {out_path}")
        return
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr = load_train(args.n)
    mu, sd = float(x_tr.mean()), float(x_tr.std())
    d0 = dataset(1001)
    sp0 = split_indices(len(d0["y"]))
    x_va = to_input(d0["x"][sp0["val"]].astype(np.float32), mu, sd).to(dev)
    y_va = torch.tensor(d0["y"][sp0["val"]], dtype=torch.float32, device=dev)
    X = to_input(x_tr, mu, sd)
    Y = torch.from_numpy(y_tr)
    net = build_net().to(dev)
    max_steps = 60_000
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    def predict(inp: torch.Tensor) -> torch.Tensor:
        net.eval()
        with torch.no_grad():
            return torch.cat([net(b) for b in inp.split(4096)])

    best, best_state, bad, step = np.inf, None, 0, 0
    g = torch.Generator().manual_seed(0)
    eval_every = 1000
    while step < max_steps and bad < 10:
        for b in torch.randperm(len(X), generator=g).split(512):
            net.train()
            opt.zero_grad()
            loss_fn(net(X[b].to(dev, non_blocking=True)), Y[b].to(dev)).backward()
            opt.step()
            sched.step()
            step += 1
            if step % eval_every == 0:
                loss = float(loss_fn(predict(x_va), y_va))
                if loss < best - 1e-4:
                    best, bad = loss, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                print(f"[n={args.n}] step {step} val_loss {loss:.4f} best {best:.4f}", flush=True)
            if step >= max_steps or bad >= 10:
                break
    net.load_state_dict(best_state)
    res: dict = {"n_train": args.n, "steps": step, "val_loss": best, "seeds": {}}
    for seed in SEEDS:
        d = dataset(seed)
        y = d["y"].astype(np.int64)
        sp = split_indices(len(y))
        p = torch.sigmoid(predict(to_input(d["x"][sp["test"]].astype(np.float32), mu, sd).to(dev)))
        p = p.cpu().numpy()
        np.save(DIR / f"specialist_{args.n}_seed{seed}_test.npy", p)
        row = {"acc": float(np.mean((p > 0.5) == y[sp["test"]]))}
        if "kinds" in d:
            k = d["kinds"][sp["test"]]
            row["acc_by_kind"] = {
                str(int(c)): float(np.mean((p[k == c] > 0.5) == y[sp["test"]][k == c]))
                for c in np.unique(k)
            }
        res["seeds"][str(seed)] = row
        print(f"[n={args.n}] seed {seed} test acc {row['acc']:.4f}", flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_path}", flush=True)


def cmd_summarize(_: argparse.Namespace) -> None:
    rep = json.loads(Path("outputs/replication/results.json").read_text())["runs"]
    out: dict = {"sizes": {}, "seeds": {}}
    for n in SIZES:
        f = DIR / f"specialist_{n}.json"
        if f.exists():
            out["sizes"][str(n)] = json.loads(f.read_text())
    n_max = str(max(int(k) for k in out["sizes"]))
    for seed in SEEDS:
        y = dataset(seed)["y"].astype(np.int64)
        sp = split_indices(len(y))
        y_te = y[sp["test"]]
        _, _, p_c = fit_predict_tuned(controls(seed), y, sp, len(sp["train"]))
        p_s = np.load(DIR / f"specialist_{n_max}_seed{seed}_test.npy")
        cmp = compare(p_s, p_c, y_te)
        headroom = cmp["acc_a"] - cmp["acc_b"]
        gains = {
            r["model"]: float(
                np.mean(
                    [
                        a["h4_attention_given_controls"]["tost_0.02"]["diff"]
                        for a in r["attention"].values()
                    ]
                )
            )
            for r in rep
            if r["seed"] == seed
        }
        out["seeds"][str(seed)] = {
            "controls_acc": cmp["acc_b"],
            "specialist_acc": cmp["acc_a"],
            "specialist_vs_controls": cmp,
            "headroom_lower_bound": headroom,
            "attention_gain": gains,
            "share_upper_bound": {m: g / headroom for m, g in gains.items()}
            if headroom > 0
            else None,
        }
        print(
            f"[seed {seed}] specialist {cmp['acc_a']:.4f} controls {cmp['acc_b']:.4f} "
            f"headroom >= {headroom:.4f}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data").add_argument("--jobs", type=int, default=20)
    sub.add_parser("train").add_argument("--n", type=int, required=True, choices=SIZES)
    sub.add_parser("summarize")
    args = ap.parse_args(argv)
    {"data": cmd_data, "train": cmd_train, "summarize": cmd_summarize}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
