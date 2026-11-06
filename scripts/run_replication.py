"""Replication of the large-sample anomaly findings over fresh data seeds and probe initialisations.

The resolution study (scripts/run_resolution_study.py) fixes one dataset (seed 1001) and one
training run per probe. This script repeats its three central comparisons on the realistic
generator for data seeds 1001, 1002, 1003 (50,000 windows each, same 60/20/20 split rule) and, for
the attention view, three probe initialisations per data seed:

- H1: best mean-pooled model vs. the union of all feature controls (hand-crafted, tsfresh efficient,
  ROCKET-style), paired on the test windows;
- H3: what the mean-pooled representation adds to all controls (conditional probe, C tuned on
  validation);
- H4: what an attention-pooled view adds to all controls. The attention view is stacked as a
  one-dimensional cross-fitted logit: training rows get out-of-fold logits from attention nets
  trained on the other four fifths, validation and test rows the mean logit of the five fold nets.
  A scalar logit has the same meaning in every fold, unlike pooled vectors whose query order is not
  identified across separately trained nets.

Each model is read at the layer selected on seed 1001 (realistic task, 30,000 training windows), so
seeds 1002 and 1003 replicate a fixed analysis. Models: TimesFM 2.0, PatchTST-FM, Moirai 2.0,
Chronos-Bolt (the four strongest on seed 1001).

Usage:
    TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_replication.py data
    ... run_replication.py controls --seed 1002
    ... run_replication.py extract --seed 1002 --model timesfm
    ... run_replication.py fit --seed 1002 --model timesfm
    ... run_replication.py summarize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.run_resolution_study import (
    STORE,
    attention_probe,
    compare,
    fit_predict_tuned,
    load_data,
    rocket_parallel,
    selected_layer,
    split_indices,
    tsfresh_features,
)

SEEDS = (1001, 1002, 1003)
MODELS = ("timesfm", "patchtst_fm", "moirai", "chronos")
ATTN_SEEDS = (0, 1, 2)
FOLDS = 5
N = 50_000
OUT = Path("outputs/replication")


def seed_dir(seed: int) -> Path:
    return STORE / "replication" / f"seed{seed}"


def dataset(seed: int) -> dict[str, NDArray]:
    if seed == 1001:
        return load_data("realistic")
    with np.load(seed_dir(seed) / "data.npz") as z:
        return {k: z[k] for k in z.files}


def tokens_path(seed: int, model: str) -> Path:
    if seed == 1001:
        return (
            STORE / "tokens" / f"{model}-pretrained" / "realistic" / f"{selected_layer(model)}.npy"
        )
    return seed_dir(seed) / "tokens" / f"{model}.npy"


def controls(seed: int) -> NDArray:
    if seed == 1001:
        base = STORE / "controls" / "realistic"
        names = ("hand_crafted", "rocket", "tsfresh_efficient")
        return np.concatenate([np.load(base / f"{n}.npy") for n in names], axis=1)
    return np.load(seed_dir(seed) / "all_controls.npy")


def cmd_data(_: argparse.Namespace) -> None:
    from scripts.run_realistic_anomaly import realistic_anomaly_dataset

    for seed in SEEDS[1:]:
        path = seed_dir(seed) / "data.npz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        x, y, k = realistic_anomaly_dataset(N, 512, seed, return_kinds=True)
        np.savez(path, x=x, y=y, kinds=k)
        print(f"wrote {path}", flush=True)


def cmd_controls(args: argparse.Namespace) -> None:
    from scripts.run_canonical_baselines import hand_crafted_features

    path = seed_dir(args.seed) / "all_controls.npy"
    if path.exists():
        return
    x = dataset(args.seed)["x"].astype(np.float64)
    parts = [
        hand_crafted_features(x).astype(np.float32),
        rocket_parallel(x, args.jobs),
        tsfresh_features(x, args.jobs),
    ]
    np.save(path, np.concatenate(parts, axis=1).astype(np.float32))
    print(f"wrote {path}", flush=True)


def cmd_extract(args: argparse.Namespace) -> None:
    import torch

    from scripts.extract_representations import load_model_wrapper
    from src.extractors.hook_manager import HookManager
    from src.utils.seed import seed_everything

    path = tokens_path(args.seed, args.model)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = selected_layer(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    wrapper = load_model_wrapper(args.model, None, device)
    name = next(n for n in wrapper.get_layer_names() if n.replace(".", "_") == layer)
    x = dataset(args.seed)["x"].astype(np.float32)
    out = None
    with HookManager(wrapper.model, [name]) as hm, torch.inference_mode():
        for s in range(0, len(x), 128):
            wrapper.forward(torch.from_numpy(x[s : s + 128]).to(device))
            a = hm.get_activations()[name].float()
            a = a.reshape(a.shape[0], -1, a.shape[-1])
            if out is None:
                out = np.lib.format.open_memmap(path, "w+", np.float16, (len(x), *a.shape[1:]))
            out[s : s + len(a)] = a.cpu().numpy().astype(np.float16)
            hm.clear()
    out.flush()
    print(f"wrote {path} {out.shape}", flush=True)


def crossfit_logits_full(tok: NDArray, y: NDArray, sp: dict[str, NDArray], seed: int) -> NDArray:
    """Cross-fitted attention logits for every window (see module docstring)."""
    tr = sp["train"]
    logit = np.zeros(len(y), dtype=np.float64)
    parts = np.array_split(np.random.default_rng(seed).permutation(len(tr)), FOLDS)
    for k, held in enumerate(parts):
        fit = np.setdiff1d(np.arange(len(tr)), held)
        score = np.concatenate([tr[held], sp["val"], sp["test"]])
        sub = {"train": tr[fit], "val": sp["val"], "test": score}
        _, p = attention_probe(tok, y, sub, seed=100 * seed + k)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p))
        logit[tr[held]] = lg[: len(held)]
        logit[sp["val"]] += lg[len(held) : len(held) + len(sp["val"])] / FOLDS
        logit[sp["test"]] += lg[len(held) + len(sp["val"]) :] / FOLDS
    return logit


def cmd_fit(args: argparse.Namespace) -> None:
    out_path = seed_dir(args.seed) / f"fit_{args.model}.json"
    if out_path.exists():
        print(f"SKIP {out_path}")
        return
    d = dataset(args.seed)
    y = d["y"].astype(np.int64)
    sp = split_indices(len(y))
    y_te = y[sp["test"]]
    n_tr = len(sp["train"])
    C = controls(args.seed)
    tok = np.load(tokens_path(args.seed, args.model), mmap_mode="r")
    mean = np.concatenate(
        [tok[s : s + 5000].astype(np.float32).mean(axis=1) for s in range(0, len(y), 5000)]
    )
    _, _, p_c = fit_predict_tuned(C, y, sp, n_tr)
    _, _, p_z = fit_predict_tuned(mean, y, sp, n_tr)
    _, _, p_cz = fit_predict_tuned(np.concatenate([C, mean], axis=1), y, sp, n_tr)
    res: dict = {
        "seed": args.seed,
        "model": args.model,
        "layer": selected_layer(args.model),
        "controls_acc": float(np.mean((p_c > 0.5) == y_te)),
        "mean_acc": float(np.mean((p_z > 0.5) == y_te)),
        "h1_mean_vs_controls": compare(p_z, p_c, y_te),
        "h3_mean_given_controls": compare(p_cz, p_c, y_te),
        "attention": {},
    }
    for a in ATTN_SEEDS:
        lg = crossfit_logits_full(tok, y, sp, seed=a)
        p_att = 1.0 / (1.0 + np.exp(-lg[sp["test"]]))
        _, _, p_joint = fit_predict_tuned(
            np.concatenate([C, lg[:, None].astype(np.float32)], axis=1), y, sp, n_tr
        )
        res["attention"][str(a)] = {
            "attention_acc": float(np.mean((p_att > 0.5) == y_te)),
            "attention_vs_controls": compare(p_att, p_c, y_te),
            "h4_attention_given_controls": compare(p_joint, p_c, y_te),
        }
        g = res["attention"][str(a)]["h4_attention_given_controls"]
        acc = res["attention"][str(a)]["attention_acc"]
        print(
            f"[{args.seed} {args.model} attn{a}] att={acc:.4f}"
            f" gain={g['tost_0.02']['diff']:+.4f} p={g['mcnemar']['p_value']:.3g}",
            flush=True,
        )
    out_path.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_path}", flush=True)


def cmd_summarize(_: argparse.Namespace) -> None:
    rows = []
    for seed in SEEDS:
        for m in MODELS:
            f = seed_dir(seed) / f"fit_{m}.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
    summary: dict = {"seeds": list(SEEDS), "models": list(MODELS), "attn_seeds": list(ATTN_SEEDS)}
    for m in MODELS:
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            continue
        h1 = [r["h1_mean_vs_controls"] for r in rs]
        h3 = [r["h3_mean_given_controls"] for r in rs]
        h4 = [a["h4_attention_given_controls"] for r in rs for a in r["attention"].values()]
        att = [a["attention_acc"] for r in rs for a in r["attention"].values()]
        summary[m] = {
            "n_data_seeds": len(rs),
            "n_attention_runs": len(h4),
            "controls_acc": [r["controls_acc"] for r in rs],
            "mean_acc": [r["mean_acc"] for r in rs],
            "h1_lead": [-c["tost_0.02"]["diff"] for c in h1],
            "h1_n_control_significant": sum(
                c["mcnemar"]["p_value"] < 0.05 and c["tost_0.02"]["diff"] < 0 for c in h1
            ),
            "h3_gain": [c["tost_0.02"]["diff"] for c in h3],
            "h3_n_equiv01": sum(c["tost_0.01"]["equivalent"] for c in h3),
            "h3_bits": [c["logloss_bits"]["bits"] for c in h3],
            "attention_acc": att,
            "h4_gain": [c["tost_0.02"]["diff"] for c in h4],
            "h4_gain_mean": float(np.mean([c["tost_0.02"]["diff"] for c in h4])),
            "h4_gain_sd": float(np.std([c["tost_0.02"]["diff"] for c in h4], ddof=1))
            if len(h4) > 1
            else 0.0,
            "h4_n_positive_significant": sum(
                c["mcnemar"]["p_value"] < 0.05 and c["tost_0.02"]["diff"] > 0 for c in h4
            ),
            "h4_n_equiv01": sum(c["tost_0.01"]["equivalent"] for c in h4),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=1))
    print(json.dumps(summary, indent=1)[:3000])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    c = sub.add_parser("controls")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--jobs", type=int, default=32)
    e = sub.add_parser("extract")
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--model", required=True, choices=MODELS)
    f = sub.add_parser("fit")
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--model", required=True, choices=MODELS)
    sub.add_parser("summarize")
    args = ap.parse_args(argv)
    {
        "data": cmd_data,
        "controls": cmd_controls,
        "extract": cmd_extract,
        "fit": cmd_fit,
        "summarize": cmd_summarize,
    }[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
