"""Two controls for the large-sample anomaly findings (realistic task, data seeds 1001-1003).

H2, token-level control: the attention view adds accuracy to all controls for every pretrained
model (scripts/run_replication.py). The attention probe reads tokens, while every control is pooled
over the whole window, so the gain could come from token-level read-out rather than from
pretraining. ``rocket-tokens`` builds a token map from the same 1,024 ROCKET-style kernels the
control uses: per kernel and per patch of 32 steps, the proportion of positive values and the
maximum (16 tokens x 2,048 dimensions; the first 192 kernels give a 384-dimensional variant, the
width of Moirai 2.0). ``token-fit`` runs the replication's analysis unchanged on these tokens:
cross-fitted attention logit stacked with all controls, three probe seeds, paired McNemar and TOST.

H7, dimension-matched controls: all controls together have about 20,000 dimensions, a mean-pooled
model 384-1,280. ``dim-match`` reduces both to the same k (StandardScaler + PCA fitted on the
training split; k in {64, 256, model width}) and compares them on the test windows.

Usage:
    TSFMI_RESOLUTION_STORE=... PYTHONPATH=. python scripts/run_revision_controls.py \
        rocket-tokens --seed 1001
    ... run_revision_controls.py token-fit --seed 1001 --tokens rocket1024
    ... run_revision_controls.py dim-match --seed 1001
    ... run_revision_controls.py summarize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.run_replication import (
    ATTN_SEEDS,
    MODELS,
    SEEDS,
    controls,
    crossfit_logits_full,
    dataset,
    seed_dir,
    tokens_path,
)
from scripts.run_resolution_study import compare, fit_predict_tuned, split_indices

PATCH = 32
TOKEN_SETS = {"rocket1024": 1024, "rocket192": 192}
DIMS = (64, 256)
OUT = Path("outputs/revision_controls")


def rocket_token_path(seed: int) -> Path:
    return seed_dir(seed) / "tokens" / "rocket1024.npy"


def load_tokens(seed: int, name: str) -> NDArray:
    tok = np.load(rocket_token_path(seed), mmap_mode="r")
    return tok if name == "rocket1024" else tok[:, :, : 2 * TOKEN_SETS[name]]


def cmd_rocket_tokens(args: argparse.Namespace) -> None:
    """Per-patch PPV and max of the control's ROCKET kernels, computed on the GPU."""
    import torch

    from scripts.run_rocket_baseline import N_KERNELS, random_kernels

    path = rocket_token_path(args.seed)
    if path.exists():
        print(f"SKIP {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernels = random_kernels(seed=42, num_kernels=N_KERNELS)  # the control's kernels
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (w, dil, _) in enumerate(kernels):
        groups.setdefault((len(w), dil), []).append(i)
    x = dataset(args.seed)["x"].astype(np.float32)
    n, length = x.shape
    n_tok = length // PATCH
    out = np.lib.format.open_memmap(path, "w+", np.float16, (n, n_tok, 2 * N_KERNELS))
    with torch.inference_mode():
        for s in range(0, n, 1000):
            xb = torch.from_numpy(x[s : s + 1000]).to(dev)[:, None, :]
            ppv = torch.empty(len(xb), N_KERNELS, n_tok, device=dev)
            mx = torch.empty_like(ppv)
            for (klen, dil), idx in groups.items():
                w = torch.tensor(
                    np.stack([kernels[i][0] for i in idx]), dtype=torch.float32, device=dev
                )[:, None, :]
                pad = (klen - 1) * dil // 2  # as in apply_kernel: output length = input length
                conv = torch.nn.functional.conv1d(xb, w, padding=pad, dilation=dil)
                conv = conv[..., :length].reshape(len(xb), len(idx), n_tok, PATCH)
                ppv[:, idx] = (conv > 0).float().mean(-1)
                mx[:, idx] = conv.amax(-1)
            # Interleave as in rocket_features: [ppv_0, max_0, ppv_1, max_1, ...].
            tok = torch.stack([ppv, mx], dim=2).reshape(len(xb), 2 * N_KERNELS, n_tok)
            out[s : s + len(xb)] = tok.transpose(1, 2).cpu().numpy().astype(np.float16)
    out.flush()
    print(f"wrote {path} {out.shape}", flush=True)


def cmd_token_fit(args: argparse.Namespace) -> None:
    out_path = seed_dir(args.seed) / f"fit_tokens_{args.tokens}.json"
    if out_path.exists():
        print(f"SKIP {out_path}")
        return
    y = dataset(args.seed)["y"].astype(np.int64)
    sp = split_indices(len(y))
    y_te, n_tr = y[sp["test"]], len(sp["train"])
    C = controls(args.seed)
    tok = np.ascontiguousarray(load_tokens(args.seed, args.tokens))
    _, _, p_c = fit_predict_tuned(C, y, sp, n_tr)
    res: dict = {
        "seed": args.seed,
        "tokens": args.tokens,
        "dim": int(tok.shape[-1]),
        "controls_acc": float(np.mean((p_c > 0.5) == y_te)),
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
            f"[{args.seed} {args.tokens} attn{a}] att={acc:.4f}"
            f" gain={g['tost_0.02']['diff']:+.4f} p={g['mcnemar']['p_value']:.3g}",
            flush=True,
        )
    out_path.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_path}", flush=True)


def pca_view(X: NDArray, sp: dict[str, NDArray], k: int) -> NDArray[np.float32]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    tr = sp["train"]
    scaler = StandardScaler().fit(X[tr])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(scaler.transform(X[tr]))
    out = np.empty((len(X), k), dtype=np.float32)
    for s in range(0, len(X), 10_000):
        out[s : s + 10_000] = pca.transform(scaler.transform(X[s : s + 10_000]))
    return out


def cmd_dim_match(args: argparse.Namespace) -> None:
    out_path = seed_dir(args.seed) / "dim_match.json"
    if out_path.exists():
        print(f"SKIP {out_path}")
        return
    y = dataset(args.seed)["y"].astype(np.int64)
    sp = split_indices(len(y))
    y_te, n_tr = y[sp["test"]], len(sp["train"])
    C = controls(args.seed).astype(np.float32)
    means = {}
    for m in MODELS:
        tok = np.load(tokens_path(args.seed, m), mmap_mode="r")
        means[m] = np.concatenate(
            [tok[s : s + 5000].astype(np.float32).mean(axis=1) for s in range(0, len(y), 5000)]
        )
    ks = sorted({*DIMS, *(z.shape[1] for z in means.values())})
    ctrl_p: dict[int, NDArray] = {}
    res: dict = {"seed": args.seed, "controls_dim": int(C.shape[1]), "models": {}}
    for k in ks:
        _, _, ctrl_p[k] = fit_predict_tuned(pca_view(C, sp, k), y, sp, n_tr)
        print(
            f"[{args.seed}] controls k={k} acc={np.mean((ctrl_p[k] > 0.5) == y_te):.4f}", flush=True
        )
    res["controls_acc"] = {str(k): float(np.mean((p > 0.5) == y_te)) for k, p in ctrl_p.items()}
    for m, Z in means.items():
        rows = {}
        for k in [k for k in ks if k <= Z.shape[1]]:
            Zk = Z if k == Z.shape[1] else pca_view(Z, sp, k)
            _, _, p_z = fit_predict_tuned(Zk, y, sp, n_tr)
            rows[str(k)] = compare(p_z, ctrl_p[k], y_te)
            print(
                f"[{args.seed} {m}] k={k} model={rows[str(k)]['acc_a']:.4f}"
                f" controls={rows[str(k)]['acc_b']:.4f}",
                flush=True,
            )
        res["models"][m] = {"width": int(Z.shape[1]), "by_k": rows}
    out_path.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_path}", flush=True)


def cmd_summarize(_: argparse.Namespace) -> None:
    summary: dict = {"tokens": {}, "dim_match": {}}
    for name in TOKEN_SETS:
        runs = [
            json.loads(f.read_text())
            for s in SEEDS
            if (f := seed_dir(s) / f"fit_tokens_{name}.json").exists()
        ]
        h4 = [a["h4_attention_given_controls"] for r in runs for a in r["attention"].values()]
        if not h4:
            continue
        gains = [c["tost_0.02"]["diff"] for c in h4]
        summary["tokens"][name] = {
            "dim": runs[0]["dim"],
            "n_runs": len(h4),
            "attention_acc": [a["attention_acc"] for r in runs for a in r["attention"].values()],
            "h4_gain": gains,
            "h4_gain_mean": float(np.mean(gains)),
            "h4_n_positive_significant": sum(
                c["mcnemar"]["p_value"] < 0.05 and c["tost_0.02"]["diff"] > 0 for c in h4
            ),
            "h4_n_equiv01": sum(c["tost_0.01"]["equivalent"] for c in h4),
        }
    for s in SEEDS:
        f = seed_dir(s) / "dim_match.json"
        if f.exists():
            summary["dim_match"][str(s)] = json.loads(f.read_text())
    leads = {}
    for r in summary["dim_match"].values():
        for m, v in r["models"].items():
            for k, c in v["by_k"].items():
                leads.setdefault(m, {}).setdefault(k, []).append(
                    {"lead": -c["tost_0.02"]["diff"], "p": c["mcnemar"]["p_value"]}
                )
    summary["dim_match_leads"] = leads
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"tokens": summary["tokens"], "leads": leads}, indent=1)[:4000])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rocket-tokens", "dim-match"):
        sub.add_parser(name).add_argument("--seed", type=int, required=True, choices=SEEDS)
    t = sub.add_parser("token-fit")
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--tokens", required=True, choices=tuple(TOKEN_SETS))
    sub.add_parser("summarize")
    args = ap.parse_args(argv)
    {
        "rocket-tokens": cmd_rocket_tokens,
        "token-fit": cmd_token_fit,
        "dim-match": cmd_dim_match,
        "summarize": cmd_summarize,
    }[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
