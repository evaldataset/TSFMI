"""Rewrite machine-specific absolute paths in result files to repository-relative defaults.

Result JSONs record where representations were read from and the command that produced them. On the
machine that ran the experiments these are absolute paths (a large-disk store, a scratch directory),
which do not exist elsewhere and can identify the machine's user. This script maps them onto the
defaults a reproducer gets from the Makefile (``outputs/representations_v2`` etc.) and reports any
absolute path it does not know how to map. It is idempotent.

Usage:
    python scripts/sanitize_result_paths.py --store /big/disk/tsfmi_store \
        --scratch /tmp/some/scratchpad --repo "$PWD" [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SUFFIXES = (".json", ".md", ".txt", ".csv", ".tex")


def rules(store: str, scratch: str, repo: str) -> list[tuple[re.Pattern[str], str]]:
    s, t, r = (re.escape(x.rstrip("/")) for x in (store, scratch, repo))
    return [
        (re.compile(rf"{s}/representations_v2\b"), "outputs/representations_v2"),
        (
            re.compile(rf"{s}/representations_dataseed(\d+)\b"),
            r"outputs/representations_v2_dataseed\1",
        ),
        (re.compile(rf"{s}/(representations_[A-Za-z0-9_]+)\b"), r"outputs/\1"),
        (re.compile(rf"{t}/representations-in-tsfms\b"), "external/representations-in-tsfms"),
        (re.compile(rf"{t}/wil_work\b"), "external/wilinski_work"),
        (re.compile(rf"{t}\b"), "scratch"),
        (re.compile(rf"{r}/"), ""),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True, help="absolute store root to replace")
    ap.add_argument("--scratch", required=True, help="absolute scratch directory to replace")
    ap.add_argument("--repo", required=True, help="absolute path of this repository")
    ap.add_argument("--root", type=Path, default=Path("outputs"))
    ap.add_argument("--check", action="store_true", help="only report, do not rewrite")
    ap.add_argument("--forbid", nargs="*", default=[], help="strings that must not remain")
    args = ap.parse_args()
    rs = rules(args.store, args.scratch, args.repo)
    changed, leftovers = 0, []
    for f in sorted(p for p in args.root.rglob("*") if p.is_file() and p.suffix in SUFFIXES):
        text = f.read_text(errors="ignore")
        new = text
        for pat, rep in rs:
            new = pat.sub(rep, new)
        if new != text:
            changed += 1
            if not args.check:
                f.write_text(new)
        for bad in args.forbid:
            if bad in new:
                leftovers.append(f"{f}: {bad}")
    print(f"{'would rewrite' if args.check else 'rewrote'} {changed} files")
    for line in leftovers:
        print(f"REMAINING {line}")
    return 1 if leftovers else 0


if __name__ == "__main__":
    sys.exit(main())
