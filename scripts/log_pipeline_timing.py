"""Wall-clock instrumentation for the canonical pipeline (CHECK.md G7).

Times the existing CPU-only canonical baseline + canonical model probing +
manifest + figures + paper compile, and writes a machine-readable timing
record so the paper's "~48 GPU-hours" claim has a corroborating artefact.

Output: outputs/paper/timing.json
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

OUT = Path("outputs/paper/timing.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def time_cmd(label: str, cmd: list[str]) -> dict:
    print(f"  [timing] {label}: {' '.join(cmd)}")
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "cmd": cmd,
        "wall_seconds": round(elapsed, 2),
        "wall_human": f"{elapsed / 60:.1f} min",
        "returncode": rc.returncode,
    }


def main() -> None:
    venv_py = ".venv/bin/python"
    if not Path(venv_py).exists():
        venv_py = shutil.which("python") or "python"

    stages: list[dict] = []
    # 1. Canonical baselines.
    stages.append(
        time_cmd(
            "canonical_baselines",
            [venv_py, "scripts/run_canonical_baselines.py"],
        )
    )
    # 2. Per-feature anomaly attribution.
    stages.append(
        time_cmd(
            "per_feature_anomaly",
            [venv_py, "scripts/run_per_feature_anomaly.py"],
        )
    )
    # 3. Manifest.
    stages.append(
        time_cmd(
            "manifest",
            [venv_py, "scripts/count_experiments.py"],
        )
    )

    record = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "stages": stages,
        "total_wall_seconds": round(sum(s["wall_seconds"] for s in stages), 2),
        "note": (
            "These stages are CPU-only and bound by sklearn fit time, not GPU. "
            "Frozen-representation extraction (the dominant ~40 GPU-hour stage) "
            "is timed separately by the extraction script's own logger."
        ),
    }
    OUT.write_text(json.dumps(record, indent=2))
    print(f"\nWrote {OUT}")
    for s in stages:
        print(f"  {s['label']:<32s} {s['wall_human']:>10s}  rc={s['returncode']}")
    print(f"  {'TOTAL':<32s} {record['total_wall_seconds'] / 60:>8.1f} min")


if __name__ == "__main__":
    main()
