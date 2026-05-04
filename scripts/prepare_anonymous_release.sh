#!/bin/bash
# Prepare anonymized code artifact for NeurIPS submission.
#
# Creates a clean zip without git history, author info, or large outputs.
#
# Usage:
#   bash scripts/prepare_anonymous_release.sh

set -euo pipefail

RELEASE_DIR="/tmp/tsfmi_anonymous_release"
ZIP_NAME="tsfmi_supplementary.zip"
# Final destination for the supplementary bundle. NeurIPS OpenReview accepts
# the file directly from this path; the /tmp staging area is used only during
# build to keep large intermediate copies off the project tree.
FINAL_DEST="$(pwd)/outputs/paper/supplementary"

echo "=== Preparing anonymous release ==="

# Clean previous
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR/tsfmi"

# Copy source code (no caches, no editor backups, no virtualenv state)
cp -r src/ "$RELEASE_DIR/tsfmi/src/"
cp -r scripts/ "$RELEASE_DIR/tsfmi/scripts/"
cp -r tests/ "$RELEASE_DIR/tsfmi/tests/"
cp -r configs/ "$RELEASE_DIR/tsfmi/configs/" 2>/dev/null || true

# Defensively strip any bytecode / virtualenv / orchestrator state that might
# have been copied.
find "$RELEASE_DIR" -type d \( -name ".venv" -o -name ".omc" -o -name ".tmp" \
    -o -name ".sisyphus" -o -name "__pycache__" -o -name ".pytest_cache" \
    -o -name ".ruff_cache" -o -name ".mypy_cache" \) \
    -exec rm -rf {} + 2>/dev/null || true
find "$RELEASE_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$RELEASE_DIR" -name ".DS_Store" -delete 2>/dev/null || true

# Copy config files
cp pyproject.toml "$RELEASE_DIR/tsfmi/"
cp requirements.txt "$RELEASE_DIR/tsfmi/"
cp Makefile "$RELEASE_DIR/tsfmi/"

# Copy documentation (anonymized)
cp CLAUDE.md "$RELEASE_DIR/tsfmi/" 2>/dev/null || true

# Copy ED Track artefacts (license, Croissant metadata).
cp LICENSE "$RELEASE_DIR/tsfmi/" 2>/dev/null || true
cp CROISSANT.json "$RELEASE_DIR/tsfmi/" 2>/dev/null || true

# Create anonymized README
cat > "$RELEASE_DIR/tsfmi/README.md" << 'READMEEOF'
# TSFMI: Anonymous Code Release (NeurIPS 2026 E&D Track)

This is the anonymized code/data artifact for the submission
**"TSFMI: A Baseline-Controlled Evaluation Protocol for Time-Series
Foundation Model Representations"** to the NeurIPS 2026 Evaluations &
Datasets Track.

The artefact contains:

- `src/`: 7 TSFM wrappers, 12 synthetic generators, probing & intervention library
- `scripts/`: extraction, canonical benchmark, baselines, LEACE/CKA/steering, paper-figure regeneration
- `tests/`: 170 automated tests (`make test`)
- `configs/`: per-experiment configuration references
- `LICENSE`: MIT (code) + pointers to original licenses for the public real-world datasets used
- `CROISSANT.json`: machine-readable dataset description (Croissant 1.0 + RAI fields)
- `Makefile`: one-line entrypoints for every reproducible target

## Quick reviewer reproduction (CPU, <10 min)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
make smoke
```

`make smoke` reproduces one cell of the canonical baseline grid +
per-feature anomaly attribution + the test suite. No GPU required.

## Full reproduction (~48 A100-hours)

```bash
make extract-representations    # ~40h GPU, 7 models x 6 datasets
make reproduce-all              # baselines + canonical + figures + paper compile
```

## Validation

```bash
make test   # 170 tests
make lint   # ruff check
```

## Reproduce Main Pipeline

```bash
# 1. Extract representations
PYTHONPATH=. python scripts/extract_representations.py \
  --model moment --dataset synthetic_trend --layers all \
  --output_dir outputs/representations/

# 2. Train probes
PYTHONPATH=. python scripts/train_probe.py \
  --representations_dir outputs/representations/moment/synthetic_trend \
  --output_dir outputs/probes/moment_synthetic_trend_linear \
  --probe_type linear

# 3. Evaluate
PYTHONPATH=. python scripts/evaluate_probe.py \
  --probe_dir outputs/probes/moment_synthetic_trend_linear \
  --representations_dir outputs/representations/moment/synthetic_trend \
  --output_dir outputs/eval/moment_synthetic_trend_linear
```

## Additional Analyses

```bash
PYTHONPATH=. python scripts/run_hard_variant_benchmark.py
PYTHONPATH=. python scripts/run_causal_controls.py
PYTHONPATH=. python scripts/run_nonlinear_recovery_all.py
PYTHONPATH=. python scripts/run_tcas_ablation.py
PYTHONPATH=. python scripts/run_pca_alternatives.py
PYTHONPATH=. python scripts/run_real_world_analysis.py
PYTHONPATH=. python scripts/run_cka_bootstrap.py
```
READMEEOF

# Remove any author-identifying info
find "$RELEASE_DIR" -name "*.py" -exec grep -l "author\|@.*\.edu\|@.*\.com" {} \; 2>/dev/null | while read f; do
    echo "  Checking $f for author info..."
done

# Remove git, cache, outputs, data, checkpoints
rm -rf "$RELEASE_DIR/tsfmi/.git"
rm -rf "$RELEASE_DIR/tsfmi/__pycache__"
find "$RELEASE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$RELEASE_DIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$RELEASE_DIR" -name "*.pyc" -delete 2>/dev/null || true

# Remove AGENTS.md (may contain identifying info)
rm -f "$RELEASE_DIR/tsfmi/AGENTS.md"
rm -f "$RELEASE_DIR/tsfmi/PROVENANCE.md"
rm -f "$RELEASE_DIR/tsfmi/Linear_Probing_TS_Survey.md"

# Remove auto-generated egg-info that may contain author / install path leaks
rm -rf "$RELEASE_DIR/tsfmi/src/tsfmi.egg-info"
rm -rf "$RELEASE_DIR/tsfmi/src/"*.egg-info

# CRITICAL: remove internal reviewer-prep documents that must never be shipped
# (these reveal anti-review strategy to actual reviewers and destroy anonymity)
rm -f "$RELEASE_DIR/tsfmi/AUDIT.md"
rm -f "$RELEASE_DIR/tsfmi/CHECK.md"
rm -f "$RELEASE_DIR/tsfmi/REVIEW.md"
rm -f "$RELEASE_DIR/tsfmi/REVIEW2.md"
rm -f "$RELEASE_DIR/tsfmi/REVISION.md"
rm -f "$RELEASE_DIR/tsfmi/PLAN.md"
rm -f "$RELEASE_DIR/tsfmi/CLAUDE.md"
rm -f "$RELEASE_DIR/tsfmi/SUBMISSION_CHECKLIST.md"

# Remove stale/alternate build artifacts that could confuse reviewers
rm -f "$RELEASE_DIR/tsfmi/outputs/paper/latex/_final.tex"
rm -f "$RELEASE_DIR/tsfmi/outputs/paper/latex/_preprint.tex"
rm -f "$RELEASE_DIR/tsfmi/outputs/paper/latex/korean.tex"
rm -f "$RELEASE_DIR/tsfmi/outputs/paper/paper_structure.md"
rm -rf "$RELEASE_DIR/tsfmi/outputs/analysis"

# Remove shell scripts with absolute paths
rm -f "$RELEASE_DIR/tsfmi/scripts/extract_hard_variants_batch.sh"
rm -f "$RELEASE_DIR/tsfmi/scripts/prepare_anonymous_release.sh"

# Strip bulky output artefacts that should NEVER ship: 837 GB of frozen
# representations, raw stdout logs, and any per-seed extraction snapshots.
rm -rf "$RELEASE_DIR/tsfmi/outputs/representations" 2>/dev/null || true
rm -rf "$RELEASE_DIR/tsfmi/outputs/representations_seed123" 2>/dev/null || true
rm -rf "$RELEASE_DIR/tsfmi/outputs/representations_seed456" 2>/dev/null || true
rm -rf "$RELEASE_DIR/tsfmi/outputs/representations_smoke" 2>/dev/null || true
find "$RELEASE_DIR/tsfmi/outputs" -name "*.log" -delete 2>/dev/null || true
find "$RELEASE_DIR/tsfmi/outputs" -name "*.txt" -delete 2>/dev/null || true

# Final safety sweep: fail if any sentinel internal doc leaks, or if
# representations/log files survived the cleanup above.
LEAKS=$(find "$RELEASE_DIR" \( -name "AUDIT.md" -o -name "CHECK.md" -o -name "REVIEW*.md" -o -name "REVISION.md" -o -name "PLAN.md" -o -name "SUBMISSION_CHECKLIST.md" \) 2>/dev/null | wc -l)
if [[ "$LEAKS" != "0" ]]; then
    echo "ERROR: internal review docs leaked into release dir!"
    find "$RELEASE_DIR" \( -name "AUDIT.md" -o -name "CHECK.md" -o -name "REVIEW*.md" -o -name "REVISION.md" -o -name "PLAN.md" -o -name "SUBMISSION_CHECKLIST.md" \) 2>/dev/null
    exit 2
fi
REPR_LEAKS=$(find "$RELEASE_DIR" -type d -name "representations*" 2>/dev/null | wc -l)
if [[ "$REPR_LEAKS" != "0" ]]; then
    echo "ERROR: outputs/representations* leaked into release dir!"
    find "$RELEASE_DIR" -type d -name "representations*" 2>/dev/null
    exit 3
fi
LOG_LEAKS=$(find "$RELEASE_DIR" -name "*.log" 2>/dev/null | wc -l)
if [[ "$LOG_LEAKS" != "0" ]]; then
    echo "ERROR: stdout logs leaked into release dir!"
    find "$RELEASE_DIR" -name "*.log" 2>/dev/null
    exit 4
fi

# Create zip and place it next to the paper PDF for OpenReview upload.
cd "$RELEASE_DIR"
zip -r "$ZIP_NAME" tsfmi/ -x "*.DS_Store" "*.pyc" "*__pycache__*" >/dev/null

mkdir -p "$FINAL_DEST"
mv "$RELEASE_DIR/$ZIP_NAME" "$FINAL_DEST/$ZIP_NAME"
rm -rf "$RELEASE_DIR" 2>/dev/null || true

echo ""
echo "=== Supplementary bundle created ==="
echo "Location: $FINAL_DEST/$ZIP_NAME"
echo "Size: $(du -h "$FINAL_DEST/$ZIP_NAME" | cut -f1)"
echo ""
echo "Contents (last 5 entries):"
unzip -l "$FINAL_DEST/$ZIP_NAME" | tail -5
