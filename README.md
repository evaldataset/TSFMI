# TSFMI

TSFMI is a research codebase for probing and interpreting temporal concept
representations in time series foundation models.

Current repository status:
- 170 tests passing
- `ruff check src scripts tests` clean
- NeurIPS paper compiles with `tectonic`
- Appendix includes a numeric 7x7 cross-model CKA table and Timer main-text summary

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

## Common Commands

```bash
make test
make lint
make format
make paper
```

`make paper` uses `latexmk` from your `PATH` (override with `make paper LATEXMK=/path/to/latexmk` if needed).

## Reproduce Main Pipeline

1. Extract representations

```bash
PYTHONPATH=. .venv/bin/python scripts/extract_representations.py \
  --model moment \
  --dataset synthetic_trend \
  --layers all \
  --output_dir outputs/representations/
```

2. Train probes

```bash
PYTHONPATH=. .venv/bin/python scripts/train_probe.py \
  --representations_dir outputs/representations/moment/synthetic_trend \
  --output_dir outputs/probes/moment_synthetic_trend_linear \
  --probe_type linear
```

3. Evaluate probes

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_probe.py \
  --probe_dir outputs/probes/moment_synthetic_trend_linear \
  --representations_dir outputs/representations/moment/synthetic_trend \
  --output_dir outputs/eval/moment_synthetic_trend_linear
```

4. Build the paper

```bash
make paper
# or, manually: cd outputs/paper/latex && latexmk -pdf main.tex
```

## Provenance

For command-level provenance of paper figures, enhancement outputs, LEACE results,
and steering artifacts, see `PROVENANCE.md`.

## Validation

```bash
ruff check src scripts tests
pytest tests/ -q
```

## Notes

- `configs/*.yaml` are reference configuration documents; the current scripts use CLI arguments.
- Cross-model CKA results separate encoder-family models from decoder-only models, while Timer achieves perfect trend/frequency/stationarity/change-point separation but remains weak on anomaly detection.
