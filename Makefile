VENV := .venv/bin
PYTHON := PYTHONPATH=. $(VENV)/python
# Use whichever LaTeX engine is available on PATH; users may override via:
#   make paper LATEXMK=/path/to/latexmk
LATEXMK ?= latexmk
LATEX_DIR := outputs/paper/latex
PAPER_MAIN := $(LATEX_DIR)/main.tex

.PHONY: test lint format paper paper-submission paper-preprint paper-final \
        reproduce-canonical reproduce-baselines reproduce-figures reproduce-all \
        reproduce-all-from-scratch extract-representations smoke benchmark-time \
        per-feature-anomaly paired-wilcoxon rocket-baselines dataset-seed-bootstrap \
        cross-leace-bootstrap mdl-probe realistic-anomaly \
        manifest clean help

test:
	$(VENV)/python -m pytest tests/ -q

lint:
	$(VENV)/ruff check src scripts tests

format:
	$(VENV)/ruff format src scripts tests

# Reproducibility targets (assume representations are already extracted into
# outputs/representations/; otherwise run `make extract-representations` first).
reproduce-baselines:
	$(PYTHON) scripts/run_canonical_baselines.py

reproduce-canonical:
	bash scripts/run_canonical_all.sh

reproduce-figures:
	$(PYTHON) scripts/generate_all_paper_figures.py

manifest:
	$(PYTHON) scripts/count_experiments.py

reproduce-all: reproduce-baselines reproduce-canonical manifest reproduce-figures paper
	@echo "=== Full reproduction pipeline complete ==="

# End-to-end from a clean checkout (extraction + canonical + figures + paper).
# Expected wall-clock on a single A100: ~48h. Logs to outputs/paper/timing.json.
reproduce-all-from-scratch: extract-representations reproduce-all
	@echo "=== Full from-scratch reproduction complete ==="

# Extracts frozen representations for the 7 confirmatory models x 6 synthetic properties.
# Skips any (model, dataset) whose tensor directory already exists.
extract-representations:
	bash scripts/extract_all_canonical.sh

# Small-footprint reproduction: one (model, property) cell end-to-end in <10 min.
# Useful for reviewers who want to verify the pipeline without 48 GPU-hours.
smoke:
	bash scripts/run_smoke.sh

# Wall-clock instrumentation for the canonical pipeline. Writes outputs/paper/timing.json.
benchmark-time:
	$(PYTHON) scripts/log_pipeline_timing.py

# Pre-submission additional experiments (CHECK.md G1-G7).
per-feature-anomaly:
	$(PYTHON) scripts/run_per_feature_anomaly.py

paired-wilcoxon:
	$(PYTHON) scripts/run_paired_wilcoxon.py

rocket-baselines:
	$(PYTHON) scripts/run_rocket_baseline.py

dataset-seed-bootstrap:
	$(PYTHON) scripts/run_dataset_seed_bootstrap.py

cross-leace-bootstrap:
	$(PYTHON) scripts/run_cross_property_leace_bootstrap.py

mdl-probe:
	$(PYTHON) scripts/run_mdl_probe.py

realistic-anomaly:
	$(PYTHON) scripts/run_realistic_anomaly.py

paper:
	$(MAKE) paper-submission

paper-submission:
	cd $(LATEX_DIR) && $(LATEXMK) -pdf -bibtex -interaction=nonstopmode main.tex

paper-preprint:
	$(VENV)/python -c 'from pathlib import Path; src = Path("$(PAPER_MAIN)").read_text(); dst = src.replace("\\usepackage[eandd]{neurips_2026}", "\\usepackage[preprint]{neurips_2026}", 1); dst = dst.replace("\\input{authors_submission.tex}", "\\input{authors_preprint.tex}", 1); dst = dst.replace("\\input{ack_submission.tex}", "\\input{ack_preprint.tex}", 1); Path("$(LATEX_DIR)/_preprint.tex").write_text(dst)'
	cd $(LATEX_DIR) && $(LATEXMK) -pdf -bibtex -interaction=nonstopmode _preprint.tex
	cd $(LATEX_DIR) && $(LATEXMK) -pdf -bibtex -interaction=nonstopmode _preprint.tex

paper-final:
	$(VENV)/python -c 'from pathlib import Path; src = Path("$(PAPER_MAIN)").read_text(); dst = src.replace("\\usepackage[eandd]{neurips_2026}", "\\usepackage[eandd,final]{neurips_2026}", 1); dst = dst.replace("\\input{authors_submission.tex}", "\\input{authors_final.tex}", 1); dst = dst.replace("\\input{ack_submission.tex}", "\\input{ack_final.tex}", 1); Path("$(LATEX_DIR)/_final.tex").write_text(dst)'
	cd $(LATEX_DIR) && $(LATEXMK) -pdf -bibtex -interaction=nonstopmode _final.tex
	cd $(LATEX_DIR) && $(LATEXMK) -pdf -bibtex -interaction=nonstopmode _final.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null
	rm -rf .coverage .mypy_cache
	rm -f $(LATEX_DIR)/_preprint.tex $(LATEX_DIR)/_final.tex $(LATEX_DIR)/*.aux $(LATEX_DIR)/*.bbl $(LATEX_DIR)/*.blg $(LATEX_DIR)/*.out $(LATEX_DIR)/*.log $(LATEX_DIR)/*.fdb_latexmk $(LATEX_DIR)/*.fls

help:
	@echo "test    - Run pytest"
	@echo "lint    - Run ruff check"
	@echo "format  - Run ruff format"
	@echo "paper   - Compile submission-mode paper"
	@echo "paper-submission - Compile NeurIPS submission mode"
	@echo "paper-preprint   - Compile NeurIPS preprint mode"
	@echo "paper-final      - Compile NeurIPS final mode"
	@echo "extract-representations - Run frozen-model extraction for the 7 confirmatory models"
	@echo "smoke   - Reproduce one (model, property) cell end-to-end (<10 min)"
	@echo "reproduce-all-from-scratch - Extraction + canonical + figures + paper"
	@echo "benchmark-time - Instrument the canonical pipeline wall-clock"
	@echo "per-feature-anomaly  - HC per-feature attribution on anomaly (CHECK.md G1)"
	@echo "paired-wilcoxon      - HC vs each TSFM paired Wilcoxon (CHECK.md G4)"
	@echo "rocket-baselines     - ROCKET/MiniROCKET baseline (CHECK.md G3)"
	@echo "dataset-seed-bootstrap - Re-bootstrap canonical with multiple data seeds (CHECK.md G2)"
	@echo "cross-leace-bootstrap  - Bootstrap entanglement diagnostic (CHECK.md H2)"
	@echo "mdl-probe              - Voita-Titov MDL probe (CHECK.md G5)"
	@echo "realistic-anomaly      - Realistic anomaly generator + canonical re-run (CHECK.md G6)"
	@echo "clean   - Remove cache files"
