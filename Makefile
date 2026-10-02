VENV := .venv/bin
PYTHON := PYTHONPATH=. $(VENV)/python
# Use whichever LaTeX engine is available on PATH; users may override via:
#   make paper LATEXMK=/path/to/latexmk
LATEXMK ?= latexmk
LATEX_DIR := outputs/paper/latex
PAPER_MAIN := $(LATEX_DIR)/main.tex

.PHONY: test lint format paper paper-submission paper-preprint paper-final \
        reproduce-canonical reproduce-baselines reproduce-figures reproduce-all \
        reproduce-all-from-scratch extract-representations smoke reproduce-one-cell \
        benchmark-time per-feature-anomaly paired-wilcoxon paired-bootstrap \
        rocket-baselines dataset-seed-bootstrap cross-leace-bootstrap mdl-probe \
        realistic-anomaly realistic-anomaly-tsfm pca-leakfree-rerun reproduce-canonical-v2 \
        realistic-anomaly-v2 hard-variants-v2 dataseed-v2 hl-selectivity-v2 \
        statistic-recoverability statistic-erasure anomaly-diagnostics ucr-probe \
        representation-storage extraction-throughput wilinski-ldr reproduce-icde \
        per-feature-all-properties bayes-ceiling strong-baselines multicapacity-probe \
        view-sensitivity \
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

# CPU-only mechanical check: HC baseline + per-feature attribution + tests.
# Does NOT extract representations or run a TSFM probe; use `make reproduce-one-cell`
# if you want to regenerate one row of the canonical TSFM table from a clean checkout.
smoke:
	bash scripts/run_smoke.sh

# Genuine end-to-end reproduction of ONE canonical (model, property) cell.
# Default: PatchTST-Pre / synthetic_anomaly. Override via ONLY_MODEL / ONLY_DATASET.
reproduce-one-cell:
	bash scripts/run_reproduce_one_cell.sh

# Wall-clock instrumentation for the canonical pipeline. Writes outputs/paper/timing.json.
benchmark-time:
	$(PYTHON) scripts/log_pipeline_timing.py

# Pre-submission additional experiments (CHECK.md G1-G7, REVIEW.md priorities).
per-feature-anomaly:
	$(PYTHON) scripts/run_per_feature_anomaly.py

paired-wilcoxon:
	$(PYTHON) scripts/run_paired_wilcoxon.py

paired-bootstrap:
	$(PYTHON) scripts/run_paired_bootstrap.py

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

realistic-anomaly-tsfm:
	$(PYTHON) scripts/run_realistic_anomaly_tsfm.py

pca-leakfree-rerun:
	$(PYTHON) scripts/run_pca_leakfree_rerun.py

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

# Canonical benchmark v2: one protocol (n=1000, 60/20/20, seeds 0-4, val-selected layer,
# train-only PCA inside the per-seed pipeline for MOMENT/GPT4TS, mean-pool elsewhere) and one
# environment for all 7 TSFMs + 3 baselines. Writes outputs/canonical_v2/ (see its README.md).
# Override REPR_ROOT / GPUS / BATCH_SIZE / BENCH_JOBS as environment variables.
reproduce-canonical-v2:
	bash scripts/run_canonical_v2.sh

# --- ICDE manuscript artifacts (same protocol and environment as reproduce-canonical-v2) --------
# Each target writes one outputs/ directory that the manuscript reads; `make reproduce-icde` runs all.
REPR_ROOT ?= outputs/representations_v2
HARD_PROPS := synthetic_trend_hard:trend_hard:classification \
    synthetic_frequency_hard:frequency_hard:classification \
    synthetic_anomaly_hard:anomaly_hard:classification \
    synthetic_stationarity_hard:stationarity_hard:classification \
    synthetic_change_point_hard:change_point_hard:classification
ANOMALY_PROPS := synthetic_anomaly:anomaly:classification \
    synthetic_anomaly_realistic:anomaly_realistic:classification
WILINSKI_REPO ?= external/representations-in-tsfms
WILINSKI_COMMIT := 72337c6e4072d4c213588c116633be71a1b87760

realistic-anomaly-v2: realistic-anomaly
	REPR_ROOT=$(REPR_ROOT) OUT_ROOT=outputs/realistic_anomaly_v2 STAGES="env extract pool bench" \
	PROPS_OVERRIDE="synthetic_anomaly_realistic:anomaly_realistic:classification" \
	bash scripts/run_canonical_v2.sh

hard-variants-v2:
	REPR_ROOT=$(REPR_ROOT) OUT_ROOT=outputs/hard_variants_v2 STAGES="env extract pool bench" \
	PROPS_OVERRIDE="$(HARD_PROPS)" bash scripts/run_canonical_v2.sh
	$(PYTHON) scripts/run_canonical_baselines.py --output_dir outputs/hard_variants_v2/baselines \
	    --properties trend_hard frequency_hard anomaly_hard stationarity_hard change_point_hard

dataseed-v2:
	for s in 43 44; do \
	  DATA_SEED=$$s REPR_ROOT=$(REPR_ROOT)_dataseed$$s OUT_ROOT=outputs/dataseed_v2/seed$$s \
	  STAGES="env extract pool bench" PROPS_OVERRIDE="$(ANOMALY_PROPS)" \
	  bash scripts/run_canonical_v2.sh || exit 1; \
	done
	$(PYTHON) scripts/summarize_dataseed.py --out outputs/dataseed_v2/summary.json

hl-selectivity-v2:
	$(PYTHON) scripts/run_hl_selectivity.py --protocol v2 --ks 20 50 --baseline_num_samples 1000 \
	    --negative_control --n_threads 16

statistic-recoverability:
	$(PYTHON) scripts/run_statistic_recoverability.py --repr-root $(REPR_ROOT) \
	    --canonical-root outputs/canonical_v2 --out outputs/statistic_recoverability

statistic-erasure:
	$(PYTHON) scripts/run_statistic_erasure.py --out outputs/statistic_erasure
	$(PYTHON) scripts/run_statistic_erasure.py --task canonical --concept handcrafted \
	    --out outputs/statistic_erasure/canonical_handcrafted
	$(PYTHON) scripts/run_statistic_erasure.py --task realistic --concept handcrafted \
	    --out outputs/statistic_erasure/realistic_handcrafted
	$(PYTHON) scripts/run_statistic_erasure.py --task realistic --concept statistics \
	    --out outputs/statistic_erasure/realistic_statistics

view-sensitivity:
	$(PYTHON) scripts/run_view_sensitivity.py --out outputs/view_sensitivity

per-feature-all-properties:
	$(PYTHON) scripts/run_per_feature_all_properties.py --out outputs/per_feature_all_properties

bayes-ceiling:
	$(PYTHON) scripts/run_bayes_ceiling.py --out outputs/bayes_ceiling

strong-baselines:
	$(PYTHON) scripts/run_strong_baselines.py

# Probes the mean-pooled Timer, TimesFM and Moirai views of the canonical v2 store.
multicapacity-probe:
	TSFMI_REPR_ROOT=$(REPR_ROOT) $(PYTHON) scripts/run_multicapacity_adversarial.py

anomaly-diagnostics:
	$(PYTHON) scripts/run_anomaly_diagnostics.py --out outputs/anomaly_diagnostics

ucr-probe:
	$(PYTHON) scripts/run_ucr_tsfm_probe.py --models moment chronos gpt4ts timer timesfm moirai \
	    patchtst_fm --datasets ECG200 Wafer Earthquakes --out outputs/ucr_tsfm_probe_v2
	$(PYTHON) scripts/run_ucr_tsfm_probe.py --models moment chronos gpt4ts timer timesfm moirai \
	    patchtst_fm --datasets FordA FordB Strawberry --out outputs/ucr_tsfm_probe_ext

representation-storage:
	$(PYTHON) scripts/measure_representation_storage.py --repr-root $(REPR_ROOT) \
	    --out outputs/representation_storage

extraction-throughput:
	$(PYTHON) scripts/measure_extraction_throughput.py --out outputs/extraction_throughput

wilinski-ldr:
	test -d $(WILINSKI_REPO) || git clone \
	    https://github.com/moment-timeseries-foundation-model/representations-in-tsfms $(WILINSKI_REPO)
	git -C $(WILINSKI_REPO) checkout $(WILINSKI_COMMIT)
	$(PYTHON) scripts/run_wilinski_moment_ldr.py --wilinski-repo $(WILINSKI_REPO)

reproduce-icde: reproduce-canonical-v2 realistic-anomaly-v2 hard-variants-v2 dataseed-v2 \
    hl-selectivity-v2 statistic-recoverability statistic-erasure anomaly-diagnostics ucr-probe \
    representation-storage extraction-throughput wilinski-ldr per-feature-anomaly rocket-baselines \
    per-feature-all-properties bayes-ceiling strong-baselines multicapacity-probe view-sensitivity

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
	@echo "reproduce-canonical-v2 - One-protocol canonical re-run, 7 TSFMs + baselines (outputs/canonical_v2)"
	@echo "reproduce-icde         - Every result the ICDE manuscript reads (v2 protocol; GPU; many hours)"
	@echo "clean   - Remove cache files"
