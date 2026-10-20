# Resolution study (`make resolution-study`)

`results.json` is written by `scripts/run_resolution_study.py analyze` and read by
`tools/make_numbers.py` (`rs/*` keys). Large arrays (datasets, pooled activations, token tensors,
control features, per-window test probabilities) live in `RESOLUTION_STORE`
(`TSFMI_RESOLUTION_STORE`, default `outputs/resolution_store`, gitignored; about 60 GB).

Design: anomaly tasks regenerated with 50,000 windows (data seed 1001; magnitude sweep: 20,000 windows,
seed 2001), one fixed 60/20/20 split, probes at 600 / 3,000 / 30,000 training windows, layer selected
on validation. Fields per dataset: `table` (test accuracy by training size, V-information),
`best_model_vs_best_control` and `at_min_train` (paired McNemar, TOST at 0.02 and 0.01, log-loss
bits), `pretrained_vs_random`, `conditional_tuned` (probes on [control, Z] with C tuned on validation;
`conditional` is the fixed-C version, kept to show the overfitting it causes), `recall_by_kind`.
Top level: `surrogate_drop` (original vs IAAFT), `views` (mean / max / last / attention, pretrained and
re-initialised), `capacity_controls`, `views_conditional` (cross-fitted attention pooling stacked on
all feature controls) and `views_conditional_nocrossfit` (the biased estimate before cross-fitting).
