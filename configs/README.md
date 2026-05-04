# Configuration Reference

These YAML files document the hyperparameters used for each model's probing experiments. They serve as **reference documentation** — the actual scripts use command-line arguments and hardcoded defaults rather than loading these files.

To reproduce an experiment using the documented settings, translate the YAML values into CLI flags:

```bash
# Example: MOMENT linear probing (from probe_moment.yaml)
PYTHONPATH=. python scripts/train_probe.py \
    --representations_dir outputs/representations/moment/synthetic_trend \
    --output_dir outputs/probes/moment_synthetic_trend_linear \
    --probe_type linear
```
