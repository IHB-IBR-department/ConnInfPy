# Existing significant-edge transfer experiment

This independent workflow asks whether training-cohort statistical edge masks support transfer of EO/EC classification to the other cohort. It is separate from atlas contrast stability and from post-fit model importance stability.

`run_ml_feature_selection.py` compares selected edges with all edges, size-matched random subsets and univariate top-k masks. Its maps and metrics retain the historical `examples/open_close/results/ml/` path. `audit_ml_feature_selection.py` summarizes saved metrics. `ml_validation.py` supplies reusable transfer/CV primitives. `openclose_ml_transfer_from_pvals.py` reads the older pickle-map format using the supported Schaefer loader; the default target is China.

```bash
python -m examples.open_close.ml_transfer.run_ml_feature_selection --help
python -m examples.open_close.ml_transfer.openclose_ml_transfer_from_pvals --help
```

This folder preserves the existing analysis design. Grouped tuning does not make cohort-wide statistical selection nested inside every validation fold; within-cohort CV metrics must be interpreted with that distinction. Independent target-cohort evaluation, method agreement and feature reproducibility are different endpoints. Runtime includes permutation maps and many classifier fits.
