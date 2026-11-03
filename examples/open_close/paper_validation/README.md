# Paired inference and publication checks

These runners generate paired EO/EC maps, empirical EC-run controls and descriptive agreement panels. They use `_shared/data.py` and `_shared/inference.py`; no classifier is fitted in this workflow. Outputs go to `../results/paper_validation/` (written `results/` below); figures go to `../results/plots/`.

| Script | Input and output |
|---|---|
| `run_paired_tfnbs.py` | Schaefer time series; cohort/retest TFNBS maps and edge CSVs in `results/` |
| `run_state_methods.py` | Paired state maps for all configured methods in `results/methods/` |
| `harmonize_pooled_cohorts.py` | Pooled state-preserving ComBat; `results/openclose_harmonized.npz` |
| `run_pooled_methods.py` | Explicit harmonized NPZ; pooled method maps in `results/methods/` |
| `run_retest_methods.py` | China complete EC-run pairs; control maps in `results/methods/` |
| `run_eh_sensitivity.py` | Parameter grid; `results/sensitivity/eh/` and a plot |
| `audit_openclose_agreement.py` | Saved cohort maps; `results/audit/` and the block-mass plot |
| `plot*.py` | Saved maps; publication plots |

```bash
python -m examples.open_close.paper_validation.run_paired_tfnbs --analyses ihb china china_retest
python -m examples.open_close.paper_validation.run_state_methods --n-perm 5000
python -m examples.open_close.paper_validation.regenerate_validation_results --dry-run
```

Inference runners accept `--data-dir`; paired/state/retest runners accept `--output`. Pooled inference accepts `--input` and `--output`. Harmonization accepts an output NPZ path. Plot defaults deliberately retain the publication artifact locations. Use each runner's `--help` for permutation budgets and seeds.

Default TFNBS paper runs use E=0.4, H=3, n=10 and 5,000 permutations. The historical sensitivity runner uses GPD acceleration; its outputs should not be described as exact-permutation maps. Both tails are separate families in these scripts. Agreement and detection counts are descriptive, and the EC-run control is not a known population null.
