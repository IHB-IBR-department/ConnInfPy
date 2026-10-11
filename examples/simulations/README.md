# Calibration, power and sensitivity simulations

| Location | Question |
|---|---|
| `fpr/` | Directional and two-sided FWER plus edge-level FPR under simulated nulls |
| `power/` | TPR, false-discovery proportion and precision for planted correlation effects |
| `glm/` | GLM null calibration and planted-effect power |
| `sensitivity/` | Parameter, normalization and prior-misspecification comparisons |
| `sc_prior/` | ENIGMA SC extraction and structural-prior feasibility |
| `_shared/` | Shared binomial calibration summaries |
| `topology_gallery.py`, `regenerate_paper_figures.py` | Figures/tables from simulated graphs and saved outputs |

Select a configuration explicitly:

```bash
python -m examples.simulations.fpr.fpr_calibration --config examples/simulations/configs/fpr_config_quick.yaml
python -m examples.simulations.power.power_analysis --config examples/simulations/configs/sweep_config_power_quick.yaml --n-jobs 1
python -m examples.simulations.glm.glm_validation --help
```

YAML output paths resolve relative to the configuration file. Bundled configurations use `../results/...`; CLI output paths are explicit caller paths. Default paths resolve from the repository, not cwd. Full grids can take hours; quick configurations are reduced research runs, not unit tests.

`n_jobs` controls parallel simulation tasks (`-1`: available CPU budget, `1`: serial); process counts are capped by pending tasks. Child workers limit native threads to one. Power always runs inner permutations serially, replacing its previously ineffective `use_mp` control. FPR/GLM retain `use_mp` for inner permutations; select it only when using serial outer execution, since nested process pools are suppressed by the package runtime.

Calibration is conditional on the tested generative models and settings. Binomial intervals and simultaneous checks quantify sampling uncertainty; they do not guarantee exactly 5% error for every dataset. Power here uses planted population correlation shifts, not Cohen's d. Existing checkpoints and scientific scoring rules are preserved.
