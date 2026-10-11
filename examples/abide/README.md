# ABIDE observational multi-site workflows

Prepare the dataset once, then choose the age or diagnosis question. Both use `_shared/data.py` to load prepared matrices and construct consistent atlas metadata.

| Location | Purpose |
|---|---|
| `prepare_data.py` | Time-series QC, connectivity and phenotype preparation; `results/abide_prepared.npz` |
| `age/` | Age/selected covariate analyses, enhancement-method comparison and parameter sensitivity |
| `diagnosis/` | ASD–HC contrasts under naive, ComBat-only, site-only and combined strategies; network-informed methods |
| `interpretation/` | Brainnetome annotation of saved/recomputed results |
| `plots/` | Publication panels from saved maps and prepared data |
| `configs/` | Standalone ConnInfPy CLI configuration examples |

```bash
python -m examples.abide.prepare_data
python -m examples.abide.age.run_age_strategy_d --variant age
python -m examples.abide.diagnosis.run_dx_site_glm --help
```

Preparation defaults to `datasets/ABIDE_TS_Schaefer100/` (override with `ABIDE_TS_DIR`). Age and diagnosis runners accept `--data-file` and `--output`. Generated maps retain `results/age_development/` and `results/diagnosis/`; plots retain `results/plots/`. All paths are based on the repository, independently of cwd. Default full inference is expensive; inspect `--help` before running.

These are observational analyses. Changes after site adjustment, enhancement-method agreement and sensitivity across settings do not by themselves establish biological discovery or the necessity of one harmonization strategy. Age and diagnosis have separate documented covariate designs. Atlas annotation changes interpretation, not the underlying inference.
