# ABIDE observational multi-site workflows

Prepare the dataset once, then choose the age or diagnosis question. Both use `_shared/data.py` to load prepared matrices and construct consistent atlas metadata.

## Which ABIDE dataset?

Three ABIDE variants are referenced across the repository:

| Dataset | Atlas / Size | Where it lives | Used by |
|---|---|---|---|
| **ABIDE I CC200** *(primary manuscript)* | Craddock-200 (`rois_cc200`) | External (`../local_datasets/ABIDE_pcp/` or `~/nilearn_data/ABIDE_pcp/`) | `prepare_cc200.py` → `age/` & `diagnosis/` (`--prepared-npz`) |
| **ABIDE-mini** *(bundled demo)* | Schaefer-100 (204 subjects, 5 sites) | Tracked in `datasets/demo/ABIDE_mini/` | `datasets/abide_mini.yaml`, Streamlit demo, `configs/abide_mini_age.yaml` |
| **Legacy Schaefer-100** | Schaefer-100 (full cohort) | External `ABIDE_TS_DIR` (outside repo) | `prepare_data.py` → `results/abide_prepared.npz` |

| Location | Purpose |
|---|---|
| `prepare_cc200.py` | **Primary** ABIDE I CC200 preparation (NaN-preserving, 11-block partition) |
| `prepare_data.py` | Legacy Schaefer-100 time-series QC, connectivity and phenotype preparation |
| `age/` | Age/selected covariate analyses, enhancement-method comparison and parameter sensitivity |
| `diagnosis/` | ASD–HC contrasts under naive, ComBat-only, site-only and combined strategies; network-informed methods |
| `brainnetome_interpretation.py` | Brainnetome annotation and connectograms of the age and diagnosis maps |
| `plots/` | Publication panels from saved maps and prepared data |
| `configs/` | Standalone ConnInfPy CLI configuration examples |

## Primary ABIDE I CC200 workflow

By default, `prepare_cc200.py` checks `../local_datasets/ABIDE_pcp/` (if present) and otherwise falls back to `~/nilearn_data/ABIDE_pcp/`, reading the ABIDE Preprocessed Connectomes Project (CPAC `cpac/filt_global` strategy) files (with `--roi-dir`, `--pheno`, `--atlas`, and `--roi-table` overrides):

- `cpac/filt_global/*_rois_cc200.1D` (ROI time series)
- `Phenotypic_V1_0b_preprocessed1.csv`
- `cc200_roi_atlas.nii.gz` (and optional `CC200_ROI_labels.csv`)

Download them with `nilearn` if not already present:

```python
from nilearn.datasets import fetch_abide_pcp

fetch_abide_pcp(
    derivatives=["rois_cc200"],
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=True,
)
```

(Ensure `cc200_roi_atlas.nii.gz` and `CC200_ROI_labels.csv` from PCP/`nilearn` are placed in `~/nilearn_data/ABIDE_pcp/`.)

Prepare the CC200 dataset and run age or diagnosis analyses with `--prepared-npz` (plus optional `--roi-subset assigned` or `--complete-case`):

```bash
python examples/abide/prepare_cc200.py --output examples/abide/results/cc200/abide_cc200_prepared.npz

python -m examples.abide.age.run_age_strategy_d \
    --prepared-npz examples/abide/results/cc200/abide_cc200_prepared.npz \
    --roi-subset assigned \
    --variant age

python -m examples.abide.diagnosis.run_dx_site_glm \
    --prepared-npz examples/abide/results/cc200/abide_cc200_prepared.npz \
    --complete-case
```

## Results layout

One `results/` folder; each analysis writes to the subfolder named like its code folder. Interpretation is written next to the result it describes.

```
results/
  cc200/abide_cc200_prepared.npz   prepare_cc200.py output (primary CC200)
  abide_prepared.npz               prepare_data.py output (legacy Schaefer-100)
  age/                             Strategy D age map and edge table
    methods/                       enhancement-method comparison
    sensitivity/eh/                E/H grid
    brainnetome/                   annotation, connectograms, report.md
  diagnosis/                       Strategy D diagnosis map and network-informed variants
    naive/  combat_only/  strategy_e/
    brainnetome/                   annotation, connectograms, report.md
  plots/                           publication figures (tracked)
```

## Legacy Schaefer-100 run

```bash
ABIDE_TS_DIR=/path/to/ABIDE_TS_Schaefer100 python -m examples.abide.prepare_data
python -m examples.abide.age.run_age_strategy_d --variant age
python -m examples.abide.diagnosis.run_dx_site_glm --help
python -m examples.abide.brainnetome_interpretation --help
```

`prepare_data.py` requires `ABIDE_TS_DIR` pointing to an external Schaefer-100 time-series directory (or `datasets/ABIDE_TS_Schaefer100/` if placed locally). Age and diagnosis runners accept `--prepared-npz` / `--data-file` and `--output`. All paths are based on the repository, independently of cwd. Default full inference is expensive; inspect `--help` before running.

These are observational analyses. Changes after site adjustment, enhancement-method agreement and sensitivity across settings do not by themselves establish biological discovery or the necessity of one harmonization strategy. Age and diagnosis have separate documented covariate designs. Atlas annotation changes interpretation, not the underlying inference.
