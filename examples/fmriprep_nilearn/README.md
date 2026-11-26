# fMRIPrep + nilearn Volumetric Workflow

This example shows how to go from **fMRIPrep derivatives** and **on-demand `nilearn` volumetric atlases** (Schaefer-100 or Craddock-200) to `conninfpy` connectivity estimation, TFNBS inference with motion QC, and volumetric Brainnetome + SUIT anatomical annotation (`support="parcel"`).

`conninfpy` keeps `nilearn` optional and does **not** bundle third-party NIfTI atlases or fMRI datasets.

## Prerequisites

Install `conninfpy` with the optional app/neuroimaging dependencies:

```bash
pip install "conninfpy[app]"
# or: pip install nilearn nibabel
```

## Quick start (on-demand nilearn download)

Run the end-to-end script using `nilearn.datasets.fetch_development_fmri()` and `fetch_atlas_schaefer_2018()` (or `--atlas craddock200`):

```bash
python examples/fmriprep_nilearn/run_fmriprep_workflow.py \
    --n-subjects 12 \
    --atlas schaefer100 \
    --n-permutations 100
```

For an offline smoke check with tiny synthetic fMRIPrep derivatives:

```bash
python examples/fmriprep_nilearn/run_fmriprep_workflow.py --synthetic --n-permutations 10
```

Outputs are written to `examples/fmriprep_nilearn/results/` (`manifest.yaml`, `pheno.csv`, `connectogram_positive.png`, and `methods_boilerplate.md`).

## Adapting `manifest_template.yaml` to your own BIDS/fMRIPrep study

1. Copy [`manifest_template.yaml`](manifest_template.yaml) to your project and set:
   - `paths.derivatives_dir`: your `bids/derivatives/fmriprep` folder
   - `paths.pheno_path`: your `participants.tsv` (or phenotype CSV with `subject_id` / `participant_id`)
   - `paths.atlas_image_path`: a 3-D integer parcel NIfTI in `MNI152NLin2009cAsym` (fetched via `nilearn.datasets.fetch_atlas_schaefer_2018(...)` or your own parcellation)
2. Run the `conninfpy` CLI pipeline:

```bash
conninfpy init --manifest manifest.yaml --out analysis.yaml
conninfpy qc analysis.yaml --motion-qc
conninfpy run analysis.yaml --out results/my_contrast
conninfpy interpret results/my_contrast \
    --support parcel \
    --parcel-nifti /path/to/atlas_MNI152NLin2009cAsym.nii.gz \
    --space MNI152NLin2009cAsym \
    --cerebellum anatom \
    --methods
```
