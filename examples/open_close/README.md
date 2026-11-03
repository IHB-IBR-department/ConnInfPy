# Open-Close workflows

The IHB and Beijing cohorts contain paired eyes-open (EO) and eyes-closed (EC) scans. The current atlas question concerns **EC−EO contrast stability**: how native Brainnetome parcellation and Brainnetome annotation of another atlas change the reproducibility and anatomical interpretation of the contrast.

| Folder | Scientific question | Entry point |
|---|---|---|
| [paper_validation/](paper_validation/) | Paired state contrasts, EC-run control, method and parameter agreement, publication plots | `run_paired_tfnbs.py`, `run_state_methods.py`, `run_retest_methods.py`, `plot*.py` |
| [atlas_stability/](atlas_stability/) | Contrast stability across Schaefer, native BNA and a coarser BNA resolution control | `run_atlas_stability.py` |
| [structural_prior/](structural_prior/) | SC-restricted edge families versus SC-defined NI-TFNBS blocks | `run_sc_inference.py`, `run_sc_blocks.py` |
| [ml_transfer/](ml_transfer/) | Existing significant-edge cross-cohort transfer experiment | `run_ml_feature_selection.py` |
| [feature_stability/](feature_stability/) | Existing edge-model importance and overlap across bootstrap fits, summarized after fitting | `run_feature_stability.py`, `feature_stability.py` |
| `brainnetome_interpretation.py` | Anatomical attribution of the paired maps; native BNA comparison | this script |

The atlas-stability workflow contains no classifiers. Feature stability and ML transfer are independent existing studies with different questions; running the atlas study does not invoke them.

`_shared/` holds the common study adapter (`data.py`), selector inference, agreement metrics and `paths.py`, which defines the results layout below. Defaults read `datasets/open_close/`: Schaefer-200 `strategy-4_GSR` arrays, ROI labels and subject order; native Brainnetome workflows additionally need the corresponding BNA arrays and atlas resources. The retained Schaefer set has 182 ROIs; native BNA uses 242 after the documented provisional dropout exclusions. China has 48 participants; workflows requiring both EC runs use 47 complete participants. Each workflow states its run choice.

## Results layout

One `results/` folder; each workflow writes to the subfolder of the same name. Interpretation is written next to the result it describes.

```
results/
  paper_validation/      paired TFNBS maps (ihb, china run 0, china EC-run retest, pooled),
                         harmonized NPZ and edge tables
    methods/             per-method paired p-maps (state, retest, pooled)
    sensitivity/eh/      E/H grid
    audit/               IHB vs China agreement
    brainnetome/         Brainnetome annotation of the Schaefer maps, one folder per analysis
    brainnetome_native/  native BNA-246 paired contrast and its annotation
  atlas_stability/
  structural_prior/      SC masks; blocks/ for the SC-block NI-TFNBS runs
  feature_stability/     one folder per run
  ml_transfer/           transfer metrics and the selector p-maps it recomputes
  plots/                 publication figures (tracked)
```

Outputs are git-ignored except publication plots and a few small summary tables. Saved configurations and provenance inside existing runs keep the paths they were written with.

## Brainnetome interpretation

`brainnetome_interpretation.py` applies Brainnetome anatomy to the stored Schaefer paired maps (`--atlas schaefer200`, default) or runs the separate native Brainnetome contrast (`--atlas brainnetome`). Annotation preserves the Schaefer result arrays; parcel overlap, purity and attribution level constrain anatomical precision. Native parcellation changes the measured signals and edge family. Each analysis folder gets its tables, connectograms and `report.md`; `--report PATH` also writes one combined Markdown narrative (e.g. into the project vault). The atlas-stability workflow provides the matched participant/bootstrap comparison; annotation alone is not evidence of improved reproducibility.

## Run

From the repository root:

```bash
python -m examples.open_close.paper_validation.run_paired_tfnbs --help
python -m examples.open_close.paper_validation.regenerate_validation_results --dry-run
python -m examples.open_close.atlas_stability.run_atlas_stability --help
python -m examples.open_close.brainnetome_interpretation --help
```

The regeneration runner defaults to inference and plots. `--scope ml` selects the separate transfer experiment; `--scope all` includes both. It uses the active interpreter. Full inference and sensitivity runs are expensive; `--quick` reduces budgets but still analyzes real data.

Paired sign flips require the stated exchangeability/symmetry assumptions for participant differences. EC-run comparisons are empirical controls, not guaranteed null data: vigilance, order and drift can differ. ComBat diagnostics describe adjusted cohort variation, not proof that scanner effects or confounding have disappeared. Annotation and grouped summaries do not create new group-level p-values or demonstrate biological validity.
