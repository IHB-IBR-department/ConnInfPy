# Open-Close workflows

The IHB and Beijing cohorts contain paired eyes-open (EO) and eyes-closed (EC) scans. The current atlas question concerns **EC−EO contrast stability**: how native Brainnetome parcellation and Brainnetome annotation of another atlas change the reproducibility and anatomical interpretation of the contrast.

| Folder | Scientific question | Entry point |
|---|---|---|
| [paper_validation/](paper_validation/) | Paired state contrasts, EC-run control, method and parameter agreement | `run_paired_tfnbs.py`, `run_state_methods.py`, `run_retest_methods.py` |
| [atlas_stability/](atlas_stability/) | Contrast stability across Schaefer, native BNA and a coarser BNA resolution control | `run_atlas_stability.py` |
| [interpretation/](interpretation/) | Anatomical attribution of edge inference; native BNA comparison | `brainnetome_interpretation.py` |
| [structural_prior/](structural_prior/) | SC-restricted edge families versus SC-defined NI-TFNBS blocks | `run_sc_inference.py`, `run_sc_blocks.py` |
| [ml_transfer/](ml_transfer/) | Existing significant-edge cross-cohort transfer experiment | `run_ml_feature_selection.py` |
| [feature_stability/](feature_stability/) | Existing edge-model importance and overlap across bootstrap fits, summarized after fitting | `run_feature_stability.py`, `feature_stability.py` |

The atlas-stability workflow contains no classifiers. Feature stability and ML transfer are independent existing studies with different questions; running the atlas study does not invoke them.

`_shared/data.py` is the common study adapter. Defaults read `datasets/open_close/`: Schaefer-200 `strategy-4_GSR` arrays, ROI labels and subject order; native Brainnetome workflows additionally need the corresponding BNA arrays and atlas resources. The retained Schaefer set has 182 ROIs; native BNA uses 242 after the documented provisional dropout exclusions. China has 48 participants; workflows requiring both EC runs use 47 complete participants. Each workflow states its run choice.

Publication maps and plots remain in `results/`, including the historical `results/ml/` location for method maps. Atlas, structural-prior and feature-stability results reside within their respective workflow directories. Existing result files and provenance are preserved.

From the repository root:

```bash
python -m examples.open_close.atlas_stability.run_atlas_stability --help
python -m examples.open_close.paper_validation.run_paired_tfnbs --help
python -m examples.open_close.paper_validation.regenerate_validation_results --dry-run
```

The regeneration runner defaults to inference and plots. `--scope ml` selects the separate transfer experiment; `--scope all` includes both. It uses the active interpreter. Full inference and sensitivity runs are expensive; `--quick` reduces budgets but still analyzes real data.

Paired sign flips require the stated exchangeability/symmetry assumptions for participant differences. EC-run comparisons are empirical controls, not guaranteed null data: vigilance, order and drift can differ. ComBat diagnostics describe adjusted cohort variation, not proof that scanner effects or confounding have disappeared. Annotation and grouped summaries do not create new group-level p-values or demonstrate biological validity.
