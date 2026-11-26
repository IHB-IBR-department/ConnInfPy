# Reproducing the manuscript figures

The ConnInfPy 2.0 manuscript figures (600 DPI PNG) are built from locked
`examples/*/results/` outputs and `paper/data/` summary CSVs by two reproducible
scripts:

```bash
python paper/build_fig1_pipeline.py
python paper/build_manuscript_figures.py --group all   # all | sim | abide | openclose
```

Add `--repo-only` to either Python command to write just to `paper/figures/`.
The root README uses a separate [detailed workflow PNG](../docs/conninfpy-workflow.png).

`paper/reproduce_figures.sh` runs the underlying `examples/` computations and its
`collect` stage invokes both figure builders and copies simulation summary tables
into `paper/figures/` (git-ignored):

```bash
bash paper/reproduce_figures.sh              # all stages
bash paper/reproduce_figures.sh collect      # rebuild 2.0 figures from existing results
```

Inputs: ABIDE I CC200/Schaefer-100 data prepared in `examples/abide/results/cc200/`
(via `examples/abide/prepare_cc200.py`), Open-Close data in `datasets/open_close/`,
and self-contained repository CSVs under `paper/data/` (`exact_grid_paired_comparison.csv`,
`openclose_agreement_summary.csv`). Simulations and benchmarks require no external
datasets.

## Figure map (ConnInfPy 2.0)

| Figure | Produced file (`paper/figures/...`) | Builder |
|---|---|---|
| Fig 1 | `figures/fig1_pipeline.png` | `paper/build_fig1_pipeline.py` |
| Fig 2 | `simulation/fig2_fwer_calibration.png` | `paper/build_manuscript_figures.py --group sim` |
| Fig 3 | `simulation/fig3_power_heatmap.png` | `paper/build_manuscript_figures.py --group sim` |
| Fig 4 | `abide/fig4_abide_cc200_workflow.png` | `paper/build_manuscript_figures.py --group abide` |
| Fig 5 | `abide/fig5_abide_cc200_bna_suit_showcase.png` | `paper/build_manuscript_figures.py --group abide` |
| Fig 6 | `openclose/fig6_openclose_bna242_showcase.png` | `paper/build_manuscript_figures.py --group openclose` |
| Fig 7 | `figures/fig7_timing_benchmark.png` | `examples/benchmarks/performance/timing_benchmark.py` → `build_manuscript_figures.py --group sim` |
| Fig S1 | `abide/figS1_abide_cc200_combat_qc.png` | `paper/build_manuscript_figures.py --group abide` |
| Fig S2 | `abide/figS2_abide_cc200_eh_multiverse.png` | `paper/build_manuscript_figures.py --group abide` |
| Fig S3 | `openclose/figS3_openclose_retest_orthogonality.png` | `paper/build_manuscript_figures.py --group openclose` |
| Fig S4 | `simulation/figS4_topologies_gallery.png` | `paper/build_manuscript_figures.py --group sim` |
| Fig S5 | `simulation/figS5_tfnbs_eh_sweep_and_paired_diff.png` | `paper/build_manuscript_figures.py --group sim` |

## Notes on simulation baselines

- **Exact integration (`n=None`)**: All TFNBS-family bars/heatmaps in Fig 2 and
  Fig 3 (`TFNBS`, `NI-TFNBS`, `FBC-TFNBS`) use ConnInfPy 2.0 exact piecewise
  integration (`n=None`, `E=0.4, H=3.0, start_thres=1.65`).
- **`NBS-intensity`**: Uses the MATLAB NBS v1.2-aligned suprathreshold excess
  $\sum (t - \tau)$.
- **Fig 2**: The shaded reference spans the central 95% of the rejection
  fraction under Binomial(1,000, 0.05). Error bars show the confidence interval
  for the point estimate in each panel.
- **Fig 3**: Both panels report edge-level localization. Each detected
  component or block is projected onto its constituent edges before scoring.
- **Fig S5**: Panels a–b show the 176-cell $(E, H)$ hyperparameter sweep at
  $\Delta r = 0.125$ (`n=10`). Panel c shows the paired Monte Carlo difference
  (`Exact n=None` minus `n=10` grid, 95% CI) across $\Delta r \in \{0.075, 0.125, 0.25\}$.

## Details for manuscript captions

- **Fig 1**: Panel b shows the statistical design. The optional site-handling
  choices in c belong to the GLM path. In d, exact piecewise TFNBS integration
  computes the enhancement scores. Monte Carlo permutation inference computes
  the corrected p-values.
- **Fig 4**: Each matrix shows the diagonal and lower triangle of a symmetric
  count matrix. The ASD–HC strategies share a color scale. The two Age tails
  share their own scale.
- **Fig 5**: The connectogram shows the 80 gyrus/network chords with the most
  detected positive edges among placed ROIs (249 of 561 detected edges).
  Of 200 ROIs, 193 were placed, and 58 detected edges touch an unplaced ROI.
  Panel b reports descriptive counts. Panel c shows descriptive coupling
  classes alongside a separate edge-significance mask. Homology is unknown
  for the CC200 ROIs.
- **Fig 6**: Panel a uses Schaefer-200 edges grouped by Yeo-7 network. Panels
  b–c use native BNA-242. The connectogram shows 90 chords covering 554 of
  1,257 detected positive edges. All 120 matched homologs have positive
  coupling in both eyes-open and eyes-closed conditions.
- **Fig 7**: Each plotted value is the mean wall-clock time of an inference
  call divided by its 100 permutations. Data generation and JIT warm-up are
  outside the timed call.
- **Fig S1**: The variance ratio uses the mean across edges of the variance
  of site means divided by the mean across edges of total participant variance.
  The boxplots show participant-level mean residual connectivity.
- **Fig S2**: The TFNBS heatmaps order the grid by increasing E and then
  increasing H within each E group.
- **Fig S3**: The state contrast uses 48 pairs and the repeat-run contrast
  uses 47. Both use the first eyes-closed run (EC₀). NBS counts are member
  edges of detected components. The overlap metrics in b refer to TFNBS.
