# Reproducing the manuscript figures

The ConnInfPy 2.0 manuscript figures (300 DPI) are built from locked
`examples/*/results/` outputs and `paper/data/` summary CSVs by two reproducible
scripts:

```bash
python paper/build_fig1_pipeline.py
python paper/build_manuscript_figures.py --group all   # all | sim | abide | openclose
```

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
- **Fig S5**: Panels A–B show the 176-cell $(E, H)$ hyperparameter sweep at
  $\Delta r = 0.125$ (`n=10`); Panel C shows the paired Monte Carlo difference
  (`Exact n=None` minus `n=10` grid, 95% CI) across $\Delta r \in \{0.075, 0.125, 0.25\}$.
