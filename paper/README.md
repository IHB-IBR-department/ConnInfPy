# Reproducing the manuscript figures

`paper/reproduce_figures.sh` regenerates every data figure and table in the
ConnInfPy manuscript (`PaperNN/main.tex`) from the scripts in `examples/`.

```bash
bash paper/reproduce_figures.sh              # all stages, ~1.5 h on an Apple M5 Pro
bash paper/reproduce_figures.sh abide        # one stage: abide | openclose | simulation | benchmarks | collect
```

Intermediate results go to `examples/*/results/`. The `collect` stage copies
the figures into `paper/figures/` (git-ignored), mirroring the manuscript's
`figs/` folders. Retired figure assets (an unused data-driven Fig 1 script and
old pipeline images) are kept locally in `paper/legacy/`.

Inputs: ABIDE I time series in `datasets/ABIDE_TS_Schaefer100/` (override
with `ABIDE_TS_DIR`), Open-Close data in `datasets/open_close/`; the
simulations and benchmarks need no data.

## Figure and table map

| Manuscript | Produced file | Script(s) |
|---|---|---|
| Fig `fig:pipeline` | `pipeline_firsted.pdf` — hand-drawn diagram, no data | — |
| Fig `fig:abide-global-alignment` + `fig:abide-site-effects` | `abide/plot1_combat_site_variance.png` (one figure; the manuscript splits it into two) | `abide_validation/plot_paper_figures.py` |
| Fig `fig:abide-age-networks` | `abide/plot3_network_block_mass.png` | `run_age_strategy_d.py` → `plot_paper_figures.py` |
| Table `tab:abide-dx`, Fig `fig:abide-dx` | `abide/plot6_diagnosis_methodology.png`; counts from `results/diagnosis/*_edges.csv` | `run_naive_baseline.py`, `run_dx_combat_only.py`, `run_dx_strategy_e.py`, `run_dx_site_glm.py` → `plot_diagnosis_quadrant.py` |
| Table `tab:method-spread` | `results/age_development/methods/*.npz` | `run_age_method_comparison.py` (+ `plot_method_sensitivity.py`) |
| Table `tab:tfnbs-multiverse`, Fig `fig:abide-multiverse` | `abide/plot5_tfnbs_grid_sensitivity.png`; `results/age_development/sensitivity/eh/` | `run_eh_sensitivity.py` |
| Table `tab:oc-within` | `results/{ihb,china}_paired_tfnbs*_edges.csv` | `openclose_validation/run_paired_tfnbs.py` |
| Table `tab:oc-cross`, Fig `fig:oc-replication` | `results/audit/openclose_agreement.csv`, `openclose/plot2_block_mass_convergence.png` | `audit_openclose_agreement.py` |
| Fig `fig:oc-specificity` | `openclose/plot5_retest_specificity.png` | `ml/run_ml_feature_selection.py`, `run_retest_methods.py` → `plot5_retest_specificity.py` |
| Table `tab:timing`, Fig `fig:timing` | `res_benchmarks/timing_benchmark.csv`, `figures/timing_benchmark.png` | `benchmarks/timing_benchmark.py` → `plot_results.py` |
| Fig `fig:fwer` | `simulation/fig1_fwer_calibration.png` | `fpr/fpr_calibration.py` → `regenerate_paper_figures.py` |
| Fig `fig:topologies` | `simulation/topologies_gallery_es0.3_n30.png` | `regenerate_paper_figures.py` |
| Tables `tab:tpr-matrix`, `tab:fdr-ranking`, Fig `fig:power-heatmap` | `results/power/power_analysis_1000_30/power_vs_effect_size.csv`, `simulation/fig2_power_heatmap_d0.25.png` (one figure; the manuscript splits TPR / FDR) | `power/power_analysis.py` → `regenerate_paper_figures.py` |
| Fig `fig:tfnbs-eh` | `simulation/fig4_tfnbs_eh_tpr_fdr.png` | `power/tfnbs_eh_sweep.py` → `regenerate_paper_figures.py` |

No script writes the LaTeX tables; their values are read from the CSV / NPZ
outputs above.

## Run time (measured on an Apple M5 Pro, 6 + 12 cores, 24 GB, branch `arch-refactor`)

| Stage | Work | Time |
|---|---|---|
| ABIDE | 14 GLM runs × 5,000 permutations (~66 s each: 10.9 ms/permutation + ~12 s setup), (E,H) multiverse, figures | ~18 min |
| Open-Close | ~27 paired runs (≤ 6 s each), ML transfer, agreement, figures | ~5–10 min |
| Simulation — FWER | 4,000 null datasets (1,000 × 4 sample sizes) × 11 methods × 1,000 permutations; 6.5–8.8 s per dataset single-process, run in parallel | ~38 min |
| Simulation — power | 900 datasets × 9 permutation methods (TFNBS families over the 176-cell (E,H) grid) × 1,000 permutations; ~28 s per dataset | ~33 min |
| Benchmarks | timing at 60/100/200/400 nodes, 3 repeats | ~3 min |
| **Total** | | **~1.5 h (1.3–2 h)** |

With the stored 100-null calibration instead of 1,000 nulls the total drops to
about 1 h, but that weakens the calibration claim (issue 2).

**Accelerations.** Already in place on `arch-refactor`: numba-compiled TFNBS,
one BLAS thread per permutation worker, workers no longer re-run the calling
script, single-pass (E,H) grids, and parallelism across simulated datasets.
The ABIDE GLM is limited by per-permutation linear algebra: multiprocessing
gives ~2× over the BLAS-threaded single process, and sending the data to each
worker only once did not help (measured). The remaining big lever is GPD tail
acceleration (200 instead of 5,000 permutations, ABIDE stage ~18 → ~5 min),
which approximates the p-values — use it for drafts only unless the manuscript
reports GPD. A quick draft pass (GPD for the empirical analyses, 100 nulls)
takes ~30–40 min.

## Issues to resolve before submission

Reproducibility

1. **Figures are restyled outside the repository.** All 12 data figures in
   `main.tex` are `*_alt` versions (serif font, shorter titles) that no script
   produces; the `plot1a`/`plot1b` and `fig2a`/`fig2b` splits are not scripted
   either. Port the restyle into the plotting scripts (one shared paper style)
   or point `main.tex` at the repository outputs. The ABIDE plots are saved at
   150 dpi; journals usually need ≥ 300.
2. **Null calibration does not match the text.** `main.tex`: 1,000 null
   datasets, n ∈ {30, 50, 100, 200}. Stored results: 100 nulls,
   n ∈ {10, 30, 50, 100}; the only 1,000-null run (`_archive/.../fpr_calibration_V1`)
   was made with pre-refactor code and also has n ∈ {10, 30, 50, 100}. The
   script defaults to 1,000 nulls with the stored sample sizes; fix the text or
   set `FPR_SAMPLE_SIZES="30 50 100 200"`.
3. **Power tables are from an earlier run.** The stored results reproduce
   `tab:tpr-matrix` and `tab:fdr-ranking` within 0.01–0.03 (e.g. NBS-intensity
   on hub 0.45 vs 0.48; NBS-extent FDP 0.282 vs 0.290). Regenerate the tables
   from the final run.
4. **Timing.** `tab:timing` says "8-core CPU"; re-measure and state the
   machine.
5. **Data provenance.** No script derives `abide_ts.npy` from the public ABIDE
   PCP release (the readme says "loaded using nilearn"); the IHB Open-Close
   cohort has no public archive yet. (`prepare_data.py` previously read a
   personal path; it now defaults to `datasets/ABIDE_TS_Schaefer100`.)

Scientific validity

6. **FWER is controlled per direction.** The calibration figure shows the
   per-tail FWER (≈ 0.05); the probability of a false finding in either
   direction is ≈ 0.09–0.11 in the same runs. The empirical tables report
   positive and negative edges, each at α = 0.05. State that FWER is
   controlled per direction, or use α/2 per tail for two-sided claims.
7. **ABIDE age model.** The text says age was adjusted for sex, motion *and
   diagnosis*; the reported 170 + 86 edges come from the model without
   diagnosis (`run_age_strategy_d.py --variant age`; the `age_dx` variant
   exists). The age map, the method comparison and the multiverse use ComBat +
   site dummies, while the text calls the site-dummies GLM primary. The
   multiverse uses yet another design (ComBat-harmonised data without site
   dummies or site-blocked permutation) and GPD with 200 permutations, neither
   of which is reported.
8. **ABIDE "naive" arm changes two things.** It drops the age / sex / motion
   covariates as well as site handling, so 520 vs 187 edges confounds
   covariate adjustment with site handling.
9. **Open-Close specificity figure.** The caption and text describe similarity
   to the primary map against a random baseline; the figure shows counts of
   significant edges. The "uniform-null expected α·E" line is the expectation
   for uncorrected tests, not for FWER-controlled methods, and the
   closed-vs-closed contrast is not a null (TFNBS 780, NBS τ = 2 3,860 edges),
   so it cannot serve as an empirical null.
10. **Open-Close block-mass measure.** `main.tex` says "block counts"; the
    correlations (0.79 / 0.61) are computed on summed −log₁₀ p
    (`result.network_summary(atlas)`, column `sum_neglog10p_*`). The
    2026-10-05 revision already says −log₁₀ p.
11. **TFNBS settings differ by experiment** — empirical (start threshold 1.65,
    n = 10), FWER calibration (1.65, n = 50), power (1.25, n = 30). FWER
    control holds for any setting under the max-statistic null, but the
    operating characteristics depend on them; report them per experiment.
12. **Fig 1 diagram.** Panel C says ComBat applies to both pipelines
    (`analyze()` skips it for two-sample designs); "Schaefer100-1000" is listed
    as available (bundled: 100 / 200 / 400).

Housekeeping (fixed): the simulation README's garbled first lines and the
stale comments in `sweep_config_power_1000.yaml`.
