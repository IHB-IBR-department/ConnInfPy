# Reproducing the manuscript figures

`paper/reproduce_figures.sh` regenerates every data figure and table in the
ConnInfPy manuscript (`PaperNN/main.tex`) from the scripts in `examples/`.

```bash
bash paper/reproduce_figures.sh              # all stages, ~3 h on an Apple M5 Pro
bash paper/reproduce_figures.sh simulation   # one or more of: abide openclose simulation sensitivity benchmarks collect
```

Intermediate results go to `examples/*/results/`. The `collect` stage copies
the figures and tables into `paper/figures/` (git-ignored), mirroring the
manuscript's `figs/` folders. Retired figure assets are kept locally in
`paper/legacy/`. Every simulation step resumes from its saved results
(`--resume`), so re-running a finished stage costs seconds.

Inputs: ABIDE I time series in `datasets/ABIDE_TS_Schaefer100/` (override
with `ABIDE_TS_DIR`), Open-Close data in `datasets/open_close/`; the
simulations and benchmarks need no data.

## Figure and table map

| Manuscript | Produced file | Script(s) |
|---|---|---|
| Fig `fig:pipeline` | `pipeline_firsted.pdf` — hand-drawn diagram, no data | — |
| Fig `fig:abide-global-alignment` + `fig:abide-site-effects` | `abide/plot1_combat_site_variance.png` (one figure; the manuscript splits it into two) | `abide_validation/plot_paper_figures.py` |
| Fig `fig:abide-age-networks` | `abide/plot3_network_block_mass.png` | `run_age_strategy_d.py --variant age` → `plot_paper_figures.py` |
| Table `tab:abide-dx`, Fig `fig:abide-dx` | `abide/plot6_diagnosis_methodology.png`; counts from `results/diagnosis/*_edges.csv` | `run_naive_baseline.py`, `run_dx_combat_only.py`, `run_dx_strategy_e.py`, `run_dx_site_glm.py` → `plot_diagnosis_quadrant.py` |
| Table `tab:method-spread` | `results/age_development/methods/*.npz` | `run_age_method_comparison.py` (+ `plot_method_sensitivity.py`) |
| Table `tab:tfnbs-multiverse`, Fig `fig:abide-multiverse` | `results/age_development/sensitivity/eh/multiverse_summary.csv` (+ pairwise matrices); `abide/plot5_tfnbs_grid_sensitivity.png` (counts per cell; the manuscript figure shows a Spearman matrix, which no script plots) | `run_eh_sensitivity.py` |
| Table `tab:oc-within` | `results/{ihb,china}_paired_tfnbs*_edges.csv` | `openclose_validation/run_paired_tfnbs.py` |
| Table `tab:oc-cross`, Fig `fig:oc-replication` | `results/audit/openclose_agreement.csv`, `openclose/plot2_block_mass_convergence.png` | `audit_openclose_agreement.py` |
| Fig `fig:oc-specificity` | `openclose/plot5_retest_specificity.png` | `ml/run_ml_feature_selection.py`, `run_retest_methods.py` → `plot5_retest_specificity.py` |
| Table `tab:timing`, Fig `fig:timing` | `res_benchmarks/timing_benchmark.csv`, `figures/timing_benchmark.png` | `benchmarks/timing_benchmark.py` → `plot_results.py` |
| Fig `fig:fwer` | `simulation/fig1_fwer_calibration.png`; verdict in `results/fpr_calibration/fpr_summary.txt` | `fpr/fpr_calibration.py` → `regenerate_paper_figures.py` |
| Fig `fig:topologies` | `simulation/topologies_gallery_dr0.125_n30.png` | `regenerate_paper_figures.py` |
| Table `tab:tpr-matrix` | `tables/tpr_matrix_dr0.125.csv` | `power/power_analysis.py` → `regenerate_paper_figures.py` |
| Table `tab:fdr-ranking` | `tables/fdr_ranking_dr0.25.csv` | same |
| Fig `fig:power-heatmap` | `simulation/fig2_power_heatmap_dr0.125.png` (one figure; the manuscript splits TPR / FDR) | same |
| Fig `fig:tfnbs-eh` | `simulation/fig4_tfnbs_eh_tpr_fdr.png` | `power/tfnbs_eh_sweep.py` → `regenerate_paper_figures.py` |
| Robustness text (`sec:evaluation-robustness`, appendix on NI normalization) | `results/sensitivity/{prior_misspec,mmin,ablation}/comparison/`, `simulation/fig5_prior_misspecification.png`, `simulation/mmin_sensitivity.png` | `sensitivity` stage |

`tables/power_by_method_topology.csv` holds every method × topology × effect
cell (mean and Monte-Carlo SE of TPR, FDP and strict FDP).

## Run time (measured on an Apple M5 Pro, 6 + 12 cores, 24 GB)

| Stage | Work | Time |
|---|---|---|
| ABIDE | 12 GLM runs × 5,000 permutations (~66 s each), (E,H) multiverse, figures | ~15 min |
| Open-Close | ~27 paired runs (≤ 6 s each), ML transfer, agreement, figures | ~5–10 min |
| Simulation — FWER | 4,000 null datasets × 11 methods × 1,000 permutations | 55 min |
| Simulation — power | 3,000 datasets (10 topologies × 3 Δr × 100) × 11 methods × 1,000 permutations; TFNBS families over the 176-cell (E,H) grid | 73 min |
| Simulation — (E,H) summaries, figures, tables | | ~1 min |
| Sensitivity | 12 sweeps × 800 datasets (prior misspecification, m_min, NI normalization, permutation BH) | 12 min |
| Benchmarks | timing at 60/100/200/400 nodes, 3 repeats | ~3 min |
| **Total** | | **~2 h 50 min** |

`power_analysis.py` prints a runtime estimate before it starts (calibrated on
this run: 71 min predicted vs 73 min measured for the power stage).

**Accelerations.** In place: numba-compiled TFNBS, one BLAS thread per
worker, single-pass (E,H) grids, parallelism across simulated datasets, and
checkpoints throttled to one full write every 5 minutes (writing every 10
tasks made the 1.6 M-row power run quadratic in I/O). The ABIDE GLM is
limited by per-permutation linear algebra; the remaining lever there is GPD
tail acceleration (200 instead of 5,000 permutations), which approximates the
p-values — drafts only unless the manuscript reports GPD. A quick draft pass
(`FPR_N_NULL=100`, GPD for the empirical analyses) takes ~40 min.

## Simulation benchmark: what changed (2026-10-06/07)

All simulation results were recomputed after the fixes below; the earlier
results are archived in `examples/simulation_validation/results/_archive/pre_fix_2026-10-06/`.

1. **Planted effects were applied at half strength** (commit `cc9ef6c`). Masks
   filled in one triangle were symmetrised by averaging with the empty
   triangle. Every earlier power, FDP, (E,H) and sensitivity number was
   therefore obtained at half the stated effect.
2. **Effect sizes are Δr, not Cohen's d.** The effect is a shift of the
   population correlation on the planted edges. The new values 0.075 /
   0.125 / 0.25 equal the *realised* effects of the earlier runs (stated
   "d" = 0.15 / 0.25 / 0.50), so the regimes are comparable. Each subject's
   matrix is the sample correlation of 30 time points, so per planted edge
   Δr = 0.075 / 0.125 / 0.25 is about Cohen's d = 0.4 / 0.7 / 1.4 on the
   Fisher-z values.
3. **NBS τ = 3 was never run in the power study.** `nbs_extent_3.0` and
   `nbs_intensity_3.0` fell back to the default τ = 2, so their columns
   duplicated τ = 2. `tab:tpr-matrix` uses τ = 2 only, but the heatmap
   figure showed the wrong τ = 3 columns.
4. **Effect leaking onto unplanted edges.** When the shifted correlation
   matrix is not positive definite (Δr = 0.25 in the hub, bipartite,
   scattered and fragmented topologies) the projection moves some unplanted
   edges by up to ~0.06 and attenuates the planted ones (hub: 0.18 instead
   of 0.25). Edges moved by more than 0.01 are now left out of TP/FP
   scoring; `FDR_strict` counts their detections as false. At Δr ≤ 0.125
   no edge is moved by more than 0.01.
5. **Failures were recorded as "nothing significant"** in the FWER, power
   and GLM scripts, which would bias FWER down and lower TPR. They now stop
   the run.
6. **One TFNBS setting everywhere:** E = 0.4, H = 3, 10 thresholds from
   1.65 — as in the empirical analyses. The power and FWER studies used
   start 1.25 / 30 thresholds and start 1.65 / 50 thresholds before. Both
   simulations now use 30 time points per subject (the null used 200).
7. **Scale:** 1,000 null datasets per cell (was 100), 100 power repeats per
   topology × effect (was 30), sensitivity sweeps at 100 repeats and 1,000
   permutations (were 10 and 100). Significance is `p ≤ α` throughout.
8. **FWER figure.** Panel (a) shows one pre-specified direction
   (group 2 > group 1) — the earlier "worse tail" estimate is biased upwards
   by about 0.004. Panel (b) shows the two-sided FWER with α/2 per tail.
   The permutation max-t method was mislabelled "uncorrected". The
   calibration verdict is now a simultaneous test (exact binomial,
   Bonferroni over cells); per-cell intervals are descriptive.
9. `effect_size = 0` is now an exact null in `TopologyDatasetGenerator`
   (per-edge effect noise used to be added even without an effect; the FWER
   study was not affected because it uses an empty mask).

## New simulation results (all at n = 30 per group, α = 0.05)

**Null calibration** (`results/fpr_calibration/fpr_summary.txt`; 1,000 null
datasets per cell, n ∈ {10, 30, 50, 100}, 11 methods = 44 cells).

| Family | FWER, pooled | Range over cells | Simultaneous test |
|---|---|---|---|
| One direction, α per tail | 0.047 | 0.025–0.065 | smallest exact binomial p = 0.021; Bonferroni over 88 tail-cells: 1.0 → pass |
| Two-sided, α/2 per tail | 0.047 | 0.027–0.061 | smallest p = 0.067; Bonferroni over 44 cells: 1.0 → pass |
| Either tail at α (no correction) | 0.092 | 0.059–0.109 | expected ≈ 2α — not a valid two-sided procedure |

One of the 88 directional intervals lies above 0.05 (FBC-TFNBS, n = 30,
negative tail, 0.065 [0.051, 0.082]) — about two are expected by chance.

**TPR at Δr = 0.125** (`tables/tpr_matrix_dr0.125.csv`; replaces
`tab:tpr-matrix`, columns in the manuscript's order; NBS τ = 2, TFNBS
E = 0.4, H = 3; max-t is the permutation max-statistic test).

| Method | wm-d | bm-d | hub | rich | scat | chain | grad | p-bip | xb-ch | frag |
|---|---|---|---|---|---|---|---|---|---|---|
| cNBS | 0.99 | 0.98 | 0.08 | 0.04 | 0.04 | 0.01 | 0.75 | 0.67 | 0.01 | 0.99 |
| NBS-int | 0.76 | 0.69 | 0.33 | 0.47 | 0.53 | 0.21 | 0.11 | 0.29 | 0.21 | 0.34 |
| NBS-ext | 0.73 | 0.69 | 0.24 | 0.27 | 0.35 | 0.14 | 0.07 | 0.19 | 0.16 | 0.21 |
| FBC-TFNBS | 0.56 | 0.48 | 0.24 | 0.20 | 0.16 | 0.08 | 0.09 | 0.36 | 0.18 | 0.47 |
| TFNBS | 0.42 | 0.35 | 0.23 | 0.27 | 0.18 | 0.11 | 0.04 | 0.20 | 0.10 | 0.30 |
| NI-TFNBS | 0.37 | 0.27 | 0.16 | 0.16 | 0.15 | 0.13 | 0.04 | 0.19 | 0.13 | 0.30 |
| max-t | 0.17 | 0.10 | 0.11 | 0.12 | 0.12 | 0.11 | 0.02 | 0.10 | 0.10 | 0.16 |

The pattern of the old table holds (it was run at the same realised effect).
Newly available: NBS at τ = 3 (hub 0.40, rich 0.43, frag 0.47, chain 0.07)
and BH-FDR (0.04–0.46).

**Mean projected-edge FDP at Δr = 0.25** over the ten topologies
(`tables/fdr_ranking_dr0.25.csv`; replaces `tab:fdr-ranking`; strict = leak
edges counted as false).

| Method | FDP | strict FDP | mean TPR |
|---|---|---|---|
| max-t | 0.002 | 0.002 | 0.69 |
| Bonferroni | 0.002 | 0.002 | 0.68 |
| NI-TFNBS | 0.004 | 0.004 | 0.79 |
| TFNBS | 0.019 | 0.019 | 0.83 |
| FBC-TFNBS | 0.020 | 0.021 | 0.83 |
| BH-FDR | 0.047 | 0.048 | 0.86 |
| NBS-extent, τ = 3 | 0.049 | 0.051 | 0.87 |
| NBS-intensity, τ = 3 | 0.049 | 0.051 | 0.87 |
| NBS-extent, τ = 2 | 0.261 | 0.263 | 0.57 |
| cNBS | 0.405 | 0.421 | 0.60 |
| NBS-intensity, τ = 2 | 0.419 | 0.423 | 0.90 |

Old table: t-stat 0.002, NI 0.005, FBC 0.023, TFNBS 0.026, NBS-ext 0.290,
cNBS 0.422, NBS-int 0.427 — same ranking. The leak edges matter mainly for
cNBS, whose block decision spreads to them (partial bipartite: 0.63 vs 0.72
strict).

**Analytical check.** BH controls FDR at π₀·α (equality under independence,
≤ under positive dependence), where π₀ is the fraction of null edges. Here
π₀·α = 0.048 on average; the simulated BH FDR is 0.044 / 0.051 / 0.047 at
Δr = 0.075 / 0.125 / 0.25, and 26 of the 30 topology × effect cells are
within 2 SE of π₀·α (the other four lie below it, as positive dependence
allows). No closed form exists for the cluster and block methods, hence the
simulation. The FDP intervals above are the normal (CLT) interval of a mean,
mean ± 1.96·SD/√100; distribution-free bounds (Hoeffding, empirical
Bernstein) are about ±0.1 at 100 repeats and too wide to be useful.

**(E,H) grid** (176 cells: E ∈ 0.05–1.5, H ∈ 0.5–10). Criterion as in the
manuscript: mean FDP over the 100 datasets ≤ 0.05 in each of the five
plotted topologies, with non-zero TPR. On the point estimates, 71 cells meet
it at Δr = 0.125 (74 at Δr = 0.25; 14 in the old run). Allowing for
Monte-Carlo error (95% normal interval of each mean; SE 0.003–0.02):

| Cells (of 176) | Δr = 0.125 | Δr = 0.25 |
|---|---|---|
| upper bound ≤ 0.05 in all five (shown ≤ 0.05) | 66 | 70 |
| lower bound > 0.05 in some topology (shown above) | 92 | 96 |
| undecided | 18 | 10 |

(0.4, 3) is undecided at Δr = 0.125 — hub FDP 0.040, upper bound 0.054 — and
shown ≤ 0.05 at Δr = 0.25 (hub 0.035, upper bound 0.044). Cells along
E ≈ H/10, (0.6, 6) … (0.9, 9), have about the same FDP and mean TPR
(0.28–0.30 vs 0.28 at Δr = 0.125).

**Robustness** (4 topologies: hub, rich club, chain, between-module dense;
`results/sensitivity/`).

| Mean over topologies | Δr | correct (4 blocks) | 7 blocks | 2 blocks |
|---|---|---|---|---|
| cNBS FDP | 0.125 / 0.25 | 0.14 / 0.43 | 0.28 / 0.67 | 0.32 / 0.66 |
| NI-TFNBS FDP | 0.125 / 0.25 | 0.012 / 0.004 | 0.008 / 0.003 | 0.020 / 0.011 |
| FBC-TFNBS FDP | 0.125 / 0.25 | 0.066 / 0.027 | 0.072 / 0.037 | 0.074 / 0.043 |
| FBC-TFNBS TPR | 0.125 / 0.25 | 0.25 / 0.83 | 0.21 / 0.66 | 0.32 / 0.90 |

- FBC-TFNBS m_min ∈ {1, 2, 3, 5, 10}, mean TPR / FDP at Δr = 0.125:
  0.19/0.014, 0.25/0.045, 0.25/0.066, 0.24/0.084, 0.20/0.093. So m_min = 2
  is as sensitive as 3 with lower FDP, and FDP grows with m_min.
- NI-TFNBS normalization (√|B| / 1/|B| / none) changes TPR by ≤ 0.04 and FDP
  by ≤ 0.01, except on between-module dense (TPR 0.27 / 0.17 / 0.37 at
  Δr = 0.125). FDP ≤ 0.022 in every variant.
- Permutation BH-FDR at 1,000 permutations has low power (TPR 0.09 vs 0.29
  for parametric BH at Δr = 0.125). Its smallest attainable p (~0.001) is
  above BH's first thresholds over 1,770 edges. It is not used in the
  manuscript.

## Empirical analyses: reproduction check (2026-10-07)

ABIDE and Open-Close were re-run end to end with the current code and
compared with the stored results and the manuscript.

**Reproduce exactly** (edge sets identical; p-values within 0.005 from the
refactored permutation engine): ABIDE age 170 + 86; diagnosis naive 520,
site dummies 187, ComBat + site dummies 187; the seven-method age table
(35 / 61 / 117 / 239 / 256 / 440 / 892); Open-Close within-cohort counts
(939 / 695 / 2,220 / 898), cross-cohort agreement (Jaccard 0.148 / 0.159,
Spearman 0.25 / 0.26, block-mass r 0.79 / 0.61) and the negative control
(Jaccard 0.006 / 0.000 against random 0.023 / 0.015).

Fixed on the way:

1. **TFNBS parameters were inherited from a changed default.** The package
   default E moved from 0.4 to 0.3 on 2026-05-25 (`e565d65`). The four
   diagnosis scripts, NI-/FBC-TFNBS in the method comparison and FBC-TFNBS
   in three Open-Close scripts passed no E and silently switched to 0.3
   (diagnosis naive 520 → 280, NI 61 → 48, FBC 117 → 105). All paper
   scripts now pass (E, H, n) = (0.4, 3, 10).
2. **The `abide` stage always stopped** at a BH-FDR call the GLM pipeline
   cannot run (not in the manuscript); removed.
3. **Unreported GPD p-values.** The stored diagnosis results used GPD tail
   acceleration (the old `analyze()` default). With exact permutation
   p-values (5,000 permutations) three arms are unchanged, but
   **ComBat-only gives 315 (53 + 262), not 177 (53 + 124)**. See open issue
   10 for why GPD moved this decision.
4. **The (E,H) multiverse was not reproducible and used a different
   design.** No script produced the Spearman matrices behind
   `tab:tfnbs-multiverse` (files from 2026-05-24), and the sweep used
   ComBat data without site dummies or within-site permutation, with GPD at
   200 permutations. `run_eh_sensitivity.py` now uses the age map's design
   and exact p-values (5,000 permutations) and writes the matrices; its
   (0.4, 3.0) cell reproduces the age map (TFNBS 170 + 86, NI 61, FBC 117).

New multiverse results (TFNBS, 36 cells, 630 pairs):

| | positive | negative |
|---|---|---|
| significant edges per cell | 32–527 (median 248) | 13–344 (median 86) |
| edges at p = 1 per cell | 89–95% | 93–96% |
| Spearman of −log₁₀ p, all edges | median 1.000 [0.690, 1.000] | 1.000 [0.747, 1.000] |
| Spearman, edges with p < 1 in either cell | median 0.963 [0.609, 1.000] | 0.994 [0.910, 1.000] |
| Jaccard of significant sets | median 0.47 [0.06, 1.00] | 0.30 [0.04, 1.00] |

The all-edge Spearman is close to 1 mostly because ~90% of edges tie at
p = 1 in every cell; the below-ceiling Spearman and the Jaccard describe
the agreement better.

## Manuscript text to update

Simulation section (`sec:evaluation`):

1. **Null calibration (l. 484–486).** n ∈ {30, 50, 100, 200} → {10, 30, 50,
   100}; "nine prespecified method variants" → eleven; add 30 time points
   per subject and TFNBS (0.4, 3, 10 thresholds from 1.65). Replace the
   "[0.033, 0.063], 36 intervals" sentence with the pooled values and the
   simultaneous test above. State that FWER is controlled per direction and
   that two-sided claims need α/2 per tail (panel b).
2. **Generator (l. 497–499).** The two-sample benchmark does not inject
   effects after the Fisher transform. It shifts group 2's *population*
   correlation by Δr on the planted edges (per-edge variability SD ≈ 0.035),
   projects to the nearest positive-definite correlation matrix when needed,
   and draws each subject's matrix as the sample correlation of 30
   multivariate-normal time points. It is generative, not an edge-feature
   benchmark. Edges moved by > 0.01 in the projection are not scored. The
   Fisher-z injection caveat (l. 668) applies only to the GLM benchmark.
3. **Effect sizes.** Every "d" in the simulation section → Δr:
   gallery d = 0.3 → Δr = 0.125 (l. 501); TPR table and heatmaps d = 0.25 →
   Δr = 0.125 (l. 516, 537); FDR table d = 0.50 → Δr = 0.25 (l. 550); (E,H)
   d = 0.25 → Δr = 0.125 (l. 570, 577). Optionally give the per-edge Cohen's
   d (0.4 / 0.7 / 1.4).
4. **Tables.** Replace `tab:tpr-matrix` and `tab:fdr-ranking` values with
   the tables above. Consider adding NBS τ = 3 and BH-FDR rows; the heatmap
   already shows all 11 variants. "t-stat" is the permutation max-t test.
5. **FDP narrative (l. 544–546, bullets l. 508–512).** cNBS 0.422 → 0.405.
   The NBS-intensity absorption of null edges is a τ = 2 property: at τ = 3,
   NBS has FDP 0.05 with TPR 0.87 at Δr = 0.25.
6. **(E,H) (l. 570).** "36 tested combinations" → 176. "(0.4, 3.0) was the
   only tested configuration with FDP ≤ 0.05 in all five topologies" is
   false: 71 of 176 cells meet it on point estimates (14 already in the old
   run), 66 with the upper 95% bound ≤ 0.05; at Δr = 0.125, (0.4, 3.0)
   itself is not distinguishable from 0.05 on the hub topology (0.040,
   upper bound 0.054). Say instead that (0.4, 3.0) was fixed a priori from
   Hao et al., lies in a broad region of settings with mean FDP ≤ 0.05, and
   report the hub FDP with its interval.
7. **Robustness (l. 572–573).** Supported; numbers above. Add that m_min = 2
   performed as well as 3 with lower FDP.
8. **Appendix, NI normalization rationale (l. 709).** Not supported by the
   ablation. The "no normalization" variant does not show elevated FDP;
   1/|B| does not approach plain TFNBS. NI-TFNBS (any normalization) is less
   sensitive than plain TFNBS on the module-aligned dense topologies
   (between-module dense 0.27 vs 0.35, within-module dense 0.37 vs 0.42),
   so the block prior does not add sensitivity there. Rewrite as "the
   normalization has little effect in the tested regimes; √|B| is kept as
   the default", or drop the paragraph.

ABIDE section:

9. **Age model (l. 291).** "adjusting for sex, head motion, and diagnosis" →
   "adjusting for sex and head motion". Age and diagnosis are separate
   analyses. The reported 170 + 86 edges and the seven-method comparison use
   this model and are unchanged.
10. **Diagnosis, ComBat-only arm (l. 300, `tab:abide-dx`).** 177 (53 + 124) →
    315 (53 + 262). The other three arms are unchanged. The count is
    fragile: 138 of the 262 negative edges form one TFNBS cluster with a
    shared exact p of 0.0496 (SE 0.003), so a different permutation draw
    could put it on either side of 0.05. Worth a sentence, or report this
    arm at block level.
11. **Multiverse (l. 347, `tab:tfnbs-multiverse`).** Replace the table with
    the numbers above (state the design: same model as the age map, exact
    p-values, 5,000 permutations). "Positive-tail minimum correlation 0.812"
    → 0.609 (edges below p = 1) or 0.690 (all edges). Report the count
    range and the Jaccard — they show how much the thresholded maps move.
    The figure caption describes a Spearman matrix; the repository figure
    shows counts per cell.
12. **Methods.** State that all empirical p-values are exact permutation
    p-values (5,000 permutations, no tail approximation) and that every
    TFNBS-family operator uses (E, H, n) = (0.4, 3, 10).

## Issues still open

1. **Figures are restyled outside the repository.** All 12 data figures in
   `main.tex` are `*_alt` versions (serif font, shorter titles) that no script
   produces; the `plot1a`/`plot1b` and `fig2a`/`fig2b` splits are not scripted
   either. Port the restyle into the plotting scripts (one shared paper style)
   or point `main.tex` at the repository outputs. The ABIDE plots are saved at
   150 dpi; journals usually need ≥ 300 (the simulation plots are now 300).
2. **Timing.** `tab:timing` says "8-core CPU"; re-measure and state the
   machine.
3. **Data provenance.** No script derives `abide_ts.npy` from the public ABIDE
   PCP release (the readme says "loaded using nilearn"); the IHB Open-Close
   cohort has no public archive yet.
4. **ABIDE site strategy.** The age map, the method comparison and the
   multiverse use ComBat + site dummies (Strategy D), while the text calls
   the site-dummies GLM primary (the multiverse now matches the age map).
5. **ABIDE "naive" arm changes two things.** It drops the age / sex / motion
   covariates as well as site handling, so 520 vs 187 edges confounds
   covariate adjustment with site handling.
6. **Open-Close specificity figure.** The caption and text describe similarity
   to the primary map against a random baseline; the figure shows counts of
   significant edges. The "uniform-null expected α·E" line is the expectation
   for uncorrected tests, not for FWER-controlled methods, and the
   closed-vs-closed contrast is not a null (TFNBS 780, NBS τ = 2 3,860 edges),
   so it cannot serve as an empirical null.
7. **Open-Close block-mass measure.** `main.tex` says "block counts"; the
   correlations (0.79 / 0.61) are computed on summed −log₁₀ p
   (`result.network_summary(atlas)`, column `sum_neglog10p_*`).
8. **Fig 1 diagram.** Panel C says ComBat applies to both pipelines
   (`analyze()` skips it for two-sample designs); "Schaefer100-1000" is
   listed as available (bundled: 100 / 200 / 400).
9. **GLM calibration (optional).** The null calibration covers the
   two-sample permutation test only, while the ABIDE analyses rely on the
   Freedman–Lane GLM with covariates. `glm/glm_validation.py` provides a
   GLM null calibration; its stored results predate the effect-size fix and
   it is not in the manuscript. Either run it (1,000 nulls; run time not yet measured) and
   report it, or drop the GLM caveat paragraph (l. 668).
10. **GPD tail approximation (package, `acceleration.py`).** `fit_gpd_tail`
    starts the threshold at the median of the max-statistic null
    (`np.linspace(0.50, 0.90, 5)`), not at the upper quartile its docstring
    gives (citing Winkler et al., 2016). With thousands of exceedances the
    first candidate passes the Anderson–Darling check, so the GPD is fitted
    to the upper half of the null maxima. It then replaces *every* p-value
    above that threshold — including p ≈ 0.05, which 5,000 permutations
    estimate directly from ~250 exceedances (Monte-Carlo SE 0.003). On the
    ABIDE ComBat-only negative tail the fitted tail is 0.057 where the
    empirical tail is 0.050 (+14%). That moved one 138-edge cluster with an
    exact p of 0.0496 to 0.056, hence 262 vs 124 edges. Suggested fix: start
    at the 0.75 quantile and use the GPD only where fewer than ~10 null
    maxima exceed the observed statistic (Knijnenburg et al., 2009), keeping
    the empirical p-value elsewhere; on this arm that returns the exact
    result (262). The paper no longer uses GPD.
11. **Package default E = 0.3 vs the paper's E = 0.4.** The paper scripts now
    pass E explicitly; decide whether the package default should match the
    setting the paper recommends. `openclose_validation/run_eh_sensitivity.py`
    (not in the manuscript) still uses GPD with 200 permutations.

Resolved by this revision: null calibration design and size, per-direction
vs two-sided FWER, power tables from the final run, TFNBS settings
differing by experiment, the ABIDE age-model description (text change
only), the multiverse design, and unreported GPD in the empirical results.
