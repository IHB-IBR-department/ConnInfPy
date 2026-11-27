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
(via `examples/abide/prepare_cc200.py`), Open-Close data in `datasets/demo/open_close/`,
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

3D BrainNet Viewer supplementary/companion renders (2400×1600, 200 DPI):
`abide/abide_cc200_dx_strategy_d_tfnbs_top150_ch2cb.png` (`BrainMesh_Ch2withCerebellum.nv`, 6-view),
`openclose/open_close_ihb_tfnbs_top150.png`, and `openclose/open_close_china_tfnbs_top150.png` (ICBM152, 4-view).

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

## Calibration and power tables (v2.0)

Compact, auditable summaries of the 1,000-null 2.0 runs are committed under
`examples/simulations/results/tables/`; the per-run CSVs they are derived from
stay local (git-ignored) in `examples/simulations/results/`.

| Table | Rows | Derived from | Settings |
|---|---|---|---|
| `v2_glm_fwer_summary.csv` | 9 methods × n ∈ {30, 60, 100} | `glm/glm_validation_2_0_1000/glm_fwer_summary.csv` | GLM, `two_tailed=True`, exact integration `n=None`, E = 0.4, H = 3.0, 1,000 permutations |
| `v2_glm_power_summary.csv` | 9 methods × 5 topologies × Δ ∈ {0.1, 0.2, 0.3} | `glm/glm_validation_2_0_1000/glm_power_summary.csv` | n = 60, 30 repeats, same inference settings |
| `v2_pairwise_fpr_summary.csv` | 5 methods × n ∈ {10, 30, 50, 100} | `fpr_calibration_tfnbs_exact/`, `fpr_calibration_ni_fbc_exact/`, `fpr_calibration_nbs_int_v2/` | two-sample, per-tail nulls, 1,000 permutations |
| `v2_exact_power_summary.csv` | 4 methods × 10 topologies × Δr ∈ {0.075, 0.125, 0.25} | `power/power_tfnbs_exact_04_30/`, `power/power_ni_fbc_exact_04_30/` | n = 30, 100 repeats; `arm` marks exact (`n=None`) vs the legacy 10-point grid |

`v2_tables_provenance.json` lists every source file with its SHA-256 and
modification time, the run parameters, the column semantics and the GLM run's
recorded commit (a pre-rewrite object whose tree matches the current main
commit carrying the same message). The three October 2026 tables in the same
directory predate the 2.0 contract and are kept unchanged.

Two points matter when reading the FWER tables:

- **`fwer_two_sided_joint` is not `fwer_max_tail`.** In the GLM table
  `fwer_two_sided_joint` is the share of null runs with any edge of either
  sign at `min(p_pos, p_neg) ≤ α`, both p-maps being corrected against the
  joint max-null. It is the FWER of the single two-sided family that the 2.0
  contract holds at α (0.039–0.065 over the 27 cells). `fwer_pos`, `fwer_neg`
  and `fwer_max_tail` are one-directional diagnostics (`fwer_max_tail`
  0.021–0.039) and `fwer_two_sided_bonf` the Bonferroni-over-tails one (α/2 per
  tail; 0.013–0.036).
- **The pairwise FPR and power runs predate the two-sided default.** They never
  passed `two_tailed=` and ran while the default was `False`, so they carry no
  joint column: `fwer_max_tail` is judged against α per tail (0.045–0.063),
  `fwer_two_sided_bonf` against α (0.032–0.061), and `fwer_either_at_alpha` is
  the diagnostic with expected upper bound 2α (0.081–0.110). The `*_exact`
  methods force `n=None`; the `n = 10` printed in those runs' text headers is
  the configuration value that those methods ignore.

`scripts/ci_fwer_check.py` (nightly and on manual dispatch through
`.github/workflows/fwer-calibration.yml`, never on pull requests) re-checks the
joint two-sided FWER of `tstat`, exact TFNBS, NBS (τ = 3.0), cNBS and the
multi-site GLM path on 200 null datasets × 199 permutations at N = 20 in about
20 s, and fails when any arm exceeds its exact 99.5 % binomial bound.

## Open-Close: Beijing repeat run and scan order

The Beijing EOEC cohort has two eyes-closed runs per participant. The Fig S3
repeat-run contrast (EC₁ − EC₀, 47 pairs) yields 556 positive and 416 negative
edges in the 2.0 pipeline (972 edges;
`paper/data/openclose_agreement_summary.csv`). This contrast does not serve as an
empirical null. FWER control at level α bounds the probability of any false
positive under the complete null, Pr(V ≥ 1) ≤ α: at most about 5 % of
exchangeable null datasets yield even one significant edge. It does not bound
the number of edges flagged in a single run — with dependent edges, one
significant component can contain hundreds of edges — so the 972 edges here
are not by themselves evidence of miscalibration, but the contrast cannot be
used as a null reference. Its spatial agreement with the state
contrast EC₀ − EO (1,919 / 1,344 edges) is close to zero or negative (edge
Jaccard 0.004 / 0.001, Spearman −0.10 / −0.12, block-mass Pearson +0.02 / −0.11),
so it does not reproduce the eyes-closed pattern. Session or vigilance drift
between the two runs is a plausible hypothesis for the EC₁ − EC₀ difference,
not an established cause.

Scan order is a known confound of the primary China contrast. Beijing scan 1 is
always eyes closed and precedes eyes open; scans 2 and 3 are eyes open / eyes
closed in random order. Close run 0 (EC₀, the paper contrast) is scan 1 and is
therefore always acquired before eyes open (the loader's naming of the
incomplete run of `sub-3258811` as "close second run" fixes the run
numbering), whereas close run 1 (EC₁) is counterbalanced with eyes open. The
sensitivity arms `openclose_china_run1` (EC₁ − EO) and `openclose_china_run0_n47`
(EC₀ − EO on the same 47 participants) are defined in
`examples/open_close/brainnetome_interpretation.py`. In the October 2026
Brainnetome analysis (paired TFNBS, 10-point threshold grid, 5,000
permutations, α = 0.05 per tail) they gave 1,795 / 1,247 (EC₁) versus 2,216 /
857 (EC₀) edges, and the whole-map correlation between the IHB and China
EC − EO maps rose from 0.38 with EC₀ to 0.49 with EC₁. EC₁ agrees better with
IHB; the paper's primary China contrast keeps EC₀, and the two definitions
should be read together.

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
