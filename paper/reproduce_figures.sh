#!/usr/bin/env bash
# Reproduce every data figure and table in the ConnInfPy manuscript
# (PaperNN/main.tex). See paper/README.md for the figure -> script map,
# expected run times and open issues.
#
# Usage (from anywhere in the repository):
#   bash paper/reproduce_figures.sh                 # everything (~3 h on an M5 Pro)
#   bash paper/reproduce_figures.sh abide collect   # selected stages
# Stages: abide | openclose | simulation | sensitivity | benchmarks | collect | all
#
# Results go to examples/*/results/; `collect` copies the figures into
# paper/figures/ under the manuscript's file names (without the `_alt`
# restyle suffix).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"

# Null-calibration design (override via environment, e.g. FPR_N_NULL=100 for
# a quick pass).
FPR_N_NULL="${FPR_N_NULL:-1000}"
FPR_SAMPLE_SIZES="${FPR_SAMPLE_SIZES:-10 30 50 100}"

step() { printf '\n=== %s ===\n' "$*"; }

abide() {
  step "ABIDE: QC + connectivity (764 subjects, 15 sites)"
  $PY examples/abide_validation/prepare_data.py
  # Age and diagnosis are separate analyses: the age model adjusts for sex and
  # motion only (not diagnosis).
  step "ABIDE: age (sex + motion covariates), ComBat + site dummies, 5000 permutations"
  $PY examples/abide_validation/run_age_strategy_d.py --variant age
  step "ABIDE: diagnosis under four site strategies, 5000 permutations each"
  $PY examples/abide_validation/run_naive_baseline.py
  $PY examples/abide_validation/run_dx_combat_only.py
  $PY examples/abide_validation/run_dx_strategy_e.py
  $PY examples/abide_validation/run_dx_site_glm.py
  # Every TFNBS-family call passes (E, H, n) = (0.4, 3, 10) explicitly: the
  # package default E is 0.3 since 2026-05-25. All p-values are exact
  # permutation p-values (no GPD tail approximation).
  step "ABIDE: seven-method comparison on age, 5000 permutations each"
  $PY examples/abide_validation/run_age_method_comparison.py
  step "ABIDE: TFNBS (E,H) multiverse, 36 cells x 3 methods, same design as the age map, 5000 permutations"
  $PY examples/abide_validation/run_eh_sensitivity.py
  step "ABIDE: figures"
  $PY examples/abide_validation/plot_paper_figures.py
  $PY examples/abide_validation/plot_method_sensitivity.py
  $PY examples/abide_validation/plot_diagnosis_quadrant.py   # last: owns plot6
}

openclose() {
  step "Open-Close: paired TFNBS (IHB, Beijing, Beijing EC-run retest), 5000 permutations"
  $PY examples/openclose_validation/run_paired_tfnbs.py --analyses ihb china china_retest
  step "Open-Close: selector p-maps (8 methods x 2 cohorts, 5000 permutations) + ML transfer"
  $PY examples/openclose_validation/ml/run_ml_feature_selection.py
  step "Open-Close: retest maps for all methods, 1000 permutations"
  $PY examples/openclose_validation/run_retest_methods.py
  step "Open-Close: cross-cohort agreement (Table oc-cross) + block-mass figure"
  $PY examples/openclose_validation/audit_openclose_agreement.py
  step "Open-Close: specificity figure"
  $PY examples/openclose_validation/plot5_retest_specificity.py
}

simulation() {
  local SV=examples/simulation_validation
  step "Simulation: FWER calibration, ${FPR_N_NULL} nulls x n in {${FPR_SAMPLE_SIZES}} x 11 methods, 1000 permutations"
  $PY $SV/fpr/fpr_calibration.py --config $SV/configs/fpr_config_1000.yaml \
      --n-null "$FPR_N_NULL" --sample-sizes $FPR_SAMPLE_SIZES --resume
  step "Simulation: power, 10 topologies x Delta r {0.075, 0.125, 0.25} x 100 repeats, 1000 permutations"
  $PY $SV/power/power_analysis.py --config $SV/configs/sweep_config_power_1000.yaml \
      --mode effect-size --resume
  step "Simulation: TFNBS (E,H) summaries"
  $PY $SV/power/tfnbs_eh_sweep.py \
      --results-dir $SV/results/power/power_analysis_1000_30 --effect-size 0.125 --sample-size 30
  step "Simulation: figures and power tables"
  $PY $SV/regenerate_paper_figures.py
}

sensitivity() {
  local SV=examples/simulation_validation
  local P=$SV/power/power_analysis.py C=$SV/configs R=$SV/results/sensitivity
  step "Sensitivity: prior misspecification (true / 7 / 2 blocks)"
  for L in true 7blocks 2blocks; do
    $PY $P --config $C/sweep_config_prior_misspec.yaml --mode effect-size --resume \
        --label-override $L --output-dir $R/prior_misspec/prior_misspec_$L
  done
  $PY $SV/sensitivity/analyze_prior_misspec.py --true-dir $R/prior_misspec/prior_misspec_true \
      --seven-dir $R/prior_misspec/prior_misspec_7blocks --two-dir $R/prior_misspec/prior_misspec_2blocks \
      --output-dir $R/prior_misspec/comparison
  step "Sensitivity: FBC-TFNBS m_min in {1, 2, 3, 5, 10}"
  for M in 1 2 3 5 10; do
    $PY $P --config $C/sweep_config_mmin.yaml --mode effect-size --resume \
        --fbc-min-cluster $M --output-dir $R/mmin/mmin_$M
  done
  $PY $SV/sensitivity/analyze_mmin_sensitivity.py --mmin-values 1 2 3 5 10 \
      --results-dirs $R/mmin/mmin_1 $R/mmin/mmin_2 $R/mmin/mmin_3 $R/mmin/mmin_5 $R/mmin/mmin_10 \
      --output-dir $R/mmin/comparison
  step "Sensitivity: NI-TFNBS block normalization (sqrt / linear / none)"
  for N in sqrt linear none; do
    $PY $P --config $C/sweep_config_norm_ablation.yaml --mode effect-size --resume \
        --ni-normalization $N --output-dir $R/ablation/norm_$N
  done
  $PY $SV/sensitivity/analyze_norm_ablation.py --sqrt-dir $R/ablation/norm_sqrt \
      --linear-dir $R/ablation/norm_linear --none-dir $R/ablation/norm_none \
      --output-dir $R/ablation/comparison
  step "Sensitivity: permutation BH-FDR"
  $PY $P --config $C/sweep_config_bh_fdr_perm.yaml --mode effect-size --resume \
      --output-dir $R/bh_fdr_perm
  step "Sensitivity: figures"
  $PY $SV/regenerate_paper_figures.py
}

benchmarks() {
  step "Timing: NBS / TFNBS / GLM-TFNBS, 100 permutations, n=30, 60/100/200/400 nodes"
  $PY examples/benchmarks/timing_benchmark.py --n_nodes 60 100 200 400 --n_perms 100 --n_samples 30
  $PY examples/benchmarks/plot_results.py --only timing_benchmark
}

copy() {  # copy SRC DEST_DIR [NEW_NAME]
  local src="$1" dest="$2/${3:-$(basename "$1")}"
  if [ -f "$src" ]; then cp "$src" "$dest"; echo "  $dest"; else echo "  MISSING $src"; fi
}

collect() {
  step "Collect figures into paper/figures (manuscript names)"
  local A=examples/abide_validation/results/plots O=examples/openclose_validation/results/plots
  local S=examples/simulation_validation/results/plots F=paper/figures
  mkdir -p "$F"/{abide,openclose,simulation,figures}
  # Manuscript plot1a/plot1b are two panels of this one figure (split not scripted)
  copy "$A/plot1_combat_site_variance.png"   "$F/abide"
  copy "$A/plot3_network_block_mass.png"     "$F/abide"
  copy "$A/plot5_tfnbs_grid_sensitivity.png" "$F/abide"
  copy "$A/plot6_diagnosis_methodology.png"  "$F/abide"
  copy "$O/plot2_block_mass_convergence.png" "$F/openclose"
  copy "$O/plot5_retest_specificity.png"     "$F/openclose"
  copy "$S/fig1_fwer_calibration.png"        "$F/simulation"
  # Manuscript fig2a (TPR) / fig2b (FDR) are two panels of this figure
  copy "$S/fig2_power_heatmap_dr0.125.png"     "$F/simulation"
  copy "$S/fig4_tfnbs_eh_tpr_fdr.png"          "$F/simulation"
  copy "$S/topologies_gallery_dr0.125_n30.png" "$F/simulation"
  copy "$S/fig5_prior_misspecification.png"    "$F/simulation"
  copy "$S/mmin_sensitivity.png"               "$F/simulation"
  mkdir -p "$F/tables"
  for t in examples/simulation_validation/results/tables/*.csv; do copy "$t" "$F/tables"; done
  copy examples/benchmarks/res_benchmarks/timing_benchmark.png "$F/figures"
}

for stage in "${@:-all}"; do
  case "$stage" in
    abide|openclose|simulation|sensitivity|benchmarks|collect) "$stage" ;;
    all) abide; openclose; simulation; sensitivity; benchmarks; collect ;;
    *) echo "unknown stage: $stage (abide|openclose|simulation|sensitivity|benchmarks|collect|all)" >&2; exit 2 ;;
  esac
done
