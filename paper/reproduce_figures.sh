#!/usr/bin/env bash
# Reproduce every data figure and table in the ConnInfPy manuscript
# (PaperNN/main.tex). See paper/README.md for the figure -> script map,
# expected run times and open issues.
#
# Usage (from anywhere in the repository):
#   bash paper/reproduce_figures.sh                 # everything (~3 h on an M5 Pro)
#   bash paper/reproduce_figures.sh abide collect   # selected stages
# Stages: abide | openclose | mltransfer | simulation | sensitivity | benchmarks | collect | all
#
# Results go to examples/*/results/; `collect` builds the 2.0.0 600 dpi
# manuscript figures into paper/figures/ (and copies simulation tables).
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
  $PY examples/abide/prepare_data.py
  # Age and diagnosis are separate analyses: the age model adjusts for sex and
  # motion only (not diagnosis).
  step "ABIDE: age (sex + motion covariates), ComBat + site dummies, 5000 permutations"
  $PY examples/abide/age/run_age_strategy_d.py --variant age
  step "ABIDE: diagnosis under four site strategies, 5000 permutations each"
  $PY examples/abide/diagnosis/run_naive_baseline.py
  $PY examples/abide/diagnosis/run_dx_combat_only.py
  $PY examples/abide/diagnosis/run_dx_strategy_e.py
  $PY examples/abide/diagnosis/run_dx_site_glm.py
  # Every TFNBS-family call passes (E, H, n) = (0.4, 3, 10) explicitly: the
  # package default E is 0.3 since 2026-05-25. All p-values are exact
  # permutation p-values (no GPD tail approximation).
  step "ABIDE: seven-method comparison on age, 5000 permutations each"
  $PY examples/abide/age/run_age_method_comparison.py
  step "ABIDE: TFNBS (E,H) multiverse, 36 cells x 3 methods, same design as the age map, 5000 permutations"
  $PY examples/abide/age/run_eh_sensitivity.py
  step "ABIDE: figures"
  $PY examples/abide/plots/plot_paper_figures.py
  $PY examples/abide/plots/plot_method_sensitivity.py
  $PY examples/abide/plots/plot_diagnosis_quadrant.py   # last: owns plot6
}

openclose() {
  step "Open-Close: paired TFNBS (IHB, Beijing, Beijing EC-run retest), 5000 permutations"
  $PY examples/open_close/paper_validation/run_paired_tfnbs.py --analyses ihb china china_retest
  step "Open-Close: paired state maps (8 methods x 2 cohorts, 5000 permutations)"
  $PY examples/open_close/paper_validation/run_state_methods.py --n-perm 5000
  step "Open-Close: retest maps for all methods, 1000 permutations"
  $PY examples/open_close/paper_validation/run_retest_methods.py
  step "Open-Close: cross-cohort agreement (Table oc-cross) + block-mass figure"
  $PY examples/open_close/paper_validation/audit_openclose_agreement.py
  step "Open-Close: specificity figure"
  $PY examples/open_close/paper_validation/plot5_retest_specificity.py
}

mltransfer() {
  step "Open-Close: separate significant-edge ML transfer study"
  $PY examples/open_close/ml_transfer/run_ml_feature_selection.py
  $PY examples/open_close/ml_transfer/audit_ml_feature_selection.py
  $PY examples/open_close/paper_validation/plot7_ml_transfer_auc.py
}

simulation() {
  local SV=examples/simulations
  step "Simulation: FWER calibration (baseline + exact TFNBS family), ${FPR_N_NULL} nulls x n in {${FPR_SAMPLE_SIZES}}, 1000 permutations"
  $PY $SV/fpr/fpr_calibration.py --config $SV/configs/fpr_config_1000.yaml \
      --n-null "$FPR_N_NULL" --sample-sizes $FPR_SAMPLE_SIZES --resume
  $PY $SV/fpr/fpr_calibration.py --config $SV/configs/fpr_config_ni_fbc_exact.yaml \
      --n-null "$FPR_N_NULL" --sample-sizes $FPR_SAMPLE_SIZES --resume
  step "Simulation: Freedman-Lane GLM 2.0 null calibration"
  $PY $SV/glm/glm_validation.py --config $SV/configs/glm_config_2_0_1000.yaml --resume
  step "Simulation: power (10 topologies x Delta r {0.075, 0.125, 0.25} x 100 repeats, 1000 permutations, exact TFNBS)"
  $PY $SV/power/power_analysis.py --config $SV/configs/sweep_config_tfnbs_exact_04_30.yaml \
      --mode effect-size --resume
  $PY $SV/power/power_analysis.py --config $SV/configs/sweep_config_ni_fbc_exact_04_30.yaml \
      --mode effect-size --resume
  $PY $SV/power/power_analysis.py --config $SV/configs/sweep_config_power_1000.yaml \
      --mode effect-size --resume
}

sensitivity() {
  local SV=examples/simulations
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
}

benchmarks() {
  step "Timing: NBS / TFNBS / GLM-TFNBS, 100 permutations, n=30, 60/100/200/400 nodes"
  $PY examples/benchmarks/performance/timing_benchmark.py --n_nodes 60 100 200 400 --n_perms 100 --n_samples 30
  $PY examples/benchmarks/performance/plot_results.py --only timing_benchmark
}

copy() {  # copy SRC DEST_DIR [NEW_NAME]
  local src="$1" dest="$2/${3:-$(basename "$1")}"
  if [ -f "$src" ]; then cp "$src" "$dest"; echo "  $dest"; else echo "  MISSING $src"; fi
}

collect() {
  step "Build 2.0.0 manuscript figures (600 dpi) into paper/figures"
  local F=paper/figures
  mkdir -p "$F"/{abide,openclose,simulation,figures,tables}
  $PY paper/build_fig1_pipeline.py
  $PY paper/build_manuscript_figures.py --group all
  for t in examples/simulations/results/tables/*.csv; do copy "$t" "$F/tables"; done
}

for stage in "${@:-all}"; do
  case "$stage" in
    abide|openclose|mltransfer|simulation|sensitivity|benchmarks|collect) "$stage" ;;
    all) abide; openclose; mltransfer; simulation; sensitivity; benchmarks; collect ;;
    *) echo "unknown stage: $stage (abide|openclose|mltransfer|simulation|sensitivity|benchmarks|collect|all)" >&2; exit 2 ;;
  esac
done
