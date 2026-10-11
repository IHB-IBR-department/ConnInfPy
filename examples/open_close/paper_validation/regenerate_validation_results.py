"""Regenerate selected Open-Close workflows using the active Python environment."""
from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from pathlib import Path
import sys

import argparse
import subprocess
from examples._shared.paths import ROOT

PAPER_STEPS = (
    "paper_validation.harmonize_pooled_cohorts",
    "paper_validation.run_paired_tfnbs",
    "paper_validation.run_pooled_methods",
    "paper_validation.run_retest_methods",
    "paper_validation.run_state_methods",
    "paper_validation.run_eh_sensitivity",
    "paper_validation.audit_openclose_agreement",
    "paper_validation.plots.plot1_combat_impact",
    "paper_validation.plots.plot2_block_mass_convergence",
    "paper_validation.plots.plot3_method_sensitivity",
    "paper_validation.plots.plot5_retest_specificity",
    "paper_validation.plots.plot6_network_informed_hierarchy",
    "paper_validation.plots.plot6_retest_hierarchy",
)
ML_STEPS = (
    "ml_transfer.run_ml_feature_selection",
    "ml_transfer.audit_ml_feature_selection",
    "paper_validation.plots.plot7_ml_transfer_auc",
)


def commands(scope):
    steps = PAPER_STEPS if scope == "paper" else ML_STEPS if scope == "ml" else PAPER_STEPS + ML_STEPS
    return [[sys.executable, "-m", "examples.open_close." + step] for step in steps]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("paper", "ml", "all"), default="paper")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without computing")
    args = parser.parse_args()
    for command in commands(args.scope):
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
