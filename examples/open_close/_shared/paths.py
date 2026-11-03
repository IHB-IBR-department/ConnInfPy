"""Open-Close study inputs and the results layout: one folder per workflow."""
from examples._shared.paths import ROOT, DATASETS

STUDY = ROOT / "examples" / "open_close"
DATA_DIR = DATASETS / "open_close"
RESULTS = STUDY / "results"
PLOTS = RESULTS / "plots"                                # publication figures
PAPER = RESULTS / "paper_validation"                     # paired maps, controls, sensitivity, audit
METHOD_MAPS = PAPER / "methods"                          # per-method paired p-maps
ATLAS_STABILITY = RESULTS / "atlas_stability"
STRUCTURAL_PRIOR = RESULTS / "structural_prior"
FEATURE_STABILITY = RESULTS / "feature_stability"
ML_TRANSFER = RESULTS / "ml_transfer"
