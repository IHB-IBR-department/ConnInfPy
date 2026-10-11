"""ConnInfPy — Connectivity Inference in Python.

Permutation-based statistical inference on brain connectivity networks (fMRI,
EEG). One permutation engine shared across nine inference methods, edge-wise
GLM (Freedman--Lane), parametric empirical-Bayes ComBat, and tail-approximation
acceleration.

Quick-start
-----------
>>> import numpy as np
>>> import conninfpy
>>> rng = np.random.default_rng(0)
>>> r = np.tanh(rng.normal(0, 0.3, (16, 8, 8)))      # 16 correlation matrices
>>> r = (r + r.transpose(0, 2, 1)) / 2
>>> out = conninfpy.analyze(group1=r[:8], group2=r[8:], input="r",
...                         test_type="two-sample", method="tstat",
...                         n_permutations=50, rng=0, use_mp=False)
>>> out["positive"].shape
(8, 8)

:func:`analyze` is the main entry point; it returns an
:class:`AnalyzeResult` whose ``positive`` / ``negative`` maps hold
FWER-corrected p-values. Everything else is grouped by module:

- :mod:`conninfpy.stats` — advanced permutation pipelines (custom designs,
  F-contrasts, null distributions)
- :mod:`conninfpy.enhance` — network enhancement (TFNBS, NBS, cNBS, ...)
- :mod:`conninfpy.harmonize` — ComBat and design diagnostics
- :mod:`conninfpy.acceleration` — GPD / gamma tail approximation
- :mod:`conninfpy.loaders` — manifest-based data loading (also :func:`load`)
- :mod:`conninfpy.sim` — synthetic data and topology scenarios
- :mod:`conninfpy.plot`, :mod:`conninfpy.eeg_utils`

Names imported directly from ``conninfpy`` in earlier versions remain
importable; ``__all__`` lists the recommended surface.
"""

__version__ = "2.0.0"

from conninfpy import (
    defaults,
    eeg_utils,
    enhance,
    harmonize,
    interpret,
    loaders,
    sim,
    stats,
)
from conninfpy._compat import TailResult
from conninfpy._shared.arrays import (
    binarize,
    create_prior_weights,
    fisher_r_to_z,
    fisher_z_to_r,
    get_components,
)
from conninfpy.analysis import AnalyzeResult, analyze
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.atlas.metadata import AtlasInfo
from conninfpy.atlas.resources import (
    fetch_bna_behavioral_decoding,
    get_bna_246_nifti_path,
)
from conninfpy.defaults import (
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_N_THRESHOLDS_PERMUTATION,
    DEFAULT_N_THRESHOLDS_SCORING,
    DEFAULT_NBS_STAT,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_START_THRESHOLD,
)
from conninfpy.eeg_utils import (
    Bands,
    EEGData,
    Electrodes,
    PairsElectrodes1020,
    inverse_reshape_eeg_data,
    read_from_eeg_dataframe,
    reshape_eeg_data,
)
from conninfpy.enhance import (
    apply_cnbs,
    apply_fbc_tfnbs,
    apply_nbs,
    apply_ni_tfnbs,
    apply_tfnbs,
)
from conninfpy.enhance.nbs import nbs_bct
from conninfpy.enhance.tfnbs import (
    get_fbc_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_tfnbs_score,
)
from conninfpy.harmonize import (
    ComBatModel,
    CombatResult,
    block_mass,
    combat_apply,
    combat_fit,
    combat_harmonize,
    compute_vif,
    design_diagnostics,
    flatten_upper,
    unflatten_upper,
)
from conninfpy.interpret.annotate import (
    RoiAnnotation,
    annotate_bna_regions,
    annotate_edges,
    annotate_rois,
    summarize_annotation,
)
from conninfpy.loaders import load
from conninfpy.results import InferenceResult, OmnibusInferenceResult
from conninfpy.sim.datasets import (
    ModularDatasetGenerator,
    generate_fc_matrices,
    generate_multisite_glm_dataset,
)
from conninfpy.sim.topologies import (
    TopologyDataset,
    TopologyDatasetGenerator,
    TopologyScenario,
    get_scenario,
    get_scenarios,
    list_scenarios,
)
from conninfpy.stats import acceleration as acceleration
from conninfpy.stats._design import build_design_matrix
from conninfpy.stats._types import GLMStatType, StatMethod, TestType
from conninfpy.stats.acceleration import (
    compute_p_values_accelerated,
    fit_gamma_tail,
    fit_gpd_tail,
)
from conninfpy.stats.glm import (
    compute_glm_stat,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
)
from conninfpy.stats.pairwise import (
    compute_null_dist,
    compute_p_val,
    compute_t_stat,
    compute_t_stat_diff,
)

# The recommended public surface. Everything imported above stays
# importable from ``conninfpy`` for backward compatibility, but new code
# should reach advanced functions through their module (``conninfpy.stats``,
# ``conninfpy.enhance``, ...).
__all__ = [
    "__version__",
    # main entry point and results
    "analyze",
    "AnalyzeResult",
    "InferenceResult",
    "OmnibusInferenceResult",
    # data in
    "load",
    "AtlasInfo",
    "fetch_bna_behavioral_decoding",
    "fisher_r_to_z",
    "fisher_z_to_r",
    # modules
    "stats",
    "interpret",
    "enhance",
    "harmonize",
    "acceleration",
    "loaders",
    "sim",
    "eeg_utils",
    "defaults",
]
