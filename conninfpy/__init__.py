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

from .analysis import AnalyzeResult, analyze
from ._compat import TailResult
from .results import InferenceResult, OmnibusInferenceResult
from .atlas import AtlasInfo, get_bna_246_nifti_path, fetch_bna_behavioral_decoding
from .defaults import (
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_START_THRESHOLD,
    DEFAULT_N_THRESHOLDS_SCORING,
    DEFAULT_N_THRESHOLDS_PERMUTATION,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_NBS_STAT,
    DEFAULT_MIN_CLUSTER_SIZE,
)

from .synth_datasets import (
    generate_fc_matrices,
    generate_multisite_glm_dataset,
    ModularDatasetGenerator,
)
from .topologies import (
    TopologyDataset,
    TopologyDatasetGenerator,
    TopologyScenario,
    get_scenario,
    get_scenarios,
    list_scenarios,
)

from .eeg_utils import (
    read_from_eeg_dataframe,
    reshape_eeg_data,
    inverse_reshape_eeg_data,
    EEGData,
    Electrodes,
    Bands,
    PairsElectrodes1020,
)
from .nbs_score import nbs_bct
from .pairwise_stats import (
    compute_p_val,
    compute_null_dist,
    compute_t_stat,
    compute_t_stat_diff,
    StatMethod,
    TestType,
)

from .enhance import (
    apply_tfnbs,
    apply_nbs,
    apply_cnbs,
    apply_ni_tfnbs,
    apply_fbc_tfnbs,
)

from .tfnbs_score import (
    get_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_fbc_tfnbs_score,
)
from .utils import (
    fisher_r_to_z,
    fisher_z_to_r,
    get_components,
    binarize,
    create_prior_weights,
)

from .glm_stats import (
    GLMStatType,
    compute_glm_stat,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
    build_design_matrix,
)

from .harmonize import (
    ComBatModel,
    CombatResult,
    combat_harmonize,
    combat_fit,
    combat_apply,
    compute_vif,
    design_diagnostics,
    flatten_upper,
    unflatten_upper,
    block_mass,
)

from .acceleration import (
    fit_gpd_tail,
    fit_gamma_tail,
    compute_p_values_accelerated,
)

from . import (
    acceleration,
    defaults,
    eeg_utils,
    enhance,
    harmonize,
    loaders,
    sim,
    stats,
)
from .loaders import load


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
    "enhance",
    "harmonize",
    "acceleration",
    "loaders",
    "sim",
    "eeg_utils",
    "defaults",
]
