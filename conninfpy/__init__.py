"""ConnInfPy — Connectivity Inference in Python.

Permutation-based statistical inference on brain connectivity networks (fMRI,
EEG). One permutation engine shared across nine inference methods, edge-wise
GLM (Freedman--Lane), parametric empirical-Bayes ComBat, and tail-approximation
acceleration.

Public quick-start
------------------
>>> import numpy as np
>>> from conninfpy import compute_p_val, fisher_r_to_z
>>> rng = np.random.default_rng(0)
>>> corr = np.clip(rng.uniform(-0.5, 0.5, (16, 8, 8)), -0.99, 0.99)
>>> corr = (corr + corr.transpose(0, 2, 1)) / 2
>>> idx = np.arange(8); corr[:, idx, idx] = 0.0
>>> z = fisher_r_to_z(corr)
>>> p = compute_p_val(z[:8], z[8:], test_type='two-sample',
...                   method='tstat', n_permutations=50, rng=0,
...                   use_mp=False)
>>> p['positive'].shape
(8, 8)

Returns a :class:`~conninfpy._compat.TailResult` with canonical keys
``'positive'`` and ``'negative'``. The legacy keys ``'g2>g1'`` and
``'g1>g2'`` from v1.x remain readable but emit a :class:`DeprecationWarning`
and will be removed in v2.1.
"""
__version__ = "2.0.0"

from ._analyze import AnalyzeResult, analyze
from ._compat import TailResult
from ._result import InferenceResult, OmnibusInferenceResult
from .atlas import AtlasInfo, get_bna_246_nifti_path
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

from ._enhancement import (
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

from . import loaders


__all__ = [
    # version + result types
    "__version__",
    "TailResult",
    "InferenceResult",
    "OmnibusInferenceResult",
    "AnalyzeResult",
    "analyze",
    "AtlasInfo",
    "get_bna_246_nifti_path",
    # defaults
    "DEFAULT_EXTENT_EXPONENT",
    "DEFAULT_HEIGHT_EXPONENT",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_N_THRESHOLDS_SCORING",
    "DEFAULT_N_THRESHOLDS_PERMUTATION",
    "DEFAULT_N_PERMUTATIONS",
    "DEFAULT_NBS_THRESHOLD",
    "DEFAULT_NBS_STAT",
    "DEFAULT_MIN_CLUSTER_SIZE",
    # statistical pipelines
    "compute_p_val",
    "compute_null_dist",
    "compute_t_stat",
    "compute_t_stat_diff",
    "StatMethod",
    "TestType",
    "compute_p_val_glm",
    "compute_p_val_glm_multi",
    "compute_p_val_paired_glm",
    "compute_glm_stat",
    "build_design_matrix",
    "GLMStatType",
    # enhancement (apply_*)
    "apply_tfnbs",
    "apply_nbs",
    "apply_cnbs",
    "apply_ni_tfnbs",
    "apply_fbc_tfnbs",
    # raw scoring
    "get_tfnbs_score",
    "get_network_informed_tfnbs_score",
    "get_fbc_tfnbs_score",
    "nbs_bct",
    # acceleration
    "fit_gpd_tail",
    "fit_gamma_tail",
    "compute_p_values_accelerated",
    # harmonization + design diagnostics
    "ComBatModel",
    "CombatResult",
    "combat_harmonize",
    "combat_fit",
    "combat_apply",
    "compute_vif",
    "design_diagnostics",
    "flatten_upper",
    "unflatten_upper",
    "block_mass",
    # synthetic data + topologies
    "generate_fc_matrices",
    "generate_multisite_glm_dataset",
    "ModularDatasetGenerator",
    "TopologyDataset",
    "TopologyDatasetGenerator",
    "TopologyScenario",
    "get_scenario",
    "get_scenarios",
    "list_scenarios",
    # utilities
    "fisher_r_to_z",
    "fisher_z_to_r",
    "get_components",
    "binarize",
    "create_prior_weights",
    # EEG
    "read_from_eeg_dataframe",
    "reshape_eeg_data",
    "inverse_reshape_eeg_data",
    "EEGData",
    "Electrodes",
    "Bands",
    "PairsElectrodes1020",
    "loaders",
]
