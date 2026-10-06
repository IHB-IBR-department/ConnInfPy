"""Advanced inference API: the permutation pipelines behind :func:`conninfpy.analyze`.

Use these when you need control that ``analyze()`` does not expose — a
custom design matrix and contrast, an F-contrast, or the raw null
distribution. They expect data already on the analysis scale (e.g. Fisher
z) and return :class:`~conninfpy.results.InferenceResult`.
"""
from .glm_stats import (
    GLMStatType,
    build_design_matrix,
    compute_glm_stat,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
)
from .pairwise_stats import (
    StatMethod,
    TestType,
    compute_null_dist,
    compute_p_val,
    compute_t_stat,
    compute_t_stat_diff,
)

__all__ = [
    # permutation pipelines
    "compute_p_val",
    "compute_p_val_glm",
    "compute_p_val_glm_multi",
    "compute_p_val_paired_glm",
    "compute_null_dist",
    # edge-wise statistics and designs
    "compute_t_stat",
    "compute_t_stat_diff",
    "compute_glm_stat",
    "build_design_matrix",
    # enums
    "StatMethod",
    "TestType",
    "GLMStatType",
]
