"""
Unified default parameters for ConnInfPy.

Single source of truth for all default constants used across modules.

TFNBS defaults use E=0.4 and H=3.0. By default n=None integrates the
piecewise support exactly, independently for each tail. An explicit positive
integer n selects the historical right-endpoint grid, with threshold levels
start + i * (max - start) / n and inclusive active edges at each level.
Legacy finite-grid scores and the exact integral need not be equal. Record
E, H, n and start explicitly when reproducing historical results.

The permutation-based BH-FDR method uses raw edge statistics and does not
apply TFNBS enhancement or its integration settings.

References
----------
- Smith & Nichols (2009). Threshold-free cluster enhancement.
  NeuroImage 44(1):83-98.
- Baggio et al. (2018). Statistical inference in brain graphs using
  threshold-free network-based statistics. Hum Brain Mapp 39(6):2289-2302.
  doi:10.1002/hbm.24007.
- Vinokur, L., Smith, R. E., Dhollander, T., Vaughan, D., Jackson, G. D., &
  Connelly, A. (2023). Parameter Sensitivity of Network-Based Statistical
  Inference. Research Square (preprint), doi:10.21203/rs.3.rs-3081615/v1.
- Hao, Z., Wang, P., Xia, X., Pan, Y., & Dou, W. (2024). Threshold-Free
  Network-Oriented Statistics in Neuroscience (TFNOS). bioRxiv (preprint),
  doi:10.1101/2024.01.03.574016. For edge-wise t-statistics in TFNOS-TT,
  ``E=0.4, H=3.0`` yielded empirical FDR < 10% and ``E=0.4, H=7.0``
  empirical FDR < 5%; earlier voxel/network defaults such as Baggio 2018
  were overly liberal in the reported conditions.
"""

# =============================================================================
# TFCE parameters (validation-paper network setting)
# =============================================================================

DEFAULT_EXTENT_EXPONENT: float = 0.4
"""Default extent exponent E."""

DEFAULT_HEIGHT_EXPONENT: float = 3.0
"""Default height exponent H."""

DEFAULT_START_THRESHOLD: float = 1.65
"""Initial t-threshold for cluster formation (asymptotic normal z ≈ 1.645 reference, one-tailed p ≈ 0.05 at large df; not a universal finite-df p-value)."""

# =============================================================================
# Integration resolution (path-specific)
# =============================================================================

DEFAULT_N_THRESHOLDS_SCORING = None
"""None selects exact integration; a positive integer selects the legacy grid."""

DEFAULT_N_THRESHOLDS_PERMUTATION = None
"""None selects exact integration inside the permutation loop."""

# =============================================================================
# Permutation testing
# =============================================================================

DEFAULT_N_PERMUTATIONS: int = 1000
"""Default number of permutations for null distribution."""

DEFAULT_ACCELERATION = None
"""Default tail acceleration: ``None`` (exact empirical permutation p-values).

``'gpd'`` / ``'gamma'`` are opt-in for exploration with ~200 permutations: a
fitted tail replaces the p-values that fewer than 10 permutations reach
(Winkler 2016); all others stay exact."""

# =============================================================================
# NBS defaults
# =============================================================================

DEFAULT_NBS_THRESHOLD: float = 3.0
"""Default t-statistic threshold for classical NBS edge inclusion."""

DEFAULT_NBS_STAT: str = "extent"
"""Default cluster statistic type for NBS ('extent' or 'intensity')."""

# =============================================================================
# FBC-TFNBS
# =============================================================================

DEFAULT_MIN_CLUSTER_SIZE: int = 3
"""Default minimum edges in a functional block to form a cluster (for FBC-TFNBS)."""


def warn_low_nbs_threshold(method, threshold):
    """Warn about interpretation of components at a low NBS threshold."""
    if method == "nbs" and threshold < 3.0:
        import warnings
        warnings.warn(
            "NBS threshold < 3.0 increases non-null edge spillover within "
            "detected components; NBS inference is at the component level, "
            "not the individual edge level.", UserWarning, stacklevel=3
        )


def warn_bh_permutation_resolution(m, n_permutations):
    """Warn when permutation resolution cannot support sparse BH rejections."""
    import math
    import warnings

    rank = m / (0.05 * (n_permutations + 1.0))
    if rank > 1.0:
        warnings.warn(
            f"bh_fdr_perm resolution: m={m} edges, B={n_permutations}, "
            f"p_min={1 / (n_permutations + 1.0):.6g}; minimum rejection "
            f"rank k_min={math.ceil(rank)} at q=0.05. For single-edge "
            f"sensitivity use n_permutations >= {math.ceil(m / 0.05)} "
            "or consider parametric bh_fdr.", UserWarning, stacklevel=3
        )
