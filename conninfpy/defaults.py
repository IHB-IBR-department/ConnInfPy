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
- Baggio et al. (2018). TFNBS: a threshold-free method for the analysis
  of brain networks. Hum Brain Mapp 39(6):2289-2302.
- Vinokur et al. (2023). Parameter sensitivity of NBS / TFNBS in
  connectivity inference.
- Hao et al. (2024). TFNOS: threshold-free network omnibus statistics
  (recommends ``E=0.4, H in [3.0, 7.0]`` for FDR < 10%).
"""

# =============================================================================
# TFCE parameters (validation-paper network setting)
# =============================================================================

DEFAULT_EXTENT_EXPONENT: float = 0.4
"""Default extent exponent E."""

DEFAULT_HEIGHT_EXPONENT: float = 3.0
"""Default height exponent H."""

DEFAULT_START_THRESHOLD: float = 1.65
"""Initial threshold for cluster formation (p < 0.05 one-tailed)."""

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

DEFAULT_NBS_THRESHOLD: float = 2.0
"""Default t-statistic threshold for classical NBS edge inclusion."""

DEFAULT_NBS_STAT: str = "extent"
"""Default cluster statistic type for NBS ('extent' or 'intensity')."""

# =============================================================================
# FBC-TFNBS
# =============================================================================

DEFAULT_MIN_CLUSTER_SIZE: int = 3
"""Default minimum edges in a functional block to form a cluster (for FBC-TFNBS)."""
