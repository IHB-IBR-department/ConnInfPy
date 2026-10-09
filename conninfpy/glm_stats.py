"""
GLM-based statistical testing for brain connectivity networks.

Implements edge-wise General Linear Model with Freedman-Lane permutation
for confound-aware inference on connectivity matrices. Supports all
enhancement methods (TFNBS, NBS, cNBS, NI-TFNBS, FBC-TFNBS).

Main Functions
--------------
compute_glm_stat : Compute GLM t-statistics for connectivity data
compute_p_val_glm : Full GLM pipeline with permutation testing
build_design_matrix : Convenience builder for design matrix + contrast

References
----------
Freedman & Lane (1983). A nonstochastic interpretation of reported significance levels.
Anderson & Legendre (1999). An empirical comparison of permutation methods.
Winkler et al. (2014). Permutation inference for the general linear model.
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from .defaults import (
    DEFAULT_ACCELERATION,
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_N_THRESHOLDS_PERMUTATION as DEFAULT_N_THRESHOLDS,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_START_THRESHOLD,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_NBS_STAT,
)
from .enhance import (
    apply_cnbs,
    apply_fbc_tfnbs,
    apply_nbs,
    apply_ni_tfnbs,
    apply_tfnbs,
)
from . import _degenerate, _estimates
from ._engine import (
    _collect_results_to_arrays,
    _compute_p_values_from_null,
    _encode_strata,
    _extract_max_stats,
    _is_worker_process,
    _stratified_perm,
    get_available_cores,
)
from .pairwise_stats import (
    CONSTRAINED_METHODS,
    StatMethod,
    _grid_from_kwargs,
    compute_p_val,
)
import time

from .acceleration import compute_p_values_accelerated
from ._compat import TailResult, make_tail_result, normalize_keys
from ._progress import run_permutations
from .results import InferenceResult, OmnibusInferenceResult
from ._prepare import check_connectivity, check_finite_design
from ._rng import RngLike, resolve_seed, warn_legacy_random_state


__all__ = [
    "GLMStatType",
    "compute_glm_stat",
    "compute_p_val_glm",
    "compute_p_val_paired_glm",
    "build_design_matrix",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class GLMStatType(str, Enum):
    """Statistic type for GLM inference."""

    TSTAT = "tstat"
    """t-statistic: beta / SE(beta). Scale-invariant, start_thres=1.65 valid."""

    BETA = "beta"
    """Raw regression coefficient. Interpretable, needs adapted threshold."""

    FSTAT = "fstat"
    """F-statistic for an omnibus (multi-row) contrast.

    Non-negative by construction — the enhancement pipeline sees a single
    ``'omnibus'`` key instead of a ``positive``/``negative`` split. For a
    single-row contrast, F == t² (same rejection regions, same rankings)."""


# =============================================================================
# OLS precomputation
# =============================================================================

def _precompute_ols(
    X: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Precompute OLS quantities that are constant across permutations.

    Parameters
    ----------
    X : ndarray of shape (n_subjects, p)
        Design matrix (must include intercept if desired).

    Returns
    -------
    X_pinv : ndarray of shape (p, n_subjects)
        Moore-Penrose pseudoinverse of the design matrix.
    XtX_inv_diag : ndarray of shape (p,)
        Diagonal of the generalized inverse of X'X, needed for SE computation.
    XtX_inv : ndarray of shape (p, p)
        Generalized inverse of X'X, needed for contrast SE computation.
    """
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        logger.warning(
            "Design matrix is rank deficient "
            f"(rank={rank}, columns={X.shape[1]}). "
            "Using Moore-Penrose pseudoinverse; non-estimable contrasts may "
            "still be unstable."
        )
    else:
        cond = np.linalg.cond(X)
        if cond > 1e12:
            logger.warning(
                f"Design matrix is ill-conditioned (cond={cond:.1e}). "
                f"Results may be numerically unstable."
            )

    X_pinv = np.linalg.pinv(X)
    XtX_inv = X_pinv @ X_pinv.T
    XtX_inv_diag = np.diag(XtX_inv)
    return X_pinv, XtX_inv_diag, XtX_inv


def _mean_square(Y_flat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-column mean of Y**2, the scale for a zero residual variance."""
    return np.einsum("ij,ij->j", Y_flat, Y_flat) / Y_flat.shape[0]


def contrast_is_estimable(
    contrast: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    X_pinv: Optional[npt.NDArray[np.float64]] = None,
) -> bool:
    """True when every contrast row is a combination of the rows of X (C = C X⁺X).

    Only then is Cβ the same for every least-squares solution β. With
    collinear columns (two copies of a variable, a group dummy nested in site
    dummies) a coefficient on one of them alone is not identified, and the
    pseudoinverse would report an arbitrary split of their joint effect.
    """
    C = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
    if X_pinv is None:
        X_pinv = np.linalg.pinv(X)
    return bool(np.allclose(C @ X_pinv @ X, C, atol=1e-8 * max(1.0, np.abs(C).max())))


def _check_estimable(contrast: npt.NDArray[np.float64], X: npt.NDArray[np.float64],
                     name: str = "Contrast") -> None:
    if not contrast_is_estimable(contrast, X):
        raise ValueError(
            f"{name} is not estimable: it is not a combination of the rows of the "
            "design matrix (C != C X⁺X), so its value depends on an arbitrary choice "
            "among equally good fits. This happens with collinear columns (e.g. two "
            "copies of one variable, or a group indicator that is a sum of site "
            "dummies). Drop the redundant column or test a combination that is "
            "identified (e.g. the sum of the collinear coefficients)."
        )


def _infinite_edges(Y, X, contrast, stat_type: str) -> npt.NDArray[np.bool_]:
    """(N, N) mask of edges whose observed t or F is infinite (exact fit, nonzero contrast)."""
    N = Y.shape[1]
    iu = np.triu_indices(N, k=1)
    mask = np.zeros((N, N), dtype=bool)
    mask[iu] = _degenerate.glm_infinite(Y[:, iu[0], iu[1]], X, contrast, stat_type)
    return mask


def _residual_degrees_of_freedom(X: npt.NDArray[np.float64]) -> int:
    """Return residual df based on design rank, not raw column count."""
    df = X.shape[0] - int(np.linalg.matrix_rank(X))
    if df <= 0:
        raise ValueError(
            "Design matrix has no residual degrees of freedom "
            f"(n={X.shape[0]}, rank={np.linalg.matrix_rank(X)})."
        )
    return df


def _warn_beta_threshold_scale(
    stat_type_str: str,
    method_enum: StatMethod,
) -> None:
    """Warn when beta-scale statistics are paired with threshold enhancers."""
    threshold_methods = {
        StatMethod.NBS,
        StatMethod.TFNBS,
        StatMethod.NI_TFNBS,
        StatMethod.FBC_TFNBS,
    }
    if (
        stat_type_str == GLMStatType.BETA.value
        and method_enum in threshold_methods
    ):
        warnings.warn(
            "stat_type='beta' uses raw coefficient units. Ensure NBS/TFNBS "
            "thresholds are chosen on the beta scale, not the default t-stat "
            "scale.",
            UserWarning,
            stacklevel=3,
        )


# =============================================================================
# Core GLM computation
# =============================================================================

def compute_glm_stat(
    Y: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    stat_type: Union[str, GLMStatType] = GLMStatType.TSTAT,
    X_pinv: Optional[npt.NDArray[np.float64]] = None,
    XtX_inv_diag: Optional[npt.NDArray[np.float64]] = None,
    XtX_inv: Optional[npt.NDArray[np.float64]] = None,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute edge-wise GLM statistics for connectivity matrices.

    For each edge (i,j): Y_ij = X @ beta_ij + epsilon_ij.

    - ``tstat``/``beta``: single-column contrast, returns ``{positive, negative}``.
    - ``fstat``: multi-row contrast (omnibus F), returns ``{omnibus}`` — F is
      non-negative by construction, so there is no direction to split.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Connectivity matrices (symmetric, zero diagonal).
    X : ndarray of shape (n_subjects, p)
        Design matrix (should include intercept column if needed).
    contrast : ndarray of shape (p,) or (r, p)
        Contrast vector (for ``tstat``/``beta``: 1D) or contrast matrix (for
        ``fstat``: 2D with ``r`` rows). A 1D contrast passed with ``stat_type
        ='fstat'`` is treated as a 1×p matrix (F == t² at each edge).
    stat_type : {'tstat', 'beta', 'fstat'} or GLMStatType, default='tstat'
        Type of statistic to compute.
    X_pinv : ndarray of shape (p, n_subjects), optional
        Precomputed Moore-Penrose pseudoinverse. If None, computed
        internally by :func:`_precompute_ols`.
    XtX_inv_diag : ndarray of shape (p,), optional
        Precomputed diagonal of the generalized inverse of ``X'X``.
        Required when ``stat_type='tstat'`` and ``X_pinv`` is provided.
    XtX_inv : ndarray of shape (p, p), optional
        Generalized inverse of ``X'X``. Required for contrast standard
        errors when ``X_pinv`` is provided.

    Returns
    -------
    dict
        For ``tstat``/``beta``: ``{'positive': (N, N), 'negative': (N, N)}``,
        both non-negative.
        For ``fstat``: ``{'omnibus': (N, N)}``, non-negative.
    """
    stat_type_str = stat_type.value if isinstance(stat_type, GLMStatType) else stat_type

    n, N, _ = Y.shape
    contrast = np.asarray(contrast, dtype=np.float64)
    p = X.shape[1]

    # Precompute if not provided
    if X_pinv is None:
        X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)

    # Compute betas: (p, N, N)
    # Y reshaped to (n, N*N), then beta = X_pinv @ Y_flat → (p, N*N) → (p, N, N)
    Y_flat = Y.reshape(n, -1)  # (n, N*N)
    beta_flat = X_pinv @ Y_flat  # (p, N*N)
    beta = beta_flat.reshape(p, N, N)  # (p, N, N)

    if stat_type_str == GLMStatType.FSTAT.value:
        # --- F-statistic: (Cβ)' [C(X'X)^+ C']^-1 (Cβ) / (r · σ²) ---
        if contrast.ndim == 1:
            C = contrast.reshape(1, -1)
        elif contrast.ndim == 2:
            C = contrast
        else:
            raise ValueError(
                f"F-stat contrast must be 1D or 2D, got ndim={contrast.ndim}."
            )
        r = C.shape[0]
        if C.shape[1] != p:
            raise ValueError(
                f"Contrast has {C.shape[1]} columns but design matrix has {p}."
            )

        # Residual variance per edge
        residuals_flat = Y_flat - X @ beta_flat  # (n, N*N)
        df = _residual_degrees_of_freedom(X)
        sigma2_flat = np.sum(residuals_flat ** 2, axis=0) / df
        sigma2 = sigma2_flat.reshape(N, N)

        if XtX_inv is None:
            _, _, XtX_inv = _precompute_ols(X)

        # M = C (X'X)^+ C' shape (r, r); invert once (small)
        M = C @ XtX_inv @ C.T
        if np.linalg.matrix_rank(M) < r:
            raise ValueError(
                "C(X'X)⁻¹C' is singular — contrast rows are not linearly "
                "independent on the column space of X."
            )
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "C(X'X)⁻¹C' is singular — contrast rows are not linearly "
                "independent on the column space of X."
            ) from err

        # Cβ shape (r, N, N); quadratic form per edge via einsum
        Cbeta = np.tensordot(C, beta, axes=([1], [0]))  # (r, N, N)
        quad = np.einsum("ixy,ij,jxy->xy", Cbeta, M_inv, Cbeta)  # (N, N)

        # zero residual variance: F = 0 (see _degenerate; entry points
        # reject an exact fit with a nonzero contrast)
        floor = _degenerate.RTOL * _mean_square(Y_flat).reshape(N, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = quad / (r * sigma2)
        F = np.where(sigma2 <= floor, 0.0, F)
        F = np.where(F < 0, 0.0, F)  # guard numerical noise near zero

        return {"omnibus": F}

    # --- t-stat / beta path: single-column contrast, signed output ---
    if contrast.ndim != 1:
        raise ValueError(
            f"stat_type='{stat_type_str}' requires a 1D contrast; got "
            f"ndim={contrast.ndim}. Use stat_type='fstat' for multi-row contrasts."
        )

    # Contrast beta: scalar statistic per edge
    # contrast @ beta → (N, N)
    c_beta = np.tensordot(contrast, beta, axes=([0], [0]))  # (N, N)

    if stat_type_str == GLMStatType.BETA.value:
        stat = c_beta
    elif stat_type_str == GLMStatType.TSTAT.value:
        # Residuals: (n, N*N)
        residuals_flat = Y_flat - X @ beta_flat  # (n, N*N)

        # Residual variance per edge: sigma^2 = RSS / (n - rank(X))
        df = _residual_degrees_of_freedom(X)
        sigma2_flat = np.sum(residuals_flat ** 2, axis=0) / df  # (N*N,)
        sigma2 = sigma2_flat.reshape(N, N)

        # SE of contrast beta: sqrt(c' (X'X)^+ c * sigma^2)
        if XtX_inv is None:
            _, _, XtX_inv = _precompute_ols(X)
        c_XtX_inv_c = contrast @ XtX_inv @ contrast  # scalar
        # zero residual variance: t = 0 (see _degenerate; entry points reject
        # an exact fit with a nonzero contrast)
        floor = _degenerate.RTOL * _mean_square(Y_flat).reshape(N, N)
        stat = _degenerate.t_ratio(c_beta, c_XtX_inv_c * sigma2, c_XtX_inv_c * floor)
    else:
        raise ValueError(
            f"Invalid stat_type: '{stat_type_str}'. "
            f"Must be one of: {[s.value for s in GLMStatType]}"
        )

    # Separate positive and negative effects
    positive = np.where(stat > 0, stat, 0.0)
    negative = np.where(stat < 0, -stat, 0.0)

    return make_tail_result(positive, negative)


# =============================================================================
# Design matrix builder
# =============================================================================

def build_design_matrix(
    interest: npt.NDArray[np.float64],
    confounds: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build design matrix and contrast vector from interest and confound variables.

    Constructs X = [intercept, confounds, interest] and contrast = [0, ..., 0, 1]
    targeting the last column (interest variable).

    Parameters
    ----------
    interest : ndarray of shape (n_subjects,)
        Variable of interest (continuous or binary).
    confounds : ndarray of shape (n_subjects,) or (n_subjects, q), optional
        Confound variables. If 1D, treated as single confound.

    Returns
    -------
    X : ndarray of shape (n_subjects, p)
        Design matrix with intercept, confounds, and interest.
    contrast : ndarray of shape (p,)
        Contrast vector [0, ..., 0, 1] targeting the interest column.
    """
    n = len(interest)
    interest = check_finite_design(interest, "interest")
    if interest.ndim == 1:
        interest = interest[:, np.newaxis]

    intercept = np.ones((n, 1), dtype=np.float64)

    if confounds is not None:
        confounds = check_finite_design(confounds, "confounds")
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        X = np.hstack([intercept, confounds, interest])
    else:
        X = np.hstack([intercept, interest])

    contrast = np.zeros(X.shape[1], dtype=np.float64)
    contrast[-1] = 1.0

    return X, contrast


# =============================================================================
# Freedman-Lane permutation helpers
# =============================================================================

def _compute_reduced_model(
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Reduced (null-hypothesis) design for Freedman-Lane permutation.

    Under ``H0: C beta = 0`` the fitted means are ``X beta`` with ``beta`` in
    the null space of ``C``, so the reduced design is ``X0 = X null(C)``
    (Winkler et al. 2014). For contrasts that pick single columns this spans
    the remaining columns, as before; for a general contrast such as
    ``beta_x - beta_z`` it keeps the common effect ``beta_x = beta_z`` in the
    null model, which deleting every touched column wrongly removed. When
    ``C`` has full column rank (every column tested) ``X0`` has no columns.

    Parameters
    ----------
    X : ndarray of shape (n_subjects, p)
        Full design matrix.
    contrast : ndarray of shape (p,) or (r, p)
        Contrast vector (1D, t-test) or contrast matrix (2D, F-test).

    Returns
    -------
    X_reduced : ndarray of shape (n_subjects, p - rank(C))
        Reduced design matrix (possibly with zero columns).
    """
    from scipy.linalg import null_space

    C = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
    return X @ null_space(C)


def _reduced_fit(
    X_reduced: npt.NDArray[np.float64],
    Y_flat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fitted values of the reduced model (zero when it has no columns)."""
    if X_reduced.shape[1] == 0 or not np.any(X_reduced):
        return np.zeros_like(Y_flat)
    return X_reduced @ (np.linalg.pinv(X_reduced) @ Y_flat)


_SCHEMES = ("auto", "permutation", "sign-flip")


def _resolve_scheme(
    X: npt.NDArray[np.float64],
    X_reduced: npt.NDArray[np.float64],
    scheme: str,
) -> str:
    """Row permutation or sign flipping of the reduced-model residuals.

    ``'auto'`` uses sign flips when the full model contains the constant and
    the null model does not — the tested effect then involves the mean (a
    one-sample or paired-difference GLM), which no row permutation can
    change. Sign flips assume symmetric errors; row permutations assume
    exchangeable errors (Winkler et al. 2014).
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}; got {scheme!r}.")
    if scheme != "auto":
        return scheme
    ones = np.ones(X.shape[0])

    def spans_constant(A: npt.NDArray[np.float64]) -> bool:
        if A.shape[1] == 0 or not np.any(A):
            return False
        r = ones - A @ (np.linalg.pinv(A) @ ones)
        return bool(np.linalg.norm(r) < 1e-8 * np.sqrt(ones.size))

    return "sign-flip" if spans_constant(X) and not spans_constant(X_reduced) else "permutation"


def _shuffle_residuals(
    residuals: npt.NDArray[np.float64],
    rng: np.random.RandomState,
    scheme: str,
    strata_codes: Optional[npt.NDArray[np.int_]],
) -> npt.NDArray[np.float64]:
    """One Freedman-Lane draw: permuted (within strata) or sign-flipped residuals."""
    n = residuals.shape[0]
    if scheme == "sign-flip":
        signs = rng.choice([1.0, -1.0], n)
        return residuals * signs.reshape((n,) + (1,) * (residuals.ndim - 1))
    perm_idx = rng.permutation(n) if strata_codes is None else _stratified_perm(strata_codes, rng)
    return residuals[perm_idx]


def _freedman_lane_permutation_task(
    Y_hat_reduced: npt.NDArray[np.float64],
    residuals_reduced: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    X_pinv: npt.NDArray[np.float64],
    XtX_inv_diag: npt.NDArray[np.float64],
    XtX_inv: npt.NDArray[np.float64],
    reference_shape: Tuple[int, ...],
    stat_type: str,
    enhance_func: Optional[Callable] = None,
    seed: Optional[int] = None,
    strata_codes: Optional[npt.NDArray[np.int_]] = None,
    scheme: str = "permutation",
    **enhance_kwargs,
) -> Dict[str, np.float64]:
    """
    Single Freedman-Lane permutation step.

    1. Permute reduced-model residuals across subjects (within-stratum
       only when ``strata_codes`` is provided — PALM ``-eb`` semantics).
    2. Reconstruct Y_perm = Y_hat_reduced + permuted residuals
    3. Fit full model to Y_perm
    4. Optionally apply enhancement
    5. Return max statistics

    Parameters
    ----------
    Y_hat_reduced : ndarray of shape (n_subjects, N, N)
        Predicted values from reduced model.
    residuals_reduced : ndarray of shape (n_subjects, N, N)
        Residuals from reduced model.
    X, contrast, X_pinv, XtX_inv_diag, XtX_inv : precomputed GLM quantities
    reference_shape : tuple
        Shape (N, N) for _extract_max_stats.
    stat_type : str
        'tstat' or 'beta'.
    enhance_func : callable, optional
        Enhancement function to apply to GLM stats.
    seed : int
        Random seed for this permutation.
    strata_codes : ndarray of shape (n_subjects,), optional
        Integer stratum codes for within-block residual permutation.
    **enhance_kwargs
        Keyword arguments for enhance_func.

    Returns
    -------
    dict
        Maximum statistics per direction.
    """
    rng = np.random.RandomState(seed)
    Y_perm = Y_hat_reduced + _shuffle_residuals(residuals_reduced, rng, scheme, strata_codes)

    stat_dict = compute_glm_stat(
        Y_perm, X, contrast,
        stat_type=stat_type,
        X_pinv=X_pinv,
        XtX_inv_diag=XtX_inv_diag,
        XtX_inv=XtX_inv,
    )

    if enhance_func is not None:
        stat_dict = enhance_func(stat_dict, **enhance_kwargs)

    return _extract_max_stats(stat_dict, reference_shape)


# =============================================================================
# Enhancement registry
# =============================================================================
# Wrappers live in enhance.py and are shared with the t-test pipeline —
# see pairwise_stats.compute_p_val.

_ENHANCE_MAP = {
    StatMethod.TSTAT: None,  # No enhancement
    StatMethod.TFNBS: apply_tfnbs,
    StatMethod.NBS: apply_nbs,
    StatMethod.CNBS: apply_cnbs,
    StatMethod.NI_TFNBS: apply_ni_tfnbs,
    StatMethod.FBC_TFNBS: apply_fbc_tfnbs,
}


# =============================================================================
# Main orchestrator
# =============================================================================

def compute_p_val_glm(
    Y: npt.NDArray[np.float64],
    # --- Advanced API ---
    design_matrix: Optional[npt.NDArray[np.float64]] = None,
    contrast: Optional[npt.NDArray[np.float64]] = None,
    # --- Convenience API ---
    interest: Optional[npt.NDArray[np.float64]] = None,
    confounds: Optional[npt.NDArray[np.float64]] = None,
    # --- Common parameters ---
    stat_type: Union[str, GLMStatType] = GLMStatType.TSTAT,
    method: Union[str, StatMethod] = StatMethod.TFNBS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    two_tailed: bool = False,
    acceleration: Optional[str] = DEFAULT_ACCELERATION,
    use_mp: bool = True,
    rng: RngLike = None,
    random_state: Optional[int] = None,  # deprecated alias for `rng`
    n_processes: Optional[int] = None,
    verbose: bool = False,
    # --- Method-specific parameters ---
    net_labels: Optional[npt.NDArray[np.int_]] = None,
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    normalization: str = "sqrt",
    strata: Optional[npt.NDArray[Any]] = None,
    missing: str = "raise",
    min_coverage: float = 0.8,
    scheme: str = "auto",
    cnbs_correction: str = "bonferroni",
    design_columns: Optional[Sequence[str]] = None,
) -> Union[InferenceResult, OmnibusInferenceResult]:
    """
    Compute p-values for connectivity data using GLM with Freedman-Lane permutation.

    Supports two APIs that share the same core engine:

    - **Advanced API**: provide ``design_matrix`` and ``contrast`` directly.
    - **Convenience API**: provide ``interest`` (and optional ``confounds``);
      the design matrix and contrast are built automatically.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Connectivity matrices (symmetric, zero diagonal).
    design_matrix : ndarray of shape (n_subjects, p), optional
        Full design matrix (advanced API). Mutually exclusive with ``interest``.
    contrast : ndarray of shape (p,) or (r, p), optional
        Contrast vector (1D, t/beta) or contrast matrix (2D, F-stat).
        Required with ``design_matrix``. For ``stat_type='fstat'``, pass a
        2D matrix whose rows encode the linear combinations being tested
        jointly (e.g., ``[[1, -1, 0], [1, 0, -1]]`` tests β₁=β₂=β₃).
    interest : ndarray of shape (n_subjects,), optional
        Variable of interest (convenience API). Mutually exclusive with
        ``design_matrix``. Produces a single-column contrast — use the
        advanced API for multi-row F-contrasts.
    confounds : ndarray of shape (n_subjects,) or (n_subjects, q), optional
        Confound variables (convenience API).
    stat_type : {'tstat', 'beta', 'fstat'} or GLMStatType, default='tstat'
        Type of statistic to compute. ``'fstat'`` enables an omnibus F-test
        for a multi-row contrast matrix and returns a single ``'omnibus'``
        p-value map (no positive/negative split).
    method : str or StatMethod, default='tfnbs'
        Enhancement method.
    n_permutations : int, default=1000
        Number of permutations for null distribution.
    two_tailed : bool, default=False
        If False (default), per-tail FWER control (separate null for positive
        and negative). If True, joint null from max(max_positive, max_negative).
    acceleration : {'gpd', 'gamma'} or None, default=None
        Permutation acceleration method (Winkler et al., 2016). If None,
        use standard empirical p-values. 'gpd' fits a Generalized Pareto
        Distribution to the tail of the null distribution, 'gamma' a gamma
        distribution by the method of moments; either replaces only the
        p-values that fewer than 10 null maxima reach. Both are
        approximation accelerators with empirical fallback; use
        ``acceleration=None`` for exact empirical finite-permutation
        p-values.
    use_mp : bool, default=True
        Use multiprocessing for permutation testing.
    random_state : int, optional
        Random seed for reproducibility.
    n_processes : int, optional
        Number of CPU cores for parallel computing.
    net_labels : ndarray of shape (N,), optional
        Network labels (required for cnbs, ni_tfnbs, fbc_tfnbs).
    threshold, nbs_stat, n, start_thres, min_cluster_size, normalization
        Method-specific parameters (see ``compute_p_val`` docstring).
    e, h : float or sequence of float
        TFNBS extent and height exponents. Pass equal-length sequences to
        evaluate a whole ``(E, H)`` grid in one permutation pass. In grid
        mode the returned :class:`InferenceResult` carries the parameter
        axis (``result.is_grid``, ``result.e_grid``, ``result.h_grid``);
        use ``result.select(param_idx)`` or pass ``param_idx=`` to
        ``significant_edges`` / ``to_csv`` to project to a single cell.
    missing : {'raise', 'available_case'}, default='raise'
        Missing connectivity values (NaN). ``'raise'`` stops with a report.
        ``'available_case'`` fits each edge with the subjects who have it:
        edges are grouped by missingness pattern, each pattern gets its own
        full and reduced fits, and every pattern uses the global (within-
        stratum) permutation restricted to its subjects, so the shuffles
        stay coherent and equal the standard ones on complete data. Edges
        kept by fewer than ``min_coverage`` of the subjects, with fewer than
        2 residual degrees of freedom, or with a contrast that is not
        estimable on their subjects are not tested (p = NaN); see
        ``result.missing_report``. Symmetric connectivity only.
    min_coverage : float, default=0.8
        Minimum share of subjects an edge needs under
        ``missing='available_case'``.
    scheme : {'auto', 'permutation', 'sign-flip'}, default='auto'
        How the reduced-model residuals are resampled. ``'permutation'``
        shuffles subjects (within ``strata``; exchangeable errors).
        ``'sign-flip'`` flips their signs (symmetric errors; ``strata`` has
        no effect). ``'auto'`` sign-flips when the tested effect involves
        the mean — the full design contains the constant but the null model
        does not, as in a one-sample or paired-difference GLM, where no row
        permutation can change the tested mean — and permutes otherwise.
    cnbs_correction : {'bonferroni', 'bh'}, default='bonferroni'
        For ``method='cnbs'`` (block-level inference, Noble & Scheinost,
        2020): correction of the block p-values across blocks. With
        ``two_tailed=True`` each block's test is two-sided.
    design_columns : sequence of str, optional
        Names of the design columns, stored with the coefficient maps
        (``result.beta``, ``result.design``). Defaults: ``intercept``,
        ``confound_1`` ... and ``interest`` for the convenience API, else
        ``intercept`` / ``x1`` ...

    Returns
    -------
    InferenceResult or OmnibusInferenceResult
        For ``stat_type`` in ``{'tstat', 'beta'}``: an
        :class:`InferenceResult` with ``positive`` and ``negative`` p-value
        maps of shape ``(N, N)`` or ``(N, N, K)`` in grid mode. For
        ``stat_type='fstat'``: an :class:`OmnibusInferenceResult` with a
        single ``omnibus`` map of shape ``(N, N)``.

    Raises
    ------
    ValueError
        If API arguments are inconsistent or contrast is invalid.
    """
    _t0 = time.perf_counter()

    # ---- Resolve rng → seed ----
    if random_state is not None and rng is None:
        warn_legacy_random_state("random_state")
    random_state = resolve_seed(rng, legacy_random_state=random_state)

    if missing not in ("raise", "available_case"):
        raise ValueError(f"missing must be 'raise' or 'available_case'; got {missing!r}.")
    Y = check_connectivity(Y, "Y", allow_missing=missing == "available_case")

    # ---- Resolve API mode ----
    if design_matrix is not None and interest is not None:
        raise ValueError(
            "Provide either (design_matrix, contrast) or (interest, confounds), "
            "not both."
        )
    if design_matrix is None and interest is None:
        raise ValueError(
            "Must provide either design_matrix or interest."
        )

    if interest is not None:
        X, contrast_vec = build_design_matrix(interest, confounds)
        if design_columns is None:
            design_columns = (["intercept"] + [f"confound_{k + 1}" for k in range(X.shape[1] - 2)]
                              + ["interest"])
    else:
        X = check_finite_design(design_matrix, "design_matrix")
        if contrast is None:
            raise ValueError("contrast is required when using design_matrix.")
        contrast_vec = np.asarray(contrast, dtype=np.float64)

    # ---- Validate inputs ----
    n_subjects = Y.shape[0]
    N = Y.shape[1]
    p = X.shape[1]

    if X.shape[0] != n_subjects:
        raise ValueError(
            f"Design matrix has {X.shape[0]} rows but Y has {n_subjects} subjects."
        )

    # Contrast may be 1D (t/beta) or 2D (F-stat). Check the column dimension.
    if contrast_vec.ndim == 1:
        if contrast_vec.shape[0] != p:
            raise ValueError(
                f"Contrast vector length ({contrast_vec.shape[0]}) must match "
                f"number of columns in design matrix ({p})."
            )
    elif contrast_vec.ndim == 2:
        if contrast_vec.shape[1] != p:
            raise ValueError(
                f"Contrast matrix has {contrast_vec.shape[1]} columns but design "
                f"matrix has {p}."
            )
    else:
        raise ValueError(
            f"Contrast must be 1D or 2D, got ndim={contrast_vec.ndim}."
        )
    if np.all(contrast_vec == 0):
        raise ValueError("Contrast must have at least one non-zero entry.")
    _check_estimable(contrast_vec, X)

    if acceleration is not None and acceleration not in ("gpd", "gamma"):
        raise ValueError(
            f"Invalid acceleration: '{acceleration}'. Must be 'gpd', 'gamma', or None."
        )

    stat_type_str = stat_type.value if isinstance(stat_type, GLMStatType) else stat_type
    method_str = method.value if isinstance(method, StatMethod) else method

    try:
        method_enum = StatMethod(method_str)
    except ValueError:
        valid = [m.value for m in StatMethod if m in _ENHANCE_MAP]
        raise ValueError(f"Invalid method: '{method_str}'. Must be one of: {valid}")

    if method_enum not in _ENHANCE_MAP:
        raise ValueError(
            f"Method '{method_str}' is not supported in the GLM pipeline. "
            f"Use compute_p_val() for parametric baselines."
        )

    _warn_beta_threshold_scale(stat_type_str, method_enum)

    if method_enum in CONSTRAINED_METHODS and net_labels is None:
        raise ValueError(
            f"Method '{method_str}' requires net_labels."
        )

    # ---- Build enhancement kwargs ----
    enhance_kwargs: Dict[str, Any] = {}
    if method_enum == StatMethod.NBS:
        enhance_kwargs = {"threshold": threshold, "nbs_stat": nbs_stat}
    elif method_enum in {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}:
        enhance_kwargs = {"e": e, "h": h, "n": n, "start_thres": start_thres}
        if method_enum in CONSTRAINED_METHODS:
            enhance_kwargs["net_labels"] = net_labels
        if method_enum == StatMethod.FBC_TFNBS:
            enhance_kwargs["min_cluster_size"] = min_cluster_size
        if method_enum == StatMethod.NI_TFNBS:
            enhance_kwargs["normalization"] = normalization
    elif method_enum == StatMethod.CNBS:
        from ._cnbs import check_correction
        check_correction(cnbs_correction)
        enhance_kwargs = {"net_labels": net_labels}
    block_inference = (dict(net_labels=net_labels, correction=cnbs_correction)
                       if method_enum == StatMethod.CNBS else None)

    enhance_func = _ENHANCE_MAP[method_enum]

    # ---- Missing values (NaN): available-case analysis ----
    if missing == "available_case" and np.isnan(Y).any():
        from ._available_case_glm import available_case_glm
        from .pairwise_stats import _grid_from_kwargs

        p_ac, raw_ac, report = available_case_glm(
            Y, X, contrast_vec, _compute_reduced_model(X, contrast_vec),
            stat_type_str, enhance_func, enhance_kwargs,
            n_permutations=n_permutations, random_state=random_state,
            use_mp=use_mp, n_processes=n_processes, verbose=verbose,
            strata=strata, acceleration=acceleration, two_tailed=two_tailed,
            min_coverage=min_coverage,
            scheme=_resolve_scheme(X, _compute_reduced_model(X, contrast_vec), scheme),
            block_inference=block_inference,
        )
        block_report = report.pop("block_report", None)
        beta_ac = report.pop("beta")
        design = _estimates.glm_design(X, contrast_vec, design_columns)
        design["available_case"] = ("each edge fitted with its own subjects; predictions use the "
                                    "full-sample column means")
        method_str = method.value if hasattr(method, "value") else str(method)
        if stat_type_str == GLMStatType.FSTAT.value:
            return OmnibusInferenceResult(
                p_ac["omnibus"], method=method_str, n_permutations=n_permutations,
                acceleration=acceleration, wall_time_s=time.perf_counter() - _t0,
                stat_omnibus=raw_ac["omnibus"], stat_type="fstat",
                strata_provided=strata is not None, missing_report=report,
                block_report=block_report, beta=beta_ac, design=design,
            )
        e_grid, h_grid = _grid_from_kwargs(method_enum, enhance_kwargs)
        return InferenceResult(
            p_ac["positive"], p_ac["negative"],
            method=method_str, n_permutations=n_permutations,
            acceleration=acceleration, wall_time_s=time.perf_counter() - _t0,
            stat_positive=raw_ac["positive"], stat_negative=raw_ac["negative"],
            stat_type=stat_type_str, strata_provided=strata is not None,
            e_grid=e_grid, h_grid=h_grid, missing_report=report,
            block_report=block_report,
            two_tailed=two_tailed, beta=beta_ac, design=design,
        )

    # An exact fit with a nonzero contrast has an infinite t (or F)
    _degenerate.raise_if_infinite(_infinite_edges(Y, X, contrast_vec, stat_type_str),
                                  "F" if stat_type_str == GLMStatType.FSTAT.value else "t")

    # ---- Precompute OLS ----
    X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)
    # Coefficient maps of the observed fit, saved with the p-maps
    beta = (X_pinv @ Y.reshape(n_subjects, -1)).reshape(X.shape[1], N, N)
    design = _estimates.glm_design(X, contrast_vec, design_columns)

    # ---- Compute observed statistics ----
    emp_stat_dict = compute_glm_stat(
        Y, X, contrast_vec,
        stat_type=stat_type_str,
        X_pinv=X_pinv,
        XtX_inv_diag=XtX_inv_diag,
        XtX_inv=XtX_inv,
    )
    # Keep the raw (pre-enhancement) stat dict so InferenceResult can
    # surface the t/β/F statistic map even when the enhanced score is a
    # multi-(E, H) tensor.
    raw_stat_dict = emp_stat_dict
    if enhance_func is not None:
        emp_stat_dict = enhance_func(emp_stat_dict, **enhance_kwargs)

    # ---- Freedman-Lane: compute reduced model ----
    X_reduced = _compute_reduced_model(X, contrast_vec)
    scheme_used = _resolve_scheme(X, X_reduced, scheme)

    Y_flat = Y.reshape(n_subjects, -1)
    Y_hat_reduced_flat = _reduced_fit(X_reduced, Y_flat)
    Y_hat_reduced = Y_hat_reduced_flat.reshape(n_subjects, N, N)
    residuals_reduced = Y - Y_hat_reduced

    reference_shape = (N, N)

    # ---- Encode strata for within-block exchangeability (PALM -eb) ----
    strata_codes = None
    if strata is not None:
        strata_codes = _encode_strata(strata)
        if strata_codes.shape[0] != n_subjects:
            raise ValueError(
                f"strata length {strata_codes.shape[0]} does not match "
                f"n_subjects={n_subjects}."
            )

    # ---- Generate seeds ----
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    # ---- Build permutation task ----
    task_func = partial(
        _freedman_lane_permutation_task,
        Y_hat_reduced,
        residuals_reduced,
        X,
        contrast_vec,
        X_pinv,
        XtX_inv_diag,
        XtX_inv,
        reference_shape,
        stat_type_str,
        enhance_func,
        strata_codes=strata_codes,
        scheme=scheme_used,
        **enhance_kwargs,
    )

    # ---- Run permutations ----
    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="glm_perm",
    )

    max_null_dict = _collect_results_to_arrays(results, n_permutations)

    # ---- Two-tailed FWER ----
    if two_tailed:
        if stat_type_str == GLMStatType.FSTAT.value:
            # F is non-negative; there is no sign to split and nothing to pool.
            # Silently ignore rather than raise — keeps the API uniform for
            # callers that pass two_tailed=True by default.
            pass
        else:
            # Build joint null from max(max_positive, max_negative)
            joint_null = np.maximum(
                max_null_dict["positive"],
                max_null_dict["negative"],
            )
            max_null_dict = {
                "positive": joint_null,
                "negative": joint_null,
            }

    # ---- Compute p-values ----
    block_report = None
    if block_inference is not None:
        from ._cnbs import block_p_values
        result, block_report = block_p_values(
            emp_stat_dict, max_null_dict, net_labels, N,
            correction=cnbs_correction, acceleration=acceleration, two_tailed=two_tailed,
        )
    elif acceleration is not None:
        result = compute_p_values_accelerated(
            emp_stat_dict, max_null_dict, method=acceleration,
        )
    else:
        result = _compute_p_values_from_null(emp_stat_dict, max_null_dict)
    method_str = method.value if hasattr(method, "value") else str(method)
    from .pairwise_stats import _grid_from_kwargs
    e_grid, h_grid = _grid_from_kwargs(method_enum, enhance_kwargs)
    if set(result.keys()) == {"positive", "negative"}:
        return InferenceResult(
            result["positive"], result["negative"],
            method=method_str, n_permutations=n_permutations,
            acceleration=acceleration,
            wall_time_s=time.perf_counter() - _t0,
            stat_positive=raw_stat_dict["positive"],
            stat_negative=raw_stat_dict["negative"],
            stat_type=stat_type_str,
            strata_provided=strata is not None,
            e_grid=e_grid,
            h_grid=h_grid,
            block_report=block_report,
            two_tailed=two_tailed,
            beta=beta,
            design=design,
        )
    # F-stat path: single 'omnibus' map → OmnibusInferenceResult
    return OmnibusInferenceResult(
        result["omnibus"],
        method=method_str, n_permutations=n_permutations,
        acceleration=acceleration,
        wall_time_s=time.perf_counter() - _t0,
        stat_omnibus=raw_stat_dict["omnibus"],
        stat_type="fstat",
        strata_provided=strata is not None,
        block_report=block_report,
        beta=beta,
        design=design,
    )


# =============================================================================
# Multi-contrast GLM (shared nuisance, single permutation pass)
# =============================================================================

def _multi_contrast_perm_task(
    nulls: Tuple[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str], ...],
    X: npt.NDArray[np.float64],
    contrast_names: Tuple[str, ...],
    contrast_arrays: Tuple[npt.NDArray[np.float64], ...],
    X_pinv: npt.NDArray[np.float64],
    XtX_inv_diag: npt.NDArray[np.float64],
    XtX_inv: npt.NDArray[np.float64],
    reference_shape: Tuple[int, ...],
    stat_type: str,
    enhance_func: Optional[Callable],
    enhance_kwargs: Dict[str, Any],
    seed: Optional[int] = None,
    strata_codes: Optional[npt.NDArray[np.int_]] = None,
) -> Dict[str, Dict[str, np.float64]]:
    """One Freedman-Lane permutation, evaluated for every contrast.

    ``nulls[k]`` is contrast k's reduced-model fit, residuals and resampling
    scheme. Each contrast restarts the generator from ``seed``, so all
    contrasts see the same permutation (or sign flips) and each result equals
    a single-contrast :func:`compute_p_val_glm` run with the same seed.
    Returns a nested dict ``{contrast_name: {tail: max_stat}}``.
    """
    out: Dict[str, Dict[str, np.float64]] = {}
    for name, contrast, (Y_hat_reduced, residuals_reduced, scheme) in zip(
            contrast_names, contrast_arrays, nulls):
        rng = np.random.RandomState(seed)
        Y_perm = Y_hat_reduced + _shuffle_residuals(residuals_reduced, rng, scheme, strata_codes)
        stat_dict = compute_glm_stat(
            Y_perm, X, contrast,
            stat_type=stat_type,
            X_pinv=X_pinv,
            XtX_inv_diag=XtX_inv_diag,
            XtX_inv=XtX_inv,
        )
        if enhance_func is not None:
            stat_dict = enhance_func(stat_dict, **enhance_kwargs)
        out[name] = _extract_max_stats(stat_dict, reference_shape)
    return out


def compute_p_val_glm_multi(
    Y: npt.NDArray[np.float64],
    design_matrix: npt.NDArray[np.float64],
    contrasts: Dict[str, npt.NDArray[np.float64]],
    *,
    nuisance_contrast: Optional[npt.NDArray[np.float64]] = None,
    stat_type: Union[str, GLMStatType] = GLMStatType.TSTAT,
    method: Union[str, StatMethod] = StatMethod.TFNBS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    two_tailed: bool = False,
    acceleration: Optional[str] = DEFAULT_ACCELERATION,
    use_mp: bool = True,
    rng: RngLike = None,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    verbose: bool = False,
    # --- Method-specific parameters (forwarded to enhancement) ---
    net_labels: Optional[npt.NDArray[np.int_]] = None,
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    normalization: str = "sqrt",
    strata: Optional[npt.NDArray[Any]] = None,
    missing: str = "raise",
    min_coverage: float = 0.8,
    scheme: str = "auto",
    cnbs_correction: str = "bonferroni",
    design_columns: Optional[Sequence[str]] = None,
) -> Dict[str, InferenceResult]:
    """Run several contrasts of one design in a single permutation pass.

    For studies with several predictors of interest in one model (e.g.
    testing age, sex and motion, each adjusted for the others), every
    contrast is tested against its own Freedman-Lane null: the reduced model
    ``X null(c)`` keeps the other predictors as nuisance, so a real effect of
    one predictor stays in the fitted part when another is tested. The
    permutations are shared (one seed per permutation, the same draw for
    every contrast), so each result equals :func:`compute_p_val_glm` for that
    contrast with the same ``rng``; the pass shares the design
    pseudoinverse and the worker pool.

    Contrasts are corrected separately (one family per contrast name, and
    per ``(e, h)`` cell); selecting the most significant contrast needs an
    extra correction.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Connectivity tensor.
    design_matrix : ndarray of shape (n_subjects, p)
        Full design matrix (intercept + all regressors of interest +
        nuisance).
    contrasts : dict[str, ndarray]
        Named 1D contrast vectors (length ``p``). For a 4-column
        ``[intercept, age, sex, motion]`` design, pass
        ``{'age': [0,1,0,0], 'sex': [0,0,1,0], 'motion': [0,0,0,1]}``.
        F-stat / multi-row contrasts are not supported here — use
        :func:`compute_p_val_glm` for those.
    nuisance_contrast : ndarray, optional
        Request a *joint* null instead: one reduced model, shared by all
        contrasts, that excludes every column touched by
        ``nuisance_contrast``. The permuted data then carry none of those
        columns' effects (the null is "all of them are zero"), not the null
        of each coefficient; with a real effect on another excluded column
        this is conservative (raw beta, 1,000 simulated datasets: FWER
        0.017-0.025 vs 0.050-0.060 with the default). Default: None
        (per-contrast nulls).
    stat_type : ``'tstat'`` or ``'beta'``
        F-stat is unsupported here.
    method, two_tailed, acceleration, n_permutations, use_mp, rng,
    n_processes, verbose, net_labels, threshold, nbs_stat, e, h, n,
    start_thres, min_cluster_size, normalization
        Forwarded to the underlying GLM pipeline (see
        :func:`compute_p_val_glm`).
    missing, min_coverage, scheme
        See :func:`compute_p_val_glm`. With missing values and
        ``missing='available_case'``, each contrast is run separately with
        the same seed (so every contrast sees the same permutations) and
        tests its predictor against its own nuisance model.

    Returns
    -------
    dict[str, InferenceResult]
        One :class:`InferenceResult` per contrast name. Wall-time on
        each result reflects the *amortised* cost (total wall-time / K),
        and ``n_permutations`` is the shared count.
    """
    _t0 = time.perf_counter()

    if random_state is not None and rng is None:
        warn_legacy_random_state("random_state")
    random_state = resolve_seed(rng, legacy_random_state=random_state)

    if missing not in ("raise", "available_case"):
        raise ValueError(f"missing must be 'raise' or 'available_case'; got {missing!r}.")
    Y = check_connectivity(Y, "Y", allow_missing=missing == "available_case")

    if not contrasts:
        raise ValueError("contrasts must be a non-empty dict.")
    if missing == "available_case" and np.isnan(Y).any():
        return {
            name: compute_p_val_glm(
                Y, design_matrix=design_matrix, contrast=c, stat_type=stat_type,
                method=method, n_permutations=n_permutations, two_tailed=two_tailed,
                acceleration=acceleration, use_mp=use_mp, rng=random_state,
                n_processes=n_processes, verbose=verbose, net_labels=net_labels,
                threshold=threshold, nbs_stat=nbs_stat, e=e, h=h, n=n,
                start_thres=start_thres, min_cluster_size=min_cluster_size,
                normalization=normalization, strata=strata,
                missing="available_case", min_coverage=min_coverage, scheme=scheme,
                cnbs_correction=cnbs_correction, design_columns=design_columns,
            )
            for name, c in contrasts.items()
        }

    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
        raise ValueError(
            f"Y must have shape (n_subjects, N, N), got {Y.shape}."
        )
    n_subjects, N, _ = Y.shape

    X = check_finite_design(design_matrix, "design_matrix")
    if X.shape[0] != n_subjects:
        raise ValueError(
            f"design_matrix has {X.shape[0]} rows but Y has {n_subjects} subjects."
        )
    p = X.shape[1]

    # Validate contrasts: all 1D, all length p
    contrast_names: List[str] = list(contrasts.keys())
    contrast_arrays: List[npt.NDArray[np.float64]] = []
    for name in contrast_names:
        c = np.asarray(contrasts[name], dtype=np.float64)
        if c.ndim != 1 or c.shape[0] != p:
            raise ValueError(
                f"contrast {name!r}: must be 1D of length {p}, got shape {c.shape}."
            )
        _check_estimable(c, X, f"Contrast {name!r}")
        contrast_arrays.append(c)

    stat_type_str = (
        stat_type.value if isinstance(stat_type, GLMStatType) else stat_type
    )
    if stat_type_str == GLMStatType.FSTAT.value:
        raise ValueError(
            "F-stat is unsupported in compute_p_val_glm_multi (per-contrast "
            "directional split is undefined). Call compute_p_val_glm once "
            "per multi-row contrast for omnibus F tests."
        )

    method_str = method.value if isinstance(method, StatMethod) else method
    try:
        method_enum = StatMethod(method_str)
    except ValueError:
        valid = [m.value for m in StatMethod]
        raise ValueError(f"Invalid method: {method_str!r}. Must be one of {valid}.")
    if method_enum not in _ENHANCE_MAP:
        valid = [m.value for m in _ENHANCE_MAP]
        raise ValueError(
            f"Method {method_str!r} is not supported in the GLM multi-contrast "
            f"pipeline. Must be one of {valid}."
        )

    _warn_beta_threshold_scale(stat_type_str, method_enum)

    if method_enum in CONSTRAINED_METHODS and net_labels is None:
        raise ValueError(
            f"Method {method_str!r} requires net_labels to be provided."
        )

    enhance_func = _ENHANCE_MAP.get(method_enum)
    enhance_kwargs: Dict[str, Any] = {}
    if method_enum == StatMethod.NBS:
        enhance_kwargs = {"threshold": threshold, "nbs_stat": nbs_stat}
    elif method_enum in {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}:
        enhance_kwargs = {"e": e, "h": h, "n": n, "start_thres": start_thres}
        if method_enum in CONSTRAINED_METHODS:
            enhance_kwargs["net_labels"] = net_labels
        if method_enum == StatMethod.FBC_TFNBS:
            enhance_kwargs["min_cluster_size"] = min_cluster_size
        if method_enum == StatMethod.NI_TFNBS:
            enhance_kwargs["normalization"] = normalization
    elif method_enum == StatMethod.CNBS:
        from ._cnbs import check_correction
        check_correction(cnbs_correction)
        enhance_kwargs = {"net_labels": net_labels}

    for name, c in zip(contrast_names, contrast_arrays):
        _degenerate.raise_if_infinite(_infinite_edges(Y, X, c, stat_type_str))

    # ---- Reduced models: one per contrast (X null(c)), or a shared joint null ----
    if nuisance_contrast is None:
        reduced_models = [_compute_reduced_model(X, c) for c in contrast_arrays]
    else:
        reduced_marker = np.asarray(nuisance_contrast, dtype=np.float64)
        if reduced_marker.ndim == 1 and reduced_marker.shape[0] != p:
            raise ValueError(
                f"nuisance_contrast length {reduced_marker.shape[0]} does not "
                f"match design width {p}."
            )
        # the marker names the columns of interest (non-zero in any row);
        # the joint null spans the other columns
        marked = np.flatnonzero(np.any(np.atleast_2d(reduced_marker) != 0, axis=0))
        reduced_models = [_compute_reduced_model(X, np.eye(p)[marked])] * len(contrast_arrays)
    X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)
    beta = (X_pinv @ Y.reshape(n_subjects, -1)).reshape(p, N, N)   # shared by every contrast

    # ---- Empirical stats per contrast ----
    emp_dicts: Dict[str, Dict[str, npt.NDArray[np.float64]]] = {}
    raw_stat_dicts: Dict[str, Dict[str, npt.NDArray[np.float64]]] = {}
    for name, contrast in zip(contrast_names, contrast_arrays):
        emp = compute_glm_stat(
            Y, X, contrast, stat_type=stat_type_str,
            X_pinv=X_pinv, XtX_inv_diag=XtX_inv_diag, XtX_inv=XtX_inv,
        )
        # Snapshot the raw (pre-enhancement) statistic for the result
        # object — the user-facing effect map should be t/β, not the
        # enhanced score.
        raw_stat_dicts[name] = emp
        if enhance_func is not None:
            emp = enhance_func(emp, **enhance_kwargs)
        emp_dicts[name] = emp

    # ---- Fixed-residuals reduced fits (one per distinct reduced model) ----
    Y_flat = Y.reshape(n_subjects, -1)
    nulls = []
    fitted: Dict[int, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]] = {}
    for X_reduced in reduced_models:
        if id(X_reduced) not in fitted:
            Y_hat_reduced = _reduced_fit(X_reduced, Y_flat).reshape(n_subjects, N, N)
            fitted[id(X_reduced)] = (Y_hat_reduced, Y - Y_hat_reduced,
                                     _resolve_scheme(X, X_reduced, scheme))
        nulls.append(fitted[id(X_reduced)])

    reference_shape = (N, N)

    # ---- Encode strata for within-block exchangeability (PALM -eb) ----
    strata_codes = None
    if strata is not None:
        strata_codes = _encode_strata(strata)
        if strata_codes.shape[0] != n_subjects:
            raise ValueError(
                f"strata length {strata_codes.shape[0]} does not match "
                f"n_subjects={n_subjects}."
            )

    # ---- Permutation seeds ----
    rng_local = np.random.RandomState(random_state)
    seeds = rng_local.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    task_func = partial(
        _multi_contrast_perm_task,
        tuple(nulls),
        X,
        tuple(contrast_names),
        tuple(contrast_arrays),
        X_pinv,
        XtX_inv_diag,
        XtX_inv,
        reference_shape,
        stat_type_str,
        enhance_func,
        enhance_kwargs,
        strata_codes=strata_codes,
    )

    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    perm_results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="glm_multi",
    )

    # ---- Re-shape into per-contrast null arrays ----
    null_max_per_contrast: Dict[str, Dict[str, npt.NDArray[np.float64]]] = {}
    for name in contrast_names:
        per_perm = [pr[name] for pr in perm_results]
        null_max_per_contrast[name] = _collect_results_to_arrays(
            per_perm, n_permutations
        )

    # ---- Two-tailed pooling ----
    if two_tailed:
        for name in contrast_names:
            joint = np.maximum(
                null_max_per_contrast[name]["positive"],
                null_max_per_contrast[name]["negative"],
            )
            null_max_per_contrast[name] = {"positive": joint, "negative": joint}

    # ---- Per-contrast p-values + InferenceResult ----
    out: Dict[str, InferenceResult] = {}
    elapsed = time.perf_counter() - _t0
    per_contrast_wall = elapsed / max(1, len(contrast_names))
    for name in contrast_names:
        emp = emp_dicts[name]
        null = null_max_per_contrast[name]
        block_report = None
        if method_enum == StatMethod.CNBS:
            from ._cnbs import block_p_values
            res, block_report = block_p_values(emp, null, net_labels, N, correction=cnbs_correction,
                                               acceleration=acceleration, two_tailed=two_tailed)
        elif acceleration is not None:
            res = compute_p_values_accelerated(emp, null, method=acceleration)
        else:
            res = _compute_p_values_from_null(emp, null)
        raw = raw_stat_dicts[name]
        out[name] = InferenceResult(
            res["positive"], res["negative"],
            method=method_str, n_permutations=n_permutations,
            acceleration=acceleration,
            wall_time_s=per_contrast_wall,
            stat_positive=raw["positive"],
            stat_negative=raw["negative"],
            stat_type=stat_type_str,
            strata_provided=strata is not None,
            block_report=block_report,
            two_tailed=two_tailed,
            beta=beta,
            design=_estimates.glm_design(X, contrast_arrays[contrast_names.index(name)], design_columns),
        )
    return out


# =============================================================================
# Paired-difference GLM convenience wrapper
# =============================================================================

def compute_p_val_paired_glm(
    Y_A: npt.NDArray[np.float64],
    Y_B: npt.NDArray[np.float64],
    confounds_A: Optional[npt.NDArray[np.float64]] = None,
    confounds_B: Optional[npt.NDArray[np.float64]] = None,
    **kwargs,
) -> Union[InferenceResult, OmnibusInferenceResult]:
    """
    Paired-difference GLM: test ``Y_A`` vs ``Y_B`` within-subject, optionally
    controlling for per-subject differences in confounds.

    Two routes depending on ``confounds_*``:

    - **No confounds** → delegates to
      :func:`conninfpy.compute_p_val` with ``test_type='paired'`` (sign-flip
      permutation). This is numerically equivalent to a one-sample GLM on
      the differences and avoids the Freedman-Lane degeneracy that occurs
      when the contrast targets the only column in the design matrix.
    - **With confounds** → builds ``Δ_Y = Y_B − Y_A`` and
      ``Δ_C = confounds_B − confounds_A``, then calls
      :func:`compute_p_val_glm` with ``design_matrix = [1, Δ_C]``,
      ``contrast = [1, 0, ..., 0]`` (intercept as the effect of interest).

    Use this when the same subjects are measured under two conditions (task
    A vs task B, pre vs post, open vs close) and a confound whose value
    *differs between the two conditions for the same subject* (e.g.,
    motion, arousal, response time) needs to be partialled out.

    Parameters
    ----------
    Y_A, Y_B : ndarray of shape (n_subjects, N, N)
        Aligned per-subject connectivity matrices for the two conditions.
        ``Y_A[s]`` and ``Y_B[s]`` must correspond to the same subject.
    confounds_A, confounds_B : ndarray of shape (n_subjects,) or (n_subjects, k), optional
        Aligned per-subject confound values for the two conditions. Either
        pass both or neither. If passed, the *difference*
        ``confounds_B - confounds_A`` enters the GLM as a nuisance regressor.
    **kwargs
        Forwarded to the downstream call (``compute_p_val`` or
        ``compute_p_val_glm``). ``test_type`` is fixed to ``'paired'`` when
        routing to the t-test pipeline and must not be supplied.

    Returns
    -------
    InferenceResult
        Directional ``'positive'`` / ``'negative'`` p-values. On both
        routes ``positive`` means ``Y_B > Y_A`` (B has greater
        connectivity), the same orientation as
        ``compute_p_val(Y_A, Y_B, test_type='paired')``; with confounds it
        is the ``Δ_Y = Y_B − Y_A`` intercept adjusted for ``Δ_C``. Legacy keys ``'g2>g1'`` / ``'g1>g2'`` remain accessible with a
        :class:`DeprecationWarning` until v2.1.

    Raises
    ------
    ValueError
        If ``Y_A`` and ``Y_B`` shapes mismatch, only one of the confound
        arrays is supplied, or confound shapes mismatch.
    """
    Y_A = np.asarray(Y_A, dtype=np.float64)
    Y_B = np.asarray(Y_B, dtype=np.float64)
    if Y_A.shape != Y_B.shape:
        raise ValueError(
            f"Y_A and Y_B must have identical shapes, got {Y_A.shape} vs {Y_B.shape}."
        )
    if Y_A.ndim != 3 or Y_A.shape[1] != Y_A.shape[2]:
        raise ValueError(
            f"Y_A/Y_B must have shape (n_subjects, N, N), got {Y_A.shape}."
        )

    has_A = confounds_A is not None
    has_B = confounds_B is not None
    if has_A != has_B:
        raise ValueError(
            "Provide either both confounds_A and confounds_B, or neither. "
            "Paired-difference inference requires matched confound values "
            "in both conditions."
        )

    # --- No-confound path: delegate to sign-flip paired test ---
    if not has_A:
        if "test_type" in kwargs:
            raise ValueError(
                "test_type is fixed to 'paired' by compute_p_val_paired_glm; "
                "do not pass it explicitly."
            )
        # GLM-only options: a mean test needs sign flips, and a statistic
        # other than t runs as the one-sample GLM on the differences (which
        # sign-flips; with t it equals the paired test exactly).
        scheme = kwargs.pop("scheme", "auto")
        if scheme == "permutation":
            raise ValueError(
                "A paired difference is a mean test: row permutation cannot change "
                "the tested mean. Use scheme='auto' or 'sign-flip'."
            )
        stat_type = kwargs.pop("stat_type", GLMStatType.TSTAT)
        stat_type_str = stat_type.value if isinstance(stat_type, GLMStatType) else stat_type
        if stat_type_str != GLMStatType.TSTAT.value:
            return compute_p_val_glm(
                Y_B - Y_A, design_matrix=np.ones((Y_A.shape[0], 1)),
                contrast=np.array([1.0]), stat_type=stat_type_str, scheme="sign-flip",
                **kwargs,
            )
        return compute_p_val(Y_A, Y_B, test_type="paired", **kwargs)

    # --- With-confound path: one-sample GLM on differences ---
    confounds_A = np.asarray(confounds_A, dtype=np.float64)
    confounds_B = np.asarray(confounds_B, dtype=np.float64)
    if confounds_A.shape != confounds_B.shape:
        raise ValueError(
            f"confounds_A and confounds_B must have identical shapes, "
            f"got {confounds_A.shape} vs {confounds_B.shape}."
        )
    if confounds_A.shape[0] != Y_A.shape[0]:
        raise ValueError(
            f"confounds have {confounds_A.shape[0]} subjects but Y_A has "
            f"{Y_A.shape[0]}."
        )

    # B − A, so `positive` means Y_B > Y_A exactly as on the no-confound
    # route (compute_p_val paired: positive = group2 > group1).
    Delta_Y = Y_B - Y_A
    Delta_C = confounds_B - confounds_A
    if Delta_C.ndim == 1:
        Delta_C = Delta_C[:, np.newaxis]

    n_subjects = Y_A.shape[0]
    intercept = np.ones((n_subjects, 1), dtype=np.float64)
    X = np.hstack([intercept, Delta_C])
    contrast = np.zeros(X.shape[1], dtype=np.float64)
    contrast[0] = 1.0  # test the intercept (= mean of Δ_Y)

    # Disallow caller from overriding the design; these keys conflict.
    for forbidden in ("design_matrix", "contrast", "interest", "confounds", "design_columns"):
        if forbidden in kwargs:
            raise ValueError(
                f"compute_p_val_paired_glm builds the design internally; "
                f"do not pass '{forbidden}'."
            )

    q = Delta_C.shape[1]
    res = compute_p_val_glm(
        Delta_Y,
        design_matrix=X,
        contrast=contrast,
        design_columns=["delta_intercept"] + [f"delta_confound_{k + 1}" for k in range(q)],
        **kwargs,
    )
    # Coefficients: [condition-A mean, delta model]. Profile B adds the change
    # with no confound change (the tested intercept) to the A mean.
    if getattr(res, "beta", None) is not None:
        with _estimates._quiet():
            mean_a = np.nanmean(Y_A, axis=0)
        res.beta = np.concatenate([mean_a[None], res.beta])
        zeros = [0.0] * q
        res.design = {
            "kind": "paired-glm", "columns": ["condition_A_mean"] + res.design["columns"],
            "column_means": [1.0] + res.design["column_means"],
            "contrast": [0.0, 1.0] + zeros,
            "profiles": {"A": [1.0, 0.0] + zeros, "B (no confound change)": [1.0, 1.0] + zeros,
                         "B (observed)": [1.0, 1.0] + Delta_C.mean(axis=0).tolist()},
            "comparison": ["A", "B (no confound change)"],
            "n_subjects": int(n_subjects), "scale": "unknown", "interest": None,
            "note": "B - A is modelled as delta_intercept + delta_confounds x (C_B - C_A); "
                    "the tested effect is the change with no confound change.",
        }
    return res
