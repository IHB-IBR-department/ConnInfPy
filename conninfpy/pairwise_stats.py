"""
Pairwise statistical testing module for network analysis.

This module provides functions for computing t-statistics and p-values
using permutation testing, with optional enhancement (TFNBS, NBS, cNBS,
NI-TFNBS, FBC-TFNBS) applied via shared ``apply_*`` wrappers from
``conninfpy.enhance``.

Main Functions
--------------
compute_p_val : Compute p-values using permutation testing (with enhancement)
compute_t_stat : Compute t-statistics for paired, one-sample, or two-sample tests
compute_null_dist : Compute null distribution via permutation testing
"""

from __future__ import annotations
import logging
from enum import Enum
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import stats

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
import time

from .acceleration import compute_p_values_accelerated
from ._compat import TailResult, make_tail_result, normalize_keys
from ._engine import (  # shared permutation machinery (re-exported)
    _encode_strata,
    _stratified_perm,
    _stratified_choice_n1,
    _extract_max_stats,
    _collect_results_to_arrays,
    _is_worker_process,
    get_available_cores,
    _compute_p_values_from_null,
    _bh_fdr_correction,
)
from ._progress import run_permutations
from .results import InferenceResult, make_inference_result
from . import _degenerate
from ._prepare import check_connectivity
from ._rng import RngLike, resolve_seed, warn_legacy_random_state


__all__ = [
    # Enums and constants
    "TestType",
    "StatMethod",
    "DEFAULT_N_PERMUTATIONS",
    "DEFAULT_EXTENT_EXPONENT",
    "DEFAULT_HEIGHT_EXPONENT",
    "DEFAULT_N_THRESHOLDS",
    # Main functions
    "compute_p_val",
    "compute_t_stat",
    "compute_null_dist",
    # T-statistic primitive (accepts pre-computed diffs)
    "compute_t_stat_diff",
    "get_available_cores",
    "_is_worker_process",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TestType(str, Enum):
    """Statistical test types for permutation testing."""

    PAIRED = "paired"
    """Paired samples t-test (within-subjects design)."""

    ONE_SAMPLE = "one-sample"
    """One-sample t-test against zero."""

    TWO_SAMPLE = "two-sample"
    """Independent samples t-test (between-subjects design)."""


class StatMethod(str, Enum):
    """Statistical method for network analysis."""

    TFNBS = "tfnbs"
    """Threshold-Free Network-Based Statistics (TFCE-style)."""

    TSTAT = "tstat"
    """Raw t-statistics without enhancement."""

    NBS = "nbs"
    """Classical Network-Based Statistics with fixed threshold."""

    CNBS = "cnbs"
    """Constrained NBS with predefined network partitions."""

    NI_TFNBS = "ni_tfnbs"
    """Network-Informed TFNBS with functional block density weighting."""

    FBC_TFNBS = "fbc_tfnbs"
    """Functional Block Clustering TFNBS (block-defined clustering)."""

    BONFERRONI = "bonferroni"
    """Parametric Bonferroni correction (no permutation testing)."""

    BH_FDR = "bh_fdr"
    """Parametric Benjamini-Hochberg FDR correction (no permutation testing)."""

    BH_FDR_PERM = "bh_fdr_perm"
    """Permutation-based BH-FDR correction (per-edge permutation p-values)."""


CONSTRAINED_METHODS = {StatMethod.CNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}

# Enhancement-function registry. Maps StatMethod → pure `stat_dict → stat_dict`
# wrapper from .enhance (shared with glm_stats.compute_p_val_glm). Methods
# without enhancement (raw t-stat, parametric baselines, bh_fdr_perm) map to None.
_ENHANCE_MAP = {
    StatMethod.TSTAT: None,
    StatMethod.TFNBS: apply_tfnbs,
    StatMethod.NBS: apply_nbs,
    StatMethod.CNBS: apply_cnbs,
    StatMethod.NI_TFNBS: apply_ni_tfnbs,
    StatMethod.FBC_TFNBS: apply_fbc_tfnbs,
    StatMethod.BONFERRONI: None,
    StatMethod.BH_FDR: None,
    StatMethod.BH_FDR_PERM: None,
}
PARAMETRIC_METHODS = {StatMethod.BONFERRONI, StatMethod.BH_FDR}

TFNBS_FAMILY = {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}


def _grid_from_kwargs(
    method_enum: "StatMethod",
    enhance_kwargs: Dict[str, Any],
) -> Tuple[Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
    """Return ``(e_grid, h_grid)`` arrays iff ``e``/``h`` were arrays.

    Used to label the parameter axis of multi-(E, H) :class:`InferenceResult`
    p-maps. Returns ``(None, None)`` for scalar (E, H) calls or
    non-TFNBS-family methods.
    """
    if method_enum not in TFNBS_FAMILY:
        return None, None
    e = enhance_kwargs.get("e")
    h = enhance_kwargs.get("h")
    if e is None or h is None:
        return None, None
    if np.isscalar(e) and np.isscalar(h):
        return None, None
    return (
        np.asarray(e, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
    )


# =============================================================================
# Helper functions for permutation testing
# =============================================================================


def _permutation_task_ind(
    full_group: npt.NDArray[np.float64],
    func: Callable[..., Any],
    n1: int,
    seed: int,
    **func_kwargs
) -> Dict[str, Union[float, npt.NDArray[np.float64]]]:
    """
    Compute maximum t-statistic for a single permutation (two-sample test).

    Parameters
    ----------
    full_group : ndarray
        Concatenated data array of shape (n_samples_1 + n_samples_2, *dims).
    func : callable
        Function to compute the t-statistic.
    n1 : int
        Number of samples in group 1.
    seed : int
        Random seed for this permutation.
    **func_kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    dict
        Dictionary with max statistics for 'negative' and 'positive' directions.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(full_group.shape[0])
    new_group1 = full_group[idx[:n1]]
    new_group2 = full_group[idx[n1:]]
    perm_stat_dict = func(new_group1, new_group2, test_type='two-sample', **func_kwargs)
    return _extract_max_stats(perm_stat_dict, full_group[0].shape)


def _permutation_task_paired(
    diffs: npt.NDArray[np.float64],
    func: Callable[..., Any],
    seed: Optional[int] = None,
    **func_kwargs
) -> Dict[str, Union[float, npt.NDArray[np.float64]]]:
    """
    Compute maximum t-statistic for a single permutation (paired/one-sample test).

    Parameters
    ----------
    diffs : ndarray
        Array of shape (n_samples, *dims) containing paired differences.
    func : callable
        Function to compute the t-statistic.
    seed : int, optional
        Random seed for this permutation.
    **func_kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    dict
        Dictionary with max statistics for 'negative' and 'positive' directions.
    """
    n_dims = len(diffs.shape) - 1
    faked_dims = [1] * n_dims
    rng = np.random.RandomState(seed)
    signs = rng.choice([1, -1], diffs.shape[0]).reshape(-1, *faked_dims)
    new_diffs = signs * diffs
    perm_stat_dict = func(new_diffs, **func_kwargs)
    return _extract_max_stats(perm_stat_dict, diffs[0].shape)


def _check_paired_counts(group1, group2, test_type: str) -> None:
    """Paired data must hold the same subjects in both conditions.

    Without this check NumPy broadcasting pairs every subject of one array
    with a single subject of the other (20 vs 1) and tests pseudo-pairs.
    """
    if test_type == TestType.PAIRED.value and group2 is not None \
            and group1.shape[0] != group2.shape[0]:
        raise ValueError(
            "A paired test needs the same subjects in both conditions, in the same "
            f"order; got {group1.shape[0]} in group1 and {group2.shape[0]} in group2."
        )


# =============================================================================
# Pre-computed sums fast path (method='tstat' and 'bh_fdr_perm')
# =============================================================================
# Exploits the algebraic variance identity Var(x) = (Σx² − (Σx)²/n)/(n−1) with
# the sign-flip-invariant Σx² pre-computed once, so each permutation reduces to
# a matrix-vector dot product over upper-triangle edges (~6-8x faster per perm
# vs. recomputing mean/std from full (nSub, N, N) arrays).


def _onesample_tstat_from_sums(sum_vec, sumsq_vec, n):
    """One-sample t-statistic from pre-computed sums. Returns (t_pos, t_neg) edge vectors."""
    mean = sum_vec / n
    var = np.maximum((sumsq_vec - sum_vec ** 2 / n) / (n - 1), 0)
    t = _degenerate.t_ratio(mean, var / n, _degenerate.RTOL * sumsq_vec / n ** 2)
    return np.maximum(t, 0.0), np.maximum(-t, 0.0)


def _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2):
    """Welch two-sample t-statistic from pre-computed sums. Returns (t_pos, t_neg) edge vectors."""
    mean1 = sum1 / n1
    mean2 = sum2 / n2
    var1 = np.maximum((sumsq1 - sum1 ** 2 / n1) / (n1 - 1), 0)
    var2 = np.maximum((sumsq2 - sum2 ** 2 / n2) / (n2 - 1), 0)
    meansq = (sumsq1 + sumsq2) / (n1 + n2)
    t = _degenerate.t_ratio(mean2 - mean1, var1 / n1 + var2 / n2,
                            _degenerate.RTOL * meansq * (1 / n1 + 1 / n2))
    return np.maximum(t, 0.0), np.maximum(-t, 0.0)


def _precompute_edge_sums(data_3d):
    """Extract upper-triangle edges + sum-of-squares for sign-flip fast path.

    Returns (X, sumsq_all): X is (n_samples, n_edges) contiguous; sumsq_all is
    (n_edges,) and is sign-flip invariant since s² = 1 for s ∈ {+1, −1}.
    """
    N = data_3d.shape[1]
    triu_idx = np.triu_indices(N, k=1)
    X = np.ascontiguousarray(data_3d[:, triu_idx[0], triu_idx[1]])
    sumsq_all = np.sum(X ** 2, axis=0)
    return X, sumsq_all


def _precompute_twosample_sums(Xall_3d):
    """Extract upper-triangle edges + pooled sums for group-label fast path.

    Returns (Xall, Xall2, sum_all, sumsq_all). Xall2 caches element-wise square
    so per-permutation cost is two matvecs + two subtractions.
    """
    N = Xall_3d.shape[1]
    triu_idx = np.triu_indices(N, k=1)
    Xall = np.ascontiguousarray(Xall_3d[:, triu_idx[0], triu_idx[1]])
    Xall2 = Xall ** 2
    sum_all = np.sum(Xall, axis=0)
    sumsq_all = np.sum(Xall2, axis=0)
    return Xall, Xall2, sum_all, sumsq_all


def _fast_permutation_task_paired(X, sumsq_all, seed):
    """Paired/one-sample permutation via sign-flip + sums. Returns max stats."""
    rng = np.random.RandomState(seed)
    n_sub = X.shape[0]
    signs = rng.choice([1, -1], n_sub).astype(np.float64)
    sum_perm = signs @ X
    t_pos, t_neg = _onesample_tstat_from_sums(sum_perm, sumsq_all, n_sub)
    return {
        "positive": np.float64(np.max(t_pos)),
        "negative": np.float64(np.max(t_neg)),
    }


def _fast_permutation_task_ind(Xall, Xall2, sum_all, sumsq_all, n1, seed,
                               strata=None, n1_per_stratum=None):
    """Two-sample permutation via group-label shuffle + sums. Returns max stats.

    When ``strata`` is provided (with ``n1_per_stratum``), the group-label
    draw honors per-stratum group totals — exchangeability blocking.
    """
    rng = np.random.RandomState(seed)
    n_all = Xall.shape[0]
    n2 = n_all - n1
    if strata is None:
        g = np.zeros(n_all, dtype=np.float64)
        g[rng.choice(n_all, n1, replace=False)] = 1.0
    else:
        g = _stratified_choice_n1(strata, n1_per_stratum, rng)
    sum1 = g @ Xall
    sumsq1 = g @ Xall2
    sum2 = sum_all - sum1
    sumsq2 = sumsq_all - sumsq1
    t_pos, t_neg = _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2)
    return {
        "positive": np.float64(np.max(t_pos)),
        "negative": np.float64(np.max(t_neg)),
    }


def _fast_permutation_task_paired_edges(X, sumsq_all, seed):
    """Paired/one-sample permutation returning upper-triangle t-stat vectors (for BH-FDR-perm)."""
    rng = np.random.RandomState(seed)
    n_sub = X.shape[0]
    signs = rng.choice([1, -1], n_sub).astype(np.float64)
    sum_perm = signs @ X
    t_pos, t_neg = _onesample_tstat_from_sums(sum_perm, sumsq_all, n_sub)
    return {"positive": t_pos, "negative": t_neg}


def _fast_permutation_task_ind_edges(Xall, Xall2, sum_all, sumsq_all, n1, seed,
                                     strata=None, n1_per_stratum=None):
    """Two-sample permutation returning upper-triangle t-stat vectors (for BH-FDR-perm)."""
    rng = np.random.RandomState(seed)
    n_all = Xall.shape[0]
    n2 = n_all - n1
    if strata is None:
        g = np.zeros(n_all, dtype=np.float64)
        g[rng.choice(n_all, n1, replace=False)] = 1.0
    else:
        g = _stratified_choice_n1(strata, n1_per_stratum, rng)
    sum1 = g @ Xall
    sumsq1 = g @ Xall2
    sum2 = sum_all - sum1
    sumsq2 = sumsq_all - sumsq1
    t_pos, t_neg = _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2)
    return {"positive": t_pos, "negative": t_neg}


# =============================================================================
# Enhancement-method fast permutation tasks (sums trick + shared enhancers)
# =============================================================================

def _tstat_edges_to_matrix(t_pos, t_neg, triu_idx, N):
    """Expand upper-triangle t-stat vectors to symmetric (N, N) matrices."""
    t_pos_mat = np.zeros((N, N))
    t_pos_mat[triu_idx] = t_pos
    t_pos_mat = t_pos_mat + t_pos_mat.T
    t_neg_mat = np.zeros((N, N))
    t_neg_mat[triu_idx] = t_neg
    t_neg_mat = t_neg_mat + t_neg_mat.T
    return t_pos_mat, t_neg_mat


def _fast_perm_task_enhanced_paired(
    X, sumsq_all, triu_idx, N, enhance_func, enhance_kwargs, seed,
):
    """Paired/one-sample permutation: sign-flip + fast t-stat + enhancement → max."""
    rng = np.random.RandomState(seed)
    n_sub = X.shape[0]
    signs = rng.choice([1, -1], n_sub).astype(np.float64)
    sum_perm = signs @ X
    t_pos, t_neg = _onesample_tstat_from_sums(sum_perm, sumsq_all, n_sub)
    t_pos_mat, t_neg_mat = _tstat_edges_to_matrix(t_pos, t_neg, triu_idx, N)
    stat_dict = {"positive": t_pos_mat, "negative": t_neg_mat}
    if enhance_func is not None:
        stat_dict = enhance_func(stat_dict, **enhance_kwargs)
    return _extract_max_stats(stat_dict, (N, N))


def _fast_perm_task_enhanced_ind(
    Xall, Xall2, sum_all, sumsq_all, n1, triu_idx, N,
    enhance_func, enhance_kwargs, seed,
    strata=None, n1_per_stratum=None,
):
    """Two-sample permutation: group shuffle + fast t-stat + enhancement → max.

    When ``strata`` is provided, the group-label draw is stratified
    (per-stratum group totals held fixed).
    """
    rng = np.random.RandomState(seed)
    n_all = Xall.shape[0]
    n2 = n_all - n1
    if strata is None:
        g = np.zeros(n_all, dtype=np.float64)
        g[rng.choice(n_all, n1, replace=False)] = 1.0
    else:
        g = _stratified_choice_n1(strata, n1_per_stratum, rng)
    sum1 = g @ Xall
    sumsq1 = g @ Xall2
    sum2 = sum_all - sum1
    sumsq2 = sumsq_all - sumsq1
    t_pos, t_neg = _twosample_tstat_from_sums(sum1, sumsq1, n1, sum2, sumsq2, n2)
    t_pos_mat, t_neg_mat = _tstat_edges_to_matrix(t_pos, t_neg, triu_idx, N)
    stat_dict = {"positive": t_pos_mat, "negative": t_neg_mat}
    if enhance_func is not None:
        stat_dict = enhance_func(stat_dict, **enhance_kwargs)
    return _extract_max_stats(stat_dict, (N, N))


# =============================================================================
# Helper functions for multiprocessing
# =============================================================================


# =============================================================================
# Null distribution computation
# =============================================================================

def compute_null_dist(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    func: Optional[Callable[..., Any]] = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    test_type: Union[str, TestType] = TestType.PAIRED,
    rng: RngLike = None,
    random_state: Optional[int] = None,  # deprecated alias for `rng`
    n_processes: Optional[int] = None,
    use_mp: bool = True,
    verbose: bool = False,
    strata: Optional[npt.NDArray[Any]] = None,
    **func_kwargs
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute null distribution of maximum t-statistics via permutation testing.

    Optimized implementation with:

    - Fixed indexing bug in sequential mode
    - Efficient result collection
    - Context-aware multiprocessing (auto-disables inside worker processes)

    Parameters
    ----------
    group1 : ndarray of shape ``(n_samples_1, *dims)``
        Data array for group 1.
    group2 : ndarray of shape ``(n_samples_2, *dims)``, optional
        Data array for group 2. Required for 'paired' and 'two-sample' tests.
    func : callable, optional
        Function to compute the t-statistic.
    n_permutations : int, default=1000
        Number of permutations.
    test_type : {'paired', 'one-sample', 'two-sample'} or TestType
        Type of statistical test.
    rng : int, numpy.random.Generator, or None
        Seed for reproducibility. ``random_state`` is a deprecated alias.
    n_processes : int, optional
        Number of parallel processes. Defaults to CPU count.
    use_mp : bool, default=True
        Whether to use multiprocessing. Automatically disabled when called
        from inside a multiprocessing worker to prevent nested pools.
    **func_kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    dict
        Dictionary with 'negative' and 'positive' arrays of shape (n_permutations,)
        or (n_permutations, n_params) for multi-parameter TFCE.

    Raises
    ------
    ValueError
        If test_type is invalid or required group2 is missing.
    """
    # Normalize test_type to string for comparison
    test_type_str = test_type.value if isinstance(test_type, TestType) else test_type

    # Input validation
    if test_type_str in (TestType.PAIRED.value, TestType.TWO_SAMPLE.value):
        if group2 is None:
            raise ValueError(f"group2 is required for test_type='{test_type_str}'")
        if group1.shape[1:] != group2.shape[1:]:
            raise ValueError("Trailing dimensions of group1 and group2 must match.")
        n1, n2 = group1.shape[0], group2.shape[0]
        if n1 < 2 or n2 < 2:
            raise ValueError("Each group must have at least 2 samples.")
        _check_paired_counts(group1, group2, test_type_str)

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    # Fast path: raw t-statistic with no enhancement — uses pre-computed sums
    # trick (upper-triangle only, sign-flip-invariant Σx²). ~6-8x faster per perm.
    use_fast_tstat = (
        func in (compute_t_stat_diff, _compute_t_stat_ind, compute_t_stat)
        and not func_kwargs
        and group1.ndim == 3
        and group1.shape[1] == group1.shape[2]
    )

    # Prepare data for permutation
    if test_type_str == TestType.PAIRED.value:
        assert group2 is not None  # validated above: required for 'paired'
        if use_fast_tstat:
            diffs = group2 - group1
            X, sumsq_all = _precompute_edge_sums(diffs)
            task_func = partial(_fast_permutation_task_paired, X, sumsq_all)
        else:
            array_to_permute = group2 - group1
            task_func = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)

    elif test_type_str == TestType.TWO_SAMPLE.value:
        strata_codes = None
        n1_per_stratum = None
        if strata is not None:
            strata_codes = _encode_strata(strata)
            if strata_codes.shape[0] != n1 + n2:
                raise ValueError(
                    f"strata length {strata_codes.shape[0]} does not match "
                    f"n1+n2={n1+n2} for the two-sample test."
                )
            # Observed group-1 counts per stratum (first n1 entries are group 1
            # by construction of the concatenated stack).
            n1_per_stratum = np.bincount(
                strata_codes[:n1], minlength=int(strata_codes.max()) + 1
            )
        if use_fast_tstat:
            Xall_3d = np.concatenate((group1, group2), axis=0)
            Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(Xall_3d)
            task_func = partial(
                _fast_permutation_task_ind, Xall, Xall2, sum_all, sumsq_all, n1,
                strata=strata_codes, n1_per_stratum=n1_per_stratum,
            )
        else:
            if strata is not None:
                raise NotImplementedError(
                    "strata= is only wired into the fast t-stat path for "
                    "two-sample tests. Pass one of the canonical t-stat "
                    "functions (compute_t_stat / _compute_t_stat_ind) to "
                    "exercise the stratified-permutation path."
                )
            array_to_permute = np.concatenate((group1, group2), axis=0)
            task_func = partial(_permutation_task_ind, array_to_permute, func, n1, **func_kwargs)

    elif test_type_str == TestType.ONE_SAMPLE.value:
        if use_fast_tstat:
            X, sumsq_all = _precompute_edge_sums(group1)
            task_func = partial(_fast_permutation_task_paired, X, sumsq_all)
        else:
            array_to_permute = group1
            task_func = partial(_permutation_task_paired, array_to_permute, func, **func_kwargs)
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    # Generate seeds for reproducibility
    if random_state is not None:
        warn_legacy_random_state("random_state")
    seed_rng = np.random.RandomState(resolve_seed(rng, legacy_random_state=random_state))
    seeds = seed_rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="compute_null_dist",
    )

    return _collect_results_to_arrays(results, n_permutations)


def _compute_enhanced_null_dist(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type_str: str,
    enhance_func: Callable,
    enhance_kwargs: Dict[str, Any],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    random_state: Optional[int] = None,
    n_processes: Optional[int] = None,
    use_mp: bool = True,
    verbose: bool = False,
    strata: Optional[npt.NDArray[Any]] = None,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Null distribution for enhancement methods via pre-computed sums fast path.

    Each permutation: sign-flip or group-shuffle → fast t-stat via sums →
    reconstruct (N, N) → apply shared enhancement wrapper → extract max.

    When ``strata`` is provided, the two-sample group-shuffle path
    honors per-stratum group totals (exchangeability blocking). Sign-
    flip paths are stratum-invariant by construction; ``strata`` is
    silently accepted but has no effect there.
    """
    N = group1.shape[1]
    triu_idx = np.triu_indices(N, k=1)

    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        X, sumsq_all = _precompute_edge_sums(diffs)
        task_func = partial(
            _fast_perm_task_enhanced_paired,
            X, sumsq_all, triu_idx, N, enhance_func, enhance_kwargs,
        )
    elif test_type_str == TestType.ONE_SAMPLE.value:
        X, sumsq_all = _precompute_edge_sums(group1)
        task_func = partial(
            _fast_perm_task_enhanced_paired,
            X, sumsq_all, triu_idx, N, enhance_func, enhance_kwargs,
        )
    elif test_type_str == TestType.TWO_SAMPLE.value:
        n1 = group1.shape[0]
        n_all = n1 + group2.shape[0]
        strata_codes = None
        n1_per_stratum = None
        if strata is not None:
            strata_codes = _encode_strata(strata)
            if strata_codes.shape[0] != n_all:
                raise ValueError(
                    f"strata length {strata_codes.shape[0]} does not match "
                    f"n1+n2={n_all} for the two-sample test."
                )
            n1_per_stratum = np.bincount(
                strata_codes[:n1], minlength=int(strata_codes.max()) + 1
            )
        Xall_3d = np.concatenate((group1, group2), axis=0)
        Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(Xall_3d)
        task_func = partial(
            _fast_perm_task_enhanced_ind,
            Xall, Xall2, sum_all, sumsq_all, n1, triu_idx, N,
            enhance_func, enhance_kwargs,
            strata=strata_codes, n1_per_stratum=n1_per_stratum,
        )
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="enhanced_null",
    )

    return _collect_results_to_arrays(results, n_permutations)


def _compute_bh_fdr_perm_p_values(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type_str: str,
    n_permutations: int,
    random_state: Optional[int],
    use_mp: bool,
    n_processes: Optional[int],
    verbose: bool = False,
    strata: Optional[npt.NDArray[Any]] = None,
    two_tailed: bool = False,
) -> Tuple[Dict[str, npt.NDArray[np.float64]], Dict[str, npt.NDArray[np.float64]]]:
    """
    Compute BH-FDR corrected p-values using permutation-based per-edge nulls.

    1. Compute observed t-stats per edge.
    2. For each permutation, compute per-edge t-stats.
    3. For each edge, p = (# perm t >= observed t) / n_perm.
    4. Apply BH-FDR correction to per-edge p-values.
    """
    # Compute observed t-stats
    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        emp_t_dict = compute_t_stat_diff(diffs)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        emp_t_dict = compute_t_stat_diff(group1)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        emp_t_dict = compute_t_stat(group1, group2, test_type=test_type_str)
    else:
        raise ValueError(f"Invalid test_type: '{test_type_str}'")

    N = emp_t_dict["positive"].shape[0]
    triu_idx = np.triu_indices(N, k=1)
    n_edges = len(triu_idx[0])

    # Generate seeds
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)

    # Prepare permutation task — pre-computed sums fast path (edge vectors).
    # Skips the redundant (N, N) reshape and triu extraction inside the perm loop.
    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        X, sumsq_all = _precompute_edge_sums(diffs)
        task_func = partial(_fast_permutation_task_paired_edges, X, sumsq_all)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        X, sumsq_all = _precompute_edge_sums(group1)
        task_func = partial(_fast_permutation_task_paired_edges, X, sumsq_all)
    else:  # two-sample
        Xall_3d = np.concatenate((group1, group2), axis=0)
        Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(Xall_3d)
        n1 = group1.shape[0]
        strata_codes = None
        n1_per_stratum = None
        if strata is not None:
            strata_codes = _encode_strata(strata)
            n_all = Xall_3d.shape[0]
            if strata_codes.shape[0] != n_all:
                raise ValueError(
                    f"strata length {strata_codes.shape[0]} does not match "
                    f"n1+n2={n_all} for the two-sample test."
                )
            n1_per_stratum = np.bincount(
                strata_codes[:n1], minlength=int(strata_codes.max()) + 1
            )
        task_func = partial(
            _fast_permutation_task_ind_edges, Xall, Xall2, sum_all, sumsq_all, n1,
            strata=strata_codes, n1_per_stratum=n1_per_stratum,
        )

    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    perm_results = run_permutations(
        task_func, list(seeds),
        use_mp=_use_mp, n_processes=n_processes,
        verbose=verbose, desc="bh_fdr_perm",
    )

    # Compute per-edge p-values (with +1 correction — Phipson & Smyth 2010)
    # and apply BH-FDR correction
    p_values = {}
    if two_tailed:
        observed = (emp_t_dict["positive"] + emp_t_dict["negative"])[triu_idx]
        count = sum(
            (r["positive"] + r["negative"] >= observed).astype(float)
            for r in perm_results
        )
        corrected = _bh_fdr_correction((count + 1.0) / (n_permutations + 1.0))
        for key in ("positive", "negative"):
            p_mat = np.ones((N, N), dtype=np.float64)
            directional = np.where(emp_t_dict[key][triu_idx] > 0, corrected, 1.0)
            p_mat[triu_idx] = directional
            p_mat[(triu_idx[1], triu_idx[0])] = directional
            p_values[key] = p_mat
        return p_values, emp_t_dict
    for key in ("positive", "negative"):
        emp_upper = emp_t_dict[key][triu_idx]

        # Count how many permutation t-stats >= observed for each edge
        count_ge = np.zeros(n_edges, dtype=np.float64)
        for perm_dict in perm_results:
            # perm_dict[key] is already the upper-triangle vector (fast path)
            count_ge += (perm_dict[key] >= emp_upper).astype(np.float64)

        # +1 correction: prevents p = 0 with finite permutations
        per_edge_p = (count_ge + 1.0) / (n_permutations + 1.0)

        corrected_p = _bh_fdr_correction(per_edge_p)

        # Reconstruct full symmetric matrix
        p_mat = np.ones((N, N), dtype=np.float64)
        p_mat[triu_idx] = corrected_p
        p_mat[(triu_idx[1], triu_idx[0])] = corrected_p
        p_values[key] = p_mat

    return p_values, emp_t_dict


# =============================================================================
# P-value computation
# =============================================================================


def _compute_degrees_of_freedom(
    n1: int,
    n2: int,
    test_type_str: str
) -> int:
    """Compute degrees of freedom for a t-test.

    Parameters
    ----------
    n1 : int
        Number of samples in group 1.
    n2 : int
        Number of samples in group 2 (0 for one-sample).
    test_type_str : str
        Test type string ('paired', 'one-sample', 'two-sample').

    Returns
    -------
    int
        Degrees of freedom.
    """
    if test_type_str == TestType.TWO_SAMPLE.value:
        return n1 + n2 - 2
    else:
        # paired or one-sample: df = n - 1
        return n1 - 1


def _compute_welch_degrees_of_freedom(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute edge-wise Welch-Satterthwaite degrees of freedom."""
    n1 = group1.shape[0]
    n2 = group2.shape[0]
    var1 = np.var(group1, axis=0, ddof=1)
    var2 = np.var(group2, axis=0, ddof=1)
    se1 = var1 / n1
    se2 = var2 / n2
    numerator = (se1 + se2) ** 2
    denominator = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)
    pooled_df = float(n1 + n2 - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = numerator / denominator
    return np.where(denominator > 0, df, pooled_df)


def _compute_parametric_p_values(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type_str: str,
    method_enum: "StatMethod",
    alpha: float = 0.05,
    two_tailed: bool = False,
) -> Tuple[Dict[str, npt.NDArray[np.float64]], Dict[str, npt.NDArray[np.float64]]]:
    """
    Compute parametric p-values with Bonferroni or BH-FDR correction.

    Parameters
    ----------
    group1 : ndarray of shape (n_subjects, N, N)
        Connectivity matrices for group 1.
    group2 : ndarray of shape (n_subjects, N, N), optional
        Connectivity matrices for group 2.
    test_type_str : str
        Test type ('paired', 'one-sample', 'two-sample').
    method_enum : StatMethod
        BONFERRONI or BH_FDR.
    alpha : float, default=0.05
        Significance level (used for BH-FDR step-up).

    Returns
    -------
    (p_values, t_dict) : tuple of dict
        ``p_values`` — per-tail corrected p-value arrays of shape (N, N).
        ``t_dict`` — per-tail observed one-tail-clipped (non-negative)
        t-statistic maps, used to populate ``stat_positive``/
        ``stat_negative`` on :class:`InferenceResult`.
    """
    # Compute t-statistics using existing infrastructure
    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        t_dict = compute_t_stat_diff(diffs)
        df = _compute_degrees_of_freedom(group1.shape[0], 0, test_type_str)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        t_dict = compute_t_stat_diff(group1)
        df = _compute_degrees_of_freedom(group1.shape[0], 0, test_type_str)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        t_dict = _compute_t_stat_ind(group1, group2)
        df = _compute_welch_degrees_of_freedom(group1, group2)
    else:
        raise ValueError(f"Invalid test_type: '{test_type_str}'")

    if two_tailed:
        signed = t_dict["positive"] - t_dict["negative"]
        triu_idx = np.triu_indices(signed.shape[0], k=1)
        raw = (2 * stats.t.sf(np.abs(signed), df))[triu_idx]
        corrected = (np.minimum(raw * raw.size, 1.0)
                     if method_enum == StatMethod.BONFERRONI else _bh_fdr_correction(raw))
        p_values = {}
        for key in ("positive", "negative"):
            p = np.ones_like(signed)
            directional = np.where(t_dict[key][triu_idx] > 0, corrected, 1.0)
            p[triu_idx] = directional
            p[(triu_idx[1], triu_idx[0])] = directional
            p_values[key] = p
        return p_values, t_dict

    p_values = {}
    for key in ("negative", "positive"):
        t_vals = t_dict[key]  # Non-negative (one-tailed)
        N = t_vals.shape[0]

        # Compute one-tailed p-values from t-distribution
        raw_p = stats.t.sf(t_vals, df)

        # Extract upper triangle (unique edges for symmetric matrices)
        triu_idx = np.triu_indices(N, k=1)
        p_upper = raw_p[triu_idx]

        # Number of unique comparisons
        m = len(p_upper)

        if method_enum == StatMethod.BONFERRONI:
            # Bonferroni: multiply by number of comparisons, cap at 1.0
            p_corrected_upper = np.minimum(p_upper * m, 1.0)
        elif method_enum == StatMethod.BH_FDR:
            # Benjamini-Hochberg step-up procedure
            p_corrected_upper = _bh_fdr_correction(p_upper)
        else:
            raise ValueError(f"Unsupported parametric method: {method_enum}")

        # Reconstruct full symmetric matrix
        p_corrected = np.ones((N, N), dtype=np.float64)
        p_corrected[triu_idx] = p_corrected_upper
        p_corrected[(triu_idx[1], triu_idx[0])] = p_corrected_upper

        p_values[key] = p_corrected

    return p_values, t_dict


def _with_estimates(fn):
    """Attach the equivalent-design coefficient maps (``beta`` / ``design``) to the result.

    Two-sample: [group-1 mean, group 2 - group 1]; paired: [condition-A mean,
    B - A]; one-sample: [mean] (NaN-aware). Computed after ``fn`` validated
    its inputs, from the same validated stacks.
    """
    import functools
    import inspect

    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if getattr(res, "beta", None) is None:
            from ._estimates import pairwise_estimates

            b = sig.bind(*args, **kwargs)
            b.apply_defaults()
            tt = b.arguments["test_type"]
            tt = tt.value if isinstance(tt, TestType) else tt
            g1 = check_connectivity(b.arguments["group1"], "group1", allow_missing=True)
            g2 = b.arguments["group2"]
            g2 = None if g2 is None else check_connectivity(g2, "group2", allow_missing=True)
            res.beta, res.design = pairwise_estimates(g1, g2, tt)
        return res

    return wrapper


@_with_estimates
def compute_p_val(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    test_type: Union[str, TestType] = TestType.PAIRED,
    method: Union[str, StatMethod] = StatMethod.TFNBS,
    use_mp: bool = True,
    rng: RngLike = None,
    random_state: Optional[int] = None,  # deprecated alias for `rng`
    n_processes: Optional[int] = None,
    acceleration: Optional[str] = DEFAULT_ACCELERATION,
    verbose: bool = False,
    # Method-specific parameters
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
    cnbs_correction: str = "bonferroni",
    two_tailed: bool = False,
    edge_mask: Optional[npt.ArrayLike] = None,
) -> InferenceResult:
    """
    Compute p-values using permutation testing with various network-based methods.

    .. warning:: **Welch + group-label permutation under unequal variances**

       The two-sample path uses Welch's t (unequal-variance) unconditionally.
       Under unequal variances with unbalanced group sizes the exchangeability
       assumption underlying permutation breaks and Type I error inflates
       (Anderson & Robinson 2001; Hayes 2000). A pooled-variance option will
       be added in v2.2 with an explicit ``var_equal=True|False`` knob; until
       then, treat two-sample results with caution when both group sizes
       and variances differ substantially.

    Supports multiple statistical methods: tfnbs, tstat, nbs, cnbs, ni_tfnbs,
    fbc_tfnbs, bonferroni, bh_fdr, bh_fdr_perm.

    .. note:: **cNBS: block-level inference**

       ``method='cnbs'`` is the constrained NBS of Noble & Scheinost (2020):
       the signed t-statistic is averaged over every edge of each within- and
       between-network block (``net_labels``), each block is tested against
       its own permutation null (the same block's mean in every permuted
       map), and the block p-values are corrected across blocks
       (``cnbs_correction``: Bonferroni for FWER, 'bh' for FDR). Every edge
       carries its block's p-value; the result's ``block_report`` lists the
       blocks. A significant block means that the block's mean effect is
       nonzero, not that each of its edges is affected.

    .. note:: **Bonferroni and BH-FDR methods**

       ``StatMethod.BONFERRONI`` and ``StatMethod.BH_FDR`` are parametric baselines
       that do **not** use permutation testing. They compute p-values from the
       t-distribution and apply multiple comparison corrections. The
       ``n_permutations`` parameter is ignored for these methods. In the
       two-sample path, the parametric baseline uses edge-wise
       Welch-Satterthwaite degrees of freedom to match the Welch statistic.

    .. note:: **Permutation p-values**

       Empirical max-statistic p-values count ties conservatively
       (``null >= observed``) and use the Phipson-Smyth +1 correction:
       ``p = (count + 1) / (B + 1)``.

    Parameters
    ----------
    group1 : ndarray of shape (n_subjects_g1, N, N)
        Input connectivity matrices for group 1.
    group2 : ndarray of shape (n_subjects_g2, N, N), optional
        Input connectivity matrices for group 2. Required for paired/two-sample tests.
    n_permutations : int, default=1000
        Number of permutations for null distribution.
    test_type : {'paired', 'one-sample', 'two-sample'} or TestType, default='paired'
        Type of statistical test.
    method : {'tfnbs', 'tstat', 'nbs', 'cnbs', 'ni_tfnbs', 'fbc_tfnbs'} or StatMethod, default='tfnbs'
        Statistical method to use for scoring.
    use_mp : bool, default=True
        Use multiprocessing for permutation testing. Automatically disabled
        when called from inside a multiprocessing worker to prevent deadlocks.
    rng : int, numpy.random.Generator, or None, optional
        Random seed for reproducibility. ``random_state`` is a deprecated
        alias that will be removed in v2.1.
    n_processes : int, optional
        Number of CPU cores for parallel computing.
    net_labels : ndarray of shape (N,), optional
        Network labels for each node. Required for cnbs, ni_tfnbs, and fbc_tfnbs.
    threshold : float, default=2.0
        T-statistic threshold for NBS (only used when method='nbs').
    nbs_stat : {'extent', 'intensity'}, default='extent'
        Cluster statistic for NBS (only used when method='nbs').
    e : float or sequence of float, default=0.3
        Extent exponent for TFNBS-based methods. Pass an equal-length
        sequence with ``h`` to evaluate a whole ``(E, H)`` grid in one
        permutation pass — TFNBS's threshold integration runs once
        and broadcasts the per-cell exponentiation, so a K-cell grid
        costs ~the same wall-clock as a single cell. In grid mode the
        returned :class:`~conninfpy.InferenceResult` carries the
        parameter axis (``result.is_grid``, ``result.e_grid``,
        ``result.h_grid``); use ``result.select(param_idx)`` or pass
        ``param_idx=`` to ``significant_edges`` / ``to_csv`` to project
        to a single cell.
    h : float or sequence of float, default=3.0
        Height exponent for TFNBS-based methods. See ``e`` for grid mode.
    n : int, default=10
        Number of threshold steps for TFNBS-based methods.
    start_thres : float, default=1.65
        Starting threshold for TFNBS integration.
    min_cluster_size : int, default=3
        Minimum cluster size for FBC-TFNBS (only used when method='fbc_tfnbs').
    normalization : {'sqrt', 'linear', 'none'}, default='sqrt'
        Block density normalization for NI-TFNBS (only used when method='ni_tfnbs').
    edge_mask : (N, N) bool array, optional
        Restrict inference to the marked edges (a declared tested family, e.g.
        structurally connected pairs). Statistics outside the mask are set to 0
        in the observed map and in every permutation, so those edges join no
        TFNBS / NBS cluster, count toward no NI-TFNBS block density or capacity,
        and never set the null maximum: FWER is controlled over the masked
        family. Their p-values are NaN (untested). Supported for 'tfnbs',
        'ni_tfnbs', 'nbs' and 'tstat'; must be symmetric.
    strata : array-like of shape (n,), optional
        Exchangeability-block labels per subject (e.g. site IDs after ComBat
        harmonization). When provided, the two-sample group-label permutation
        is restricted to *within-stratum* swaps, holding per-stratum group
        totals fixed. This prevents the shadow-of-H₀ leak that occurs when
        ComBat is fit on observed labels but downstream permutation reshuffles
        across sites. Paired / one-sample sign-flip paths are stratum-invariant
        by construction; ``strata`` is silently accepted but has no effect
        there.
    missing : {'raise', 'available_case'}, default='raise'
        Missing connectivity values (NaN). ``'raise'`` stops with a report
        of the affected subjects and regions. ``'available_case'`` tests
        each edge with the subjects who have it (paired: complete pairs);
        permutations move whole subjects, so the test stays exact given
        the missingness. Edges kept by fewer than ``min_coverage`` of a
        group's subjects (or by fewer than 3) are not tested: p = NaN.
        Infinite values are always rejected. The coverage report is in
        ``result.missing_report``.
    min_coverage : float, default=0.8
        Minimum share of each group's subjects (or of the pairs) an edge
        needs to be tested under ``missing='available_case'``.
    cnbs_correction : {'bonferroni', 'bh'}, default='bonferroni'
        For ``method='cnbs'``: correction of the block p-values across blocks
        (Bonferroni: FWER over blocks; 'bh': FDR over blocks).
    two_tailed : bool, default=False
        Correct one two-sided family. Max-statistic methods pool positive
        and negative null maxima; cNBS uses absolute block means; BH methods
        correct two-sided marginal p-values once across eligible edges.
        Directional maps report the corrected p-value in the observed
        direction and 1 in the opposite direction. Each (E, H) setting is
        corrected separately; selecting settings is not jointly corrected.
        False keeps separate one-sided families (use alpha/2 per tail for
        combined-direction FWER). Unsupported keyword arguments raise.

    Returns
    -------
    InferenceResult
        Dict-like result with p-value arrays:

        - 'negative': P-values for group 1 > group 2.
        - 'positive': P-values for group 2 > group 1.

    Raises
    ------
    ValueError
        If constrained methods (cnbs, ni_tfnbs, fbc_tfnbs) are used without net_labels.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(2)
    >>> group1 = np.random.rand(5, 3, 3)
    >>> for arr in group1: np.fill_diagonal(arr, 0)
    >>> group2 = np.random.rand(8, 3, 3)
    >>> for arr in group2: np.fill_diagonal(arr, 0)
    >>> # Standard t-test
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10,
    ...                        test_type='two-sample', method='tstat',
    ...                        use_mp=False, rng=0)
    >>> # TFNBS
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10,
    ...                        test_type='two-sample', method='tfnbs',
    ...                        use_mp=False, rng=0)
    >>> # cNBS with network labels
    >>> labels = np.array([0, 0, 1])
    >>> p_vals = compute_p_val(group1, group2, n_permutations=10,
    ...                        test_type='two-sample', method='cnbs',
    ...                        net_labels=labels, use_mp=False, rng=0)
    """
    # Normalize inputs
    test_type_str = test_type.value if isinstance(test_type, TestType) else test_type
    method_str = method.value if isinstance(method, StatMethod) else method
    if random_state is not None and rng is None:
        warn_legacy_random_state("random_state")
    random_state = resolve_seed(rng, legacy_random_state=random_state)

    if missing not in ("raise", "available_case"):
        raise ValueError(f"missing must be 'raise' or 'available_case'; got {missing!r}.")
    allow_missing = missing == "available_case"
    if method_str == StatMethod.CNBS.value:
        from ._cnbs import check_correction
        check_correction(cnbs_correction)
    group1 = check_connectivity(group1, "group1", allow_missing=allow_missing)
    if group2 is not None:
        group2 = check_connectivity(group2, "group2", allow_missing=allow_missing)
        if group2.shape[1:] != group1.shape[1:]:
            raise ValueError(
                f"group1 and group2 must have the same (N, N) shape; got "
                f"{group1.shape[1:]} and {group2.shape[1:]}."
            )
    if test_type_str not in [t.value for t in TestType]:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )
    if test_type_str in (TestType.PAIRED.value, TestType.TWO_SAMPLE.value) and group2 is None:
        raise ValueError(f"group2 is required for test_type='{test_type_str}'.")
    _check_paired_counts(group1, group2, test_type_str)

    # Validate method
    try:
        method_enum = StatMethod(method_str)
    except ValueError:
        valid_methods = [m.value for m in StatMethod]
        raise ValueError(
            f"Invalid method: '{method_str}'. Must be one of: {valid_methods}"
        )

    # Validate constrained methods require net_labels
    if method_enum in CONSTRAINED_METHODS and net_labels is None:
        raise ValueError(
            f"Method '{method_str}' requires net_labels to be provided. "
            f"Constrained methods are: {[m.value for m in CONSTRAINED_METHODS]}"
        )

    # An edge with zero variance and a nonzero effect has an infinite t
    _degenerate.raise_if_infinite(_degenerate.pairwise_infinite(group1, group2, test_type_str))

    _t0 = time.perf_counter()

    # Resolve enhancement wrapper (None for raw t-stat)
    enhance_func = _ENHANCE_MAP[method_enum]

    # Build enhancement kwargs once (method-specific)
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
        enhance_kwargs = {"net_labels": net_labels}

    # Declared tested family (edge_mask): mask the statistic before any enhancement
    family = None
    if edge_mask is not None:
        family = _check_edge_mask(edge_mask, group1.shape[1])
        if method_enum not in (StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.NBS, StatMethod.TSTAT):
            raise NotImplementedError(f"edge_mask is supported for tfnbs, ni_tfnbs, nbs and tstat; got {method_str!r}.")
        if allow_missing and (np.isnan(group1).any() or (group2 is not None and np.isnan(group2).any())):
            raise NotImplementedError("edge_mask cannot be combined with missing='available_case' yet.")
        from .enhance import apply_edge_mask
        enhance_kwargs = {**enhance_kwargs, "edge_mask": family, "inner": enhance_func}
        enhance_func = apply_edge_mask

    # Missing values (NaN), allowed only under missing='available_case'
    if allow_missing and (np.isnan(group1).any() or (group2 is not None and np.isnan(group2).any())):
        from ._available_case import available_case_p_val

        if method_enum in PARAMETRIC_METHODS:
            mode = f"parametric_{method_enum.value}"
        elif method_enum == StatMethod.BH_FDR_PERM:
            mode = "bh_fdr_perm"
        else:
            mode = "max"
        p_ac, raw_t_ac, report = available_case_p_val(
            group1, group2, test_type_str, mode, enhance_func, enhance_kwargs,
            n_permutations=n_permutations, random_state=random_state,
            use_mp=use_mp, n_processes=n_processes, verbose=verbose,
            strata=strata, acceleration=acceleration if mode == "max" else None,
            min_coverage=min_coverage, two_tailed=two_tailed,
            block_inference=(dict(net_labels=net_labels, correction=cnbs_correction)
                             if method_enum == StatMethod.CNBS else None),
        )
        e_grid, h_grid = _grid_from_kwargs(method_enum, enhance_kwargs)
        return InferenceResult(
            p_ac["positive"], p_ac["negative"],
            method=method_str,
            n_permutations=0 if mode.startswith("parametric_") else n_permutations,
            acceleration=acceleration if mode == "max" else None,
            wall_time_s=time.perf_counter() - _t0,
            stat_positive=raw_t_ac["positive"], stat_negative=raw_t_ac["negative"],
            stat_type="tstat", strata_provided=strata is not None,
            e_grid=e_grid, h_grid=h_grid, missing_report=report,
            block_report=report.pop("block_report", None),
            two_tailed=two_tailed,
        )

    # Parametric methods: compute p-values directly from t-distribution
    if method_enum in PARAMETRIC_METHODS:
        result, obs_t_dict = _compute_parametric_p_values(
            group1, group2, test_type_str, method_enum, two_tailed=two_tailed
        )
        return InferenceResult(
            result["positive"], result["negative"],
            method=method_str, n_permutations=0, acceleration=None,
            wall_time_s=time.perf_counter() - _t0,
            stat_positive=obs_t_dict["positive"],
            stat_negative=obs_t_dict["negative"],
            stat_type="tstat",
            two_tailed=two_tailed,
        )

    # BH-FDR with permutation p-values: separate code path
    if method_enum == StatMethod.BH_FDR_PERM:
        result, obs_t_dict = _compute_bh_fdr_perm_p_values(
            group1, group2, test_type_str,
            n_permutations=n_permutations,
            random_state=random_state,
            use_mp=use_mp,
            n_processes=n_processes,
            verbose=verbose,
            strata=strata,
            two_tailed=two_tailed,
        )
        return InferenceResult(
            result["positive"], result["negative"],
            method=method_str, n_permutations=n_permutations,
            acceleration=None,
            wall_time_s=time.perf_counter() - _t0,
            stat_positive=obs_t_dict["positive"],
            stat_negative=obs_t_dict["negative"],
            stat_type="tstat",
            strata_provided=strata is not None,
            two_tailed=two_tailed,
        )

    # Compute observed t-stat once, then apply enhancement (if any)
    if test_type_str == TestType.PAIRED.value:
        diffs = group2 - group1
        emp_t_dict = compute_t_stat_diff(diffs)
    elif test_type_str == TestType.ONE_SAMPLE.value:
        emp_t_dict = compute_t_stat_diff(group1)
    elif test_type_str == TestType.TWO_SAMPLE.value:
        emp_t_dict = _compute_t_stat_ind(group1, group2)
    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )

    # Keep a reference to the raw (pre-enhancement) t-stat dict for
    # InferenceResult.stat_positive/stat_negative — the user-facing
    # test-statistic map is kept independently of the enhanced score.
    raw_t_dict = emp_t_dict
    if enhance_func is not None:
        emp_t_dict = enhance_func(emp_t_dict, **enhance_kwargs)

    # Compute null distribution
    # - method='tstat' (no enhancement): use compute_null_dist fast path
    # - enhancement methods: use fast enhancement perm tasks (sums + enhancement)
    if enhance_func is None:
        group2_for_null = group2 if test_type_str != TestType.ONE_SAMPLE.value else None
        tstat_func = (
            _compute_t_stat_ind
            if test_type_str == TestType.TWO_SAMPLE.value
            else compute_t_stat_diff
        )
        max_null_dict = compute_null_dist(
            group1, group2_for_null, tstat_func,
            n_permutations=n_permutations,
            test_type=test_type,
            use_mp=use_mp,
            rng=random_state,
            n_processes=n_processes,
            verbose=verbose,
            strata=strata,
        )
    else:
        max_null_dict = _compute_enhanced_null_dist(
            group1, group2, test_type_str,
            enhance_func=enhance_func,
            enhance_kwargs=enhance_kwargs,
            n_permutations=n_permutations,
            random_state=random_state,
            use_mp=use_mp,
            n_processes=n_processes,
            verbose=verbose,
            strata=strata,
        )

    block_report = None
    if two_tailed and method_enum != StatMethod.CNBS:
        joint = np.maximum(max_null_dict["positive"], max_null_dict["negative"])
        max_null_dict = {"positive": joint, "negative": joint}
    if method_enum == StatMethod.CNBS:
        from ._cnbs import block_p_values
        result, block_report = block_p_values(
            emp_t_dict, max_null_dict, net_labels, group1.shape[1],
            correction=cnbs_correction, acceleration=acceleration, two_tailed=two_tailed,
        )
    elif acceleration is not None:
        result = compute_p_values_accelerated(
            emp_t_dict, max_null_dict, method=acceleration,
        )
    else:
        result = _compute_p_values_from_null(emp_t_dict, max_null_dict)
    if family is not None:
        # Edges outside the declared family are untested: NaN, never "not significant"
        outside = ~family & ~np.eye(family.shape[0], dtype=bool)
        result = {k: np.where(outside, np.nan, np.asarray(result[k], dtype=np.float64))
                  for k in ("positive", "negative")}
    # Capture the (E, H) grid for TFNBS-family methods so the result
    # carries the parameter axis labels when an array was passed.
    e_grid, h_grid = _grid_from_kwargs(method_enum, enhance_kwargs)
    return InferenceResult(
        result["positive"], result["negative"],
        method=method_str, n_permutations=n_permutations,
        acceleration=acceleration,
        wall_time_s=time.perf_counter() - _t0,
        stat_positive=raw_t_dict["positive"],
        stat_negative=raw_t_dict["negative"],
        stat_type="tstat",
        strata_provided=strata is not None,
        e_grid=e_grid,
        h_grid=h_grid,
        block_report=block_report,
        two_tailed=two_tailed,
    )


def _check_edge_mask(edge_mask: npt.ArrayLike, n_nodes: int) -> npt.NDArray[np.bool_]:
    """Validate a tested-family mask: (N, N), symmetric, at least one edge; diagonal ignored."""
    m = np.asarray(edge_mask)
    if m.shape != (n_nodes, n_nodes):
        raise ValueError(f"edge_mask must have shape {(n_nodes, n_nodes)}; got {m.shape}.")
    m = m.astype(bool)
    if not np.array_equal(m, m.T):
        raise ValueError("edge_mask must be symmetric.")
    m = m.copy()
    np.fill_diagonal(m, False)
    if not m.any():
        raise ValueError("edge_mask selects no edge.")
    return m


# =============================================================================
# T-statistic computation functions
# =============================================================================


def compute_t_stat(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    test_type: Union[str, TestType] = TestType.PAIRED
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for paired, one-sample, or two-sample tests.

    Parameters
    ----------
    group1 : ndarray of shape (n_samples_1, N, N)
        Data array for group 1.
    group2 : ndarray of shape (n_samples_2, N, N), optional
        Data array for group 2. Required for paired/two-sample tests.
    test_type : {'paired', 'one-sample', 'two-sample'} or TestType
        Type of statistical test.

    Returns
    -------
    dict
        Dictionary with 'positive' and 'negative' t-statistic arrays.

    Raises
    ------
    ValueError
        If test_type is invalid or dimensions don't match.
    """
    # Normalize test_type to string
    test_type_str = test_type.value if isinstance(test_type, TestType) else test_type

    if test_type_str == TestType.ONE_SAMPLE.value:
        if group2 is not None:
            logger.warning("group2 is provided but test_type is 'one-sample'. It will be ignored.")
        if group1.ndim != 3:
            raise ValueError("Dimensions of group 1 data should be: (subjects, N, N).")
        return compute_t_stat_diff(group1)

    elif test_type_str == TestType.TWO_SAMPLE.value:
        if group1.shape[1:] != group2.shape[1:]:
            raise ValueError("Trailing dimensions of group1 and group2 must match.")
        return _compute_t_stat_ind(group1, group2)

    elif test_type_str == TestType.PAIRED.value:
        _check_paired_counts(group1, group2, test_type_str)
        diffs = group2 - group1
        return compute_t_stat_diff(diffs)

    else:
        raise ValueError(
            f"Invalid test_type: '{test_type_str}'. "
            f"Must be one of: {[t.value for t in TestType]}"
        )


def compute_t_stat_diff(
    diff: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute t-statistics for paired differences.

    Parameters
    ----------
    diff : ndarray of shape ``(n_samples, *dims)``
        Array containing paired differences.

    Returns
    -------
    dict
        Dictionary with:

        - 'positive': Positive t-values (where group 2 > group 1).
        - 'negative': Negative t-values converted to positive.

    Raises
    ------
    ValueError
        If fewer than 2 samples are provided.
    """
    n = diff.shape[0]
    if n < 2:
        raise ValueError("At least 2 samples required for t-statistic.")

    x_mean = np.mean(diff, axis=0)
    x_var = np.var(diff, axis=0, ddof=1)
    meansq = (n - 1) / n * x_var + x_mean ** 2

    # zero variance: t = 0 (see _degenerate; entry points reject a nonzero mean)
    t_stat = _degenerate.t_ratio(x_mean, x_var / n, _degenerate.RTOL * meansq / n)

    pos_t = np.where(t_stat > 0, t_stat, 0)
    neg_t = np.where(t_stat < 0, -t_stat, 0)

    return make_tail_result(pos_t, neg_t)


def _compute_t_stat_ind(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Compute Welch's t-statistics for independent samples.

    Parameters
    ----------
    group1 : ndarray of shape (n_samples_1, *dims)
        Data array for group 1.
    group2 : ndarray of shape (n_samples_2, *dims)
        Data array for group 2.

    Returns
    -------
    dict
        Dictionary with:

        - 'positive': Positive t-values (where group 2 > group 1).
        - 'negative': Negative t-values converted to positive.

    Raises
    ------
    ValueError
        If either group has fewer than 2 samples.
    """
    n1, n2 = group1.shape[0], group2.shape[0]
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 samples.")

    x_mean_1 = np.mean(group1, axis=0)
    x_mean_2 = np.mean(group2, axis=0)
    x_var_1 = np.var(group1, axis=0, ddof=1) / n1
    x_var_2 = np.var(group2, axis=0, ddof=1) / n2
    meansq = ((n1 - 1) * x_var_1 + x_mean_1 ** 2) * n1 / (n1 + n2) \
        + ((n2 - 1) * x_var_2 + x_mean_2 ** 2) * n2 / (n1 + n2)

    # zero variance: t = 0 (see _degenerate; entry points reject a nonzero difference)
    t_stat = _degenerate.t_ratio(x_mean_2 - x_mean_1, x_var_1 + x_var_2,
                                 _degenerate.RTOL * meansq * (1 / n1 + 1 / n2))

    pos_t = np.where(t_stat > 0, t_stat, 0)
    neg_t = np.where(t_stat < 0, -t_stat, 0)

    return make_tail_result(pos_t, neg_t)
