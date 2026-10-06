"""Shared permutation machinery used by both the t-test and GLM pipelines.

Stratified (within-block) permutation, max-statistic extraction, collection
of per-permutation results, empirical p-values with the +1 Phipson–Smyth
correction, BH-FDR, and the worker-process / core-count helpers.
:mod:`conninfpy.pairwise_stats` and :mod:`conninfpy.glm_stats` build their
permutation loops from these pieces.
"""
from __future__ import annotations

import multiprocessing
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from ._compat import make_tail_result


def _encode_strata(strata: Any) -> npt.NDArray[np.int_]:
    """Map an arbitrary stratum-label sequence to contiguous integer codes."""
    arr = np.asarray(strata)
    _, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int_)


def _stratified_perm(
    strata: npt.NDArray[np.int_], rng: np.random.RandomState
) -> npt.NDArray[np.int_]:
    """Return a permutation index honoring exchangeability blocks.

    Within each unique stratum value, indices are independently
    permuted; across strata, indices do not move. This implements
    within-block exchangeability used by Freedman-Lane permutation
    on multi-site or otherwise blocked designs.

    Parameters
    ----------
    strata : ndarray of shape (n,)
        Integer stratum codes (use :func:`_encode_strata` first).
    rng : numpy.random.RandomState
        Already-seeded RNG.

    Returns
    -------
    perm : ndarray of shape (n,)
        Permutation index such that ``perm[i]`` is the new position of
        the data point originally at ``i``. Equivalent: for any array
        ``a``, ``a[perm]`` is the stratum-respecting permutation of ``a``.
    """
    n = strata.shape[0]
    perm = np.arange(n)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if idx.size > 1:
            perm[idx] = rng.permutation(idx)
    return perm


def _stratified_choice_n1(
    strata: npt.NDArray[np.int_],
    n1_per_stratum: npt.NDArray[np.int_],
    rng: np.random.RandomState,
) -> npt.NDArray[np.float64]:
    """Length-n binary mask honoring per-stratum group-1 counts.

    For the two-sample fast permutation path under exchangeability
    blocking: within each stratum ``s``, sample exactly
    ``n1_per_stratum[s]`` indices to label as group 1 (the rest are
    group 2). Across strata, the per-stratum group totals are held
    fixed.

    Parameters
    ----------
    strata : ndarray of shape (n,)
        Integer stratum codes.
    n1_per_stratum : ndarray of shape (n_unique_strata,)
        Group-1 count per stratum (computed once from the observed
        group labels and reused across permutations).
    rng : numpy.random.RandomState
    """
    n = strata.shape[0]
    g = np.zeros(n, dtype=np.float64)
    for s_code, n1_s in enumerate(n1_per_stratum):
        if n1_s <= 0:
            continue
        idx = np.where(strata == s_code)[0]
        chosen = rng.choice(idx, size=int(n1_s), replace=False)
        g[chosen] = 1.0
    return g


def _extract_max_stats(
    stat_dict: Dict[str, npt.NDArray],
    reference_shape: Tuple[int, ...]
) -> Dict[str, np.float64]:
    """
    Extract maximum statistics from a stat dictionary.

    Handles both 2D matrices and multi-parameter 3D arrays.
    Key-agnostic: works with any dictionary keys (e.g., 'negative'/'positive'
    for t-test pipeline, 'positive'/'negative' for GLM pipeline).

    Parameters
    ----------
    stat_dict : dict
        Dictionary with statistic arrays (any string keys).
    reference_shape : tuple
        Shape of a single sample for determining dimensionality.

    Returns
    -------
    dict
        Dictionary with maximum statistics for each key.
    """
    result: Dict[str, Any] = {}
    for key, arr in stat_dict.items():
        if arr.shape == reference_shape:
            result[key] = np.max(arr).astype(np.float64)
        else:
            # Multi-parameter case: max over spatial dims, keep param dim
            spatial_axes = tuple(range(arr.ndim - 1))
            result[key] = np.max(arr, axis=spatial_axes).astype(np.float64)
    if set(result.keys()) == {"positive", "negative"}:
        return make_tail_result(result["positive"], result["negative"])
    return result


def _collect_results_to_arrays(
    results: List[Dict[str, Any]],
    n_permutations: int
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Efficiently collect permutation results into numpy arrays.

    Uses pre-allocation and direct indexing instead of Python loops.

    Parameters
    ----------
    results : list of dict
        List of permutation result dictionaries.
    n_permutations : int
        Number of permutations.

    Returns
    -------
    dict
        Dictionary with arrays of shape (n_permutations,) or (n_permutations, n_params).
    """
    group_keys = list(results[0].keys())
    first_val = results[0][group_keys[0]]
    output_shape = first_val.shape if hasattr(first_val, 'shape') else ()

    t_maxes_dict = {
        key: np.empty((n_permutations, *output_shape), dtype=np.float64)
        for key in group_keys
    }

    for key in group_keys:
        for i, perm_dict in enumerate(results):
            t_maxes_dict[key][i] = perm_dict[key]

    if set(t_maxes_dict.keys()) == {"positive", "negative"}:
        return make_tail_result(t_maxes_dict["positive"], t_maxes_dict["negative"])
    return t_maxes_dict


def _is_worker_process() -> bool:
    """Check if running inside a multiprocessing worker.

    Returns True when the current process was spawned by a Pool,
    preventing nested Pool creation which causes deadlocks and
    unintentional process multiplication.
    """
    proc_name = multiprocessing.current_process().name
    return (
        proc_name != 'MainProcess' or 
        'PoolWorker' in proc_name or 
        'SpawnPoolWorker' in proc_name or
        'ForkPoolWorker' in proc_name
    )


def get_available_cores():
    """Number of CPU cores usable for worker pools."""
    try:
        # Linux
        affinity = os.sched_getaffinity(0)
        return len(affinity)
    except AttributeError:
        # Fallback Windows/Mac
        return multiprocessing.cpu_count()


def _compute_p_values_from_null(
    emp_t_dict: Dict[str, npt.NDArray],
    max_null_dict: Dict[str, npt.NDArray]
) -> Dict[str, npt.NDArray]:
    """
    Compute max-statistic p-values from empirical statistics and a null.

    Ties are counted in the conservative direction (``null >= observed``)
    and the Phipson-Smyth +1 correction is applied:
    ``p = (count + 1) / (B + 1)``. This keeps an all-zero observed map
    against an all-zero null at ``p = 1`` rather than ``1 / (B + 1)``.

    Parameters
    ----------
    emp_t_dict : dict
        Dictionary with empirical t-statistic arrays.
    max_null_dict : dict
        Dictionary with null distribution arrays.

    Returns
    -------
    dict
        Dictionary with p-value arrays for each direction.
    """
    keys = list(emp_t_dict.keys())
    p_values = {}

    is_2d = len(emp_t_dict[keys[0]].shape) == 2

    for key in keys:
        emp_t = emp_t_dict[key]
        null_dist = max_null_dict[key]

        if is_2d:
            # Shape: (N, N) vs (n_permutations,)
            n_perm = null_dist.shape[0]
            emp_t_expanded = emp_t[..., np.newaxis]
            count = np.sum(emp_t_expanded <= null_dist, axis=-1)
        else:
            # Multi-param: (N, N, n_params) vs (n_permutations, n_params)
            n_perm = null_dist.shape[0]
            emp_t_expanded = emp_t[..., np.newaxis]
            null_reshaped = null_dist.swapaxes(0, 1)[None, None, ...]
            count = np.sum(emp_t_expanded <= null_reshaped, axis=-1)

        # +1 correction (Phipson & Smyth 2010): prevents p = 0 with finite perms
        p_values[key] = (count + 1.0) / (n_perm + 1.0)

    return p_values


def _bh_fdr_correction(
    p_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Apply Benjamini-Hochberg FDR correction to a 1D array of p-values.

    Parameters
    ----------
    p_values : ndarray of shape (m,)
        Uncorrected p-values.

    Returns
    -------
    ndarray of shape (m,)
        BH-corrected p-values (adjusted so that thresholding at alpha
        controls FDR at level alpha).
    """
    m = len(p_values)
    if m == 0:
        return p_values.copy()

    # Sort p-values and track original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH adjustment: p_adj[i] = p[i] * m / rank[i]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * m / ranks

    # Enforce monotonicity (step-up): working backwards,
    # each adjusted p-value must be <= the next one
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Restore original order
    result = np.empty(m, dtype=np.float64)
    result[sorted_idx] = adjusted

    return result
