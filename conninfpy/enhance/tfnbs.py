"""
Threshold-Free Network-Based Statistics (TFNBS) scoring module.

This module provides implementations of TFCE (Threshold-Free Cluster Enhancement)
adapted for network/connectivity analysis. The main function is `get_tfnbs_score()`
which transforms t-statistic matrices into TFNBS scores.

Functions
---------
get_tfnbs_score : Main optimized TFNBS implementation using scipy
get_tfnbs_score_baseline : Non-optimized version for benchmarking
get_tfnbs_score_networkx : Legacy networkx implementation for comparison with other packages
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from conninfpy.enhance._graph import _get_edges, _is_symmetric
from conninfpy.enhance._thresholds import (
    ArrayLike,
    _check_domain,
    _compute_thresholds,
    _extent_power,
    _validate_params,
)

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Fallback decorator when numba is not installed."""

        def decorator(func):
            """No-op stand-in for numba's njit when numba is absent."""
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


__all__ = [
    "get_tfnbs_score",
    "get_network_informed_tfnbs_score",
    "get_fbc_tfnbs_score",
    "HAS_NUMBA",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_EXTENT_EXPONENT",
    "DEFAULT_HEIGHT_EXPONENT",
    "DEFAULT_N_THRESHOLDS",
    "DEFAULT_MIN_CLUSTER_SIZE",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Constants — imported from unified defaults
# =============================================================================

from conninfpy.defaults import (
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_START_THRESHOLD,
)
from conninfpy.defaults import DEFAULT_N_THRESHOLDS_SCORING as DEFAULT_N_THRESHOLDS

# =============================================================================
# Type aliases
# =============================================================================


# =============================================================================
# Helper functions
# =============================================================================


def get_tfnbs_score(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
    backend: str = "auto",
) -> npt.NDArray[np.floating]:
    """
    Transform the connectivity matrix using Threshold-Free Network-Based Statistics.

    Optimized implementation using scipy's csgraph module with:

    - Pre-extracted edges to avoid repeated masking
    - Buffer reuse to minimize memory allocations
    - Vectorized cluster size computation

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent. Can be a scalar or list of values for parameter sweep.
    h : float or array-like
        Height exponent. Can be a scalar or list of values for parameter sweep.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.
    backend : {'auto', 'numba', 'scipy'}, default='auto'
        Computation backend. 'auto' uses numba if available, otherwise scipy.
        'numba' uses JIT-compiled union-find for connected components
        (8-30x faster). Falls back to 'scipy' if numba is not installed.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix. Shape is (N, N) if e and h are scalars,
        otherwise (N, N, num_params).

    Examples
    --------
    >>> t = np.array([[0, 2.1, 0.5],[2.1, 0, 2.5],[0.5, 2.5, 0]])
    >>> np.round(get_tfnbs_score(t, e=0.5, h=2.0, n=10), 2)
    array([[0.  , 2.19, 0.  ],
           [2.19, 0.  , 4.5 ],
           [0.  , 4.5 , 0.  ]])
    """
    use_numba = (backend == "numba") or (backend == "auto" and HAS_NUMBA)
    if use_numba:
        if not HAS_NUMBA:
            logger.warning("Numba not installed, falling back to scipy backend.")
        else:
            return _get_tfnbs_score_numba(t_stats, e, h, n, start_thres)
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    nroi = t_stats.shape[0]
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    is_symm = _is_symmetric(t_stats)
    edge_rows, edge_cols, edge_weights = _get_edges(
        t_stats, start_thres, symmetric=is_symm
    )

    if len(edge_rows) == 0:
        return tfnbs

    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr
        h_bc = h_arr

    # Pre-allocate reusable buffers
    mask = np.zeros((nroi, nroi), dtype=bool)
    clustsize = np.zeros((nroi, nroi), dtype=np.float64)

    for threshold in threshs:
        active_edges = edge_weights >= threshold

        if not np.any(active_edges):
            continue

        active_rows = edge_rows[active_edges]
        active_cols = edge_cols[active_edges]

        # Build mask for active edges
        mask.fill(False)
        mask[active_rows, active_cols] = True
        if is_symm:
            # For symmetric matrices, mirror to lower triangle
            mask[active_cols, active_rows] = True

        # For connected_components, always use symmetric mask (undirected graph)
        mask_for_cc = mask | mask.T
        sparse_mat = csr_matrix(mask_for_cc)
        n_components, node_labels = connected_components(sparse_mat, directed=False)

        # Count edges per component
        edge_component_ids = node_labels[active_rows]
        component_edge_counts = np.bincount(edge_component_ids, minlength=n_components)
        edge_sizes = component_edge_counts[edge_component_ids].astype(np.float64)

        clustsize.fill(0)
        clustsize[active_rows, active_cols] = edge_sizes
        if is_symm:
            # For symmetric matrices, copy to lower triangle
            clustsize[active_cols, active_rows] = edge_sizes

        if scalar_mode:
            tfnbs += _extent_power(clustsize, e_bc) * (threshold**h_bc)
        else:
            tfnbs += _extent_power(clustsize[..., np.newaxis], e_bc) * (threshold**h_bc)

    tfnbs *= dh
    return tfnbs


# =============================================================================
# Numba JIT backend
# =============================================================================


@njit(cache=True)
def _union_find_labels(rows, cols, n_nodes):
    """Union-Find connected components with path compression and union-by-rank.

    Parameters
    ----------
    rows, cols : int64 arrays
        Edge list (row[i], col[i]) for each edge.
    n_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    labels : int32 array of shape (n_nodes,)
        Component label for each node (compressed so root = label).
    """
    parent = np.arange(n_nodes, dtype=np.int32)
    rank = np.zeros(n_nodes, dtype=np.int32)

    for i in range(len(rows)):
        # find root of rows[i]
        a = rows[i]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        # find root of cols[i]
        b = cols[i]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            if rank[a] < rank[b]:
                parent[a] = b
            elif rank[a] > rank[b]:
                parent[b] = a
            else:
                parent[b] = a
                rank[a] += 1

    # Flatten labels
    labels = np.empty(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        labels[i] = x
    return labels


@njit(cache=True)
def _tfnbs_core_numba_scalar(
    edge_rows, edge_cols, edge_weights, thresholds, e, h, dh, nroi, is_symm
):
    """JIT-compiled TFNBS inner loop for scalar (e, h).

    Returns a flat array of shape (nroi * nroi,) representing the score matrix.
    """
    tfnbs = np.zeros(nroi * nroi, dtype=np.float64)

    for t_idx in range(len(thresholds)):
        threshold = thresholds[t_idx]

        # Count active edges
        n_active = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                n_active += 1

        if n_active == 0:
            continue

        # Collect active edges
        active_rows = np.empty(n_active, dtype=np.int64)
        active_cols = np.empty(n_active, dtype=np.int64)
        idx = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                active_rows[idx] = edge_rows[k]
                active_cols[idx] = edge_cols[k]
                idx += 1

        # Build symmetric edge list for union-find
        if is_symm:
            sym_rows = np.empty(n_active * 2, dtype=np.int64)
            sym_cols = np.empty(n_active * 2, dtype=np.int64)
            for k in range(n_active):
                sym_rows[k] = active_rows[k]
                sym_cols[k] = active_cols[k]
                sym_rows[n_active + k] = active_cols[k]
                sym_cols[n_active + k] = active_rows[k]
            labels = _union_find_labels(sym_rows, sym_cols, nroi)
        else:
            # For asymmetric, symmetrize for connected components
            sym_rows = np.empty(n_active * 2, dtype=np.int64)
            sym_cols = np.empty(n_active * 2, dtype=np.int64)
            for k in range(n_active):
                sym_rows[k] = active_rows[k]
                sym_cols[k] = active_cols[k]
                sym_rows[n_active + k] = active_cols[k]
                sym_cols[n_active + k] = active_rows[k]
            labels = _union_find_labels(sym_rows, sym_cols, nroi)

        # Count edges per component (using the original active edges)
        # First, find max label for bincount size
        max_label = 0
        for k in range(n_active):
            lbl = labels[active_rows[k]]
            if lbl > max_label:
                max_label = lbl
        comp_edge_counts = np.zeros(max_label + 1, dtype=np.int64)
        for k in range(n_active):
            comp_edge_counts[labels[active_rows[k]]] += 1

        # Accumulate scores
        h_power = threshold**h
        for k in range(n_active):
            edge_size = float(comp_edge_counts[labels[active_rows[k]]])
            score_inc = (edge_size**e) * h_power
            r, c = active_rows[k], active_cols[k]
            tfnbs[r * nroi + c] += score_inc
            if is_symm:
                tfnbs[c * nroi + r] += score_inc

    # Multiply by dh
    for i in range(len(tfnbs)):
        tfnbs[i] *= dh

    return tfnbs


@njit(cache=True)
def _tfnbs_core_numba_multi(
    edge_rows,
    edge_cols,
    edge_weights,
    thresholds,
    e_arr,
    h_arr,
    dh,
    nroi,
    n_params,
    is_symm,
):
    """JIT-compiled TFNBS inner loop for multiple (e, h) parameter pairs.

    Returns a flat array of shape (nroi * nroi * n_params,).
    """
    tfnbs = np.zeros(nroi * nroi * n_params, dtype=np.float64)

    for t_idx in range(len(thresholds)):
        threshold = thresholds[t_idx]

        # Count active edges
        n_active = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                n_active += 1

        if n_active == 0:
            continue

        # Collect active edges
        active_rows = np.empty(n_active, dtype=np.int64)
        active_cols = np.empty(n_active, dtype=np.int64)
        idx = 0
        for k in range(len(edge_weights)):
            if edge_weights[k] >= threshold:
                active_rows[idx] = edge_rows[k]
                active_cols[idx] = edge_cols[k]
                idx += 1

        # Build symmetric edge list for union-find
        sym_rows = np.empty(n_active * 2, dtype=np.int64)
        sym_cols = np.empty(n_active * 2, dtype=np.int64)
        for k in range(n_active):
            sym_rows[k] = active_rows[k]
            sym_cols[k] = active_cols[k]
            sym_rows[n_active + k] = active_cols[k]
            sym_cols[n_active + k] = active_rows[k]
        labels = _union_find_labels(sym_rows, sym_cols, nroi)

        # Count edges per component
        max_label = 0
        for k in range(n_active):
            lbl = labels[active_rows[k]]
            if lbl > max_label:
                max_label = lbl
        comp_edge_counts = np.zeros(max_label + 1, dtype=np.int64)
        for k in range(n_active):
            comp_edge_counts[labels[active_rows[k]]] += 1

        # Accumulate scores for all parameter pairs
        for k in range(n_active):
            edge_size = float(comp_edge_counts[labels[active_rows[k]]])
            r, c = active_rows[k], active_cols[k]
            for p in range(n_params):
                score_inc = (edge_size ** e_arr[p]) * (threshold ** h_arr[p])
                tfnbs[(r * nroi + c) * n_params + p] += score_inc
                if is_symm:
                    tfnbs[(c * nroi + r) * n_params + p] += score_inc

    # Multiply by dh
    for i in range(len(tfnbs)):
        tfnbs[i] *= dh

    return tfnbs


def _get_tfnbs_score_numba(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """TFNBS scoring using Numba JIT backend.

    Same interface as get_tfnbs_score() but uses JIT-compiled union-find
    for connected components instead of scipy.sparse.csgraph.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent.
    h : float or array-like
        Height exponent.
    n : int
        Number of threshold steps.
    start_thres : float, default=1.65
        Initial threshold for cluster formation.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix.
    """
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)
    t_stats = np.round(t_stats, decimals=10)

    nroi = t_stats.shape[0]
    num_params = len(e_arr)

    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
        return np.zeros(tfnbs_shape)

    is_symm = _is_symmetric(t_stats)
    edge_rows, edge_cols, edge_weights = _get_edges(
        t_stats, start_thres, symmetric=is_symm
    )

    if len(edge_rows) == 0:
        tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
        return np.zeros(tfnbs_shape)

    # Ensure int64 for numba
    edge_rows = edge_rows.astype(np.int64)
    edge_cols = edge_cols.astype(np.int64)
    edge_weights = edge_weights.astype(np.float64)
    e_arr = e_arr.astype(np.float64)
    h_arr = h_arr.astype(np.float64)

    if scalar_mode:
        flat = _tfnbs_core_numba_scalar(
            edge_rows,
            edge_cols,
            edge_weights,
            threshs,
            e_arr[0],
            h_arr[0],
            dh,
            nroi,
            is_symm,
        )
        return flat.reshape(nroi, nroi)
    else:
        flat = _tfnbs_core_numba_multi(
            edge_rows,
            edge_cols,
            edge_weights,
            threshs,
            e_arr,
            h_arr,
            dh,
            nroi,
            num_params,
            is_symm,
        )
        return flat.reshape(nroi, nroi, num_params)


def get_tfnbs_score_baseline(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """
    Baseline (non-optimized) implementation of TFNBS for performance comparison.

    .. deprecated:: 2.0
        Removed from the public API in v2.0; the function lives on only as
        an equivalence reference for internal tests. Will be deleted in
        v2.1. Use :func:`get_tfnbs_score` instead — it is the optimised
        production implementation.

    This is the original scipy-based implementation without optimizations.
    Use `get_tfnbs_score()` for production - it is ~1.5x faster.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent. Can be a scalar or list of values.
    h : float or array-like
        Height exponent. Can be a scalar or list of values.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix.

    Examples
    --------
    >>> t = np.array([[0, 2.1, 0.5],[2.1, 0, 2.5],[0.5, 2.5, 0]])
    >>> np.round(get_tfnbs_score_baseline(t, e=0.5, h=2.0, n=10), 2)
    array([[0.  , 2.19, 0.  ],
           [2.19, 0.  , 4.5 ],
           [0.  , 4.5 , 0.  ]])
    """
    if not np.all(np.diag(t_stats) == 0):
        raise ValueError(
            "Diagonal elements of the connectivity matrix must be zero (no self-connections)."
        )

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    scalar_mode = np.isscalar(e) and np.isscalar(h)
    if scalar_mode:
        e, h = [e], [h]

    e = np.array(e)
    h = np.array(h)
    if e.shape != h.shape:
        raise ValueError("e and h must have the same shape!")
    _check_domain(e, h)

    nroi = t_stats.shape[0]
    num_params = len(e)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    max_stat = np.max(t_stats)
    dh = (max_stat - start_thres) / n
    if dh == 0:
        return tfnbs
    threshs = np.linspace(start_thres + dh, max_stat, n)

    if not scalar_mode:
        e = e.reshape(1, 1, num_params)
        h = h.reshape(1, 1, num_params)

    for threshold in threshs:
        mask = t_stats >= threshold
        np.fill_diagonal(mask, False)
        n_components, labels = connected_components(mask.astype(int), directed=False)

        unique, counts = np.unique(labels, return_counts=True)
        clustsize = 1.0 * mask.copy()

        for lbl, size in zip(unique, counts):
            if size >= 2:
                sz_links = np.sum(mask[np.ix_(labels == lbl, labels == lbl)]) / 2
                clustsize[np.ix_(labels == lbl, labels == lbl)] *= sz_links

        np.fill_diagonal(clustsize, 0)

        if scalar_mode:
            tfnbs += _extent_power(clustsize, e[0]) * (threshold ** h[0])
        else:
            tfnbs += _extent_power(clustsize[..., np.newaxis], e) * (threshold**h)

    tfnbs *= dh
    return tfnbs


def get_tfnbs_score_networkx(
    t_stats: npt.NDArray[np.floating],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """
    Transform the connectivity matrix using Threshold-Free Network-Based Statistics.

    .. deprecated:: 2.0
        Removed from the public API in v2.0; the function lives on only as
        an equivalence reference for internal tests. Will be deleted in
        v2.1. Use :func:`get_tfnbs_score` instead.

    Legacy networkx implementation - use `get_tfnbs_score()` for better performance.
    This function is kept for comparison with other packages that use networkx.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    e : float or array-like
        Extent exponent. Can be a scalar or list of values.
    h : float or array-like
        Height exponent. Can be a scalar or list of values.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        TFNBS score matrix.

    Examples
    --------
    >>> t = np.array([[0, 2.1, 0.5],[2.1, 0, 2.5],[0.5, 2.5, 0]])
    >>> np.round(get_tfnbs_score_networkx(t, e=0.5, h=2.0, n=10), 2)
    array([[0.  , 2.19, 0.  ],
           [2.19, 0.  , 4.5 ],
           [0.  , 4.5 , 0.  ]])
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for get_tfnbs_score_networkx(). "
            "Use get_tfnbs_score() instead."
        )

    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)

    nroi = t_stats.shape[0]
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr.reshape(1, 1, num_params)
        h_bc = h_arr.reshape(1, 1, num_params)

    clustsize = np.zeros((nroi, nroi), dtype=np.float64)

    for threshold in threshs:
        mask = t_stats >= threshold
        np.fill_diagonal(mask, False)

        if not np.any(mask):
            continue

        G = nx.from_numpy_array(mask)
        components = list(nx.connected_components(G))

        clustsize.fill(0)
        clustsize[mask] = 1.0

        for component in components:
            if len(component) >= 2:
                component_list = list(component)
                sz_links = np.sum(mask[np.ix_(component_list, component_list)]) / 2
                clustsize[np.ix_(component_list, component_list)] *= sz_links

        np.fill_diagonal(clustsize, 0)

        if scalar_mode:
            tfnbs += _extent_power(clustsize, e_bc) * (threshold**h_bc)
        else:
            tfnbs += _extent_power(clustsize[..., np.newaxis], e_bc) * (threshold**h_bc)

    tfnbs *= dh
    return tfnbs


"""
Network-Informed TFNBS scoring module.

This module extends TFNBS to incorporate functional network architecture.
It weights topological clusters based on the density of the functional blocks
they span, increasing sensitivity to effects that align with known
functional anatomy.
"""


# =============================================================================
# Specific Helper Functions
# =============================================================================


# =============================================================================
# Main Function
# =============================================================================


# =============================================================================
# Functional Block Clustering TFNBS (FBC-TFNBS)
# =============================================================================


# Compatibility exports of the other TFNBS family members.
from conninfpy.enhance._blocks import (
    _compute_canonical_block_ids,
    _compute_edge_block_weights,
    _precompute_block_metadata,
    _validate_net_labels,
)
from conninfpy.enhance.fbc_tfnbs import get_fbc_tfnbs_score
from conninfpy.enhance.ni_tfnbs import get_network_informed_tfnbs_score
