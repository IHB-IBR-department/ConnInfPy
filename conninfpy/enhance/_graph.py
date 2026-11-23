"""Graph checks shared by network enhancement scores."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from conninfpy.enhance._backend import HAS_NUMBA, njit


@njit(cache=True)
def _extract_numba(matrix, cutoff, symmetric):
    n = matrix.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1 if symmetric else 0, n):
            if i != j and matrix[i, j] > cutoff:
                count += 1
    rows = np.empty(count, dtype=np.int64)
    cols = np.empty(count, dtype=np.int64)
    weights = np.empty(count, dtype=matrix.dtype)
    k = 0
    for i in range(n):
        for j in range(i + 1 if symmetric else 0, n):
            if i != j and matrix[i, j] > cutoff:
                rows[k], cols[k], weights[k] = i, j, matrix[i, j]
                k += 1
    return rows, cols, weights


def _is_symmetric(matrix: npt.NDArray[np.floating], rtol: float = 1e-8) -> bool:
    """True when the matrix equals its transpose up to rounding.

    The tolerance scales with the largest magnitude (as in the input check):
    a purely relative test fails on near-zero statistics, whose two triangles
    can differ by ~1e-14 in absolute terms, and the map would then be scored
    as directed (each undirected edge counted twice).
    """
    m = np.asarray(matrix, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        scale = max(1.0, float(np.nanmax(np.abs(m))) if np.isfinite(m).any() else 1.0)
    return bool(np.allclose(m, m.T, rtol=0.0, atol=rtol * scale, equal_nan=True))


def _get_edges(
    t_stats: npt.NDArray[np.floating],
    start_thres: float,
    symmetric: bool,
    use_numba: bool = True,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """
    Extract edges with values above threshold.

    For symmetric matrices, extracts only upper triangle for efficiency.
    For asymmetric matrices, extracts all off-diagonal elements.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix.
    start_thres : float
        Minimum threshold for edge inclusion.
    symmetric : bool
        Whether matrix is symmetric (use upper triangle only).
    use_numba : bool, default=True
        Use the JIT extractor when Numba is installed and dtype is supported.

    Returns
    -------
    row_indices : ndarray
        Row indices of valid edges.
    col_indices : ndarray
        Column indices of valid edges.
    edge_weights : ndarray
        Edge weight values.
    """
    # Inputs have already been rounded by the calling scorer. Only dispatch to
    # the Numba extractor when Numba is actually installed; otherwise use
    # vectorized NumPy so the fallback does not run an interpreted double loop.
    if (
        HAS_NUMBA
        and use_numba
        and t_stats.dtype in (np.dtype("float64"), np.dtype("float32"), np.dtype("int64"))
    ):
        return _extract_numba(t_stats, start_thres, symmetric)
    if symmetric:
        rows, cols = np.triu_indices_from(t_stats, k=1)
    else:
        nroi = t_stats.shape[0]
        rows, cols = np.where(~np.eye(nroi, dtype=bool))
    weights = t_stats[rows, cols]
    valid_mask = weights > start_thres
    return rows[valid_mask], cols[valid_mask], weights[valid_mask]
