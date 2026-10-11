"""Graph checks shared by network enhancement scores."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


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
    t_stats: npt.NDArray[np.floating], start_thres: float, symmetric: bool
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

    Returns
    -------
    row_indices : ndarray
        Row indices of valid edges.
    col_indices : ndarray
        Column indices of valid edges.
    edge_weights : ndarray
        Edge weight values.
    """
    if symmetric:
        rows, cols = np.triu_indices_from(t_stats, k=1)
    else:
        nroi = t_stats.shape[0]
        rows, cols = np.where(~np.eye(nroi, dtype=bool))

    weights = t_stats[rows, cols]
    valid_mask = weights > start_thres
    return rows[valid_mask], cols[valid_mask], weights[valid_mask]
