"""Upper-triangle conversions used by ComBat."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


def flatten_upper(
    Y: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], Tuple[npt.NDArray, npt.NDArray], int]:
    """
    Flatten the upper triangle (k=1) of a (n, N, N) connectivity stack.

    Parameters
    ----------
    Y : ndarray of shape (n_subjects, N, N)
        Symmetric connectivity matrices with zero diagonal.

    Returns
    -------
    features : ndarray of shape (n_subjects, N*(N-1)/2)
        Flattened upper triangles — one feature per edge.
    triu_idx : tuple of ndarray
        ``(rows, cols)`` index tuple usable as ``M[triu_idx]`` to invert.
    N : int
        Side length of the original matrices.
    """
    if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
        raise ValueError(f"Y must have shape (n, N, N), got {Y.shape}.")
    N = Y.shape[1]
    triu_idx = np.triu_indices(N, k=1)
    features = Y[:, triu_idx[0], triu_idx[1]]
    return features, triu_idx, N


def unflatten_upper(
    features: npt.NDArray[np.float64],
    triu_idx: Tuple[npt.NDArray, npt.NDArray],
    N: int,
) -> npt.NDArray[np.float64]:
    """
    Reconstruct (n, N, N) symmetric matrices (zero diagonal) from flattened
    upper-triangle features.
    """
    n = features.shape[0]
    Y = np.zeros((n, N, N), dtype=features.dtype)
    Y[:, triu_idx[0], triu_idx[1]] = features
    Y[:, triu_idx[1], triu_idx[0]] = features
    return Y
