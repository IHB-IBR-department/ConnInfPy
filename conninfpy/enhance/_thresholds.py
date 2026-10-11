"""Threshold grids, parameter validation and extent powers for TFNBS methods."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

ArrayLike = Union[float, List[float], npt.NDArray[np.floating]]


def _validate_params(
    t_stats: npt.NDArray[np.floating], e: ArrayLike, h: ArrayLike
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], bool]:
    """
    Validate input dimensions and unify scalar/array parameters.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Input statistical matrix.
    e : float or array-like
        Extent parameter(s).
    h : float or array-like
        Height parameter(s).

    Returns
    -------
    e_arr : ndarray
        Extent parameter as 1D array.
    h_arr : ndarray
        Height parameter as 1D array.
    scalar_mode : bool
        True if both e and h were originally scalars.

    Raises
    ------
    ValueError
        If diagonal elements are non-zero or e/h shapes don't match.
    """
    if not np.all(np.diag(t_stats) == 0):
        raise ValueError(
            "Diagonal elements of the connectivity matrix must be zero (no self-connections)."
        )

    scalar_mode = np.isscalar(e) and np.isscalar(h)

    e_arr = np.atleast_1d(e)
    h_arr = np.atleast_1d(h)

    if e_arr.shape != h_arr.shape:
        raise ValueError("Parameters 'e' and 'h' must have the same shape.")
    _check_domain(e_arr, h_arr)

    return e_arr, h_arr, scalar_mode


def _check_domain(e_arr, h_arr) -> None:
    """E and H must be finite and >= 0.

    E = 0 drops the extent term (height-only integration); a negative E would
    reward small components and is infinite for unsupported edges (0**E). A
    negative H would weight low heights above high ones.
    """
    for name, arr in (("e", e_arr), ("h", h_arr)):
        arr = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(
                f"TFNBS parameter '{name}' must be finite and >= 0; got {arr.tolist()}."
            )


def _extent_power(size, e):
    """``size ** e`` on supported edges (size > 0) and 0 elsewhere.

    The explicit mask keeps unsupported edges at 0 for E = 0 as well, where
    0**0 = 1 would otherwise score every inactive edge and the diagonal (the
    Numba core only visits active edges, so this also keeps the backends equal).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(size > 0, size**e, 0.0)


def _compute_thresholds(
    t_stats: npt.NDArray[np.floating], n: int, start_thres: float
) -> Tuple[Optional[npt.NDArray[np.floating]], float]:
    """
    Compute threshold range for TFCE integration.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix.
    n : int
        Number of threshold steps.
    start_thres : float
        Initial threshold.

    Returns
    -------
    thresholds : ndarray or None
        Array of threshold values, or None if no valid range exists.
    dh : float
        Step size for integration.
    """
    if not np.isfinite(start_thres) or start_thres < 0:
        # heights are magnitudes (each tail is enhanced separately); a negative
        # cutoff makes threshold**H undefined for fractional H
        raise ValueError(f"start_thres must be finite and >= 0; got {start_thres}.")
    if int(n) != n or n < 1:
        raise ValueError(f"n (integration steps) must be a positive integer; got {n}.")
    max_stat = np.max(t_stats)
    dh = (max_stat - start_thres) / n

    if dh <= 0:
        return None, 0

    thresholds = np.linspace(start_thres + dh, max_stat, n)
    return thresholds, dh
