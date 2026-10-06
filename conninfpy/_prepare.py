"""Input preparation shared by every inference entry point.

Connectivity stacks enter the package in exactly one way:

- :func:`check_connectivity` — shape check and zero diagonal. Used by the
  advanced pipelines (``compute_p_val*``, ``nbs_bct``), which expect data
  already on the analysis scale (e.g. Fisher z).
- :func:`prepare_connectivity` — the same check plus a single Fisher r→z
  transform when the input holds correlations. Used by
  :func:`conninfpy.analyze`.
- :func:`infer_input_kind` — tells correlations from Fisher-z values when
  the data make it unambiguous.

The caller's array is never modified; a copy is made only when the
diagonal actually needs zeroing. Symmetry is not required — directed
(asymmetric) connectivity is supported downstream.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from .utils import fisher_r_to_z

# Subjects scanned per block in infer_input_kind, so large stacks don't
# allocate a full-size |Y| temporary.
_SCAN_BLOCK = 64


def check_connectivity(
    Y: npt.ArrayLike, name: str = "Y"
) -> npt.NDArray[np.float64]:
    """Validate an ``(n_subjects, N, N)`` stack and zero its diagonal.

    Parameters
    ----------
    Y : array_like of shape (n_subjects, N, N)
        Per-subject connectivity matrices.
    name : str, optional
        Argument name used in error messages.

    Returns
    -------
    np.ndarray of shape (n_subjects, N, N)
        float64 stack with a zero diagonal. The input itself when it is
        already float64 with a zero diagonal; otherwise a new array.

    Raises
    ------
    ValueError
        If ``Y`` is not a stack of square matrices.
    """
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(
            f"{name} must have shape (n_subjects, N, N); got {arr.shape}."
        )
    idx = np.arange(arr.shape[-1])
    if np.any(arr[:, idx, idx] != 0):  # NaN != 0, so NaN diagonals are zeroed too
        if isinstance(Y, np.ndarray) and np.shares_memory(arr, Y):
            arr = arr.copy()  # never write into the caller's array
        arr[:, idx, idx] = 0.0
    return arr


def infer_input_kind(*arrays: npt.NDArray[np.float64]) -> Optional[str]:
    """Guess whether stacks hold correlations (``'r'``) or Fisher-z (``'z'``).

    Returns ``'z'`` when any off-diagonal value exceeds 1 in magnitude,
    ``'r'`` when the diagonal is 1, and ``None`` when the data cannot
    tell (all values in [-1, 1] and a diagonal that is not ones — e.g.
    correlations whose diagonal was already zeroed, or weak Fisher-z
    connectivity). Empty rows (all zero or NaN: a region with no usable
    signal in that subject) are ignored when checking the diagonal.
    """
    offdiag_max = 0.0
    diag_is_one = True
    any_one = False
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        n_nodes = arr.shape[-1]
        offdiag = ~np.eye(n_nodes, dtype=bool)
        idx = np.arange(n_nodes)
        for start in range(0, arr.shape[0], _SCAN_BLOCK):
            raw = arr[start:start + _SCAN_BLOCK]
            block = np.abs(raw)
            if block[:, offdiag].size:
                offdiag_max = max(offdiag_max, float(np.nanmax(block[:, offdiag])))
            diag = raw[:, idx, idx]
            is_one = np.isclose(diag, 1.0)
            empty_row = np.all((raw == 0) | np.isnan(raw), axis=2)
            diag_is_one = diag_is_one and bool(np.all(is_one | empty_row))
            any_one = any_one or bool(is_one.any())
    if offdiag_max > 1.0:
        return "z"
    if diag_is_one and any_one:
        return "r"
    return None


def prepare_connectivity(
    Y: npt.ArrayLike, input: str, name: str = "Y"
) -> npt.NDArray[np.float64]:
    """Check ``Y``, zero its diagonal, and apply Fisher r→z once if needed.

    Parameters
    ----------
    Y : array_like of shape (n_subjects, N, N)
        Per-subject connectivity matrices.
    input : {'r', 'z'}
        ``'r'`` for correlations (Fisher r→z is applied), ``'z'`` for data
        already on the analysis scale (left unchanged apart from the
        diagonal).
    name : str, optional
        Argument name used in error messages.

    Returns
    -------
    np.ndarray of shape (n_subjects, N, N)
        Prepared float64 stack with a zero diagonal.
    """
    if input not in ("r", "z"):
        raise ValueError(f"input must be 'r' or 'z'; got {input!r}.")
    arr = check_connectivity(Y, name)
    if input == "r":
        # Diagonal is already zero, so arctanh never sees r = 1 there.
        arr = fisher_r_to_z(arr)
    return arr
