"""Statistical utilities for ConnInfPy.

Includes:

- :func:`fisher_r_to_z` / :func:`fisher_z_to_r` — Fisher transformations.
- :func:`get_components` — connected components (BCT-style).
- :func:`binarize` — threshold-and-binarize a connectivity matrix.
- :func:`create_prior_weights` — block-density priors used by NI-TFNBS.
"""

import warnings
from typing import Literal, Optional, Union, overload

import numpy as np
import numpy.typing as npt
from scipy.sparse import csgraph

__all__ = [
    "fisher_r_to_z",
    "fisher_z_to_r",
    "get_components",
    "binarize",
    "create_prior_weights",
]


def _clipped_mask(r: npt.NDArray[np.float64], *, policy: str, max_z: float,
                    clip_r: Optional[float], z: Optional[npt.NDArray[np.float64]]) -> npt.NDArray[np.bool_]:
    """Boolean mask of the values the active policy replaces (NaN: False)."""
    if policy == "clip_r":
        assert clip_r is not None  # the clip_r policy always carries its bound
        return np.abs(r) > clip_r
    # clip_z policy: |z| beyond max_z (r = ±1 gives |z| = inf, so it is included)
    assert z is not None  # the transformed values are what this policy inspects
    return np.abs(z) > max_z


def _clip_report(mask: npt.NDArray[np.bool_], r: npt.NDArray[np.float64], *,
                 policy: str, max_z: float, clip_r: Optional[float],
                 z: Optional[npt.NDArray[np.float64]] = None) -> dict:
    """Explicit clipping counts with separate units.

    ``matrix cells`` are counted per observation matrix (a cell clipped in
    two subjects counts twice); ``n_unique_undirected_edges_clipped`` counts
    upper-triangle positions (i < j) clipped in at least one observation;
    ``n_observations`` is the number of leading (N, N) matrices. The
    correlation diagonal (r = 1 by construction) is reported separately from
    observed off-diagonal values, so a structural diagonal is never reported
    as saturation of measured edges. Under the unbounded policy nothing is
    clipped: the clipped counts are zero and the values left at ±infinity
    are reported separately in ``left_at_infinity`` (same count units).
    """
    if policy == "clip_r":
        assert clip_r is not None  # the clip_r policy always carries its bound
        parameters = {"clip_r": float(clip_r)}
    elif policy == "unbounded":
        parameters = {}
    else:
        parameters = {"max_z": float(max_z)}
    report: dict = {
        "policy": policy,
        "parameters": parameters,
        "n_values_clipped": int(mask.sum()),
        "matrix_shape": None,
        "n_observations": None,
        "diagonal": None,
        "off_diagonal": None,
        "left_at_infinity": None,
        "count_units": (
            "n_values_clipped counts matrix cells when the input is a stack of "
            "(N, N) matrices (each cell counted once per observation); "
            "diagonal/off_diagonal n_cells count matrix cells; "
            "n_unique_undirected_edges_clipped counts upper-triangle positions "
            "(i < j) clipped in at least one observation; n_observations is the "
            "number of (N, N) matrices in the input stack."
        ),
    }
    arr = np.asarray(r)
    if arr.ndim < 2 or arr.shape[-2] != arr.shape[-1]:
        return report
    n = arr.shape[-1]
    cells = mask.reshape(-1, n, n)
    report["matrix_shape"] = (n, n)
    report["n_observations"] = int(cells.shape[0])
    diag_mask = np.zeros_like(cells)
    idx = np.arange(n)
    diag_mask[:, idx, idx] = cells[:, idx, idx]
    off_mask = cells & ~diag_mask
    report["diagonal"] = {"n_cells_clipped": int(diag_mask.sum())}
    iu, ju = np.triu_indices(n, k=1)
    upper = off_mask[:, iu, ju] | off_mask[:, ju, iu]  # either symmetric cell
    report["off_diagonal"] = {
        "n_cells_clipped": int(off_mask.sum()),
        "n_unique_undirected_edges_clipped": int(np.any(upper, axis=0).sum()),
    }
    if policy == "unbounded" and z is not None:
        inf_cells = np.isinf(np.asarray(z)).reshape(-1, n, n)
        inf_diag = np.zeros_like(inf_cells)
        inf_diag[:, idx, idx] = inf_cells[:, idx, idx]
        inf_off = inf_cells & ~inf_diag
        report["left_at_infinity"] = {
            "n_values": int(inf_cells.sum()),
            "diagonal": {"n_cells": int(inf_diag.sum())},
            "off_diagonal": {
                "n_cells": int(inf_off.sum()),
                "n_unique_undirected_edges": int(np.any(
                    inf_off[:, iu, ju] | inf_off[:, ju, iu], axis=0).sum()),
            },
        }
    return report


@overload
def fisher_r_to_z(
    r: npt.NDArray[np.float64], handle_bounds: bool = ..., max_z: float = ...,
    *, clip_r: Optional[float] = ..., return_report: Literal[False] = ...,
) -> "npt.NDArray[np.float64]": ...


@overload
def fisher_r_to_z(
    r: npt.NDArray[np.float64], handle_bounds: bool = ..., max_z: float = ...,
    *, clip_r: Optional[float] = ..., return_report: Literal[True],
) -> "tuple[npt.NDArray[np.float64], dict]": ...


def fisher_r_to_z(
    r: npt.NDArray[np.float64], handle_bounds: bool = True, max_z: float = 5,
    *, clip_r: Optional[float] = None, return_report: bool = False,
) -> "Union[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], dict]]":
    """
    Apply Fisher r-to-z transformation to correlation coefficients, handling boundary cases.

    Parameters
    ----------
    r : np.ndarray
        Array of correlation coefficients with values in ``[-1, 1]``.
        Can be any shape (scalar, 1D, 2D, ...).
    handle_bounds : bool, optional
        Clip-z policy: if True (default), clip z to ``[-max_z, max_z]``
        (this also makes ``r = ±1`` finite). If False, keep the exact
        transform, with infinities at ``r = ±1`` and a warning.
    max_z : float
        Clipping bound of the clip-z policy (default: 5, i.e.
        ``|r| > tanh(5) ≈ 0.99991``). Every value beyond it is clipped, so
        the transform stays monotone in ``r``.
    clip_r : float, optional
        Clip-r policy: clip correlations to ``[-clip_r, clip_r]`` *before*
        ``arctanh`` instead of clipping z afterwards (``0 < clip_r < 1``).
        This is the historical in-app "Apply Fisher-z now" behaviour
        (``clip_r=0.9999``, z bounded by ``atanh(0.9999) ≈ 4.9517``, no
        warning). Mutually exclusive with ``handle_bounds=False``; out-of-
        range input still raises.
    return_report : bool, optional
        Also return a report dict with the number of clipped values,
        separating the structural diagonal (``r = 1`` by construction) from
        observed off-diagonal values, with explicit count units (matrix
        cells vs unique undirected edges vs observations). Under the
        unbounded policy nothing is clipped: the clipped counts are zero
        and values left at ±infinity are reported separately
        (``left_at_infinity``). Default False returns only the transformed
        array.

    Returns
    -------
    z : np.ndarray
        Array of z-scores with the same shape as ``r`` (plus the report
        dict when ``return_report=True``).

    Raises
    ------
    ValueError
        If any value in ``r`` is outside ``[-1, 1]``, or ``clip_r`` is not
        a number in (0, 1), or both ``clip_r`` and
        ``handle_bounds=False`` are given.

    Warnings
    --------
    UserWarning
        Under the clip-z policy: if ``r`` contains ``±1`` or values beyond
        ``tanh(max_z)``. With ``handle_bounds=False``: if ``r`` contains
        ``±1`` (infinities are kept). The clip-r policy stays silent, as it
        has always been; use ``return_report`` for its counts.

    Notes
    -----
    The transformation is :math:`z = \\operatorname{arctanh}(r)`. For
    :math:`r = \\pm 1`, :math:`z` approaches :math:`\\pm\\infty`. When
    ``handle_bounds=True``, z is clipped at ``±max_z``. (Earlier versions
    replaced only values within ``np.isclose`` of ±1, so r = 0.99998 gave
    z = 5.76 while the larger r = 0.99999 gave 5.) Missing values (NaN)
    stay NaN.

    Examples
    --------
    >>> r_vals = np.array([1.0, -1.0])
    >>> fisher_r_to_z(r_vals, handle_bounds=True, max_z=4.0)
    array([ 4., -4.])

    """
    r = np.asarray(r, dtype=np.float64)

    if clip_r is not None:
        if isinstance(clip_r, bool) or not isinstance(clip_r, (int, float)) or not (0 < clip_r < 1):
            raise ValueError("clip_r must be a number in (0, 1).")
        if not handle_bounds:
            raise ValueError("clip_r and handle_bounds=False are mutually exclusive policies.")

    if np.any((r < -1) | (r > 1)):
        raise ValueError("Correlation coefficients must be in the range [-1, 1].")

    if clip_r is not None:
        # Clip-r policy: bound the correlations first, then transform. The
        # bounded r makes z finite by construction; no clipping of z, no
        # warning (historical in-app behaviour; counts via return_report).
        clipped = np.abs(r) > clip_r  # NaN compares False
        z = np.arctanh(np.clip(r, -clip_r, clip_r))
        if return_report:
            return z, _clip_report(clipped, r, policy="clip_r", max_z=max_z, clip_r=float(clip_r))
        return z

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # arctanh(±1): divide-by-zero → ±inf, handled below
        z = np.arctanh(r)

    if handle_bounds:
        over = np.abs(z) > max_z  # includes r = ±1; NaN compares False
        if np.any(over):
            warnings.warn(
                f"Input contains r = ±1 or |r| > tanh({max_z:g}) = {np.tanh(max_z):.5f} "
                f"({int(np.sum(over))} value(s)); z clipped at ±{max_z:g}.",
                UserWarning,
            )
            z = np.clip(z, -max_z, max_z)
        if return_report:
            return z, _clip_report(over, r, policy="clip_z", max_z=max_z, clip_r=None)
    else:
        if np.any(np.isinf(z)):
            warnings.warn(
                "Input contains r = ±1, resulting in infinite z-values. Left as infinity.",
                UserWarning,
            )
        if return_report:
            return z, _clip_report(np.zeros_like(np.isinf(z)), r, policy="unbounded",
                                   max_z=max_z, clip_r=None, z=z)

    return z


def fisher_z_to_r(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Fisher z-scores back to correlation coefficients.

    Parameters
    ----------
    z : np.ndarray
        Fisher z-scores, any shape.

    Returns
    -------
    np.ndarray
        Correlation coefficients in (-1, 1), same shape as ``z``.

    Examples
    --------
    >>> z_vals = np.array([0.0, 1.0, -1.0, 2.0])
    >>> fisher_z_to_r(z_vals).round(3)
    array([ 0.   ,  0.762, -0.762,  0.964])
    """
    z = np.asarray(z, dtype=np.float64)
    return np.tanh(z)


def get_components(A, no_depend=False):
    """Connected components of an undirected binary graph.

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Binary undirected adjacency matrix.
    no_depend : bool, optional
        Does nothing; kept for backwards compatibility.

    Returns
    -------
    comps : np.ndarray of shape (N,)
        Component index (1-based) assigned to each node.
    comp_sizes : np.ndarray of shape (M,)
        Number of nodes in each component.

    Raises
    ------
    ValueError
        If ``A`` is not symmetric.

    Notes
    -----
    Isolated nodes form components of size 1. Component numbering is not
    guaranteed to match the BCT Matlab implementation, although the
    component topology is. Thanks to Nick Cullen for the original
    implementation.

    Examples
    --------
    >>> A = np.eye(3)
    >>> comps, sizes = get_components(A)
    >>> comps
    array([1, 2, 3])
    >>> sizes
    array([1, 1, 1])
    """
    if not np.array_equal(
        A, A.T, equal_nan=True
    ):  # NaN != NaN; binarize drops it below
        raise ValueError(
            "get_components can only be computed for undirected"
            " matrices. If your matrix is noisy, correct it with np.around"
        )

    A = binarize(A, copy=True)
    np.fill_diagonal(A, 1)

    n_comp, comps = csgraph.connected_components(A, directed=False)
    comps = comps.astype(np.int64) + 1
    comp_sizes = np.bincount(comps)[1:]
    return comps, comp_sizes


def binarize(W, copy=True):
    """Binarize a weighted connectivity matrix: nonzero → 1, zero → 0.

    NaN counts as absent data and becomes 0 — missing values must not
    turn into present edges. With ``copy=False`` the input array is
    modified in place.

    Parameters
    ----------
    W : np.ndarray of shape (N, N)
        Weighted connectivity matrix.
    copy : bool, optional
        If True (default), return a new array; otherwise modify ``W``
        in place.

    Returns
    -------
    np.ndarray of shape (N, N)
        Binary adjacency matrix (0/1, same dtype behavior as input copy).

    Examples
    --------
    >>> W = np.array([
    ...     [0.0, 2.5, 0.0],
    ...     [1.1, 0.0, 0.3],
    ...     [0.0, 0.0, 0.0]
    ... ])
    >>> binarize(W)
    array([[0., 1., 0.],
           [1., 0., 1.],
           [0., 0., 0.]])
    >>> binarize(np.array([np.nan, 0.0, 2.0]))
    array([0., 0., 1.])
    """
    if copy:
        W = W.copy()
    W[np.isnan(W)] = 0  # missing data is not an edge
    W[W != 0] = 1
    return W


def create_prior_weights(
    node_labels: npt.NDArray[np.int_],
    target_network_id: Optional[int] = None,
    boost_factor: float = 2.0,
) -> npt.NDArray[np.float64]:
    """
    Create an (N, N) symmetric weight matrix based on node network labels.

    Parameters
    ----------
    node_labels : np.ndarray[int]
        1D array of integer network IDs for each node (length N).
    target_network_id : int or None, optional
        If provided, only edges where both nodes belong to this network are
        boosted. If None (default), all intra-network edges (nodes sharing
        the same label) are boosted.
    boost_factor : float, optional
        Multiplicative weight for priority edges (default 2.0).

    Returns
    -------
    weights : np.ndarray[float]
        Symmetric (N, N) weight matrix with background value 1.0 and
        `boost_factor` for priority intra-network edges. The diagonal is
        left at 1.0.

    Examples
    --------
    >>> labels = np.array([1, 1, 2, 2])
    >>> create_prior_weights(labels, boost_factor=3.0)
    array([[1., 3., 1., 1.],
           [3., 1., 1., 1.],
           [1., 1., 1., 3.],
           [1., 1., 3., 1.]])
    """
    labels = np.asarray(node_labels)
    if labels.ndim != 1:
        raise ValueError("`node_labels` must be a 1D array of integer labels")

    N = labels.shape[0]
    weights = np.ones((N, N), dtype=np.float64)

    same_network = labels[:, None] == labels[None, :]

    if target_network_id is not None:
        target_mask = labels == target_network_id
        same_network = np.logical_and(
            same_network, target_mask[:, None] & target_mask[None, :]
        )

    np.fill_diagonal(same_network, False)

    weights[same_network] = float(boost_factor)

    return weights
