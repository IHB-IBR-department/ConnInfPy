"""Statistical utilities for ConnInfPy.

Includes:

- :func:`fisher_r_to_z` / :func:`fisher_z_to_r` — Fisher transformations.
- :func:`get_components` — connected components (BCT-style).
- :func:`binarize` — threshold-and-binarize a connectivity matrix.
- :func:`create_prior_weights` — block-density priors used by NI-TFNBS.
"""

import warnings
from typing import Optional

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


def fisher_r_to_z(r: npt.NDArray[np.float64],
                  handle_bounds: bool = True,
                  max_z: float = 5) -> npt.NDArray[np.float64]:
    """
    Apply Fisher r-to-z transformation to correlation coefficients, handling boundary cases.

    Parameters
    ----------
    r : np.ndarray
        Array of correlation coefficients with values in ``[-1, 1]``.
        Can be any shape (scalar, 1D, 2D, ...).
    handle_bounds : bool, optional
        If True, replace infinite z-values (from ``r = ±1``) with finite
        values. If False, allow infinities and raise a warning.
        (default: True).
    max_z : float
        Maximum absolute z-value to use when ``handle_bounds=True``
        (default: 5).

    Returns
    -------
    z : np.ndarray
        Array of z-scores with the same shape as ``r``.

    Raises
    ------
    ValueError
        If any value in ``r`` is outside ``[-1, 1]``.

    Warnings
    --------
    UserWarning
        If ``r`` contains ``±1``, indicating boundary values were
        encountered.

    Notes
    -----
    The transformation is :math:`z = \\operatorname{arctanh}(r)`. For
    :math:`r = \\pm 1`, :math:`z` approaches :math:`\\pm\\infty`. When
    ``handle_bounds=True``, these are capped at ``±max_z``.

    Examples
    --------
    >>> r_vals = np.array([1.0, -1.0])
    >>> fisher_r_to_z(r_vals, handle_bounds=True, max_z=4.0)
    array([ 4., -4.])

    """
    r = np.asarray(r, dtype=np.float64)

    if np.any((r < -1) | (r > 1)):
        raise ValueError("Correlation coefficients must be in the range [-1, 1].")

    with np.errstate(invalid='ignore'):  # arctanh(±1) = ±inf, handled below
        z = np.arctanh(r)

    bounds_mask = np.isclose(r, 1.0) | np.isclose(r, -1.0)
    if np.any(bounds_mask):
        warnings.warn(
            "Input contains r = ±1, resulting in infinite z-values. "
            f"{'Capped at ±' + str(max_z) if handle_bounds else 'Left as infinity.'}",
            UserWarning
        )
        if handle_bounds:
            z = np.where(bounds_mask, np.sign(r) * max_z, z)

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
    if not np.all(A == A.T):
        raise AssertionError('get_components can only be computed for undirected'
                             ' matrices.  If your matrix is noisy, correct it with np.around')

    A = binarize(A, copy=True)
    np.fill_diagonal(A, 1)

    n_comp, comps = csgraph.connected_components(A, directed=False)
    comps = comps.astype(np.int64) + 1
    comp_sizes = np.bincount(comps)[1:]
    return comps, comp_sizes


def binarize(W, copy=True):
    '''
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix

    >>> W = np.array([
    ...     [0.0, 2.5, 0.0],
    ...     [1.1, 0.0, 0.3],
    ...     [0.0, 0.0, 0.0]
    ... ])
    >>> binarize(W)
    array([[0., 1., 0.],
           [1., 0., 1.],
           [0., 0., 0.]])
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def create_prior_weights(node_labels: npt.NDArray[np.int_],
                         target_network_id: Optional[int] = None,
                         boost_factor: float = 2.0) -> npt.NDArray[np.float64]:
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
        same_network = np.logical_and(same_network, target_mask[:, None] & target_mask[None, :])

    np.fill_diagonal(same_network, False)

    weights[same_network] = float(boost_factor)

    return weights
