"""Functional-block-clustering TFNBS scoring."""

from __future__ import annotations


import numpy as np
import numpy.typing as npt

from conninfpy.defaults import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_START_THRESHOLD,
)
from conninfpy.enhance._blocks import (
    _compute_canonical_block_ids,
    _validate_net_labels,
)
from conninfpy.enhance._graph import _get_edges, _is_symmetric
from conninfpy.enhance._thresholds import (
    ArrayLike,
    _compute_thresholds,
    _extent_power,
    _validate_params,
)


def get_fbc_tfnbs_score(
    t_stats: npt.NDArray[np.floating],
    net_labels: npt.NDArray[np.integer],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> npt.NDArray[np.floating]:
    """
    Functional Block Clustering TFNBS (FBC-TFNBS).

    This method clusters edges by functional block membership rather than
    topological connectivity. At each threshold:

    1. Edges are grouped by their functional block (e.g., Visual-Visual, DMN-DMN)
    2. If a block has k >= min_cluster_size edges, those edges support each other
       (support = k)
    3. If a block has fewer edges, they are considered "isolated" and suppressed
       (support = 0)

    Key difference from standard TFNBS:
        - Standard TFNBS: edges cluster if they share a node (topological)
        - FBC-TFNBS: edges cluster if they're in the same functional block

    Key difference from NI-TFNBS:
        - NI-TFNBS: edge-level density weighting within predefined blocks
        - FBC-TFNBS: no topological clustering, isolated edges are suppressed

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed (usually absolute t-values).
    net_labels : ndarray of shape (N,)
        Integer labels assigning each node to a functional network.
    e : float or array-like
        Extent exponent. Controls how much cluster size matters.
    h : float or array-like
        Height exponent. Controls how much threshold height matters.
    n : int
        Number of threshold steps between start_thres and max(t_stats).
    start_thres : float, default=1.65
        Initial threshold for cluster formation.
    min_cluster_size : int, default=3
        Minimum number of edges in a functional block to form a cluster.
        Blocks with fewer edges get support=0 (suppressed).

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        FBC-TFNBS score matrix.

    Notes
    -----
    The key insight is that edges in the same functional block should support
    each other even if they don't share any nodes. This captures the biological
    reality that effects within a functional system (e.g., Visual cortex) are
    likely related, while isolated edges are more likely noise.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple t-statistic matrix
    >>> t = np.array([[0, 2.5, 1.8, 0.5],
    ...               [2.5, 0, 2.2, 0.3],
    ...               [1.8, 2.2, 0, 2.0],
    ...               [0.5, 0.3, 2.0, 0]])
    >>> # Nodes 0,1 in network 0; nodes 2,3 in network 1
    >>> labels = np.array([0, 0, 1, 1])
    >>> scores = get_fbc_tfnbs_score(t, labels, e=0.5, h=2.0, n=20,
    ...                               start_thres=1.5, min_cluster_size=2)
    """
    # 1. Validation
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)
    nroi = t_stats.shape[0]

    normalized_net_labels = _validate_net_labels(np.asarray(net_labels), nroi)
    n_networks = np.max(normalized_net_labels) + 1

    # Round to avoid float precision issues
    t_stats = np.round(t_stats, decimals=10)

    # Initialize output
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    # 2. Compute thresholds
    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    # 3. Pre-processing
    is_symm = _is_symmetric(t_stats)

    # Extract edges above start threshold
    edge_rows, edge_cols, edge_weights = _get_edges(
        t_stats, start_thres, symmetric=is_symm
    )

    if len(edge_rows) == 0:
        return tfnbs

    # Compute canonical block IDs for all extracted edges
    edge_block_ids = _compute_canonical_block_ids(
        edge_rows, edge_cols, normalized_net_labels, n_networks
    )

    # Maximum possible block ID (for bincount)
    max_block_id = n_networks * n_networks

    # Prepare broadcasting parameters
    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr
        h_bc = h_arr

    # 4. Main Integration Loop
    for threshold in threshs:
        # Find active edges at this threshold
        active_mask = edge_weights >= threshold

        if not np.any(active_mask):
            continue

        active_indices = np.where(active_mask)[0]
        active_rows = edge_rows[active_indices]
        active_cols = edge_cols[active_indices]
        active_block_ids = edge_block_ids[active_indices]

        # Count edges per functional block at this threshold
        block_counts = np.bincount(active_block_ids, minlength=max_block_id)

        # Compute support for each active edge
        # Support = block_count if block_count >= min_cluster_size, else 0
        edge_block_counts = block_counts[active_block_ids]
        edge_supports = np.where(
            edge_block_counts >= min_cluster_size, edge_block_counts, 0
        ).astype(np.float64)

        # Calculate score increment: h^H * support^E * dh
        # Suppressed edges (support 0) add nothing, also at E = 0 (0**0 = 1)
        if scalar_mode:
            increment = (threshold**h_bc) * _extent_power(edge_supports, e_bc) * dh
            tfnbs[active_rows, active_cols] += increment
            if is_symm:
                tfnbs[active_cols, active_rows] += increment
        else:
            # Handle parameter sweep
            increment = (
                (threshold**h_bc)
                * _extent_power(edge_supports[..., np.newaxis], e_bc)
                * dh
            )
            tfnbs[active_rows, active_cols, :] += increment
            if is_symm:
                tfnbs[active_cols, active_rows, :] += increment

    return tfnbs
