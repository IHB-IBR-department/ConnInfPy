"""Network-informed TFNBS: fixed-block capacity weighting."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from conninfpy.defaults import (
    DEFAULT_START_THRESHOLD,
)
from conninfpy.enhance._blocks import (
    _compute_edge_block_weights,
    _precompute_block_metadata,
    _validate_net_labels,
)
from conninfpy.enhance._graph import _get_edges, _is_symmetric
from conninfpy.enhance._thresholds import (
    ArrayLike,
    _compute_thresholds,
    _extent_power,
    _validate_params,
)


def get_network_informed_tfnbs_score(
    t_stats: npt.NDArray[np.floating],
    net_labels: npt.NDArray[np.integer],
    e: ArrayLike,
    h: ArrayLike,
    n: int,
    start_thres: float = DEFAULT_START_THRESHOLD,
    normalization: str = "sqrt",
    edge_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating]:
    """
    Transform connectivity matrix using Network-Informed TFNBS (Edge-Level Weighting).

    This method uses Edge-Level Weighting: each edge's support depends only on
    the density of its functional block, with NO topological clustering.
    Capacity M_b counts fixed eligible undirected edges, regardless of the
    observed/permuted statistic. Suprathreshold noise can still contribute
    to a block's density; this is not a guarantee that noise is excluded.

    Weighting Logic (depends on normalization):
        - ``'sqrt'``: S_edge = k / sqrt(M_b) (default)
        - ``'linear'``: S_edge = k / M_b (density)
        - ``'none'``: S_edge = k (raw active count)

    Key difference from standard TFNBS:
        - Standard TFNBS: edges cluster if they share a node (topological)
        - NI-TFNBS: each edge scored independently by block density

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Statistical matrix to be transformed.
    net_labels : ndarray of shape (N,)
        Integer labels assigning each node to a functional network.
    e : float or array-like
        Extent exponent.
    h : float or array-like
        Height exponent.
    n : int
        Number of threshold steps.
    start_thres : float, default=1.65
        Initial threshold for cluster formation.
    normalization : {'sqrt', 'linear', 'none'}, default='sqrt'
        Block density normalization variant for ablation study.
    edge_mask : ndarray of shape (N, N), optional
        Symmetric fixed eligibility mask. Ineligible edges contribute neither
        scores nor capacity. Defaults to all off-diagonal undirected pairs.

    Returns
    -------
    tfnbs : ndarray of shape (N, N) or (N, N, num_params)
        Network-Informed TFNBS score matrix.
    """
    # 1. Validation
    e_arr, h_arr, scalar_mode = _validate_params(t_stats, e, h)
    nroi = t_stats.shape[0]

    normalized_net_labels = _validate_net_labels(np.asarray(net_labels), nroi)
    if normalization not in ("sqrt", "linear", "none"):
        raise ValueError("normalization must be 'sqrt', 'linear', or 'none'.")
    if not _is_symmetric(t_stats):
        raise ValueError("NI-TFNBS requires symmetric undirected statistics.")
    if edge_mask is not None:
        edge_mask = np.asarray(edge_mask, dtype=bool)
        if edge_mask.shape != t_stats.shape or not np.array_equal(
            edge_mask, edge_mask.T
        ):
            raise ValueError(
                "edge_mask must be symmetric and match the statistic matrix."
            )
        t_stats = np.where(edge_mask, t_stats, 0.0)

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    # Initialize output
    num_params = len(e_arr)
    tfnbs_shape = (nroi, nroi) if scalar_mode else (nroi, nroi, num_params)
    tfnbs = np.zeros(tfnbs_shape)

    # 2. Thresholds
    threshs, dh = _compute_thresholds(t_stats, n, start_thres)
    if threshs is None:
        return tfnbs

    # 3. Pre-processing
    # Check symmetry to optimize edge extraction
    is_symm = _is_symmetric(t_stats)

    # Extract ALL potential edges > start_thres.
    # We do this ONCE to avoid repeated masking of the full NxN matrix.
    edge_rows, edge_cols, edge_weights = _get_edges(
        t_stats, start_thres, symmetric=is_symm
    )

    if len(edge_rows) == 0:
        return tfnbs

    # Pre-compute Block Metadata for these specific edges
    # This gives us the Block ID for every edge in our list and the Capacity of blocks
    edge_block_ids, sqrt_capacities = _precompute_block_metadata(
        edge_rows, edge_cols, normalized_net_labels, edge_mask
    )

    # Prepare broadcasting parameters
    if scalar_mode:
        e_bc = e_arr[0]
        h_bc = h_arr[0]
    else:
        e_bc = e_arr
        h_bc = h_arr

    # 4. Main Integration Loop (NO topological clustering - edge-level weighting)
    for threshold in threshs:
        # Determine which of our pre-extracted edges are active at this level
        active_indices_mask = edge_weights >= threshold

        if not np.any(active_indices_mask):
            continue

        # Get active edge indices
        active_rows = edge_rows[active_indices_mask]
        active_cols = edge_cols[active_indices_mask]

        # Compute edge-level weights based on block density
        edge_supports = _compute_edge_block_weights(
            active_indices_mask,
            edge_block_ids,
            sqrt_capacities,
            normalization=normalization,
        )

        # Calculate score increment: h^H * support^E * dh
        if scalar_mode:
            increment = (threshold**h_bc) * _extent_power(edge_supports, e_bc) * dh
            tfnbs[active_rows, active_cols] += increment
            if is_symm:
                tfnbs[active_cols, active_rows] += increment
        else:
            # Handle parameter sweep (broadcast dimensions)
            increment = (
                (threshold**h_bc)
                * _extent_power(edge_supports[..., np.newaxis], e_bc)
                * dh
            )
            tfnbs[active_rows, active_cols, :] += increment
            if is_symm:
                tfnbs[active_cols, active_rows, :] += increment

    return tfnbs
