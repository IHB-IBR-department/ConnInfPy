"""Fixed network-pair assignments and eligible block capacities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


def _validate_net_labels(
    net_labels: npt.NDArray[np.integer], n_nodes: int
) -> npt.NDArray[np.integer]:
    """
    Validate and normalize network labels.

    Parameters
    ----------
    net_labels : ndarray of shape (N,)
        Network assignments.
    n_nodes : int
        Expected number of nodes.

    Returns
    -------
    normalized_labels : ndarray
        Labels mapped to 0..K-1 range.
    """
    if net_labels.shape[0] != n_nodes:
        raise ValueError(
            f"net_labels shape {net_labels.shape} does not match "
            f"number of nodes {n_nodes}."
        )

    # Map to continuous range 0..K-1 for efficient bincount indexing
    _, inverse = np.unique(net_labels, return_inverse=True)
    return inverse


def _precompute_block_metadata(
    edge_rows: npt.NDArray[np.intp],
    edge_cols: npt.NDArray[np.intp],
    node_labels: npt.NDArray[np.integer],
    edge_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """
    Pre-compute canonical block assignments and capacity factors.

    Block IDs are canonical for undirected network pairs:
    ``min(label_i, label_j) * K + max(label_i, label_j)``. This avoids
    ROI-order-dependent density weights and avoids re-calculating block IDs
    and capacities inside the threshold loop.

    Parameters
    ----------
    edge_rows : ndarray
        Row indices of all potential suprathreshold edges.
    edge_cols : ndarray
        Column indices of all potential suprathreshold edges.
    node_labels : ndarray
        Normalized network labels (0..K-1) for each node.
    edge_mask : ndarray, optional
        Fixed eligible undirected opportunities used to count capacity.

    Returns
    -------
    edge_block_ids : ndarray
        Canonical block ID assigned to each edge in `edge_rows`.
    sqrt_capacities : ndarray
        Lookup table for sqrt(M_total) indexed by Block ID.
    """
    n_nets = np.max(node_labels) + 1

    labels_i = node_labels[edge_rows]
    labels_j = node_labels[edge_cols]

    # Canonical block ID for undirected network pairs. This keeps block-density
    # weights invariant to ROI ordering within the upper triangle.
    min_labels = np.minimum(labels_i, labels_j)
    max_labels = np.maximum(labels_i, labels_j)
    edge_block_ids = min_labels * n_nets + max_labels

    # Total number of possible blocks
    n_blocks_total = n_nets * n_nets

    # Capacity counts fixed eligible opportunities, independently of the
    # observed/permuted statistics and the integration thresholds.
    rows, cols = np.triu_indices(len(node_labels), k=1)
    if edge_mask is not None:
        eligible = edge_mask[rows, cols]
        rows, cols = rows[eligible], cols[eligible]
    potential_blocks = np.minimum(
        node_labels[rows], node_labels[cols]
    ) * n_nets + np.maximum(node_labels[rows], node_labels[cols])
    block_capacities = np.bincount(potential_blocks, minlength=n_blocks_total).astype(
        np.float64
    )

    # Pre-calculate sqrt(Capacity) for the weighting formula: W = k / sqrt(M)
    # Avoid division by zero
    sqrt_capacities = np.sqrt(block_capacities)
    sqrt_capacities[sqrt_capacities == 0] = 1.0

    return edge_block_ids, sqrt_capacities


def _compute_edge_block_weights(
    active_edge_indices: npt.NDArray[np.bool_],
    edge_block_ids: npt.NDArray[np.intp],
    sqrt_capacities: npt.NDArray[np.floating],
    normalization: str = "sqrt",
) -> npt.NDArray[np.float64]:
    """
    Compute edge-level weights based on functional block density.

    Each edge's weight depends only on its functional block density,
    with no topological clustering (Edge-Level Weighting approach).

    Parameters
    ----------
    active_edge_indices : ndarray
        Boolean mask indicating which pre-extracted edges are active.
    edge_block_ids : ndarray
        Block ID for every pre-extracted edge.
    sqrt_capacities : ndarray
        Sqrt of total edges per block (precomputed).
    normalization : {'sqrt', 'linear', 'none'}, default='sqrt'
        Normalization variant for block density weighting:
        - ``'sqrt'``: W = k / sqrt(M_b) (default, current)
        - ``'linear'``: W = k / M_b (density normalization)
        - ``'none'``: W = k (raw active count, no normalization)

    Returns
    -------
    edge_weights : ndarray
        Weight for each active edge based on its block density.
    """
    # 1. Get block IDs for active edges
    active_block_ids = edge_block_ids[active_edge_indices]

    # 2. Count active edges (k) per block
    block_active_counts = np.bincount(active_block_ids, minlength=len(sqrt_capacities))

    # 3. Calculate weight per block based on normalization variant
    if normalization == "sqrt":
        # W = k / sqrt(M_b) — default
        block_weights = block_active_counts / sqrt_capacities
    elif normalization == "linear":
        # W = k / M_b = k / (sqrt(M_b))^2
        block_weights = block_active_counts / (sqrt_capacities**2)
    elif normalization == "none":
        # W = k — raw active count
        block_weights = block_active_counts.astype(np.float64)
    else:
        raise ValueError(
            f"Invalid normalization: '{normalization}'. "
            f"Must be one of: 'sqrt', 'linear', 'none'"
        )

    # 4. Return weight for each active edge
    return block_weights[active_block_ids]


def _compute_canonical_block_ids(
    edge_rows: npt.NDArray[np.intp],
    edge_cols: npt.NDArray[np.intp],
    node_labels: npt.NDArray[np.integer],
    n_networks: int,
) -> npt.NDArray[np.intp]:
    """
    Compute canonical block IDs for edges (handles undirected networks).

    For undirected networks, Block(A→B) == Block(B→A), so we use
    min(label_i, label_j) * n_networks + max(label_i, label_j).

    Parameters
    ----------
    edge_rows : ndarray
        Row indices of edges.
    edge_cols : ndarray
        Column indices of edges.
    node_labels : ndarray
        Network label for each node (0..K-1).
    n_networks : int
        Number of distinct networks.

    Returns
    -------
    block_ids : ndarray
        Canonical block ID for each edge.
    """
    labels_i = node_labels[edge_rows]
    labels_j = node_labels[edge_cols]

    # Canonical ordering for undirected: min * n_networks + max
    min_labels = np.minimum(labels_i, labels_j)
    max_labels = np.maximum(labels_i, labels_j)

    return min_labels * n_networks + max_labels
