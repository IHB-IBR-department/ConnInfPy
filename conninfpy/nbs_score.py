"""
Network-Based Statistics (NBS) scoring module.

Provides classical NBS cluster scoring (extent or intensity) and constrained
cNBS scoring in a single module, mirroring the role of tfnbs_score.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ._prepare import check_connectivity
from ._rng import RngLike, resolve_seed, warn_legacy_random_state


__all__ = [
    "DEFAULT_NBS_THRESHOLD",
    "DEFAULT_NBS_STAT",
    "get_nbs_score",
    "get_cnbs_score",
    "cnbs_blocks",
    "cnbs_block_means",
    "CNBSBlocks",
    "nbs_bct",
]


# =============================================================================
# Constants — imported from unified defaults
# =============================================================================

from .defaults import DEFAULT_NBS_STAT, DEFAULT_NBS_THRESHOLD, DEFAULT_N_PERMUTATIONS


# =============================================================================
# Helper utilities
# =============================================================================

ArrayF = npt.NDArray[np.float64]


def _is_symmetric(matrix: npt.NDArray[np.floating], rtol: float = 1e-10) -> bool:
    """True when the matrix equals its transpose."""
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=0)


def _get_edges(
    t_stats: npt.NDArray[np.floating],
    min_threshold: float,
    symmetric: bool,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating]]:
    """Upper-triangle edge indices of a square matrix."""
    if symmetric:
        rows, cols = np.triu_indices_from(t_stats, k=1)
    else:
        nroi = t_stats.shape[0]
        rows, cols = np.where(~np.eye(nroi, dtype=bool))

    weights = t_stats[rows, cols]
    valid_mask = weights > min_threshold
    return rows[valid_mask], cols[valid_mask], weights[valid_mask]


def _check_square(t_stat: npt.NDArray[np.floating]) -> None:
    """Raise ValueError unless ``arr`` is a square (N, N) matrix."""
    if t_stat.ndim != 2 or t_stat.shape[0] != t_stat.shape[1]:
        raise ValueError("t_stat must be a square (N, N) matrix.")
    if not np.allclose(np.diag(t_stat), 0.0):
        raise ValueError("Diagonal elements must be zero (no self-connections).")


# =============================================================================
# NBS scoring
# =============================================================================

def get_nbs_score(
    t_stats: npt.NDArray[np.floating],
    threshold: float = DEFAULT_NBS_THRESHOLD,
    stat_type: str = DEFAULT_NBS_STAT,
) -> npt.NDArray[np.floating]:
    """
    Compute NBS cluster statistics for a t-statistic matrix.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Non-negative t-statistic matrix (one tail, already separated).
    threshold : float
        T-statistic threshold for edge inclusion.
    stat_type : {'extent', 'intensity'}
        - 'extent': cluster size (# edges).
        - 'intensity': sum of t-values within the cluster.

    Returns
    -------
    ndarray of shape (N, N)
        Matrix where each suprathreshold edge is assigned its cluster statistic.
        Non-suprathreshold edges are 0.
    """
    if stat_type not in ("extent", "intensity"):
        raise ValueError("stat_type must be 'extent' or 'intensity'.")
    _check_square(t_stats)

    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)

    nroi = t_stats.shape[0]
    scores = np.zeros((nroi, nroi), dtype=np.float64)

    is_symm = _is_symmetric(t_stats)
    edge_rows, edge_cols, edge_weights = _get_edges(t_stats, threshold, symmetric=is_symm)
    if edge_rows.size == 0:
        return scores

    mask = np.zeros((nroi, nroi), dtype=bool)
    mask[edge_rows, edge_cols] = True
    if is_symm:
        mask[edge_cols, edge_rows] = True

    mask_for_cc = mask | mask.T
    sparse_mat = csr_matrix(mask_for_cc)
    n_components, node_labels = connected_components(sparse_mat, directed=False)

    edge_component_ids = node_labels[edge_rows]

    if stat_type == "extent":
        component_stats = np.bincount(edge_component_ids, minlength=n_components).astype(np.float64)
    else:
        component_stats = np.bincount(
            edge_component_ids,
            weights=edge_weights.astype(np.float64),
            minlength=n_components,
        )

    edge_stats = component_stats[edge_component_ids]
    scores[edge_rows, edge_cols] = edge_stats
    if is_symm:
        scores[edge_cols, edge_rows] = edge_stats

    return scores


# =============================================================================
# cNBS scoring (Noble & Scheinost, 2020)
# =============================================================================

@dataclass(frozen=True)
class CNBSBlocks:
    """The blocks of a network partition: every within- and between-network pair.

    Attributes
    ----------
    pairs : tuple of (label_a, label_b)
        Network labels of each block (``label_a <= label_b`` in sorted order);
        ``(a, a)`` is the within-network block of ``a``.
    edge_block : ndarray of shape (N, N)
        Block index of every edge; -1 on the diagonal.
    n_edges : ndarray of shape (K,)
        Undirected edges per block.
    """

    pairs: Tuple[Tuple[Any, Any], ...]
    edge_block: npt.NDArray[np.intp]
    n_edges: npt.NDArray[np.intp]

    @property
    def n_blocks(self) -> int:
        return len(self.pairs)


_BLOCK_CACHE: Dict[Tuple[bytes, str, int], CNBSBlocks] = {}


def cnbs_blocks(net_labels: npt.NDArray[Any], n_nodes: int) -> CNBSBlocks:
    """Blocks of ``net_labels``: one per pair of networks that has at least one edge."""
    labels = np.asarray(net_labels)
    if labels.ndim != 1 or labels.shape[0] != n_nodes:
        raise ValueError(f"net_labels must have shape ({n_nodes},); got {labels.shape}.")
    key = (labels.tobytes(), labels.dtype.str, n_nodes)
    if key in _BLOCK_CACHE:
        return _BLOCK_CACHE[key]
    values, codes = np.unique(labels, return_inverse=True)
    k = values.size
    lo = np.minimum(codes[:, None], codes[None, :])
    hi = np.maximum(codes[:, None], codes[None, :])
    canonical = lo * k + hi
    iu = np.triu_indices(n_nodes, k=1)
    present = np.unique(canonical[iu])            # a one-node network has no within block
    lookup = np.full(k * k, -1, dtype=np.intp)
    lookup[present] = np.arange(present.size)
    edge_block = lookup[canonical]
    np.fill_diagonal(edge_block, -1)
    as_py = lambda v: v.item() if hasattr(v, "item") else v  # noqa: E731
    blocks = CNBSBlocks(
        pairs=tuple((as_py(values[c // k]), as_py(values[c % k])) for c in present),
        edge_block=edge_block,
        n_edges=np.bincount(edge_block[iu], minlength=present.size),
    )
    if len(_BLOCK_CACHE) > 32:
        _BLOCK_CACHE.clear()
    _BLOCK_CACHE[key] = blocks
    return blocks


def cnbs_block_means(
    stat: npt.NDArray[np.float64],
    blocks: CNBSBlocks,
    edge_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.float64]:
    """Mean of ``stat`` over every edge of each block (upper triangle).

    ``edge_mask`` restricts the mean to the edges it marks (available-case
    analysis: untested edges are left out); a block with no edge left is NaN.
    """
    iu = np.triu_indices(stat.shape[0], k=1)
    values, ids = stat[iu], blocks.edge_block[iu]
    if edge_mask is not None:
        keep = np.asarray(edge_mask, dtype=bool)[iu]
        values, ids = values[keep], ids[keep]
    sums = np.bincount(ids, weights=values, minlength=blocks.n_blocks)
    counts = np.bincount(ids, minlength=blocks.n_blocks)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def get_cnbs_score(
    t_stat: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    """
    cNBS statistic (Noble & Scheinost, 2020) as an edge map.

    Every edge gets the mean **signed** statistic of its block (all edges of
    the within- or between-network pair, whatever their sign or size). The
    inference unit is the block: see :func:`conninfpy.compute_p_val` with
    ``method='cnbs'``, which tests each block against its own permutation
    null and corrects across blocks.

    .. versionchanged:: 2.0
        Previously the mean of the strictly positive values of one tail,
        i.e. not the published statistic.

    Parameters
    ----------
    t_stat : ndarray of shape (N, N)
        Signed, symmetric edge statistic (e.g. t).
    net_labels : ndarray of shape (N,)
        Network label of each node.

    Returns
    -------
    ndarray of shape (N, N)
        Block mean on every off-diagonal edge; 0 on the diagonal.
    """
    _check_square(t_stat)
    blocks = cnbs_blocks(net_labels, t_stat.shape[0])
    means = cnbs_block_means(np.asarray(t_stat, dtype=np.float64), blocks)
    out = np.where(blocks.edge_block >= 0, means[np.maximum(blocks.edge_block, 0)], 0.0)
    return out


# =============================================================================
# Reference NBS (comparison/testing only)
# =============================================================================

def _score_nbs_from_diffs(
    diffs: npt.NDArray[np.float64],
    threshold: float,
    stat_type: str,
) -> Dict[str, npt.NDArray[np.float64]]:
    """NBS component scores from per-edge statistics."""
    from ._compat import make_tail_result
    from .pairwise_stats import compute_t_stat_diff

    t_stat_dict = compute_t_stat_diff(diffs)
    return make_tail_result(
        get_nbs_score(t_stat_dict["positive"], threshold=threshold, stat_type=stat_type),
        get_nbs_score(t_stat_dict["negative"], threshold=threshold, stat_type=stat_type),
    )


def _score_nbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    threshold: float,
    stat_type: str,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Two-sample NBS component scores."""
    from ._compat import make_tail_result
    from .pairwise_stats import compute_t_stat

    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    return make_tail_result(
        get_nbs_score(t_stat_dict["positive"], threshold=threshold, stat_type=stat_type),
        get_nbs_score(t_stat_dict["negative"], threshold=threshold, stat_type=stat_type),
    )


def nbs_bct(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]] = None,
    threshold: float = DEFAULT_NBS_THRESHOLD,
    stat_type: str = DEFAULT_NBS_STAT,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    test_type: str = "paired",
    use_mp: bool = True,
    random_state: Optional[int] = None,  # deprecated alias for `rng`
    rng: RngLike = None,
    n_processes: Optional[int] = None,
    **kwargs,
) -> Tuple[
    Dict[str, npt.NDArray[np.float64]],
    Dict[str, npt.NDArray[np.uint8]],
    Dict[str, npt.NDArray[np.float64]],
]:
    """
    Compute classical NBS with permutation testing.

    Returns p-values, adjacency matrices (thresholded t-stats), and null
    maxima. P-values use tie-inclusive max-statistic counting and the
    Phipson-Smyth +1 correction, matching the main ``compute_p_val`` path.
    """
    from .pairwise_stats import compute_null_dist, compute_t_stat, compute_t_stat_diff

    if random_state is not None and rng is None:
        warn_legacy_random_state("random_state")
    random_state = resolve_seed(rng, legacy_random_state=random_state)

    group1 = check_connectivity(group1, "group1")
    if group2 is not None:
        group2 = check_connectivity(group2, "group2")

    if test_type in ("paired", "two-sample") and group2 is None:
        raise ValueError(f"group2 is required for test_type='{test_type}'.")

    if test_type == "paired":
        diffs = group2 - group1
        emp_t_dict = _score_nbs_from_diffs(diffs, threshold=threshold, stat_type=stat_type)
        t_func = _score_nbs_from_diffs

    elif test_type == "two-sample":
        emp_t_dict = _score_nbs_two_sample(group1, group2, test_type="two-sample", threshold=threshold, stat_type=stat_type)
        t_func = _score_nbs_two_sample

    elif test_type == "one-sample":
        if group2 is not None:
            raise ValueError("group2 must be None for one-sample tests.")
        if group1.ndim != 3:
            raise ValueError("Dimensions of group 1 data should be: (subjects, N, N).")
        emp_t_dict = _score_nbs_from_diffs(group1, threshold=threshold, stat_type=stat_type)
        t_func = _score_nbs_from_diffs

    else:
        raise ValueError("test_type must be one of {'paired', 'one-sample', 'two-sample'}.")

    # Build adjacency masks for reporting (raw thresholding on t-stats)
    adj_matrices: Dict[str, npt.NDArray[np.uint8]] = {}
    if test_type == "paired":
        raw_t = compute_t_stat_diff(group2 - group1)
    elif test_type == "two-sample":
        raw_t = compute_t_stat(group1, group2, test_type="two-sample", **kwargs)
    else:
        raw_t = compute_t_stat_diff(group1)

    for key in raw_t:
        adj = (raw_t[key] > threshold).astype(np.uint8)
        if adj.shape[-1] == adj.shape[-2]:
            adj = np.triu(adj, 1)
            adj = adj + adj.T
        adj_matrices[key] = adj

    # Null distribution for max cluster statistic
    group2_for_null = group2 if test_type != "one-sample" else None
    max_null_dict = compute_null_dist(
        group1,
        group2_for_null,
        t_func,
        n_permutations=n_permutations,
        test_type=test_type,
        use_mp=use_mp,
        rng=random_state,
        n_processes=n_processes,
        threshold=threshold,
        stat_type=stat_type,
    )

    # P-values via max-stat (cluster extent or intensity)
    keys = list(emp_t_dict.keys())
    p_values: Dict[str, npt.NDArray[np.float64]] = {}
    for key in keys:
        emp_t = emp_t_dict[key][..., np.newaxis]
        null_dist = max_null_dict[key]
        count = np.sum(emp_t <= null_dist, axis=-1)
        p_values[key] = (count + 1.0) / (null_dist.shape[0] + 1.0)

    return p_values, adj_matrices, max_null_dict
