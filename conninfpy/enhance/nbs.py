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

from conninfpy._prepare import check_connectivity
from conninfpy._shared.rng import RngLike, resolve_seed, warn_legacy_random_state
from conninfpy.enhance._backend import HAS_NUMBA, _record_executed_backend, njit
from conninfpy.enhance._graph import _get_edges, _is_symmetric

__all__ = [
    "DEFAULT_NBS_THRESHOLD",
    "DEFAULT_NBS_STAT",
    "get_nbs_score",
    "nbs_components",
    "NBSComponents",
    "get_cnbs_score",
    "cnbs_blocks",
    "cnbs_block_means",
    "CNBSBlocks",
    "nbs_bct",
]


# =============================================================================
# Constants — imported from unified defaults
# =============================================================================

from conninfpy.defaults import (
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_NBS_STAT,
    DEFAULT_NBS_THRESHOLD,
)

# =============================================================================
# Helper utilities
# =============================================================================

ArrayF = npt.NDArray[np.float64]


def _check_square(t_stat: npt.NDArray[np.floating]) -> None:
    """Raise ValueError unless ``arr`` is a square (N, N) matrix."""
    if t_stat.ndim != 2 or t_stat.shape[0] != t_stat.shape[1]:
        raise ValueError("t_stat must be a square (N, N) matrix.")
    if not np.allclose(np.diag(t_stat), 0.0):
        raise ValueError("Diagonal elements must be zero (no self-connections).")


@njit(cache=True)
def _nbs_union_find_labels(rows, cols, n_nodes):
    """Weakly-connected component labels (0..n_components-1) from an edge list."""
    parent = np.arange(n_nodes, dtype=np.int32)
    rank = np.zeros(n_nodes, dtype=np.int32)

    for i in range(len(rows)):
        a = rows[i]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        b = cols[i]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            if rank[a] < rank[b]:
                parent[a] = b
            elif rank[a] > rank[b]:
                parent[b] = a
            else:
                parent[b] = a
                rank[a] += 1

    comp_map = np.full(n_nodes, -1, dtype=np.int32)
    labels = np.empty(n_nodes, dtype=np.int32)
    n_comp = 0
    for i in range(n_nodes):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        cid = comp_map[x]
        if cid < 0:
            cid = n_comp
            comp_map[x] = cid
            n_comp += 1
        labels[i] = cid
    return n_comp, labels


@njit(cache=True)
def _nbs_max_numba(rows, cols, weights, n_nodes, threshold, is_intensity):
    """Single-pass union-find returning max component extent or intensity."""
    parent = np.arange(n_nodes, dtype=np.int32)
    rank = np.zeros(n_nodes, dtype=np.int32)

    n_edges = len(rows)
    for i in range(n_edges):
        a = rows[i]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        b = cols[i]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            if rank[a] < rank[b]:
                parent[a] = b
            elif rank[a] > rank[b]:
                parent[b] = a
            else:
                parent[b] = a
                rank[a] += 1

    comp_stat = np.zeros(n_nodes, dtype=np.float64)
    max_stat = 0.0
    for i in range(n_edges):
        a = rows[i]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        val = (weights[i] - threshold) if is_intensity else 1.0
        new_val = comp_stat[a] + val
        comp_stat[a] = new_val
        if new_val > max_stat:
            max_stat = new_val
    return max_stat


def _connected_components_from_edges(
    edge_rows: npt.NDArray[np.intp],
    edge_cols: npt.NDArray[np.intp],
    nroi: int,
    *,
    emit_fallback_warning: bool = True,
) -> Tuple[int, npt.NDArray[np.intp]]:
    """Compute weakly connected node labels directly from sparse edge coordinates."""
    if HAS_NUMBA:
        _record_executed_backend(
            "numba",
            method_label="NBS",
            emit_fallback_warning=emit_fallback_warning,
        )
        n_comp, labels = _nbs_union_find_labels(
            edge_rows.astype(np.int64, copy=False),
            edge_cols.astype(np.int64, copy=False),
            nroi,
        )
        return int(n_comp), labels.astype(np.intp, copy=False)

    _record_executed_backend(
        "scipy",
        method_label="NBS",
        emit_fallback_warning=emit_fallback_warning,
        fallback_reason="no_numba",
    )
    ones = np.ones(edge_rows.size, dtype=np.int8)
    sparse_mat = csr_matrix((ones, (edge_rows, edge_cols)), shape=(nroi, nroi))
    n_components, node_labels = connected_components(sparse_mat, directed=False)
    return int(n_components), node_labels.astype(np.intp, copy=False)


def _nbs_max_from_edges(
    edge_rows: npt.NDArray[np.intp],
    edge_cols: npt.NDArray[np.intp],
    edge_weights: npt.NDArray[np.floating],
    nroi: int,
    threshold: float,
    stat_type: str,
    *,
    emit_fallback_warning: bool = True,
) -> np.float64:
    """Return max NBS component statistic directly from sparse suprathreshold edges."""
    if HAS_NUMBA:
        _record_executed_backend(
            "numba",
            method_label="NBS",
            emit_fallback_warning=emit_fallback_warning,
        )
        if edge_rows.size == 0:
            return np.float64(0.0)
        return np.float64(
            _nbs_max_numba(
                edge_rows.astype(np.int64, copy=False),
                edge_cols.astype(np.int64, copy=False),
                edge_weights.astype(np.float64, copy=False),
                nroi,
                float(threshold),
                stat_type == "intensity",
            )
        )

    _record_executed_backend(
        "scipy",
        method_label="NBS",
        emit_fallback_warning=emit_fallback_warning,
        fallback_reason="no_numba",
    )
    if edge_rows.size == 0:
        return np.float64(0.0)
    ones = np.ones(edge_rows.size, dtype=np.int8)
    sparse_mat = csr_matrix((ones, (edge_rows, edge_cols)), shape=(nroi, nroi))
    n_components, node_labels = connected_components(sparse_mat, directed=False)
    edge_component = node_labels[edge_rows]
    if stat_type == "extent":
        return np.float64(
            np.bincount(edge_component, minlength=n_components).max()
        )
    return np.float64(
        np.bincount(
            edge_component,
            weights=edge_weights.astype(np.float64, copy=False) - threshold,
            minlength=n_components,
        ).max()
    )


# =============================================================================
# NBS scoring
# =============================================================================


@dataclass(frozen=True)
class _Suprathreshold:
    """Suprathreshold edges of one statistic map and their connected components.

    ``rows``/``cols``/``weights`` list the edges (the upper triangle when the
    map is symmetric); ``edge_component`` gives the component of each edge
    (labels from sparse union-find or :func:`scipy.sparse.csgraph.connected_components`);
    ``node_labels`` gives the component of every node.
    """

    rows: npt.NDArray[np.intp]
    cols: npt.NDArray[np.intp]
    weights: npt.NDArray[np.floating]
    edge_component: npt.NDArray[np.intp]
    n_components: int
    node_labels: npt.NDArray[np.intp]
    symmetric: bool


def _label_suprathreshold(
    t_stats: npt.NDArray[np.floating], threshold: float
) -> Optional[_Suprathreshold]:
    """Edges with ``t > threshold`` and their connected components.

    The single place where NBS components are defined: :func:`get_nbs_score`
    (the statistic the permutation test uses) and :func:`nbs_components`
    (the component report) both read it, so the reported components are
    the components that were tested. Returns ``None`` when no edge exceeds
    the threshold.
    """
    # Round to avoid float precision issues at threshold boundaries
    t_stats = np.round(t_stats, decimals=10)
    nroi = t_stats.shape[0]
    is_symm = _is_symmetric(t_stats)
    edge_rows, edge_cols, edge_weights = _get_edges(
        t_stats, threshold, symmetric=is_symm
    )
    if edge_rows.size == 0:
        _record_executed_backend(
            "numba" if HAS_NUMBA else "scipy",
            method_label="NBS",
            fallback_reason=None if HAS_NUMBA else "no_numba",
        )
        return None

    n_components, node_labels = _connected_components_from_edges(
        edge_rows, edge_cols, nroi
    )
    return _Suprathreshold(
        rows=edge_rows,
        cols=edge_cols,
        weights=edge_weights,
        edge_component=node_labels[edge_rows],
        n_components=n_components,
        node_labels=node_labels,
        symmetric=is_symm,
    )


def _component_statistics(
    found: _Suprathreshold, threshold: float, stat_type: str
) -> npt.NDArray[np.float64]:
    """Extent or intensity of every connected component (isolated nodes: 0)."""
    if stat_type == "extent":
        return np.bincount(
            found.edge_component, minlength=found.n_components
        ).astype(np.float64)
    # Mass above the threshold (NBS v1.2 'Intensity').
    return np.bincount(
        found.edge_component,
        weights=found.weights.astype(np.float64) - threshold,
        minlength=found.n_components,
    )


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
        - 'intensity': sum over the cluster's edges of the statistic minus
          the threshold, ``sum(t - threshold)``, as in the MATLAB NBS
          toolbox (``NBSstats.m``). Only the excess above the threshold
          counts: an edge just above it adds almost nothing, so the score
          does not jump when an edge crosses the threshold, and a small
          cluster of strong edges can outrank a large cluster of weak ones.
          Summing ``t`` itself would add ``threshold * extent``, which makes
          intensity close to a rescaled extent.

    Returns
    -------
    ndarray of shape (N, N)
        Matrix where each suprathreshold edge is assigned its cluster statistic.
        Non-suprathreshold edges are 0.

    See Also
    --------
    nbs_components : the components themselves (ids, sizes, nodes).
    """
    if stat_type not in ("extent", "intensity"):
        raise ValueError("stat_type must be 'extent' or 'intensity'.")
    _check_square(t_stats)

    nroi = t_stats.shape[0]
    scores = np.zeros((nroi, nroi), dtype=np.float64)

    found = _label_suprathreshold(t_stats, threshold)
    if found is None:
        return scores

    component_stats = _component_statistics(found, threshold, stat_type)
    edge_stats = component_stats[found.edge_component]
    scores[found.rows, found.cols] = edge_stats
    if found.symmetric:
        scores[found.cols, found.rows] = edge_stats

    return scores


@dataclass(frozen=True)
class NBSComponents:
    """Connected components of the suprathreshold edges of one tail.

    Components are numbered ``1..K`` by decreasing statistic (ties: more
    edges first, then the lower first ROI index), so a component keeps its
    number whatever else is reported.

    Fields: ``labels``, an (N, N) int32 array with the component number of
    every edge (both triangles; 0 for an edge not above the threshold and on
    the diagonal); ``n_edges``, the undirected edges of each component (its
    extent); ``nodes``, the sorted ROI indices of each component (a single
    suprathreshold edge is a component of two nodes and one edge);
    ``statistic``, the extent or the intensity ``sum(t - threshold)`` of
    each component; ``stat_type`` (``'extent'`` or ``'intensity'``) and
    ``threshold``.
    """

    labels: npt.NDArray[np.int32]
    n_edges: npt.NDArray[np.intp]
    nodes: Tuple[npt.NDArray[np.intp], ...]
    statistic: npt.NDArray[np.float64]
    stat_type: str
    threshold: float

    @property
    def n_components(self) -> int:
        return int(self.n_edges.size)

    @property
    def n_nodes(self) -> npt.NDArray[np.intp]:
        return np.array([n.size for n in self.nodes], dtype=np.intp)


def nbs_components(
    t_stats: npt.NDArray[np.floating],
    threshold: float = DEFAULT_NBS_THRESHOLD,
    stat_type: str = DEFAULT_NBS_STAT,
) -> NBSComponents:
    """Connected components of the edges above ``threshold`` in one tail.

    The components :func:`get_nbs_score` scores, with their numbers, sizes,
    nodes and statistics; the permutation test of ``method='nbs'`` assigns
    to every edge the p-value of its component's statistic. Used for the
    component report of :class:`~conninfpy.InferenceResult`.

    Parameters
    ----------
    t_stats : ndarray of shape (N, N)
        Non-negative, symmetric statistic map of one tail with a zero
        diagonal (the same input as :func:`get_nbs_score`).
    threshold : float
        Edge threshold: an edge belongs to a component iff ``t > threshold``.
    stat_type : {'extent', 'intensity'}
        Component statistic, as in :func:`get_nbs_score`.
    """
    if stat_type not in ("extent", "intensity"):
        raise ValueError("stat_type must be 'extent' or 'intensity'.")
    _check_square(t_stats)
    nroi = t_stats.shape[0]
    found = _label_suprathreshold(t_stats, threshold)
    if found is None:
        return NBSComponents(
            labels=np.zeros((nroi, nroi), dtype=np.int32),
            n_edges=np.zeros(0, dtype=np.intp),
            nodes=(),
            statistic=np.zeros(0, dtype=np.float64),
            stat_type=stat_type,
            threshold=float(threshold),
        )
    if not found.symmetric:
        raise ValueError("t_stats must be symmetric to report components.")

    raw_stat = _component_statistics(found, threshold, stat_type)
    raw_ids, n_edges = np.unique(found.edge_component, return_counts=True)
    statistic = raw_stat[raw_ids]
    node_lists = [np.flatnonzero(found.node_labels == c) for c in raw_ids]
    first_node = np.array([n[0] for n in node_lists])
    # lexsort: last key is primary
    order = np.lexsort((first_node, -n_edges, -statistic))
    rank = np.empty(raw_ids.size, dtype=np.int32)
    rank[order] = np.arange(1, raw_ids.size + 1, dtype=np.int32)

    lookup = np.zeros(found.n_components, dtype=np.int32)
    lookup[raw_ids] = rank
    edge_rank = lookup[found.edge_component]
    labels = np.zeros((nroi, nroi), dtype=np.int32)
    labels[found.rows, found.cols] = edge_rank
    labels[found.cols, found.rows] = edge_rank

    return NBSComponents(
        labels=labels,
        n_edges=n_edges[order].astype(np.intp),
        nodes=tuple(node_lists[i] for i in order),
        statistic=statistic[order],
        stat_type=stat_type,
        threshold=float(threshold),
    )


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
        raise ValueError(
            f"net_labels must have shape ({n_nodes},); got {labels.shape}."
        )
    key = (labels.tobytes(), labels.dtype.str, n_nodes)
    if key in _BLOCK_CACHE:
        return _BLOCK_CACHE[key]
    values, codes = np.unique(labels, return_inverse=True)
    k = values.size
    lo = np.minimum(codes[:, None], codes[None, :])
    hi = np.maximum(codes[:, None], codes[None, :])
    canonical = lo * k + hi
    iu = np.triu_indices(n_nodes, k=1)
    present = np.unique(canonical[iu])  # a one-node network has no within block
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
    from conninfpy._compat import make_tail_result
    from conninfpy.stats.pairwise import compute_t_stat_diff

    t_stat_dict = compute_t_stat_diff(diffs)
    return make_tail_result(
        get_nbs_score(
            t_stat_dict["positive"], threshold=threshold, stat_type=stat_type
        ),
        get_nbs_score(
            t_stat_dict["negative"], threshold=threshold, stat_type=stat_type
        ),
    )


def _score_nbs_two_sample(
    group1: npt.NDArray[np.float64],
    group2: npt.NDArray[np.float64],
    test_type: str,
    threshold: float,
    stat_type: str,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Two-sample NBS component scores."""
    from conninfpy._compat import make_tail_result
    from conninfpy.stats.pairwise import compute_t_stat

    t_stat_dict = compute_t_stat(group1, group2, test_type=test_type)
    return make_tail_result(
        get_nbs_score(
            t_stat_dict["positive"], threshold=threshold, stat_type=stat_type
        ),
        get_nbs_score(
            t_stat_dict["negative"], threshold=threshold, stat_type=stat_type
        ),
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
    from conninfpy.stats.pairwise import (
        compute_null_dist,
        compute_t_stat,
        compute_t_stat_diff,
    )

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
        emp_t_dict = _score_nbs_from_diffs(
            diffs, threshold=threshold, stat_type=stat_type
        )
        t_func = _score_nbs_from_diffs

    elif test_type == "two-sample":
        emp_t_dict = _score_nbs_two_sample(
            group1,
            group2,
            test_type="two-sample",
            threshold=threshold,
            stat_type=stat_type,
        )
        t_func = _score_nbs_two_sample

    elif test_type == "one-sample":
        if group2 is not None:
            raise ValueError("group2 must be None for one-sample tests.")
        if group1.ndim != 3:
            raise ValueError("Dimensions of group 1 data should be: (subjects, N, N).")
        emp_t_dict = _score_nbs_from_diffs(
            group1, threshold=threshold, stat_type=stat_type
        )
        t_func = _score_nbs_from_diffs

    else:
        raise ValueError(
            "test_type must be one of {'paired', 'one-sample', 'two-sample'}."
        )

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
