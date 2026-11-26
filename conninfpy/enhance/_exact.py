"""Exact integration of the existing piecewise TFNBS support functions.

The graph sweep uses a disjoint set with lazy component integrals. Block
scores use a separate sweep because block support is not graph connectivity.
"""
from __future__ import annotations

import numpy as np

from conninfpy.enhance._backend import (
    HAS_NUMBA,
    _record_executed_backend,
    njit,
)


@njit(cache=True)
def _find(parent, offset, node):
    root = node
    total = 0.0
    while parent[root] != root:
        total += offset[root]
        root = parent[root]
    remaining = total
    while parent[node] != node:
        next_node = parent[node]
        old = offset[node]
        parent[node] = root
        offset[node] = remaining
        remaining -= old
        node = next_node
    return root, total


def _find_py(parent, offset, node):
    root = node
    total = 0.0
    while parent[root] != root:
        total += offset[root]
        root = parent[root]
    remaining = total
    while parent[node] != node:
        next_node = parent[node]
        old = offset[node]
        parent[node] = root
        offset[node] = remaining
        remaining -= old
        node = next_node
    return root, total


@njit(cache=True)
def _graph_integral(rows, cols, weights, n_nodes, exponent, height, start):
    order = np.argsort(-weights)
    parent = np.arange(n_nodes)
    offset = np.zeros(n_nodes)
    accum = np.zeros(n_nodes)
    edges = np.zeros(n_nodes)
    nodes = np.ones(n_nodes)
    since = np.zeros(n_nodes)
    base = np.zeros(weights.size)
    power = height + 1.0
    for index in order:
        level = weights[index]
        phi = level ** power / power
        a, off_a = _find(parent, offset, rows[index])
        b, off_b = _find(parent, offset, cols[index])
        if edges[a] > 0:
            accum[a] += edges[a] ** exponent * (since[a] ** power / power - phi)
        if b != a and edges[b] > 0:
            accum[b] += edges[b] ** exponent * (since[b] ** power / power - phi)
        base[index] = off_a + accum[a]
        if a != b:
            if nodes[a] < nodes[b]:
                a, b = b, a
            parent[b] = a
            offset[b] = accum[b] - accum[a]
            nodes[a] += nodes[b]
            edges[a] += edges[b]
        edges[a] += 1
        since[a] = level
    phi_start = start ** power / power
    for root in range(n_nodes):
        if parent[root] == root and edges[root] > 0:
            accum[root] += edges[root] ** exponent * (since[root] ** power / power - phi_start)
    result = np.zeros(weights.size)
    for index in range(weights.size):
        root, off = _find(parent, offset, rows[index])
        result[index] = off + accum[root] - base[index]
    return result


def _graph_integral_py(rows, cols, weights, n_nodes, exponent, height, start):
    """Pure-Python fallback using native lists (no Numba FFI overhead)."""
    order = np.argsort(-weights).tolist()
    rows_l = rows.tolist()
    cols_l = cols.tolist()
    weights_l = weights.astype(np.float64, copy=False).tolist()
    parent = list(range(n_nodes))
    offset = [0.0] * n_nodes
    accum = [0.0] * n_nodes
    edges = [0] * n_nodes
    nodes = [1] * n_nodes
    since = [0.0] * n_nodes
    n_edges = len(weights_l)
    base = [0.0] * n_edges
    power = float(height) + 1.0
    start_f = float(start)
    exp_f = float(exponent)
    for index in order:
        level = weights_l[index]
        phi = (level ** power) / power
        a, off_a = _find_py(parent, offset, rows_l[index])
        b, off_b = _find_py(parent, offset, cols_l[index])
        if edges[a] > 0:
            accum[a] += (edges[a] ** exp_f) * ((since[a] ** power) / power - phi)
        if b != a and edges[b] > 0:
            accum[b] += (edges[b] ** exp_f) * ((since[b] ** power) / power - phi)
        base[index] = off_a + accum[a]
        if a != b:
            if nodes[a] < nodes[b]:
                a, b = b, a
            parent[b] = a
            offset[b] = accum[b] - accum[a]
            nodes[a] += nodes[b]
            edges[a] += edges[b]
        edges[a] += 1
        since[a] = level
    phi_start = (start_f ** power) / power
    for root in range(n_nodes):
        if parent[root] == root and edges[root] > 0:
            accum[root] += (edges[root] ** exp_f) * ((since[root] ** power) / power - phi_start)
    result = [0.0] * n_edges
    for index in range(n_edges):
        root, off = _find_py(parent, offset, rows_l[index])
        result[index] = off + accum[root] - base[index]
    return np.asarray(result, dtype=np.float64)


@njit(cache=True)
def _block_integral(blocks, weights, exponent, height, start, scale, minimum):
    # Sorting first by block and then height keeps tied edges in the same epoch.
    by_weight = np.argsort(weights, kind="mergesort")
    order = by_weight[np.argsort(blocks[by_weight], kind="mergesort")]
    scores = np.zeros(weights.size)
    power = height + 1.0
    i = 0
    while i < weights.size:
        block = blocks[order[i]]
        end = i
        while end < weights.size and blocks[order[end]] == block:
            end += 1
        remaining = end - i
        previous = start
        integral = 0.0
        j = i
        while j < end:
            value = weights[order[j]]
            k = j
            while k < end and weights[order[k]] == value:
                k += 1
            support = remaining * scale[block] if remaining >= minimum else 0.0
            if support > 0:
                integral += support ** exponent * (value ** power - previous ** power) / power
            for q in range(j, k):
                scores[order[q]] = integral
            remaining -= k - j
            previous = value
            j = k
        i = end
    return scores


def _block_integral_numpy(blocks, weights, exponent, height, start, scale, minimum):
    """Vectorized NumPy exact block integral fallback when Numba is unavailable."""
    m = weights.size
    by_weight = np.argsort(weights, kind="mergesort")
    order = by_weight[np.argsort(blocks[by_weight], kind="mergesort")]
    b_s = blocks[order]
    w_s = weights[order].astype(np.float64, copy=False)

    is_block_start = np.empty(m, dtype=bool)
    is_block_start[0] = True
    is_block_start[1:] = b_s[1:] != b_s[:-1]

    is_epoch_start = np.empty(m, dtype=bool)
    is_epoch_start[0] = True
    is_epoch_start[1:] = is_block_start[1:] | (w_s[1:] != w_s[:-1])

    epoch_starts = np.flatnonzero(is_epoch_start)
    block_starts_idx = np.flatnonzero(is_block_start)
    block_ends_idx = np.empty(block_starts_idx.size, dtype=np.intp)
    block_ends_idx[:-1] = block_starts_idx[1:]
    block_ends_idx[-1] = m

    epoch_is_first_in_block = is_block_start[epoch_starts]
    epoch_block_idx = np.cumsum(epoch_is_first_in_block) - 1
    epoch_block_end = block_ends_idx[epoch_block_idx]

    remaining = (epoch_block_end - epoch_starts).astype(np.float64)
    epoch_block_id = b_s[epoch_starts]
    previous = np.where(
        epoch_is_first_in_block, float(start), w_s[np.maximum(epoch_starts - 1, 0)]
    )
    value = w_s[epoch_starts]

    support = np.where(remaining >= minimum, remaining * scale[epoch_block_id], 0.0)
    power = float(height) + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(
            support > 0.0,
            (support ** float(exponent)) * (value ** power - previous ** power) / power,
            0.0,
        )

    cs = np.cumsum(delta)
    first_epoch_of_block = np.flatnonzero(epoch_is_first_in_block)
    block_offset = np.zeros(first_epoch_of_block.size, dtype=np.float64)
    if first_epoch_of_block.size > 1:
        block_offset[1:] = cs[first_epoch_of_block[1:] - 1]
    epoch_integral = cs - block_offset[epoch_block_idx]

    edge_epoch_id = np.cumsum(is_epoch_start) - 1
    scores = np.empty(m, dtype=np.float64)
    scores[order] = epoch_integral[edge_epoch_id]
    return scores


def _integrate_edge_scores(
    rows,
    cols,
    weights,
    n_nodes,
    e,
    h,
    start,
    *,
    blocks=None,
    scale=None,
    minimum=1,
    use_numba=True,
    method_label="TFNBS (exact)",
    emit_fallback_warning=True,
    fallback_reason=None,
):
    if not np.isfinite(start) or start < 0:
        raise ValueError(f"start_thres must be finite and >= 0; got {start}.")

    effective_numba = bool(use_numba and HAS_NUMBA)
    if blocks is None:
        actual_backend = "numba" if effective_numba else "python"
    else:
        actual_backend = "numba" if effective_numba else "numpy"

    if fallback_reason is None and not effective_numba:
        fallback_reason = "no_numba" if not HAS_NUMBA else "disabled"

    _record_executed_backend(
        actual_backend,
        method_label=method_label,
        emit_fallback_warning=emit_fallback_warning,
        fallback_reason=fallback_reason,
    )

    n_edges = len(weights)
    n_params = len(e)
    if not n_edges:
        return np.zeros((0, n_params), dtype=np.float64)

    all_scores = np.empty((n_edges, n_params), dtype=np.float64)
    if blocks is None:
        fn = _graph_integral if effective_numba else _graph_integral_py
        for p, (exponent, height) in enumerate(zip(e, h, strict=True)):
            all_scores[:, p] = fn(
                rows, cols, weights, n_nodes, float(exponent), float(height), start
            )
    else:
        fn = _block_integral if effective_numba else _block_integral_numpy
        for p, (exponent, height) in enumerate(zip(e, h, strict=True)):
            all_scores[:, p] = fn(
                blocks, weights, float(exponent), float(height), start, scale, minimum
            )
    return all_scores


def integrate(
    rows,
    cols,
    weights,
    n_nodes,
    e,
    h,
    scalar,
    symmetric,
    start,
    blocks=None,
    scale=None,
    minimum=1,
    use_numba=True,
    *,
    method_label="TFNBS (exact)",
    emit_fallback_warning=True,
    fallback_reason=None,
):
    """Return scores for scalar or paired (E,H) parameters, preserving direction."""
    all_scores = _integrate_edge_scores(
        rows,
        cols,
        weights,
        n_nodes,
        e,
        h,
        start,
        blocks=blocks,
        scale=scale,
        minimum=minimum,
        use_numba=use_numba,
        method_label=method_label,
        emit_fallback_warning=emit_fallback_warning,
        fallback_reason=fallback_reason,
    )
    out = np.zeros((n_nodes, n_nodes) if scalar else (n_nodes, n_nodes, len(e)))
    if not len(weights):
        return out
    for p in range(len(e)):
        scores = all_scores[:, p]
        target = out if scalar else out[:, :, p]
        target[rows, cols] = scores
        if symmetric:
            target[cols, rows] = scores
    return out


def integrate_max(
    rows,
    cols,
    weights,
    n_nodes,
    e,
    h,
    scalar,
    start,
    blocks=None,
    scale=None,
    minimum=1,
    use_numba=True,
    *,
    method_label="TFNBS (exact)",
    emit_fallback_warning=True,
    fallback_reason=None,
):
    """Return max enhanced score over active edges without allocating (N, N)."""
    all_scores = _integrate_edge_scores(
        rows,
        cols,
        weights,
        n_nodes,
        e,
        h,
        start,
        blocks=blocks,
        scale=scale,
        minimum=minimum,
        use_numba=use_numba,
        method_label=method_label,
        emit_fallback_warning=emit_fallback_warning,
        fallback_reason=fallback_reason,
    )
    if all_scores.shape[0] == 0:
        return np.float64(0.0) if scalar else np.zeros(len(e), dtype=np.float64)
    maxes = all_scores.max(axis=0).astype(np.float64)
    return maxes[0] if scalar else maxes
