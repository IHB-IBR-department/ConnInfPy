"""Exact integration of the existing piecewise TFNBS support functions.

The graph sweep uses a disjoint set with lazy component integrals. Block
scores use a separate sweep because block support is not graph connectivity.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn


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


def integrate(rows, cols, weights, n_nodes, e, h, scalar, symmetric, start,
              blocks=None, scale=None, minimum=1, use_numba=True):
    """Return scores for scalar or paired (E,H) parameters, preserving direction."""
    if not np.isfinite(start) or start < 0:
        raise ValueError(f"start_thres must be finite and >= 0; got {start}.")
    out = np.zeros((n_nodes, n_nodes) if scalar else (n_nodes, n_nodes, len(e)))
    if not len(weights):
        return out
    graph = _graph_integral if use_numba else getattr(_graph_integral, "py_func", _graph_integral)
    block = _block_integral if use_numba else getattr(_block_integral, "py_func", _block_integral)
    # The Python graph fallback must also use the Python _find when numba is absent;
    # a compiled _find is harmless when only the graph core is requested as Python.
    for p, (exponent, height) in enumerate(zip(e, h)):
        if blocks is None:
            scores = graph(rows, cols, weights, n_nodes, float(exponent), float(height), start)
        else:
            scores = block(blocks, weights, float(exponent), float(height), start, scale, minimum)
        target = out if scalar else out[:, :, p]
        target[rows, cols] = scores
        if symmetric:
            target[cols, rows] = scores
    return out
