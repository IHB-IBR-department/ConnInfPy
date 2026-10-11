"""
Network enhancement: the ``apply_*`` wrappers plus the raw scoring functions.

The ``apply_*`` wrappers are shared by the t-test and GLM pipelines. Each wrapper is a pure transformation: ``stat_dict → score_dict``. It takes a
per-direction statistic dict (``{'positive': ..., 'negative': ...}``) and
applies the corresponding network-based enhancement to each direction
independently.

Wrappers accept the legacy keys ``'g2>g1'`` / ``'g1>g2'`` from the v1.x
t-test pipeline (silently remapped via :func:`conninfpy._compat.normalize_keys`)
and always return a :class:`~conninfpy._compat.TailResult` with canonical
``'positive'`` / ``'negative'`` keys. The legacy keys remain readable on
the result with a :class:`DeprecationWarning` until v2.1.

Wrappers do NOT compute statistics internally — the caller is responsible for
providing the raw statistic (t-stat, β, etc.). This matches the design
pattern documented in developer guide.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from conninfpy._compat import TailResult, make_tail_result, normalize_keys
from conninfpy._shared.arrays import create_prior_weights
from conninfpy.defaults import (
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_NBS_STAT,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_START_THRESHOLD,
)
from conninfpy.defaults import DEFAULT_N_THRESHOLDS_PERMUTATION as DEFAULT_N_THRESHOLDS
from conninfpy.enhance.nbs import (
    cnbs_block_means,
    cnbs_blocks,
    get_cnbs_score,
    get_nbs_score,
    nbs_bct,
)
from conninfpy.enhance.tfnbs import (
    get_fbc_tfnbs_score,
    get_network_informed_tfnbs_score,
    get_tfnbs_score,
)

__all__ = [
    "apply_tfnbs",
    "apply_nbs",
    "apply_cnbs",
    "apply_ni_tfnbs",
    "apply_fbc_tfnbs",
    "apply_edge_mask",
    # raw scoring (no permutation)
    "get_tfnbs_score",
    "get_network_informed_tfnbs_score",
    "get_fbc_tfnbs_score",
    "create_prior_weights",
    # classical BCT-style NBS reference (own permutation loop)
    "nbs_bct",
]


def _wrap(stat_dict: Dict[str, npt.NDArray[np.float64]], score_fn) -> TailResult:
    """Apply ``score_fn`` to each tail and wrap the result in a TailResult.

    Accepts legacy keys via :func:`normalize_keys`; always emits canonical keys.
    Falls back to a plain dict if the input has unexpected keys (e.g. F-stat
    omnibus path with ``{'omnibus': ...}``).
    """
    norm = normalize_keys(stat_dict)
    if set(norm.keys()) == {"positive", "negative"}:
        return make_tail_result(
            score_fn(norm["positive"]),
            score_fn(norm["negative"]),
        )
    return {key: score_fn(arr) for key, arr in norm.items()}


def apply_tfnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    **kwargs,
) -> TailResult:
    """Apply TFNBS (threshold-free cluster enhancement) to each direction."""
    return _wrap(
        stat_dict,
        lambda arr: get_tfnbs_score(arr, e, h, n, start_thres=start_thres),
    )


def apply_nbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    threshold: float = DEFAULT_NBS_THRESHOLD,
    nbs_stat: str = DEFAULT_NBS_STAT,
    **kwargs,
) -> TailResult:
    """Apply classical NBS (fixed threshold) to each direction."""
    return _wrap(
        stat_dict,
        lambda arr: get_nbs_score(arr, threshold=threshold, stat_type=nbs_stat),
    )


def apply_cnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    net_labels: npt.NDArray[np.int_],
    edge_mask: Optional[npt.NDArray[np.bool_]] = None,
    **kwargs,
) -> TailResult:
    """Constrained NBS (Noble & Scheinost, 2020): one statistic per block.

    The signed edge statistic (``positive - negative``) is averaged over every
    edge of each within- and between-network block. Returns block vectors of
    shape (K,): the mean for the positive tail and minus the mean for the
    negative tail (or the mean F for an omnibus map). Inference is per block
    (see :mod:`conninfpy._cnbs`), not by the maximum over edges.
    """
    norm = normalize_keys(stat_dict)
    first = next(iter(norm.values()))
    blocks = cnbs_blocks(net_labels, first.shape[0])
    if set(norm.keys()) == {"positive", "negative"}:
        means = cnbs_block_means(norm["positive"] - norm["negative"], blocks, edge_mask)
        return make_tail_result(means, -means)
    return {key: cnbs_block_means(arr, blocks, edge_mask) for key, arr in norm.items()}


def apply_ni_tfnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    normalization: str = "sqrt",
    edge_mask: Optional[npt.NDArray[np.bool_]] = None,
    **kwargs,
) -> TailResult:
    """Apply network-informed TFNBS (block-density weighted) to each direction."""
    return _wrap(
        stat_dict,
        lambda arr: get_network_informed_tfnbs_score(
            arr,
            net_labels,
            e,
            h,
            n,
            start_thres=start_thres,
            normalization=normalization,
            edge_mask=edge_mask,
        ),
    )


def apply_fbc_tfnbs(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    net_labels: npt.NDArray[np.int_],
    e: Union[float, List[float]] = DEFAULT_EXTENT_EXPONENT,
    h: Union[float, List[float]] = DEFAULT_HEIGHT_EXPONENT,
    n: int = DEFAULT_N_THRESHOLDS,
    start_thres: float = DEFAULT_START_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    **kwargs,
) -> TailResult:
    """Apply functional-block-clustering TFNBS to each direction."""
    return _wrap(
        stat_dict,
        lambda arr: get_fbc_tfnbs_score(
            arr,
            net_labels,
            e,
            h,
            n,
            start_thres=start_thres,
            min_cluster_size=min_cluster_size,
        ),
    )


def apply_edge_mask(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    edge_mask: npt.NDArray[np.bool_],
    inner=None,
    **kwargs,
) -> TailResult:
    """Restrict an enhancement to the edges of ``edge_mask`` (the tested family).

    Statistics outside the mask are set to 0 before ``inner`` runs, so those
    edges cannot join a TFNBS / NBS cluster or count toward an NI-TFNBS block
    density; their score is 0 and never sets a permutation maximum, which
    makes the max-statistic FWER apply to the masked family. ``edge_mask`` is
    also passed to ``inner`` (NI-TFNBS counts block capacity over it). With
    ``inner=None`` the raw statistic is masked (``method='tstat'``).
    Module-level so it pickles to worker processes.
    """
    mask = np.asarray(edge_mask, dtype=bool)
    masked = _wrap(stat_dict, lambda arr: np.where(mask, arr, 0.0))
    return masked if inner is None else inner(masked, edge_mask=mask, **kwargs)
