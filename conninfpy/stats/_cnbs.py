"""Block-level inference for the constrained NBS (Noble & Scheinost, 2020).

cNBS averages the signed edge statistic over every edge of each predefined
block (each within-network and each between-network pair of an atlas
partition). Each block is calibrated against its own permutation null, the
mean of the same block in every permuted map, and the block p-values are
then corrected across blocks: Bonferroni (FWER, the default) or
Benjamini-Hochberg (FDR). This follows the authors' implementation
(NBS_benchmarking, ``get_constrained_stats.m`` and ``NBSstats_smn.m``,
'Constrained_FWER' / 'Constrained'), with the Phipson-Smyth (b+1)/(B+1)
count used everywhere in this package.

The inference unit is the block. Every edge of a significant block carries
the block's p-value, including edges whose own effect is zero or opposite:
a significant block says that the block's mean effect is nonzero, not that
each of its edges is.

The positive tail tests a positive block mean, the negative tail a negative
one; as for the other methods, each tail is its own family (use alpha/2 per
tail, or ``two_tailed=True``, for one two-sided family).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from conninfpy.enhance.nbs import cnbs_blocks
from conninfpy.stats._engine import _bh_fdr_correction

logger = logging.getLogger(__name__)

CORRECTIONS = ("bonferroni", "bh")


def check_correction(correction: str) -> str:
    if correction not in CORRECTIONS:
        raise ValueError(
            f"cnbs_correction must be one of {CORRECTIONS}; got {correction!r}."
        )
    return correction


def block_p_values(
    emp: Dict[str, npt.NDArray[np.float64]],
    null: Dict[str, npt.NDArray[np.float64]],
    net_labels: npt.NDArray[Any],
    n_nodes: int,
    *,
    correction: str = "bonferroni",
    edge_mask: Optional[npt.NDArray[np.bool_]] = None,
    acceleration: Optional[str] = None,
    two_tailed: bool = False,
) -> Tuple[Dict[str, npt.NDArray[np.float64]], Dict[str, Any]]:
    """Edge maps of corrected block p-values, and the block table.

    Parameters
    ----------
    emp : dict
        Observed block statistics per tail, each of shape (K,).
    null : dict
        Permutation block statistics per tail, each of shape (B, K).
    net_labels, n_nodes
        The partition that defined the blocks.
    correction : {'bonferroni', 'bh'}
        Across-block correction within each tail.
    edge_mask : ndarray of shape (N, N), optional
        Tested edges (available-case analysis); other edges get NaN.
    acceleration : str, optional
        Not used: each block's p-value is counted from its own null.
    two_tailed : bool, default=False
        Compare absolute block means with each block's absolute permutation
        null, correct once across blocks, and project to the observed sign.
    """
    check_correction(correction)
    if acceleration is not None:
        logger.warning(
            "acceleration=%r is not applied to cNBS: each block's p-value is "
            "counted from its own permutation null.",
            acceleration,
        )
    blocks = cnbs_blocks(net_labels, n_nodes)
    eb = blocks.edge_block
    off = eb >= 0
    p_edges: Dict[str, npt.NDArray[np.float64]] = {}
    table: Dict[str, Any] = {}
    two_sided_p = None
    if two_tailed and set(emp) == {"positive", "negative"}:
        observed = np.abs(np.asarray(emp["positive"]))
        joint = np.maximum(null["positive"], null["negative"])
        testable = np.isfinite(observed)
        if not np.isfinite(joint[:, testable]).all():
            raise RuntimeError(
                "Nonfinite block statistics in the two-sided permutation null."
            )
        p_unc = np.where(
            testable,
            (np.sum(joint >= observed, axis=0) + 1.0) / (joint.shape[0] + 1.0),
            np.nan,
        )
        p_corr = np.full_like(p_unc, np.nan)
        if testable.any():
            p_corr[testable] = (
                np.minimum(p_unc[testable] * testable.sum(), 1.0)
                if correction == "bonferroni"
                else _bh_fdr_correction(p_unc[testable])
            )
        two_sided_p = p_unc, p_corr
    for key, obs in emp.items():
        obs = np.asarray(obs, dtype=np.float64)
        nul = np.asarray(null[key], dtype=np.float64)
        B = nul.shape[0]
        testable = np.isfinite(obs)
        if not np.isfinite(nul[:, testable]).all():
            raise RuntimeError(f"NaN block statistics in the {key!r} permutation null.")
        count = np.sum(nul >= obs[None, :], axis=0)
        p_unc = np.where(testable, (count + 1.0) / (B + 1.0), np.nan)
        p_corr = np.full_like(p_unc, np.nan)
        m = int(testable.sum())
        if two_sided_p is not None:
            p_unc = np.where(testable, np.where(obs > 0, two_sided_p[0], 1.0), np.nan)
            p_corr = np.where(testable, np.where(obs > 0, two_sided_p[1], 1.0), np.nan)
        elif m:
            if correction == "bonferroni":
                p_corr[testable] = np.minimum(p_unc[testable] * m, 1.0)
            else:
                p_corr[testable] = _bh_fdr_correction(p_unc[testable])
        edge = np.ones((n_nodes, n_nodes))
        edge[off] = p_corr[eb[off]]
        if edge_mask is not None:
            edge[off & ~np.asarray(edge_mask, dtype=bool)] = np.nan
        p_edges[key] = edge
        table[key] = (obs, p_unc, p_corr)

    m = int(np.isfinite(next(iter(emp.values()))).sum())
    B = next(iter(null.values())).shape[0]
    if correction == "bonferroni" and m / (B + 1.0) > 0.05:
        logger.warning(
            "cNBS: with %d blocks and %d permutations the smallest Bonferroni-corrected "
            "p-value is %.3f, so no block can reach 0.05; use at least %d permutations.",
            m,
            B,
            m / (B + 1.0),
            int(np.ceil(m / 0.05)),
        )

    rows = []
    signed_key = "positive" if "positive" in table else next(iter(table))
    for b, (a, c) in enumerate(blocks.pairs):
        row = {
            "network_a": a,
            "network_b": c,
            "n_edges": int(blocks.n_edges[b]),
            "mean_stat": float(table[signed_key][0][b]),
        }
        for key, (_, p_unc, p_corr) in table.items():
            row[f"p_{key}"] = float(p_corr[b])
            row[f"p_{key}_uncorrected"] = float(p_unc[b])
        if two_sided_p is not None:
            row["p_two_sided_uncorrected"] = float(two_sided_p[0][b])
            row["p_two_sided"] = float(two_sided_p[1][b])
        rows.append(row)
    report = {
        "unit": "block",
        "statistic": "mean signed edge statistic per block (Noble & Scheinost, 2020)",
        "correction": correction,
        "tail_policy": "two-sided" if two_sided_p is not None else "separate-one-sided",
        "n_blocks": blocks.n_blocks,
        "n_blocks_tested": m,
        "n_permutations": B,
        "blocks": rows,
    }
    return p_edges, report
