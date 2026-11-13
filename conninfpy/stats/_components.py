"""Component report for the classical NBS (Zalesky, Fornito & Bullmore, 2010).

The inference unit of ``method='nbs'`` is the connected component of the
edges whose statistic exceeds the threshold. Each component is scored by its
extent (number of edges) or intensity (``sum(t - threshold)``) and
calibrated against the permutation distribution of the *maximum* component
statistic, which controls the family-wise error rate. Every edge of a
component carries the component's p-value in the edge map; this module
reports the components themselves, as :mod:`conninfpy.stats._cnbs` reports
the blocks of cNBS.

The report is built from the same suprathreshold labelling that scores the
permutations (:func:`conninfpy.enhance.nbs.nbs_components`) and takes each
component's p-value from the p-value map that was returned; nothing is
recomputed or re-corrected. Every edge of a component should carry the same
p-value in that map, and this is checked (to float tolerance, NaN included):
if the edges of a component disagree, a :class:`RuntimeWarning` names the
tail and the component and the component's ``p_fwer`` is NaN, so an
ambiguous p-value is never reported.

Components of every observed suprathreshold cluster are listed, significant
or not (as ``block_report`` lists every block); ``p_fwer <= alpha`` decides
significance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from conninfpy.enhance.nbs import nbs_components

__all__ = [
    "build_component_report",
    "restrict_component_report",
    "component_table",
    "component_edge_columns",
]

_STAT_DESCRIPTION = {
    "extent": "extent: number of edges of the component",
    "intensity": "intensity: sum over the component's edges of (statistic - threshold)",
}
_TAILS = ("positive", "negative", "omnibus")


def build_component_report(
    stat: Dict[str, npt.NDArray[np.float64]],
    p: Dict[str, npt.NDArray[np.float64]],
    *,
    method: Any,
    threshold: float,
    nbs_stat: str,
    edge_stat: str = "tstat",
    n_permutations: int = 0,
    two_tailed: bool = False,
    edge_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Optional[Dict[str, Any]]:
    """Components of every tail of an NBS result, or ``None`` for other methods.

    Parameters
    ----------
    stat : dict of str -> ndarray (N, N)
        Observed, non-negative statistic map per tail (``'positive'`` /
        ``'negative'``, or ``'omnibus'``), before any enhancement.
    p : dict of str -> ndarray (N, N)
        The corrected p-value map per tail that the result carries.
    method : str or StatMethod
        The report exists for ``'nbs'`` only.
    threshold, nbs_stat
        The NBS edge threshold and component statistic of the run.
    edge_stat : str
        Statistic the maps hold (``'tstat'``, ``'beta'``, ``'fstat'``).
    n_permutations : int
        Permutations behind the null distribution of the maximum.
    two_tailed : bool
        Whether both tails were corrected against one joint null.
    edge_mask : ndarray of bool (N, N), optional
        Declared tested family: edges outside it never join a component
        (the permutation test masks them the same way).

    Returns
    -------
    dict or None
        ``unit``, ``statistic`` (what the number is), ``stat_type``,
        ``threshold``, ``edge_statistic``, ``correction``, ``tail_policy``,
        ``tested_tail``, ``n_permutations``, ``n_components`` (per tail),
        ``components`` (one row per component: ``tail``, ``component_id``,
        ``n_edges``, ``n_nodes``, ``statistic``, ``p_fwer``, ``nodes``) and
        ``labels`` (per tail, an (N, N) int array: the component number of
        every edge, 0 outside any component).
    """
    if getattr(method, "value", method) != "nbs":
        return None
    tails = [tail for tail in _TAILS if tail in stat]
    labels: Dict[str, npt.NDArray[np.int32]] = {}
    n_components: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for tail in tails:
        t = np.asarray(stat[tail], dtype=np.float64)
        if edge_mask is not None:
            t = np.where(np.asarray(edge_mask, dtype=bool), t, 0.0)
        found = nbs_components(t, threshold=threshold, stat_type=nbs_stat)
        labels[tail] = found.labels
        n_components[tail] = found.n_components
        if not found.n_components:
            continue
        iu = np.triu_indices(t.shape[0], k=1)
        edge_label = found.labels[iu]
        in_component = edge_label > 0
        ids, p_component = _component_p_values(
            tail,
            edge_label[in_component],
            np.asarray(p[tail], dtype=np.float64)[iu][in_component],
        )
        for k, cid in enumerate(ids):
            rows.append(
                {
                    "tail": tail,
                    "component_id": int(cid),
                    "n_edges": int(found.n_edges[cid - 1]),
                    "n_nodes": int(found.nodes[cid - 1].size),
                    "statistic": float(found.statistic[cid - 1]),
                    "p_fwer": float(p_component[k]),
                    "nodes": [int(v) for v in found.nodes[cid - 1]],
                }
            )
    return {
        "unit": "component",
        "statistic": _STAT_DESCRIPTION[nbs_stat],
        "stat_type": nbs_stat,
        "threshold": float(threshold),
        "edge_statistic": edge_stat,
        "correction": "permutation_max",
        "tail_policy": "omnibus"
        if tails == ["omnibus"]
        else ("two-sided" if two_tailed else "separate-one-sided"),
        "tested_tail": None,
        "n_permutations": int(n_permutations),
        "n_components": n_components,
        "components": rows,
        "labels": labels,
    }


_P_RTOL = 1e-12
_P_ATOL = 1e-12


def _component_p_values(
    tail: str,
    labels: npt.NDArray[np.int32],
    p_edges: npt.NDArray[np.float64],
) -> "tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]":
    """One p-value per component, checked to be the same on all its edges.

    ``labels`` and ``p_edges`` list the component number and the p-value of
    every edge that is in a component. A component whose edges disagree
    (beyond float tolerance, or some NaN and some not) gets NaN and a
    :class:`RuntimeWarning` naming the tail and the component; a component
    whose edges are all NaN (untested) is NaN without a warning.
    """
    order = np.argsort(labels, kind="stable")
    lab, pv = labels[order], p_edges[order]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    ids = lab[starts]
    sizes = np.diff(np.r_[starts, lab.size])
    n_nan = np.add.reduceat(np.isnan(pv).astype(np.intp), starts)
    lo = np.fmin.reduceat(pv, starts)
    hi = np.fmax.reduceat(pv, starts)
    all_nan = n_nan == sizes
    agree = all_nan | ((n_nan == 0) & np.isclose(lo, hi, rtol=_P_RTOL, atol=_P_ATOL))
    p_component = np.where(agree, pv[starts], np.nan)
    for cid, ok, a, b, n_bad in zip(ids, agree, lo, hi, n_nan):
        if not ok:
            warnings.warn(
                f"NBS component {int(cid)} of the {tail} tail: its edges do not share one "
                f"p-value (range {a:.6g} to {b:.6g}, {int(n_bad)} NaN); its p_fwer is "
                "reported as NaN.",
                RuntimeWarning,
                stacklevel=4,
            )
    return ids, p_component


def restrict_component_report(
    report: Optional[Dict[str, Any]], tail: str, *, selection: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """The report of a one-sided result: only the tested tail's components.

    :meth:`InferenceResult.one_sided` sets the other tail's p-values to 1 (not
    tested); its components are not reported either. Component numbers are
    kept, so a component keeps its id. ``selection`` records how the
    direction was chosen (``'prespecified'`` before the run,
    ``'post-hoc'`` after it) as ``tail_selection`` in the report; ``None``
    (the default) leaves the marker unset, as in reports saved before it
    existed.
    """
    if report is None:
        return None
    restricted = {
        **report,
        "tail_policy": f"one-sided-{tail}",
        "tested_tail": tail,
        "n_components": {tail: report["n_components"].get(tail, 0)},
        "components": [dict(r) for r in report["components"] if r["tail"] == tail],
        "labels": {tail: report["labels"][tail]} if tail in report["labels"] else {},
    }
    if selection is not None:
        restricted["tail_selection"] = selection
    return restricted


def component_edge_columns(
    labels: Optional[Dict[str, npt.NDArray[np.int32]]],
    iu: npt.NDArray[np.intp],
    ju: npt.NDArray[np.intp],
) -> Dict[str, Any]:
    """``component_tail`` and ``component_id`` of the upper-triangle edges.

    An edge is in at most one component (its statistic is positive in one tail
    only). ``component_id`` is ``<NA>`` and ``component_tail`` ``''`` for an
    edge in no component. Ids are numbered within each tail: the pair
    (``component_tail``, ``component_id``) is the key of the component report.
    """
    n = len(iu)
    ids = np.zeros(n, dtype=np.int64)
    tails = np.full(n, "", dtype=object)
    for tail, arr in (labels or {}).items():
        here = np.asarray(arr)[iu, ju]
        hit = here > 0
        ids[hit] = here[hit]
        tails[hit] = tail
    return {
        "component_tail": tails,
        "component_id": pd.arrays.IntegerArray(ids, ids == 0),
    }


def _usable_networks(atlas: Any) -> Optional[npt.NDArray[np.object_]]:
    """Network label per ROI, or ``None`` when the atlas has none worth reporting."""
    networks = getattr(atlas, "networks", None)
    if networks is None:
        return None
    text = np.array([str(x).strip() for x in networks], dtype=object)
    present = {x for x in text if x}
    if not present or present == {"Unknown"}:
        return None
    return np.where(text == "", "unassigned", text).astype(object)


def component_table(
    report: Optional[Dict[str, Any]],
    *,
    atlas: Optional[Any] = None,
    alpha: float = 0.05,
    tail: str = "both",
    significant_only: bool = False,
) -> pd.DataFrame:
    """The component report as a table, optionally annotated with an atlas.

    Columns: ``tail``, ``component_id``, ``n_edges``, ``n_nodes``,
    ``statistic`` and ``statistic_type`` (extent or intensity), ``p_fwer``,
    ``significant`` (``p_fwer <= alpha``) and ``nodes`` (ROI indices). With an
    atlas, ``roi_names`` is added, and, when the atlas has network labels,
    ``networks`` (networks of the component's ROIs with their ROI counts) and
    ``network_pairs`` (network pairs of its edges with their edge counts,
    pairs written as in ``significant_edges``). Without an atlas those
    columns are absent; atlas metadata is never guessed.
    """
    if report is None:
        raise ValueError(
            "this result has no component report: it is produced by method='nbs' only."
        )
    if tail not in ("both", "positive", "negative"):
        raise ValueError(f"tail must be 'both', 'positive' or 'negative'; got {tail!r}.")
    rows = [
        r
        for r in report["components"]
        if (tail == "both" or r["tail"] == tail)
        and (not significant_only or (np.isfinite(r["p_fwer"]) and r["p_fwer"] <= alpha))
    ]
    columns = [
        "tail",
        "component_id",
        "n_edges",
        "n_nodes",
        "statistic",
        "statistic_type",
        "p_fwer",
        "significant",
        "nodes",
    ]
    df = pd.DataFrame(
        {
            "tail": [r["tail"] for r in rows],
            "component_id": pd.array([r["component_id"] for r in rows], dtype="int64"),
            "n_edges": pd.array([r["n_edges"] for r in rows], dtype="int64"),
            "n_nodes": pd.array([r["n_nodes"] for r in rows], dtype="int64"),
            "statistic": pd.array([r["statistic"] for r in rows], dtype="float64"),
            "statistic_type": [report["stat_type"]] * len(rows),
            "p_fwer": pd.array([r["p_fwer"] for r in rows], dtype="float64"),
            "significant": pd.array(
                [bool(np.isfinite(r["p_fwer"]) and r["p_fwer"] <= alpha) for r in rows],
                dtype="bool",
            ),
            "nodes": [list(r["nodes"]) for r in rows],
        },
        columns=columns,
    )
    if atlas is not None:
        n_nodes = _n_nodes_of(report)
        if n_nodes is not None and len(atlas) != n_nodes:
            raise ValueError(
                f"atlas has {len(atlas)} ROIs but the result is on {n_nodes} nodes; "
                "cannot annotate."
            )
        names = np.asarray(atlas.labels, dtype=object)
        df["roi_names"] = [[str(names[i]) for i in nodes] for nodes in df["nodes"]]
        networks = _usable_networks(atlas)
        if networks is not None:
            df["networks"] = [_count_labels(networks[nodes]) for nodes in df["nodes"]]
            df["network_pairs"] = _network_pair_counts(report, rows, networks)
    df.attrs.update(
        {
            "threshold": report["threshold"],
            "statistic_type": report["stat_type"],
            "edge_statistic": report["edge_statistic"],
            "tail_policy": report["tail_policy"],
            "tested_tail": report.get("tested_tail"),
            "tail_selection": report.get("tail_selection"),
            "n_permutations": report["n_permutations"],
            "alpha": alpha,
        }
    )
    return df


def _n_nodes_of(report: Dict[str, Any]) -> Optional[int]:
    labels = report["labels"]
    return int(next(iter(labels.values())).shape[0]) if labels else None


def _count_labels(values: Any) -> List[str]:
    """``['A (3)', 'B (1)']``: labels by decreasing count, then name."""
    labels, counts = np.unique(np.asarray(values, dtype=object).astype(str), return_counts=True)
    order = sorted(range(len(labels)), key=lambda i: (-counts[i], labels[i]))
    return [f"{labels[i]} ({counts[i]})" for i in order]


def _network_pair_counts(
    report: Dict[str, Any],
    rows: List[Dict[str, Any]],
    networks: npt.NDArray[np.object_],
) -> List[List[str]]:
    """Per component: its edges counted by network pair (``'A—B (n)'``)."""
    out: List[List[str]] = []
    for r in rows:
        labels = report["labels"][r["tail"]]
        i, j = np.nonzero(np.triu(labels == r["component_id"], k=1))
        pairs = ["—".join(sorted((str(networks[a]), str(networks[b])))) for a, b in zip(i, j)]
        out.append(_count_labels(pairs))
    return out
