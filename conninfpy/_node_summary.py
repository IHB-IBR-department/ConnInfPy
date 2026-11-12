"""Node-level (ROI) description of the detections of an inference result.

:meth:`InferenceResult.node_summary` answers *which ROIs gather the detected
connections*. It is descriptive: it introduces no p-value and no new
significance, and it never modifies the result. What a "detection" of a node
is depends on the inference unit of the result (``result.inference_unit``):

``'edge'`` (``tstat``, ``bh_fdr``, ``bh_fdr_perm``, ``tfnbs``, ``ni_tfnbs``, ``fbc_tfnbs``)
    Every edge has its own p-value. Per ROI and tail: the number of
    significant incident edges (``p <= alpha``), out of the incident edges that
    were *tested* (an edge outside an ``edge_mask`` family or without enough
    subjects under ``missing='available_case'`` has a NaN p-value and is in
    neither the numerator nor the denominator), their share, and the mean
    contrast effect (``c' beta``, :meth:`InferenceResult.effect_map`) over them.

``'component'`` (``nbs``) and ``'block'`` (``cnbs``)
    All edges of a component or block carry one p-value, so a count of
    "significant edges" or a mass of ``-log10 p`` says nothing about the ROI.
    The summary lists, per ROI and tail, the significant components or blocks
    that contain it and how many of its tested edges lie in them. No p-value
    mass is formed.

For ROIs with at least one detection and an atlas with network labels the
partner networks of those edges (edges of the significant components or blocks
for ``nbs`` / ``cnbs``) are tabulated, with the participation coefficient of
Guimera & Amaral (2005), ``P_i = 1 - sum_s (k_is / k_i)**2``. It is NaN, not
0, for a ROI without detections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["build_node_summary"]

_NOTE = {
    "edge": "descriptive: counts of significant edges per ROI; no new p-value or significance is introduced",
    "component": (
        "descriptive: the inference unit is the component (all its edges share one p-value); "
        "no per-ROI significance, p-value mass or count of significant edges is formed"
    ),
    "block": (
        "descriptive: the inference unit is the block (all its edges share one p-value); "
        "no per-ROI significance, p-value mass or count of significant edges is formed"
    ),
}


def _suffix(tails: Sequence[str]) -> Dict[str, str]:
    """Column suffix per tail: none for the single ``'omnibus'`` tail."""
    return {t: ("" if list(tails) == ["omnibus"] else f"_{t}") for t in tails}


def _count_edges(n: int, i: npt.NDArray[np.intp], j: npt.NDArray[np.intp],
                 mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """Edges per node among the upper-triangle edges ``(i, j)`` selected by ``mask``."""
    return np.bincount(i[mask], minlength=n) + np.bincount(j[mask], minlength=n)


def _partner_columns(
    n: int,
    i: npt.NDArray[np.intp],
    j: npt.NDArray[np.intp],
    selected: npt.NDArray[np.bool_],
    networks: npt.NDArray[np.object_],
) -> Dict[str, Any]:
    """Partner networks of each node's selected edges and the participation coefficient."""
    from conninfpy.stats._components import _count_labels

    names, code = np.unique(networks.astype(str), return_inverse=True)
    k = np.zeros((n, names.size), dtype=np.float64)
    np.add.at(k, (i[selected], code[j[selected]]), 1.0)
    np.add.at(k, (j[selected], code[i[selected]]), 1.0)
    total = k.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        participation = np.where(total > 0, 1.0 - ((k / total[:, None]) ** 2).sum(axis=1), np.nan)
    partners = [
        _count_labels(np.repeat(names, k[v].astype(int))) if total[v] > 0 else []
        for v in range(n)
    ]
    return {
        "n_networks_reached": (k > 0).sum(axis=1).astype(np.int64),
        "participation": participation,
        "partner_networks": partners,
    }


def build_node_summary(
    *,
    unit: str,
    p_maps: Dict[str, npt.NDArray[np.float64]],
    tails: Sequence[str],
    alpha: float,
    effect: Optional[npt.NDArray[np.float64]] = None,
    has_effect: bool = True,
    report: Optional[Dict[str, Any]] = None,
    atlas: Optional[Any] = None,
    sort: str = "detections",
    top_k: Optional[int] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Per-ROI description of the detections of one (2D) result; see the module docstring.

    Parameters
    ----------
    unit : {'edge', 'component', 'block'}
        ``result.inference_unit``.
    p_maps : dict of tail -> ndarray (N, N)
        Corrected p-value maps (NaN for edges that were not tested).
    tails : sequence of str
        The tails that were tested and are reported.
    alpha : float
        ``p <= alpha`` marks a significant edge, component or block.
    effect : ndarray (N, N), optional
        Contrast effect map (edge mode); ``None`` gives NaN columns.
    has_effect : bool
        False for results without a signed effect concept (F omnibus): no effect columns.
    report : dict, optional
        ``component_report`` (unit ``'component'``) or ``block_report`` (``'block'``).
    atlas : AtlasInfo, optional
    sort : {'detections', 'roi'}
    top_k : int, optional
    attrs : dict, optional
        Extra ``DataFrame.attrs`` entries.
    """
    from conninfpy.stats._components import _usable_networks

    if sort not in ("detections", "roi"):
        raise ValueError(f"sort must be 'detections' or 'roi'; got {sort!r}.")
    tails = list(tails)
    first = np.asarray(p_maps[tails[0]])
    n = first.shape[0]
    if first.ndim != 2 or first.shape[1] != n:
        raise ValueError("node_summary needs square (N, N) p-value maps.")
    if atlas is not None and len(atlas) != n:
        raise ValueError(f"atlas has {len(atlas)} ROIs but the result is on {n} nodes; cannot annotate.")

    i, j = np.triu_indices(n, k=1)
    p_edge = {t: np.asarray(p_maps[t], dtype=np.float64)[i, j] for t in tails}
    tested = np.logical_and.reduce([np.isfinite(p_edge[t]) for t in tails])
    n_tested = _count_edges(n, i, j, tested)
    suffix = _suffix(tails)
    cols: Dict[str, Any] = {"n_tested": n_tested.astype(np.int64)}
    selected = np.zeros(i.size, dtype=bool)   # the edges that count as detections of their endpoints

    if unit == "edge":
        sig = {t: tested & (p_edge[t] <= alpha) for t in tails}
        for t in tails:
            selected |= sig[t]
        n_sig = {t: _count_edges(n, i, j, sig[t]) for t in tails}
        n_total = _count_edges(n, i, j, selected)
        eff = None if effect is None else np.asarray(effect, dtype=np.float64)[i, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            prop = lambda k: np.where(n_tested > 0, k / n_tested, np.nan)  # noqa: E731
            if suffix[tails[0]] != "":
                for t in tails:
                    cols[f"n_sig{suffix[t]}"] = n_sig[t].astype(np.int64)
            cols["n_sig_total"] = n_total.astype(np.int64)
            if suffix[tails[0]] != "":
                for t in tails:
                    cols[f"prop_sig{suffix[t]}"] = prop(n_sig[t])
            cols["prop_sig"] = prop(n_total)
            if has_effect:
                for t in tails:
                    if eff is None:
                        cols[f"mean_effect{suffix[t]}"] = np.full(n, np.nan)
                        continue
                    w = np.where(sig[t], eff, 0.0)
                    total = np.bincount(i, weights=w, minlength=n) + np.bincount(j, weights=w, minlength=n)
                    cols[f"mean_effect{suffix[t]}"] = np.where(n_sig[t] > 0, total / n_sig[t], np.nan)
        detections = n_total
    elif unit in ("component", "block"):
        if report is None:
            raise ValueError(
                f"this result has no {unit} report (it was made before it was recorded); "
                "rerun the inference to get a node summary."
            )
        name = "components" if unit == "component" else "blocks"
        ids: Dict[str, List[List[Any]]] = {t: [[] for _ in range(n)] for t in tails}
        in_unit = {t: np.zeros(i.size, dtype=np.int64) for t in tails}
        edge_in_unit = {t: np.zeros(i.size, dtype=bool) for t in tails}
        if unit == "component":
            for row in report["components"]:
                t = row["tail"]
                if t not in tails or not (np.isfinite(row["p_fwer"]) and row["p_fwer"] <= alpha):
                    continue
                members = (np.asarray(report["labels"][t]) == row["component_id"])[i, j] & tested
                edge_in_unit[t] |= members
                for v in row["nodes"]:
                    ids[t][v].append(int(row["component_id"]))
        else:
            labels = report.get("node_labels")
            if labels is None:
                raise ValueError(
                    "this block_report does not record the node labels of the partition "
                    "(result made before node_summary existed); rerun the inference."
                )
            lab = np.asarray(labels)
            if lab.shape != (n,):
                raise ValueError(f"block_report has {lab.shape[0]} node labels but the result is on {n} nodes.")
            for row in report["blocks"]:
                for t in tails:
                    p = row.get(f"p_{t}", np.nan)
                    if not (np.isfinite(p) and p <= alpha):
                        continue
                    a, b = row["network_a"], row["network_b"]
                    members = (((lab[i] == a) & (lab[j] == b)) | ((lab[i] == b) & (lab[j] == a))) & tested
                    edge_in_unit[t] |= members
                    pair = "—".join(sorted((str(a), str(b))))
                    for v in np.flatnonzero((lab == a) | (lab == b)):
                        ids[t][v].append(pair)
        for t in tails:
            selected |= edge_in_unit[t]
            cols[f"n_{name}{suffix[t]}"] = np.array([len(x) for x in ids[t]], dtype=np.int64)
            cols[f"{unit}_ids{suffix[t]}"] = ids[t]
            cols[f"n_edges_in_{name}{suffix[t]}"] = _count_edges(n, i, j, edge_in_unit[t]).astype(np.int64)
        total_edges = _count_edges(n, i, j, selected)
        if suffix[tails[0]] != "":
            cols[f"n_edges_in_{name}"] = total_edges.astype(np.int64)
        detections = total_edges
    else:
        raise ValueError(f"unknown inference unit {unit!r}.")

    df = pd.DataFrame({"roi": np.arange(n)})
    networks = None
    if atlas is not None:
        df["roi_name"] = [str(x) for x in atlas.labels]
        networks = _usable_networks(atlas)
        if networks is not None:
            df["network"] = networks
        if getattr(atlas, "hemisphere", None) is not None:
            df["hemisphere"] = [str(x) for x in atlas.hemisphere]
    df["inference_unit"] = unit
    for key, value in cols.items():
        df[key] = value
    if networks is not None:
        for key, value in _partner_columns(n, i, j, selected, networks).items():
            df[key] = value

    if sort == "detections":
        df = df.assign(_k=detections).sort_values("_k", ascending=False, kind="mergesort").drop(columns="_k")
    if top_k is not None:
        df = df.head(int(top_k))
    df = df.reset_index(drop=True)
    df.attrs.update(
        {
            "alpha": alpha,
            "inference_unit": unit,
            "tails": tails,
            "descriptive": _NOTE[unit],
            "denominator": "tested incident edges (edges with a NaN p-value are in neither numerator nor denominator)",
            "effect_source": (
                "contrast effect map c'beta (InferenceResult.effect_map); NaN when the result carries none"
                if unit == "edge" and has_effect
                else None
            ),
            "participation": (
                "Guimera & Amaral (2005) coefficient of the partner networks of the detected edges; "
                "NaN without detections; 0 for a ROI whose detections all go to one network"
                if networks is not None
                else None
            ),
            **(attrs or {}),
        }
    )
    return df
