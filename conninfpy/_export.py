"""Edge-level export helpers for :class:`InferenceResult` /
:class:`OmnibusInferenceResult`.

Implements the shared logic behind :meth:`significant_edges` and
:meth:`to_csv` on both result classes. Kept in a single module so the
tail and omnibus paths share the same column-naming and atlas-merging
conventions.

.. note::

   *Significance is a statistical, not mechanistic, statement.* The
   tables this module produces are FWER/FDR-controlled claims about
   edge-wise null rejection; they do not by themselves establish that
   an edge corresponds to a specific cognitive or biological
   mechanism. Block-level patterns
   (``roi_i_network`` × ``roi_j_network``) carry more interpretive
   weight than individual edges; prefer ``sort='network_pair'`` for
   reporting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .atlas import AtlasInfo


_VALID_TAILS = ("both", "positive", "negative")
_VALID_SORTS = ("p", "statistic", "effect_size", "network_pair")


def _check_square(arr: npt.NDArray, name: str) -> None:
    """Raise ValueError unless ``arr`` is a square (N, N) matrix."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a square (N, N) matrix; got shape {arr.shape}."
        )


def _row_tail_label(
    p_pos: npt.NDArray[np.float64],
    p_neg: npt.NDArray[np.float64],
    alpha: float,
) -> npt.NDArray[np.object_]:
    """Per-edge significance label: 'positive', 'negative', 'both', or ''."""
    pos_sig = p_pos <= alpha
    neg_sig = p_neg <= alpha
    out = np.full(p_pos.shape, "", dtype=object)
    out[pos_sig & ~neg_sig] = "positive"
    out[~pos_sig & neg_sig] = "negative"
    out[pos_sig & neg_sig] = "both"
    return out


def _annotate_with_atlas(
    df: pd.DataFrame,
    atlas: "AtlasInfo",
    *,
    n_nodes: int,
) -> pd.DataFrame:
    """Merge ROI names / networks / hemisphere / network_pair into ``df``."""
    if len(atlas) != n_nodes:
        raise ValueError(
            f"atlas has {len(atlas)} ROIs but the result is on "
            f"{n_nodes} nodes; cannot annotate."
        )
    labels = np.asarray(atlas.labels, dtype=object)
    networks = np.asarray(atlas.networks, dtype=object)
    hemi = (
        np.asarray(atlas.hemisphere, dtype=object)
        if atlas.hemisphere is not None
        else None
    )

    i_idx = df["roi_i"].to_numpy()
    j_idx = df["roi_j"].to_numpy()

    df = df.copy()
    df["roi_i_name"] = labels[i_idx]
    df["roi_j_name"] = labels[j_idx]
    df["roi_i_network"] = networks[i_idx]
    df["roi_j_network"] = networks[j_idx]
    # Canonicalize network_pair as a sorted tuple-string so that a
    # within-block (X, Y) is the same key as (Y, X).
    pair = np.array(
        [
            "—".join(sorted((a, b)))
            for a, b in zip(df["roi_i_network"], df["roi_j_network"])
        ],
        dtype=object,
    )
    df["network_pair"] = pair
    if hemi is not None:
        df["hemisphere_i"] = hemi[i_idx]
        df["hemisphere_j"] = hemi[j_idx]
    return df


def _column_order_tailed(with_atlas: bool) -> list:
    """Preferred column order for the tailed edge table."""
    if with_atlas:
        cols = [
            "edge_id",
            "roi_i",
            "roi_i_name",
            "roi_i_network",
            "roi_j",
            "roi_j_name",
            "roi_j_network",
            "network_pair",
            "t_signed",
            "p_positive",
            "p_negative",
            "p_min",
            "tail",
            "hemisphere_i",
            "hemisphere_j",
        ]
    else:
        cols = [
            "edge_id",
            "roi_i",
            "roi_j",
            "t_signed",
            "p_positive",
            "p_negative",
            "p_min",
            "tail",
        ]
    return cols


def _column_order_omnibus(with_atlas: bool) -> list:
    """Preferred column order for the omnibus edge table."""
    if with_atlas:
        return [
            "edge_id",
            "roi_i",
            "roi_i_name",
            "roi_i_network",
            "roi_j",
            "roi_j_name",
            "roi_j_network",
            "network_pair",
            "F",
            "p_omnibus",
            "hemisphere_i",
            "hemisphere_j",
        ]
    return ["edge_id", "roi_i", "roi_j", "F", "p_omnibus"]


def build_tailed_dataframe(
    p_positive: npt.NDArray[np.float64],
    p_negative: npt.NDArray[np.float64],
    *,
    stat_signed: Optional[npt.NDArray[np.float64]] = None,
    atlas: Optional["AtlasInfo"] = None,
    alpha: float = 0.05,
    tail: str = "both",
    sort: str = "p",
    include_nonsig: bool = False,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Build the significant-edges table for a t / β inference result."""
    if tail not in _VALID_TAILS:
        raise ValueError(
            f"tail must be one of {_VALID_TAILS}; got {tail!r}."
        )
    if sort not in _VALID_SORTS:
        raise ValueError(
            f"sort must be one of {_VALID_SORTS}; got {sort!r}."
        )
    if sort == "network_pair" and atlas is None:
        raise ValueError(
            "sort='network_pair' requires an atlas (no network labels "
            "available without one)."
        )
    if stat_signed is None:
        raise ValueError(
            "stat maps not populated; v2.0 pickled results are not "
            "exportable — rerun with v2.1+ pipeline."
        )

    _check_square(p_positive, "p_positive")
    _check_square(p_negative, "p_negative")
    if p_negative.shape != p_positive.shape:
        raise ValueError("p_positive and p_negative must have the same shape.")
    _check_square(stat_signed, "stat_signed")
    if stat_signed.shape != p_positive.shape:
        raise ValueError("stat_signed must match p_positive shape.")

    N = p_positive.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    p_pos_v = p_positive[iu, ju]
    p_neg_v = p_negative[iu, ju]
    t_v = stat_signed[iu, ju]
    p_min_v = np.minimum(p_pos_v, p_neg_v)

    # Filter by `tail` parameter.
    if tail == "both":
        sig_mask = p_min_v <= alpha
    elif tail == "positive":
        sig_mask = p_pos_v <= alpha
    else:  # negative
        sig_mask = p_neg_v <= alpha

    keep = np.ones_like(sig_mask) if include_nonsig else sig_mask

    tail_label_full = _row_tail_label(
        p_positive, p_negative, alpha
    )[iu, ju]

    edge_id = np.arange(len(iu))[keep]
    df = pd.DataFrame(
        {
            "edge_id": edge_id,
            "roi_i": iu[keep],
            "roi_j": ju[keep],
            "t_signed": t_v[keep],
            "p_positive": p_pos_v[keep],
            "p_negative": p_neg_v[keep],
            "p_min": p_min_v[keep],
            "tail": tail_label_full[keep],
        }
    )

    with_atlas = atlas is not None
    if atlas is not None:
        df = _annotate_with_atlas(df, atlas, n_nodes=N)

    df = _sort_dataframe(df, sort, by="p_min")

    if top_k is not None:
        df = df.head(int(top_k))

    cols = [c for c in _column_order_tailed(with_atlas) if c in df.columns]
    return df[cols].reset_index(drop=True)


def build_omnibus_dataframe(
    p_omnibus: npt.NDArray[np.float64],
    *,
    stat_omnibus: Optional[npt.NDArray[np.float64]] = None,
    atlas: Optional["AtlasInfo"] = None,
    alpha: float = 0.05,
    sort: str = "p",
    include_nonsig: bool = False,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Build the significant-edges table for an F-stat omnibus result."""
    if sort not in _VALID_SORTS:
        raise ValueError(
            f"sort must be one of {_VALID_SORTS}; got {sort!r}."
        )
    if sort == "network_pair" and atlas is None:
        raise ValueError(
            "sort='network_pair' requires an atlas."
        )
    if stat_omnibus is None:
        raise ValueError(
            "stat map not populated; v2.0 pickled F-stat results are "
            "not exportable — rerun with v2.1+ pipeline."
        )

    _check_square(p_omnibus, "p_omnibus")
    _check_square(stat_omnibus, "stat_omnibus")
    if stat_omnibus.shape != p_omnibus.shape:
        raise ValueError("stat_omnibus must match p_omnibus shape.")

    N = p_omnibus.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    p_v = p_omnibus[iu, ju]
    f_v = stat_omnibus[iu, ju]

    keep = np.ones_like(p_v, dtype=bool) if include_nonsig else (p_v <= alpha)

    df = pd.DataFrame(
        {
            "edge_id": np.arange(len(iu))[keep],
            "roi_i": iu[keep],
            "roi_j": ju[keep],
            "F": f_v[keep],
            "p_omnibus": p_v[keep],
        }
    )

    with_atlas = atlas is not None
    if atlas is not None:
        df = _annotate_with_atlas(df, atlas, n_nodes=N)

    df = _sort_dataframe(df, sort, by="p_omnibus", effect_col="F")

    if top_k is not None:
        df = df.head(int(top_k))

    cols = [c for c in _column_order_omnibus(with_atlas) if c in df.columns]
    return df[cols].reset_index(drop=True)


def _sort_dataframe(
    df: pd.DataFrame,
    sort: str,
    *,
    by: str,
    effect_col: str = "t_signed",
) -> pd.DataFrame:
    """Sort an edge table by p, test statistic, or network pair."""
    if sort == "effect_size":
        warnings.warn("sort='effect_size' ranks a test statistic; use sort='statistic'.",
                      DeprecationWarning, stacklevel=3)
        sort = "statistic"
    if df.empty:
        return df
    if sort == "p":
        return df.sort_values(by=by, ascending=True, kind="mergesort")
    if sort == "statistic":
        if effect_col == "t_signed":
            key = df[effect_col].abs()
        else:
            key = df[effect_col]
        return df.assign(_k=key).sort_values(
            by="_k", ascending=False, kind="mergesort"
        ).drop(columns="_k")
    # network_pair: group by (sorted) network pair then by p within group
    return df.sort_values(
        by=["network_pair", by], ascending=[True, True], kind="mergesort"
    )


# ---------------------------------------------------------------------------
# Network-pair aggregation
# ---------------------------------------------------------------------------

def _network_codes(networks) -> "tuple[np.ndarray, list]":
    """Integer code per ROI and the network names, in first-appearance order.

    The same order as :meth:`AtlasInfo.network_index` and the network plots.
    """
    seen: dict = {}
    for name in networks:
        if name not in seen:
            seen[name] = len(seen)
    codes = np.array([seen[name] for name in networks], dtype=np.int_)
    return codes, list(seen)


def build_network_summary(
    p_maps: "dict[str, npt.NDArray[np.float64]]",
    networks,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate edge-wise p-value maps over pairs of networks.

    Edges are the upper triangle of each ``(N, N)`` map; an edge belongs to
    the unordered pair of its endpoints' networks. For every tail
    ``t`` in ``p_maps`` the table holds ``n_sig_t`` (edges with
    ``p <= alpha``), ``prop_sig_t`` (``n_sig_t / n_edges``) and the
    ``-log10 p`` mass ``sum_neglog10p_t`` / ``mean_neglog10p_t``. With both
    ``'positive'`` and ``'negative'`` tails it adds ``signed_proportion =
    (n_sig_positive - n_sig_negative) / n_edges``. ``n_edges`` counts the
    tested edges: an edge with a NaN p-value (not tested, e.g. too few
    subjects under available-case analysis) is excluded. Pairs without a
    tested edge (e.g. a single-ROI network's within-block) are omitted.
    """
    names = list(p_maps)
    first = np.asarray(p_maps[names[0]], dtype=np.float64)
    _check_square(first, names[0])
    n_nodes = first.shape[0]
    if len(networks) != n_nodes:
        raise ValueError(
            f"{len(networks)} network labels for {n_nodes} nodes; one label "
            "per ROI is required."
        )
    codes, net_names = _network_codes(networks)
    n_nets = len(net_names)
    iu, ju = np.triu_indices(n_nodes, k=1)
    lo = np.minimum(codes[iu], codes[ju])
    hi = np.maximum(codes[iu], codes[ju])

    tested = np.isfinite(first[iu, ju])
    n_edges = np.zeros((n_nets, n_nets))
    np.add.at(n_edges, (lo, hi), tested.astype(float))
    a, b = np.nonzero(np.triu(n_edges) > 0)
    table = {
        "network_a": [net_names[k] for k in a],
        "network_b": [net_names[k] for k in b],
        "network_pair": ["—".join(sorted((net_names[i], net_names[j]))) for i, j in zip(a, b)],
        "within_network": a == b,
        "n_edges": n_edges[a, b].astype(int),
    }
    for tail in names:
        p = np.asarray(p_maps[tail], dtype=np.float64)
        if p.shape != first.shape:
            raise ValueError(f"p-value map {tail!r} has shape {p.shape}; expected {first.shape}.")
        p_edges = p[iu, ju]
        sig = np.zeros((n_nets, n_nets))
        np.add.at(sig, (lo, hi), (p_edges <= alpha).astype(float))
        # Permutation p-values are never 0; the floor only guards parametric p.
        neglog = np.nan_to_num(-np.log10(np.clip(p_edges, 1e-300, 1.0)), nan=0.0)
        mass = np.zeros((n_nets, n_nets))
        np.add.at(mass, (lo, hi), neglog)
        table[f"n_sig_{tail}"] = sig[a, b].astype(int)
        table[f"prop_sig_{tail}"] = sig[a, b] / n_edges[a, b]
        table[f"sum_neglog10p_{tail}"] = mass[a, b]
        table[f"mean_neglog10p_{tail}"] = mass[a, b] / n_edges[a, b]
    if "positive" in p_maps and "negative" in p_maps:
        table["signed_proportion"] = (
            table["n_sig_positive"] - table["n_sig_negative"]
        ) / table["n_edges"]
    summary = pd.DataFrame(table)
    summary.attrs["networks"] = net_names   # row/column order for matrices
    return summary


def network_summary_matrix(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot one column of a network summary into a symmetric K × K table.

    Rows and columns are the networks in first-appearance order; pairs with
    no possible edge are 0.
    """
    if value not in summary.columns:
        raise ValueError(
            f"{value!r} is not a summary column; choose from "
            f"{[c for c in summary.columns if c not in ('network_a', 'network_b', 'network_pair', 'within_network')]}."
        )
    order = summary.attrs.get("networks") or list(
        dict.fromkeys(list(summary["network_a"]) + list(summary["network_b"])))
    mat = pd.DataFrame(0.0, index=order, columns=order)
    for a, b, v in zip(summary["network_a"], summary["network_b"], summary[value]):
        mat.loc[a, b] = v
        mat.loc[b, a] = v
    return mat


__all__ = [
    "build_tailed_dataframe",
    "build_omnibus_dataframe",
    "build_network_summary",
    "network_summary_matrix",
]
