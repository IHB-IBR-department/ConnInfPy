"""Networks plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover (env-specific)
    raise ImportError(
        "conninfpy.plot requires matplotlib; install via `pip install matplotlib`."
    ) from exc

import pandas as pd

from conninfpy._export import build_network_summary, network_summary_matrix
from conninfpy.harmonize import block_mass as _block_mass
from conninfpy.results import InferenceResult, OmnibusInferenceResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from conninfpy.atlas.metadata import AtlasInfo


def plot_block_mass(
    p_map: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    *,
    network_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    annotate: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """Yeo-style block-mass heatmap of significant edges.

    Parameters
    ----------
    p_map : ndarray of shape (N, N)
        Edge-level p-value matrix (symmetric, diagonal=1).
    net_labels : ndarray of shape (N,)
        Integer network assignment per node, ``0..K-1``.
    network_names : sequence of str, optional
        Display names for the K networks. Defaults to ``["Net 0", ...]``.
    alpha : float, default 0.05
        Significance threshold for the underlying :func:`block_mass`
        aggregation.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new ``(5, 4)`` figure is
        created.
    cmap : str, default "viridis"
        Matplotlib colormap.
    annotate : bool, default True
        Print edge counts inside each cell.
    title : str, optional
        Axis title.
    """
    M = _block_mass(p_map, net_labels, alpha=alpha, return_upper=True)
    K = M.shape[0]
    if network_names is None:
        network_names = [f"Net {i}" for i in range(K)]
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4))
    vmax = max(int(M.max()), 1)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax)
    if annotate:
        for i in range(K):
            for j in range(K):
                if i <= j and M[i, j] > 0:
                    color = "white" if M[i, j] < vmax / 2 else "black"
                    ax.text(
                        j,
                        i,
                        str(int(M[i, j])),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(network_names, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(network_names, fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_network_summary(
    result: Any,
    *,
    atlas: "AtlasInfo",
    alpha: float = 0.05,
    ax: Optional[Axes] = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """K × K block heatmap of the signed proportion of significant edges.

    For each pair of networks ``(i, j)`` (canonical, unordered), the
    cell value is

    .. math::

       s_{ij} = \\frac{n_{ij}^{+} - n_{ij}^{-}}{n_{ij}^{\\text{total}}}

    where ``n_{ij}^{+}`` and ``n_{ij}^{-}`` are counts of significant
    positive- and negative-tail edges with one endpoint in network
    ``i`` and the other in network ``j``, and the denominator is the
    total number of edges in that block. The ratio is bounded in
    ``[-1, 1]``; ``+1`` means every block edge is significantly
    positive, ``-1`` significantly negative, ``0`` no net direction.

    For the F-stat omnibus path the cell holds the *unsigned*
    proportion ``n_{ij} / n_{ij}^{total}`` and the colormap defaults
    to a sequential scheme.

    Parameters
    ----------
    result : InferenceResult or OmnibusInferenceResult
        Inference output (uses ``result['positive']`` / ``['negative']``
        or ``['omnibus']`` p-value maps).
    atlas : :class:`AtlasInfo`
        Required — supplies the per-ROI network labels.
    alpha : float, default 0.05
        Significance threshold.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    cmap : str, default "RdBu_r"
        Matplotlib colormap (sequential for the F-stat path).
    annotate : bool, default True
        Print the cell value inside each block.
    title : str, optional
        Axis title.
    """
    if not isinstance(result, (InferenceResult, OmnibusInferenceResult)):
        raise TypeError(
            f"Unsupported result type {type(result).__name__}; expected "
            "InferenceResult or OmnibusInferenceResult."
        )
    omnibus = isinstance(result, OmnibusInferenceResult)
    if omnibus:
        p_maps = {"omnibus": np.asarray(result["omnibus"], dtype=np.float64)}
    else:
        p_maps = {
            "positive": np.asarray(result["positive"], dtype=np.float64),
            "negative": np.asarray(result["negative"], dtype=np.float64),
        }
    N = next(iter(p_maps.values())).shape[0]
    if len(atlas) != N:
        raise ValueError(
            f"atlas has {len(atlas)} ROIs but the result is on "
            f"{N} nodes; cannot summarize."
        )

    # Same numbers as result.network_summary(atlas): one source of truth.
    summary = build_network_summary(p_maps, list(atlas.networks), alpha=alpha)
    block_names = summary.attrs["networks"]
    K = len(block_names)
    value = "prop_sig_omnibus" if omnibus else "signed_proportion"
    cell_full = network_summary_matrix(summary, value).to_numpy()
    total = network_summary_matrix(summary, "n_edges").to_numpy()
    if omnibus:
        vmin, vmax = 0.0, 1.0
        cmap_eff = "magma" if cmap == "RdBu_r" else cmap
    else:
        vmax = float(np.max(np.abs(cell_full))) if cell_full.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        vmin = -vmax
        cmap_eff = cmap

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cell_full, cmap=cmap_eff, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if isinstance(result, OmnibusInferenceResult):
        cbar.set_label("sig fraction", fontsize=8)
    else:
        cbar.set_label("signed sig fraction", fontsize=8)
    if annotate:
        for i in range(K):
            for j in range(K):
                if total[i, j] <= 0:
                    continue
                ax.text(
                    j,
                    i,
                    f"{cell_full[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(block_names, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(block_names, fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_connectome_graph(
    edges_df,
    atlas: "AtlasInfo",
    tail: str,
    *,
    alpha: float = 0.05,
    top_n: int = 100,
    display_mode: str = "z",
    color_by: str = "Network",
    weight_by: str = "-log10(p)",
    rank_by: str = "p-value",
    selected_networks: Optional[Sequence[str]] = None,
    network_filter_mode: str = "Either endpoint",
    title: Optional[str] = None,
) -> Figure:
    """Plot significant edges on a brain connectome model.

    Requires `nilearn` and `pandas` packages to be installed.
    """
    try:
        from nilearn import plotting
    except ImportError as exc:
        raise ImportError(
            "plot_connectome_graph requires nilearn; install via `pip install nilearn`."
        ) from exc

    if getattr(atlas, "coords", None) is None:
        raise ValueError(
            "plot_connectome_graph requires an atlas with MNI coordinates."
        )

    # 1. Filter edges for tail, significance, and top_n
    p_col = "p_positive" if tail == "positive" else "p_negative"
    if edges_df is None or edges_df.empty:
        df = pd.DataFrame()
    else:
        df = edges_df.copy()
        df = df[df[p_col] <= alpha]

        if selected_networks and getattr(atlas, "networks", None) is not None:
            networks = np.asarray(atlas.networks, dtype=object)
            roi_i_net = (
                df["roi_i"]
                .astype(int)
                .map(lambda idx: networks[idx] if idx < len(networks) else "Unknown")
            )
            roi_j_net = (
                df["roi_j"]
                .astype(int)
                .map(lambda idx: networks[idx] if idx < len(networks) else "Unknown")
            )
            selected = set(selected_networks)
            if network_filter_mode == "Both endpoints":
                keep = roi_i_net.isin(selected) & roi_j_net.isin(selected)
            else:
                keep = roi_i_net.isin(selected) | roi_j_net.isin(selected)
            df = df[keep]

        if not df.empty:
            if rank_by == "|t|":
                df = df.assign(_rank_value=df["t_signed"].abs())
                df = df.sort_values("_rank_value", ascending=False)
            else:
                df = df.sort_values(p_col, ascending=True)
            df = df.head(int(top_n))

    # 2. Build adjacency matrix and calculate degree
    n_nodes = len(atlas)
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    degrees = np.zeros(n_nodes, dtype=float)

    if not df.empty:
        for _, row in df.iterrows():
            i = int(row["roi_i"])
            j = int(row["roi_j"])
            if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
                continue
            if weight_by == "|t|":
                value = abs(float(row.get("t_signed", 1.0)))
            else:
                value = -np.log10(max(float(row.get(p_col, 1.0)), 1e-12))
            adj[i, j] = value
            adj[j, i] = value
            degrees[i] += 1
            degrees[j] += 1

    # 3. Determine node colors
    node_colors = []
    color_map = {}
    if color_by == "Degree":
        if np.max(degrees) > 0:
            normed = degrees / np.max(degrees)
        else:
            normed = degrees
        cmap = plt.get_cmap("viridis")
        node_colors = [cmap(float(v)) for v in normed]
    elif color_by == "Hemisphere" and getattr(atlas, "hemisphere", None) is not None:
        categories = list(dict.fromkeys(str(v) for v in atlas.hemisphere))
        cmap = plt.get_cmap("tab20")
        color_map = {cat: cmap(idx % cmap.N) for idx, cat in enumerate(categories)}
        node_colors = [color_map[str(v)] for v in atlas.hemisphere]
    elif getattr(atlas, "networks", None) is not None:
        categories = list(dict.fromkeys(str(v) for v in atlas.networks))
        cmap = plt.get_cmap("tab20")
        color_map = {cat: cmap(idx % cmap.N) for idx, cat in enumerate(categories)}
        node_colors = [color_map[str(v)] for v in atlas.networks]
    else:
        node_colors = ["#9CA3AF"] * n_nodes
        color_map = {"ROI": "#9CA3AF"}

    node_sizes = 18 + 18 * np.sqrt(degrees)
    edge_cmap = "Reds" if tail == "positive" else "Blues"

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    plotting.plot_connectome(
        adj,
        atlas.coords,
        node_color=node_colors,
        node_size=node_sizes,
        edge_cmap=edge_cmap,
        edge_threshold=None,
        display_mode=display_mode,
        figure=fig,
        title=title,
        colorbar=False,
    )

    if color_map and color_by != "Degree":
        legend_items = []
        for name, color in color_map.items():
            legend_items.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=7,
                    label=name,
                )
            )
        if legend_items:
            fig.legend(
                handles=legend_items[:14],
                loc="lower center",
                ncol=min(4, len(legend_items)),
                fontsize=7,
                frameon=False,
                bbox_to_anchor=(0.5, -0.02),
            )
    return fig
