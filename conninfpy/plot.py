"""Diagnostic plotting helpers for ConnInfPy results.

Per-panel helpers and a one-call publication figure:

- :func:`plot_block_mass` — Yeo-7 (or arbitrary partition) block-mass
  heatmap of significant edges.
- :func:`plot_p_map` — symmetric edge-level p-value heatmap with
  optional significant-edge highlight.
- :func:`plot_null_max_distribution` — max-stat null distribution
  histogram with optional GPD tail-fit overlay.
- :func:`plot_effect_matrix` — signed effect heatmap (RdBu_r, symmetric
  vmax) with significant edges outlined; optional network-block
  ordering when an :class:`AtlasInfo` is supplied.
- :func:`plot_network_summary` — P × P block heatmap of the signed
  proportion of significant edges per (network_i × network_j) cell;
  more robust than the edge-level map at high node counts.
- :func:`summary_figure` — publication-ready 2×2 layout combining the
  effect matrix, block summary, p-maps, and top-k edge table.
- :func:`plot_connectogram` — Brainnetome-style circular connectogram of
  the detected connections, aggregated to BNA subregions or gyri, with
  chords coloured by the endpoints' functional networks.

Each per-panel helper accepts an optional ``ax=`` argument (creating a
fresh figure if omitted) and returns the
:class:`matplotlib.axes.Axes` so the caller can compose multi-panel
figures. :func:`summary_figure` returns a :class:`matplotlib.figure.Figure`.

Importing ``conninfpy.plot`` requires :mod:`matplotlib`. The plotting
helpers are intentionally not imported at the top level of
:mod:`conninfpy` — call ``import conninfpy.plot`` only if you need
them, so headless environments (CI, batch jobs) don't pull matplotlib.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Tuple

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

from .harmonize import block_mass as _block_mass
import pandas as pd

from ._export import build_network_summary, network_summary_matrix
from .results import InferenceResult, OmnibusInferenceResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .atlas import AtlasInfo


__all__ = [
    "plot_block_mass",
    "plot_p_map",
    "plot_null_max_distribution",
    "plot_effect_matrix",
    "plot_network_summary",
    "summary_figure",
    "plot_connectome_graph",
    "plot_connectogram",
]


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
                    ax.text(j, i, str(int(M[i, j])),
                            ha="center", va="center",
                            fontsize=7, color=color)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(network_names, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(network_names, fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_p_map(
    p_map: npt.NDArray[np.float64],
    *,
    alpha: float = 0.05,
    ax: Optional[Axes] = None,
    cmap: str = "viridis_r",
    log_scale: bool = True,
    title: Optional[str] = None,
    show_significant: bool = True,
) -> Axes:
    """Edge-level p-value heatmap with optional log-scale and sig-mask.

    Parameters
    ----------
    p_map : ndarray of shape (N, N)
        Edge-level p-values.
    alpha : float, default 0.05
        Threshold for the optional significance mask.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    cmap : str, default "viridis_r"
        Reverse viridis (so darker = more significant).
    log_scale : bool, default True
        Plot ``-log10(p)`` rather than raw ``p``.
    title : str, optional
        Axis title.
    show_significant : bool, default True
        Draw a red contour around edges with ``p <= alpha``.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.5, 4))
    if log_scale:
        with np.errstate(divide="ignore"):
            data = -np.log10(np.where(p_map <= 0, 1e-300, p_map))
        cbar_label = r"$-\log_{10}(p)$"
        ref_line = -np.log10(alpha)
    else:
        data = p_map
        cbar_label = r"$p$"
        ref_line = alpha
    im = ax.imshow(data, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    if log_scale:
        cbar.ax.axhline(ref_line, color="red", linewidth=0.6,
                        linestyle="--")
    if show_significant:
        sig = (p_map <= alpha) & np.isfinite(p_map)
        np.fill_diagonal(sig, False)
        if sig.any():
            ax.contour(sig.astype(float), levels=[0.5],
                       colors="red", linewidths=0.4)
    ax.set_xlabel("node")
    ax.set_ylabel("node")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_null_max_distribution(
    null_max: npt.NDArray[np.float64],
    *,
    observed_max: Optional[float] = None,
    gpd_tail: Optional[Tuple[float, float, float]] = None,
    threshold_quantile: float = 0.75,
    ax: Optional[Axes] = None,
    bins: int = 40,
    title: Optional[str] = None,
) -> Axes:
    """Max-statistic null histogram with optional GPD-tail overlay.

    Parameters
    ----------
    null_max : ndarray of shape (J,)
        Per-permutation maximum statistic.
    observed_max : float, optional
        Observed max statistic, drawn as a red vertical line for
        eyeballing FWER significance.
    gpd_tail : tuple ``(u, sigma, xi)``, optional
        GPD parameters from :func:`conninfpy.fit_gpd_tail`. If supplied,
        the GPD survival function is overlaid as a smooth red curve on
        the upper tail.
    threshold_quantile : float, default 0.75
        Empirical quantile at which the GPD tail is anchored — only
        used to position the dashed grey reference line; should match
        the value passed to ``fit_gpd_tail``.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    bins : int, default 40
        Histogram bin count.
    title : str, optional
        Axis title.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(null_max, bins=bins, density=True, alpha=0.55,
            edgecolor="black", linewidth=0.3,
            label=f"empirical null (J={len(null_max)})")
    u_emp = float(np.quantile(null_max, threshold_quantile))
    ax.axvline(u_emp, color="grey", linestyle="--", linewidth=0.6,
               label=f"u = {threshold_quantile:.0%}-tile")
    if gpd_tail is not None:
        u, sigma, xi = gpd_tail
        xs = np.linspace(u, null_max.max() * 1.02, 200)
        z = (xs - u) / sigma
        # Differentiate the survival function: f(x) = (1/σ) (1 + ξ z)^{-1 - 1/ξ}
        if abs(xi) < 1e-8:
            pdf = (1.0 / sigma) * np.exp(-z)
        else:
            base = np.maximum(0, 1 + xi * z)
            pdf = (1.0 / sigma) * np.power(base, -1.0 - 1.0 / xi)
        ax.plot(xs, pdf, color="red", linewidth=1.3,
                label=f"GPD tail (ξ={xi:.2f})")
    if observed_max is not None:
        ax.axvline(observed_max, color="black", linewidth=0.9,
                   label=f"observed max = {observed_max:.2f}")
    ax.set_xlabel("max statistic across edges")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


# =============================================================================
# Result-driven helpers (consume InferenceResult / OmnibusInferenceResult)
# =============================================================================

def _effect_and_pmin_from_result(
    result: Any,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]:
    """Extract a signed effect map and per-edge p-min from a result.

    Returns ``(effect, p_min, label)`` where ``label`` is the colorbar
    label (``"t"``, ``"β"``, or ``"F"`` for the omnibus path).
    """
    if isinstance(result, OmnibusInferenceResult):
        if result.stat_omnibus is None:
            raise ValueError(
                "stat_omnibus is None — v2.0 pickled results cannot be "
                "plotted with plot_effect_matrix; rerun the pipeline."
            )
        return (
            np.asarray(result.stat_omnibus, dtype=np.float64),
            np.asarray(result["omnibus"], dtype=np.float64),
            "F",
        )
    if isinstance(result, InferenceResult):
        signed = result.stat_signed
        if signed is None:
            raise ValueError(
                "stat_positive / stat_negative are None — v2.0 pickled "
                "results cannot be plotted with plot_effect_matrix; "
                "rerun the pipeline."
            )
        p_min = np.minimum(result["positive"], result["negative"])
        label = "β" if result.stat_type == "beta" else "t"
        return (
            np.asarray(signed, dtype=np.float64),
            np.asarray(p_min, dtype=np.float64),
            label,
        )
    raise TypeError(
        f"Unsupported result type {type(result).__name__}; expected "
        "InferenceResult or OmnibusInferenceResult."
    )


def _network_order(
    atlas: "AtlasInfo",
) -> Tuple[npt.NDArray[np.int_], list, list]:
    """Return ``(perm, block_starts, block_names)`` for atlas-ordered plots.

    ``perm`` is the row/column permutation that groups ROIs by network
    in their first-appearance order. ``block_starts`` lists the first
    index of each network block (length K) and ``block_names`` the
    network labels in the same order.
    """
    networks = np.asarray(atlas.networks, dtype=object)
    seen_order: list = []
    for n in networks:
        if n not in seen_order:
            seen_order.append(n)
    perm_parts = [np.where(networks == name)[0] for name in seen_order]
    perm = np.concatenate(perm_parts).astype(int)
    block_starts = [0]
    for part in perm_parts[:-1]:
        block_starts.append(block_starts[-1] + len(part))
    return perm, block_starts, seen_order


def plot_effect_matrix(
    result: Any,
    *,
    alpha: float = 0.05,
    vmax: Optional[float] = None,
    atlas: Optional["AtlasInfo"] = None,
    ax: Optional[Axes] = None,
    order_by_network: bool = True,
    show_network_lines: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """Signed effect heatmap with significant edges outlined.

    Parameters
    ----------
    result : InferenceResult or OmnibusInferenceResult
        Inference output. The signed observed statistic is read from
        ``result.stat_signed`` (t / β path) or ``result.stat_omnibus``
        (F path, unsigned).
    alpha : float, default 0.05
        Significance threshold for the edge-outline overlay
        (``min(p_positive, p_negative) <= alpha`` on the tail path;
        ``p_omnibus <= alpha`` on the F path).
    vmax : float, optional
        Symmetric color bound. Defaults to the 98th percentile of the
        absolute observed statistic.
    atlas : :class:`AtlasInfo`, optional
        When supplied (and ``order_by_network=True``), rows and columns
        are reordered so that same-network ROIs are contiguous; thin
        lines mark the network blocks if ``show_network_lines=True``.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new figure is created.
    order_by_network : bool, default True
        Permute rows/columns to group by network. Ignored without an
        atlas.
    show_network_lines : bool, default True
        Draw light separator lines between network blocks. Ignored
        without an atlas or when ``order_by_network=False``.
    title : str, optional
        Axis title.
    """
    effect, p_min, stat_label = _effect_and_pmin_from_result(result)
    if effect.ndim != 2 or effect.shape[0] != effect.shape[1]:
        raise ValueError(
            f"effect matrix must be square; got {effect.shape}."
        )
    N = effect.shape[0]

    perm: Optional[np.ndarray] = None
    block_starts: list = []
    if atlas is not None and order_by_network:
        if len(atlas) != N:
            raise ValueError(
                f"atlas has {len(atlas)} ROIs but the result is on "
                f"{N} nodes; cannot reorder."
            )
        perm, block_starts, _ = _network_order(atlas)
        effect = effect[np.ix_(perm, perm)]
        p_min = p_min[np.ix_(perm, perm)]

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4.5))

    # Symmetric color bound.
    if vmax is None:
        finite = effect[np.isfinite(effect)]
        if finite.size:
            vmax = float(np.quantile(np.abs(finite), 0.98))
        if not vmax or vmax <= 0:
            vmax = 1.0
    is_unsigned = stat_label == "F"
    if is_unsigned:
        im = ax.imshow(effect, cmap="magma", vmin=0, vmax=vmax)
    else:
        im = ax.imshow(effect, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(stat_label, fontsize=8)

    # Significant-edge contour
    sig = (p_min <= alpha) & np.isfinite(p_min)
    np.fill_diagonal(sig, False)
    if sig.any():
        ax.contour(sig.astype(float), levels=[0.5],
                   colors="black", linewidths=0.5)

    # Network block separators
    if atlas is not None and order_by_network and show_network_lines:
        for s in block_starts[1:]:
            ax.axvline(s - 0.5, color="grey", linewidth=0.4, alpha=0.7)
            ax.axhline(s - 0.5, color="grey", linewidth=0.4, alpha=0.7)

    ax.set_xlabel("node")
    ax.set_ylabel("node")
    if title:
        ax.set_title(title, fontsize=9)
    elif atlas is not None and order_by_network:
        ax.set_title("effect matrix (network-ordered)", fontsize=9)
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
        p_maps = {"positive": np.asarray(result["positive"], dtype=np.float64),
                  "negative": np.asarray(result["negative"], dtype=np.float64)}
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
                ax.text(j, i, f"{cell_full[i, j]:+.2f}",
                        ha="center", va="center",
                        fontsize=6, color="black")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(block_names, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(block_names, fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def summary_figure(
    result: Any,
    *,
    atlas: Optional["AtlasInfo"] = None,
    alpha: float = 0.05,
    top_k: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12.0, 9.0),
) -> Figure:
    """Publication-ready 2 × 2 summary of a ConnInfPy inference run.

    Panels:

    - **top-left**: :func:`plot_effect_matrix` (network-ordered if an
      atlas is supplied).
    - **top-right**: :func:`plot_network_summary` if an atlas is
      supplied, otherwise :func:`plot_block_mass` with a fallback
      single-block partition.
    - **bottom-left**: :func:`plot_p_map` of the positive tail
      (``-log10 p`` heatmap; for the F-stat path the single
      ``omnibus`` map is shown).
    - **bottom-right**: top-``top_k`` row table from
      ``result.significant_edges(atlas, sort='p')``.

    Parameters
    ----------
    result : InferenceResult or OmnibusInferenceResult
        Inference output.
    atlas : :class:`AtlasInfo`, optional
        Atlas for the network-ordered effect matrix, network summary
        block, and the edge-table annotation.
    alpha : float, default 0.05
        Significance threshold used by every panel.
    top_k : int, default 10
        Number of rows in the edge table.
    title : str, optional
        Figure-level title.
    figsize : tuple of float, default ``(12, 9)``
        Inches.
    """
    if not isinstance(result, (InferenceResult, OmnibusInferenceResult)):
        raise TypeError(
            f"Unsupported result type {type(result).__name__}; expected "
            "InferenceResult or OmnibusInferenceResult."
        )

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top-left: effect matrix
    plot_effect_matrix(result, alpha=alpha, atlas=atlas, ax=axes[0, 0])

    # Top-right: network / block summary
    if atlas is not None:
        plot_network_summary(result, atlas=atlas, alpha=alpha,
                             ax=axes[0, 1])
    else:
        # Fall back to the existing block_mass helper with a single-block
        # partition; not very informative but keeps the layout consistent.
        if isinstance(result, InferenceResult):
            p_for_block = np.minimum(result["positive"], result["negative"])
        else:
            p_for_block = result["omnibus"]
        N = p_for_block.shape[0]
        plot_block_mass(p_for_block, np.zeros(N, dtype=int),
                        network_names=["all"], alpha=alpha,
                        ax=axes[0, 1],
                        title="block mass (no atlas)")

    # Bottom-left: p-map
    if isinstance(result, InferenceResult):
        plot_p_map(result["positive"], alpha=alpha, ax=axes[1, 0],
                   title="positive-tail p-map")
    else:
        plot_p_map(result["omnibus"], alpha=alpha, ax=axes[1, 0],
                   title="omnibus p-map")

    # Bottom-right: top-k edge table
    ax_t = axes[1, 1]
    ax_t.axis("off")
    try:
        df = result.significant_edges(atlas=atlas, alpha=alpha,
                                      sort="p", top_k=top_k)
        if df.empty:
            ax_t.text(0.5, 0.5, "No significant edges at α = "
                      f"{alpha:.3g}", ha="center", va="center",
                      fontsize=10)
        else:
            # Pick a small, readable subset of columns.
            if atlas is not None:
                if isinstance(result, OmnibusInferenceResult):
                    cols = ["roi_i_name", "roi_j_name",
                            "network_pair", "F", "p_omnibus"]
                else:
                    cols = ["roi_i_name", "roi_j_name",
                            "network_pair", "t_signed", "p_min", "tail"]
            else:
                if isinstance(result, OmnibusInferenceResult):
                    cols = ["roi_i", "roi_j", "F", "p_omnibus"]
                else:
                    cols = ["roi_i", "roi_j", "t_signed", "p_min", "tail"]
            cols = [c for c in cols if c in df.columns]
            disp = df[cols].copy()
            # Round numeric columns for compact display.
            for c in disp.columns:
                if disp[c].dtype.kind in "fc":
                    disp[c] = disp[c].map(lambda v: f"{v:.3g}")
            table = ax_t.table(
                cellText=disp.values.tolist(),
                colLabels=list(disp.columns),
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.2)
            ax_t.set_title(f"top {len(disp)} significant edges",
                           fontsize=9, loc="left")
    except (ValueError, TypeError) as exc:
        # Stat maps missing or other export-time error — show the message
        # so the figure still composes cleanly.
        ax_t.text(0.5, 0.5, f"(edge table unavailable: {exc})",
                  ha="center", va="center", fontsize=8, wrap=True)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


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
        raise ValueError("plot_connectome_graph requires an atlas with MNI coordinates.")

    # 1. Filter edges for tail, significance, and top_n
    p_col = "p_positive" if tail == "positive" else "p_negative"
    if edges_df is None or edges_df.empty:
        df = pd.DataFrame()
    else:
        df = edges_df.copy()
        df = df[df[p_col] <= alpha]

        if selected_networks and getattr(atlas, "networks", None) is not None:
            networks = np.asarray(atlas.networks, dtype=object)
            roi_i_net = df["roi_i"].astype(int).map(lambda idx: networks[idx] if idx < len(networks) else "Unknown")
            roi_j_net = df["roi_j"].astype(int).map(lambda idx: networks[idx] if idx < len(networks) else "Unknown")
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
                    [0], [0],
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


# ----------------------------------------------------------------------
# Brainnetome connectogram
# ----------------------------------------------------------------------
# Yeo et al. (2011) 7-network colours; Limbic darkened for contrast on white.
_YEO7_COLORS = {
    "Visual": "#781286", "Somatomotor": "#4682B4", "Dorsal Attention": "#00760E",
    "Ventral Attention": "#C43AFA", "Limbic": "#A8C66C", "Frontoparietal": "#E69422",
    "Default": "#CD3E4E",
}
_YEO7_ALIASES = {
    "Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Ventral Attention", "Salience/Ventral Attention": "Ventral Attention",
    "Cont": "Frontoparietal", "Frontoparietal Control": "Frontoparietal",
    "Default Mode": "Default", "DMN": "Default",
}
# Lobe order and abbreviations of the Brainnetome atlas connectogram (atlas.brainnetome.org).
_LOBES = {"Frontal": ("FRO", "#D9C7A7"), "Insular": ("INS", "#B9DCDC"), "Limbic": ("LIM", "#E6C5D6"),
          "Temporal": ("TEM", "#CADDAA"), "Parietal": ("PAR", "#AFC8E2"), "Occipital": ("OCC", "#D7C9EA"),
          "Subcortical": ("SUB", "#D6D6D6")}


def _network_key(name: Any) -> str:
    return _YEO7_ALIASES.get(str(name), str(name))


def _connectogram_layout(regions: Any, lobe_gap: float = 1.5, hemi_gap: float = 4.0) -> Any:
    """Slot of each BNA subregion along its half of the circle, in degrees from the top.

    The right hemisphere runs clockwise from 12 o'clock and the left mirrors it
    (homologs face each other); lobes follow the Brainnetome connectogram order
    and subregions their BNA ID within a lobe, so gyri are contiguous.
    """
    rank = {lobe: k for k, lobe in enumerate(_LOBES)}
    right = regions[regions["hemisphere"] == "R"]
    order = sorted(right.index, key=lambda r: (rank[right.at[r, "lobe"]], r))
    s, prev, slots = hemi_gap / 2, None, {}
    for rid in order:
        lobe = right.at[rid, "lobe"]
        if prev is not None and lobe != prev:
            s += lobe_gap
        slots[rid] = s
        s, prev = s + 1, lobe
    deg = 180.0 / (s + hemi_gap / 2)
    out = regions[["name", "gyrus", "lobe", "hemisphere"]].copy()
    for rid in order:
        out.loc[rid, "s0"] = slots[rid] * deg
        out.loc[int(regions.at[rid, "homolog_id"]), "s0"] = slots[rid] * deg
    out["s1"] = out["s0"] + deg
    return out


def _side_angle(s: Any, hemisphere: str) -> Any:
    """Polar angle (degrees, counter-clockwise from east) of side coordinate ``s``."""
    return 90.0 - s if hemisphere == "R" else 90.0 + s


def plot_connectogram(
    result: Any,
    rois: Any,
    *,
    tail: str = "positive",
    level: str = "gyrus",
    alpha: float = 0.05,
    assign: str = "dominant",
    networks: Optional[Sequence[str]] = None,
    weight: str = "count",
    min_weight: float = 1.0,
    max_links: Optional[int] = None,
    labels: str = "gyrus",
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    param_idx: Optional[int] = None,
) -> Axes:
    """Brainnetome-style connectogram of one tail's detected connections.

    The layout follows the BNA-246 connectogram: left hemisphere on the left,
    right mirrored on the right, lobes from frontal (top) to subcortical
    (bottom). Rings, from outside in: the 246 subregions (pale = not sampled by
    any ROI), the gyri, and the network make-up of each unit at ``level`` (the
    share of its sampled tissue that comes from ROIs of each network).

    Connections are aggregated with
    :func:`conninfpy.interpret.annotate.aggregate_connections`, split by each ROI's
    network label (``networks``, default ``rois.table['network']``: the source
    atlas's division, see :func:`conninfpy.interpret.annotate.annotate_rois`). By default
    each ROI is placed in the unit holding most of its tissue
    (``assign='dominant'``), so every chord is a whole number of connections
    and none is dropped. ``assign='weighted'`` spreads each ROI over the units
    its support covers; that is faithful to mixed ROIs, but splits a
    connection between large parcels into many fractional chords, most below
    ``min_weight`` (the footer gives the share of connections drawn).

    A chord joins two (unit, network) slots; its colour runs from one
    endpoint's network to the other's, and its width scales with the number of
    detected connections (``weight='count'``) or detected / tested
    (``'rate'``). Chords below ``min_weight`` connections are not drawn.

    Descriptive only: the result's correction and inference unit are unchanged.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Wedge

    from .interpret.annotate import _unit_weights, aggregate_connections

    if tail not in ("positive", "negative"):
        raise ValueError("tail must be 'positive' or 'negative' (draw one connectogram per tail).")
    if weight not in ("count", "rate"):
        raise ValueError("weight must be 'count' or 'rate'.")
    if labels not in ("gyrus", "subregion", "none"):
        raise ValueError("labels must be 'gyrus', 'subregion' or 'none'.")
    regions = rois.reference.regions
    lay = _connectogram_layout(regions)
    nets = [str(x) for x in (rois.grouping("network") if networks is None else networks)]
    W, units = _unit_weights(rois, level, assign)
    agg = aggregate_connections(result, rois, level=level, tail=tail, alpha=alpha, assign=assign,
                                networks=nets, param_idx=param_idx)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")

    # rings: subregions (sampled or not) and gyri
    sampled = np.nan_to_num(rois.labelled_weights(), nan=0.0).sum(axis=0) >= 0.1
    for rid, r in lay.iterrows():
        a0, a1 = sorted((_side_angle(r.s0, r.hemisphere), _side_angle(r.s1, r.hemisphere)))
        on = sampled[int(rid) - 1]
        ax.add_patch(Wedge((0, 0), 1.0, a0, a1, width=0.045, lw=0.3,
                           facecolor=_LOBES[r.lobe][1] if on else "white",
                           edgecolor="white" if on else "#C8C8C8"))
    gyri = lay.groupby(["gyrus", "hemisphere"], sort=False).agg(s0=("s0", "min"), s1=("s1", "max"),
                                                                lobe=("lobe", "first"))
    for k, ((g, h), r) in enumerate(gyri.sort_values("s0").iterrows()):
        a0, a1 = sorted((_side_angle(r.s0, h), _side_angle(r.s1, h)))
        ax.add_patch(Wedge((0, 0), 0.945, a0, a1, width=0.035, lw=0.4, edgecolor="white",
                           facecolor="#4D4D4D" if k % 2 else "#8A8A8A"))

    # network ring at the units of `level`; chords attach to each network's share
    key = lay["name"] if level == "subregion" else lay[level] + "_" + lay["hemisphere"]
    spans = lay.assign(unit=key.to_numpy()).groupby("unit").agg(s0=("s0", "min"), s1=("s1", "max"),
                                                                 hemisphere=("hemisphere", "first"))
    net_order = [n for n in _YEO7_COLORS] + [n for n in dict.fromkeys(map(_network_key, nets))
                                             if n not in _YEO7_COLORS]
    keyed = np.array([_network_key(x) for x in nets])
    attach = {}
    for u_idx, u in enumerate(units):
        mass = {n: float(W[keyed == n, u_idx].sum()) for n in net_order}
        tot = sum(mass.values())
        if tot <= 0 or u not in spans.index:
            continue
        s0, s1, h = spans.at[u, "s0"], spans.at[u, "s1"], spans.at[u, "hemisphere"]
        pos = s0
        for n in net_order:
            if mass[n] <= 0:
                continue
            width = (s1 - s0) * mass[n] / tot
            a0, a1 = sorted((_side_angle(pos, h), _side_angle(pos + width, h)))
            ax.add_patch(Wedge((0, 0), 0.895, a0, a1, width=0.03, lw=0,
                               facecolor=_YEO7_COLORS.get(n, "#8C8C8C")))
            attach[(u, n)] = (pos + width / 2, h, width)
            pos += width

    # chords
    links = agg[agg["detected"] >= min_weight].copy()
    links["value"] = links["detected"] if weight == "count" else links["rate"]
    if max_links is not None:
        links = links.nlargest(int(max_links), "value")
    links = links.sort_values("value")
    r_in, t = 0.86, np.linspace(0.0, 1.0, 60)[:, None]

    def xy(s, h):
        a = np.deg2rad(_side_angle(s, h))
        return np.array([r_in * np.cos(a), r_in * np.sin(a)])

    segs, cols, lws = [], [], []
    vmax = float(links["value"].max()) if len(links) else 1.0
    for row in links.itertuples(index=False):
        na, nb = _network_key(row.network_a), _network_key(row.network_b)
        sa, ha, wa = attach[(row.unit_a, na)]
        sb, hb, wb = attach[(row.unit_b, nb)]
        if (row.unit_a, na) == (row.unit_b, nb):        # within one slot: a small inward loop
            p0, p2 = xy(sa - wa / 4, ha), xy(sa + wa / 4, ha)
            c = 0.7 * xy(sa, ha)
        else:
            p0, p2 = xy(sa, ha), xy(sb, hb)
            c = (p0 + p2) / 2 * (0.15 + 0.6 * (1 - np.linalg.norm(p2 - p0) / (2 * r_in)))
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2
        f = row.value / vmax
        ca, cb = np.array(to_rgba(_YEO7_COLORS.get(na, "#8C8C8C"))), np.array(to_rgba(_YEO7_COLORS.get(nb, "#8C8C8C")))
        mix = 1 / (1 + np.exp(-14 * (t[:-1] - 0.5)))   # each half keeps its endpoint's colour
        grad = (1 - mix) * ca + mix * cb
        grad[:, 3] = 0.3 + 0.6 * f
        segs.extend(np.stack([pts[:-1], pts[1:]], axis=1))
        cols.extend(grad)
        lws.extend([0.4 + 5.6 * f] * (len(t) - 1))
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="butt", joinstyle="round"))

    # labels
    def radial(text, s, h, r, size, **kw):
        a = _side_angle(s, h)
        ax.text(r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a)), text, fontsize=size,
                rotation=a if h == "R" else a - 180, rotation_mode="anchor",
                ha="left" if h == "R" else "right", va="center", **kw)

    r_gyrus = 1.015
    if labels == "subregion":
        for _, r in lay.iterrows():
            radial(r["name"].rsplit("_", 1)[0], (r.s0 + r.s1) / 2, r.hemisphere, 1.015, 3.2)
        r_gyrus = 1.16
    if labels != "none":
        for (g, h), r in gyri.iterrows():
            radial(g, (r.s0 + r.s1) / 2, h, r_gyrus, 6.0, color="#333333")
    r_lobe = {"gyrus": 1.17, "subregion": 1.33, "none": 1.08}[labels]
    lobes = lay.groupby(["lobe", "hemisphere"], sort=False).agg(s0=("s0", "min"), s1=("s1", "max"))
    for (lobe, h), r in lobes.iterrows():
        a = np.deg2rad(_side_angle((r.s0 + r.s1) / 2, h))
        ax.text(r_lobe * np.cos(a), r_lobe * np.sin(a), f"{_LOBES[lobe][0]}-{h}", fontsize=8.5,
                ha="center", va="center", color="#333333", fontweight="bold")
    lim = r_lobe + 0.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - 0.06, lim)

    # title and legends
    design = getattr(result, "design", None) or {}
    comp = design.get("comparison")
    if title is None:
        n_det = int((result.significant_edges(alpha=alpha, tail=tail, param_idx=param_idx).shape[0]))
        if comp:
            a, b = comp
            title = f"{b} > {a}" if tail == "positive" else f"{a} > {b}"
        else:
            title = f"{tail} tail"
        title = f"{title}: {n_det} connections"
    ax.set_title(title, fontsize=11, pad=2)
    present = [n for n in net_order if any(k[1] == n for k in attach)]
    leg = ax.legend([Line2D([0], [0], color=_YEO7_COLORS.get(n, "#8C8C8C"), lw=3) for n in present], present,
                    loc="upper left", fontsize=6.5, frameon=False, title="network", title_fontsize=7)
    ax.add_artist(leg)
    if len(links):
        if weight == "count":
            ticks = sorted({max(1, round(v)) for v in (vmax, vmax / 2, vmax / 4)}, reverse=True)
            fmt = lambda v: f"{v:g} connection{'s' if v != 1 else ''}"   # noqa: E731
        else:
            ticks, fmt = [vmax, vmax / 2, vmax / 4], (lambda v: f"rate {v:.2f}")
        ax.legend([Line2D([0], [0], color="#555555", lw=0.4 + 5.6 * v / vmax) for v in ticks],
                  [fmt(v) for v in ticks], loc="upper right", fontsize=6.5, frameon=False)
    total = float(agg["detected"].sum())
    shown = float(links["detected"].sum()) if len(links) else 0.0
    how = "placed in the unit holding most of it" if assign == "dominant" else "spread over the units it covers"
    ax.text(0, -lim - 0.03, f"Each of {len(rois)} ROIs {how} (Brainnetome {level}); {len(links)} chords ≥ "
            f"{min_weight:g} connection carry {shown:.0f} of {total:.0f} connections ({shown / total:.0%})"
            if total else f"Each of {len(rois)} ROIs {how} (Brainnetome {level}); no detected connections",
            ha="center", va="center", fontsize=6.5, color="#555555")
    return ax
