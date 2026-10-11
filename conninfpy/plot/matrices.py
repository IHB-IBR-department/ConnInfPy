"""Matrices plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import numpy.typing as npt

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as exc:  # pragma: no cover (env-specific)
    raise ImportError(
        "conninfpy.plot requires matplotlib; install via `pip install matplotlib`."
    ) from exc


from conninfpy.results import InferenceResult, OmnibusInferenceResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from conninfpy.atlas.metadata import AtlasInfo


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
        cbar.ax.axhline(ref_line, color="red", linewidth=0.6, linestyle="--")
    if show_significant:
        sig = (p_map <= alpha) & np.isfinite(p_map)
        np.fill_diagonal(sig, False)
        if sig.any():
            ax.contour(sig.astype(float), levels=[0.5], colors="red", linewidths=0.4)
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
    ax.hist(
        null_max,
        bins=bins,
        density=True,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.3,
        label=f"empirical null (J={len(null_max)})",
    )
    u_emp = float(np.quantile(null_max, threshold_quantile))
    ax.axvline(
        u_emp,
        color="grey",
        linestyle="--",
        linewidth=0.6,
        label=f"u = {threshold_quantile:.0%}-tile",
    )
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
        ax.plot(xs, pdf, color="red", linewidth=1.3, label=f"GPD tail (ξ={xi:.2f})")
    if observed_max is not None:
        ax.axvline(
            observed_max,
            color="black",
            linewidth=0.9,
            label=f"observed max = {observed_max:.2f}",
        )
    ax.set_xlabel("max statistic across edges")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


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
        raise ValueError(f"effect matrix must be square; got {effect.shape}.")
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
        ax.contour(sig.astype(float), levels=[0.5], colors="black", linewidths=0.5)

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
