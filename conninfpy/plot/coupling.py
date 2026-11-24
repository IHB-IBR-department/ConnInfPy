"""Descriptive coupling-change class matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover (env-specific)
    raise ImportError(
        "conninfpy.plot requires matplotlib; install via `pip install matplotlib`."
    ) from exc

from conninfpy.stats._estimates import COUPLING_CLASSES

#: One fixed colour per exact string of :data:`COUPLING_CLASSES`.
COUPLING_CLASS_COLORS: Dict[str, str] = {
    "stronger positive coupling": "#2166ac",
    "weaker positive coupling": "#9ecae1",
    "stronger anticorrelation": "#b2182b",
    "weaker anticorrelation": "#f4a582",
    "sign change: positive to negative": "#762a83",
    "sign change: negative to positive": "#e08214",
    "emerges from ~0": "#1b7837",
    "returns to ~0": "#a6dba0",
    "no change": "#d9d9d9",
}

#: Label of the neutral category for cells without a class (NaN in ``a`` or ``b``).
MISSING_LABEL = "missing"
#: Colour of the neutral category.
MISSING_COLOR = "#7f7f7f"

#: Integer code of each class inside the plotted matrix; ``-1`` is :data:`MISSING_LABEL`.
COUPLING_CLASS_CODES: Dict[str, int] = {
    name: code for code, name in enumerate(COUPLING_CLASSES)
}

if TYPE_CHECKING:  # pragma: no cover - typing only
    from conninfpy.atlas.brainnetome import BrainnetomeReference
    from conninfpy.atlas.metadata import AtlasInfo


def _class_codes(classes: npt.NDArray[np.object_]) -> npt.NDArray[np.float64]:
    """Integer code per cell (``-1`` = missing, ``nan`` = diagonal)."""
    codes = np.full(classes.shape, -1.0, dtype=np.float64)
    for code, name in enumerate(COUPLING_CLASSES):
        codes[classes == name] = float(code)
    return codes


def _homotopy_pairs(
    label_ids: npt.NDArray[np.int64],
    reference: "BrainnetomeReference",
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Matrix-index pairs ``(i, j)`` that are BNA homologues, and unknown ROIs.

    A pair is homologous iff the two BNA ids differ and the hierarchy's
    ``homolog_id`` of one is the other. Ids absent from ``reference.regions``
    are reported as unknown; nothing is inferred from ROI order or names.
    """
    regions = reference.regions
    known_ids = set(int(v) for v in getattr(regions, "index", []))
    homolog: Dict[int, int] = {}
    unknown: List[int] = []
    for k, roi_id in enumerate(int(v) for v in label_ids):
        if roi_id not in known_ids:
            unknown.append(k)
            continue
        homolog[roi_id] = int(regions.at[roi_id, "homolog_id"])
    pairs: List[Tuple[int, int]] = []
    n = len(label_ids)
    for i in range(n):
        li = int(label_ids[i])
        for j in range(i + 1, n):
            lj = int(label_ids[j])
            if li == lj or li not in homolog or lj not in homolog:
                continue
            if homolog[li] == lj:
                pairs.append((i, j))
    return pairs, unknown


def plot_coupling_classes(
    result: Any,
    *,
    profiles: Optional[Sequence[str]] = None,
    near_zero: float = 0.0,
    atlas: Optional["AtlasInfo"] = None,
    reference: Optional["BrainnetomeReference"] = None,
    alpha: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Axes:
    """Categorical heatmap of how coupling changes between two profiles.

    The matrix is descriptive and covers every cell whose two predicted
    profiles are both available: the class of a cell is the class of its
    ``a`` → ``b`` change at the recorded scale, computed from
    ``result.predicted()`` for the two profiles and taken verbatim from
    :meth:`conninfpy.InferenceResult.coupling` (see
    :data:`conninfpy.stats._estimates.COUPLING_CLASSES`) — this helper
    classifies nothing itself. A NaN in either predicted profile leaves the
    cell without a class, drawn as the neutral ``'missing'`` category. The
    plot is symmetric and drawn in full; the diagonal is masked. Classes are
    descriptive labels on the recorded scale, not evidence of significance,
    and the colour matrix stays fully descriptive at any ``alpha``.

    Homotopy pairs are marked from BNA metadata only: with an atlas that
    carries ``label_ids`` (BNA ``roi_id`` per matrix row) and a
    :class:`~conninfpy.atlas.brainnetome.BrainnetomeReference``, a cell is
    marked when the two ids are each other's ``homolog_id``. Without those
    metadata all pairs are unknown and the legend says so; an id absent from
    ``reference.regions`` makes its pairs unknown as well. Homology is never
    inferred from ROI order, symmetry or name suffixes.

    Parameters
    ----------
    result : InferenceResult
        Result carrying coefficient maps and a design with a ``comparison``
        (see :meth:`conninfpy.InferenceResult.coupling`).
    profiles : sequence of str, optional
        Two profile names, forwarded to ``result.coupling``. Default: the
        design's recorded comparison.
    near_zero : float, default 0.0
        Prespecified "~0" band forwarded to ``result.coupling``; it decides
        the emerges-from / returns-to classes. The plot introduces no band
        of its own.
    atlas : :class:`~conninfpy.atlas.metadata.AtlasInfo`, optional
        Supplies ``label_ids`` (BNA ROI id per matrix row) and labels for
        the axes. Its length must match the matrix.
    reference : :class:`~conninfpy.atlas.brainnetome.BrainnetomeReference`, optional
        Source of ``homolog_id``. Requires ``atlas.label_ids``; otherwise
        every pair is unknown.
    alpha : float, optional
        If given, outline the edges returned by
        ``result.significant_edges(alpha=alpha, tail='both')``. For an
        edge-level result these are significant edges; for ``nbs`` /
        ``cnbs`` results they are the edges of the detected components /
        blocks, and the legend says exactly that. The contour bounds the
        overlay only; the colour matrix keeps its descriptive meaning. No
        new threshold, p-value or edge discovery is introduced here, and the
        contour is not a joint detection claim: as noted for
        :meth:`InferenceResult.significant_edges`, with separate one-sided
        families filtering both tails at ``alpha`` does not imply joint FWER
        at ``alpha``.
    ax : matplotlib Axes, optional
        Plot into this axis; otherwise a new ``(5.6, 5.0)`` figure is
        created.
    title : str, optional
        Axis title. Defaults to the compared profiles.

    Returns
    -------
    matplotlib.axes.Axes
        The axis, carrying ``attrs``:

        ``descriptive`` (True), ``inference_unit`` (``result.inference_unit``),
        ``applies_to`` (always the scope of the colour matrix:
        ``'descriptive coupling classes of cells with available predicted
        profiles'``), ``contour`` (``None`` when ``alpha`` is ``None`` -
        no contour is drawn; otherwise a dict with ``'applies_to'``
        (``'significant edges'`` for edge results, else ``'edges of the
        detected components|blocks'``), ``'alpha'`` (float) and ``'label'``
        (the legend text of the overlay)), ``homology`` (``'bna homolog_id'``
        or ``'unknown'``), and the descriptive extras ``homotopy_pairs``,
        ``homology_unknown_rois``, ``profiles``, ``near_zero``, ``alpha``.

        The legend lists the class strings present in the matrix in
        :data:`COUPLING_CLASSES` order — the same strings as the
        ``coupling_change`` column of :meth:`InferenceResult.annotated_edges`
        — plus ``'missing'`` for NaN cells, the homotopy marker and, where
        applicable, an explicit "homology unknown" entry.

    Raises
    ------
    ValueError
        If the result carries no design / comparison (raised by
        :meth:`InferenceResult.coupling`), or if ``atlas`` / ``label_ids``
        length does not match the matrix.

    Notes
    -----
    Interpretation follows :attr:`InferenceResult.inference_unit`: with
    ``cnbs`` (blocks) or ``nbs`` (components) the ``alpha`` overlay shows
    the edges *inside* detected units, never edge-level significance.
    Negative baselines and the classes derived from them depend on
    preprocessing and carry no physiological or causal meaning on their own.
    """
    coupling = result.coupling(profiles, near_zero=near_zero)
    classes = np.asarray(coupling["classes"], dtype=object)
    if classes.ndim != 2 or classes.shape[0] != classes.shape[1]:
        raise ValueError(f"coupling classes must be square; got {classes.shape}.")
    N = classes.shape[0]
    names = list(coupling["profiles"])

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.6, 5.0))

    codes = _class_codes(classes)
    np.fill_diagonal(codes, np.nan)

    # Legend entries follow the classes actually present off the diagonal.
    off_diagonal = ~np.eye(N, dtype=bool)
    present = [
        name
        for name in COUPLING_CLASSES
        if bool(np.any((classes == name) & off_diagonal))
    ]
    has_missing = bool(np.any((codes[off_diagonal] < 0)))

    colors = [MISSING_COLOR] + [COUPLING_CLASS_COLORS[name] for name in COUPLING_CLASSES]
    cmap = ListedColormap(colors)
    if hasattr(cmap, "with_extremes"):  # masked diagonal reads as background
        cmap = cmap.with_extremes(bad="white")
    bounds = np.arange(-1.5, len(COUPLING_CLASSES) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(np.ma.masked_invalid(codes), cmap=cmap, norm=norm, interpolation="nearest")

    if atlas is not None:
        if len(atlas) != N:
            raise ValueError(
                f"atlas has {len(atlas)} ROIs but the result is on {N} nodes."
            )
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        tick_font = 6 if N > 12 else 8
        ax.set_xticklabels(list(atlas.labels), rotation=90, fontsize=tick_font)
        ax.set_yticklabels(list(atlas.labels), fontsize=tick_font)

    # ---------------------------------------------------------------- overlay
    unit = str(result.inference_unit)
    contour: Optional[Dict[str, Any]] = None
    overlay_label = None
    if alpha is not None:
        overlay_label = (
            f"significant edges (α={alpha:g})"
            if unit == "edge"
            else f"edges of detected {unit}s (α={alpha:g})"
        )
        contour = {
            "applies_to": (
                "significant edges"
                if unit == "edge"
                else f"edges of the detected {unit}s"
            ),
            "alpha": float(alpha),
            "label": overlay_label,
        }
        edges = result.significant_edges(alpha=float(alpha), tail="both")
        overlay = np.zeros((N, N), dtype=bool)
        if len(edges):
            i_idx = np.asarray(edges["roi_i"], dtype=int)
            j_idx = np.asarray(edges["roi_j"], dtype=int)
            overlay[i_idx, j_idx] = True
            overlay[j_idx, i_idx] = True
        np.fill_diagonal(overlay, False)
        if overlay.any():
            ax.contour(overlay.astype(float), levels=[0.5], colors="black", linewidths=0.6)

    # --------------------------------------------------------------- homology
    label_ids = None if atlas is None else getattr(atlas, "label_ids", None)
    homology = "unknown"
    pairs: List[Tuple[int, int]] = []
    unknown_rois: List[int] = []
    if label_ids is not None and reference is not None:
        label_ids = np.asarray(label_ids, dtype=np.int64)
        if label_ids.shape != (N,):
            raise ValueError(
                f"atlas.label_ids has shape {label_ids.shape}; expected ({N},)."
            )
        homology = "bna homolog_id"
        pairs, unknown_rois = _homotopy_pairs(label_ids, reference)
    if pairs:
        xs, ys = [], []
        for i, j in pairs:
            xs.extend([j, i])
            ys.extend([i, j])
        ax.scatter(
            xs,
            ys,
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            label="homotopy pair (BNA homolog_id)",
        )

    # ---------------------------------------------------------------- legend
    handles: List[Any] = [
        Patch(facecolor=COUPLING_CLASS_COLORS[name], edgecolor="0.4", label=name)
        for name in present
    ]
    if has_missing:
        handles.append(
            Patch(facecolor=MISSING_COLOR, edgecolor="0.4", label=MISSING_LABEL)
        )
    if pairs:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="black",
                label="homotopy pair (BNA homolog_id)",
            )
        )
    if homology == "unknown":
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="0.6",
                label="homotopy pairs: unknown (no BNA metadata)",
            )
        )
    elif unknown_rois:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                label=f"homology unknown for {len(unknown_rois)} ROI(s)",
            )
        )
    if alpha is not None:
        handles.append(Line2D([], [], color="black", linewidth=0.8, label=overlay_label))
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            frameon=False,
            title=(
                "descriptive labels, not significance\n"
                f"inference unit: {unit}\n"
                f"homology: {homology}"
            ),
            title_fontsize=7,
        )

    ax.set_xlabel("node")
    ax.set_ylabel("node")
    ax.set_title(title or f"Coupling change: {names[0]} → {names[1]}", fontsize=9)
    metadata = {
        "descriptive": True,
        "inference_unit": unit,
        "applies_to": "descriptive coupling classes of cells with available predicted profiles",
        "contour": contour,
        "homology": homology,
        "homotopy_pairs": pairs,
        "homology_unknown_rois": unknown_rois,
        "profiles": names,
        "near_zero": near_zero,
        "alpha": alpha,
    }
    if hasattr(ax, "attrs"):  # matplotlib >= ? exposes a mapping
        ax.attrs.update(metadata)
    else:
        ax.attrs = metadata
    return ax


__all__ = [
    "plot_coupling_classes",
    "COUPLING_CLASS_COLORS",
    "COUPLING_CLASS_CODES",
    "MISSING_LABEL",
    "MISSING_COLOR",
]
