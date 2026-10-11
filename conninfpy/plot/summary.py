"""Summary plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover (env-specific)
    raise ImportError(
        "conninfpy.plot requires matplotlib; install via `pip install matplotlib`."
    ) from exc


from conninfpy.results import InferenceResult, OmnibusInferenceResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from conninfpy.atlas.metadata import AtlasInfo


from conninfpy.plot.matrices import plot_effect_matrix, plot_p_map
from conninfpy.plot.networks import plot_block_mass, plot_network_summary


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
        plot_network_summary(result, atlas=atlas, alpha=alpha, ax=axes[0, 1])
    else:
        # Fall back to the existing block_mass helper with a single-block
        # partition; not very informative but keeps the layout consistent.
        if isinstance(result, InferenceResult):
            p_for_block = np.minimum(result["positive"], result["negative"])
        else:
            p_for_block = result["omnibus"]
        N = p_for_block.shape[0]
        plot_block_mass(
            p_for_block,
            np.zeros(N, dtype=int),
            network_names=["all"],
            alpha=alpha,
            ax=axes[0, 1],
            title="block mass (no atlas)",
        )

    # Bottom-left: p-map
    if isinstance(result, InferenceResult):
        plot_p_map(
            result["positive"], alpha=alpha, ax=axes[1, 0], title="positive-tail p-map"
        )
    else:
        plot_p_map(result["omnibus"], alpha=alpha, ax=axes[1, 0], title="omnibus p-map")

    # Bottom-right: top-k edge table
    ax_t = axes[1, 1]
    ax_t.axis("off")
    try:
        df = result.significant_edges(atlas=atlas, alpha=alpha, sort="p", top_k=top_k)
        if df.empty:
            ax_t.text(
                0.5,
                0.5,
                f"No significant edges at α = {alpha:.3g}",
                ha="center",
                va="center",
                fontsize=10,
            )
        else:
            # Pick a small, readable subset of columns.
            if atlas is not None:
                if isinstance(result, OmnibusInferenceResult):
                    cols = [
                        "roi_i_name",
                        "roi_j_name",
                        "network_pair",
                        "F",
                        "p_omnibus",
                    ]
                else:
                    cols = [
                        "roi_i_name",
                        "roi_j_name",
                        "network_pair",
                        "t_signed",
                        "p_min",
                        "tail",
                    ]
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
            ax_t.set_title(f"top {len(disp)} significant edges", fontsize=9, loc="left")
    except (ValueError, TypeError) as exc:
        # Stat maps missing or other export-time error — show the message
        # so the figure still composes cleanly.
        ax_t.text(
            0.5,
            0.5,
            f"(edge table unavailable: {exc})",
            ha="center",
            va="center",
            fontsize=8,
            wrap=True,
        )

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
