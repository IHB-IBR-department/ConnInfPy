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
them, so headless environments (CI, batch jobs) don't pull matplotlib."""

from conninfpy.plot.connectograms import plot_connectogram
from conninfpy.plot.matrices import (
    plot_effect_matrix,
    plot_null_max_distribution,
    plot_p_map,
)
from conninfpy.plot.networks import (
    plot_block_mass,
    plot_connectome_graph,
    plot_network_summary,
)
from conninfpy.plot.summary import summary_figure

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


def __getattr__(name):
    from conninfpy.plot import connectograms, matrices, networks, summary

    for module in (matrices, networks, connectograms, summary):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
