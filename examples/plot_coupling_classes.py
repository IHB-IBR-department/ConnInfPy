"""Descriptive coupling-class matrix with homotopy pairs.

Two panels of the same result: with BNA metadata (known homotopy pairs plus
ROIs whose ids are not in the reference) and without any reference (all
pairs unknown). Cells are the descriptive classes of ``result.coupling()``;
a few edges are simulated as untested (NaN) to show the neutral category.

Usage::

    python3 examples/plot_coupling_classes.py [output.png]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from conninfpy import AtlasInfo, InferenceResult, analyze
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.plot import plot_coupling_classes
from conninfpy.stats._estimates import COUPLING_CLASSES

N = 12
# BNA ids: three left/right homolog pairs, one more pair, then ids that are
# deliberately absent from the reference hierarchy.
ROI_IDS = np.array([1, 2, 3, 4, 7, 8, 11, 12, 901, 902, 903, 904])
UNTESTED = [(0, 5), (2, 7), (3, 9)]


def _synthetic_groups(seed: int = 49, n_sub: int = 12):
    """Two groups on N nodes with a few edges whose coupling changes."""
    rng = np.random.default_rng(seed)
    g1 = np.tanh(rng.normal(0.0, 0.4, (n_sub, N, N)))
    g2 = np.tanh(rng.normal(0.0, 0.4, (n_sub, N, N)))
    for g in (g1, g2):
        g[:] = (g + g.transpose(0, 2, 1)) / 2
        g[:, np.arange(N), np.arange(N)] = 1.0
    # Group 2: stronger / weaker coupling and two sign changes (Fisher z).
    changes = {
        (0, 1): (0.3, 1.1),    # stronger positive coupling
        (2, 3): (1.0, 0.4),    # weaker positive coupling
        (4, 5): (-0.4, -1.2),  # stronger anticorrelation
        (6, 7): (-1.1, -0.4),  # weaker anticorrelation
        (8, 9): (0.6, -0.5),   # sign change: positive to negative
        (10, 11): (-0.5, 0.7),  # sign change: negative to positive
    }
    for (i, j), (before, after) in changes.items():
        g1[:, i, j] = g1[:, j, i] = before + rng.normal(0.0, 0.08, n_sub)
        g2[:, i, j] = g2[:, j, i] = after + rng.normal(0.0, 0.08, n_sub)
    return g1, g2


def _atlas(reference: BrainnetomeReference) -> AtlasInfo:
    known = set(int(v) for v in reference.regions.index)
    labels, networks = [], []
    for roi_id in ROI_IDS:
        if int(roi_id) in known:
            row = reference.regions.loc[int(roi_id)]
            labels.append(str(row["name"]))
            networks.append(str(row.get("yeo7", "unknown")))
        else:
            labels.append(f"ROI{int(roi_id)} (no metadata)")
            networks.append("unknown")
    return AtlasInfo(labels=labels, networks=networks, label_ids=ROI_IDS)


def _with_untested_edges(result, edges):
    """Mark a few edges as untested (NaN), as available-case coverage does."""
    beta = np.array(result.beta, dtype=float)
    positive = np.array(result["positive"], dtype=float)
    negative = np.array(result["negative"], dtype=float)
    for i, j in edges:
        beta[:, i, j] = beta[:, j, i] = np.nan
        positive[i, j] = positive[j, i] = np.nan
        negative[i, j] = negative[j, i] = np.nan
    return InferenceResult(
        positive=positive,
        negative=negative,
        method=result.method,
        n_permutations=result.n_permutations,
        stat_positive=result.stat_positive,
        stat_negative=result.stat_negative,
        stat_type=result.stat_type,
        beta=beta,
        design=result.design,
    )


def _print_summary(classes) -> None:
    off_diagonal = classes[~np.eye(len(classes), dtype=bool)]
    values = [c for c in off_diagonal if c is not None]
    print(f"classified edges: {len(values)} of {off_diagonal.size} off-diagonal cells")
    for name in COUPLING_CLASSES:
        count = sum(1 for c in values if c == name)
        if count:
            print(f"  {name}: {count}")
    missing = sum(1 for c in off_diagonal if c is None)
    print(f"  missing (NaN): {missing}")


def main() -> None:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(tempfile.mkdtemp()) / "coupling_classes.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    reference = BrainnetomeReference.load(profiles=False, resolution_mm=2)
    atlas = _atlas(reference)
    g1, g2 = _synthetic_groups()
    analyzed = analyze(
        group1=g1,
        group2=g2,
        test_type="two-sample",
        input="z",
        r_to_z=False,
        method="tstat",
        n_permutations=30,
        acceleration=None,
        use_mp=False,
        rng=0,
        atlas=atlas,
    )
    result = _with_untested_edges(analyzed.inference, UNTESTED)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    plot_coupling_classes(
        result,
        atlas=atlas,
        reference=reference,
        ax=axes[0],
        title="with BNA metadata",
    )
    plot_coupling_classes(
        result,
        atlas=atlas,
        ax=axes[1],
        title="without reference: homology unknown",
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.55)  # room for the legends outside the axes
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pairs = axes[0].attrs["homotopy_pairs"]
    unknown = axes[0].attrs["homology_unknown_rois"]
    print(f"homotopy pairs: {len(pairs)} (BNA homolog_id)")
    print(f"ROIs without hierarchy entry: {len(unknown)}")
    _print_summary(result.coupling()["classes"])
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
