"""Tests for the descriptive coupling-class plot (NETSTAT-49).

The classes are read from :meth:`InferenceResult.coupling` only; the plot
must not reclassify, and homotopy markers must come from BNA metadata
alone. Results are small synthetic objects carrying beta / design directly,
so no permutation run is needed.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pytest
from matplotlib.axes import Axes

from conninfpy import AtlasInfo, InferenceResult
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.plot.coupling import (
    COUPLING_CLASS_CODES,
    MISSING_LABEL,
    plot_coupling_classes,
)
from conninfpy.stats._estimates import COUPLING_CLASSES

#: Synthetic per-edge predicted z at profiles A and B (a, b):
#: (i, j, z_A, z_B). Cells not listed are predicted 0 at both -> "no change".
_EDGES = [
    (0, 1, 0.1, 0.9),    # stronger positive coupling
    (0, 2, 0.9, 0.6),    # weaker positive coupling
    (1, 2, -0.2, -0.8),  # stronger anticorrelation
    (1, 3, -1.1, -0.65),  # weaker anticorrelation
    (2, 3, 0.7, -0.3),   # sign change: positive to negative
    (0, 3, -0.7, 0.3),   # sign change: negative to positive
]

_DESIGN = {
    "kind": "two-sample",
    "columns": ["c0", "c1"],
    "contrast": [0.0, 1.0],
    "profiles": {"A": [1.0, -0.5], "B": [1.0, 0.5]},
    "comparison": ["A", "B"],
    "scale": "fisher_z",
    "n_subjects": 12,
}


def _synthetic_result(
    N: int = 6,
    *,
    method: str = "tstat",
    edges=None,
    nan_edges=(),
    p_sig_edges=(),
):
    """InferenceResult with coefficient maps that produce known classes."""
    edges = _EDGES if edges is None else edges
    base = np.zeros((N, N))
    diff = np.zeros((N, N))
    for i, j, za, zb in edges:
        base[i, j] = base[j, i] = 0.5 * (za + zb)
        diff[i, j] = diff[j, i] = zb - za
    beta = np.stack([base, diff])
    for i, j in nan_edges:
        beta[:, i, j] = np.nan
        beta[:, j, i] = np.nan
    p = np.full((N, N), 0.5)
    for i, j in p_sig_edges:
        p[i, j] = p[j, i] = 0.001
    stat = np.where(p <= 0.05, 4.0, 0.0)
    return InferenceResult(
        p,
        p.copy(),
        method=method,
        n_permutations=20,
        stat_positive=stat,
        stat_negative=np.zeros_like(stat),
        beta=beta,
        design=dict(_DESIGN),
    )


def _codes(ax: Axes) -> np.ndarray:
    """The plotted categorical matrix as float codes (-1 missing, nan diagonal)."""
    return np.ma.filled(np.ma.asarray(ax.images[0].get_array()).astype(float), np.nan)


def _legend(ax: Axes):
    legend = ax.get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


def _markers(ax: Axes):
    """Matrix-index points carrying the homotopy marker."""
    out = []
    for coll in ax.collections:
        if coll.get_label() == "homotopy pair (BNA homolog_id)":
            offsets = np.ma.asarray(coll.get_offsets())
            out.extend((int(round(x)), int(round(y))) for x, y in offsets)
    return out


def _atlas(ids, n: int) -> AtlasInfo:
    ids = np.asarray(ids, dtype=np.int64)
    assert len(ids) == n
    return AtlasInfo(
        labels=[f"ROI{k}" for k in range(n)],
        networks=["Net"] * n,
        label_ids=ids,
    )


# ---------------------------------------------------------------------------
# 1. codes / legend follow the exact COUPLING_CLASSES strings
# ---------------------------------------------------------------------------

def test_matrix_layout_matches_coupling_classes_and_legend():
    res = _synthetic_result()
    ax = plot_coupling_classes(res)
    assert isinstance(ax, Axes)
    codes = _codes(ax)
    classes = res.coupling()["classes"]
    N = codes.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j:
                assert np.isnan(codes[i, j])  # diagonal masked
                continue
            if classes[i, j] is None:
                assert codes[i, j] == -1
            else:
                assert COUPLING_CLASSES[int(codes[i, j])] == classes[i, j]

    present = {classes[i, j] for i in range(N) for j in range(N)
               if i != j and classes[i, j] is not None}
    assert len(present) >= 5  # the fixture really spans several classes
    legend = _legend(ax)
    assert [name for name in COUPLING_CLASSES if name in legend] == [
        name for name in COUPLING_CLASSES if name in present
    ]
    assert set(legend) & set(COUPLING_CLASSES) == present


# ---------------------------------------------------------------------------
# 2. NaN in a / b -> neutral "missing" category, no exception
# ---------------------------------------------------------------------------

def test_nan_cells_become_missing_category():
    res = _synthetic_result(nan_edges=[(0, 1)])
    assert res.coupling()["classes"][0, 1] is None
    ax = plot_coupling_classes(res)
    codes = _codes(ax)
    assert codes[0, 1] == -1 and codes[1, 0] == -1
    assert MISSING_LABEL in _legend(ax)
    assert np.isfinite(codes[0, 2])  # other cells keep their class


# ---------------------------------------------------------------------------
# 3. homotopy markers come from BNA homolog_id only
# ---------------------------------------------------------------------------

def test_homotopy_from_bna_metadata_and_order_invariant():
    ref = BrainnetomeReference.load(profiles=False, resolution_mm=2)
    ids = np.array([1, 2, 3, 4, 20, 24])  # (1,2) and (3,4) are BNA homologues
    res = _synthetic_result()
    ax = plot_coupling_classes(res, atlas=_atlas(ids, 6), reference=ref)
    assert ax.attrs["homology"] == "bna homolog_id"
    assert set(ax.attrs["homotopy_pairs"]) == {(0, 1), (2, 3)}
    assert set(_markers(ax)) == {(0, 1), (1, 0), (2, 3), (3, 2)}

    # Same ROI ids in another matrix order -> the same true pairs.
    perm = [3, 0, 5, 1, 4, 2]
    ids_perm = ids[perm]
    res_perm = _synthetic_result(edges=[])
    ax_perm = plot_coupling_classes(res_perm, atlas=_atlas(ids_perm, 6), reference=ref)
    pairs_as_ids = {frozenset((int(ids_perm[i]), int(ids_perm[j])))
                    for i, j in ax_perm.attrs["homotopy_pairs"]}
    assert pairs_as_ids == {frozenset({1, 2}), frozenset({3, 4})}
    assert set(_markers(ax_perm)) == {
        (i, j) for i, j in ax_perm.attrs["homotopy_pairs"]
    } | {(j, i) for i, j in ax_perm.attrs["homotopy_pairs"]}

    # Not every pair is marked: only the real homologues.
    assert len(ax_perm.attrs["homotopy_pairs"]) == 2


# ---------------------------------------------------------------------------
# 4. unknown homology is stated, never guessed
# ---------------------------------------------------------------------------

def test_unknown_homology_is_explicit():
    ref = BrainnetomeReference.load(profiles=False, resolution_mm=2)
    res = _synthetic_result()
    ax = plot_coupling_classes(res, atlas=_atlas([1, 2, 3, 4, 20, 24], 6))
    assert ax.attrs["homology"] == "unknown"
    assert ax.attrs["homotopy_pairs"] == []
    assert _markers(ax) == []
    assert any("unknown" in text for text in _legend(ax))

    # An id absent from reference.regions is unknown for that ROI only.
    ids = np.array([1, 2, 900, 901, 4, 3])
    ax_ids = plot_coupling_classes(res, atlas=_atlas(ids, 6), reference=ref)
    assert ax_ids.attrs["homology"] == "bna homolog_id"
    assert set(ax_ids.attrs["homology_unknown_rois"]) == {2, 3}
    assert set(ax_ids.attrs["homotopy_pairs"]) == {(0, 1), (4, 5)}
    assert any("homology unknown" in text for text in _legend(ax_ids))
    marker_points = set(_markers(ax_ids))
    assert marker_points and not any(2 in p or 3 in p for p in marker_points)


# ---------------------------------------------------------------------------
# 5. inference unit drives the wording of the alpha overlay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method, unit", [("nbs", "component"), ("cnbs", "block")])
def test_unit_specific_overlay_wording(method, unit):
    res = _synthetic_result(method=method, p_sig_edges=[(2, 3)])
    ax = plot_coupling_classes(res, alpha=0.05)
    assert ax.attrs["inference_unit"] == unit
    assert ax.attrs["descriptive"] is True
    assert ax.attrs["applies_to"] == f"edges of the detected {unit}s"
    legend = _legend(ax)
    assert any(f"edges of detected {unit}s" in text for text in legend)
    assert not any("significant edges" in text for text in legend)
    assert any("α=0.05" in text for text in legend)


def test_edge_unit_overlay_says_significant_edges():
    res = _synthetic_result(p_sig_edges=[(2, 3)])
    ax = plot_coupling_classes(res, alpha=0.05)
    assert ax.attrs["inference_unit"] == "edge"
    assert ax.attrs["applies_to"] == "edge"
    legend = _legend(ax)
    assert any(text.startswith("significant edges (") for text in legend)
    assert not any("detected" in text for text in legend)
    # works with an existing axis and keeps its identity
    import matplotlib.pyplot as plt

    fig, existing = plt.subplots()
    assert plot_coupling_classes(res, ax=existing) is existing
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. no design / comparison -> ValueError from coupling()
# ---------------------------------------------------------------------------

def test_result_without_design_raises():
    N = 6
    res = InferenceResult(
        np.full((N, N), 0.5),
        np.full((N, N), 0.5),
        method="tstat",
        n_permutations=20,
    )
    with pytest.raises(ValueError):
        plot_coupling_classes(res)


# ---------------------------------------------------------------------------
# 7. legend class rows are exactly the classes of the plotted edges
# ---------------------------------------------------------------------------

def test_legend_rows_equal_class_values():
    res = _synthetic_result()
    c = res.coupling()
    values = {c["classes"][i, j] for i in range(6) for j in range(6)
              if i != j and c["classes"][i, j] is not None}
    legend_classes = [text for text in _legend(plot_coupling_classes(res))
                      if text in COUPLING_CLASSES]
    assert legend_classes and set(legend_classes) == values
    # every class row is one of the exact strings and maps to its own colour code
    for text in legend_classes:
        assert COUPLING_CLASS_CODES[text] == COUPLING_CLASSES.index(text)
    assert not any(text.lower().startswith(("p ", "p<", "significan"))
                   for text in legend_classes)
