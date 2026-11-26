"""NETSTAT-52: runtime cerebellar companion layer (SUIT atl-Anatom) tests."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conninfpy.atlas.brainnetome import BrainnetomeReference, N_REGIONS
from conninfpy.interpret.annotate import (
    annotate_edges,
    annotate_rois,
    summarize_annotation,
)
from conninfpy.plot.connectograms import _connectogram_layout, _side_angle, plot_connectogram


class _FakeResult:
    """Minimal NetworkTestResult-like stub for annotation & connectogram tests."""

    def __init__(self, n_rois: int, sig_pairs: list[tuple[int, int]]):
        self.n_rois = n_rois
        self.inference_unit = "edge"
        self.design = {"comparison": ("HC", "ASD")}
        rows = []
        sig_set = {tuple(sorted(p)) for p in sig_pairs}
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                is_sig = (i, j) in sig_set
                rows.append(
                    {
                        "roi_i": i,
                        "roi_j": j,
                        "t_signed": 3.5 if is_sig else 0.2,
                        "p_positive": 0.001 if is_sig else 0.6,
                        "p_negative": 0.999 if is_sig else 0.4,
                        "p_min": 0.001 if is_sig else 0.4,
                    }
                )
        self._edges = pd.DataFrame(rows)

    def significant_edges(self, alpha=0.05, tail="both", include_nonsig=False, param_idx=None):
        if include_nonsig:
            return self._edges.copy()
        if tail == "positive":
            return self._edges[self._edges["p_positive"] <= alpha].copy()
        if tail == "negative":
            return self._edges[self._edges["p_negative"] <= alpha].copy()
        return self._edges[self._edges["p_min"] <= alpha].copy()


from conninfpy.atlas.resources import atlas_resource_available as _atlas_resource_available

_HAS_CEREBELLAR_CACHE = all(
    _atlas_resource_available(n)
    for n in (
        "BN_Atlas_246_1mm.nii.gz",
        "bna246_hierarchy.csv",
        "bna246.csv",
        "atl-Anatom_space-MNI_dseg.nii.gz",
        "atl-Anatom_space-MNI_dseg.tsv",
    )
)
_CACHE_SKIP_REASON = "Brainnetome/SUIT atlas files not in local cache; run scripts/download_atlases.py"


@pytest.mark.skipif(not _HAS_CEREBELLAR_CACHE, reason=_CACHE_SKIP_REASON)
def test_default_cerebellum_none_preserves_bna246_bitwise():
    ref = BrainnetomeReference.load()
    assert ref.n_regions == N_REGIONS == 246
    assert list(ref.regions.index) == list(range(1, 247))
    assert (ref.regions["atlas_source"] == "BNA-246").all()
    assert "cerebellum_atlas" not in ref.provenance
    assert int(ref.labels.max()) == 246
    # Pure BNA connectogram layout has no Cerebellum sector
    lay = _connectogram_layout(ref.regions)
    assert "Cerebellum" not in set(lay["lobe"])
    assert not lay["s0"].isna().any()


@pytest.mark.skipif(not _HAS_CEREBELLAR_CACHE, reason=_CACHE_SKIP_REASON)
def test_cerebellum_anatom_composite_and_metadata():
    ref_bna = BrainnetomeReference.load()
    ref = BrainnetomeReference.load(cerebellum="anatom")
    assert ref.n_regions == 280
    assert list(ref.regions.index) == list(range(1, 281))
    assert ref.provenance["clipped_voxels"] == 1033
    assert ref.provenance["cerebellar_voxels"] > 0
    assert (ref.regions.loc[1:246, "atlas_source"] == "BNA-246").all()
    assert (ref.regions.loc[247:280, "atlas_source"] == "SUIT").all()
    # Strict BNA priority: every BNA>0 voxel is bitwise identical
    bna_mask = ref_bna.labels > 0
    assert np.array_equal(ref.labels[bna_mask], ref_bna.labels[bna_mask])

    # Hemispheres & homologs (Vermis is M with no homolog)
    assert ref.regions.at[251, "name"] == "Left_VI"
    assert ref.regions.at[251, "hemisphere"] == "L"
    assert ref.homolog(251) == 253  # Right_VI
    assert ref.regions.at[252, "name"] == "Vermis_VI"
    assert ref.regions.at[252, "hemisphere"] == "M"
    assert ref.homolog(252) is None

    # BrainMap profiles explicitly unavailable for cerebellar regions
    assert ref.profile("BD", 1) is not None
    assert ref.profile("BD", 247) is None
    assert ref.profile("PC", 280) is None

    # Invalid cerebellum option or 2mm resolution raises ValueError
    with pytest.raises(ValueError):
        BrainnetomeReference.load(cerebellum="unknown")
    with pytest.raises(ValueError):
        BrainnetomeReference.load(resolution_mm=2, cerebellum="anatom")


@pytest.mark.skipif(not _HAS_CEREBELLAR_CACHE, reason=_CACHE_SKIP_REASON)
def test_annotate_rois_edges_summary_and_connectogram_with_cerebellum():
    ref = BrainnetomeReference.load(cerebellum="anatom")
    # Pick exact centroids of:
    # 0: A8m_L (cortex L, id=1)
    # 1: A8m_R (cortex R, id=2)
    # 2: Left_CrusI (cerebellum L, id=254)
    # 3: Right_CrusI (cerebellum R, id=256)
    # 4: Vermis_VI (cerebellum M, id=252)
    ids = [1, 2, 254, 256, 252]
    coords = ref.regions.loc[ids, ["x", "y", "z"]].to_numpy(float)
    rois = annotate_rois(
        coords,
        reference=ref,
        names=["Cortex_L", "Cortex_R", "CB_L", "CB_R", "Vermis"],
        support="sphere",
        radius_mm=4.0,
        min_coverage=0.1,
    )
    assert rois.weights.shape == (5, 280)
    assert rois.table.loc[2, "lobe"] == "Cerebellum"
    assert rois.table.loc[2, "hemisphere"] == "L"
    assert rois.table.loc[4, "hemisphere"] == "M"
    assert rois.table.loc[2, "profile_status"].startswith("not available for companion atlas")

    # Edges: (0, 2)=ipsilateral, (0, 3)=crossed cerebello-cortical, (0, 4)=midline, (2, 3)=homotopic intra-cerebellar
    res = _FakeResult(5, [(0, 3), (0, 2), (2, 3), (0, 4)])
    edges = annotate_edges(res.significant_edges(tail="positive"), rois)
    assert list(edges["hemispheric_relation"]) == [
        "ipsilateral cerebello-cortical",
        "crossed cerebello-cortical",
        "bilateral or midline endpoint",
        "homotopic",
    ]
    assert list(edges["circuit"]) == [
        "cerebello-cortical",
        "cerebello-cortical",
        "cerebello-cortical",
        "intra-cerebellar",
    ]
    assert edges["sc"].isna().all()

    summary = summarize_annotation(res, rois, tail="positive")
    assert "relations" in summary and "regions" in summary

    # Headless connectogram rendering and symmetric midline layout
    lay = _connectogram_layout(ref.regions)
    mid = lay[lay["hemisphere"] == "M"]
    mid_angles = [_side_angle((r.s0 + r.s1) / 2, "M") for _, r in mid.iterrows()]
    assert abs(float(np.mean(mid_angles)) - 270.0) < 1.0
    for rid in ref.regions[ref.regions["hemisphere"] == "R"].index:
        hom = int(ref.regions.at[rid, "homolog_id"])
        ar = _side_angle((lay.at[rid, "s0"] + lay.at[rid, "s1"]) / 2, "R")
        al = _side_angle((lay.at[hom, "s0"] + lay.at[hom, "s1"]) / 2, "L")
        assert np.isclose(np.cos(np.deg2rad(ar)), -np.cos(np.deg2rad(al)))
        assert np.isclose(np.sin(np.deg2rad(ar)), np.sin(np.deg2rad(al)))

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_connectogram(res, rois, tail="positive", level="gyrus", min_weight=1.0, ax=ax)
    assert any("Brainnetome + SUIT gyrus" in t.get_text() for t in ax.texts)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    plot_connectogram(res, rois, tail="positive", level="subregion", labels="subregion", min_weight=1.0, ax=ax2)
    texts2 = {t.get_text() for t in ax2.texts}
    assert "Left_V" in texts2 and "Left" not in texts2
    assert "A8m" in texts2  # BNA-246 _L/_R suffix still stripped
    plt.close(fig2)
