"""Small oracles for saved descriptive results, identity and spatial context."""
import copy
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from conninfpy import ResultBundle, brainnetome_crosswalk
from conninfpy.atlas import AtlasInfo, assign_coordinates
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.loaders.base import LoadedDataset


def _synthetic_bna_regions() -> pd.DataFrame:
    rows = []
    for rid in range(1, 247):
        hemi = "L" if rid % 2 else "R"
        hom = rid + 1 if rid % 2 else rid - 1
        name = "A8m_L" if rid == 1 else "A8m_R" if rid == 2 else f"R{rid}_{hemi}"
        rows.append(
            {
                "roi_id": rid,
                "name": name,
                "hemisphere": hemi,
                "homolog_id": hom,
                "code": f"C_{rid}",
                "description": f"Region {name}",
                "gyrus": "SFG" if rid <= 2 else f"G{(rid - 1) // 2}",
                "gyrus_name": "Superior Frontal Gyrus",
                "lobe": "Frontal" if rid <= 2 else "Temporal",
                "structure": "cortex",
                "yeo7_id": 6,
                "yeo7": "Frontoparietal",
                "yeo17_id": 12,
                "yeo17": "Control A",
                "x": -10.0 if hemi == "L" else 10.0,
                "y": float(rid),
                "z": 20.0,
            }
        )
    return pd.DataFrame(rows).set_index("roi_id")
from conninfpy.loaders.manifest import filter_dataset_rois
from conninfpy.results import InferenceResult, OmnibusInferenceResult


def matrix(n, edges, fill=0.0):
    a = np.full((n, n), fill, float)
    for i, j, value in edges:
        a[i, j] = a[j, i] = value
    return a


def fixture_result(method="tstat", grid=False, unit_report=None):
    effect = matrix(4, [(0, 1, 0), (0, 2, 2), (0, 3, -1), (1, 2, 4), (1, 3, 3), (2, 3, 6)])
    p = matrix(4, [(0, 1, .01), (0, 2, .4), (0, 3, .01), (1, 2, .2), (1, 3, .4), (2, 3, .3)], 1)
    maps = np.stack([p, p * .5], axis=2) if grid else p
    result = InferenceResult(maps, np.ones_like(maps), method=method,
        e_grid=np.array([.4, .7]) if grid else None, h_grid=np.array([3., 4.]) if grid else None,
        beta=effect[None], design={"kind": "one-sample", "contrast": [1], "scale": "fisher_z",
                                   "enhancement": {"n": None, "integration": "exact", "start_thres": 1.65}},
        component_report=unit_report if method == "nbs" else None,
        block_report=unit_report if method == "cnbs" else None)
    eligible = ~np.eye(4, dtype=bool)
    estimable = eligible.copy(); estimable[1, 3] = estimable[3, 1] = False
    tested = eligible.copy(); tested[2, 3] = tested[3, 2] = False
    masks = np.stack([tested, tested], axis=2) if grid else tested
    return result, dict(roi_ids=[1, 5, 12, 190], groupings={"functional": ["A", "A", "B", "B"],
            "hemisphere": ["L", "R", "L", "R"], "custom": ["unknown", "not_applicable", "X", "X"]},
        eligible=eligible, estimable=estimable, tested={"positive": masks, "negative": masks},
        model_id="model-A", contrast_id="mean", units="Fisher z", scale="fisher_z", family="whole-connectome")


def test_roundtrip_summary_masks_zero_and_selection(tmp_path):
    r, kw = fixture_result(grid=True)
    original = copy.deepcopy(r)
    bundle = ResultBundle.from_result(r, **kw)
    path = tmp_path / "result.pkl"; bundle.save(path)
    loaded = ResultBundle.load(path)
    assert not hasattr(loaded, "stat_positive") and "null_max_dist" not in loaded.__dict__
    assert loaded.roi_ids == (1, 5, 12, 190)
    assert loaded.metadata["design"]["enhancement"]["integration"] == "exact"
    s = loaded.summarize(grouping="functional", measure="contrast_beta", contrast_id="mean", tail="positive", param_idx=0)
    aa = s[(s.group_a == "A") & (s.group_b == "A")].iloc[0]
    ab = s[(s.group_a == "A") & (s.group_b == "B")].iloc[0]
    bb = s[(s.group_a == "B") & (s.group_b == "B")].iloc[0]
    assert aa.n_effect_estimable == 1 and aa.mean_effect == 0 and aa.share_zero == 1
    assert ab.n_eligible == 4 and ab.n_effect_estimable == 3 and ab.n_tested == 4
    assert ab.mean_effect == pytest.approx(5 / 3) and ab.share_negative == pytest.approx(1 / 3)
    assert bb.n_tested == 0 and np.isnan(bb.detection_rate)
    assert s.attrs["family"] == "whole-connectome"
    assert not loaded.summarize(grouping="hemisphere", measure="contrast_beta", contrast_id="mean", tail="positive", param_idx=0).equals(s)
    with pytest.raises(ValueError): loaded.summarize(grouping="functional", measure="contrast_beta", contrast_id="wrong", tail="positive", param_idx=0)
    with pytest.raises(ValueError): loaded.summarize(grouping="functional", measure="contrast_beta", contrast_id="mean", tail="positive")
    np.testing.assert_equal(r.positive, original.positive)
    np.testing.assert_equal(r.beta, original.beta)
    assert r.design == original.design
    csv = tmp_path / "summary.csv"; s.to_csv(csv, index=False)
    assert pd.read_csv(csv).shape == s.shape
    ax = loaded.plot_summary(s); assert ax.images[0].get_array()[0, 1] == pytest.approx(5 / 3)


def test_bundle_roundtrip_never_uses_pickle(tmp_path, monkeypatch):
    result, kwargs = fixture_result(grid=True)
    bundle = ResultBundle.from_result(result, **kwargs)
    bundle.metadata["mc"] = {"exceedance_counts": {"positive": np.zeros_like(bundle.corrected_p["positive"], dtype=int)}}
    monkeypatch.setattr(pickle, "load", lambda *a, **k: pytest.fail("pickle.load called"))
    monkeypatch.setattr(pickle, "dump", lambda *a, **k: pytest.fail("pickle.dump called"))
    path = tmp_path / "bundle.zip"
    bundle.save(path)
    restored = ResultBundle.load(path)
    assert restored.schema_version == bundle.schema_version
    assert restored.roi_ids == bundle.roi_ids
    assert restored.groupings == bundle.groupings
    assert restored.annotation == bundle.annotation
    np.testing.assert_array_equal(restored.effect, bundle.effect)
    np.testing.assert_array_equal(restored.metadata["mc"]["exceedance_counts"]["positive"], bundle.metadata["mc"]["exceedance_counts"]["positive"])


def test_schema_validation_and_f_omnibus(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema"): ResultBundle.load(bad)
    r, kw = fixture_result()
    kw["estimable"] = kw["eligible"].copy()
    kw["estimable"][0, 1] = True
    r.beta[0, 0, 1] = r.beta[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="finite effects"): ResultBundle.from_result(r, **kw)
    result = OmnibusInferenceResult(r.positive, beta=np.zeros((2, 4, 4)), design={"contrast": [[1, 0], [0, 1]], "scale": "fisher_z"})
    kw["estimable"] = np.zeros((4, 4), bool); kw["tested"] = {"omnibus": kw["eligible"]}
    out = ResultBundle.from_result(result, **kw)
    s = out.summarize(grouping="functional", measure="contrast_beta", contrast_id="mean", tail="omnibus")
    assert out.effect is None and s.mean_effect.isna().all()


def test_native_component_and_block_ids_are_not_independent_edges():
    component = matrix(4, [(0, 1, 1), (0, 3, 1)], 0).astype(int)
    report = {"labels": {"positive": component, "negative": np.zeros((4, 4), int)}}
    r, kw = fixture_result(method="nbs", unit_report=report)
    b = ResultBundle.from_result(r, **kw)
    units = b.native_units(tail="positive")
    assert len(units) == 1 and units.iloc[0].native_id == ("positive", 1)
    assert units.iloc[0].n_locations == 2 and units.iloc[0].roi_ids == [1, 5, 190]
    block = {"node_labels": ["X", "Y", "X", "Y"], "correction": "bonferroni", "blocks": []}
    r, kw = fixture_result(method="cnbs", unit_report=block)
    b = ResultBundle.from_result(r, **kw)
    assert b.detected_pairs(tail="positive").native_id.tolist() == [("X", "Y"), ("X", "Y")]
    assert len(b.native_units(tail="positive")) == 1


def test_filter_original_ids_and_assignment_preserved():
    atlas = AtlasInfo(labels=["one", "five", "twelve"], networks=["a", "b", "c"],
                      label_ids=np.array([1, 5, 12]), assignment=pd.DataFrame({"roi": ["one", "five", "twelve"]}))
    data = np.arange(9).reshape(1, 3, 3)
    ds = LoadedDataset(data, pd.DataFrame(), "correlation", atlas=atlas)
    filter_dataset_rois(ds, [12, 1], by="id")
    assert ds.atlas.label_ids.tolist() == [12, 1]
    assert ds.atlas.assignment.roi.tolist() == ["twelve", "one"]
    np.testing.assert_equal(ds.data[0], data[0][np.ix_([2, 0], [2, 0])])


def test_coordinate_contract_and_nearest_is_opt_in():
    img = nib.Nifti1Image(np.array([[[1]], [[0]], [[0]]], dtype=np.int16), np.eye(4))
    atlas = AtlasInfo.from_nifti(img)
    with pytest.raises(ValueError, match="spaces"): assign_coordinates([[0, 0, 0]], atlas)
    with pytest.raises(ValueError, match="finite"): assign_coordinates([[np.nan, 0, 0]], atlas, space="S", reference_space="S", units="mm")
    t = assign_coordinates([[1, 0, 0]], atlas, space="S", reference_space="S", units="mm")
    assert t.method.iloc[0] == "unassigned"
    t = assign_coordinates([[1, 0, 0]], atlas, space="S", reference_space="S", units="mm", max_distance=1.1)
    assert t.method.iloc[0] == "nearest voxel"
    centroid = AtlasInfo(labels=["x"], networks=["n"], coords=np.array([[0, 0, 0]]))
    t = assign_coordinates([[100, 0, 0]], centroid, space="S", reference_space="S", units="mm", max_distance=5)
    assert t.method.iloc[0] == "unassigned"


def test_crosswalk_profiles_homolog_unknown_and_overlap():
    data = np.zeros((5, 1, 1), dtype=np.int16); data[0, 0, 0] = 1; data[4, 0, 0] = 2
    affine = np.eye(4); affine[0, 3] = -2
    image = nib.Nifti1Image(data, affine)
    ref = BrainnetomeReference(regions=_synthetic_bna_regions(), label_image=image,
        profiles={"BD": {1: {"Cognition.Memory": 2.5}, 2: {}}, "PC": {1: {}, 2: {"Task": 1.2}}},
        provenance={"atlas": "synthetic", "profiles": {"source": "fixture"}})
    cross = brainnetome_crosswalk(roi_ids=[12, 3, 9], coords=[[-2, 0, 0], [2, 0, 0], [0, 0, 0]],
        space="MNI152NLin6Asym", units="mm", reference=ref, networks=["A", "not_applicable", "unknown"])
    assert cross["roi_ids"] == (12, 3, 9)
    assert cross["groupings"]["functional"] == ("A", "not_applicable", "unknown")
    assert set(cross["groupings"]) == {"anatomical", "gyrus", "lobe", "hemisphere", "structure", "functional"}
    a, b, u = cross["endpoints"]
    assert a["profiles"]["BD"]["terms"][0]["score"] == 2.5
    assert a["profiles"]["PC"]["status"] == "no_source_filtered_terms"
    assert a["homolog_reference"][0]["homolog_bna_id"] == 2
    assert b["profiles"]["PC"]["terms"][0]["score"] == 1.2
    assert u["unknown_fraction"] == 1 and u["profiles"]["BD"]["status"] == "unassigned"
    parcel = np.zeros((5, 1, 1), dtype=np.int16); parcel[0:2, 0, 0] = 7
    cp = brainnetome_crosswalk(roi_ids=[7], space="MNI152NLin6Asym", units="mm", reference=ref,
                               support="parcel", parcel_image=nib.Nifti1Image(parcel, affine), parcel_labels=[7])
    assert cp["endpoints"][0]["unknown_fraction"] == pytest.approx(.5)
    with pytest.raises(ValueError): brainnetome_crosswalk(roi_ids=[1], coords=[[0, 0, 0]], space="Talairach", units="mm", reference=ref)
    with pytest.raises(ValueError): brainnetome_crosswalk(roi_ids=[1], coords=[[0, 0, 0]], space="MNI152", units="cm", reference=ref)
    with pytest.raises(ValueError): brainnetome_crosswalk(roi_ids=[1], coords=[[0, 0, 0]], space="MNI152", units="mm", reference=ref, nearest_mm=5)


def test_annotation_endpoint_pair_and_missing_profile():
    data = np.zeros((5, 1, 1), dtype=np.int16); data[0, 0, 0] = 1; data[4, 0, 0] = 2
    affine = np.eye(4); affine[0, 3] = -2
    ref = BrainnetomeReference(regions=_synthetic_bna_regions(), label_image=nib.Nifti1Image(data, affine),
        profiles={"BD": {1: {"shared": 2}, 2: {"shared": 3}}, "PC": {1: {"task": 5}}},
        provenance={"atlas": "fixture", "profiles": {"source": "test"}})
    annotation = brainnetome_crosswalk(roi_ids=[1, 5, 12, 190], coords=[[-2, 0, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0]],
              space="MNI152NLin6Asym", units="mm", reference=ref)
    assert annotation["endpoints"][1]["profiles"]["PC"]["status"] == "profile_missing"
    r, kw = fixture_result(); kw["annotation"] = annotation
    bundle = r.to_bundle(**kw)
    pairs = bundle.detected_pairs(tail="positive", shared_terms=True)
    first = pairs.iloc[0]
    assert (first.roi_i, first.roi_j) == (1, 5)
    assert first.endpoint_i["profiles"]["BD"]["terms"][0]["score"] == 2
    assert first.endpoint_j["profiles"]["BD"]["terms"][0]["score"] == 3
    assert first.shared_BD_terms == ["shared"]
    assert first.endpoint_i["homolog_reference"][0]["name"] == "A8m_R"


def test_one_sided_grid_validation_and_export_metadata(tmp_path):
    r, kw = fixture_result(grid=True)
    r.tested_tail = "positive"; kw["tested"].pop("negative")
    bundle = ResultBundle.from_result(r, **kw)
    with pytest.raises(ValueError, match="not saved"): bundle.detected_pairs(tail="negative", param_idx=0)
    s = bundle.summarize(grouping="functional", measure="contrast_beta", contrast_id="mean", tail="positive", param_idx=1)
    assert s.attrs["e"] == .7 and s.attrs["h"] == 4.
    path = tmp_path / "selected.csv"
    bundle.export_summary(s, path)
    import json
    meta = json.loads(path.with_suffix(".csv.json").read_text())
    assert meta["grouping"] == "functional" and meta["param_idx"] == 1
    assert meta["measure"]["standardizer"] is None
    bundle.tested["positive"][0, 1, 0] = False  # invalid asymmetric saved object
    with pytest.raises(ValueError, match="symmetric"): bundle.save(tmp_path / "invalid.pkl")


def test_sphere_coverage_includes_outside_reference_field():
    data = np.ones((1, 1, 1), dtype=np.int16)
    ref = BrainnetomeReference(regions=_synthetic_bna_regions(), label_image=nib.Nifti1Image(data, np.eye(4)),
                               profiles=None, provenance={"atlas": "fixture"})
    cross = brainnetome_crosswalk(roi_ids=[1], coords=[[0, 0, 0]], space="MNI152NLin6Asym", units="mm",
                                 reference=ref, support="sphere", radius_mm=1.01, min_coverage=.1)
    ep = cross["endpoints"][0]
    assert ep["bna_overlap"][1] == pytest.approx(1 / 7)
    assert ep["unknown_fraction"] == pytest.approx(6 / 7)


def test_endpoint_profiles_preserve_small_overlap_terms():
    data = np.zeros((5, 1, 1), dtype=np.int16)
    data[0, 0, 0] = 1; data[4, 0, 0] = 2
    affine = np.eye(4); affine[0, 3] = -2
    ref = BrainnetomeReference(regions=_synthetic_bna_regions(), label_image=nib.Nifti1Image(data, affine),
        profiles={"BD": {1: {"first": 1.1}, 2: {"second": 1.2}}, "PC": {1: {}, 2: {}}},
        provenance={"atlas": "fixture"})
    source = nib.Nifti1Image(np.full((5, 1, 1), 7, dtype=np.int16), affine)
    cross = brainnetome_crosswalk(roi_ids=[7], space="MNI152NLin6Asym", units="mm", reference=ref,
                                 support="parcel", parcel_image=source, parcel_labels=[7], min_weight=.9,
                                 min_coverage=.3)
    ep = cross["endpoints"][0]
    assert {x["term"] for x in ep["profiles"]["BD"]["terms"]} == {"first", "second"}
    assert ep["unknown_fraction"] == pytest.approx(.6)
