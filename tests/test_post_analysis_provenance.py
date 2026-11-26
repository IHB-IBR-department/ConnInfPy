import pickle

import numpy as np
import pytest

from conninfpy import (
    AtlasInfo,
    ResultBundle,
    analyze,
    compute_p_val,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
)
from conninfpy.results import InferenceResult


def _stack(rng, n=8):
    y = rng.normal(size=(n, 3, 3))
    y = (y + y.transpose(0, 2, 1)) / 2
    y[:, np.arange(3), np.arange(3)] = 0
    return y


def _pair_summary(bundle, a, b):
    summary = bundle.summarize(
        grouping="input_position", measure="contrast_beta",
        contrast_id=bundle.metadata["contrast_id"], tail="positive",
    )
    return summary[(summary.group_a == a) & (summary.group_b == b)].iloc[0]


@pytest.mark.parametrize("test_type,var_equal", [
    ("one-sample", False), ("paired", False),
    ("two-sample", False), ("two-sample", True),
])
def test_pairwise_available_case_real_producer_roundtrip(test_type, var_equal, tmp_path):
    rng = np.random.default_rng(11)
    g1 = _stack(rng, 6)
    g2 = None if test_type == "one-sample" else _stack(rng, 6)
    if test_type == "paired":
        g2[:, 0, 1] = g2[:, 1, 0] = g1[:, 0, 1]
    elif test_type == "two-sample":
        g1[:, 0, 1] = g1[:, 1, 0] = 0
        g2[:, 0, 1] = g2[:, 1, 0] = 0
    else:
        g1[:, 0, 1] = g1[:, 1, 0] = 0
    for group in (g1, g2):
        if group is None:
            continue
        group[2:, 0, 2] = group[2:, 2, 0] = np.nan
        group[:, 1, 2] = group[:, 2, 1] = np.nan
    result = compute_p_val(
        g1, g2, test_type=test_type, var_equal=var_equal, method="tstat",
        missing="available_case", min_coverage=.8, n_permutations=2,
        acceleration=None, use_mp=False, rng=0,
    )
    bundle = result.to_bundle()
    assert bundle.eligible[0, 2] and bundle.estimable[0, 2]
    assert not bundle.tested["positive"][0, 2]
    assert not bundle.estimable[1, 2]
    assert bundle.effect[0, 1] == 0
    assert bundle.tested["positive"][0, 1]
    expected_n = 4 if test_type == "two-sample" else 2
    assert bundle.metadata["coverage"]["edge_n"][0, 2] == expected_n
    assert _pair_summary(bundle, "0", "1").mean_effect == 0
    assert _pair_summary(bundle, "0", "2").n_effect_estimable == 1
    assert _pair_summary(bundle, "0", "2").n_tested == 0
    path = tmp_path / f"{test_type}-{var_equal}.pkl"
    bundle.save(path)
    loaded = ResultBundle.load(path)
    assert not hasattr(loaded, "stat_positive") and "null_max_dist" not in loaded.__dict__
    assert _pair_summary(loaded, "1", "2").n_effect_estimable == 0


def test_pairwise_edge_mask_atlas_and_conflict():
    y = _stack(np.random.default_rng(2), 6)
    eligible = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    atlas = AtlasInfo(labels=["a", "b", "c"], networks=["A", "A", "B"],
                      label_ids=np.array([10, 20, 30]))
    result = compute_p_val(
        y, test_type="one-sample", method="tstat", edge_mask=eligible,
        n_permutations=2, acceleration=None, use_mp=False, rng=0,
        atlas=atlas, groupings={"custom": ["x", "x", "unknown"]}, units="input z units",
    )
    bundle = result.to_bundle()
    assert bundle.roi_ids == (10, 20, 30)
    assert np.array_equal(bundle.eligible, eligible)
    assert np.array_equal(bundle.estimable, eligible)
    assert np.array_equal(bundle.tested["negative"], eligible)
    assert bundle.metadata["family"]["scope"] == "explicit_edge_mask"
    assert bundle.effect_spec["units"] == "input z units"
    assert set(bundle.groupings) == {"atlas_label", "network", "custom"}
    with pytest.raises(ValueError, match="conflicts"):
        result.to_bundle(eligible=~np.eye(3, dtype=bool))


def test_analyze_captures_scale_and_metadata_after_reload(tmp_path):
    y = _stack(np.random.default_rng(3))
    atlas = AtlasInfo(labels=["a", "b", "c"], networks=["A", "A", "B"],
                      hemisphere=["L", "R", "L"], label_ids=np.array([4, 8, 9]))
    result = analyze(
        group1=y, test_type="one-sample", input="z", method="tstat",
        n_permutations=2, acceleration=None, use_mp=False, rng=0,
        atlas=atlas, groupings={"custom": ["left", "right", "left"]},
    )
    bundle = result.to_bundle()
    assert bundle.roi_ids == (4, 8, 9)
    assert bundle.effect_spec["scale"] == "fisher_z"
    assert bundle.effect_spec["units"] == "Fisher z"
    assert bundle.groupings["custom"] == ("left", "right", "left")
    path = tmp_path / "analyze.pkl"
    bundle.save(path)
    loaded = ResultBundle.load(path)
    summary = loaded.summarize(
        grouping="custom", measure="contrast_beta", contrast_id=bundle.metadata["contrast_id"],
        tail="positive",
    )
    assert not summary.empty and summary.attrs["family"]["scope"] == "all_off_diagonal_edges"


def test_original_result_pickle_retains_provenance_for_later_bundle():
    result = analyze(
        group1=_stack(np.random.default_rng(31), 6), test_type="one-sample",
        input="z", method="tstat", n_permutations=2, acceleration=None,
        use_mp=False, rng=0, roi_ids=[7, 11, 19],
        groupings={"custom": ["a", "a", "b"]}, units="declared z units",
    )
    restored = pickle.loads(pickle.dumps(result))
    bundle = restored.to_bundle()
    assert bundle.roi_ids == (7, 11, 19)
    assert bundle.groupings["custom"] == ("a", "a", "b")
    assert bundle.effect_spec["units"] == "declared z units"
    assert np.array_equal(bundle.tested["positive"], result.inference.analysis_provenance["tested"]["positive"])


def test_glm_available_case_and_f_omnibus():
    y = _stack(np.random.default_rng(5))
    y[2:, 0, 2] = y[2:, 2, 0] = np.nan
    y[:, 0, 1] = y[:, 1, 0] = 0
    result = compute_p_val_glm(
        y, interest=np.arange(8), method="tstat", missing="available_case",
        min_coverage=.8, n_permutations=2, acceleration=None, use_mp=False, rng=0,
    )
    bundle = result.to_bundle()
    assert bundle.effect[0, 1] == 0
    assert not bundle.estimable[0, 2] and not bundle.tested["positive"][0, 2]
    assert "edge_residual_df" in bundle.metadata["coverage"]
    complete = _stack(np.random.default_rng(6))
    x = np.column_stack([np.ones(8), np.arange(8)])
    omnibus = compute_p_val_glm(
        complete, design_matrix=x, contrast=[[1, 0], [0, 1]], stat_type="fstat",
        method="tstat", n_permutations=2, acceleration=None, use_mp=False, rng=0,
    ).to_bundle()
    assert omnibus.effect is None and not omnibus.estimable.any()
    assert omnibus.tested["omnibus"][0, 1]
    summary = omnibus.summarize(
        grouping="input_position", measure="contrast_beta",
        contrast_id=omnibus.metadata["contrast_id"], tail="omnibus",
    )
    assert summary.mean_effect.isna().all()


@pytest.mark.parametrize("with_missing", [False, True])
def test_glm_multi_contrast_identity(with_missing):
    y = _stack(np.random.default_rng(7))
    if with_missing:
        y[0, 0, 1] = y[0, 1, 0] = np.nan
    x = np.column_stack([np.ones(8), np.arange(8), np.arange(8) % 2])
    results = compute_p_val_glm_multi(
        y, design_matrix=x, contrasts={"age": [0, 1, 0], "group": [0, 0, 1]},
        method="tstat", missing="available_case" if with_missing else "raise",
        n_permutations=2, acceleration=None, use_mp=False, rng=0,
    )
    assert set(results) == {"age", "group"}
    for name, result in results.items():
        bundle = result.to_bundle()
        assert bundle.metadata["contrast_id"] == name
        assert bundle.tested["positive"].shape == (3, 3)
        assert bundle.metadata["family"]["correction"] == "permutation_max"


def test_paired_glm_wrapper_refreshes_model_identity():
    a = _stack(np.random.default_rng(8))
    b = _stack(np.random.default_rng(9))
    result = compute_p_val_paired_glm(
        a, b, confounds_A=np.arange(8), confounds_B=2 * np.arange(8) + 1,
        method="tstat", n_permutations=2, acceleration=None, use_mp=False, rng=0,
    )
    bundle = result.to_bundle()
    assert bundle.metadata["model_id"].startswith("paired-glm[")
    assert np.array_equal(bundle.effect, result.effect_map())


def test_grid_select_and_one_sided_keep_aligned_masks():
    y = _stack(np.random.default_rng(10), 6)
    result = compute_p_val(
        y, test_type="one-sample", method="tfnbs", e=[.4, .7], h=[3., 4.],
        n_permutations=2, acceleration=None, use_mp=False, rng=0, two_tailed=False,
    )
    assert result.analysis_provenance["tested"]["positive"].shape == (3, 3, 2)
    selected = result.select(1).to_bundle()
    assert selected.tested["positive"].shape == (3, 3)
    assert selected.metadata["e_grid"].tolist() == [.7]
    one_sided = result.one_sided("positive").to_bundle()
    assert set(one_sided.tested) == {"positive"}
    summary = one_sided.summarize(
        grouping="input_position", measure="contrast_beta",
        contrast_id=one_sided.metadata["contrast_id"], tail="positive", param_idx=0,
    )
    assert summary.attrs["param_idx"] == 0


def test_real_nbs_and_cnbs_native_provenance():
    y = _stack(np.random.default_rng(12), 6)
    nbs = compute_p_val(
        y, test_type="one-sample", method="nbs", threshold=.1,
        n_permutations=2, acceleration=None, use_mp=False, rng=0,
    ).to_bundle()
    assert nbs.metadata["inference_unit"] == "component"
    assert nbs.metadata["native_report"] is not None
    cnbs = compute_p_val(
        y, test_type="one-sample", method="cnbs", net_labels=[0, 0, 1],
        n_permutations=2, use_mp=False, rng=0,
    ).to_bundle()
    assert cnbs.metadata["inference_unit"] == "block"
    assert cnbs.metadata["native_report"] is not None


from conninfpy.atlas.resources import atlas_resource_available as _atlas_resource_available

_HAS_BNA_CACHE = _atlas_resource_available("bna246_hierarchy.csv")
_BNA_SKIP_REASON = "Brainnetome atlas files not in local cache; run scripts/download_atlases.py"


@pytest.mark.skipif(not _HAS_BNA_CACHE, reason=_BNA_SKIP_REASON)
def test_bna_atlas_supplies_parallel_anatomical_groupings():
    atlas = AtlasInfo(
        labels=["A8m_L", "A8m_R", "A9l_L"], networks=["Default"] * 3,
        source="Fan et al. 2016 Brainnetome Atlas", label_ids=np.array([1, 2, 5]),
    )
    result = compute_p_val(
        _stack(np.random.default_rng(13), 6), test_type="one-sample", method="tstat",
        n_permutations=2, acceleration=None, use_mp=False, rng=0, atlas=atlas,
    )
    groups = result.to_bundle().groupings
    assert groups["anatomical"][0].startswith("A8m")
    assert groups["lobe"] == ("Frontal", "Frontal", "Frontal")
    assert groups["structure"] == ("cortex",) * 3
    assert groups["yeo7"] != groups["yeo17"]
    assert groups["network"] == ("Default",) * 3


@pytest.mark.skipif(not _HAS_BNA_CACHE, reason=_BNA_SKIP_REASON)
def test_bna_mixed_cortical_subcortical_ids_and_functional_unknowns():
    atlas = AtlasInfo(
        labels=["A8m_L", "A8m_R", "mAmyg_L"],
        networks=["Default", "Default", "Subcortical"],
        source="Fan et al. 2016 Brainnetome Atlas", label_ids=np.array([1, 2, 211]),
    )
    result = compute_p_val(
        _stack(np.random.default_rng(14), 6), test_type="one-sample", method="tstat",
        n_permutations=2, acceleration=None, use_mp=False, rng=0, atlas=atlas,
        groupings={"custom": ["known", "unknown", "known"]},
    )
    bundle = result.to_bundle()
    assert bundle.roi_ids == (1, 2, 211)
    assert bundle.groupings["structure"] == ("cortex", "cortex", "amygdala")
    assert bundle.groupings["lobe"] == ("Frontal", "Frontal", "Subcortical")
    assert bundle.groupings["yeo7"][-1] == "not_applicable"
    assert bundle.groupings["custom"] == ("known", "unknown", "known")
    assert len(bundle.groupings["atlas_label"]) == len(set(bundle.groupings["atlas_label"])) == 3


def test_legacy_bundle_still_requires_explicit_metadata():
    p = np.full((3, 3), .5)
    legacy = InferenceResult(
        p, p, beta=np.ones((1, 3, 3)),
        design={"kind": "one-sample", "contrast": [1], "scale": "fisher_z"},
    )
    with pytest.raises(ValueError, match="lacks captured analysis metadata"):
        legacy.to_bundle()
