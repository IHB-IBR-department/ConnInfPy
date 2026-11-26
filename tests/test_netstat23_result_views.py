"""Tests for NETSTAT-23: network summaries, scale/units, predicted/coupling, null max, companion checks, and UI parity."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conninfpy import analyze, compute_p_val
from conninfpy.atlas import AtlasInfo
from conninfpy.results import InferenceResult
from conninfpy.stats.glm import compute_p_val_glm
from conninfpy.app.utils.result_views import (
    build_cnbs_block_figure,
    build_companion_agreement_figure,
    build_connectivity_maps_figure,
    build_network_summary_figure,
    build_null_max_figure,
    build_unit_annotation_context,
    check_companion_compatibility,
    describe_result_scale_and_stat,
    get_2d_result,
    prepare_cnbs_blocks_table,
    prepare_network_summary,
    prepare_node_summary,
    prepare_predicted_coupling_summary,
    resolve_roi_ordering,
)


APP = Path(__file__).resolve().parents[1] / "conninfpy" / "app" / "streamlit_app.py"


def _sym_batch(rng: np.random.Generator, n_sub: int, n_nodes: int) -> np.ndarray:
    a = rng.normal(scale=0.18, size=(n_sub, n_nodes, n_nodes))
    a = 0.5 * (a + a.transpose(0, 2, 1))
    a[:, np.arange(n_nodes), np.arange(n_nodes)] = 0.0
    return a


def _synthetic_group_data(n_sub: int = 16, n_nodes: int = 6, seed: int = 7):
    rng = np.random.default_rng(seed)
    g0 = _sym_batch(rng, n_sub // 2, n_nodes) + 0.35
    g1 = _sym_batch(rng, n_sub - n_sub // 2, n_nodes) + 0.35
    for k in range(len(g0)):
        np.fill_diagonal(g0[k], 0.0)
    for k in range(len(g1)):
        np.fill_diagonal(g1[k], 0.0)
    # Connected 3-edge positive component on (1, 4, 5) including Vis block (1, 4), plus negative edge (0, 2) in SMN
    for u, v in ((1, 4), (1, 5), (4, 5)):
        g1[:, u, v] += 1.0
        g1[:, v, u] += 1.0
    g1[:, 0, 2] -= 0.9
    g1[:, 2, 0] -= 0.9
    data = np.concatenate([g0, g1], axis=0)
    pheno = pd.DataFrame({
        "subject_id": [f"s{i}" for i in range(n_sub)],
        "group": [0] * (n_sub // 2) + [1] * (n_sub - n_sub // 2),
        "age": np.linspace(20.0, 40.0, n_sub),
    })
    # Deliberately interleaved network labels: first appearance order is SMN, Vis, DMN
    atlas = AtlasInfo(
        labels=[f"ROI{i}" for i in range(n_nodes)],
        networks=["SMN", "Vis", "SMN", "DMN", "Vis", "DMN"],
        coords=np.array([
            [-20, -10, 40],
            [20, -80, 10],
            [22, -12, 42],
            [-6, 48, 8],
            [-22, -82, 12],
            [8, -52, 26],
        ], dtype=float),
    )
    effect_mask = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v in ((1, 4), (1, 5), (4, 5), (0, 2)):
        effect_mask[u, v] = effect_mask[v, u] = 1.0
    return g0, g1, data, pheno, atlas, effect_mask


def _omnibus_f_result(data: np.ndarray, pheno: pd.DataFrame):
    n_sub = len(pheno)
    g = np.array([i % 3 for i in range(n_sub)])
    X = np.column_stack([np.ones(n_sub), (g == 1).astype(float), (g == 2).astype(float)])
    C = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return compute_p_val_glm(
        data, design_matrix=X, contrast=C, stat_type="fstat",
        method="tstat", n_permutations=30, rng=2, use_mp=False, acceleration=None,
    )


# ---------------------------------------------------------------------------
# AC #1: Stat maps vs native effect_map(), scale/sign/units, aligned ROI order
# ---------------------------------------------------------------------------

def test_ac1_stat_map_never_labeled_effect_size_and_separates_native_cbeta():
    _, _, data, pheno, atlas, effect_mask = _synthetic_group_data()
    res = analyze(
        data,
        interest=pheno["group"].to_numpy(dtype=float),
        confounds=pheno["age"].to_numpy(dtype=float),
        method="tstat",
        input="z",
        r_to_z=False,
        n_permutations=40,
        rng=42,
        use_mp=False,
        acceleration=None,
        atlas=atlas,
    )
    fig, info = build_connectivity_maps_figure(
        res, atlas=atlas, alpha=0.05, method_name="TSTAT", effect_mask=effect_mask
    )
    try:
        assert fig is not None
        assert info["has_effect_map"] is True
        assert info["has_truth_mask"] is True
        assert info["scale"] == "fisher_z"
        assert info["stat_symbol"] in ("t", "β")

        image_axes = [ax for ax in fig.axes if ax.get_images()]
        assert len(image_axes) == 4  # stat, signed -log10(p), native c'beta, truth
        titles = [ax.get_title() for ax in image_axes]
        assert all("Observed Effect Size" not in t for t in titles)
        assert "not effect size" in titles[0]
        assert "Significant Detections" in titles[1]
        assert "Native Contrast Effect (c' β)" in titles[2]
        assert "Synthetic Ground-Truth Mask" in titles[3]

        cbar_labels = [ax.get_ylabel() for ax in fig.axes if not ax.get_images()]
        assert any("Fisher z units" in lbl for lbl in cbar_labels)

        # PNG export of the exact figure succeeds and is valid PNG
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        assert buf.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        plt.close(fig)


def test_ac1_roi_ordering_and_network_boundaries_match_across_all_panels():
    _, _, data, pheno, atlas, effect_mask = _synthetic_group_data()
    res = analyze(
        data,
        interest=pheno["group"].to_numpy(dtype=float),
        method="tstat",
        input="r",
        r_to_z=False,
        n_permutations=40,
        rng=5,
        use_mp=False,
        acceleration=None,
        atlas=atlas,
    )
    perm, block_starts, block_names, has_nets = resolve_roi_ordering(atlas, len(atlas))
    assert has_nets is True
    # First-appearance network order: SMN [0, 2], Vis [1, 4], DMN [3, 5]
    assert block_names == ["SMN", "Vis", "DMN"]
    assert block_starts == [0, 2, 4]
    assert list(perm) == [0, 2, 1, 4, 3, 5]

    fig, info = build_connectivity_maps_figure(
        res, atlas=atlas, alpha=0.05, method_name="TSTAT", effect_mask=effect_mask
    )
    try:
        inner = res.inference
        image_axes = [ax for ax in fig.axes if ax.get_images()]
        stat_arr = np.asarray(image_axes[0].get_images()[0].get_array())
        eff_arr = np.asarray(image_axes[2].get_images()[0].get_array())
        truth_arr = np.asarray(image_axes[3].get_images()[0].get_array())

        np.testing.assert_allclose(stat_arr, inner.stat_signed[np.ix_(perm, perm)], equal_nan=True)
        np.testing.assert_allclose(eff_arr, inner.effect_map()[np.ix_(perm, perm)], equal_nan=True)
        np.testing.assert_allclose(truth_arr, (effect_mask > 0).astype(float)[np.ix_(perm, perm)])
        assert info["effect_unit_label"] == "c' β (Δr, correlation units)"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# AC #2: Network summary, node summary, cNBS block report, omnibus & grid
# ---------------------------------------------------------------------------

def test_ac2_summaries_across_tstat_tfnbs_nbs_cnbs_and_omnibus():
    g0, g1, data, pheno, atlas, _ = _synthetic_group_data()

    res_tstat = compute_p_val(g0, g1, test_type="two-sample", method="tstat", n_permutations=40, rng=1, use_mp=False, acceleration=None)
    res_tfnbs = compute_p_val(g0, g1, test_type="two-sample", method="tfnbs", e=[0.5, 0.75], h=[2.0, 2.0], n_permutations=30, rng=1, use_mp=False, acceleration=None)
    res_nbs = compute_p_val(g0, g1, test_type="two-sample", method="nbs", threshold=1.5, n_permutations=80, rng=1, use_mp=False, acceleration=None)
    res_cnbs = compute_p_val(g0, g1, test_type="two-sample", method="cnbs", net_labels=atlas.network_index(), n_permutations=140, rng=1, use_mp=False, acceleration=None)
    res_omni = _omnibus_f_result(data, pheno)

    # TFNBS grid selection produces a 2D slice without crashing
    tfnbs_2d, idx = get_2d_result(res_tfnbs, 1)
    assert idx == 1
    assert np.asarray(tfnbs_2d["positive"]).ndim == 2

    for r in (res_tstat, tfnbs_2d, res_nbs, res_cnbs, res_omni):
        net_df, reason = prepare_network_summary(r, atlas=atlas, alpha=0.05)
        assert reason is None and net_df is not None and not net_df.empty
        fig = build_network_summary_figure(r, atlas=atlas, alpha=0.05, method_name="TEST")
        assert fig is not None
        plt.close(fig)

        node_df, n_reason = prepare_node_summary(r, atlas=atlas, alpha=0.05)
        assert n_reason is None and node_df is not None and not node_df.empty

    # Unknown networks atlas hides network summary cleanly and drops 'Unknown' cols in node summary
    unk_atlas = AtlasInfo(labels=atlas.labels, networks=["Unknown"] * len(atlas))
    net_unk, reason_unk = prepare_network_summary(res_tstat, atlas=unk_atlas, alpha=0.05)
    assert net_unk is None and "no usable network labels" in reason_unk
    node_unk, _ = prepare_node_summary(res_tstat, atlas=unk_atlas, alpha=0.05)
    assert "network" not in node_unk.columns

    # cNBS block report table & figure match res_cnbs.block_report without recomputing
    blk_df, blk_info = prepare_cnbs_blocks_table(res_cnbs, atlas=atlas, alpha=0.05)
    assert blk_df is not None
    assert len(blk_df) == res_cnbs.block_report["n_blocks_tested"]
    assert blk_info["correction"] == res_cnbs.block_report["correction"]
    assert any("Vis" in b for b in blk_df["Block"])
    blk_fig = build_cnbs_block_figure(res_cnbs, atlas=atlas, alpha=0.05, method_name="CNBS")
    assert blk_fig is not None
    plt.close(blk_fig)

    # Omnibus metadata is explicit
    omni_meta = describe_result_scale_and_stat(res_omni)
    assert omni_meta["is_omnibus"] is True
    assert omni_meta["stat_symbol"] == "F"


# ---------------------------------------------------------------------------
# AC #3: predicted() / coupling(), scale conversions, unit annotation context
# ---------------------------------------------------------------------------

def test_ac3_predicted_and_coupling_scales_and_legacy_missing_beta():
    _, _, data, pheno, atlas, _ = _synthetic_group_data()
    kwargs = dict(
        interest=pheno["group"].to_numpy(dtype=float),
        confounds=pheno["age"].to_numpy(dtype=float),
        method="tstat",
        r_to_z=False,
        n_permutations=30,
        rng=9,
        use_mp=False,
        acceleration=None,
        atlas=atlas,
    )
    res_z = analyze(data, input="z", **kwargs)
    out_z = prepare_predicted_coupling_summary(res_z, atlas=atlas, alpha=0.05)
    assert out_z["available"] is True
    assert out_z["scale"] == "fisher_z"
    assert "tanh(Δz) is NOT Δr" in out_z["scale_explanation"]
    edf = out_z["edge_predictions_df"]
    p0, p1 = out_z["comparison"]
    assert {f"z ({p0})", f"z ({p1})", f"r ({p0})", f"r ({p1})", "Δr (r_B - r_A)", "coupling_class"} <= set(edf.columns)
    row_14 = edf[(edf["roi_i"] == 1) & (edf["roi_j"] == 4)].iloc[0]
    assert np.isclose(row_14[f"r ({p0})"], np.tanh(row_14[f"z ({p0})"]))
    assert np.isclose(row_14[f"r ({p1})"], np.tanh(row_14[f"z ({p1})"]))
    assert np.isclose(row_14["Δr (r_B - r_A)"], row_14[f"r ({p1})"] - row_14[f"r ({p0})"])
    assert abs(row_14["Δr (r_B - r_A)"] - np.tanh(row_14[f"z ({p1})"] - row_14[f"z ({p0})"])) > 1e-3

    res_r = analyze(data, input="r", **kwargs)
    out_r = prepare_predicted_coupling_summary(res_r, atlas=atlas, alpha=0.05)
    assert out_r["available"] is True
    assert out_r["scale"] == "correlation"
    assert f"r ({p0})" in out_r["edge_predictions_df"].columns
    assert f"z ({p0})" not in out_r["edge_predictions_df"].columns

    # Unknown scale: must NOT apply tanh or label columns as r / Δr
    inner_unk = InferenceResult(
        positive=res_z.inference["positive"],
        negative=res_z.inference["negative"],
        stat_positive=res_z.inference.stat_positive,
        stat_negative=res_z.inference.stat_negative,
        method="tstat",
        stat_type="tstat",
        beta=res_z.inference.beta,
        design={**res_z.inference.design, "scale": None},
    )
    out_unk = prepare_predicted_coupling_summary(inner_unk, atlas=atlas, alpha=0.05)
    assert out_unk["available"] is True
    assert out_unk["scale"] == "unknown"
    assert "no r / Δr or tanh conversion applied" in out_unk["scale_explanation"]
    assert "Δ (unknown units)" in out_unk["edge_predictions_df"].columns
    assert "Δr (r_B - r_A)" not in out_unk["edge_predictions_df"].columns

    # Legacy result without beta / profiles reports unavailable cleanly
    legacy = InferenceResult(
        positive=res_z.inference["positive"],
        negative=res_z.inference["negative"],
        stat_positive=res_z.inference.stat_positive,
        stat_negative=res_z.inference.stat_negative,
        method="tstat",
    )
    out_legacy = prepare_predicted_coupling_summary(legacy, atlas=atlas, alpha=0.05)
    assert out_legacy["available"] is False
    assert "unavailable" in out_legacy["reason"].lower()


def test_ac3_unit_annotation_context_for_nbs_and_cnbs():
    g0, g1, _, _, atlas, _ = _synthetic_group_data()
    res_nbs = compute_p_val(g0, g1, test_type="two-sample", method="nbs", threshold=1.5, n_permutations=140, rng=1, use_mp=False, acceleration=None)
    res_cnbs = compute_p_val(g0, g1, test_type="two-sample", method="cnbs", net_labels=atlas.network_index(), n_permutations=140, rng=1, use_mp=False, acceleration=None)

    # Note: real RoiAnnotation.table has string 'roi' names (e.g. 'A8m_L'), NOT int indices
    rois = SimpleNamespace(
        table=pd.DataFrame({
            "roi": atlas.labels,
            "network": atlas.networks,
            "lobe": ["Frontal", "Occipital", "Frontal", "Frontal", "Occipital", "Parietal"],
            "gyrus": ["SFG", "MOG", "SFG", "MFG", "MOG", "PCun"],
            "domains": ["Action; Cognition", "Perception", "Action", "Cognition", "Perception", "Cognition"],
        })
    )
    nbs_edges = res_nbs.significant_edges(atlas=atlas, alpha=0.05)
    assert not nbs_edges.empty
    ctx_nbs = build_unit_annotation_context(res_nbs, nbs_edges, rois, alpha=0.05)
    assert ctx_nbs["inference_unit"] == "component"
    assert "ROI selection rule (component unit)" in ctx_nbs["roi_selection_rule"]
    assert "does not test node/hub significance" in ctx_nbs["roi_selection_rule"]
    assert not ctx_nbs["unit_table"].empty
    assert {"Component", "Edges", "ROIs", "p (FWER)", "BrainMap domains (ROI counts)"} <= set(ctx_nbs["unit_table"].columns)

    cnbs_edges = res_cnbs.significant_edges(atlas=atlas, alpha=0.05)
    assert not cnbs_edges.empty
    ctx_cnbs = build_unit_annotation_context(res_cnbs, cnbs_edges, rois, alpha=0.05)
    assert ctx_cnbs["inference_unit"] == "block"
    assert "ROI selection rule (block unit)" in ctx_cnbs["roi_selection_rule"]
    assert not ctx_cnbs["unit_table"].empty
    assert {"Block (network pair)", "Edges in block", "ROIs in block", "Block p"} <= set(ctx_cnbs["unit_table"].columns)
    assert any("Vis" in b or "SMN" in b for b in ctx_cnbs["unit_table"]["Block (network pair)"])


# ---------------------------------------------------------------------------
# AC #4: Null max-distribution & strict companion compatibility checks
# ---------------------------------------------------------------------------

def test_ac4_null_max_figure_and_strict_companion_compatibility():
    _, _, data, pheno, atlas, _ = _synthetic_group_data()
    g = pheno["group"].to_numpy(dtype=float)
    res_p = analyze(
        data, interest=g, method="tstat", input="z", r_to_z=False,
        n_permutations=40, rng=3, use_mp=False, acceleration=None, atlas=atlas,
    )
    res_c_same = analyze(
        data, interest=g, method="tstat", input="z", r_to_z=False,
        n_permutations=40, rng=7, use_mp=False, acceleration=None, atlas=atlas,
    )
    res_nbs = analyze(
        data, interest=g, method="nbs", threshold=2.0, input="z", r_to_z=False,
        n_permutations=40, rng=3, use_mp=False, acceleration=None, atlas=atlas,
    )

    # null_max_dist plot succeeds when saved, unavailable when absent
    with_null = InferenceResult(
        positive=res_p.inference["positive"],
        negative=res_p.inference["negative"],
        stat_positive=res_p.inference.stat_positive,
        stat_negative=res_p.inference.stat_negative,
        method="tstat",
        null_max_dist={
            "positive": np.linspace(1.2, 3.8, 40),
            "negative": np.linspace(1.1, 3.6, 40),
        },
    )
    fig_null, info_null = build_null_max_figure(with_null, method_name="TSTAT")
    assert fig_null is not None and info_null["available"] is True
    plt.close(fig_null)

    no_null = InferenceResult(
        positive=res_p.inference["positive"],
        negative=res_p.inference["negative"],
        stat_positive=res_p.inference.stat_positive,
        stat_negative=res_p.inference.stat_negative,
        method="tstat",
    )
    fig_none, info_none = build_null_max_figure(no_null)
    assert fig_none is None and info_none["available"] is False

    # Compatible pair yields agreement matrix + Jaccard
    compat = check_companion_compatibility(res_p, res_c_same, alpha=0.05)
    assert compat["compatible"] is True
    agree_fig = build_companion_agreement_figure(compat, atlas=atlas, primary_name="A", comp_name="B")
    assert agree_fig is not None
    plt.close(agree_fig)

    # 1. Inference unit mismatch (edge vs component)
    incompat_unit = check_companion_compatibility(res_p, res_nbs, alpha=0.05)
    assert incompat_unit["compatible"] is False
    assert any("Inference unit" in r for r in incompat_unit["reasons"])

    # 2. Tail direction mismatch
    res_pos_only = analyze(
        data, interest=g, method="tstat", input="z", r_to_z=False,
        tested_tail="positive", n_permutations=40, rng=7, use_mp=False, acceleration=None, atlas=atlas,
    )
    incompat_tail = check_companion_compatibility(res_p, res_pos_only, alpha=0.05)
    assert incompat_tail["compatible"] is False
    assert any("Tested tail" in r or "tail family" in r for r in incompat_tail["reasons"])

    # 3. Alpha mismatch
    incompat_alpha = check_companion_compatibility(res_p, res_c_same, primary_alpha=0.05, comp_alpha=0.01)
    assert incompat_alpha["compatible"] is False
    assert any("α" in r for r in incompat_alpha["reasons"])

    # 4. ROI frame size / roi_ids mismatch
    bad_roi_res = InferenceResult(
        positive=res_c_same.inference["positive"],
        negative=res_c_same.inference["negative"],
        stat_positive=res_c_same.inference.stat_positive,
        stat_negative=res_c_same.inference.stat_negative,
        method="tstat",
        two_tailed=res_c_same.inference.two_tailed,
        design=res_c_same.inference.design,
        analysis_provenance={**res_c_same.inference.analysis_provenance, "roi_ids": ["X"] * len(atlas)},
    )
    incompat_roi = check_companion_compatibility(res_p, bad_roi_res, alpha=0.05)
    assert incompat_roi["compatible"] is False
    assert any("roi_ids" in r for r in incompat_roi["reasons"])

    # 5. Eligible/tested mask mismatch
    p_masked = np.array(res_c_same.inference["positive"], copy=True)
    p_masked[0, 1] = p_masked[1, 0] = np.nan
    bad_mask_res = InferenceResult(
        positive=p_masked,
        negative=res_c_same.inference["negative"],
        stat_positive=res_c_same.inference.stat_positive,
        stat_negative=res_c_same.inference.stat_negative,
        method="tstat",
        two_tailed=res_c_same.inference.two_tailed,
        design=res_c_same.inference.design,
        analysis_provenance=res_c_same.inference.analysis_provenance,
    )
    incompat_mask = check_companion_compatibility(res_p, bad_mask_res, alpha=0.05)
    assert incompat_mask["compatible"] is False
    assert any("mask" in r.lower() for r in incompat_mask["reasons"])


# ---------------------------------------------------------------------------
# AC #5: End-to-end Streamlit AppTest UI parity
# ---------------------------------------------------------------------------

def _apptest_results_page(
    result,
    atlas,
    pheno,
    *,
    alpha: float = 0.05,
    effect_mask=None,
    companion=None,
    companion_method: str | None = None,
    active_tab: str = "3. Inference Results",
):
    from streamlit.testing.v1 import AppTest

    n_nodes = len(atlas)
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((len(pheno), n_nodes, n_nodes))
    app.session_state.pheno_df = pheno
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = result
    app.session_state.edges_df = get_2d_result(result, 0)[0].significant_edges(atlas=atlas, alpha=alpha)
    app.session_state._synthetic_effect_mask = effect_mask
    if companion is not None:
        app.session_state.companion_inference_result = companion
        app.session_state.companion_edges_df = companion.significant_edges(atlas=atlas, alpha=alpha)
        app.session_state.companion_method = companion_method or getattr(companion, "method", "companion")
    app.session_state.run_plan = {
        "method": getattr(result, "method", "tstat"),
        "alpha": alpha,
        "loaded_settings_hash": "fixture data",
        "active_atlas_signature": "fixture atlas",
        "question_type": "Group Difference",
        "ref_group": 0,
        "target_group": 1,
        "analysis_scale": "fisher_z",
    }
    app.session_state.active_tab = active_tab
    return app.run()


def test_ac5_apptest_cnbs_predicted_null_max_and_no_effect_size_mislabeling():
    pytest.importorskip("streamlit")
    _, _, data, pheno, atlas, effect_mask = _synthetic_group_data()
    res = analyze(
        data,
        interest=pheno["group"].to_numpy(dtype=float),
        method="cnbs",
        net_labels=atlas.network_index(),
        input="z",
        r_to_z=False,
        n_permutations=140,
        rng=4,
        use_mp=False,
        acceleration=None,
        atlas=atlas,
    )
    app = _apptest_results_page(res, atlas, pheno, alpha=0.05, effect_mask=effect_mask)
    assert not app.exception

    all_text = "\n".join(
        [m.value for m in app.markdown]
        + [c.value for c in app.caption]
        + [i.value for i in app.info]
    )
    assert "Observed Effect Size" not in all_text
    assert "Network-Pair Summary" in all_text
    assert "Node Summary" in all_text
    assert "cNBS Network Blocks" in all_text
    assert "Predicted Profiles & Coupling" in all_text
    assert "Null Max-Statistic Distribution" in all_text
    assert "tanh(Δz) is NOT Δr" in all_text

    dl_labels = [b.proto.label for b in app.get("download_button")]
    assert any("Connectivity Maps PNG" in lbl for lbl in dl_labels)
    assert any("Network Summary PNG" in lbl for lbl in dl_labels)
    assert any("Blocks CSV" in lbl for lbl in dl_labels)
    assert any("Predicted Profiles CSV" in lbl for lbl in dl_labels)


def test_ac5_apptest_omnibus_and_empty_unknown_networks_and_incompatible_companion():
    pytest.importorskip("streamlit")
    _, _, data, pheno, atlas, effect_mask = _synthetic_group_data()
    res_omni = _omnibus_f_result(data, pheno)
    unk_atlas = AtlasInfo(labels=atlas.labels, networks=["Unknown"] * len(atlas))
    app_omni = _apptest_results_page(res_omni, unk_atlas, pheno, alpha=0.01)
    assert not app_omni.exception
    info_txt = "\n".join(i.value for i in app_omni.info)
    assert "no usable network labels" in info_txt

    # Incompatible companion (tstat edge vs nbs component) omits bogus agreement
    g = pheno["group"].to_numpy(dtype=float)
    res_t = analyze(data, interest=g, method="tstat", input="z", r_to_z=False, n_permutations=30, rng=1, use_mp=False, acceleration=None, atlas=atlas)
    res_nbs = analyze(data, interest=g, method="nbs", threshold=2.0, input="z", r_to_z=False, n_permutations=30, rng=1, use_mp=False, acceleration=None, atlas=atlas)
    app_cmp = _apptest_results_page(res_t, atlas, pheno, effect_mask=effect_mask, companion=res_nbs, companion_method="nbs")
    assert not app_cmp.exception
    radio = next(r for r in app_cmp.get("radio") if "Select Result View Mode" in r.proto.label)
    app_cmp = radio.set_value("Side-by-Side Comparison").run()
    assert not app_cmp.exception
    warnings_txt = "\n".join(w.value for w in app_cmp.warning)
    assert "Companion agreement omitted (incompatible runs)" in warnings_txt
    assert "Inference unit differs" in warnings_txt


def test_ac5_apptest_tfnbs_grid_and_compatible_companion_and_bna_component_context(tmp_path):
    pytest.importorskip("streamlit")
    from conninfpy.atlas.brainnetome import BrainnetomeReference
    from conninfpy.interpret.annotate import annotate_bna_regions, summarize_annotation

    g0, g1, data, pheno, atlas, _ = _synthetic_group_data()
    g = pheno["group"].to_numpy(dtype=float)
    res_tfnbs = analyze(
        data, interest=g, method="tfnbs", e=[0.5, 0.75], h=[2.0, 2.0],
        input="z", r_to_z=False, n_permutations=30, rng=1, use_mp=False, acceleration=None, atlas=atlas,
    )
    res_t1 = analyze(data, interest=g, method="tstat", input="z", r_to_z=False, n_permutations=30, rng=1, use_mp=False, acceleration=None, atlas=atlas)
    res_t2 = analyze(data, interest=g, method="tstat", input="z", r_to_z=False, n_permutations=30, rng=2, use_mp=False, acceleration=None, atlas=atlas)

    # TFNBS grid run renders and switching (E, H) slice works without exception
    app_grid = _apptest_results_page(res_tfnbs, atlas, pheno, alpha=0.05)
    assert not app_grid.exception
    grid_sel = next(s for s in app_grid.get("selectbox") if "TFNBS parameter-grid cell" in s.proto.label)
    app_grid = grid_sel.set_value(1).run()
    assert not app_grid.exception

    # Compatible companion renders agreement metrics + PNG download
    app_ok = _apptest_results_page(res_t1, atlas, pheno, companion=res_t2, companion_method="tstat_alt")
    radio = next(r for r in app_ok.get("radio") if "Select Result View Mode" in r.proto.label)
    app_ok = radio.set_value("Side-by-Side Comparison").run()
    assert not app_ok.exception
    assert any(m.label == "Jaccard Overlap" for m in app_ok.metric)
    dl_labels = [b.proto.label for b in app_ok.get("download_button")]
    assert any("Companion Agreement PNG" in lbl for lbl in dl_labels)

    # BNA Annotation view shows ROI selection rule and Aggregated Component Context for NBS
    import nibabel as nib

    res_nbs = compute_p_val(
        g0, g1, test_type="two-sample", method="nbs", threshold=1.5,
        n_permutations=140, rng=1, use_mp=False, acceleration=None,
    )
    rows = [
        {
            "roi_id": rid,
            "name": f"A{rid}_{'L' if rid % 2 else 'R'}",
            "hemisphere": "L" if rid % 2 else "R",
            "homolog_id": rid + 1 if rid % 2 else rid - 1,
            "code": f"C_{rid}",
            "description": f"Synthetic {rid}",
            "gyrus": "SFG",
            "gyrus_name": "Superior Frontal Gyrus",
            "lobe": "Frontal",
            "structure": "cortex",
            "yeo7_id": 6,
            "yeo7": "Frontoparietal",
            "yeo17_id": 12,
            "yeo17": "Control A",
            "x": -10.0 if rid % 2 else 10.0,
            "y": float(rid * 5),
            "z": 50.0,
        }
        for rid in range(1, 7)
    ]
    vol = np.zeros((6, 1, 1), dtype=np.int16)
    for rid in range(1, 7):
        vol[rid - 1, 0, 0] = rid
    ref = BrainnetomeReference(
        regions=pd.DataFrame(rows).set_index("roi_id"),
        label_image=nib.Nifti1Image(vol, np.eye(4)),
        data_dir=tmp_path,
        cache_dir=tmp_path,
        profiles=None,
        provenance={"atlas": "synthetic BNA fixture"},
    )
    rois = annotate_bna_regions([1, 2, 3, 4, 5, 6], reference=ref)
    summary = summarize_annotation(res_nbs, rois, alpha=0.05)
    edges = summary["edges"]
    detected = edges[(edges["p_positive"] <= 0.05) | (edges["p_negative"] <= 0.05)]
    assert not detected.empty

    from streamlit.testing.v1 import AppTest
    app_ann = AppTest.from_file(str(APP), default_timeout=120)
    app_ann.session_state.connectivity_data = data
    app_ann.session_state.pheno_df = pheno
    app_ann.session_state.loaded_reference_atlas = rois.to_atlas()
    app_ann.session_state.loaded_atlas_signature = "fixture bna6"
    app_ann.session_state.loaded_settings_hash = "fixture data"
    app_ann.session_state.inference_result = res_nbs
    app_ann.session_state.edges_df = detected
    app_ann.session_state.annotation = {
        "rois": rois,
        "summary": summary,
        "detected": detected,
        "support": "point",
        "level": "lobe",
        "alpha": 0.05,
        "param_idx": None,
    }
    app_ann.session_state.run_plan = {
        "method": "nbs",
        "alpha": 0.05,
        "loaded_settings_hash": "fixture data",
        "active_atlas_signature": "fixture bna6",
    }
    app_ann.session_state.active_tab = "4. Brainnetome Annotation"
    app_ann.run()
    assert not app_ann.exception
    captions = "\n".join(c.value for c in app_ann.caption)
    markdowns = "\n".join(m.value for m in app_ann.markdown)
    assert "ROI selection rule (component unit)" in captions
    assert "does not test node/hub significance" in captions
    assert "Aggregated Component Annotation Context" in markdowns
    ann_dls = [b.proto.label for b in app_ann.get("download_button")]
    assert any("Component context · CSV" in lbl for lbl in ann_dls)
