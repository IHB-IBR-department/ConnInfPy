"""Scientific presentation regressions found during independent acceptance."""

import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from conninfpy.app.utils.result_views import (
    build_null_max_figure, check_companion_compatibility,
    describe_result_scale_and_stat, prepare_predicted_coupling_summary,
    recorded_native_detection_count,
)
from conninfpy import analyze
from conninfpy.enhance.tfnbs import get_tfnbs_score
from conninfpy.results import InferenceResult


@pytest.fixture
def result():
    rng = np.random.default_rng(230)
    data = rng.normal(size=(14, 4, 4))
    data = (data + data.transpose(0, 2, 1)) / 2
    x, z = rng.normal(size=(2, 14))
    options = dict(method="tstat", input="z", r_to_z=False,
                   n_permutations=13, rng=6, use_mp=False)
    return analyze(data, interest=x, **options), analyze(data, interest=x, confounds=z, **options)


def test_enhanced_null_never_marks_raw_t_as_observed_score():
    raw = np.array([[0., 3., 2., 0.], [3., 0., 2.5, 0.],
                    [2., 2.5, 0., 0.], [0., 0., 0., 0.]])
    # The score is a different quantity, even on this tiny known component.
    enhanced_max = get_tfnbs_score(raw, e=.4, h=3., n=None).max()
    assert enhanced_max > 7 * raw.max()
    for method in ("tfnbs", "nbs", "cnbs", "ni_tfnbs", "fbc_tfnbs", "tstat"):
        saved = np.array([0., 10., 20., 30.])
        res = InferenceResult(np.ones_like(raw), np.ones_like(raw), method=method,
                              stat_positive=raw, stat_negative=np.zeros_like(raw),
                              null_max_dist={"positive": saved})
        before = saved.copy()
        fig, info = build_null_max_figure(res)
        try:
            lines = [line for ax in fig.axes for line in ax.lines
                     if line.get_label().startswith("observed max")]
            if method == "tstat":
                assert len(lines) == 1 and lines[0].get_xdata()[0] == raw.max()
                assert info["observed_max_note"] is None
            else:
                assert not lines and info["observed_max_note"]
                assert any("not saved" in text.get_text() for text in fig.texts)
            np.testing.assert_array_equal(saved, before)
        finally:
            plt.close(fig)


def test_companion_different_contrast_shapes_are_incompatible_not_an_exception(result):
    a, b = result
    assert np.asarray(a.inference.design["contrast"]).shape != np.asarray(b.inference.design["contrast"]).shape
    report = check_companion_compatibility(a, b)
    assert report["compatible"] is False
    assert any("Contrast vector" in reason for reason in report["reasons"])


def test_companion_unknown_legacy_provenance_does_not_certify_a_match(result):
    legacy = copy.deepcopy(result[0].inference)
    legacy.analysis_provenance = None
    legacy.design = None
    report = check_companion_compatibility(legacy, copy.deepcopy(legacy))
    assert report["compatible"] is False
    assert any("not recorded" in reason for reason in report["reasons"])


def _native_result(base, unit):
    r = copy.deepcopy(base.inference)
    n = 4
    p = np.ones((n, n))
    if unit == "component":
        labels = np.zeros((n, n), dtype=int)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            labels[i, j] = labels[j, i] = 1
            p[i, j] = p[j, i] = .02
        r.method = "nbs"
        r.component_report = {
            "threshold": 2., "edge_statistic": "extent", "correction": "fwer",
            "labels": {"positive": labels, "negative": np.zeros_like(labels)},
            "components": [{"tail": "positive", "component_id": 1, "p_fwer": .02}],
        }
    else:
        r.method = "cnbs"
        p[:2, 2:] = p[2:, :2] = .02
        r.block_report = {
            "correction": "bonferroni", "node_labels": [0, 0, 1, 1],
            "blocks": [
                {"network_a": 0, "network_b": 0, "p_two_sided": .5, "n_edges": 1, "mean_stat": 0.},
                {"network_a": 0, "network_b": 1, "p_two_sided": .02, "n_edges": 4, "mean_stat": 3.},
                {"network_a": 1, "network_b": 1, "p_two_sided": .5, "n_edges": 1, "mean_stat": 0.},
            ],
        }
    r["positive"] = p
    r["negative"] = np.ones_like(p)
    return r


@pytest.mark.parametrize("unit,neither", [("component", 0), ("block", 2)])
def test_native_agreement_counts_units_instead_of_projected_edges(result, unit, neither):
    r = _native_result(result[0], unit)
    report = check_companion_compatibility(r, copy.deepcopy(r))
    assert report["compatible"] is True, report
    assert report["inference_unit"] == unit
    # Three component edges or four block edges are each ONE inference unit.
    assert report["summary"]["both"] == 1
    assert report["summary"]["neither"] == neither
    assert report["summary"]["jaccard"] == 1
    assert recorded_native_detection_count(r, .05) == 1
    other = copy.deepcopy(r)
    if unit == "component":
        other.component_report["labels"]["positive"][0, 2] = 0
        other.component_report["labels"]["positive"][2, 0] = 0
    else:
        other.block_report["node_labels"] = [0, 1, 0, 1]
    mismatch = check_companion_compatibility(r, other)
    assert mismatch["compatible"] is False
    assert any("memberships differ" in reason for reason in mismatch["reasons"])


def test_untested_edges_are_excluded_from_neither_count_and_marked(result):
    r = copy.deepcopy(result[0].inference)
    tested = np.zeros((4, 4), dtype=bool)
    tested[0, 1] = tested[1, 0] = True
    r.analysis_provenance["tested"] = {"positive": tested.tolist(), "negative": tested.tolist()}
    r["positive"][:] = r["negative"][:] = 1.
    report = check_companion_compatibility(r, copy.deepcopy(r))
    assert report["compatible"] is True, report
    assert report["summary"]["neither"] == 1
    assert report["categories"][0, 1] == 0
    assert report["categories"][0, 2] == -1


def test_difference_response_profiles_keep_native_units_and_source_roi_ids():
    rng = np.random.default_rng(231)
    data = rng.normal(size=(16, 4, 4))
    data = (data + data.transpose(0, 2, 1)) / 2
    source_ids = [101, 205, 9, 44]
    result = analyze(group1=data[:8], group2=data[8:], interest=np.linspace(-1, 1, 8),
                     test_type="paired", input="z", method="tstat", roi_ids=source_ids,
                     n_permutations=13, rng=3, use_mp=False)
    summary = prepare_predicted_coupling_summary(result, alpha=.001)
    assert summary["available"] and summary["scale"] == "difference_fisher_z"
    assert "difference" in summary["effect_unit_label"]
    assert "difference-only model" in summary["scale_explanation"]
    edges = summary["edge_predictions_df"]
    assert "Δr (r_B - r_A)" not in edges and "coupling_class" not in edges
    assert summary["coupling_counts_df"] is None
    assert "Δ (Fisher z difference units)" in edges
    for _, row in edges.iterrows():
        assert row["roi_id_i"] == source_ids[int(row["roi_i"])]
        assert row["roi_id_j"] == source_ids[int(row["roi_j"])]
    result.inference.analysis_provenance["units"] = "native calibrated units"
    assert "native calibrated units" in describe_result_scale_and_stat(result)["effect_unit_label"]


@pytest.mark.parametrize("unit", ["component", "block"])
def test_real_streamlit_headline_and_agreement_use_native_counts(result, unit):
    from conninfpy.atlas import AtlasInfo
    from streamlit.testing.v1 import AppTest

    native = _native_result(result[0], unit)
    app = AppTest.from_string('''
import streamlit as st
from conninfpy.app.components.inference_results import _render_summary_metrics, _render_comparative_view
r = st.session_state['result']
atlas = st.session_state['atlas']
edges = r.significant_edges(atlas=atlas, alpha=.05)
labels = {'positive':'positive', 'negative':'negative'}
_render_summary_metrics(r, edges, r.method, labels)
_render_comparative_view(r, edges, r, edges, 'A', 'B', atlas, None, labels)
''', default_timeout=30)
    app.session_state['result'] = native
    app.session_state['atlas'] = AtlasInfo(labels=['a', 'b', 'c', 'd'], networks=['Unknown']*4)
    app.session_state['run_plan'] = {'alpha': .05}
    app.run()
    assert not app.exception, [e.message for e in app.exception]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics[f"Significant {unit.capitalize()}s"] == '1'
    assert metrics[f"Both Significant ({unit}s)"] == '1'
