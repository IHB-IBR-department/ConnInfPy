"""The results step uses the α recorded in the run plan for every view.

The design step fixes α at launch; summary counters, the significance
heatmap and the NBS components view must all read the same value instead of
a hardcoded 0.05.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conninfpy import compute_p_val
from conninfpy.atlas import AtlasInfo

APP = Path(__file__).resolve().parents[1] / "conninfpy" / "app" / "streamlit_app.py"
N = 8


def _result_with_p(p_positive):
    """A two-sample tstat result whose positive p-map is replaced by hand."""
    rng = np.random.default_rng(11)

    def sym(a):
        a = (a + a.transpose(0, 2, 1)) / 2
        a[:, np.arange(N), np.arange(N)] = 0.0
        return a

    g1, g2 = sym(rng.normal(size=(12, N, N))), sym(rng.normal(size=(12, N, N)))
    res = compute_p_val(g1, g2, test_type="two-sample", var_equal=True, method="tstat",
                        n_permutations=60, rng=3, use_mp=False, acceleration=None,
                        two_tailed=True)
    res["positive"] = np.asarray(p_positive, dtype=float)
    res["negative"] = np.ones_like(res["negative"])
    return res


ATLAS = AtlasInfo(labels=[f"ROI{i}" for i in range(N)],
                  networks=["Vis"] * 2 + ["DMN"] * 2 + ["SMN"] * 4)
ATLAS_COORDS = AtlasInfo(labels=[f"ROI{i}" for i in range(N)],
                         networks=["Vis"] * 2 + ["DMN"] * 2 + ["SMN"] * 4,
                         coords=np.column_stack([np.arange(N) * 1.0, np.zeros(N), np.zeros(N)]))


def _results_page(result, alpha):
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((24, N, N))
    app.session_state.pheno_df = pd.DataFrame(
        {"subject_id": list("abcd"), "group": [0, 0, 1, 1]})
    app.session_state.loaded_reference_atlas = ATLAS
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = result
    app.session_state.edges_df = result.significant_edges(atlas=ATLAS, alpha=alpha)
    app.session_state.run_plan = {
        "method": result.method, "alpha": alpha, "loaded_settings_hash": "fixture data",
        "active_atlas_signature": "fixture atlas", "question_type": "Group Difference",
        "reference_group": 0, "target_group": 1,
    }
    app.session_state.active_tab = "3. Inference Results"
    return app.run()


def _p_map():
    """Positive-tail p-values: two edges at 0.02 (sig at .05, not at .01)."""
    p = np.ones((N, N))
    p[0, 1] = p[1, 0] = 0.02
    p[2, 3] = p[3, 2] = 0.02
    return p


def app_metric(app):
    """(label, value) of the rendered summary metrics."""
    metrics = app.get("metric")
    assert metrics, "summary metrics not rendered"
    return metrics


def _metric_map(app):
    return {m.proto.label: m.value for m in app_metric(app)}


def test_summary_counters_follow_the_run_alpha():
    result = _result_with_p(_p_map())
    strict = _results_page(result, alpha=0.01)
    assert not strict.exception
    metrics = _metric_map(strict)
    # At α = 0.01 the 0.02 edges are not detections.
    assert metrics["Group 1 > Group 0"] == "0"
    caption = next(c.value for c in strict.caption if "Detections at" in c.value)
    assert "α = 0.01" in caption
    loose = _results_page(result, alpha=0.05)
    assert not loose.exception
    metrics05 = _metric_map(loose)
    assert metrics05["Group 1 > Group 0"] == "2"
    caption05 = next(c.value for c in loose.caption if "Detections at" in c.value)
    assert "α = 0.05" in caption05


def test_graph_cutoff_slider_defaults_to_the_run_alpha():
    """The brain-graph p-value cutoff starts at the run's α, not 0.05."""
    result = _result_with_p(_p_map())
    app = _results_page_coords(result, alpha=0.05)
    assert not app.exception
    cutoff = next(s for s in app.get("slider") if "cutoff" in s.proto.label)
    assert list(cutoff.proto.default) == [0.05]


def _results_page_coords(result, alpha):
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((24, N, N))
    app.session_state.pheno_df = pd.DataFrame(
        {"subject_id": list("abcd"), "group": [0, 0, 1, 1]})
    app.session_state.loaded_reference_atlas = ATLAS_COORDS
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = result
    app.session_state.edges_df = result.significant_edges(atlas=ATLAS_COORDS, alpha=alpha)
    app.session_state.run_plan = {
        "method": result.method, "alpha": alpha, "loaded_settings_hash": "fixture data",
        "active_atlas_signature": "fixture atlas", "question_type": "Group Difference",
        "reference_group": 0, "target_group": 1,
    }
    app.session_state.active_tab = "3. Inference Results"
    return app.run()


def test_two_sided_family_caption_explains_the_family():
    result = _result_with_p(_p_map())
    app = _results_page(result, alpha=0.05)
    assert not app.exception
    caption = next(c.value for c in app.caption if "Detections at" in c.value)
    assert "Two-sided family" in caption
    assert "jointly" in caption


def test_graph_cutoff_slider_resets_when_a_new_run_changes_alpha():
    """A new result with a different run alpha overrides stale graph slider state."""
    p = np.ones((N, N))
    p[0, 1] = p[1, 0] = 0.005
    p[2, 3] = p[3, 2] = 0.02

    res1 = _result_with_p(p)
    app = _results_page_coords(res1, alpha=0.05)
    assert not app.exception
    cutoff = next(s for s in app.get("slider") if "cutoff" in s.proto.label)
    assert cutoff.value == 0.05

    # Adjusting the slider within the same run is preserved on rerun.
    cutoff.set_value(0.03).run()
    assert not app.exception
    cutoff_adj = next(s for s in app.get("slider") if "cutoff" in s.proto.label)
    assert cutoff_adj.value == 0.03

    # A new run on the same dataset with alpha = 0.01 resets the slider to 0.01.
    res2 = _result_with_p(p)
    app.session_state.inference_result = res2
    app.session_state.edges_df = res2.significant_edges(atlas=ATLAS_COORDS, alpha=0.01)
    app.session_state.run_plan = {
        **dict(app.session_state.run_plan),
        "alpha": 0.01,
    }
    app.run()
    assert not app.exception
    cutoff2 = next(s for s in app.get("slider") if "cutoff" in s.proto.label)
    assert cutoff2.value == 0.01


def test_annotation_alpha_follows_new_run_alpha():
    """Tab 4 detection threshold defaults to the new run's alpha after a result update."""
    from conninfpy.app.utils.workspace import ANNOTATION_TAB

    res1 = _result_with_p(_p_map())
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((24, N, N))
    app.session_state.pheno_df = pd.DataFrame(
        {"subject_id": list("abcd"), "group": [0, 0, 1, 1]})
    app.session_state.loaded_reference_atlas = ATLAS_COORDS
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = res1
    app.session_state.edges_df = res1.significant_edges(atlas=ATLAS_COORDS, alpha=0.05)
    app.session_state.run_plan = {
        "method": res1.method, "alpha": 0.05, "loaded_settings_hash": "fixture data",
        "active_atlas_signature": "fixture atlas", "question_type": "Group Difference",
        "reference_group": 0, "target_group": 1,
    }
    app.session_state.active_tab = ANNOTATION_TAB
    app.run()
    assert not app.exception
    ni1 = next(w for w in app.get("number_input") if "Detection threshold" in w.proto.label)
    assert ni1.value == pytest.approx(0.05)

    res2 = _result_with_p(_p_map())
    app.session_state.inference_result = res2
    app.session_state.edges_df = res2.significant_edges(atlas=ATLAS_COORDS, alpha=0.01)
    app.session_state.annotation = None
    app.session_state.run_plan = {
        **dict(app.session_state.run_plan),
        "alpha": 0.01,
    }
    app.run()
    assert not app.exception
    ni2 = next(w for w in app.get("number_input") if "Detection threshold" in w.proto.label)
    assert ni2.value == pytest.approx(0.01)


def test_comparative_view_labels_untested_tail_not_tested():
    """Side-by-side comparison reports 'not tested' rather than 0 for an excluded tail."""
    rng = np.random.default_rng(11)

    def sym(a):
        a = (a + a.transpose(0, 2, 1)) / 2
        a[:, np.arange(N), np.arange(N)] = 0.0
        return a

    g1, g2 = sym(rng.normal(size=(12, N, N))), sym(rng.normal(size=(12, N, N)))
    base_p = compute_p_val(
        g1, g2, test_type="two-sample", var_equal=True, method="tstat",
        n_permutations=40, rng=3, use_mp=False, acceleration=None,
        two_tailed=False,
    )
    base_c = compute_p_val(
        g1, g2, test_type="two-sample", var_equal=True, method="bh_fdr",
        use_mp=False, two_tailed=False,
    )
    pre_primary = base_p._restricted_to_tail("positive", selection="prespecified")
    pre_comp = base_c._restricted_to_tail("positive", selection="prespecified")

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((24, N, N))
    app.session_state.pheno_df = pd.DataFrame(
        {"subject_id": list("abcd"), "group": [0, 0, 1, 1]})
    app.session_state.loaded_reference_atlas = ATLAS
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = pre_primary
    app.session_state.edges_df = pre_primary.significant_edges(atlas=ATLAS, alpha=0.05)
    app.session_state.companion_inference_result = pre_comp
    app.session_state.companion_edges_df = pre_comp.significant_edges(atlas=ATLAS, alpha=0.05)
    app.session_state.companion_method = "bh_fdr"
    app.session_state.run_plan = {
        "method": "tstat", "companion_method": "bh_fdr", "alpha": 0.05,
        "loaded_settings_hash": "fixture data", "active_atlas_signature": "fixture atlas",
        "question_type": "Group Difference", "reference_group": 0, "target_group": 1,
        "tested_tail": "positive", "tail_selection": "prespecified", "two_tailed": False,
    }
    app.session_state.active_tab = "3. Inference Results"
    app.run()
    assert not app.exception
    radio = next(r for r in app.get("radio") if "Result View Mode" in r.proto.label)
    radio.set_value("Side-by-Side Comparison").run()
    assert not app.exception
    md_text = "\n".join(m.value for m in app.markdown)
    assert "(not tested):** `not tested`" in md_text
