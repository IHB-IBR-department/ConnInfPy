"""Regressions for navigation and loaded/draft dataset separation; no inference."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from apps.utils import helpers

APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"


def app_with_demo():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    next(r for r in app.radio if r.label == "Select Ingestion Method").set_value("Built-in data").run()
    next(b for b in app.button if b.label == "Load, Preprocess & Validate Data").click().run()
    assert not app.exception
    return app


def saved_analysis(app):
    app.session_state.inference_result = "saved result"
    app.session_state.edges_df = pd.DataFrame()
    app.session_state.run_plan = {
        "method": "tstat", "alpha": 0.05,
        "loaded_settings_hash": app.session_state.loaded_settings_hash,
        "active_atlas_signature": app.session_state.loaded_atlas_signature,
    }


def test_round_trip_keeps_source_data_and_finished_analysis():
    app = app_with_demo()
    data = app.session_state.connectivity_data
    signature = app.session_state.loaded_settings_hash
    saved_analysis(app)
    app.button(key="nav_docs_btn").click().run()
    app.button(key="nav_step_1").click().run()
    assert not app.exception
    assert next(r for r in app.radio if r.label == "Select Ingestion Method").value == "Built-in data"
    assert app.session_state.connectivity_data is data
    assert app.session_state.loaded_settings_hash == signature
    assert app.session_state.current_settings_hash == signature
    assert app.session_state.inference_result == "saved result"
    assert app.session_state.dataset_info["name"] == "ABIDE-mini"
    assert app.button(key="nav_step_3").proto.help == "No detections"


def test_draft_source_change_keeps_loaded_provenance_and_results():
    app = app_with_demo()
    data = app.session_state.connectivity_data
    saved_analysis(app)
    next(r for r in app.radio if r.label == "Select Ingestion Method").set_value("Generate Synthetic Dataset").run()
    assert not app.exception
    assert app.session_state.connectivity_data is data
    assert app.session_state.inference_result == "saved result"
    assert app.session_state.dataset_info["name"] == "ABIDE-mini"
    assert app.session_state.current_settings_hash != app.session_state.loaded_settings_hash
    assert any("Active dataset: ABIDE-mini" in m.value for m in app.markdown)
    assert not any("Synthetic Generator" in m.value for m in app.markdown)


def test_explicit_reload_clears_results_and_completed_task():
    app = app_with_demo()
    saved_analysis(app)
    app.session_state.inference_task = SimpleNamespace(status="completed", result="old result")
    next(b for b in app.button if b.label == "Load, Preprocess & Validate Data").click().run()
    assert not app.exception
    assert app.session_state.inference_result is None
    assert app.session_state.run_plan is None
    assert "inference_task" not in app.session_state
    assert app.session_state.dataset_info["name"] == "ABIDE-mini"


def test_design_inputs_survive_visiting_another_step():
    app = app_with_demo()
    app.button(key="nav_step_2").click().run()
    assert not app.exception
    assert app.button(key="run_inference_btn").label == "🚀 Run Connectivity Inference"
    integration = next(s for s in app.selectbox if s.label == "Integration")
    assert integration.value == "exact"
    integration.set_value("right-grid").run()
    steps = next(n for n in app.number_input if n.label == "Integration steps (n)")
    assert steps.value == 50
    steps.set_value(25).run()
    budget = next(r for r in app.radio if r.label == "Permutation Budget Preset:")
    budget.set_value("Custom").run()
    next(n for n in app.number_input if n.label == "Number of Permutations (B)").set_value(300).run()
    app.button(key="nav_docs_btn").click().run()
    app.button(key="nav_step_2").click().run()
    assert not app.exception
    assert next(r for r in app.radio if r.label == "Permutation Budget Preset:").value == "Custom"
    assert next(n for n in app.number_input if n.label == "Number of Permutations (B)").value == 300
    assert next(s for s in app.selectbox if s.label == "Integration").value == "right-grid"
    assert next(n for n in app.number_input if n.label == "Integration steps (n)").value == 25


def test_design_top_actions_show_running_and_completed_states():
    app = app_with_demo()
    app.button(key="nav_step_2").click().run()
    app.session_state.inference_task = SimpleNamespace(status="running", progress_message="Test progress")
    app.run()
    assert not app.exception
    assert app.button(key="run_inference_running").disabled
    assert any("Test progress" in message.value for message in app.info)
    saved_analysis(app)
    app.session_state.inference_task.status = "success"
    app.run()
    assert not app.exception
    assert app.button(key="run_inf_new_button").label == "🚀 Configure & Run New Inference"
    assert app.button(key="design_continue_results").label == "Continue to Results →"


def test_roi_draft_does_not_change_the_active_analysis():
    app = app_with_demo()
    saved_analysis(app)
    next(r for r in app.radio if r.label == "ROI Selection Input Method").set_value("Manual List (Comma-separated)").run()
    roi_input = next(t for t in app.text_input if t.label.startswith("ROI Indices"))
    roi_input.set_value("0,1,2").run()
    assert app.session_state.roi_indices is None
    assert app.session_state.inference_result == "saved result"
    next(t for t in app.text_input if t.label.startswith("ROI Indices")).set_value("1000").run()
    assert next(b for b in app.button if b.label == "Load, Preprocess & Validate Data").disabled
    assert app.session_state.connectivity_data.shape == (203, 100, 100)


def test_changing_sidebar_atlas_keeps_the_loaded_atlas():
    app = app_with_demo()
    loaded_atlas = app.session_state.loaded_reference_atlas
    loaded_signature = app.session_state.loaded_atlas_signature
    app.radio(key="atlas_mode").set_value("No atlas metadata").run()
    assert not app.exception
    assert app.session_state.loaded_reference_atlas is loaded_atlas
    assert app.session_state.loaded_atlas_signature == loaded_signature
    assert app.session_state.active_atlas_signature != loaded_signature
    assert app.session_state.connectivity_data.shape == (203, 100, 100)
    app.button(key="nav_step_2").click().run()
    assert not app.exception


def test_preprocessing_choices_are_kept_per_source():
    app = app_with_demo()
    next(c for c in app.checkbox if c.label == "Fisher r-to-z transform").set_value(False).run()
    next(r for r in app.radio if r.label == "Select Ingestion Method").set_value("Dataset Manifest (data.yaml)").run()
    assert not app.exception
    assert next(c for c in app.checkbox if c.label == "Fisher r-to-z transform").value is True
    next(r for r in app.radio if r.label == "Select Ingestion Method").set_value("Built-in data").run()
    assert next(c for c in app.checkbox if c.label == "Fisher r-to-z transform").value is False
    assert app.session_state.connectivity_data_kind == "fisher_z"


@pytest.mark.parametrize("paradigm", ["Two-Group Contrast", "Two-Group Contrast with Confound"])
def test_synthetic_group_sizes_variability_and_output_scale(paradigm):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.radio(key="atlas_mode").set_value("No atlas metadata").run()
    next(n for n in app.number_input if n.label == "Number of ROIs / Nodes").set_value(10).run()
    next(r for r in app.radio if r.label == "Simulation Paradigm").set_value(paradigm).run()
    next(n for n in app.number_input if n.label == "Subjects in group 1").set_value(10).run()
    next(n for n in app.number_input if n.label == "Subjects in group 2").set_value(15).run()
    assert any("Group 2 (group = 1)" in caption.value and "Group 1 (group = 0)" in caption.value for caption in app.caption)
    if paradigm.endswith("with Confound"):
        next(s for s in app.slider if s.label == "Confound Effect Size").set_value(0.0).run()
    assert not any(b.key == "btn_continue_to_design" for b in app.button)
    assert app.button(key="generate_dataset_btn").label == "Generate Synthetic Dataset"
    app.button(key="generate_dataset_btn").click().run()
    assert not app.exception
    assert app.button(key="btn_continue_to_design").label == "Continue to Design & Inference →"
    assert app.button(key="generate_dataset_btn").label == "↻ Generate again"
    original = app.session_state.connectivity_data.copy()
    assert original.shape == (25, 10, 10)
    assert app.session_state.pheno_df.group.value_counts().to_dict() == {0: 10, 1: 15}
    assert (app.session_state.pheno_df.group.to_numpy() == np.repeat([0, 1], [10, 15])).all()
    assert app.session_state.connectivity_data_kind == "fisher_z"

    next(n for n in app.number_input if n.label == "Group 2 added variance (%)").set_value(100).run()
    next(c for c in app.checkbox if c.label == "Apply Fisher r-to-z transform").set_value(False).run()
    assert app.session_state.current_settings_hash != app.session_state.loaded_settings_hash
    assert app.button(key="generate_dataset_btn").label == "↻ Generate with new settings"
    assert not any(b.key == "btn_continue_to_design" for b in app.button)
    app.button(key="generate_dataset_btn").click().run()
    assert not app.exception
    assert app.session_state.connectivity_data_kind == "correlation"
    changed = np.arctanh(app.session_state.connectivity_data)
    np.testing.assert_allclose(changed[:10], original[:10], atol=1e-12)
    assert not np.allclose(changed[10:], original[10:])
    np.testing.assert_allclose(changed, changed.transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(np.diagonal(changed, axis1=1, axis2=2), 0, atol=1e-12)
    assert app.session_state.current_settings_hash == app.session_state.loaded_settings_hash
    app.button(key="btn_continue_to_design").click().run()
    assert not app.exception
    assert app.session_state.active_tab == "2. Design & Inference"
    assert any("generator Group 1 = 0" in caption.value and "generator Group 2 = 1" in caption.value for caption in app.caption)


def test_synthetic_seed_reproduces_and_replaces_data_only_on_success():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.radio(key="atlas_mode").set_value("No atlas metadata").run()
    next(n for n in app.number_input if n.label == "Number of ROIs / Nodes").set_value(10).run()
    next(n for n in app.number_input if n.label == "Subjects in group 1").set_value(10).run()
    next(n for n in app.number_input if n.label == "Subjects in group 2").set_value(10).run()
    app.button(key="generate_dataset_btn").click().run()
    first = app.session_state.connectivity_data.copy()
    assert app.session_state.dataset_info["seed"] == 42
    app.button(key="generate_dataset_btn").click().run()
    np.testing.assert_array_equal(app.session_state.connectivity_data, first)
    next(n for n in app.number_input if n.label == "Random seed").set_value(43).run()
    assert app.button(key="generate_dataset_btn").label == "↻ Generate with new settings"
    app.button(key="generate_dataset_btn").click().run()
    assert not app.exception
    assert not np.array_equal(app.session_state.connectivity_data, first)
    assert app.session_state.dataset_info["seed"] == 43
    saved_data = app.session_state.connectivity_data
    saved_analysis(app)
    with patch("apps.components.data_ingestion.TopologyDatasetGenerator", side_effect=RuntimeError("generation failed")):
        app.button(key="generate_dataset_btn").click().run()
    assert app.exception
    assert app.session_state.connectivity_data is saved_data
    assert app.session_state.inference_result == "saved result"


def test_synthetic_added_variance_is_independent_and_centered():
    from apps.components.data_ingestion import _add_group_variance

    data = np.zeros((20000, 3, 3))
    result = _add_group_variance(data, 100, np.random.default_rng(123))
    assert abs(result[:, 0, 1].mean()) < 0.005
    assert abs(result[:, 0, 1].var() - 1 / 27) < 0.002
    assert abs(np.corrcoef(result[:-1, 0, 1], result[1:, 0, 1])[0, 1]) < 0.03
    np.testing.assert_array_equal(result, result.transpose(0, 2, 1))


def test_synthetic_continuous_covariate_allows_correlation_output():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.radio(key="atlas_mode").set_value("No atlas metadata").run()
    next(n for n in app.number_input if n.label == "Number of ROIs / Nodes").set_value(10).run()
    next(r for r in app.radio if r.label == "Simulation Paradigm").set_value("Continuous Covariate GLM").run()
    next(n for n in app.number_input if n.label == "Total Subjects").set_value(20).run()
    next(c for c in app.checkbox if c.label == "Apply Fisher r-to-z transform").set_value(False).run()
    app.button(key="generate_dataset_btn").click().run()
    assert not app.exception
    assert app.session_state.connectivity_data.shape == (20, 10, 10)
    assert app.session_state.connectivity_data_kind == "correlation"
    assert np.max(np.abs(app.session_state.connectivity_data)) < 1
    assert {"interest", "confounds"} <= set(app.session_state.pheno_df.columns)


@pytest.mark.parametrize("step", [2, 3, 4])
def test_empty_steps_lead_directly_to_loading_data(step):
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.button(key=f"nav_step_{step}").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Load data").click().run()
    assert app.session_state.active_tab == "1. Data Ingestion & Preprocessing"


def test_navigation_reaches_annotation_and_docs_without_narrative_controls():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    keys = {button.key for button in app.button}
    assert {f"nav_step_{step}" for step in (1, 2, 3, 4)} <= keys
    assert "nav_docs_btn" in keys
    assert not any("narrative" in (button.key or "").lower() or "llm" in button.label.lower()
                   for button in app.button)
    app.button(key="nav_step_4").click().run()
    assert not app.exception
    assert app.session_state.active_tab == "4. Brainnetome Annotation"
    app.button(key="nav_docs_btn").click().run()
    assert not app.exception
    assert app.session_state.active_tab == "6. Workspace Documentation"


def test_editing_a_draft_does_not_make_the_loaded_result_stale():
    state = {
        "loaded_settings_hash": "loaded", "current_settings_hash": "edited draft",
        "loaded_atlas_signature": "loaded atlas", "active_atlas_signature": "different draft atlas",
    }
    plan = {"loaded_settings_hash": "loaded", "active_atlas_signature": "loaded atlas"}
    with patch.object(helpers.st, "session_state", state):
        assert not helpers.result_is_stale(plan)
        state["loaded_settings_hash"] = "replacement dataset"
        assert helpers.result_is_stale(plan)


@pytest.mark.parametrize("tested_tail", [None, "positive"])
def test_result_graph_direction_selection_and_one_sided_policy(tested_tail):
    from conninfpy.atlas import AtlasInfo
    from conninfpy.results import InferenceResult
    import matplotlib.pyplot as plt

    atlas = AtlasInfo(labels=["A", "B", "C"], networks=["N", "N", "N"], coords=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    p_positive = np.ones((3, 3))
    p_positive[0, 1] = p_positive[1, 0] = 0.01
    stat = np.array([[0, 4, 0], [4, 0, 0], [0, 0, 0]])
    result = InferenceResult(p_positive, np.ones((3, 3)), method="tstat", stat_positive=stat, stat_negative=np.zeros((3, 3)), tested_tail=tested_tail)
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((4, 3, 3))
    app.session_state.pheno_df = pd.DataFrame({"subject_id": ["a", "b", "c", "d"], "group": [0, 0, 1, 1]})
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = result
    app.session_state.edges_df = result.significant_edges(atlas=atlas)
    app.session_state.run_plan = {"method": "tstat", "loaded_settings_hash": "fixture data", "active_atlas_signature": "fixture atlas", "question_type": "Group Difference", "ref_group": 0, "target_group": 1}
    app.session_state.active_tab = "3. Inference Results"
    with patch("apps.components.inference_results.plot_connectome_graph", side_effect=lambda **kwargs: plt.figure()):
        app.run()
        assert not app.exception
        direction = next(r for r in app.radio if r.label == "Effect direction")
        assert "1 edges" in direction.options[0]
        if tested_tail is None:
            assert len(direction.options) == 2
            direction.set_value(direction.options[1]).run()
            assert not app.exception
            assert any("No edges for" in message.value for message in app.info)
        else:
            assert len(direction.options) == 1
            assert any("was tested" in caption.value and "change the hypothesis direction" in caption.value for caption in app.caption)


@pytest.mark.parametrize("has_detection", [True, False])
def test_completed_annotation_renders_and_survives_navigation(tmp_path, has_detection):
    import nibabel as nib
    from conninfpy.atlas.brainnetome import BrainnetomeReference
    from conninfpy.interpret.annotate import annotate_bna_regions, summarize_annotation
    from conninfpy.results import InferenceResult

    # Four synthetic homologous regions, fixed p-values: rendering only, no external atlas needed.
    rows = [
        {
            "roi_id": rid,
            "name": name,
            "hemisphere": hemi,
            "homolog_id": hom,
            "code": f"C_{rid}",
            "description": f"Synthetic {name}",
            "gyrus": "SFG",
            "gyrus_name": "Superior Frontal Gyrus",
            "lobe": "Frontal",
            "structure": "cortex",
            "yeo7_id": 6,
            "yeo7": "Frontoparietal",
            "yeo17_id": 12,
            "yeo17": "Control A",
            "x": x,
            "y": 15.0,
            "z": 54.0,
        }
        for rid, name, hemi, hom, x in (
            (1, "A8m_L", "L", 2, -5.0),
            (2, "A8m_R", "R", 1, 7.0),
            (3, "A8dl_L", "L", 4, -18.0),
            (4, "A8dl_R", "R", 3, 18.0),
        )
    ]
    regions = pd.DataFrame(rows).set_index("roi_id")
    vol = np.zeros((5, 1, 1), dtype=np.int16)
    vol[0, 0, 0], vol[1, 0, 0], vol[2, 0, 0], vol[3, 0, 0] = 1, 2, 3, 4
    reference = BrainnetomeReference(
        regions=regions,
        label_image=nib.Nifti1Image(vol, np.eye(4)),
        data_dir=tmp_path,
        cache_dir=tmp_path,
        profiles=None,
        provenance={"atlas": "synthetic BNA fixture"},
    )
    rois = annotate_bna_regions([1, 2, 3, 4], reference=reference)
    p = np.ones((4, 4))
    p[0, 1] = p[1, 0] = 0.01 if has_detection else 1.0
    stat = np.array([
        [0, 4.0 if has_detection else 0.5, 0.3, 0.1],
        [4.0 if has_detection else 0.5, 0, 1.2, 0.2],
        [0.3, 1.2, 0, 0.4],
        [0.1, 0.2, 0.4, 0],
    ])
    result = InferenceResult(p, np.ones((4, 4)), method="tstat", stat_positive=stat, stat_negative=np.zeros((4, 4)))
    summary = summarize_annotation(result, rois, alpha=0.05)
    edges = summary["edges"]
    detected = edges[(edges.p_positive <= 0.05) | (edges.p_negative <= 0.05)]
    annotation = {"rois": rois, "summary": summary, "detected": detected, "support": "point", "level": "lobe", "alpha": 0.05, "param_idx": None}

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((4, 4, 4))
    app.session_state.pheno_df = pd.DataFrame({"subject_id": ["a", "b", "c", "d"], "group": [0, 0, 1, 1]})
    app.session_state.loaded_reference_atlas = rois.to_atlas()
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.inference_result = result
    app.session_state.edges_df = detected
    app.session_state.annotation = annotation
    app.session_state.run_plan = {"method": "tstat", "loaded_settings_hash": "fixture data", "active_atlas_signature": "fixture atlas"}
    app.session_state.active_tab = "4. Brainnetome Annotation"
    app.run()
    assert not app.exception
    assert next(m for m in app.metric if m.label == "Detected connections").value == str(int(has_detection))
    assert next(e for e in app.expander if e.label == "Annotation settings").proto.expanded is False
    assert next(r for r in app.radio if r.label == "ROI signal source").value == "point"
    if has_detection:
        assert not any("No detected connections" in i.value for i in app.info)
    else:
        assert any("analysis is complete" in i.value for i in app.info)
    app.button(key="nav_docs_btn").click().run()
    app.button(key="nav_step_4").click().run()
    assert not app.exception
    assert app.session_state.annotation is annotation
