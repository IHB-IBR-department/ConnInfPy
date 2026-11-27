"""NETSTAT-21 tests: shared defaults across UI, worker dispatch, and docs.

Also hosts the NETSTAT-86 fast public-API default regression (signature
inspection plus the tail-policy resolver — no simulation, no UI).
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from conninfpy import defaults

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "conninfpy" / "app" / "streamlit_app.py"
DESIGN_INFERENCE = REPO / "conninfpy" / "app" / "components" / "design_inference.py"


def _design_page():
    from conninfpy.atlas import AtlasInfo

    n_rois = 8
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state.connectivity_data = np.zeros((6, n_rois, n_rois))
    app.session_state.connectivity_data_kind = "fisher_z"
    app.session_state.pheno_df = pd.DataFrame(
        {
            "subject_id": [f"sub{i}" for i in range(6)],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    app.session_state.loaded_reference_atlas = AtlasInfo(
        labels=[f"ROI{i}" for i in range(n_rois)],
        networks=["Vis"] * 4 + ["DMN"] * 4,
    )
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.active_tab = "2. Design & Inference"
    return app.run()


def _by_label(elements, label):
    matches = [el for el in elements if el.proto.label == label]
    assert matches, f"Widget with label {label!r} not found"
    return matches[0]


def _load_ast_worker():
    tree = ast.parse(DESIGN_INFERENCE.read_text(encoding="utf-8"))
    names = {"_run_inference_background", "_method_plan_parameters"}
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(module, str(DESIGN_INFERENCE), "exec"), namespace)
    return namespace["_run_inference_background"]


def test_ui_fresh_basic_and_advanced_tfnbs_defaults():
    app = _design_page()
    assert not app.exception

    assert _by_label(app.selectbox, "Select method profile:").value == "Balanced default (TFNBS)"
    assert _by_label(app.number_input, "Extent exponent (E)").value == pytest.approx(
        defaults.DEFAULT_EXTENT_EXPONENT
    )
    assert _by_label(app.number_input, "Height exponent (H)").value == pytest.approx(
        defaults.DEFAULT_HEIGHT_EXPONENT
    )
    expected_integration = (
        "exact" if defaults.DEFAULT_N_THRESHOLDS_PERMUTATION is None else "right-grid"
    )
    assert _by_label(app.selectbox, "Integration").value == expected_integration
    assert _by_label(app.number_input, "Start threshold (t)").value == pytest.approx(
        defaults.DEFAULT_START_THRESHOLD
    )
    assert (
        _by_label(app.radio, "Permutation Budget Preset:").value
        == f"Exploratory ({defaults.DEFAULT_N_PERMUTATIONS} permutations)"
    )
    expected_accel = (
        "none" if defaults.DEFAULT_ACCELERATION is None else defaults.DEFAULT_ACCELERATION
    )
    assert _by_label(app.selectbox, "Tail Acceleration").value == expected_accel

    _by_label(app.radio, "Permutation Budget Preset:").set_value("Custom")
    app = app.run()
    assert not app.exception
    assert _by_label(app.number_input, "Number of Permutations (B)").value == (
        defaults.DEFAULT_N_PERMUTATIONS
    )


def test_ui_fbc_tfnbs_and_nbs_defaults_and_bounds():
    app = _design_page()
    assert not app.exception

    _by_label(app.selectbox, "Select method profile:").set_value(
        "Diffuse block effect (FBC-TFNBS/cNBS)"
    )
    app = app.run()
    assert not app.exception

    assert _by_label(app.radio, "Choose specific block method:").value == "fbc_tfnbs"
    m_min = _by_label(app.number_input, "Minimum block size (m_min)")
    assert m_min.value == defaults.DEFAULT_MIN_CLUSTER_SIZE
    assert m_min.min <= defaults.DEFAULT_MIN_CLUSTER_SIZE

    # Explicit custom value (including min_cluster_size=3 or 2) is preserved.
    m_min.set_value(2)
    app = app.run()
    assert not app.exception
    assert _by_label(app.number_input, "Minimum block size (m_min)").value == 2

    _by_label(app.selectbox, "Select method profile:").set_value(
        "Classical connected component (NBS)"
    )
    app = app.run()
    assert not app.exception
    assert _by_label(app.number_input, "Cluster-forming threshold (tau)").value == pytest.approx(
        defaults.DEFAULT_NBS_THRESHOLD
    )
    assert _by_label(app.selectbox, "NBS statistic").value == defaults.DEFAULT_NBS_STAT


def test_ui_companion_defaults_and_explicit_overrides():
    app = _design_page()
    assert not app.exception

    _by_label(app.checkbox, "Include Sensitivity Companion Analysis").set_value(True)
    app = app.run()
    assert not app.exception

    comp_select = _by_label(app.selectbox, "Select Companion Baseline Method:")
    comp_select.set_value("fbc_tfnbs")
    app = app.run()
    assert not app.exception

    assert _by_label(app.number_input, "Companion Extent exponent (E)").value == pytest.approx(
        defaults.DEFAULT_EXTENT_EXPONENT
    )
    assert _by_label(app.number_input, "Companion Height exponent (H)").value == pytest.approx(
        defaults.DEFAULT_HEIGHT_EXPONENT
    )
    assert _by_label(app.selectbox, "Companion integration").value == "exact"
    assert _by_label(app.number_input, "Companion start threshold (t)").value == pytest.approx(
        defaults.DEFAULT_START_THRESHOLD
    )
    comp_m_min = _by_label(app.number_input, "Companion Minimum block size (m_min)")
    assert comp_m_min.value == defaults.DEFAULT_MIN_CLUSTER_SIZE
    assert comp_m_min.min <= defaults.DEFAULT_MIN_CLUSTER_SIZE

    comp_select = _by_label(app.selectbox, "Select Companion Baseline Method:")
    comp_select.set_value("nbs")
    app = app.run()
    assert not app.exception
    assert _by_label(
        app.number_input, "Companion Cluster-forming threshold (tau)"
    ).value == pytest.approx(defaults.DEFAULT_NBS_THRESHOLD)
    assert _by_label(app.selectbox, "Companion NBS statistic").value == defaults.DEFAULT_NBS_STAT

    # Explicit overrides persist across reruns.
    _by_label(app.selectbox, "Tail Acceleration").set_value("gpd")
    _by_label(app.number_input, "Extent exponent (E)").set_value(0.5)
    _by_label(app.number_input, "Companion Cluster-forming threshold (tau)").set_value(3.2)
    app = app.run()
    assert not app.exception
    assert _by_label(app.selectbox, "Tail Acceleration").value == "gpd"
    assert _by_label(app.number_input, "Extent exponent (E)").value == pytest.approx(0.5)
    assert _by_label(
        app.number_input, "Companion Cluster-forming threshold (tau)"
    ).value == pytest.approx(3.2)


@pytest.mark.parametrize("use_ast_loader", [False, True])
def test_worker_fallback_dispatch_uses_shared_defaults(use_ast_loader):
    if use_ast_loader:
        worker = _load_ast_worker()
    else:
        from conninfpy.app.components.design_inference import _run_inference_background as worker

    fake_result = SimpleNamespace(significant_edges=lambda **kw: [])
    atlas = SimpleNamespace(network_index=lambda: [0, 0, 1, 1])

    for primary_method, comp_method in [
        ("tfnbs", "fbc_tfnbs"),
        ("fbc_tfnbs", "nbs"),
        ("nbs", "tfnbs"),
    ]:
        args = {name: None for name in inspect.signature(worker).parameters}
        task = SimpleNamespace()
        args.update(
            task=task,
            data_kind="fisher_z",
            test_type="one-sample",
            operator_choice=primary_method,
            op_kwargs={},
            run_sensitivity=True,
            comp_method=comp_method,
            comp_kwargs={},
            atlas=atlas,
            n_perms=None,
            acceleration_choice="none",
        )
        with patch("conninfpy.analyze", return_value=fake_result) as run:
            worker(**args)
        assert task.status == "success", getattr(task, "error", None)
        assert run.call_count == 2

        for call, method in zip(run.call_args_list, (primary_method, comp_method), strict=True):
            assert call.kwargs["n_permutations"] == defaults.DEFAULT_N_PERMUTATIONS
            assert call.kwargs["acceleration"] is defaults.DEFAULT_ACCELERATION
            if method in ("tfnbs", "fbc_tfnbs"):
                assert call.kwargs["e"] == defaults.DEFAULT_EXTENT_EXPONENT
                assert call.kwargs["h"] == defaults.DEFAULT_HEIGHT_EXPONENT
                assert call.kwargs["n"] == defaults.DEFAULT_N_THRESHOLDS_PERMUTATION
                assert call.kwargs["start_thres"] == defaults.DEFAULT_START_THRESHOLD
            if method == "fbc_tfnbs":
                assert call.kwargs["min_cluster_size"] == defaults.DEFAULT_MIN_CLUSTER_SIZE
            if method == "nbs":
                assert call.kwargs["threshold"] == defaults.DEFAULT_NBS_THRESHOLD
                assert call.kwargs["nbs_stat"] == defaults.DEFAULT_NBS_STAT


def test_worker_preserves_explicit_overrides():
    worker = _load_ast_worker()
    fake_result = SimpleNamespace(significant_edges=lambda **kw: [])
    atlas = SimpleNamespace(network_index=lambda: [0, 0, 1, 1])
    args = {name: None for name in inspect.signature(worker).parameters}
    task = SimpleNamespace()
    args.update(
        task=task,
        data_kind="fisher_z",
        test_type="one-sample",
        operator_choice="fbc_tfnbs",
        op_kwargs={"e": 0.5, "h": 2.5, "n": 10, "start_thres": 2.0, "min_cluster_size": 7},
        run_sensitivity=True,
        comp_method="nbs",
        comp_kwargs={"threshold": 3.4, "nbs_stat": "intensity"},
        atlas=atlas,
        n_perms=200,
        acceleration_choice="gpd",
    )
    with patch("conninfpy.analyze", return_value=fake_result) as run:
        worker(**args)

    assert task.status == "success"
    p_kw, c_kw = [c.kwargs for c in run.call_args_list]
    assert p_kw["n_permutations"] == 200
    assert p_kw["acceleration"] == "gpd"
    assert p_kw["e"] == 0.5
    assert p_kw["h"] == 2.5
    assert p_kw["n"] == 10
    assert p_kw["start_thres"] == 2.0
    assert p_kw["min_cluster_size"] == 7
    assert c_kw["n_permutations"] == 200
    assert c_kw["acceleration"] == "gpd"
    assert c_kw["threshold"] == 3.4
    assert c_kw["nbs_stat"] == "intensity"


def test_readme_and_start_thres_docstring_accuracy():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "TFNBS `e=0.3`" not in readme
    assert "`e=0.4`, `h=3.0`" in readme
    assert "`n=None`" in readme
    assert "`acceleration=None`" in readme
    assert "`two_tailed=True`" in readme
    assert "`two_tailed=False`" in readme

    assert defaults.DEFAULT_START_THRESHOLD == 1.65
    defaults_src = (REPO / "conninfpy" / "defaults.py").read_text(encoding="utf-8")
    assert "asymptotic normal" in defaults_src


# ---------------------------------------------------------------------------
# NETSTAT-86: fast public-API default regression (no simulation, no UI).
# ---------------------------------------------------------------------------


def test_public_api_signature_defaults():
    import conninfpy

    p = inspect.signature(conninfpy.compute_p_val).parameters
    assert p["two_tailed"].default is True
    assert p["n_permutations"].default == defaults.DEFAULT_N_PERMUTATIONS == 1000
    assert p["threshold"].default == defaults.DEFAULT_NBS_THRESHOLD == 3.0
    assert p["nbs_stat"].default == defaults.DEFAULT_NBS_STAT == "extent"
    assert p["e"].default == defaults.DEFAULT_EXTENT_EXPONENT == 0.4
    assert p["h"].default == defaults.DEFAULT_HEIGHT_EXPONENT == 3.0
    assert p["n"].default is None
    assert p["acceleration"].default is defaults.DEFAULT_ACCELERATION is None

    g = inspect.signature(conninfpy.compute_p_val_glm).parameters
    assert g["two_tailed"].default is True
    assert g["n_permutations"].default == defaults.DEFAULT_N_PERMUTATIONS
    assert g["threshold"].default == defaults.DEFAULT_NBS_THRESHOLD
    assert g["nbs_stat"].default == defaults.DEFAULT_NBS_STAT
    assert g["e"].default == defaults.DEFAULT_EXTENT_EXPONENT
    assert g["h"].default == defaults.DEFAULT_HEIGHT_EXPONENT
    assert g["n"].default is None
    assert g["acceleration"].default is None

    a = inspect.signature(conninfpy.analyze).parameters
    # analyze defers the tail decision: two_tailed=None resolves to two-sided
    # unless tested_tail prespecifies a direction (functional check below).
    assert a["two_tailed"].default is None
    assert a["tested_tail"].default is None
    assert a["n_permutations"].default == defaults.DEFAULT_N_PERMUTATIONS
    assert a["e"].default == defaults.DEFAULT_EXTENT_EXPONENT
    assert a["h"].default == defaults.DEFAULT_HEIGHT_EXPONENT
    assert a["n"].default is None
    assert a["acceleration"].default is None

    # BCT-compatible NBS wrapper: two-sided max-statistic null is built in
    # (no two_tailed parameter); threshold/stat_type/n_permutations defaults
    # must stay aligned with the shared constants.
    b = inspect.signature(conninfpy.nbs_bct).parameters
    assert b["threshold"].default == defaults.DEFAULT_NBS_THRESHOLD == 3.0
    assert b["stat_type"].default == defaults.DEFAULT_NBS_STAT == "extent"
    assert b["n_permutations"].default == defaults.DEFAULT_N_PERMUTATIONS


def test_analyze_tail_policy_none_means_two_sided():
    from conninfpy._analysis_stages import _resolve_tail_policy

    state = SimpleNamespace(tested_tail=None, two_tailed=None, method_kwargs={}, flags=[])
    _resolve_tail_policy(state)
    assert state.effective_two_tailed is True

    state = SimpleNamespace(tested_tail=None, two_tailed=False, method_kwargs={}, flags=[])
    _resolve_tail_policy(state)
    assert state.effective_two_tailed is False

    state = SimpleNamespace(
        tested_tail="positive", two_tailed=None, method_kwargs={}, flags=[]
    )
    _resolve_tail_policy(state)
    assert state.effective_two_tailed is False


def test_docs_conf_version_matches_package():
    import runpy

    import conninfpy

    conf = runpy.run_path(str(REPO / "docs" / "source" / "conf.py"))
    assert conf["release"] == conninfpy.__version__
    assert conf["version"] == ".".join(conninfpy.__version__.split(".")[:2])
