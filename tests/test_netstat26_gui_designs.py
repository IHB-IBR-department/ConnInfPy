"""NETSTAT-26: guided GUI exposure of already-supported package designs.

One scientific question per run. No new statistical engine and no change to the
numerical inference: the tests below check that the guided interface passes
exactly what it shows (paired condition arrays with real subject IDs and
condition-specific confounds, missing-connectivity policy, tested edge family,
declared primary (E, H) cell), that unsupported combinations are blocked before
the run, and that the captured run plan, workspace and archive round-trip.

Synthetic data only; no hosted services.
"""

from __future__ import annotations

import ast
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conninfpy.stats.pairwise import _check_edge_mask

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "conninfpy" / "app" / "streamlit_app.py"


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------
def _corr_stack(n, N=5, seed=3):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, N, N))
    for s in range(n):
        a = rng.standard_normal((N, 20))
        c = np.corrcoef(a)
        np.fill_diagonal(c, 1.0)
        out[s] = c
    return out


def _paired_pheno(order=("s3", "s1", "s2")):
    """Interleaved two-condition phenotype; rows deliberately not sorted."""
    rows = []
    for k, subject in enumerate(order):
        rows.append((subject, "rest", 0.10 + 0.05 * k, "S1"))
        rows.append((subject, "task", 0.40 + 0.10 * k, "S1"))
    return pd.DataFrame(
        rows, columns=["subject_id", "condition", "Motion_FD", "site"]
    ).iloc[[3, 0, 4, 1, 5, 2]].reset_index(drop=True)


def _build_paired_arrays(Y=None, pheno=None, **overrides):
    from conninfpy.app.components.design_inference import _build_design_arrays

    pheno = _paired_pheno() if pheno is None else pheno
    Y = _corr_stack(len(pheno)) if Y is None else Y
    kwargs = dict(
        test_type="paired",
        question_choice="Paired Condition",
        condition_col="condition",
        subject_col="subject_id",
        baseline_val="rest",
        target_val="task",
        paired_confound_cols=["Motion_FD"],
        require_paired_bijection=True,
    )
    kwargs.update(overrides)
    return _build_design_arrays(Y, pheno, **kwargs)


def _worker_namespace():
    """Load the worker exactly as the app does (module-level names available)."""
    from conninfpy.app.components import design_inference

    return design_inference


def _run_worker(task, **overrides):
    """Call the real background worker with patched analyze() and capture calls."""
    from conninfpy.app.components.design_inference import _run_inference_background

    calls = []
    captured = {}

    def fake_analyze(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            significant_edges=lambda **kw: (captured.__setitem__("edges_kwargs", kw) or []),
            wall_time_s=0.01,
            fisher=None,
            design={},
            is_grid=False,
            e_grid=None,
            h_grid=None,
        )

    args = {name: None for name in inspect.signature(_run_inference_background).parameters}
    args.update(
        task=task,
        test_type="paired",
        operator_choice="tfnbs",
        n_perms=20,
        seed_val=42,
        use_mp=False,
        acceleration_choice="none",
        op_kwargs={"e": 0.4, "h": 2.0, "n": 50, "start_thres": 1.0},
        run_sensitivity=False,
        alpha=0.05,
        question_choice="Paired Condition",
        loaded_settings_hash="fixture",
        recipe_choice="Paired within-subject",
        effective_recipe_choice="Paired within-subject",
        subject_col="subject_id",
        condition_col="condition",
        baseline_val="rest",
        target_val="task",
        confound_vars=[],
        active_atlas_signature="fixture atlas",
        data_kind="fisher_z",
    )
    args.update(overrides)
    with patch("conninfpy.analyze", side_effect=fake_analyze):
        _run_inference_background(**args)
    return calls


# ----------------------------------------------------------------------
# 1. Paired design: real IDs, condition A order, condition-specific confounds
# ----------------------------------------------------------------------
def test_paired_arrays_keep_condition_a_order_and_are_not_pre_differenced():
    pheno = _paired_pheno()
    Y = _corr_stack(len(pheno))
    arrays = _build_paired_arrays(Y, pheno)

    a_rows = pheno.index[pheno["condition"] == "rest"].tolist()
    assert arrays["idx_baseline"] == a_rows
    assert arrays["paired_subjects"] == list(pheno.loc[a_rows, "subject_id"])
    # condition B follows condition A's subject order, not the file order
    assert arrays["idx_target"] == [
        int(pheno.index[(pheno["condition"] == "task") & (pheno["subject_id"] == s)][0])
        for s in arrays["paired_subjects"]
    ]
    np.testing.assert_allclose(arrays["group1"], Y[arrays["idx_baseline"]])
    np.testing.assert_allclose(arrays["group2"], Y[arrays["idx_target"]])
    np.testing.assert_allclose(
        arrays["confounds_group1"],
        pheno.loc[arrays["idx_baseline"], ["Motion_FD"]].to_numpy(float),
    )
    np.testing.assert_allclose(
        arrays["confounds_group2"],
        pheno.loc[arrays["idx_target"], ["Motion_FD"]].to_numpy(float),
    )
    assert arrays["ids_group1"] == arrays["ids_group2"] == arrays["paired_subjects"]


def test_worker_passes_condition_arrays_ids_and_confounds_without_a_gui_delta():
    from conninfpy.app.components.design_inference import InferenceTask

    pheno = _paired_pheno()
    Y = _corr_stack(len(pheno))
    arrays = _build_paired_arrays(Y, pheno)
    task = InferenceTask()
    calls = _run_worker(
        task,
        group1=arrays["group1"],
        group2=arrays["group2"],
        ids_group1=arrays["ids_group1"],
        ids_group2=arrays["ids_group2"],
        confounds_group1=arrays["confounds_group1"],
        confounds_group2=arrays["confounds_group2"],
        design_facts={
            "condition_confound_cols": ["Motion_FD"],
            "n_paired_subjects": 3,
            "paired_subject_order": "condition A (baseline) phenotype order; condition B matched one-to-one by subject ID",
        },
    )
    assert task.status == "success", task.error
    kwargs = calls[0]
    np.testing.assert_allclose(kwargs["group1"], Y[arrays["idx_baseline"]])
    np.testing.assert_allclose(kwargs["group2"], Y[arrays["idx_target"]])
    assert kwargs["ids_group1"] == arrays["paired_subjects"]
    assert kwargs["ids_group2"] == arrays["paired_subjects"]
    np.testing.assert_allclose(
        kwargs["confounds_group1"], pheno.loc[arrays["idx_baseline"], ["Motion_FD"]].to_numpy(float)
    )
    np.testing.assert_allclose(
        kwargs["confounds_group2"], pheno.loc[arrays["idx_target"], ["Motion_FD"]].to_numpy(float)
    )
    # The B - A difference is the package's job: no delta table is handed in.
    assert kwargs["confounds"] is None
    assert task.run_plan["condition_confound_cols"] == ["Motion_FD"]
    assert task.run_plan["n_paired_subjects"] == 3
    assert "condition A" in task.run_plan["paired_subject_order"]


def test_paired_design_rejections_name_the_offending_subjects():
    from conninfpy.app.components.design_inference import PairedDesignError

    pheno = _paired_pheno()
    Y = _corr_stack(len(pheno))

    duplicate = pd.concat([pheno, pheno.iloc[[0]]], ignore_index=True)
    with pytest.raises(PairedDesignError) as dup:
        _build_paired_arrays(_corr_stack(len(duplicate)), duplicate)
    assert "duplicate subject IDs" in str(dup.value) and "s1" in str(dup.value)

    # The empty ID names the affected condition directly (unit-level check:
    # the app's required-field filter would remove such a row first).
    from conninfpy.app.components.design_inference import _paired_design_violations

    blank = pheno.copy()
    blank.loc[0, "subject_id"] = None
    empty_problems = _paired_design_violations(
        blank, "condition", "subject_id", "rest", "task"
    )
    assert any("empty subject ID" in problem for problem in empty_problems)
    # Through the sliced design the pair is reported as one-sided, naming s1.
    with pytest.raises(PairedDesignError) as empty:
        _build_paired_arrays(Y, blank)
    assert "s1" in str(empty.value)

    one_sided = pheno[
        ~((pheno["subject_id"] == "s2") & (pheno["condition"] == "task"))
    ].reset_index(drop=True)
    with pytest.raises(PairedDesignError) as partial:
        _build_paired_arrays(_corr_stack(len(one_sided)), one_sided)
    assert "measured only in" in str(partial.value) and "s2" in str(partial.value)

    # Incompatible sizes with the same subject sets: one extra repeat row.
    extra = pd.concat([pheno, pheno.iloc[[1]]], ignore_index=True)
    with pytest.raises(PairedDesignError):
        _build_paired_arrays(_corr_stack(len(extra)), extra)


def test_missing_condition_confound_drops_the_whole_pair_with_a_count():
    pheno = _paired_pheno()
    pheno.loc[pheno["subject_id"] == "s1", "Motion_FD"] = np.nan
    arrays = _build_paired_arrays(_corr_stack(len(pheno)), pheno)
    assert arrays["n_pairs_dropped_confound"] == 1
    assert arrays["paired_subjects"] == ["s3", "s2"]
    assert arrays["confounds_group1"].shape == (2, 1)
    assert np.isfinite(arrays["confounds_group2"]).all()


def test_legacy_direct_callers_keep_the_permissive_slicing():
    """Without the flag the documented historical slicing is unchanged."""
    pheno = _paired_pheno().assign(subject_id=["a", "b", "c", "d", "e", "f"])
    arrays = _build_paired_arrays(
        _corr_stack(len(pheno)), pheno, require_paired_bijection=False
    )
    assert arrays["paired_subjects"] == []  # no subject is in both conditions
    assert arrays["group1"].shape[0] == 0


# ----------------------------------------------------------------------
# 2. Missing design fields vs missing connectivity
# ----------------------------------------------------------------------
def test_missing_design_rows_are_excluded_before_the_run_and_counted():
    pheno = _paired_pheno()
    pheno.loc[pheno["subject_id"] == "s2", "condition"] = None  # required design field
    arrays = _build_paired_arrays(_corr_stack(len(pheno)), pheno)
    assert arrays["n_dropped"] == 2
    assert arrays["paired_subjects"] == ["s3", "s1"]


def test_available_case_policy_and_coverage_are_forwarded_and_recorded():
    from conninfpy.app.components.design_inference import InferenceTask

    pheno = _paired_pheno()
    Y = _corr_stack(len(pheno))
    arrays = _build_paired_arrays(Y, pheno)

    task = InferenceTask()
    calls = _run_worker(
        task,
        group1=arrays["group1"],
        group2=arrays["group2"],
        missing_policy="available_case",
        min_coverage=0.9,
    )
    assert task.status == "success", task.error
    assert calls[0]["missing"] == "available_case"
    assert calls[0]["min_coverage"] == pytest.approx(0.9)
    assert task.run_plan["missing_policy"] == "available_case"
    assert task.run_plan["min_coverage"] == pytest.approx(0.9)

    task_default = InferenceTask()
    default_calls = _run_worker(
        task_default, group1=arrays["group1"], group2=arrays["group2"]
    )
    assert default_calls[0]["missing"] == "raise"
    assert task_default.run_plan["missing_policy"] == "raise"
    assert task_default.run_plan["min_coverage"] is None


def test_available_case_result_surfaces_coverage_reasons_and_unit():
    from conninfpy import analyze

    n, N = 10, 4
    Y = _corr_stack(n, N)
    Y[0, 0, 1] = Y[0, 1, 0] = np.nan
    Y[1, 2, 3] = Y[1, 3, 2] = np.nan
    group1 = Y[::2]
    group2 = Y[1::2]
    res = analyze(
        group1=group1, group2=group2, test_type="paired", method="tstat",
        input="z", r_to_z=False, n_permutations=20, rng=1, use_mp=False,
        acceleration=None, missing="available_case", min_coverage=0.9,
    )
    report = res.inference.missing_report
    assert report["policy"] == "available_case"
    assert report["min_coverage"] == pytest.approx(0.9)
    assert report["n_edges_untested"] >= 1
    assert np.asarray(report["edge_n"]).max() > np.asarray(report["edge_n"]).min() or (
        report["edge_n_min"] == int(np.asarray(report["edge_n"]).max())
    )
    assert "subjects_with_missing" in report


# ----------------------------------------------------------------------
# 3. Edge family mask: passed through, recorded as metadata, not as an array
# ----------------------------------------------------------------------
def test_edge_mask_reaches_analyze_and_run_plan_records_metadata_only():
    from conninfpy.app.components.design_inference import InferenceTask, _edge_mask_metadata

    pheno = _paired_pheno()
    Y = _corr_stack(len(pheno))
    arrays = _build_paired_arrays(Y, pheno)
    nodes = arrays["group1"].shape[1]
    raw = np.zeros((nodes, nodes), dtype=bool)
    raw[0, 1] = raw[1, 0] = True
    raw[2, 3] = raw[3, 2] = True
    mask = _check_edge_mask(raw, nodes)

    task = InferenceTask()
    calls = _run_worker(
        task,
        group1=arrays["group1"],
        group2=arrays["group2"],
        op_kwargs={"e": 0.4, "h": 2.0, "n": 50, "start_thres": 1.0, "edge_mask": mask},
        edge_mask_plan=_edge_mask_metadata(mask, "family.npy"),
    )
    assert task.status == "success", task.error
    np.testing.assert_array_equal(calls[0]["edge_mask"], mask)

    recorded = task.run_plan["edge_mask"]
    assert recorded["purpose"] == "edge_family_mask"
    assert recorded["eligible_edges"] == 2
    assert recorded["shape"] == [nodes, nodes]
    assert recorded["source_filename"] == "family.npy"
    assert recorded["sha256"] == _edge_mask_metadata(mask)["sha256"]
    # The full array never enters operator_kwargs (it is an array, not a fact).
    assert "edge_mask" not in task.run_plan["operator_kwargs"]
    # Distinct from ROI exclusion and SC community labels.
    assert task.run_plan["roi_indices"] is None
    assert task.run_plan["net_labels"] is None


def test_edge_mask_validation_reuses_engine_semantics():
    bad = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        _check_edge_mask(bad, 4)  # selects no edge
    asymmetric = np.zeros((4, 4), dtype=bool)
    asymmetric[0, 1] = True
    with pytest.raises(ValueError):
        _check_edge_mask(asymmetric, 4)
    with pytest.raises(ValueError):
        _check_edge_mask(np.ones((3, 3), dtype=bool), 4)


# ----------------------------------------------------------------------
# 4. E/H sensitivity grid with a declared primary cell
# ----------------------------------------------------------------------
def test_grid_parsing_and_range_validation():
    from conninfpy.app.components.design_inference import _parse_float_list, _validate_eh_grid

    assert _parse_float_list("0.4, 0.5")[0] == [0.4, 0.5]
    assert _parse_float_list("")[1]
    assert "not a number" in _parse_float_list("0.4, x")[1]
    assert _validate_eh_grid([0.4, 0.5], [2.0, 3.0], (0.1, 2.0), (1.0, 5.0)) == []
    assert _validate_eh_grid([0.4, 0.5], [2.0], (0.1, 2.0), (1.0, 5.0))
    assert _validate_eh_grid([3.0], [2.0], (0.1, 2.0), (1.0, 5.0))


def test_declared_primary_cell_is_the_completion_table():
    from conninfpy.app.components.design_inference import InferenceTask

    task = InferenceTask()
    grid_result = SimpleNamespace(
        significant_edges=lambda **kw: (calls.append(kw) or []),
        is_grid=True,
        e_grid=np.array([0.5, 0.75]),
        h_grid=np.array([2.0, 2.0]),
        wall_time_s=0.01,
        fisher=None,
        design={},
    )
    calls = []
    from conninfpy.app.components.design_inference import _run_inference_background

    args = {name: None for name in inspect.signature(_run_inference_background).parameters}
    args.update(
        task=task, test_type="paired", operator_choice="tfnbs", n_perms=20, seed_val=42,
        use_mp=False, acceleration_choice="none",
        op_kwargs={"e": [0.5, 0.75], "h": [2.0, 2.0], "n": 50, "start_thres": 1.0},
        run_sensitivity=False, alpha=0.05, question_choice="Paired Condition",
        subject_col="subject_id", condition_col="condition",
        baseline_val="rest", target_val="task", confound_vars=[],
        active_atlas_signature="fixture", data_kind="fisher_z",
        group1=np.zeros((3, 4, 4)), group2=np.zeros((3, 4, 4)),
        primary_cell={"e": 0.75, "h": 2.0, "param_idx": 1, "selection": "declared_before_run"},
    )
    with patch("conninfpy.analyze", return_value=grid_result):
        _run_inference_background(**args)
    assert task.status == "success", task.error
    assert calls[0]["param_idx"] == 1
    assert task.run_plan["primary_cell"]["param_idx"] == 1
    assert task.run_plan["primary_cell"]["e"] == pytest.approx(0.75)
    assert task.run_plan["primary_cell"]["resolved_from"] == "result_e_h_grid"


def test_run_plan_is_stable_and_changes_only_with_the_parameters():
    from conninfpy.app.components.design_inference import InferenceTask

    pheno = _paired_pheno()
    arrays = _build_paired_arrays(_corr_stack(len(pheno)), pheno)
    base = dict(group1=arrays["group1"], group2=arrays["group2"], min_coverage=0.8)
    first, second, changed = InferenceTask(), InferenceTask(), InferenceTask()
    _run_worker(first, missing_policy="available_case", **base)
    _run_worker(second, missing_policy="available_case", **base)
    _run_worker(changed, missing_policy="available_case", min_coverage=0.95,
                group1=arrays["group1"], group2=arrays["group2"])

    def comparable(plan):
        drop = {"started_at", "wall_time_s", "python_version", "package_version"}
        return json.dumps({k: v for k, v in plan.items() if k not in drop}, sort_keys=True, default=str)

    assert comparable(first.run_plan) == comparable(second.run_plan)
    assert comparable(first.run_plan) != comparable(changed.run_plan)
    assert changed.run_plan["min_coverage"] == pytest.approx(0.95)


# ----------------------------------------------------------------------
# 5. Replay and archive capture
# ----------------------------------------------------------------------
def _plan_with_new_facts():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    from conninfpy.app.components.design_inference import _edge_mask_metadata

    return {
        "method": "tfnbs",
        "question_type": "Paired Condition",
        "design_family": "paired",
        "condition_col": "condition",
        "subject_col": "subject_id",
        "baseline_condition": "rest",
        "target_condition": "task",
        "condition_confound_cols": ["Motion_FD"],
        "n_paired_subjects": 12,
        "alpha": 0.05,
        "seed": 42,
        "n_permutations": 1000,
        "tested_tail": None,
        "two_tailed": True,
        "input": "z",
        "use_mp": False,
        "missing_policy": "available_case",
        "min_coverage": 0.8,
        "materialized": True,
        "primary_cell": {"e": 0.5, "h": 2.0, "param_idx": 1, "selection": "declared_before_run"},
        "sensitivity_grid": {"e": [0.5, 0.75], "h": [2.0, 2.0], "n_cells": 2},
        "edge_mask": _edge_mask_metadata(mask, "family.npy"),
        "operator_kwargs": {"e": [0.5, 0.75], "h": [2.0, 2.0], "n": 50, "start_thres": 1.0},
        "method_parameters": {"E": [0.5, 0.75], "H": [2.0, 2.0]},
    }


def _replay_dataset_info(tmp_path: Path):
    manifest = tmp_path / "study.yaml"
    manifest.write_text("paths: {}\n", encoding="utf-8")
    (tmp_path / "data.npy").write_bytes(b"stub")
    return {
        "mode": "Manifest",
        "name": "fixture",
        "source": str(manifest),
        "manifest_file": str(manifest),
        "data_kind": "fisher_z",
        "loader_provenance": {"data_path": str(tmp_path / "data.npy")},
    }


def test_replay_config_captures_available_case_grid_and_mask(tmp_path: Path):
    from conninfpy.app.utils.archive_export import build_replay_config

    dataset_info = _replay_dataset_info(tmp_path)
    cfg = build_replay_config(_plan_with_new_facts(), dataset_info, project_root=tmp_path)

    assert cfg["config"]["inference"]["missing"] == "available_case"
    assert cfg["config"]["inference"]["min_coverage"] == pytest.approx(0.8)
    # The CLI's paired schema reads condition-specific confounds from design.confounds.
    assert cfg["config"]["design"]["confounds"] == ["Motion_FD"]
    assert cfg["declared_primary_cell"]["param_idx"] == 1
    kinds = {p["kind"] for p in cfg["prerequisites"]}
    assert "edge-family-mask" in kinds
    assert any("edge family" in reason for reason in cfg["reasons"])
    # The mask itself is never inlined into the YAML config.
    assert "edge_mask" not in json.dumps(cfg["config"])

    # The produced analysis file passes the CLI's own schema validation.
    import yaml
    from conninfpy.cli import load_config

    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg["config"]), encoding="utf-8")
    loaded = load_config(str(cfg_path))
    assert loaded["design"]["type"] == "paired"
    assert loaded["inference"]["missing"] == "available_case"
    assert loaded["inference"]["min_coverage"] == pytest.approx(0.8)


def test_legacy_plan_without_min_coverage_stays_not_replayable(tmp_path: Path):
    from conninfpy.app.utils.archive_export import build_replay_config

    plan = _plan_with_new_facts()
    plan.pop("min_coverage")
    cfg = build_replay_config(plan, _replay_dataset_info(tmp_path), project_root=tmp_path)
    assert cfg["status"] == "not-replayable"
    assert any("min_coverage was not captured" in r for r in cfg["reasons"])


def test_archive_bundles_the_mask_and_the_workspace_round_trips_the_facts(tmp_path: Path):
    from conninfpy import analyze
    from conninfpy.app.utils.archive_export import build_analysis_archive
    from conninfpy.app.utils.workspace_io import (
        build_workspace_archive,
        inspect_workspace_archive,
        restore_workspace_archive,
    )
    import io
    import zipfile

    n, N = 10, 4
    Y = _corr_stack(n, N, seed=11)
    group = np.array([0, 1] * (n // 2), dtype=float)
    res = analyze(
        Y, interest=group, method="tstat", input="z", r_to_z=False,
        n_permutations=10, rng=1, use_mp=False, acceleration=None,
    )
    plan = _plan_with_new_facts()
    plan["method"] = "tstat"
    plan["design_family"] = "glm"
    plan["operator_kwargs"] = {}
    plan["primary_cell"] = None
    mask = np.zeros((N, N), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    state = {
        "inference_result": res,
        "run_plan": plan,
        "dataset_info": {"mode": "Generate Synthetic Dataset", "name": "fixture"},
        "design:edge_mask_array": mask,
        "design:edge_mask_source": "family.npy",
        "_workspace_widget_keys": {
            "design:edge_mask_array": "design",
            "design:edge_mask_source": "design",
        },
        "results:grid_idx:primary": 1,
        "active_tab": "3. Inference Results",
    }
    zip_bytes, _manifest = build_analysis_archive(
        res, plan, state["dataset_info"], edge_family_mask=mask, workspace_state=state,
        include_bundle=False,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        assert "arrays/edge_family_mask.npy" in names
        stored = np.load(io.BytesIO(zf.read("arrays/edge_family_mask.npy")))
        np.testing.assert_array_equal(stored, mask)
        run_plan_json = json.loads(zf.read("run_plan.json"))
        assert run_plan_json["edge_mask"]["purpose"] == "edge_family_mask"
        assert "edge_mask_array" not in json.dumps(run_plan_json)

    ws_bytes, _ = build_workspace_archive(state, include_data=False)
    inspection = inspect_workspace_archive(ws_bytes)
    assert inspection.summary
    target = {}
    restore_workspace_archive(ws_bytes, target)
    assert target["run_plan"]["missing_policy"] == "available_case"
    assert target["run_plan"]["primary_cell"] is None
    assert target["results:grid_idx:primary"] == 1
    np.testing.assert_array_equal(target["design:edge_mask_array"], mask)
    assert target["design:edge_mask_source"] == "family.npy"


# ----------------------------------------------------------------------
# 6. Guided GUI: blocked combinations, declared primary, e2e run + reload
# ----------------------------------------------------------------------
def _design_app(method_profile="Balanced default (TFNBS)", question="Group Difference"):
    from conninfpy.atlas import AtlasInfo

    n, N = 8, 5
    Y = _corr_stack(n, N, seed=5)
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n)],
            "group": ["ctrl"] * 4 + ["case"] * 4,
            "Motion_FD": np.linspace(0.05, 0.5, n),
        }
    )
    atlas = AtlasInfo(
        labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N,
        coords=np.column_stack([np.arange(N) * 15.0, np.zeros(N), np.zeros(N)]),
    )
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state.connectivity_data = Y
    app.session_state.connectivity_data_kind = "fisher_z"
    app.session_state.pheno_df = pheno
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.active_tab = "2. Design & Inference"
    return app.run()


def test_design_page_shows_the_new_controls():
    app = _design_app()
    assert not app.exception
    policy = next(r for r in app.radio if "Missing connectivity policy" in r.proto.label)
    assert policy.value.startswith("Complete-case")
    assert any("required design fields" in c.value for c in app.caption)
    grid = next(c for c in app.checkbox if "E/H sensitivity grid" in c.proto.label)
    assert grid.value is False
    assert any("tested family" in c.value for c in app.caption)
    assert any("Tested edge family" in m.value for m in app.markdown)


def test_design_page_blocks_mask_with_unsupported_method_and_available_case():
    app = _design_app()
    N = 5
    mask = np.zeros((N, N), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    app.session_state["design:edge_mask_array"] = mask
    app.session_state["design:edge_mask_source"] = "family.npy"
    app.run()
    assert not app.exception
    assert not app.button(key="run_inference_btn").disabled
    assert any("Tested edge family" in i.value for i in app.info)

    # Unsupported method: BH-FDR has no tested-family mask in the engine.
    next(s for s in app.selectbox if s.label == "Select method profile:").set_value(
        "Fast edge-wise screen (BH-FDR)"
    ).run()
    assert not app.exception
    assert any("Edge family mask" in e.value for e in app.error)
    assert app.button(key="run_inference_btn").disabled

    # Supported method + available-case: still blocked, never a silent fallback.
    next(s for s in app.selectbox if s.label == "Select method profile:").set_value(
        "Balanced default (TFNBS)"
    ).run()
    next(r for r in app.radio if "Missing connectivity policy" in r.proto.label).set_value(
        "Available-case (test each edge on its available subjects)"
    ).run()
    assert not app.exception
    assert any("available-case" in e.value for e in app.error)
    assert app.button(key="run_inference_btn").disabled
    # The coverage floor is exposed with the engine default.
    assert next(
        n for n in app.number_input if "Minimum edge coverage" in n.label
    ).value == pytest.approx(0.8)


def test_design_page_binds_condition_confounds_in_the_paired_branch():
    from conninfpy.atlas import AtlasInfo

    n, N = 6, 4
    Y = _corr_stack(n, N, seed=7)
    labels = ["s0", "s1", "s2"] * 2
    conds = ["rest"] * 3 + ["task"] * 3
    pheno = pd.DataFrame(
        {
            "subject_id": labels,
            "condition": conds,
            "Motion_FD": np.linspace(0.1, 0.6, n),
        }
    )
    pheno = pheno.iloc[[3, 0, 1, 4, 2, 5]].reset_index(drop=True)
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N)
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state.connectivity_data = Y
    app.session_state.connectivity_data_kind = "fisher_z"
    app.session_state.pheno_df = pheno
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.active_tab = "2. Design & Inference"
    app.session_state["design:select_your_analysis_question"] = "Paired Condition"
    app.run()
    assert not app.exception
    confounds = next(
        m for m in app.multiselect if "Condition-specific confounds" in m.proto.label
    )
    confounds.set_value(["Motion_FD"]).run()
    assert not app.exception
    assert any("Paired design verified" in c.value for c in app.caption)
    assert any("computed by the engine" in c.value for c in app.caption)


def test_results_page_badges_the_declared_primary_and_reports_coverage():
    from conninfpy import analyze
    from conninfpy.atlas import AtlasInfo
    from conninfpy.app.utils.result_views import get_2d_result

    n, N = 10, 4
    Y = _corr_stack(n, N, seed=13)
    Y[0, 0, 1] = Y[0, 1, 0] = np.nan
    group = np.array([0, 1] * 5, dtype=float)
    res = analyze(
        group1=Y[:5], group2=Y[5:], test_type="paired", method="tfnbs",
        e=[0.5, 0.75], h=[2.0, 2.0], input="z", r_to_z=False,
        n_permutations=20, rng=1, use_mp=False, acceleration=None,
        missing="available_case", min_coverage=0.6,
    )
    assert res.inference.is_grid
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N)
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state.connectivity_data = Y
    app.session_state.pheno_df = pd.DataFrame({"group": group})
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.inference_result = res
    app.session_state.edges_df = get_2d_result(res, 1)[0].significant_edges(atlas=atlas, alpha=0.05)
    app.session_state.run_plan = {
        "method": "tfnbs",
        "alpha": 0.05,
        "design_family": "paired",
        "missing_policy": "available_case",
        "min_coverage": 0.6,
        "primary_cell": {"e": 0.75, "h": 2.0, "param_idx": 1, "selection": "declared_before_run"},
    }
    app.session_state.active_tab = "3. Inference Results"
    app.run()
    assert not app.exception
    success_text = "\n".join(s.value for s in app.success)
    assert "Declared primary cell" in success_text
    assert any("descriptive sensitivity" in s.value for s in app.caption)
    coverage = [e for e in app.expander if "Edge coverage" in str(e.label)]
    assert coverage, "coverage report expander is missing"
    assert any("not a complete-case analysis" in s.value for s in app.caption)
    assert any("subject_pair_difference" in s.value for s in app.caption)

    # Browsing another cell stays descriptive and never rewrites the primary.
    selector = next(s for s in app.selectbox if "TFNBS parameter-grid cell" in s.proto.label)
    selector.set_value(0).run()
    assert not app.exception
    assert any("Descriptive sensitivity slice" in w.value for w in app.warning)
    assert app.session_state.run_plan["primary_cell"]["param_idx"] == 1
    assert app.session_state.run_plan["missing_policy"] == "available_case"


def test_end_to_end_two_sample_run_with_edge_mask_records_the_family():
    """Two-sample run guided by a tested-family mask: facts, not arrays."""
    from conninfpy.atlas import AtlasInfo

    n, N = 10, 4
    Y = _corr_stack(n, N, seed=31)
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n)],
            "group": ["ctrl"] * 5 + ["case"] * 5,
            "Motion_FD": np.linspace(0.05, 0.5, n),
        }
    )
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N)
    mask = np.zeros((N, N), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    mask[1, 2] = mask[2, 1] = True

    app = AppTest.from_file(str(APP), default_timeout=240)
    app.session_state.connectivity_data = Y
    app.session_state.connectivity_data_kind = "fisher_z"
    app.session_state.pheno_df = pheno
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.active_tab = "2. Design & Inference"
    app.session_state["design:edge_mask_array"] = mask
    app.session_state["design:edge_mask_source"] = "family.npy"
    app.run()
    assert not app.exception
    # t-stat is a supported mask method on the two-sample route.
    next(s for s in app.selectbox if s.label == "Select method profile:").set_value(
        "Diagnostic baseline (t-stat max-permutation)"
    ).run()
    next(c for c in app.checkbox if c.label == "Parallel Execution (Multiprocessing)").set_value(False).run()
    next(r for r in app.radio if r.label == "Permutation Budget Preset:").set_value(
        "Quick check (100 permutations)"
    ).run()
    assert not app.exception
    assert app.button(key="run_inference_btn").disabled is False
    app.button(key="run_inference_btn").click().run()
    assert not app.exception
    task = app.session_state.inference_task
    deadline = time.time() + 120
    while task.status == "running" and time.time() < deadline:
        time.sleep(0.5)
    assert task.status == "success", getattr(task, "error", None)

    app.run()
    app.run()
    assert not app.exception
    plan = app.session_state.run_plan
    assert plan["edge_mask"]["purpose"] == "edge_family_mask"
    assert plan["edge_mask"]["eligible_edges"] == 2
    assert plan["edge_mask"]["source_filename"] == "family.npy"
    assert plan["edge_mask"]["shape"] == [N, N]
    assert isinstance(plan["edge_mask"]["sha256"], str) and len(plan["edge_mask"]["sha256"]) == 64
    assert "edge_mask" not in plan["operator_kwargs"]
    assert plan["missing_policy"] == "raise"
    assert app.session_state.inference_result is not None


def test_end_to_end_paired_run_declared_primary_then_workspace_reload(tmp_path: Path):
    """Configure -> run -> results (declared primary) -> save/reload workspace."""
    from conninfpy.atlas import AtlasInfo

    n, N = 6, 4
    Y = _corr_stack(n, N, seed=21)
    subjects = ["s0", "s1", "s2"]
    pheno = pd.DataFrame(
        {
            "subject_id": subjects * 2,
            "condition": ["rest"] * 3 + ["task"] * 3,
            "Motion_FD": [0.1, 0.2, 0.3, 0.2, 0.5, 0.7],
        }
    ).iloc[[3, 0, 1, 4, 2, 5]].reset_index(drop=True)
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N)

    app = AppTest.from_file(str(APP), default_timeout=240)
    app.session_state.connectivity_data = Y
    app.session_state.connectivity_data_kind = "fisher_z"
    app.session_state.pheno_df = pheno
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.active_tab = "2. Design & Inference"
    app.session_state["design:select_your_analysis_question"] = "Paired Condition"
    app.run()
    assert not app.exception

    next(m for m in app.multiselect if "Condition-specific confounds" in m.proto.label).set_value(
        ["Motion_FD"]
    ).run()
    # Condition-specific confounds plus the GLM route cannot take an edge mask,
    # but this run needs neither: a declared grid primary over the pairwise route.
    grid = next(c for c in app.checkbox if "E/H sensitivity grid" in c.proto.label)
    grid.set_value(True).run()
    assert not app.exception
    next(t for t in app.text_input if "Extent exponents (E)" in t.label).set_value("0.4, 0.5").run()
    next(t for t in app.text_input if "Height exponents (H)" in t.label).set_value("2, 3").run()
    next(
        s for s in app.selectbox if s.label.startswith("Declared primary cell")
    ).set_value("Cell 1: E=0.5, H=3").run()
    next(c for c in app.checkbox if c.label == "Parallel Execution (Multiprocessing)").set_value(False).run()
    next(r for r in app.radio if r.label == "Permutation Budget Preset:").set_value("Quick check (100 permutations)").run()
    assert not app.exception
    assert any("Descriptive sensitivity only" in c.value for c in app.caption)

    # Condition-specific confounds switch the paired route to the Δ-GLM, which
    # the engine does not expose to an edge mask; the run itself is still valid.
    app.button(key="run_inference_btn").click().run()
    assert not app.exception
    task = app.session_state.inference_task
    deadline = time.time() + 120
    while task.status == "running" and time.time() < deadline:
        time.sleep(0.5)
    assert task.status == "success", getattr(task, "error", None)

    app.run()
    app.run()
    assert not app.exception
    plan = app.session_state.run_plan
    assert plan["primary_cell"]["param_idx"] == 1
    assert plan["primary_cell"]["e"] == pytest.approx(0.5)
    assert plan["sensitivity_grid"]["n_cells"] == 2
    assert plan["condition_confound_cols"] == ["Motion_FD"]
    assert plan["n_paired_subjects"] == 3
    assert plan["missing_policy"] == "raise"
    assert app.session_state.inference_result is not None
    assert getattr(app.session_state.inference_result.inference, "is_grid", False)

    # Save the finished workspace and reload it into a fresh session state.
    from conninfpy.app.utils.workspace_io import build_workspace_archive, restore_workspace_archive

    session = app.session_state
    keys = {
        "_workspace_widget_keys", "inference_result", "run_plan", "edges_df",
        "companion_inference_result", "connectivity_data", "pheno_df", "dataset_info",
        "loaded_reference_atlas", "loaded_atlas_signature", "loaded_settings_hash",
        "active_tab",
    }
    try:
        keys.update(session["_workspace_widget_keys"])
    except (KeyError, AttributeError):
        pass
    state = {}
    for key in keys:
        try:
            state[key] = session[key]
        except (KeyError, AttributeError):
            continue
    ws_bytes, _ = build_workspace_archive(state, include_data=False)
    target = {}
    restore_workspace_archive(ws_bytes, target)
    assert target["run_plan"]["primary_cell"]["param_idx"] == 1
    assert target["run_plan"]["condition_confound_cols"] == ["Motion_FD"]
    assert target["run_plan"]["sensitivity_grid"]["e"] == pytest.approx([0.4, 0.5])
    assert target["design:eh_grid_primary_cell"] == "Cell 1: E=0.5, H=3"

    reloaded = AppTest.from_file(str(APP), default_timeout=240)
    for key, value in target.items():
        if key in ("inference_result", "run_plan", "connectivity_data", "pheno_df",
                   "loaded_reference_atlas", "loaded_atlas_signature", "loaded_settings_hash",
                   "active_tab", "edges_df"):
            reloaded.session_state[key] = value
    reloaded.session_state.active_tab = "3. Inference Results"
    reloaded.run()
    assert not reloaded.exception
    assert "Declared primary cell" in "\n".join(s.value for s in reloaded.success)


# ----------------------------------------------------------------------
# 7. Owner review fixes: declared cell in companion agreement; archive mask
#    must be the mask of the finished run (sha256-verified), never a later one
# ----------------------------------------------------------------------
def test_companion_agreement_selects_the_declared_primary_cell():
    """The agreement gate slices a grid primary at the declared cell, not cell 0."""
    import copy

    from conninfpy import analyze
    from conninfpy.app.utils import result_views

    n, N = 8, 4
    Y = _corr_stack(n, N, seed=21)
    group = np.array([0, 1] * (n // 2), dtype=float)
    grid_res = analyze(
        Y, interest=group, method="tfnbs", input="z", r_to_z=False,
        e=[0.5, 0.75], h=[2.0, 2.0], n=50,
        n_permutations=10, rng=1, use_mp=False, acceleration=None,
    )
    inner = result_views.unwrap_result(grid_res)
    assert getattr(inner, "is_grid", False)

    seen = []
    real_get_2d = result_views.get_2d_result

    def spy(res, param_idx=None):
        seen.append(param_idx)
        return real_get_2d(res, param_idx)

    with patch.object(result_views, "get_2d_result", side_effect=spy):
        report = result_views.check_companion_compatibility(
            grid_res, copy.deepcopy(grid_res), alpha=0.05, primary_param_idx=1
        )
    assert report["compatible"], report["reasons"]
    assert seen and seen[0] == 1  # primary sliced at the declared cell
    # Without the declaration the historical cell 0 behaviour is unchanged.
    seen.clear()
    with patch.object(result_views, "get_2d_result", side_effect=spy):
        result_views.check_companion_compatibility(grid_res, copy.deepcopy(grid_res), alpha=0.05)
    assert seen and seen[0] == 0


def test_archive_omits_a_mask_changed_after_the_run(tmp_path: Path):
    """A mask whose sha256 differs from the run plan is never bundled silently."""
    import io
    import zipfile

    from conninfpy import analyze
    from conninfpy.app.utils.archive_export import build_analysis_archive

    n, N = 8, 4
    Y = _corr_stack(n, N, seed=22)
    group = np.array([0, 1] * (n // 2), dtype=float)
    res = analyze(
        Y, interest=group, method="tstat", input="z", r_to_z=False,
        n_permutations=10, rng=1, use_mp=False, acceleration=None,
    )
    plan = _plan_with_new_facts()
    plan["method"] = "tstat"
    plan["design_family"] = "glm"
    plan["operator_kwargs"] = {}
    plan["primary_cell"] = None
    run_mask = np.zeros((N, N), dtype=bool)
    run_mask[0, 1] = run_mask[1, 0] = True
    later_mask = run_mask.copy()
    later_mask[2, 3] = later_mask[3, 2] = True  # changed after the run
    info = {"mode": "Generate Synthetic Dataset", "name": "fixture"}

    zip_bytes, manifest = build_analysis_archive(
        res, plan, info, edge_family_mask=later_mask, include_bundle=False,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert "arrays/edge_family_mask.npy" not in zf.namelist()
    assert any("does not match the sha256" in note for note in manifest.get("notes", []))

    # The matching mask of the finished run is bundled as before.
    zip_bytes, _ = build_analysis_archive(
        res, plan, info, edge_family_mask=run_mask, include_bundle=False,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert "arrays/edge_family_mask.npy" in zf.namelist()


# ----------------------------------------------------------------------
# 7. Round-1 parent review regressions (4 defects)
# ----------------------------------------------------------------------
def test_edge_mask_metadata_rejects_malformed_shapes():
    """A malformed mask is a validation error, never an IndexError in render."""
    from conninfpy.app.components.design_inference import _edge_mask_metadata

    for malformed in (
        np.zeros(3, dtype=bool),          # 1-D
        np.zeros((2, 3), dtype=bool),     # non-square 2-D
        np.zeros((0, 0), dtype=bool),     # empty
        np.zeros((2, 3, 4), dtype=bool),  # 3-D
    ):
        with pytest.raises(ValueError) as excinfo:
            _edge_mask_metadata(malformed)
        assert "square 2-D" in str(excinfo.value)
        assert not isinstance(excinfo.value, IndexError)

    # Symmetry is the engine validator's job and stays a clear error.
    asymmetric = np.zeros((4, 4), dtype=bool)
    asymmetric[0, 1] = True
    with pytest.raises(ValueError) as excinfo:
        _check_edge_mask(asymmetric, 4)
    assert "symmetric" in str(excinfo.value)

    # A valid square mask still yields the recorded facts.
    good = np.zeros((4, 4), dtype=bool)
    good[0, 1] = good[1, 0] = True
    facts = _edge_mask_metadata(good, "family.npy")
    assert facts["shape"] == [4, 4] and facts["eligible_edges"] == 1


def test_design_page_with_malformed_stored_mask_blocks_the_run():
    """A 1-D stored mask must render with a clear error and a blocked launch."""
    app = _design_app()
    app.session_state["design:edge_mask_array"] = np.zeros(3, dtype=bool)
    app.session_state["design:edge_mask_source"] = "broken.npy"
    app.run()
    assert not app.exception
    errors = "\n".join(e.value for e in app.error)
    assert "square 2-D" in errors          # retained-mask caption / run-plan line
    assert "Edge family mask" in errors    # engine validator blocking message
    assert app.button(key="run_inference_btn").disabled
    # The run-plan table reports the invalid mask instead of claiming "all edges".
    plan_md = "\n".join(m.value for m in app.markdown)
    assert "invalid retained mask" in plan_md


def _tstat_result_and_plan():
    from conninfpy import analyze

    n, N = 8, 4
    Y = _corr_stack(n, N, seed=23)
    group = np.array([0, 1] * (n // 2), dtype=float)
    res = analyze(
        Y, interest=group, method="tstat", input="z", r_to_z=False,
        n_permutations=10, rng=1, use_mp=False, acceleration=None,
    )
    plan = _plan_with_new_facts()
    plan["method"] = "tstat"
    plan["operator_kwargs"] = {}
    plan["primary_cell"] = None
    return res, plan, {"mode": "Generate Synthetic Dataset", "name": "fixture"}


def _archive_member_and_notes(res, plan, info, mask):
    import io
    import zipfile

    from conninfpy.app.utils.archive_export import build_analysis_archive

    zip_bytes, manifest = build_analysis_archive(
        res, plan, info, edge_family_mask=mask, include_bundle=False,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    return names, list(manifest.get("notes", []))


def test_archive_omits_mask_without_a_recorded_sha():
    """No recorded sha256 means the mask cannot be verified: never bundle it."""
    res, plan, info = _tstat_result_and_plan()
    plan["edge_mask"] = dict(plan["edge_mask"])
    plan["edge_mask"].pop("sha256")
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = True

    names, notes = _archive_member_and_notes(res, plan, info, mask)
    assert "arrays/edge_family_mask.npy" not in names
    assert any("did not record a sha256" in note for note in notes)
    assert any("cannot reproduce the tested family" in note for note in notes)

    for empty in (None, "", "   "):
        plan["edge_mask"]["sha256"] = empty
        names, notes = _archive_member_and_notes(res, plan, info, mask)
        assert "arrays/edge_family_mask.npy" not in names
        assert any("did not record a sha256" in note for note in notes)


def test_archive_omits_a_reshaped_mask_with_matching_bytes():
    """Same bool bytes, different ROI x ROI shape: a different tested family."""
    res, plan, info = _tstat_result_and_plan()
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    reshaped = mask.reshape(2, 8)
    assert plan["edge_mask"]["shape"] == [4, 4]
    # The bytes (and therefore the sha256) are identical.
    import hashlib

    assert (
        hashlib.sha256(np.ascontiguousarray(reshaped.astype(bool)).tobytes()).hexdigest()
        == plan["edge_mask"]["sha256"]
    )

    names, notes = _archive_member_and_notes(res, plan, info, reshaped)
    assert "arrays/edge_family_mask.npy" not in names
    assert any("recorded" in note and "shape" in note for note in notes)
    assert any("cannot reproduce the tested family" in note for note in notes)

    # The mask with the recorded shape and bytes is still bundled.
    names, notes = _archive_member_and_notes(res, plan, info, mask)
    assert "arrays/edge_family_mask.npy" in names
    assert not any("cannot reproduce the tested family" in note for note in notes)


def _run_worker_with_result(task, result, **overrides):
    """Run the real worker against a fake grid result and record edge calls."""
    from conninfpy.app.components.design_inference import _run_inference_background

    args = {name: None for name in inspect.signature(_run_inference_background).parameters}
    args.update(
        task=task, test_type="paired", operator_choice="tfnbs", n_perms=20, seed_val=42,
        use_mp=False, acceleration_choice="none",
        op_kwargs={"e": [0.5, 0.75], "h": [2.0, 2.0], "n": 50, "start_thres": 1.0},
        run_sensitivity=False, alpha=0.05, question_choice="Paired Condition",
        subject_col="subject_id", condition_col="condition",
        baseline_val="rest", target_val="task", confound_vars=[],
        active_atlas_signature="fixture", data_kind="fisher_z",
        group1=np.zeros((3, 4, 4)), group2=np.zeros((3, 4, 4)),
    )
    args.update(overrides)
    with patch("conninfpy.analyze", return_value=result):
        _run_inference_background(**args)


def _grid_result(e_grid, h_grid, calls):
    def _significant_edges(**kwargs):
        calls.append(kwargs)
        return []

    return SimpleNamespace(
        significant_edges=_significant_edges,
        is_grid=True,
        e_grid=np.asarray(e_grid, dtype=float),
        h_grid=np.asarray(h_grid, dtype=float),
        wall_time_s=0.01,
        fisher=None,
        design={},
    )


def test_worker_rejects_a_declared_primary_cell_absent_from_the_grid():
    from conninfpy.app.components.design_inference import InferenceTask

    calls = []
    task = InferenceTask()
    _run_worker_with_result(
        task,
        _grid_result([0.5, 0.75], [2.0, 2.0], calls),
        primary_cell={"e": 0.99, "h": 2.0, "param_idx": 1, "selection": "declared_before_run"},
    )
    assert task.status == "failed"
    assert "not a cell of the completed" in (task.error or "")
    assert calls == []  # no completion table for a different cell


def test_worker_rejects_an_ambiguous_declared_primary_cell():
    from conninfpy.app.components.design_inference import InferenceTask

    calls = []
    task = InferenceTask()
    _run_worker_with_result(
        task,
        _grid_result([0.5, 0.5], [2.0, 2.0], calls),
        primary_cell={"e": 0.5, "h": 2.0, "param_idx": 0, "selection": "declared_before_run"},
    )
    assert task.status == "failed"
    assert "ambiguous" in (task.error or "")
    assert calls == []


def test_worker_rejects_an_index_contradicting_the_declared_coordinates():
    from conninfpy.app.components.design_inference import InferenceTask

    calls = []
    task = InferenceTask()
    _run_worker_with_result(
        task,
        _grid_result([0.5, 0.75], [2.0, 2.0], calls),
        primary_cell={"e": 0.75, "h": 2.0, "param_idx": 0, "selection": "declared_before_run"},
    )
    assert task.status == "failed"
    assert "contradicts" in (task.error or "")
    assert calls == []


def test_results_page_never_badges_a_mismatched_declared_primary():
    """A reloaded plan whose (E, H) is not in this grid must not claim the badge."""
    from conninfpy import analyze
    from conninfpy.atlas import AtlasInfo
    from conninfpy.app.utils.result_views import get_2d_result

    n, N = 10, 4
    Y = _corr_stack(n, N, seed=29)
    group = np.array([0, 1] * 5, dtype=float)
    res = analyze(
        group1=Y[:5], group2=Y[5:], test_type="two-sample", method="tfnbs",
        e=[0.5, 0.75], h=[2.0, 2.0], input="z", r_to_z=False,
        n_permutations=20, rng=1, use_mp=False, acceleration=None,
    )
    assert res.inference.is_grid
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N)
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state.connectivity_data = Y
    app.session_state.pheno_df = pd.DataFrame({"group": group})
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.inference_result = res
    app.session_state.edges_df = get_2d_result(res, 1)[0].significant_edges(atlas=atlas, alpha=0.05)
    app.session_state.run_plan = {
        "method": "tfnbs",
        "alpha": 0.05,
        "design_family": "two-sample",
        "primary_cell": {"e": 0.99, "h": 2.0, "param_idx": 1, "selection": "declared_before_run"},
    }
    app.session_state.active_tab = "3. Inference Results"
    app.run()
    assert not app.exception
    assert "Declared primary cell" not in "\n".join(s.value for s in app.success)
    warnings_text = "\n".join(w.value for w in app.warning)
    assert "does not match this result's (E, H) grid" in warnings_text
    # Browsing stays descriptive and the run plan is untouched.
    selector = next(s for s in app.selectbox if "TFNBS parameter-grid cell" in s.proto.label)
    assert selector.value == 0
    selector.set_value(1).run()
    assert not app.exception
    assert len(app.success) == 0 or "Declared primary cell" not in "\n".join(
        s.value for s in app.success
    )
    assert app.session_state.run_plan["primary_cell"]["e"] == pytest.approx(0.99)


def test_comparative_view_validates_the_declared_primary_against_its_grid():
    """The comparison view and its detail tabs use only a validated declaration."""
    from conninfpy import analyze
    from conninfpy.atlas import AtlasInfo
    from conninfpy.app.utils.result_views import get_2d_result

    n, N = 10, 4
    Y = _corr_stack(n, N, seed=33)
    group = np.array([0, 1] * 5, dtype=float)
    grid = analyze(
        Y, interest=group, method="tfnbs", e=[0.5, 0.75], h=[2.0, 2.0], n=50,
        input="z", r_to_z=False, n_permutations=10, rng=1, use_mp=False, acceleration=None,
    )
    companion = analyze(
        Y, interest=group, method="tstat", input="z", r_to_z=False,
        n_permutations=10, rng=1, use_mp=False, acceleration=None,
    )
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N)

    def _app(primary_cell):
        app = AppTest.from_file(str(APP), default_timeout=180)
        app.session_state.connectivity_data = Y
        app.session_state.pheno_df = pd.DataFrame({"group": group})
        app.session_state.loaded_reference_atlas = atlas
        app.session_state.loaded_atlas_signature = "fixture"
        app.session_state.loaded_settings_hash = "fixture"
        app.session_state.inference_result = grid
        app.session_state.edges_df = get_2d_result(grid, 1)[0].significant_edges(
            atlas=atlas, alpha=0.05
        )
        app.session_state.companion_inference_result = companion
        app.session_state.companion_edges_df = companion.significant_edges(atlas=atlas, alpha=0.05)
        app.session_state.companion_method = "tstat"
        app.session_state.run_plan = {
            "method": "tfnbs", "alpha": 0.05, "design_family": "glm",
            "primary_cell": primary_cell,
        }
        app.session_state.active_tab = "3. Inference Results"
        app.run()
        assert not app.exception
        next(r for r in app.radio if "Select Result View Mode" in r.proto.label).set_value(
            "Side-by-Side Comparison"
        ).run()
        assert not app.exception
        return app

    valid = _app({"e": 0.75, "h": 2.0, "param_idx": 1, "selection": "declared_before_run"})
    captions = "\n".join(c.value for c in valid.caption)
    assert "compared at the declared primary cell 1" in captions
    assert "does not match this result's (E, H) grid" not in "\n".join(
        w.value for w in valid.warning
    )

    mismatched = _app({"e": 0.99, "h": 2.0, "param_idx": 1, "selection": "declared_before_run"})
    warnings_text = "\n".join(w.value for w in mismatched.warning)
    assert "does not match this result's (E, H) grid" in warnings_text
    assert "compared at the declared primary cell" not in "\n".join(
        c.value for c in mismatched.caption
    )

    out_of_range = _app({"e": 0.75, "h": 2.0, "param_idx": 9, "selection": "declared_before_run"})
    assert "does not match this result's (E, H) grid" in "\n".join(
        w.value for w in out_of_range.warning
    )


def test_replay_manifest_and_methods_match_the_actual_bundle_decision(tmp_path: Path):
    """Round-2 parent review: no export text may claim an absent archive member.

    The replay reasons/prerequisites and methods.md must follow the actual
    bundle verdict (sha+shape verified), not merely the presence of mask
    metadata in the run plan. Uses a full valid paired plan + manifest dataset
    (the parent's repro path), so the mask prerequisite yields requires-input.
    """
    import io
    import zipfile

    from conninfpy.app.utils.archive_export import build_analysis_archive

    res, _minimal_plan, _minimal_info = _tstat_result_and_plan()
    plan = _plan_with_new_facts()
    plan["method"] = "tstat"
    plan["operator_kwargs"] = {}
    plan["method_parameters"] = {}
    plan["primary_cell"] = None
    plan.pop("sensitivity_grid", None)
    info = _replay_dataset_info(tmp_path)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = True

    def _export(p, m):
        zip_bytes, manifest = build_analysis_archive(
            res, p, info, edge_family_mask=m, include_bundle=False,
        )
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            member = "arrays/edge_family_mask.npy" in zf.namelist()
            methods = zf.read("methods.md").decode("utf-8")
        replay = manifest["replay"]
        text = json.dumps(replay, ensure_ascii=False)
        return member, replay, text, methods

    # Valid sha+shape match: bundled, and the replay may name the member.
    member, replay, text, methods = _export(plan, mask)
    assert member
    assert replay["status"] == "requires-input"
    assert "bundled as arrays/edge_family_mask.npy" in text
    assert "NOT bundled" not in text
    assert "bundled in this archive as arrays/edge_family_mask.npy" in methods

    # Omission cases: the member is absent, so no export text may claim it.
    no_sha = dict(plan)
    no_sha["edge_mask"] = {k: v for k, v in plan["edge_mask"].items() if k != "sha256"}
    cases = [
        ("no recorded sha", no_sha, mask),
        ("same-byte reshape", plan, mask.reshape(2, 8)),
        ("changed bytes", plan, np.logical_not(mask)),
        ("mask unavailable", plan, None),
    ]
    for label, p, m in cases:
        member, replay, text, methods = _export(p, m)
        assert not member, label
        assert replay["status"] == "requires-input", label
        assert "NOT bundled" in text, label
        assert "bundled as arrays/edge_family_mask.npy" not in text, label
        assert "apply the bundled" not in text, label
        assert "external input" in text, label
        assert "NOT bundled in this archive" in methods, label
        assert "bundled in this archive as" not in methods, label
        # The prerequisite still names the exact recorded family facts.
        prereq = next(q for q in replay["prerequisites"] if q["kind"] == "edge-family-mask")
        assert "edge-family-mask" == prereq["kind"]
        assert "[4, 4]" in prereq["detail"], label

    # A standalone replay config (no archive, no array) is honest too.
    from conninfpy.app.utils.archive_export import build_replay_config

    cfg = build_replay_config(plan, info, project_root=tmp_path)
    text = json.dumps(cfg, ensure_ascii=False)
    assert "NOT bundled" in text
    assert "apply the bundled" not in text
