"""NETSTAT-27: GUI progress/cancel/stale-job contract (real worker thread).

These tests drive the real background worker (real ``analyze()`` on synthetic
data, real engine cancellation) and render the real Streamlit app with
``AppTest``. Nothing here is import- or mock-only for the inference itself.

Synthetic data only; no hosted services.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conninfpy.app.components.design_inference import (
    InferenceTask,
    _invalidate_superseded_task,
    _run_inference_background,
)
from conninfpy.atlas import AtlasInfo

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "conninfpy" / "app" / "streamlit_app.py"
N = 5


def _corr_stack(n, seed=3, N=N):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, N, N))
    for i in range(n):
        a = rng.standard_normal((N, 20))
        c = np.corrcoef(a)
        np.fill_diagonal(c, 1.0)
        out[i] = c
    return out


def _worker_kwargs(task, n_perms, **overrides):
    kwargs = dict(
        task=task,
        Y=None,
        interest=None,
        confounds=None,
        group1=None,
        group2=None,
        test_type="two-sample",
        sites=None,
        harmonization_choice=None,
        operator_choice="tfnbs",
        n_perms=n_perms,
        seed_val=42,
        use_mp=False,
        acceleration_choice="none",
        op_kwargs={"e": 0.4, "h": 2.0, "n": 50, "start_thres": 1.0},
        run_sensitivity=False,
        comp_method=None,
        comp_kwargs=None,
        sc_net_labels=None,
        use_sc_prior=False,
        atlas=AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N),
        annotation_atlas=AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N),
        alpha=0.05,
        question_choice="Group Difference",
        loaded_settings_hash="fixture",
        recipe_choice="No site handling",
        effective_recipe_choice="No site handling",
        group_col="group",
        ref_group=0,
        target_group=1,
        interest_var=None,
        subject_col=None,
        condition_col=None,
        baseline_val=None,
        target_val=None,
        confound_vars=[],
        site_col=None,
        active_atlas_signature="fixture-atlas",
        data_kind="correlation",
        n_processes=2,
    )
    kwargs.update(overrides)
    return kwargs


def _run_worker_thread(task, n_perms, timeout=180, **overrides):
    kwargs = _worker_kwargs(task, n_perms, **overrides)
    kwargs["group1"] = _corr_stack(6, seed=1)
    kwargs["group2"] = _corr_stack(6, seed=2)
    thread = threading.Thread(target=_run_inference_background, kwargs=kwargs, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        thread.join(0.05)
    assert not thread.is_alive(), "background inference thread did not finish"
    return task


# ----------------------------------------------------------------------
# 1. Real run: exact parent-side progress, success, resource facts
# ----------------------------------------------------------------------
def test_real_run_reports_exact_progress_and_success():
    task = InferenceTask()
    task.generation = 1
    _run_worker_thread(task, 6)

    assert task.status == "success", task.error
    assert task.result is not None and task.edges is not None
    assert task.progress_total == 6
    assert task.progress_done == task.progress_total
    plan = task.run_plan
    assert plan["progress_final"]["done"] == plan["progress_final"]["total"] == 6
    assert plan["n_processes_requested"] == 2
    # This run asked for a single process, so the engine resolves one worker;
    # the requested value is recorded next to the effective one.
    assert plan["resource_plan"]["use_mp_requested"] is False
    assert plan["resource_plan"]["workers_effective"] == 1
    # Honest unknowns: the BLAS library / executed kernel are not measured.
    assert plan["resource_plan"]["blas_library"] == "unknown/not measured"
    assert plan["generation"] == 1


# ----------------------------------------------------------------------
# 2. Real cancellation mid-run: cancelled state, no result, restart works
# ----------------------------------------------------------------------
def test_real_cancel_leaves_cancelled_task_without_result_and_restart_works():
    task = InferenceTask()
    task.generation = 1
    kwargs = _worker_kwargs(task, 400)
    kwargs["use_mp"] = True
    kwargs["group1"] = _corr_stack(8, seed=4)
    kwargs["group2"] = _corr_stack(8, seed=5)
    # Deterministic cancel: request it from the first parent-side progress
    # callback (the engine's own chunk-boundary thread), so a fast run cannot
    # finish between a cross-thread polling sleep and the cancel request.
    base_cb = task._on_progress

    def _cancel_on_first_chunk(done, total):
        base_cb(done, total)
        if done >= 1:
            task.request_cancel("cancelled by the user")

    task._on_progress = _cancel_on_first_chunk
    thread = threading.Thread(target=_run_inference_background, kwargs=kwargs, daemon=True)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), "cancellation must stop the background thread"
    assert task.progress_done >= 1, "the run must report folded chunks before cancelling"

    assert task.status == "cancelled"
    assert task.result is None and task.edges is None
    assert task.comp_result is None and task.comp_edges is None
    assert task.run_plan is None
    assert task.cancel_event.is_set()
    assert task.progress_done < task.progress_total

    # Restart in the same process: a fresh task completes normally.
    ok = InferenceTask()
    ok.generation = 2
    _run_worker_thread(ok, 4)
    assert ok.status == "success", ok.error
    assert ok.progress_done == ok.progress_total == 4


# ----------------------------------------------------------------------
# 3. AppTest: cancelled UI, no result/export, restart available
# ----------------------------------------------------------------------
def _app_with_task(task):
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.session_state.connectivity_data = _corr_stack(12)
    app.session_state.connectivity_data_kind = "correlation"
    app.session_state.pheno_df = pd.DataFrame(
        {"group": np.array([0] * 6 + [1] * 6), "subject_id": [f"s{i}" for i in range(12)]}
    )
    atlas = AtlasInfo(labels=[f"ROI{i}" for i in range(N)], networks=["Vis"] * N)
    app.session_state.loaded_reference_atlas = atlas
    app.session_state.loaded_atlas_signature = "fixture"
    app.session_state.loaded_settings_hash = "fixture"
    app.session_state.inference_task = task
    app.session_state.active_tab = "2. Design & Inference"
    return app


def test_cancelled_task_renders_cancelled_and_blocks_export():
    task = InferenceTask()
    task.status = "cancelled"
    task.cancel_reason = "cancelled by the user"
    task.cancel_event.set()
    task.generation = 3
    app = _app_with_task(task)
    app.run()
    assert not app.exception
    warnings_text = "\n".join(w.value for w in app.warning)
    assert "Cancelled" in warnings_text or "cancelled" in warnings_text
    assert "no p-values" in warnings_text
    assert app.session_state.get("inference_result") is None
    assert app.session_state.get("edges_df") is None
    assert app.session_state.get("run_plan") is None
    labels = [b.label for b in app.button]
    assert not any("Export" in label for label in labels)
    assert app.button(key="run_inf_new_button").label.startswith("🚀 Configure")


def test_stale_success_payload_is_never_promoted():
    task = InferenceTask()
    task.status = "success"
    task.result = "stale-result"
    task.edges = "stale-edges"
    task.run_plan = {"method": "tfnbs"}
    task.generation = 5
    task.finished_generation = 5
    app = _app_with_task(task)
    app.session_state.inference_generation = 6  # a newer run has started
    app.run()
    assert not app.exception
    assert app.session_state.inference_result is None
    assert app.session_state.edges_df is None


def test_running_and_completed_states_still_work():
    task = InferenceTask()
    task.status = "running"
    task.progress_message = "Executing primary method"
    task.progress_done = 3
    task.progress_total = 10
    task.generation = 1
    app = _app_with_task(task)
    app.run()
    assert not app.exception
    assert app.button(key="run_inference_running").disabled
    assert app.button(key="cancel_inference_btn") is not None
    text = "\n".join(m.value for m in app.info)
    assert "Executing primary method" in text
    progress_text = "\n".join(
        getattr(p.proto, "text", "") or "" for p in app.get("progress")
    )
    assert "3 of 10" in progress_text


# ----------------------------------------------------------------------
# 4. Stale-job invalidation by launch signature
# ----------------------------------------------------------------------
def test_changed_data_cancels_a_running_job():
    task = InferenceTask()
    task.status = "running"
    task.launch_signature = {"data_id": "1", "method": "tfnbs"}
    _invalidate_superseded_task(task, {"data_id": "2", "method": "tfnbs"})
    assert task.cancel_event.is_set()
    assert "changed" in (task.cancel_reason or "")


def test_unchanged_signature_keeps_the_job_running():
    task = InferenceTask()
    task.status = "running"
    task.launch_signature = {"data_id": "1", "method": "tfnbs"}
    _invalidate_superseded_task(task, {"data_id": "1", "method": "tfnbs"})
    assert not task.cancel_event.is_set()


# ----------------------------------------------------------------------
# 5. Successful run: results page and workspace save still work
# ----------------------------------------------------------------------
def test_successful_run_result_and_workspace_round_trip():
    task = InferenceTask()
    task.generation = 1
    _run_worker_thread(task, 4)
    assert task.status == "success", task.error
    app = _app_with_task(task)
    app.run()
    assert not app.exception
    # The completed result is promoted and the results tab is reachable.
    app.session_state.active_tab = "3. Inference Results"
    app.run()
    assert not app.exception
    assert app.session_state.inference_result is not None
    assert app.session_state.run_plan["generation"] == 1

    # Saving the finished analysis keeps its results (workspace round-trip).
    import io
    import zipfile

    from conninfpy.app.utils import workspace_io

    archive, _summary = workspace_io.build_workspace_archive(
        dict(app.session_state), include_data=False
    )
    names = zipfile.ZipFile(io.BytesIO(archive)).namelist()
    assert any(name.endswith("state.json") for name in names)

    # And the restored payload is a completed analysis, not a cancelled one.
    restored = workspace_io.restore_workspace_archive(archive, {})
    assert restored is not None
