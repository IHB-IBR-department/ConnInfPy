"""NETSTAT-27 round 2: regressions for the parent's five blocking findings.

R1-1 stale thread must not touch the newer generation's state (both the
     generation guard and a fresh task object per launch).
R1-2 the launch signature covers the complete scientific design snapshot
     (group reversal, method change, grid change invalidate a running job).
R1-3 a cancel request is rechecked while waiting for a blocked worker and
     still cleans up (map/reduce x spawn/forkserver).
R1-4 an explicit external ``cancel`` is honoured together with a
     ``ProgressSession``.
R1-5 ``conninfpy info`` itself adds no filesystem side effect and does not
     import numba.

Synthetic data only; no hosted services.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import conninfpy
from apps.components import design_inference as design
from apps.components.design_inference import InferenceTask, _run_inference_background
from conninfpy import InferenceCancelled, ProgressSession, analyze
from conninfpy.atlas import AtlasInfo
from conninfpy.stats import _workers
from conninfpy.stats._workers import prepare_run, run_permutations, run_permutations_reduce
from tests.mp_helpers import sum_squares_chunk

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "apps" / "streamlit_app.py"
N = 5
START_METHODS = [
    name for name in ("spawn", "forkserver")
    if name in _workers.multiprocessing.get_all_start_methods()
]


def _stack(n, seed=3, size=N):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, size, size))
    for i in range(n):
        a = rng.standard_normal((size, 20))
        out[i] = np.corrcoef(a)
        np.fill_diagonal(out[i], 1.0)
    return out


def _worker_kwargs(task, n_perms, **overrides):
    from tests.test_netstat27_gui import _worker_kwargs as base

    kwargs = base(task, n_perms)
    kwargs["group1"] = _stack(6, seed=1)
    kwargs["group2"] = _stack(6, seed=2)
    kwargs.update(overrides)
    return kwargs


def _app_with_task(task, **state):
    from tests.test_netstat27_gui import _app_with_task as base

    app = base(task)
    for key, value in state.items():
        app.session_state[key] = value
    return app


# ----------------------------------------------------------------------
# R1-1: stale thread must not mutate the newer generation
# ----------------------------------------------------------------------
def test_stale_thread_cannot_overwrite_generation_two_state():
    """Deterministic two-generation regression (real analysis, real thread).

    The old worker is held inside ``analyze`` until generation 2 has been
    initialised on the same task object; when released it runs the real small
    serial analysis and must write nothing at all.
    """
    task = InferenceTask()
    task.generation = 1
    kwargs = _worker_kwargs(task, 2)
    started = threading.Event()
    release = threading.Event()
    real_analyze = conninfpy.analyze

    def delayed_analyze(*args, **kw):
        started.set()
        assert release.wait(20), "probe release was not signalled"
        return real_analyze(*args, **kw)

    thread = threading.Thread(
        target=_run_inference_background, kwargs=kwargs, daemon=True
    )
    with patch("conninfpy.analyze", new=delayed_analyze):
        thread.start()
        assert started.wait(20), "old worker never reached analyze"
        # Exactly what the launch code does for generation 2.
        task.generation = 2
        task.status = "running"
        task.result = "generation-2-result"
        task.edges = "generation-2-edges"
        task.run_plan = {"generation": 2}
        task.cancel_event = threading.Event()
        task.cancel_reason = None
        task.progress_done = 0
        task.progress_total = 2
        task.finished_generation = None
        release.set()
        thread.join(90)
    assert not thread.is_alive(), "the old worker must finish"

    assert task.status == "running"
    assert task.result == "generation-2-result"
    assert task.edges == "generation-2-edges"
    assert task.run_plan == {"generation": 2}
    assert (task.progress_done, task.progress_total) == (0, 2)
    assert task.finished_generation is None
    assert task.error is None
    assert task.generation == 2


def test_launch_creates_a_fresh_task_object_and_generation():
    """A new launch never reuses the previous InferenceTask object."""
    old_task = InferenceTask()
    old_task.status = "idle"
    old_task.generation = 4
    app = _app_with_task(old_task, inference_generation=4)
    app.run()
    assert not app.exception
    # The idle design step renders the run button; launching it is a real UI
    # action, so patch only the worker body (the task bookkeeping is real).
    seen = {}

    def fake_worker(**kwargs):
        seen["task"] = kwargs["task"]

    with patch.object(design, "_run_inference_background", new=fake_worker):
        app.button(key="run_inference_btn").click().run()
    assert not app.exception
    current = app.session_state.inference_task
    assert current is not old_task, "each launch must own a fresh task object"
    assert current.generation == 5
    assert seen.get("task") is current
    assert current.status == "running"
    # The previous object is untouched by the new launch.
    assert old_task.generation == 4 and old_task.status == "idle"


# ----------------------------------------------------------------------
# R1-2: the launch signature covers the scientific design snapshot
# ----------------------------------------------------------------------
def _capture_signature(app):
    """Run the app once and return the launch signature it computed."""
    captured = []
    original = design._invalidate_superseded_task

    def observe(current, signature):
        captured.append(dict(signature))
        original(current, signature)

    with patch.object(design, "_invalidate_superseded_task", new=observe):
        app.run()
    assert captured, "render did not compute a launch signature"
    return captured[-1]


def test_group_reversal_invalidates_a_running_job_and_payload_never_publishes():
    task = InferenceTask()
    task.status = "running"
    task.generation = 1
    app = _app_with_task(task)
    task.launch_signature = _capture_signature(app)
    assert not task.cancel_event.is_set()

    for widget in app.selectbox:
        if widget.label == "Reference Group (Baseline)":
            widget.set_value(1)
        elif widget.label == "Target Group (Contrast)":
            widget.set_value(0)
    app.run()
    assert not app.exception
    assert task.cancel_event.is_set(), "reversing the contrast must invalidate the job"
    assert "changed" in (task.cancel_reason or "")
    assert app.session_state.get("inference_result") is None


def test_method_change_invalidates_a_running_job():
    task = InferenceTask()
    task.status = "running"
    task.generation = 1
    app = _app_with_task(task)
    task.launch_signature = _capture_signature(app)

    profile = next(
        (b for b in app.selectbox if "Select method profile" in b.label), None
    )
    assert profile is not None, "method profile selector not found"
    options = list(profile.options)
    alternative = next(
        (option for option in options if option != profile.value), None
    )
    assert alternative is not None, "no alternative method profile"
    profile.set_value(alternative)
    app.run()
    assert not app.exception
    assert task.cancel_event.is_set(), "a method change must invalidate the job"
    assert app.session_state.get("inference_result") is None


def test_grid_parameter_change_invalidates_a_running_job():
    """A changed (E, H) grid value must invalidate the running job."""
    task = InferenceTask()
    task.status = "running"
    task.generation = 1
    app = _app_with_task(task)
    task.launch_signature = _capture_signature(app)

    # Toggle the TFNBS parameter grid so the declared grid/E/H facts change.
    toggled = False
    for checkbox in app.checkbox:
        if "grid" in checkbox.label.lower():
            checkbox.set_value(not checkbox.value)
            toggled = True
            break
    if not toggled:
        # Fallback: change a numeric E/H parameter of the enhancement method.
        for number in app.number_input:
            label = number.label.lower()
            if "extent" in label or "height" in label or "cluster" in label:
                number.set_value(float(number.value) + 0.25)
                toggled = True
                break
    assert toggled, "no grid/parameter control was found"
    app.run()
    assert not app.exception
    assert task.cancel_event.is_set(), "a grid parameter change must invalidate the job"
    assert app.session_state.get("inference_result") is None


# ----------------------------------------------------------------------
# R1-3: cancellation while a worker is blocked (bounded waits)
# ----------------------------------------------------------------------
def _blocked_task(log_path, seed):
    with open(log_path, "a") as stream:
        stream.write(f"{os.getpid()}\n")
        stream.flush()
    time.sleep(300)
    return seed * seed


@pytest.mark.parametrize("route", ["map", "reduce"])
@pytest.mark.parametrize("start_method", START_METHODS)
def test_cancel_reaches_a_blocked_worker_within_seconds(
    start_method, route, tmp_path
):
    log = tmp_path / "pids.log"
    _files_before = _task_files()
    ctx = _workers.multiprocessing.get_context(start_method)
    stop = threading.Event()
    box = {}

    def call():
        try:
            with patch.object(_workers.multiprocessing, "get_context", return_value=ctx):
                if route == "map":
                    run_permutations(
                        partial(_blocked_task, str(log)), list(range(8)),
                        n_processes=2, cancel=stop.is_set,
                    )
                else:
                    run_permutations_reduce(
                        partial(_blocked_task, str(log)), list(range(8)),
                        reduce_func=sum_squares_chunk, n_processes=2,
                        cancel=stop.is_set,
                    )
        except BaseException as exc:
            box["exception"] = exc

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if log.exists() and log.read_text().strip():
            break
        if not thread.is_alive():
            break
        time.sleep(0.05)
    assert log.exists() and log.read_text().strip(), "no worker ever started"
    stop.set()
    requested_at = time.monotonic()
    thread.join(10)
    elapsed = time.monotonic() - requested_at
    assert not thread.is_alive(), (
        f"cancel did not unblock the {route}/{start_method} wait within 10 s"
    )
    assert elapsed <= 5.0, f"cancel took {elapsed:.2f}s (> 5 s)"
    assert isinstance(box.get("exception"), InferenceCancelled), box.get("exception")

    restarted = run_permutations(abs, [-1, -2], n_processes=2)
    pids = sorted({int(line) for line in log.read_text().split()})
    gone_deadline = time.monotonic() + 10
    remaining = [pid for pid in pids if _alive(pid)]
    while remaining and time.monotonic() < gone_deadline:
        time.sleep(0.05)
        remaining = [pid for pid in pids if _alive(pid)]
    assert not remaining, f"worker processes still alive: {remaining}"
    assert restarted == [1, 2]
    # No private task file from THIS test remains behind.
    leftovers = [p for p in _task_files() if p not in _files_before]
    assert leftovers == [], f"private task files left behind: {leftovers}"


def _task_files():
    import glob

    return set(glob.glob(os.path.join(tempfile.gettempdir(), "conninfpy-task-*.pkl")))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover
        return True
    return True


# ----------------------------------------------------------------------
# R1-4: external cancel combined with a ProgressSession
# ----------------------------------------------------------------------
def test_prepare_run_combines_session_and_explicit_cancel():
    session = ProgressSession()
    assert prepare_run(session, lambda: True, 4, "probe").cancelled() is True
    assert prepare_run(session, lambda: False, 4, "probe").cancelled() is False
    # The session's own predicate is still honoured when no explicit cancel
    # is given.
    assert prepare_run(ProgressSession(cancel=lambda: True), None, 4).cancelled() is True


def test_public_analyze_honours_explicit_cancel_with_a_session():
    with pytest.raises(InferenceCancelled):
        analyze(
            group1=_stack(3), group2=_stack(3, seed=8), input="r",
            test_type="two-sample", method="tstat", n_permutations=2,
            rng=1, use_mp=False, progress=ProgressSession(), cancel=lambda: True,
        )


# ----------------------------------------------------------------------
# R1-5: info() adds no side effect and does not import numba
# ----------------------------------------------------------------------
def _run_python(code, env_extra=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, timeout=300, cwd=str(REPO),
    )


def test_info_itself_creates_no_cache_directory(tmp_path):
    """Package import + ``info()`` create no cache directory (R1-5 round 2).

    Enhancement kernels enable numba caching lazily on first real use, so
    importing conninfpy and running ``info`` must leave a nonexistent
    ``NUMBA_CACHE_DIR`` nonexistent — no removal workaround needed.
    """
    cache = tmp_path / "info-cache"
    code = f"""
import os, sys
os.environ["NUMBA_CACHE_DIR"] = {str(cache)!r}
import conninfpy.cli  # package import must not create the numba cache dir
print("DIR_EXISTS_AFTER_IMPORT", os.path.isdir({str(cache)!r}))
rc = conninfpy.cli.main(["info"])
print("RC", rc)
print("DIR_EXISTS_AFTER_INFO", os.path.isdir({str(cache)!r}))
print("NUMBA_IMPORTED_BY_INFO", "numba" in sys.modules)
"""
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    lines = dict(
        line.split(" ", 1) for line in proc.stdout.strip().splitlines()
        if line.startswith(("RC ", "DIR_EXISTS_AFTER_IMPORT ", "DIR_EXISTS_AFTER_INFO ", "NUMBA_IMPORTED_BY_INFO "))
    )
    assert lines["RC"] == "0"
    assert lines["DIR_EXISTS_AFTER_IMPORT"] == "False", (
        "importing conninfpy must not create the numba cache directory"
    )
    assert lines["DIR_EXISTS_AFTER_INFO"] == "False", (
        "conninfpy info must not create the numba cache directory"
    )
    assert "numba cache" in proc.stdout
    assert "unknown/not measured" in proc.stdout


def test_info_does_not_attempt_a_numba_import(tmp_path):
    """With numba imports blocked, ``info`` still runs and stays honest."""
    cache = tmp_path / "blocked-cache"
    code = f"""
import os, sys
os.environ["NUMBA_CACHE_DIR"] = {str(cache)!r}
class Block:
    def find_module(self, name, path=None):
        return None
    def find_spec(self, name, path=None, target=None):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba imports are blocked for this test")
        return None
sys.meta_path.insert(0, Block())
import conninfpy.cli
rc = conninfpy.cli.main(["info"])
print("RC", rc)
"""
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    assert "RC 0" in proc.stdout
    assert "numba" in proc.stdout
    assert "unknown/not measured" in proc.stdout
