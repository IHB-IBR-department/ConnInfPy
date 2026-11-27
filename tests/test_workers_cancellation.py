"""NETSTAT-27: real spawn/forkserver cancellation, cleanup and restart.

No mocked pools: every check below starts actual worker processes, cancels at
a permutation chunk boundary, and asserts that the children are terminated and
joined, that the private task file is gone, and that the next run in the same
parent process completes normally.

Synthetic data only; no hosted services.
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
import tempfile
import threading
import time
from functools import partial
from unittest.mock import patch

import pytest

from conninfpy.stats import _workers
from conninfpy.stats._workers import (
    InferenceCancelled,
    ProgressSession,
    run_permutations,
    run_permutations_reduce,
)
from tests.mp_helpers import (
    slow_square_logging_pid,
    sum_squares_chunk,
    worker_threads,
)

START_METHODS = [
    name
    for name in ("spawn", "forkserver")
    if name in _workers.multiprocessing.get_all_start_methods()
]

TASK_FILE_PATTERN = os.path.join(tempfile.gettempdir(), "conninfpy-task-*.pkl")


@pytest.fixture(params=START_METHODS)
def worker_context(request):
    ctx = _workers.multiprocessing.get_context(request.param)
    with patch.object(_workers.multiprocessing, "get_context", return_value=ctx):
        yield ctx


def _task_files():
    return set(glob.glob(TASK_FILE_PATTERN))


def _alive(pid):
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not expected locally
        return True
    return True


def _run_guarded(func, seconds=60.0):
    """Run ``func`` in a thread with a hard timeout (no hanging tests)."""
    box = {}

    def target():
        try:
            box["value"] = func()
        except BaseException as exc:  # re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"cancellation did not finish within {seconds}s"
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _cancelling_progress(stop, records, after=1):
    def progress(done, total):
        records.append((done, total))
        if done >= after:
            stop.set()

    def cancel():
        return stop.is_set()

    return progress, cancel


def _assert_children_gone(log_path):
    pids = sorted({int(line) for line in log_path.read_text().split()})
    assert pids, "no worker process recorded a PID"
    deadline = time.monotonic() + 10
    remaining = [pid for pid in pids if _alive(pid)]
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [pid for pid in pids if _alive(pid)]
    assert not remaining, f"worker processes still alive after cancel: {remaining}"
    return pids


@pytest.mark.parametrize("route", ["map", "reduce"])
def test_cancel_mid_run_terminates_workers_removes_files_and_restarts(
    worker_context, tmp_path, route
):
    log_path = tmp_path / "pids.log"
    task = partial(slow_square_logging_pid, str(log_path))
    before = _task_files()
    stop = threading.Event()
    records = []
    progress, cancel = _cancelling_progress(stop, records)

    def cancel_route():
        with pytest.raises(InferenceCancelled):
            if route == "map":
                run_permutations(
                    task,
                    list(range(40)),
                    n_processes=2,
                    progress=progress,
                    cancel=cancel,
                )
            else:
                run_permutations_reduce(
                    task,
                    list(range(40)),
                    reduce_func=sum_squares_chunk,
                    n_processes=2,
                    progress=progress,
                    cancel=cancel,
                )

    _run_guarded(cancel_route)

    assert records, "at least one folded chunk must be reported before cancelling"
    assert records[-1][0] < records[-1][1], "a cancelled run never reports a full total"
    _assert_children_gone(log_path)
    assert _task_files() == before, "the private task file must be removed on cancel"

    # The same parent process starts a fresh pool and completes.
    out = run_permutations(abs, [-1, -2, -3], n_processes=2)
    assert out == [1, 2, 3]
    assert _task_files() == before


def test_cancel_with_no_progress_still_cleans_up(worker_context, tmp_path):
    log_path = tmp_path / "pids.log"
    task = partial(slow_square_logging_pid, str(log_path))
    before = _task_files()
    stop = threading.Event()

    def cancel_route():
        with pytest.raises(InferenceCancelled):
            run_permutations(
                task,
                list(range(40)),
                n_processes=2,
                cancel=lambda: stop.is_set(),
            )

    def cancel_soon():
        time.sleep(0.4)
        stop.set()

    threading.Thread(target=cancel_soon, daemon=True).start()
    _run_guarded(cancel_route)
    # With bounded polling the cancel may land before a worker even starts;
    # any PID that did start must still be gone.
    if log_path.exists():
        _assert_children_gone(log_path)
    assert _task_files() == before


def test_cancel_before_start_never_creates_a_pool(worker_context):
    def boom(seed):  # pragma: no cover - must not run
        raise AssertionError("a cancelled run must not execute permutations")

    with patch.object(_workers.multiprocessing, "get_context") as factory:
        with pytest.raises(InferenceCancelled):
            run_permutations(boom, [1, 2, 3], n_processes=2, cancel=lambda: True)
        factory.assert_not_called()


def test_worker_exception_still_joins_workers(worker_context, tmp_path):
    from tests.mp_helpers import fail_on_seed

    before = _task_files()
    with pytest.raises(ValueError):
        run_permutations(fail_on_seed, [1, 2, 3], n_processes=2)
    assert _task_files() == before
    # No worker leaked: the next run in this process succeeds.
    assert run_permutations(abs, [-4, -5], n_processes=2) == [4, 5]


# ----------------------------------------------------------------------
# No callback / event / UI object can reach a worker.
# ----------------------------------------------------------------------
def test_unpicklable_callback_and_cancel_still_run_workers(worker_context):
    """Lambdas are unpicklable: pool start proves neither is serialized."""
    records = []
    progress = lambda done, total: records.append((done, total))  # noqa: E731
    cancel = lambda: False  # noqa: E731
    out = run_permutations(
        worker_threads, [1, 1], n_processes=2, progress=progress, cancel=cancel
    )
    assert [r["seed"] for r in out] == [1, 1]
    assert records and records[-1] == (2, 2)
    parent_pid = os.getpid()
    assert {r["pid"] for r in out} != {parent_pid}


def test_task_payload_contains_no_event_or_ui_objects():
    """The serialized task payload carries only the analysis task."""
    session = ProgressSession(
        callback=lambda done, total: None, cancel=threading.Event().is_set
    )
    with _workers._task_file(partial(pow, 2)) as path:
        payload = open(path, "rb").read()
    assert b"threading" not in payload
    assert b"Event" not in payload
    assert b"streamlit" not in payload
    assert session.total == 0  # the session is parent-side bookkeeping only
    # A plain pickle round-trip of the payload is enough to rebuild the task.
    assert callable(pickle.loads(payload))


def test_reduce_cancel_returns_no_partial_total(worker_context):
    stop = threading.Event()

    def cancel():
        return stop.is_set()

    def on_progress(done, total):
        stop.set()

    with pytest.raises(InferenceCancelled):
        run_permutations_reduce(
            partial(slow_square_logging_pid, os.devnull),
            list(range(20)),
            reduce_func=sum_squares_chunk,
            n_processes=2,
            progress=on_progress,
            cancel=cancel,
        )
