"""NETSTAT-27: parent-side permutation progress and cooperative cancellation.

The progress callback reports permutations that were actually folded, in the
parent process only; cancellation happens at a permutation chunk boundary and
never yields a partial reduction or an ``InferenceResult``. These tests check
the semantics on the routes the app really uses (pairwise, GLM, multi-contrast
GLM, available-case) and that adding a callback or a cancel predicate changes
no number.

Synthetic data only; no hosted services.
"""

from __future__ import annotations

import numpy as np
import pytest

from conninfpy import InferenceCancelled, ProgressSession, analyze
from conninfpy.stats._workers import prepare_run
from conninfpy.stats.pairwise import compute_p_val


N_ROI = 5


def _stack(n, seed=3, N=N_ROI):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, N, N))
    for i in range(n):
        a = rng.standard_normal((N, 20))
        c = np.corrcoef(a)
        np.fill_diagonal(c, 1.0)
        out[i] = c
    return out


def _matrix_record(values):
    return np.asarray(values, dtype=float)


def _assert_same(without, with_cb):
    for key in ("positive", "negative"):
        a = np.asarray(without[key])
        b = np.asarray(with_cb[key])
        assert np.array_equal(a, b, equal_nan=True), key


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, done, total):
        self.calls.append((int(done), int(total)))

    @property
    def last(self):
        return self.calls[-1]


def _check_monotonic(calls, total_permutations):
    assert calls, "no progress callback was made"
    dones = [d for d, _ in calls]
    totals = [t for _, t in calls]
    assert dones == sorted(dones), "done must be monotonic"
    assert dones[-1] == total_permutations
    assert calls[-1][1] == total_permutations
    assert all(d <= t for d, t in calls)
    assert totals[-1] == total_permutations


# ----------------------------------------------------------------------
# progress: pairwise / GLM / multi / available-case
# ----------------------------------------------------------------------
def test_pairwise_progress_and_numerical_identity():
    g1, g2 = _stack(6), _stack(6, seed=9)
    base = analyze(
        group1=g1, group2=g2, input="r", test_type="two-sample", method="tfnbs",
        n_permutations=8, rng=1, use_mp=False,
    )
    rec = _Recorder()
    with_cb = analyze(
        group1=g1, group2=g2, input="r", test_type="two-sample", method="tfnbs",
        n_permutations=8, rng=1, use_mp=False, progress=rec,
    )
    _assert_same(base, with_cb)
    _check_monotonic(rec.calls, 8)
    # Chunk-boundary reporting, not one callback per permutation of IPC.
    assert len({done for done, _ in rec.calls}) <= 8


def test_glm_progress_and_numerical_identity():
    Y = _stack(9)
    interest = np.linspace(0.0, 1.0, 9)
    base = analyze(
        Y=Y, interest=interest, input="r", test_type="glm", method="tfnbs",
        n_permutations=6, rng=2, use_mp=False,
    )
    rec = _Recorder()
    with_cb = analyze(
        Y=Y, interest=interest, input="r", test_type="glm", method="tfnbs",
        n_permutations=6, rng=2, use_mp=False, progress=rec,
    )
    _assert_same(base, with_cb)
    _check_monotonic(rec.calls, 6)


def test_multi_contrast_progress_uses_shared_permutation_total():
    Y = _stack(10)
    rng = np.random.default_rng(17)
    interest = {
        "age": rng.standard_normal(10),
        "motion": rng.standard_normal(10),
    }
    base = analyze(
        Y=Y, interest=interest, input="r", test_type="glm", method="tfnbs",
        n_permutations=5, rng=3, use_mp=False,
    )
    rec = _Recorder()
    with_cb = analyze(
        Y=Y, interest=interest, input="r", test_type="glm", method="tfnbs",
        n_permutations=5, rng=3, use_mp=False, progress=rec,
    )
    for name in base:
        _assert_same(base[name], with_cb[name])
    # One shared permutation pass for every contrast: total is n_permutations.
    _check_monotonic(rec.calls, 5)


def test_available_case_reduce_route_progress():
    g1, g2 = _stack(6), _stack(6, seed=11)
    g1[0, 1, 2] = g1[0, 2, 1] = np.nan
    rec = _Recorder()
    out = analyze(
        group1=g1, group2=g2, input="r", test_type="two-sample",
        method="bh_fdr_perm", n_permutations=6, rng=4, use_mp=False,
        missing="available_case", progress=rec,
    )
    assert out is not None
    _check_monotonic(rec.calls, 6)


@pytest.mark.parametrize("use_mp", [False])
def test_no_callback_keeps_previous_behaviour(use_mp):
    rng = np.random.default_rng(0)
    d = rng.standard_normal((4, N_ROI, N_ROI))
    d = (d + d.transpose(0, 2, 1)) / 2
    for i in range(4):
        np.fill_diagonal(d[i], 1.0)
    first = compute_p_val(
        d[:2], d[2:], n_permutations=5, method="tstat", rng=7, use_mp=use_mp
    )
    second = compute_p_val(
        d[:2], d[2:], n_permutations=5, method="tstat", rng=7, use_mp=use_mp
    )
    assert np.array_equal(
        np.asarray(first["positive"]), np.asarray(second["positive"]), equal_nan=True
    )


# ----------------------------------------------------------------------
# ProgressSession bookkeeping (total is announced, never invented)
# ----------------------------------------------------------------------
def test_session_expect_is_redeemed_not_double_counted():
    calls = []
    session = ProgressSession(callback=lambda d, t: calls.append((d, t)))
    session.expect(10, "companion planned")
    assert session.total == 10 and session.done == 0
    run = prepare_run(session, None, 10, "primary")
    assert session.total == 10, "a redeemed announcement must not be counted twice"
    run.update(4)
    run.finish()
    assert session.done == 10 and session.total == 10
    assert calls[-1] == (10, 10)


def test_session_finalize_drops_unused_announcement():
    calls = []
    session = ProgressSession(callback=lambda d, t: calls.append((d, t)))
    session.expect(10, "companion planned")  # never redeemed by a real loop
    run = prepare_run(session, None, 4, "primary")
    assert session.total == 14
    run.update(4)
    run.finish()
    session.finalize()
    assert session.done == 4 and session.total == 4
    assert session.unused_plan == 10
    assert calls[-1] == (4, 4)


def test_prepare_run_without_progress_is_a_no_op():
    run = prepare_run(None, None, 3)
    run.update(1)
    run.finish()
    assert not run.cancelled()


# ----------------------------------------------------------------------
# cancellation
# ----------------------------------------------------------------------
def test_cancel_before_start_raises_without_any_result():
    g1, g2 = _stack(6), _stack(6, seed=5)
    with pytest.raises(InferenceCancelled):
        analyze(
            group1=g1, group2=g2, input="r", test_type="two-sample",
            method="tfnbs", n_permutations=8, rng=1, use_mp=False,
            cancel=lambda: True,
        )


def test_cancel_mid_run_stops_before_total_and_returns_no_result():
    g1, g2 = _stack(6), _stack(6, seed=5)
    seen = []
    result = {"value": "not set"}

    def cancel():
        return len(seen) >= 2

    def progress(done, total):
        seen.append((done, total))

    with pytest.raises(InferenceCancelled):
        result["value"] = analyze(
            group1=g1, group2=g2, input="r", test_type="two-sample",
            method="tfnbs", n_permutations=12, rng=1, use_mp=False,
            progress=progress, cancel=cancel,
        )
    assert result["value"] == "not set", "no InferenceResult may escape a cancelled run"
    assert seen, "progress is reported up to the cancellation boundary"
    assert seen[-1][0] < seen[-1][1], "cancellation never reports a finished total"


def test_cancel_in_reduce_route_raises():
    from functools import partial

    from conninfpy.stats._workers import run_permutations_reduce
    from tests.mp_helpers import sum_squares_chunk

    calls = []

    def cancel():
        return len(calls) >= 1

    with pytest.raises(InferenceCancelled):
        run_permutations_reduce(
            partial(pow, 2),
            list(range(20)),
            reduce_func=sum_squares_chunk,
            use_mp=False,
            progress=lambda d, t: calls.append((d, t)),
            cancel=cancel,
        )


def test_cancelled_run_leaves_no_result_on_the_analyze_contract():
    """A cancelled run must not be mistakable for a completed one."""
    g1, g2 = _stack(6), _stack(6, seed=5)
    with pytest.raises(InferenceCancelled):
        analyze(
            group1=g1, group2=g2, input="r", test_type="two-sample",
            method="tfnbs", n_permutations=6, rng=1, use_mp=False,
            cancel=lambda: True,
        )
    # An identical, uncancelled run still completes (no poisoned global state).
    out = analyze(
        group1=g1, group2=g2, input="r", test_type="two-sample",
        method="tfnbs", n_permutations=6, rng=1, use_mp=False,
    )
    assert out["positive"].shape == (N_ROI, N_ROI)
