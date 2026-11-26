"""NETSTAT-34 regression: edge-vector Freedman-Lane null path (single contrast).

The permutation null of ``compute_p_val_glm`` now refits and scores the
``E = N(N-1)/2`` upper-triangle edge vector instead of rebuilding several
``(n, N, N)`` arrays per draw (``conninfpy/stats/_glm_fast.py``). The tests
here pin what must not move:

* per-draw ``t``/``beta``/``F`` against the *unchanged* oracle
  ``compute_glm_stat`` on the same permuted stack — permutation, sign-flip,
  blocked permutation, rank-deficient design, large offset near the floor
  (tolerance ``1e-12``; zero/floor decisions exact);
* the RNG stream: same seed, same draws as ``glm._shuffle_residuals``;
* no ``n x N**2`` allocation per draw (tracemalloc peak of one task call);
* the edge-vector enhancement maximum equals the matrix maximum on stat maps
  with exact ties, values straddling the 1e-10 rounding boundary, and no
  suprathreshold edges at all;
* end to end, the fast path and the retained matrix path produce identical
  p-values and exceedance counts.

Synthetic data only.
"""

from __future__ import annotations

import tracemalloc
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from conninfpy import compute_p_val_glm
from conninfpy.defaults import DEFAULT_START_THRESHOLD
from conninfpy.enhance import apply_nbs, apply_tfnbs
from conninfpy.stats import _glm_fast as fast_mod
from conninfpy.stats import glm as glm_mod
from conninfpy.stats._design import _precompute_ols
from conninfpy.stats._engine import _encode_strata, _extract_max_stats
from conninfpy.stats.pairwise import _build_edge_max_spec, _eval_edge_max_tail

TFNBS_KW = {"e": 0.5, "h": 2.0, "n": None, "start_thres": DEFAULT_START_THRESHOLD}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sym(rng: np.random.RandomState, n: int, N: int, offset: float = 0.0,
         scale: float = 0.3) -> np.ndarray:
    """Exactly symmetric zero-diagonal subject stack (as check_connectivity)."""
    Y = rng.randn(n, N, N) * scale + offset
    Y = (Y + Y.transpose(0, 2, 1)) / 2.0
    idx = np.arange(N)
    Y[:, idx, idx] = 0.0
    return Y


def _prepare(Y, X, contrast, stat_type, enhance_func=None, enhance_kwargs=None,
             scheme="permutation", strata=None):
    """Full-matrix baseline pieces and the edge-vector plan for one run."""
    X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)
    X_reduced = glm_mod._compute_reduced_model(X, contrast)
    plan = fast_mod.build_plan(
        Y, X, contrast, stat_type, X_pinv, XtX_inv,
        enhance_func, enhance_kwargs or {}, scheme, strata,
    )
    assert plan is not None, "edge-vector plan unexpectedly unavailable"
    yhat_e, r_e = fast_mod.reduced_payload(Y, X_reduced, plan.rows, plan.cols)
    return plan, yhat_e, r_e, (X_pinv, XtX_inv_diag, XtX_inv, X_reduced)


def _matrix_payload(Y, X_reduced):
    Y_hat = glm_mod._reduced_fit(X_reduced, Y.reshape(Y.shape[0], -1))
    Y_hat = Y_hat.reshape(Y.shape)
    return Y_hat, Y - Y_hat


# ---------------------------------------------------------------------------
# (a) per-draw statistics against the unchanged oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stat_type", ["tstat", "beta", "fstat"])
@pytest.mark.parametrize("scheme,blocked", [
    ("permutation", False),
    ("sign-flip", False),
    ("permutation", True),
])
def test_edge_statistics_match_oracle_per_draw(stat_type, scheme, blocked):
    """t/beta/F per edge == compute_glm_stat on the very same Y*."""
    rng = np.random.RandomState(11)
    n, N = 14, 7
    Y = _sym(rng, n, N, offset=0.1)
    interest = rng.randn(n)
    X = np.column_stack([np.ones(n), rng.randn(n), interest])
    contrast = (
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
        if stat_type == "fstat"
        else np.array([0.0, 0.0, 1.0])
    )
    strata = _encode_strata(np.repeat([0, 1], n // 2)) if blocked else None

    plan, yhat_e, r_e, (X_pinv, XtX_inv_diag, XtX_inv, X_reduced) = _prepare(
        Y, X, contrast, stat_type, scheme=scheme, strata=strata
    )
    Y_hat, resid = _matrix_payload(Y, X_reduced)
    rows, cols = plan.rows, plan.cols

    for seed in (101, 202, 303):
        shuffled = glm_mod._shuffle_residuals(
            resid, np.random.RandomState(seed), scheme, strata
        )
        Y_perm = Y_hat + shuffled
        oracle = glm_mod.compute_glm_stat(
            Y_perm, X, contrast, stat_type=stat_type,
            X_pinv=X_pinv, XtX_inv_diag=XtX_inv_diag, XtX_inv=XtX_inv,
        )
        Y_e = fast_mod._draw_y_edge(
            yhat_e, r_e, np.random.RandomState(seed), scheme, strata
        )
        assert np.array_equal(Y_e, Y_perm[:, rows, cols])

        edge = fast_mod.edge_statistics(plan, Y_e).stat
        if stat_type == "fstat":
            ref = oracle["omnibus"][rows, cols]
        else:
            ref = oracle["positive"][rows, cols] - oracle["negative"][rows, cols]

        tol = 1e-12 * np.maximum(1.0, np.abs(ref))
        assert np.all(np.abs(edge - ref) <= tol), (
            f"max |delta| = {np.max(np.abs(edge - ref))}"
        )
        # Floor / zero decisions must be identical, not merely close.
        assert np.array_equal(edge == 0.0, ref == 0.0)


def test_edge_statistics_match_oracle_rank_deficient_and_near_floor():
    """Rank-deficient design, and a large offset whose t is exactly 0 everywhere."""
    rng = np.random.RandomState(12)
    n, N = 14, 6
    Y = _sym(rng, n, N)
    col = rng.randn(n)
    X = np.column_stack([np.ones(n), col, col.copy()])  # duplicated column
    contrast = np.array([0.0, 0.5, 0.5])  # estimable combination
    plan, yhat_e, r_e, pieces = _prepare(Y, X, contrast, "tstat")

    Y_floor = _sym(rng, n, N, offset=1e6, scale=1e-3)
    plan_floor, yhat_f, r_f, pieces_f = _prepare(Y_floor, X, contrast, "tstat")

    for label, (Yc, plan_c, yhat_c, r_c, pc) in {
        "rankdef": (Y, plan, yhat_e, r_e, pieces),
        "nearfloor": (Y_floor, plan_floor, yhat_f, r_f, pieces_f),
    }.items():
        Xp, Xd, Xi, Xr = pc
        Y_hat, resid = _matrix_payload(Yc, Xr)
        for seed in (7, 8):
            shuffled = glm_mod._shuffle_residuals(
                resid, np.random.RandomState(seed), "permutation", None
            )
            oracle = glm_mod.compute_glm_stat(
                Y_hat + shuffled, X, contrast, stat_type="tstat",
                X_pinv=Xp, XtX_inv_diag=Xd, XtX_inv=Xi,
            )
            rows, cols = plan_c.rows, plan_c.cols
            ref = oracle["positive"][rows, cols] - oracle["negative"][rows, cols]
            edge = fast_mod.edge_statistics(
                plan_c,
                fast_mod._draw_y_edge(
                    yhat_c, r_c, np.random.RandomState(seed), "permutation", None
                ),
            ).stat
            tol = 1e-12 * np.maximum(1.0, np.abs(ref))
            assert np.all(np.abs(edge - ref) <= tol), label
            assert np.array_equal(edge == 0.0, ref == 0.0), label
            if label == "nearfloor":
                # Every edge is at the residual-variance floor: t is exactly 0.
                assert np.all(edge == 0.0)


# ---------------------------------------------------------------------------
# (b) RNG stream identity with the baseline helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme,strata", [
    ("permutation", None),
    ("sign-flip", None),
    ("permutation", _encode_strata(np.repeat([0, 1, 2, 3], 4))),
])
def test_draw_consumes_rng_exactly_like_shuffle_residuals(scheme, strata):
    rng = np.random.RandomState(20)
    n, E = 16, 21
    resid = rng.randn(n, E)
    yhat = rng.randn(n, E)

    for seed in (1, 2, 3):
        rng_fast = np.random.RandomState(seed)
        got = fast_mod._draw_y_edge(yhat, resid, rng_fast, scheme, strata)
        rng_base = np.random.RandomState(seed)
        want = yhat + glm_mod._shuffle_residuals(resid, rng_base, scheme, strata)
        assert np.array_equal(got, want)
        # Same seed *and* the same number of draws: the streams stay aligned.
        assert rng_fast.randint(0, 2**31 - 1) == rng_base.randint(0, 2**31 - 1)


# ---------------------------------------------------------------------------
# (c) one task call never allocates n x N**2
# ---------------------------------------------------------------------------


def test_permutation_task_allocates_only_edge_buffers():
    """Peak of one draw at N=60, n=40 stays within a few (n, E) buffers."""
    rng = np.random.RandomState(21)
    n, N = 40, 60
    Y = _sym(rng, n, N)
    rows = np.triu_indices(N, 1)[0]
    E = rows.size
    X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
    contrast = np.array([0.0, 0.0, 1.0])

    plan, yhat_e, r_e, pieces = _prepare(Y, X, contrast, "tstat", apply_tfnbs, TFNBS_KW)
    X_pinv, XtX_inv_diag, XtX_inv, X_reduced = pieces
    assert plan.spec is not None, "expected the direct edge-max route"
    assert yhat_e.shape == (n, E) and r_e.shape == (n, E)

    fast_mod.permutation_task(plan, yhat_e, r_e, 5)  # warm-up (numba JIT)

    tracemalloc.start(1)
    out = fast_mod.permutation_task(plan, yhat_e, r_e, 6)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert set(out) == {"positive", "negative"}
    budget = 3.5 * n * E * 8 + 128 * 1024  # a few (n, E) buffers + fixups
    assert peak < budget, f"peak {peak} bytes >= {budget}"

    # Positive control: the same draw through the retained matrix task must
    # exceed the budget, so the assertion above is discriminating.
    Y_hat, resid = _matrix_payload(Y, X_reduced)
    matrix_task = partial(
        glm_mod._freedman_lane_permutation_task,
        Y_hat, resid, X, contrast, X_pinv, XtX_inv_diag, XtX_inv,
        (N, N), "tstat", apply_tfnbs, **TFNBS_KW,
    )
    matrix_task(5)  # warm-up
    tracemalloc.start(1)
    matrix_task(6)
    _, peak_matrix = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_matrix > budget, (
        f"matrix task peak {peak_matrix} not above the edge budget {budget}"
    )


# ---------------------------------------------------------------------------
# (d) edge-vector enhancement maximum == matrix maximum
# ---------------------------------------------------------------------------


def _assemble(vec, N, rows, cols):
    mat = np.zeros((N, N), dtype=np.float64)
    mat[rows, cols] = vec
    return mat + mat.T


@pytest.mark.parametrize("case", ["ties", "rounding", "empty", "graded"])
def test_edge_max_matches_matrix_max(case):
    rng = np.random.RandomState(31)
    N = 9
    rows, cols = np.triu_indices(N, 1)
    if case == "ties":
        tail = rng.choice([0.0, 0.5, 1.0, 2.0, 3.0], size=rows.size)
    elif case == "rounding":
        # values straddling the 1e-10 rounding boundary and the cutoff 1.0
        tail = rng.choice(
            [1.0 + 5e-11, 1.0 - 5e-11, 2.0 + 4e-11, 0.9999999999, 0.0],
            size=rows.size,
        )
    elif case == "empty":
        tail = np.zeros(rows.size)  # no suprathreshold edge at all
    else:
        tail = np.abs(rng.randn(rows.size)) * rng.choice([0.0, 1.0], size=rows.size)

    matrix = _assemble(tail, N, rows, cols)
    stat_dict = {"positive": matrix, "negative": matrix.copy()}

    cases = [
        (apply_tfnbs, {**TFNBS_KW}, "positive"),
        (apply_tfnbs, {"e": [0.5, 1.0], "h": [1.0, 2.0], "n": None,
                       "start_thres": DEFAULT_START_THRESHOLD}, "positive"),
        (apply_nbs, {"threshold": 1.5, "nbs_stat": "extent"}, "positive"),
        (apply_nbs, {"threshold": 1.0, "nbs_stat": "intensity"}, "positive"),
    ]
    for enhance_func, kwargs, key in cases:
        spec = _build_edge_max_spec(enhance_func, kwargs, (rows, cols), N)
        assert spec is not None
        got = _eval_edge_max_tail(tail, spec)
        want = _extract_max_stats(enhance_func(stat_dict, **kwargs), (N, N))[key]
        assert np.array_equal(np.asarray(got), np.asarray(want)), (case, kwargs)


# ---------------------------------------------------------------------------
# end to end: fast route == retained matrix route
# ---------------------------------------------------------------------------


def _run(Y, **kwargs):
    return compute_p_val_glm(Y, use_mp=False, **kwargs)


def _assert_same_result(fast_res, slow_res):
    assert list(fast_res.keys()) == list(slow_res.keys())
    for key in fast_res:
        assert np.array_equal(
            np.asarray(fast_res[key]), np.asarray(slow_res[key])
        ), key
    for attr in ("stat_positive", "stat_negative", "stat_omnibus"):
        a, b = getattr(fast_res, attr, None), getattr(slow_res, attr, None)
        assert (a is None) == (b is None), attr
        if a is not None:
            assert np.array_equal(np.asarray(a), np.asarray(b)), attr
    assert set(fast_res.exceedance_counts) == set(slow_res.exceedance_counts)
    for key, val in fast_res.exceedance_counts.items():
        if val is not None:
            assert np.array_equal(
                np.asarray(val), np.asarray(slow_res.exceedance_counts[key])
            ), key


@pytest.mark.parametrize("case", ["tstat_tfnbs", "strata", "fstat", "nbs", "nearfloor"])
def test_fast_path_matches_matrix_path_end_to_end(case):
    rng = np.random.RandomState(41)
    n, N = 14, 8
    B = 40
    if case == "nearfloor":
        Y = _sym(rng, n, N, offset=1e5, scale=1e-3)
    else:
        Y = _sym(rng, n, N)
    interest = rng.randn(n)
    confound = rng.randn(n)
    base = {"n_permutations": B, "rng": 4242}

    if case == "fstat":
        X = np.column_stack([np.ones(n), interest, confound, rng.randn(n)])
        kwargs = {
            "design_matrix": X,
            "contrast": np.array([[0, 1, 0, 0], [0, 0, 1, -1.0]]),
            "stat_type": "fstat", "method": "tfnbs",
        }
    elif case == "nbs":
        kwargs = {"interest": interest, "confounds": confound, "method": "nbs",
                      "threshold": 1.5, "nbs_stat": "extent"}
    elif case == "strata":
        kwargs = {"interest": interest, "confounds": confound, "method": "tfnbs",
                      "strata": np.repeat([0, 1], n // 2)}
    else:
        kwargs = {"interest": interest, "confounds": confound, "method": "tfnbs"}

    with patch.object(fast_mod, "build_plan", return_value=None):
        slow = _run(Y, **{**base, **kwargs})
    fast = _run(Y, **{**base, **kwargs})
    _assert_same_result(fast, slow)


def test_fast_path_is_actually_used():
    """Guard: the end-to-end comparisons above must exercise the edge route."""
    rng = np.random.RandomState(43)
    n, N = 12, 6
    Y = _sym(rng, n, N)
    interest = rng.randn(n)
    seen = {}
    original = fast_mod.build_plan

    def spy(*args, **kwargs):
        plan = original(*args, **kwargs)
        seen["plan"] = plan
        return plan

    with patch.object(fast_mod, "build_plan", side_effect=spy):
        compute_p_val_glm(
            Y, interest=interest, confounds=rng.randn(n), method="tfnbs",
            n_permutations=10, rng=1, use_mp=False,
        )
    assert seen["plan"] is not None
    assert seen["plan"].spec is not None
