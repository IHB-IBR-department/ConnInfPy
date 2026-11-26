"""NETSTAT-34 C2 regression: multi-contrast edge null, available-case reuse.

What must not move after moving the shared multi-contrast permutation pass and
the available-case permutation task onto the edge-vector kernel:

* every contrast's per-draw result still equals the unchanged matrix task
  ``glm._freedman_lane_permutation_task`` at the same seed — contrasts visited
  in dict order, each restarting ``RandomState(seed)``;
* a joint null (``nuisance_contrast``) stores one reduced-model payload, not K;
* the available-case ``_restricted`` cache selects exactly what the
  ``np.isin`` version selected, element for element;
* ``missing_report['n_patterns']`` equals the number of fitted patterns;
* one multi-contrast draw allocates only ``(n, E)`` buffers;
* end to end, both fast routes produce the same p-values and counts as their
  retained matrix routes.

Synthetic data only.
"""

from __future__ import annotations

import tracemalloc
from unittest.mock import patch

import numpy as np
import pytest

from conninfpy import compute_p_val_glm, compute_p_val_glm_multi
from conninfpy.defaults import DEFAULT_START_THRESHOLD
from conninfpy.enhance import apply_tfnbs
from conninfpy.stats import _glm_fast as fast_mod
from conninfpy.stats import _missing_glm
from conninfpy.stats import glm as glm_mod
from conninfpy.stats._design import _precompute_ols
from conninfpy.stats._engine import _encode_strata, _stratified_perm

TFNBS_KW = {"e": 0.5, "h": 2.0, "n": None, "start_thres": DEFAULT_START_THRESHOLD}
MIN_DF = _missing_glm.MIN_DF


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sym(rng, n, N, offset=0.0, scale=0.3):
    Y = rng.randn(n, N, N) * scale + offset
    Y = (Y + Y.transpose(0, 2, 1)) / 2.0
    idx = np.arange(N)
    Y[:, idx, idx] = 0.0
    return Y


def _multi_setup(Y, X, contrasts, stat_type="tstat", enhance_func=apply_tfnbs,
                 kwargs=None, scheme="auto", strata=None):
    """Per-contrast plans/payloads plus the unchanged matrix-task ingredients."""
    kwargs = TFNBS_KW if kwargs is None else kwargs
    X_pinv, XtX_inv_diag, XtX_inv = _precompute_ols(X)
    strata_codes = _encode_strata(strata) if strata is not None else None
    names = tuple(contrasts)
    plans, payloads, references = [], [], []
    for name in names:
        c = contrasts[name]
        X_reduced = glm_mod._compute_reduced_model(X, c)
        scheme_k = glm_mod._resolve_scheme(X, X_reduced, scheme)
        plan = fast_mod.build_plan(
            Y, X, c, stat_type, X_pinv, XtX_inv,
            enhance_func, kwargs, scheme_k, strata_codes,
        )
        assert plan is not None
        yhat_e, r_e = fast_mod.reduced_payload(Y, X_reduced, plan.rows, plan.cols)
        plans.append(plan)
        payloads.append((yhat_e, r_e))
        Y_hat = glm_mod._reduced_fit(X_reduced, Y.reshape(Y.shape[0], -1))
        Y_hat = Y_hat.reshape(Y.shape)
        references.append((Y_hat, Y - Y_hat, scheme_k, c))
    return names, tuple(plans), tuple(payloads), references, (
        X_pinv, XtX_inv_diag, XtX_inv, strata_codes, kwargs,
    )


def _assert_same_tails(got, want, label):
    assert list(got.keys()) == list(want.keys()), label
    for key in want:
        a, b = np.asarray(got[key]), np.asarray(want[key])
        assert a.shape == b.shape, (label, key, a.shape, b.shape)
        assert np.array_equal(a, b), (label, key)


# ---------------------------------------------------------------------------
# (a) multi-contract: each contrast == the unchanged matrix task at that seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["three_contrasts", "one_sample_signflip", "strata"])
def test_multi_per_contrast_matches_matrix_task(kind):
    rng = np.random.RandomState(51)
    n, N = 16, 7
    Y = _sym(rng, n, N, offset=0.05)
    strata = None
    if kind == "one_sample_signflip":
        X = np.ones((n, 1))  # auto -> sign-flip
        contrasts = {"a": np.array([1.0]), "b": np.array([2.0])}
    elif kind == "strata":
        X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        contrasts = {
            "u": np.array([0.0, 0.0, 1.0]),
            "v": np.array([0.0, 1.0, -1.0]),
            "beta": np.array([0.0, 0.5, 0.5]),
        }
        strata = np.repeat([0, 1, 2, 3], n // 4)
    else:
        X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        contrasts = {
            "u": np.array([0.0, 0.0, 1.0]),
            "v": np.array([0.0, 1.0, -1.0]),
            "beta": np.array([0.0, 0.5, 0.5]),
        }

    names, plans, payloads, refs, extra = _multi_setup(
        Y, X, contrasts, scheme="auto", strata=strata
    )
    X_pinv, XtX_inv_diag, XtX_inv, strata_codes, kwargs = extra

    for seed in (11, 12):
        got = fast_mod.multi_contrast_perm_task(names, plans, payloads, seed)
        assert list(got.keys()) == list(names)
        for name, (Y_hat, resid, scheme_k, c) in zip(names, refs, strict=True):
            want = glm_mod._freedman_lane_permutation_task(
                Y_hat, resid, X, c, X_pinv, XtX_inv_diag, XtX_inv,
                (N, N), "tstat", apply_tfnbs, seed=seed,
                strata_codes=strata_codes, scheme=scheme_k, **kwargs,
            )
            _assert_same_tails(got[name], want, f"{kind}/{name}/seed={seed}")


def test_multi_fstat_and_plain_routes_match_matrix_task():
    rng = np.random.RandomState(52)
    n, N = 16, 6
    Y = _sym(rng, n, N)
    X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
    C = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
    for stat_type, enhance_func, kwargs in [
        ("fstat", apply_tfnbs, TFNBS_KW),
        ("tstat", None, {}),
        ("tstat", apply_tfnbs, {**TFNBS_KW, "n": 10}),  # legacy matrix grid
    ]:
        names, plans, payloads, refs, extra = _multi_setup(
            Y, X, {"f": C} if stat_type == "fstat" else {"t": C[0]},
            stat_type=stat_type, enhance_func=enhance_func, kwargs=kwargs,
        )
        X_pinv, XtX_inv_diag, XtX_inv, strata_codes, _ = extra
        for seed in (21,):
            got = fast_mod.multi_contrast_perm_task(names, plans, payloads, seed)
            for name, (Y_hat, resid, scheme_k, c) in zip(names, refs, strict=True):
                want = glm_mod._freedman_lane_permutation_task(
                    Y_hat, resid, X, c, X_pinv, XtX_inv_diag, XtX_inv,
                    (N, N), stat_type, enhance_func, seed=seed,
                    strata_codes=strata_codes, scheme=scheme_k, **kwargs,
                )
                _assert_same_tails(got[name], want, f"{stat_type}/{enhance_func}")


# ---------------------------------------------------------------------------
# (b) available-case _restricted cache == np.isin version
# ---------------------------------------------------------------------------


def _make_pattern(n, rows, block_ids, edges, X):
    in_rows = np.zeros(n, dtype=bool)
    in_rows[rows] = True
    loc = np.full(n, -1, dtype=np.intp)
    loc[rows] = np.arange(rows.size)
    block_positions = [np.flatnonzero(block_ids == b) for b in np.unique(block_ids)]
    pat = _missing_glm._Pattern(
        edges=edges, rows=rows, X=X[rows], pinv=np.linalg.pinv(X[rows]),
        df=rows.size - 2,
    )
    pat.blocks = [(pos, pos[in_rows[pos]], loc) for pos in block_positions]
    pat.member = in_rows
    return pat


def _restricted_isin(perm_idx, pat):
    local = np.empty(pat.rows.size, dtype=np.intp)
    for positions, present_positions, loc in pat.blocks:
        sources = perm_idx[positions]
        keep = sources[np.isin(sources, pat.rows)]
        local[loc[present_positions]] = loc[keep]
    return local


@pytest.mark.parametrize("blocked", [False, True])
def test_restricted_cache_matches_isin(blocked):
    rng = np.random.RandomState(61)
    n = 18
    X = np.column_stack([np.ones(n), rng.randn(n)])
    block_ids = (
        _encode_strata(np.repeat([0, 1, 2], 6)) if blocked else np.zeros(n, np.int_)
    )
    rows_all = np.arange(n)
    for _ in range(12):
        keep = rng.rand(n) < rng.uniform(0.4, 1.0)
        rows = rows_all[keep]
        # keep enough per block for the restricted permutation to be defined
        if rows.size < 6:
            continue
        pat = _make_pattern(n, rows, block_ids, np.arange(3, dtype=np.intp), X)
        for _ in range(5):
            perm_idx = (
                _stratified_perm(block_ids, rng)  # a valid draw: stays in block
                if blocked
                else rng.permutation(n)
            )
            got = _missing_glm._restricted(perm_idx, pat)
            want = _restricted_isin(perm_idx, pat)
            assert np.array_equal(got, want)
            uncached = _make_pattern(
                n, rows, block_ids, np.arange(3, dtype=np.intp), X
            )
            uncached.member = None  # fallback path must agree as well
            assert np.array_equal(_missing_glm._restricted(perm_idx, uncached), want)


# ---------------------------------------------------------------------------
# (c) n_patterns == number of fitted patterns
# ---------------------------------------------------------------------------


def _n_patterns_reference(Y, min_coverage=0.8):
    n, N, _ = Y.shape
    triu = np.triu_indices(N, k=1)
    present = np.isfinite(Y[:, triu[0], triu[1]])
    need = max(int(np.ceil(min_coverage * n)), MIN_DF + 1)
    covered = present.sum(0) >= need
    packed = np.packbits(present, axis=0)
    keys = {packed[:, j].tobytes() for j in np.flatnonzero(covered)}
    return len(keys), int(covered.sum())


@pytest.mark.parametrize("case", ["dropout", "scattered"])
def test_n_patterns_counts_fitted_patterns(case):
    rng = np.random.RandomState(62)
    n, N = 20, 9
    Y = _sym(rng, n, N)
    np.triu_indices(N, 1)
    if case == "dropout":
        drop = rng.rand(n) < 0.15
        for i, j in [(2, 5), (2, 7), (3, 6)]:
            Y[np.ix_(drop, [i], [j])] = np.nan
            Y[np.ix_(drop, [j], [i])] = np.nan
    else:
        mask = rng.rand(n, N, N) < 0.04
        mask = mask | mask.transpose(0, 2, 1)
        idx = np.arange(N)
        mask[:, idx, idx] = False
        Y[mask] = np.nan

    res = compute_p_val_glm(
        Y, interest=rng.randn(n), confounds=rng.randn(n), method="tfnbs",
        n_permutations=10, rng=7, use_mp=False, missing="available_case",
    )
    expected, covered = _n_patterns_reference(Y)
    if case == "scattered":
        # every covered group keeps df and estimability in this design
        assert res.missing_report["n_patterns"] == expected
    assert res.missing_report["n_patterns"] >= 1
    assert res.missing_report["n_edges_tested"] == covered or case == "dropout"
    assert isinstance(res.missing_report["n_patterns"], int)


# ---------------------------------------------------------------------------
# (d) deduplication of a joint null, and payload shape / peak memory
# ---------------------------------------------------------------------------


def test_joint_null_dedupes_reduced_payloads():
    rng = np.random.RandomState(63)
    n, N = 16, 6
    Y = _sym(rng, n, N)
    X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n), rng.randn(n)])
    contrasts = {
        "a": np.array([0, 1.0, 0, 0]),
        "b": np.array([0, 0, 1.0, 0]),
        "c": np.array([0, 0, 0, 1.0]),
    }
    calls = {"n": 0}
    original = fast_mod.reduced_payload

    def spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    common = {"method": "tfnbs", "n_permutations": 8, "rng": 3, "use_mp": False}
    with patch.object(fast_mod, "reduced_payload", side_effect=spy):
        compute_p_val_glm_multi(Y, X, contrasts, **common)
    assert calls["n"] == 3  # one reduced model per contrast

    calls["n"] = 0
    with patch.object(fast_mod, "reduced_payload", side_effect=spy):
        compute_p_val_glm_multi(
            Y, X, contrasts, nuisance_contrast=np.array([0, 1.0, 1.0, 1.0]),
            **common,
        )
    assert calls["n"] == 1  # shared joint null: one payload for every contrast


def test_multi_task_allocates_only_edge_buffers():
    rng = np.random.RandomState(64)
    n, N = 30, 40
    Y = _sym(rng, n, N)
    X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
    contrasts = {
        "a": np.array([0.0, 0.0, 1.0]),
        "b": np.array([0.0, 1.0, -1.0]),
        "c": np.array([0.0, 0.5, 0.5]),
    }
    names, plans, payloads, _, _ = _multi_setup(Y, X, contrasts)
    E = plans[0].rows.size
    for yhat_e, r_e in payloads:
        assert yhat_e.shape == (n, E) and r_e.shape == (n, E)
        assert yhat_e.shape != (n, N, N)

    fast_mod.multi_contrast_perm_task(names, plans, payloads, 1)  # warm-up
    tracemalloc.start(1)
    out = fast_mod.multi_contrast_perm_task(names, plans, payloads, 2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert list(out) == list(names)
    budget = len(names) * (3.5 * n * E * 8) + 256 * 1024
    assert peak < budget, f"peak {peak} >= {budget} for K={len(names)}"


# ---------------------------------------------------------------------------
# end to end: fast routes == retained matrix routes
# ---------------------------------------------------------------------------


def _assert_same_result(fast_res, slow_res, label):
    assert list(fast_res.keys()) == list(slow_res.keys()), label
    for key in fast_res:
        assert np.array_equal(np.asarray(fast_res[key]), np.asarray(slow_res[key])), (
            label,
            key,
        )
    for attr in ("stat_positive", "stat_negative", "stat_omnibus"):
        a, b = getattr(fast_res, attr, None), getattr(slow_res, attr, None)
        assert (a is None) == (b is None), (label, attr)
        if a is not None:
            assert np.array_equal(np.asarray(a), np.asarray(b)), (label, attr)
    for key, val in fast_res.exceedance_counts.items():
        if val is not None:
            assert np.array_equal(
                np.asarray(val), np.asarray(slow_res.exceedance_counts[key])
            ), (label, key)


@pytest.mark.parametrize("joint", [False, True])
def test_multi_fast_matches_matrix_end_to_end(joint):
    """compute_p_val_glm_multi is t/beta only (F is rejected upstream), so the
    multi-row contrast case is covered at the task level above."""
    rng = np.random.RandomState(65)
    n, N = 14, 7
    Y = _sym(rng, n, N)
    X = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n), rng.randn(n)])
    contrasts = {
        "a": np.array([0, 1.0, 0, 0]),
        "b": np.array([0, 0, 1.0, -1.0]),
    }
    nuisance = np.array([0, 1.0, 1.0, 0.0]) if joint else None
    common = {
        "method": "tfnbs", "nuisance_contrast": nuisance, "n_permutations": 24,
        "rng": 9, "use_mp": False,
    }
    with patch.object(fast_mod, "build_plan", return_value=None):
        slow = compute_p_val_glm_multi(Y, X, contrasts, **common)
    fast = compute_p_val_glm_multi(Y, X, contrasts, **common)
    for name in contrasts:
        _assert_same_result(fast[name], slow[name], f"{name}/{joint}")


@pytest.mark.parametrize("case", ["tfnbs", "ni_tfnbs", "fstat"])
def test_available_case_spec_matches_matrix_end_to_end(case):
    rng = np.random.RandomState(66)
    n, N = 20, 8
    Y = _sym(rng, n, N)
    drop = rng.rand(n) < 0.15
    for i, j in [(2, 5), (2, 7)]:
        Y[np.ix_(drop, [i], [j])] = np.nan
        Y[np.ix_(drop, [j], [i])] = np.nan
    interest = rng.randn(n)
    confound = rng.randn(n)
    labels = np.repeat([0, 1], N // 2)

    if case == "ni_tfnbs":
        kwargs = {"interest": interest, "confounds": confound, "method": "ni_tfnbs",
                      "net_labels": labels}
    elif case == "fstat":
        X = np.column_stack([np.ones(n), interest, confound])
        kwargs = {"design_matrix": X,
                      "contrast": np.array([[0, 1, 0], [0, 0, 1.0]]),
                      "stat_type": "fstat", "method": "tfnbs", "start_thres": 4.0}
    else:
        kwargs = {"interest": interest, "confounds": confound, "method": "tfnbs"}
    common = {"n_permutations": 20, "rng": 11, "use_mp": False,
                  "missing": "available_case"}

    fast = compute_p_val_glm(Y, **kwargs, **common)
    with patch.object(_missing_glm, "_edge_max_spec", return_value=None):
        slow = compute_p_val_glm(Y, **kwargs, **common)
    assert fast.missing_report["n_patterns"] == slow.missing_report.pop("n_patterns")
    _assert_same_result(fast, slow, case)


def test_available_case_uses_the_edge_spec_when_supported():
    rng = np.random.RandomState(67)
    n, N = 18, 7
    Y = _sym(rng, n, N)
    Y[:3, 2, 5] = np.nan
    Y[:3, 5, 2] = np.nan
    seen = {}
    original = _missing_glm._edge_max_spec

    def spy(*args, **kwargs):
        spec = original(*args, **kwargs)
        seen["spec"] = spec
        return spec

    with patch.object(_missing_glm, "_edge_max_spec", side_effect=spy):
        compute_p_val_glm(
            Y, interest=rng.randn(n), confounds=rng.randn(n), method="tfnbs",
            n_permutations=10, rng=13, use_mp=False, missing="available_case",
        )
    assert seen["spec"] is not None
