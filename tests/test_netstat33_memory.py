"""NETSTAT-33: bounded-memory empirical p-values and streaming bh_fdr_perm.

Two properties are pinned down here:

1. Exactness — the searchsorted empirical-p counting and the streaming
   bh_fdr_perm reduction reproduce the previous broadcast / collect-list
   implementations bit for bit (ties included, +1 Phipson–Smyth kept).
2. Boundedness — the streaming reduction never lets an unbounded number of
   issued-but-unreduced chunk results accumulate in the parent, verified
   with instrumentation and a deliberately slow first chunk against fast
   followers (serial and multiprocessing with 2 workers).
"""

import multiprocessing
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from conninfpy.stats import _workers
from conninfpy.stats._engine import (
    _bh_fdr_correction,
    _compute_p_values_from_null,
)
from conninfpy.stats._missing_pairwise import (
    _build_design,
    _observed,
    _task_edges,
    available_case_p_val,
)
from conninfpy.stats.acceleration import _empirical_p
from conninfpy.stats.pairwise import (
    _compute_bh_fdr_perm_p_values,
    _fast_permutation_task_ind_edges,
    _fast_permutation_task_paired_edges,
    _precompute_edge_sums,
    _precompute_twosample_sums,
    compute_t_stat,
    compute_t_stat_diff,
)
from conninfpy.stats._workers import run_permutations, run_permutations_reduce
from tests.mp_helpers import slow_first_chunk, sum_squares_chunk

START_METHODS = [
    m for m in ("forkserver", "spawn") if m in multiprocessing.get_all_start_methods()
]


# ---------------------------------------------------------------------------
# Independent oracles: the previous broadcast / collect-list implementations.
# ---------------------------------------------------------------------------


def _oracle_broadcast_p(observed, null):
    """The previous (N, ..., B) boolean-broadcast empirical p-value count."""
    if null.ndim == 1:
        expanded = observed[..., np.newaxis]
        count = np.sum(expanded <= null, axis=-1)
    else:
        # (N, N, P) vs (B, P): null reshaped to (1, 1, P, B)
        expanded = observed[..., np.newaxis]
        null_reshaped = null.swapaxes(0, 1)[None, None, ...]
        count = np.sum(expanded <= null_reshaped, axis=-1)
    return (count + 1.0) / (null.shape[0] + 1.0), count


def _empirical_p_cases():
    rg = np.random.default_rng(33)
    # Ties matter: quantize both nulls and observations onto a coarse grid.
    null_1d = np.round(rg.normal(size=61), 1)
    obs_2d = np.round(rg.normal(size=(7, 7)), 1)
    obs_multi = np.round(rg.normal(size=(6, 6, 3)), 1)
    null_multi = np.round(rg.normal(size=(40, 3)), 1)
    return [
        ("plain_2d", obs_2d, null_1d),
        ("multi_param", obs_multi, null_multi),
        ("multi_obs_1d_null", obs_multi, null_1d),
        ("observed_above_null", obs_2d + 50.0, null_1d),
        ("observed_below_null", obs_2d - 50.0, null_1d),
    ]


def _acceleration_p_cases():
    """_empirical_p takes a 1D null; skip the per-parameter (B, P) null."""
    return [case for case in _empirical_p_cases() if case[2].ndim == 1]


def _engine_p_cases():
    """Engine grid: the multi-param branch requires a (B, P) null."""
    return [
        (name, observed, null)
        for name, observed, null in _empirical_p_cases()
        if observed.ndim == 2 or null.ndim == 2
    ]


def _base_broadcast_oracle(emp_t, null_dist):
    """The literal e8430e8 expression: broadcast + sum, +1 Phipson-Smyth.

    Raises exactly what the base raises (AxisError for a 1D null in the
    multi-param branch, ValueError for incompatible axes).
    """
    null_reshaped = null_dist.swapaxes(0, 1)[None, None, ...]
    count = np.sum(emp_t[..., np.newaxis] <= null_reshaped, axis=-1)
    return (count + 1.0) / (null_dist.shape[0] + 1.0)


@pytest.mark.parametrize("name,observed,null", _engine_p_cases(), ids=lambda x: x)
def test_engine_p_values_match_broadcast_oracle(name, observed, null):
    """Engine counting equals the previous broadcast, tie handling included."""
    result = _compute_p_values_from_null(
        {"positive": observed, "negative": -observed},
        {"positive": null, "negative": null},
    )
    expected, _ = _oracle_broadcast_p(observed, null)
    np.testing.assert_array_equal(result["positive"], expected)


@pytest.mark.parametrize("name,observed,null", _engine_p_cases(), ids=lambda x: x)
def test_engine_p_values_match_broadcast_oracle_b1(name, observed, null):
    """B = 1 nulls (single permutation) keep the +1 correction behaviour."""
    null_b1 = null[:1] if null.ndim == 1 else null[:1, :]
    result = _compute_p_values_from_null({"positive": observed}, {"positive": null_b1})
    expected, _ = _oracle_broadcast_p(observed, null_b1)
    np.testing.assert_array_equal(result["positive"], expected)


def test_engine_all_zero_maps_give_p_one():
    """All-zero observed against an all-zero null counts every null (ties)."""
    null = np.zeros(25)
    observed = np.zeros((5, 5))
    result = _compute_p_values_from_null({"positive": observed}, {"positive": null})
    np.testing.assert_array_equal(result["positive"], np.ones((5, 5)))


def test_engine_count_dtype_is_int64():
    null = np.round(np.random.default_rng(0).normal(size=30), 1)
    observed = np.round(np.random.default_rng(1).normal(size=(5, 5)), 1)
    counts = _empirical_p(null, observed)[1]
    assert counts.dtype == np.int64
    # Memory footprint: no broadcast-sized intermediate survives in the result.
    assert counts.nbytes == observed.size * 8


def _engine_broadcast_grid():
    """Owner repro grid for the multi-param broadcast contract."""
    rg = np.random.default_rng(99)
    emp3 = np.round(rg.normal(size=(2, 2, 3)), 1)
    emp3s = np.round(rg.normal(size=(2, 2, 1)), 1)
    emp1 = np.round(rg.normal(size=(3,)), 1)
    emp0 = np.round(rg.normal(), 1)
    null53 = np.round(rg.normal(size=(5, 3)), 1)
    null51 = np.round(rg.normal(size=(5, 1)), 1)
    null11 = np.round(rg.normal(size=(1, 1)), 1)
    null13 = np.round(rg.normal(size=(1, 3)), 1)
    return [
        ("(2,2,3)x(5,3)", emp3, null53),
        ("(2,2,3)x(5,1)_singleton_null_col", emp3, null51),
        ("(2,2,3)x(1,1)_B1", emp3, null11),
        ("(2,2,3)x(1,3)", emp3, null13),
        ("(2,2,1)x(5,3)_singleton_emp", emp3s, null53),
        ("(2,2,1)x(5,1)", emp3s, null51),
        ("(3,)x(5,3)_fused_axes", emp1, null53),
        ("0Dx(5,3)", emp0, null53),
        ("0Dx(5,1)", emp0, null51),
        ("ties", np.round(rg.normal(size=(2, 2, 3)), 0), np.round(rg.normal(size=(7, 3)), 0)),
        ("random_big", np.round(rg.normal(size=(6, 6, 4)), 1), np.round(rg.normal(size=(11, 4)), 1)),
    ]


def _run_engine_single(emp_t, null_dist):
    return _compute_p_values_from_null(
        {"positive": emp_t, "negative": -emp_t if emp_t.ndim else emp_t},
        {"positive": null_dist, "negative": null_dist},
    )["positive"]


@pytest.mark.parametrize("name,observed,null", _engine_broadcast_grid(), ids=lambda x: x)
def test_engine_multi_param_matches_literal_e8430e8(name, observed, null):
    """Bitwise p-values and shapes versus the literal e8430e8 expression."""
    result = _run_engine_single(observed, null)
    expected = _base_broadcast_oracle(observed, null)
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "emp_key,null_key,expected_error",
    [
        ("emp3", "null5", "AxisError"),
        ("emp3", "null52", "ValueError"),
        ("emp7", "null53", "ValueError"),
    ],
)
def test_engine_multi_param_error_types_match_e8430e8(emp_key, null_key, expected_error):
    """Accept/reject and exception TYPES equal the base expression."""
    rg = np.random.default_rng(101)
    emps = {
        "emp3": np.round(rg.normal(size=(2, 2, 3)), 1),
        "emp7": np.round(rg.normal(size=(7,)), 1),
    }
    nulls = {
        "null5": np.arange(5.0),
        "null52": np.round(rg.normal(size=(5, 2)), 1),
        "null53": np.round(rg.normal(size=(5, 3)), 1),
    }
    emp_t, null_dist = emps[emp_key], nulls[null_key]
    engine_error = base_error = None
    try:
        _run_engine_single(emp_t, null_dist)
    except Exception as exc:  # noqa: BLE001 - the exception TYPE is the assertion
        engine_error = type(exc).__name__
    try:
        _base_broadcast_oracle(emp_t, null_dist)
    except Exception as exc:  # noqa: BLE001
        base_error = type(exc).__name__
    assert engine_error == base_error == expected_error


def test_engine_2d_branch_rejects_2d_null_like_base():
    """2D observed against a (B, P) null: ValueError, as the base 2D branch."""
    rg = np.random.default_rng(103)
    observed = np.round(rg.normal(size=(4, 4)), 1)
    null_2d = np.round(rg.normal(size=(5, 3)), 1)
    engine_error = base_error = None
    try:
        _compute_p_values_from_null({"positive": observed}, {"positive": null_2d})
    except Exception as exc:  # noqa: BLE001
        engine_error = type(exc).__name__
    try:
        count = np.sum(observed[..., np.newaxis] <= null_2d, axis=-1)
        _ = (count + 1.0) / (null_2d.shape[0] + 1.0)
    except Exception as exc:  # noqa: BLE001
        base_error = type(exc).__name__
    assert engine_error == base_error == "ValueError"


def test_engine_nan_statistics_raise_runtime_error():
    null = np.arange(10.0)
    with pytest.raises(RuntimeError, match="NaN statistics in the"):
        _compute_p_values_from_null(
            {"positive": np.full((4, 4), np.nan)}, {"positive": null}
        )
    null_nan = np.arange(10.0)
    null_nan[3] = np.nan
    with pytest.raises(RuntimeError, match="NaN statistics in the"):
        _compute_p_values_from_null({"positive": np.zeros((4, 4))}, {"positive": null_nan})


@pytest.mark.parametrize("name,observed,null", _acceleration_p_cases(), ids=lambda x: x)
def test_acceleration_empirical_p_matches_broadcast_oracle(name, observed, null):
    p, count = _empirical_p(null, observed)
    expected, expected_count = _oracle_broadcast_p(observed, null)
    np.testing.assert_array_equal(p, expected)
    np.testing.assert_array_equal(count, expected_count)
    assert count.dtype == np.int64


def test_acceleration_empirical_p_one_permutation():
    p, count = _empirical_p(np.array([2.0]), np.array([[1.0, 2.0, 3.0]]))
    # null >= observed: 1, 1, 0 (ties included); Phipson-Smyth +1.
    np.testing.assert_array_equal(count, [[1, 1, 0]])
    np.testing.assert_allclose(p, [[2 / 2, 2 / 2, 1 / 2]])


def test_acceleration_empirical_p_rejects_bad_null():
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        _empirical_p(np.array([]), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        _empirical_p(np.zeros((3, 4)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        _empirical_p(np.array([1.0, np.nan]), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        _empirical_p(np.arange(4.0), np.array([[np.inf, 0.0], [0.0, 0.0]]))


# ---------------------------------------------------------------------------
# bh_fdr_perm: bitwise parity with a list-based (collect-all) oracle.
# ---------------------------------------------------------------------------


def _paired_oracle_task(group):
    emp = compute_t_stat_diff(group)
    X, sumsq = _precompute_edge_sums(group)
    return emp, partial(_fast_permutation_task_paired_edges, X, sumsq)


def _two_sample_oracle_task(group1, group2, var_equal, strata):
    emp = compute_t_stat(group1, group2, test_type="two-sample", var_equal=var_equal)
    Xall_3d = np.concatenate((group1, group2), axis=0)
    Xall, Xall2, sum_all, sumsq_all = _precompute_twosample_sums(Xall_3d)
    n1 = group1.shape[0]
    strata_codes = n1_per_stratum = None
    if strata is not None:
        from conninfpy.stats._engine import _encode_strata

        strata_codes = _encode_strata(strata)
        n1_per_stratum = np.bincount(
            strata_codes[:n1], minlength=int(strata_codes.max()) + 1
        )
    task = partial(
        _fast_permutation_task_ind_edges,
        Xall,
        Xall2,
        sum_all,
        sumsq_all,
        n1,
        strata=strata_codes,
        n1_per_stratum=n1_per_stratum,
        var_equal=var_equal,
    )
    return emp, task


def _list_based_bh_fdr_oracle(group1, group2, test_type, two_tailed, B=30, seed=123,
                              var_equal=False, strata=None):
    """The previous implementation: collect B per-seed dicts, fold in the parent."""
    if test_type == "paired":
        emp, task = _paired_oracle_task(group2 - group1)
    elif test_type == "one-sample":
        emp, task = _paired_oracle_task(group1)
    else:
        emp, task = _two_sample_oracle_task(group1, group2, var_equal, strata)
    seeds = np.random.RandomState(seed).randint(0, 2**32 - 1, size=B, dtype=np.int64)
    perm_results = run_permutations(task, list(seeds), use_mp=False)
    emp_upper = {key: emp[key][np.triu_indices(emp[key].shape[0], 1)]
                 for key in ("positive", "negative")}
    n_edges = emp_upper["positive"].shape[0]
    p_values = {}
    if two_tailed:
        observed = emp_upper["positive"] + emp_upper["negative"]
        count = sum(
            (r["positive"] + r["negative"] >= observed).astype(float)
            for r in perm_results
        )
        corrected = _bh_fdr_correction((count + 1.0) / (B + 1.0))
        for key in ("positive", "negative"):
            p_mat = np.ones(emp[key].shape, dtype=np.float64)
            triu = np.triu_indices(emp[key].shape[0], 1)
            directional = np.where(emp[key][triu] > 0, corrected, 1.0)
            p_mat[triu] = directional
            p_mat[(triu[1], triu[0])] = directional
            p_values[key] = p_mat
        return p_values, emp
    for key in ("positive", "negative"):
        count_ge = np.zeros(n_edges, dtype=np.float64)
        for perm_dict in perm_results:
            count_ge += (perm_dict[key] >= emp_upper[key]).astype(np.float64)
        per_edge_p = (count_ge + 1.0) / (B + 1.0)
        corrected_p = _bh_fdr_correction(per_edge_p)
        p_mat = np.ones(emp[key].shape, dtype=np.float64)
        triu = np.triu_indices(emp[key].shape[0], 1)
        p_mat[triu] = corrected_p
        p_mat[(triu[1], triu[0])] = corrected_p
        p_values[key] = p_mat
    return p_values, emp


def _toy_groups():
    rg = np.random.default_rng(7)
    g1 = rg.normal(0, 1, (16, 6, 6))
    g1 = (g1 + g1.transpose(0, 2, 1)) / 2
    g2 = rg.normal(0, 1, (14, 6, 6))
    g2 = (g2 + g2.transpose(0, 2, 1)) / 2
    for group in (g1, g2):
        for matrix in group:
            np.fill_diagonal(matrix, 0)
    g2[:, 0, 1] += 0.8
    g2[:, 1, 0] = g2[:, 0, 1]
    return g1, g2


@pytest.mark.parametrize("two_tailed", [False, True])
@pytest.mark.parametrize(
    "test_type",
    ["one-sample", "paired", "two-sample"],
)
def test_bh_fdr_perm_streaming_matches_list_oracle(test_type, two_tailed):
    """Streaming serial run reproduces the list-based oracle bit for bit."""
    g1, g2 = _toy_groups()
    if test_type == "paired":
        g1, g2 = g1[:12], g2[:12]  # paired designs need equal group sizes
    expected_p, expected_emp = _list_based_bh_fdr_oracle(g1, g2, test_type, two_tailed)
    p_values, emp_t = _compute_bh_fdr_perm_p_values(
        g1, g2, test_type, n_permutations=30, random_state=123,
        use_mp=False, n_processes=None, two_tailed=two_tailed,
    )
    for key in ("positive", "negative"):
        np.testing.assert_array_equal(p_values[key], expected_p[key])
        np.testing.assert_array_equal(emp_t[key], expected_emp[key])


def test_bh_fdr_perm_two_sample_options_match_list_oracle():
    """strata and var_equal variants stay bitwise identical to the oracle."""
    g1, g2 = _toy_groups()
    strata = np.array(["a"] * 15 + ["b"] * 15)  # spans n1 + n2 subjects
    for var_equal in (False, True):
        for two_tailed in (False, True):
            expected_p, _ = _list_based_bh_fdr_oracle(
                g1, g2, "two-sample", two_tailed, var_equal=var_equal, strata=strata
            )
            p_values, _ = _compute_bh_fdr_perm_p_values(
                g1, g2, "two-sample", n_permutations=30, random_state=123,
                use_mp=False, n_processes=None, two_tailed=two_tailed,
                strata=strata, var_equal=var_equal,
            )
            for key in ("positive", "negative"):
                np.testing.assert_array_equal(p_values[key], expected_p[key])


@pytest.mark.parametrize("two_tailed", [False, True])
@pytest.mark.parametrize(
    "test_type",
    ["one-sample", "paired", "two-sample"],
)
@pytest.mark.parametrize("start_method", START_METHODS)
def test_bh_fdr_perm_streaming_mp_matches_serial(test_type, two_tailed, start_method):
    """Serial and 2-worker runs produce identical p-values, per start method."""
    g1, g2 = _toy_groups()
    if test_type == "paired":
        g1, g2 = g1[:12], g2[:12]
    strata = np.array(["a"] * 15 + ["b"] * 15) if test_type == "two-sample" else None
    serial_p, serial_emp = _compute_bh_fdr_perm_p_values(
        g1, g2, test_type, n_permutations=30, random_state=123,
        use_mp=False, n_processes=None, two_tailed=two_tailed, strata=strata,
    )
    ctx = multiprocessing.get_context(start_method)
    with patch.object(_workers.multiprocessing, "get_context", return_value=ctx):
        mp_p, mp_emp = _compute_bh_fdr_perm_p_values(
            g1, g2, test_type, n_permutations=30, random_state=123,
            use_mp=True, n_processes=2, two_tailed=two_tailed, strata=strata,
        )
    for key in ("positive", "negative"):
        np.testing.assert_array_equal(mp_p[key], serial_p[key])
        np.testing.assert_array_equal(mp_emp[key], serial_emp[key])


# ---------------------------------------------------------------------------
# Available-case bh_fdr_perm: streaming path, NaN-free semantics preserved.
# ---------------------------------------------------------------------------


def _missing_groups():
    """g1 loses enough subjects at ROIs 2/3 to cross the coverage threshold."""
    g1, g2 = _toy_groups()
    g1 = g1.copy()
    for subject in range(4):
        g1[subject, 2, :] = g1[subject, :, 2] = np.nan
        g1[subject, 2, 2] = 0.0
    for subject in range(2):
        g1[subject, 3, :] = g1[subject, :, 3] = np.nan
        g1[subject, 3, 3] = 0.0
    return g1, g2


def _available_case_oracle(group1, group2, test_type, two_tailed, B=24, seed=5):
    """Previous available-case path: collect per-seed vectors, fold in parent."""
    d, _ = _build_design(group1, group2, test_type, None, 0.8)
    t_pos, t_neg = _observed(d)
    seeds = np.random.RandomState(seed).randint(0, 2**32 - 1, size=B, dtype=np.int64)
    perm = run_permutations(partial(_task_edges, d), list(seeds), use_mp=False)
    p_values = {}
    for key, t in (("positive", t_pos), ("negative", t_neg)):
        count = np.zeros(t.shape)
        for r in perm:
            count += (
                (r["positive"] + r["negative"] >= t_pos + t_neg)
                if two_tailed
                else (r[key] >= t)
            )
        per_edge = (count + 1.0) / (B + 1.0)
        p = np.full(t.shape, np.nan)
        p[d.tested] = _bh_fdr_correction(per_edge[d.tested])
        if two_tailed:
            p[d.tested & (t == 0)] = 1.0
        mat = np.full((d.N, d.N), np.nan)
        mat[d.triu] = p
        mat[(d.triu[1], d.triu[0])] = p
        np.fill_diagonal(mat, 1.0)
        p_values[key] = mat
    return p_values


@pytest.mark.parametrize("two_tailed", [False, True])
@pytest.mark.parametrize("test_type", ["two-sample", "paired", "one-sample"])
def test_available_case_bh_fdr_perm_matches_list_oracle(test_type, two_tailed):
    """Streaming available-case run equals the collect-list path bit for bit."""
    g1, g2 = _missing_groups()
    if test_type == "paired":
        g1, g2 = g1[:12], g2[:12]
    if test_type == "one-sample":
        g1, g2 = g1, None
    expected = _available_case_oracle(g1, g2, test_type, two_tailed)
    for use_mp, n_processes in ((False, None), (True, 2)):
        p_values, _, report = available_case_p_val(
            g1, g2, test_type, "bh_fdr_perm", None, {},
            n_permutations=24, random_state=5, use_mp=use_mp,
            n_processes=n_processes, verbose=False, strata=None,
            acceleration=None, two_tailed=two_tailed,
        )
        for key in ("positive", "negative"):
            np.testing.assert_array_equal(p_values[key], expected[key])
        # Untested edges stay NaN (their comparisons never counted).
        assert report["n_edges_untested"] > 0


# ---------------------------------------------------------------------------
# Boundedness: instrumented proof of the structural chunk bound.
# ---------------------------------------------------------------------------


def _assert_stats_consistent(stats, n_seeds):
    assert stats["chunks_issued"] == stats["chunks_folded"] == stats["n_chunks"]
    assert stats["chunk_size"] * (stats["n_chunks"] - 1) < n_seeds <= stats["chunk_size"] * stats["n_chunks"]
    limit = stats["issued_outstanding_limit"]
    assert stats["max_issued_outstanding"] <= limit
    assert stats["max_arrived_outstanding"] <= limit
    assert stats["max_arrived_outstanding_bytes"] <= limit * stats["max_chunk_result_bytes"]
    return limit


def test_reduction_bound_mp_slow_first_chunk():
    """2 workers, slow first chunk: fast results buffer, but never past 2*w.

    With chunk_size = 4 the first chunk (seeds 0-3) sleeps in the worker
    while the remaining seven chunks complete almost instantly. The parent
    must neither issue unbounded work nor buffer unbounded results: every
    issued-but-unfolded chunk stays within the structural bound 2 * workers.
    """
    seeds = list(range(32))
    stats_out = {}
    total = run_permutations_reduce(
        slow_first_chunk,
        seeds,
        reduce_func=sum_squares_chunk,
        use_mp=True,
        n_processes=2,
        stats_out=stats_out,
    )
    # sum of k**2 for k in 0..31, exact in float64
    assert total == float(sum(k * k for k in seeds))
    limit = _assert_stats_consistent(stats_out, len(seeds))
    assert stats_out["mode"] == "mp"
    assert stats_out["workers"] == 2
    assert stats_out["issued_outstanding_limit"] == 4  # 2 * workers
    # The slow first chunk really did let fast chunk results pile up
    # out of order — the bound held above an empty buffer.
    assert stats_out["max_arrived_outstanding"] >= 2
    assert stats_out["max_arrived_outstanding_bytes"] > 0


def test_reduction_bound_serial_one_outstanding():
    """Serial path folds chunk-by-chunk: at most one outstanding result."""
    seeds = list(range(24))
    stats_out = {}
    total = run_permutations_reduce(
        slow_first_chunk,
        seeds,
        reduce_func=sum_squares_chunk,
        use_mp=False,
        stats_out=stats_out,
    )
    assert total == float(sum(k * k for k in seeds))
    limit = _assert_stats_consistent(stats_out, len(seeds))
    assert stats_out["mode"] == "serial"
    assert limit == 1
    assert stats_out["max_issued_outstanding"] == 1
    assert stats_out["max_arrived_outstanding"] == 1


@pytest.mark.parametrize("start_method", START_METHODS)
def test_reduction_bound_holds_per_start_method(start_method):
    """The structural bound is independent of the pool start method."""
    ctx = multiprocessing.get_context(start_method)
    with patch.object(_workers.multiprocessing, "get_context", return_value=ctx):
        stats_out = {}
        total = run_permutations_reduce(
            slow_first_chunk,
            list(range(16)),
            reduce_func=sum_squares_chunk,
            use_mp=True,
            n_processes=2,
            stats_out=stats_out,
        )
    assert total == float(sum(k * k for k in range(16)))
    _assert_stats_consistent(stats_out, 16)


def test_reduction_matches_run_permutations_results():
    """Same seeds: the reduced total equals folding the list of results."""
    seeds = list(range(20))
    per_seed = run_permutations(slow_first_chunk, seeds, use_mp=False)
    total = run_permutations_reduce(
        slow_first_chunk, seeds, reduce_func=sum_squares_chunk, use_mp=False
    )
    assert total == float(sum(per_seed))


def test_reduce_empty_seeds_return_none_without_pool():
    with patch.object(_workers.multiprocessing, "get_context") as factory:
        assert run_permutations_reduce(abs, [], reduce_func=sum_squares_chunk) is None
    factory.assert_not_called()
