"""Two-sample permutation test: Welch's t (default) or Student's t (``var_equal``).

The choice changes only the statistic (the denominator of t); the observed
map, every permutation, the parametric baselines and the available-case path
must all use the same one, and the group-label permutation is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from conninfpy import analyze, compute_p_val, compute_t_stat
from conninfpy.stats.pairwise import (
    _fast_permutation_task_ind,
    _precompute_twosample_sums,
)

N = 6
IU = np.triu_indices(N, 1)


def _groups(n1=12, n2=18, seed=0):
    """Unbalanced groups with unequal variances, so Welch != Student."""
    rng = np.random.default_rng(seed)

    def stack(n, sd, shift):
        Y = sd * rng.standard_normal((n, N, N)) + shift
        Y = (Y + Y.transpose(0, 2, 1)) / 2
        Y[:, np.arange(N), np.arange(N)] = 0.0
        return Y

    return stack(n1, 1.0, 0.0), stack(n2, 0.4, 0.3)


def _signed(t):
    return (t["positive"] - t["negative"])[IU]


@pytest.mark.parametrize("var_equal", [False, True])
def test_statistic_matches_scipy(var_equal):
    g1, g2 = _groups()
    t = compute_t_stat(g1, g2, test_type="two-sample", var_equal=var_equal)
    ref = stats.ttest_ind(g2[:, IU[0], IU[1]], g1[:, IU[0], IU[1]], equal_var=var_equal).statistic
    np.testing.assert_allclose(_signed(t), ref, atol=1e-12)


def test_equal_group_sizes_give_the_same_statistic():
    g1, g2 = _groups(n1=15, n2=15)
    welch = compute_t_stat(g1, g2, test_type="two-sample")
    student = compute_t_stat(g1, g2, test_type="two-sample", var_equal=True)
    np.testing.assert_allclose(_signed(welch), _signed(student), atol=1e-12)


@pytest.mark.parametrize("var_equal", [False, True])
def test_permutation_null_uses_the_same_statistic(var_equal):
    """One permutation of the fast path equals scipy's t on the same relabelling."""
    g1, g2 = _groups()
    n1 = g1.shape[0]
    Xall, Xall2, s, q = _precompute_twosample_sums(np.concatenate((g1, g2)))
    seed = 12345
    out = _fast_permutation_task_ind(Xall, Xall2, s, q, n1, seed, var_equal=var_equal)
    # Re-draw the same relabelling as the task does.
    rng = np.random.RandomState(seed)
    g = np.zeros(Xall.shape[0], dtype=bool)
    g[rng.choice(Xall.shape[0], n1, replace=False)] = True
    ref = stats.ttest_ind(Xall[~g], Xall[g], equal_var=var_equal).statistic
    assert out["positive"] == pytest.approx(max(ref.max(), 0.0), abs=1e-10)
    assert out["negative"] == pytest.approx(max((-ref).max(), 0.0), abs=1e-10)


@pytest.mark.parametrize("method", ["tstat", "nbs", "tfnbs", "bh_fdr_perm"])
def test_permutation_methods_record_and_use_student(method):
    g1, g2 = _groups()
    kw = dict(test_type="two-sample", method=method, n_permutations=50, rng=1,
              use_mp=False, threshold=1.0)
    if method != "nbs":
        kw.pop("threshold")
    student = compute_p_val(g1, g2, var_equal=True, **kw)
    welch = compute_p_val(g1, g2, **kw)
    ref = stats.ttest_ind(g2[:, IU[0], IU[1]], g1[:, IU[0], IU[1]], equal_var=True).statistic
    np.testing.assert_allclose(student.stat_signed[IU], ref, atol=1e-12)
    assert student.design["t_statistic"] == "student"
    assert welch.design["t_statistic"] == "welch"
    assert not np.allclose(student.stat_signed, welch.stat_signed)


@pytest.mark.parametrize("var_equal", [False, True])
def test_parametric_bonferroni_degrees_of_freedom(var_equal):
    g1, g2 = _groups()
    res = compute_p_val(g1, g2, test_type="two-sample", method="bonferroni",
                        var_equal=var_equal)
    a, b = g2[:, IU[0], IU[1]], g1[:, IU[0], IU[1]]
    ref = stats.ttest_ind(a, b, equal_var=var_equal, alternative="greater").pvalue
    np.testing.assert_allclose(res.positive[IU], np.minimum(ref * IU[0].size, 1.0), rtol=1e-8)


def test_available_case_uses_student_on_the_available_subjects():
    g1, g2 = _groups()
    g1 = g1.copy()
    g1[0, 0, 1] = g1[0, 1, 0] = np.nan          # one subject misses edge (0, 1)
    res = compute_p_val(g1, g2, test_type="two-sample", method="tstat", n_permutations=20,
                        rng=1, use_mp=False, missing="available_case", min_coverage=0.5,
                        var_equal=True)
    ref = stats.ttest_ind(g2[:, 0, 1], g1[1:, 0, 1], equal_var=True).statistic
    assert res.stat_signed[0, 1] == pytest.approx(ref, abs=1e-10)
    bonf = compute_p_val(g1, g2, test_type="two-sample", method="bonferroni",
                         missing="available_case", min_coverage=0.5, var_equal=True)
    p = stats.ttest_ind(g2[:, 0, 1], g1[1:, 0, 1], equal_var=True, alternative="greater").pvalue
    assert bonf.positive[0, 1] == pytest.approx(min(p * IU[0].size, 1.0), rel=1e-8)


def test_var_equal_is_rejected_outside_the_two_sample_test():
    g1, g2 = _groups(n1=12, n2=12)
    with pytest.raises(ValueError, match="two-sample"):
        compute_p_val(g1, g2, test_type="paired", method="tstat", n_permutations=10,
                      use_mp=False, var_equal=True)
    x = np.random.default_rng(0).standard_normal(24)
    with pytest.raises(ValueError, match="two-sample"):
        analyze(np.concatenate((g1, g2)), interest=x, input="z", method="tstat",
                n_permutations=10, use_mp=False, var_equal=True)


def test_analyze_passes_var_equal_to_the_two_sample_test():
    g1, g2 = _groups()
    res = analyze(group1=g1, group2=g2, test_type="two-sample", input="z",
                  method="tstat", n_permutations=20, rng=1, use_mp=False, var_equal=True)
    assert res.inference.design["t_statistic"] == "student"
    ref = stats.ttest_ind(g2[:, IU[0], IU[1]], g1[:, IU[0], IU[1]], equal_var=True).statistic
    np.testing.assert_allclose(res.inference.stat_signed[IU], ref, atol=1e-12)
