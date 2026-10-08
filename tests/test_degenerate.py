"""
Degenerate tests: zero residual variance, non-estimable contrasts, pseudo-pairs.

* zero effect and zero variance (an edge that is 0 in every subject): t = 0;
* nonzero effect and zero variance (an edge that is 0.3 in every subject): the
  statistic is infinite and the entry points reject it, instead of returning
  t ~ 1e16 from rounding noise;
* a GLM contrast must be estimable (C = C X⁺X);
* paired data must have the same subjects in both conditions.
"""

import unittest
import warnings

import numpy as np
from scipy import stats

from conninfpy import compute_p_val
from conninfpy.glm_stats import (
    compute_glm_stat,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    contrast_is_estimable,
)
from conninfpy.pairwise_stats import compute_t_stat, compute_t_stat_diff

N, n = 6, 20
INFINITE = "zero residual variance but a nonzero effect"


def _data(seed=0, size=n):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(size, N, N))
    d = (d + d.transpose(0, 2, 1)) / 2
    for s in d:
        np.fill_diagonal(s, 0)
    return d


def _set(d, i, j, value):
    d = d.copy()
    d[:, i, j] = d[:, j, i] = value
    return d


class TestZeroEffectZeroVariance(unittest.TestCase):
    """An edge that is 0 in every subject: no evidence, t = 0, p = 1."""

    def test_pairwise_and_glm(self):
        d = _set(_data(), 2, 3, 0.0)
        for method in ("tstat", "tfnbs", "bonferroni"):
            r = compute_p_val(d, test_type="one-sample", method=method, n_permutations=49,
                              rng=0, use_mp=False)
            self.assertEqual(r.stat_positive[2, 3], 0.0)
            self.assertEqual(np.asarray(r["positive"])[2, 3], 1.0)
        r = compute_p_val_glm(d, design_matrix=np.ones((n, 1)), contrast=np.array([1.0]),
                              method="tstat", n_permutations=49, rng=0, use_mp=False)
        self.assertEqual(r.stat_positive[2, 3], 0.0)

    def test_equal_constants_in_two_groups(self):
        d = _set(_data(), 0, 1, 0.3)
        r = compute_p_val(d[:10], d[10:], test_type="two-sample", method="tstat",
                          n_permutations=19, rng=0, use_mp=False)
        self.assertEqual(r.stat_positive[0, 1], 0.0)
        self.assertEqual(r.stat_negative[0, 1], 0.0)


class TestNonzeroEffectZeroVariance(unittest.TestCase):
    """An edge that is 0.3 in every subject: infinite t, rejected."""

    def test_primitives_return_zero_not_rounding_noise(self):
        d = _set(_data(), 0, 1, 0.3)
        self.assertEqual(compute_t_stat_diff(d)["positive"][0, 1], 0.0)
        g2 = _set(d[10:], 0, 1, 0.5)
        self.assertEqual(compute_t_stat(d[:10], g2, test_type="two-sample")["positive"][0, 1], 0.0)
        t = compute_glm_stat(d, np.ones((n, 1)), np.array([1.0]), stat_type="tstat")
        self.assertEqual(t["positive"][0, 1], 0.0)

    def test_pairwise_entry_points_reject(self):
        d = _set(_data(), 0, 1, 0.3)
        for method in ("tstat", "tfnbs", "bonferroni", "bh_fdr_perm"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, INFINITE):
                compute_p_val(d, test_type="one-sample", method=method, n_permutations=9,
                              rng=0, use_mp=False)
        base = _data()
        with self.assertRaisesRegex(ValueError, INFINITE):   # every subject shifted by 0.3
            compute_p_val(base, _set(base, 0, 1, base[:, 0, 1] + 0.3),
                          test_type="paired", method="tstat", n_permutations=9, use_mp=False)
        g1, g2 = _set(base[:10], 0, 1, 0.3), _set(base[10:], 0, 1, 0.5)
        with self.assertRaisesRegex(ValueError, r"\(0, 1\)"):
            compute_p_val(g1, g2, test_type="two-sample", method="tstat", n_permutations=9,
                          use_mp=False)

    def test_glm_entry_points_reject(self):
        d = _set(_data(), 0, 1, 0.3)
        X = np.column_stack([np.ones(n), np.random.default_rng(1).normal(size=n)])
        with self.assertRaisesRegex(ValueError, INFINITE):
            compute_p_val_glm(d, design_matrix=np.ones((n, 1)), contrast=np.array([1.0]),
                              method="tstat", n_permutations=9, use_mp=False)
        with self.assertRaisesRegex(ValueError, "F statistic is infinite"):
            compute_p_val_glm(d, design_matrix=X, contrast=np.eye(2), stat_type="fstat",
                              method="tstat", n_permutations=9, use_mp=False)
        with self.assertRaisesRegex(ValueError, INFINITE):
            compute_p_val_glm_multi(d, design_matrix=X,
                                    contrasts={"mean": np.array([1.0, 0]), "x": np.array([0, 1.0])},
                                    method="tstat", n_permutations=9, use_mp=False)
        # an exact fit through the intercept is not infinite for the slope:
        # zero effect, zero variance
        r = compute_p_val_glm(d, design_matrix=X, contrast=np.array([0, 1.0]), method="tstat",
                              n_permutations=9, use_mp=False)
        self.assertEqual(r.stat_positive[0, 1] + r.stat_negative[0, 1], 0.0)

    def test_available_case_paths_reject_and_nan_leaves_the_edge_untested(self):
        d = _set(_data(), 0, 1, 0.3)
        d[0, 2, :] = d[0, :, 2] = np.nan
        d[0, 2, 2] = 0.0
        with self.assertRaisesRegex(ValueError, INFINITE):
            compute_p_val(d, test_type="one-sample", method="tstat", n_permutations=9,
                          use_mp=False, missing="available_case")
        with self.assertRaisesRegex(ValueError, INFINITE):
            compute_p_val_glm(d, design_matrix=np.ones((n, 1)), contrast=np.array([1.0]),
                              method="tstat", n_permutations=9, use_mp=False,
                              missing="available_case")
        d = _set(d, 0, 1, np.nan)
        r = compute_p_val(d, test_type="one-sample", method="tstat", n_permutations=9,
                          use_mp=False, missing="available_case")
        self.assertIn((0, 1), r.missing_report["untested_edges"])

    def test_small_but_real_variance_is_tested(self):
        rng = np.random.default_rng(3)
        d = _set(_data(), 0, 1, 0.3 + 1e-4 * rng.normal(size=n))
        r = compute_p_val(d, test_type="one-sample", method="bonferroni")
        ref = stats.ttest_1samp(d[:, 0, 1], 0.0).statistic
        self.assertAlmostEqual(r.stat_positive[0, 1], ref, places=6)


class TestEstimability(unittest.TestCase):

    def setUp(self):
        self.d = _data()
        self.x = np.random.default_rng(2).normal(size=n)
        self.X = np.column_stack([np.ones(n), self.x, self.x])     # duplicated column

    def test_coefficient_of_a_duplicated_column_is_rejected(self):
        self.assertFalse(contrast_is_estimable(np.array([0, 1.0, 0]), self.X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "not estimable"):
                compute_p_val_glm(self.d, design_matrix=self.X, contrast=np.array([0, 1.0, 0]),
                                  method="tstat", n_permutations=9, use_mp=False)
            with self.assertRaisesRegex(ValueError, "Contrast 'x' is not estimable"):
                compute_p_val_glm_multi(self.d, design_matrix=self.X,
                                        contrasts={"x": np.array([0, 1.0, 0])},
                                        method="tstat", n_permutations=9, use_mp=False)
            with self.assertRaisesRegex(ValueError, "not estimable"):
                bad = self.d.copy()
                bad[0, 1, :] = bad[0, :, 1] = np.nan
                bad[0, 1, 1] = 0.0
                compute_p_val_glm(bad, design_matrix=self.X, contrast=np.array([0, 1.0, 0]),
                                  method="tstat", n_permutations=9, use_mp=False,
                                  missing="available_case")

    def test_identified_combination_matches_the_full_rank_design(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deficient = compute_glm_stat(self.d, self.X, np.array([0, 1.0, 1.0]), stat_type="tstat")
            r = compute_p_val_glm(self.d, design_matrix=self.X, contrast=np.array([0, 1.0, 1.0]),
                                  method="tstat", n_permutations=9, use_mp=False)
        full = compute_glm_stat(self.d, self.X[:, :2], np.array([0, 1.0]), stat_type="tstat")
        for key in ("positive", "negative"):
            np.testing.assert_allclose(deficient[key], full[key], atol=1e-10)
        self.assertTrue(np.all(np.isfinite(np.asarray(r["positive"]))))


class TestPairedSubjectCounts(unittest.TestCase):

    def test_unequal_counts_are_rejected_not_broadcast(self):
        d = _data()
        for other in (d[:1], d[:19]):
            with self.assertRaisesRegex(ValueError, "same subjects in both conditions"):
                compute_p_val(d, other, test_type="paired", method="bonferroni")
            with self.assertRaisesRegex(ValueError, "same subjects in both conditions"):
                compute_t_stat(d, other, test_type="paired")


if __name__ == "__main__":
    unittest.main()
