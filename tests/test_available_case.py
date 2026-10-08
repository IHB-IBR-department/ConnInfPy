"""
Available-case inference: each edge is tested with the subjects who have it.

Calibration was checked separately (2,000 null datasets, 15 + 15 subjects,
30 % of subjects missing one region at random: FWER 0.046 / 0.047 per tail).
These tests pin down the mechanics: identical results on complete data,
per-edge sample sizes, coverage rules, parametric baselines, reporting.
"""

import unittest

import numpy as np
from scipy import stats

from conninfpy import analyze
from conninfpy._available_case import available_case_p_val
from conninfpy.pairwise_stats import _ENHANCE_MAP, StatMethod, compute_p_val
from conninfpy.results import summarize_by_network
from conninfpy.topologies import TopologyDatasetGenerator

N = 12
IU = np.triu_indices(N, 1)


def _data(n=15, seed=1, effect=0.15):
    ds = TopologyDatasetGenerator(n_nodes=N, n_modules=3, seed=seed).generate(
        "within_module_dense", effect_size=effect, n_samples=n, time_points=60)
    g1, g2 = ds.fisher_z()
    return g1, g2, ds.net_labels


def _kill(g, subject, roi):
    g = g.copy()
    g[subject, roi, :] = np.nan
    g[subject, :, roi] = np.nan
    g[subject, roi, roi] = 0.0
    return g


def _roi_edges(roi):
    return (IU[0] == roi) | (IU[1] == roi)


class TestCompleteDataUnchanged(unittest.TestCase):
    """With no missing value the available-case path equals the standard one."""

    def _compare(self, test, method, **kw):
        g1, g2, labels = _data()
        a, b = (g1, g2) if test != "one-sample" else (g2 - g1, None)
        ref = compute_p_val(a, b, test_type=test, method=method, n_permutations=99,
                            rng=3, use_mp=False, **kw)
        mode = (f"parametric_{method}" if method in ("bonferroni", "bh_fdr")
                else "bh_fdr_perm" if method == "bh_fdr_perm" else "max")
        p, _, rep = available_case_p_val(
            a, b, test, mode, _ENHANCE_MAP[StatMethod(method)], dict(kw),
            n_permutations=99, random_state=3, use_mp=False, n_processes=None,
            verbose=False, strata=None, acceleration=None)
        self.assertEqual(rep["n_edges_untested"], 0)
        for tail in ("positive", "negative"):
            np.testing.assert_allclose(p[tail][IU], np.asarray(ref[tail])[IU], atol=1e-12)

    def test_tstat_two_sample(self):
        self._compare("two-sample", "tstat")

    def test_tfnbs_paired(self):
        self._compare("paired", "tfnbs", e=0.4, h=3.0, n=10, start_thres=1.65)

    def test_one_sample(self):
        self._compare("one-sample", "tstat")

    def test_parametric_and_permutation_bh(self):
        self._compare("two-sample", "bonferroni")
        self._compare("paired", "bh_fdr")
        self._compare("two-sample", "bh_fdr_perm")


class TestMissingValues(unittest.TestCase):

    def test_raise_is_the_default(self):
        g1, g2, _ = _data()
        with self.assertRaisesRegex(ValueError, "Regions without signal"):
            compute_p_val(_kill(g1, 0, 2), g2, test_type="two-sample", method="tstat",
                          n_permutations=19, rng=0, use_mp=False)

    def test_edges_of_a_missing_region_use_the_remaining_subjects(self):
        g1, g2, _ = _data()
        r = compute_p_val(_kill(g1, 0, 2), g2, test_type="two-sample", method="tstat",
                          n_permutations=49, rng=0, use_mp=False, missing="available_case")
        rep = r.missing_report
        n = rep["edge_n"][IU]
        self.assertTrue(np.all(n[_roi_edges(2)] == 29))
        self.assertTrue(np.all(n[~_roi_edges(2)] == 30))
        self.assertEqual(rep["regions_without_signal"], {2: 1})
        self.assertEqual(rep["subjects_with_missing"], {"group1": [0], "group2": []})
        self.assertEqual(rep["n_edges_untested"], 0)
        self.assertTrue(np.all(np.isfinite(np.asarray(r["positive"])[IU])))

    def test_coverage_rule_leaves_poorly_covered_edges_untested(self):
        g1, g2, _ = _data()
        for s in range(4):                      # 11 of 15 = 73 % < 80 %
            g1 = _kill(g1, s, 5)
        g1 = _kill(g1, 7, 8)                    # 14 of 15 for ROI 8: tested
        r = compute_p_val(g1, g2, test_type="two-sample", method="tfnbs", n_permutations=49,
                          rng=0, use_mp=False, missing="available_case")
        p = np.asarray(r["positive"])[IU]
        self.assertTrue(np.all(np.isnan(p[_roi_edges(5)])))
        self.assertTrue(np.all(np.isfinite(p[~_roi_edges(5)])))
        self.assertEqual(r.missing_report["n_edges_untested"], N - 1)
        # a looser rule (70 %: 11 of 15) tests them, except edge 5-8, which
        # loses the subjects missing ROI 5 and the one missing ROI 8 (10 of 15)
        r = compute_p_val(g1, g2, test_type="two-sample", method="tstat", n_permutations=19,
                          rng=0, use_mp=False, missing="available_case", min_coverage=0.7)
        self.assertEqual(r.missing_report["untested_edges"], [(5, 8)])

    def test_paired_uses_complete_pairs(self):
        g1, g2, _ = _data()
        r = compute_p_val(g1, _kill(g2, 4, 1), test_type="paired", method="tstat",
                          n_permutations=19, rng=0, use_mp=False, missing="available_case")
        n = r.missing_report["edge_n"][IU]
        self.assertTrue(np.all(n[_roi_edges(1)] == 14))
        self.assertTrue(np.all(n[~_roi_edges(1)] == 15))

    def test_parametric_welch_on_available_subjects(self):
        g1, g2, _ = _data()
        g1 = _kill(g1, 0, 2)
        r = compute_p_val(g1, g2, test_type="two-sample", method="bonferroni",
                          missing="available_case")
        i, j = 2, 7
        a, b = g1[:, i, j], g2[:, i, j]
        ref = stats.ttest_ind(b, a[np.isfinite(a)], equal_var=False)
        t = r.stat_positive[i, j] - r.stat_negative[i, j]
        self.assertAlmostEqual(t, ref.statistic, places=10)
        one_tailed = ref.pvalue / 2 if ref.statistic > 0 else 1 - ref.pvalue / 2
        self.assertAlmostEqual(np.asarray(r["positive"])[i, j], min(one_tailed * len(IU[0]), 1.0), places=10)

    def test_infinite_values_are_still_rejected(self):
        g1, g2, _ = _data()
        g1[0, 1, 2] = g1[0, 2, 1] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            compute_p_val(g1, g2, test_type="two-sample", method="tstat", n_permutations=19,
                          rng=0, use_mp=False, missing="available_case")

    def test_strata_with_missing_values(self):
        g1, g2, _ = _data()
        r = compute_p_val(_kill(g1, 0, 2), g2, test_type="two-sample", method="tstat",
                          n_permutations=49, rng=0, use_mp=False, missing="available_case",
                          strata=np.repeat([0, 1, 2], 10))
        self.assertTrue(r.strata_provided)
        self.assertTrue(np.all(np.isfinite(np.asarray(r["positive"])[IU])))


class TestAnalyzeAndSummaries(unittest.TestCase):

    def test_analyze_reports_available_case_in_flags(self):
        g1, g2, _ = _data()
        out = analyze(group1=_kill(g1, 0, 2), group2=g2, input="z", test_type="two-sample",
                      method="tstat", n_permutations=19, rng=0, use_mp=False,
                      missing="available_case")
        self.assertTrue(any("available-case analysis" in f and "ROI 2 in 1" in f for f in out.flags))

    def test_glm_with_missing_values_is_not_supported_yet(self):
        g1, _, _ = _data()
        with self.assertRaisesRegex(NotImplementedError, "GLM designs"):
            analyze(_kill(g1, 0, 2), interest=np.linspace(0, 1, 15), input="z", method="tstat",
                    n_permutations=19, rng=0, use_mp=False, missing="available_case")

    def test_network_summary_counts_tested_edges_only(self):
        g1, g2, labels = _data()
        for s in range(4):
            g1 = _kill(g1, s, 0)                # ROI 0 untested
        r = compute_p_val(g1, g2, test_type="two-sample", method="tstat", n_permutations=19,
                          rng=0, use_mp=False, missing="available_case")
        summary = summarize_by_network(r, list(map(str, labels)))
        self.assertEqual(int(summary["n_edges"].sum()), len(IU[0]) - (N - 1))


if __name__ == "__main__":
    unittest.main()
