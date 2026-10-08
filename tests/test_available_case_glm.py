"""
Available-case GLM (Freedman-Lane per missingness pattern) and ComBat with
missing values.

Calibration was checked separately (1,000 null datasets, 36 subjects at 3
sites, ComBat + site dummies, within-site permutation, 30 % of subjects
missing one region): FWER 0.051-0.063, matching the same pipeline on the
complete data (0.054-0.059) — the small excess is Freedman-Lane's
approximation in this small design, not the missing-data handling.
"""

import unittest
import warnings

import numpy as np

from conninfpy import analyze
from conninfpy._available_case_glm import available_case_glm
from conninfpy.glm_stats import (
    _ENHANCE_MAP,
    _compute_reduced_model,
    build_design_matrix,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
)
from conninfpy.harmonize import _combat_fit_available, combat_fit, combat_harmonize
from conninfpy.pairwise_stats import StatMethod

N, n = 10, 30
IU = np.triu_indices(N, 1)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, N, N))
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    for s in Y:
        np.fill_diagonal(s, 0)
    age = rng.normal(size=n)
    sex = rng.integers(0, 2, n).astype(float)
    sites = np.repeat(["a", "b", "c"], 10)
    Y[:, 0, 1] += 0.9 * age
    Y[:, 1, 0] = Y[:, 0, 1]
    return Y, age, sex, sites


def _kill(Y, subject, roi):
    Y = Y.copy()
    Y[subject, roi, :] = np.nan
    Y[subject, :, roi] = np.nan
    Y[subject, roi, roi] = 0.0
    return Y


def _roi_edges(roi):
    return (IU[0] == roi) | (IU[1] == roi)


class TestGLMCompleteDataUnchanged(unittest.TestCase):

    def _compare(self, method, stat_type="tstat", strata=None, contrast=None, **kw):
        Y, age, sex, sites = _data()
        X, c = build_design_matrix(age, sex[:, None])
        c = c if contrast is None else contrast
        ref = compute_p_val_glm(Y, design_matrix=X, contrast=c, method=method, stat_type=stat_type,
                                n_permutations=49, rng=2, use_mp=False, strata=strata, **kw)
        p, _, _ = available_case_glm(
            Y, X, c, _compute_reduced_model(X, c), stat_type, _ENHANCE_MAP[StatMethod(method)], dict(kw),
            n_permutations=49, random_state=2, use_mp=False, n_processes=None, verbose=False,
            strata=strata, acceleration=None, two_tailed=False)
        for key in p:
            np.testing.assert_allclose(p[key][IU], np.asarray(ref[key])[IU], atol=1e-12)

    def test_tstat_with_and_without_strata(self):
        self._compare("tstat")
        self._compare("tstat", strata=np.repeat([0, 1, 2], 10))

    def test_tfnbs(self):
        self._compare("tfnbs", e=0.4, h=3.0, n=10, start_thres=1.65)

    def test_f_contrast(self):
        C = np.zeros((2, 3))
        C[0, 2] = C[1, 1] = 1.0
        self._compare("tstat", stat_type="fstat", contrast=C)


class TestGLMMissingValues(unittest.TestCase):

    def test_raise_by_default_and_available_case_on_request(self):
        Y, age, sex, _ = _data()
        Y = _kill(Y, 3, 4)
        with self.assertRaisesRegex(ValueError, "Regions without signal"):
            compute_p_val_glm(Y, interest=age, confounds=sex, method="tstat",
                              n_permutations=19, use_mp=False, rng=0)
        r = compute_p_val_glm(Y, interest=age, confounds=sex, method="tfnbs", n_permutations=49,
                              use_mp=False, rng=0, missing="available_case")
        rep = r.missing_report
        n_edge = rep["edge_n"][IU]
        self.assertTrue(np.all(n_edge[_roi_edges(4)] == n - 1))
        self.assertTrue(np.all(n_edge[~_roi_edges(4)] == n))
        self.assertEqual(rep["regions_without_signal"], {4: 1})
        self.assertEqual(rep["n_edges_untested"], 0)
        self.assertTrue(np.all(np.isfinite(np.asarray(r["positive"])[IU])))
        self.assertIsNotNone(rep["missing_by_interest_p"])

    def test_site_without_the_region_is_dropped_from_the_design(self):
        # All of site "c" lacks ROI 2: its dummy column vanishes on those edges,
        # but the age contrast stays estimable.
        Y, age, sex, sites = _data()
        for s in range(20, 30):
            Y = _kill(Y, s, 2)
        X, c = build_design_matrix(age, np.column_stack([sex, sites == "b", sites == "c"]).astype(float))
        r = compute_p_val_glm(Y, design_matrix=X, contrast=c, method="tstat", n_permutations=19,
                              use_mp=False, rng=0, missing="available_case", min_coverage=0.6)
        self.assertEqual(r.missing_report["n_edges_untested"], 0)

    def test_non_estimable_contrast_leaves_edges_untested(self):
        # The tested variable is a group label and every group-1 subject lacks ROI 5.
        Y, _, _, _ = _data()
        group = np.repeat([0.0, 1.0], 15)
        for s in range(15, 30):
            Y = _kill(Y, s, 5)
        r = compute_p_val_glm(Y, interest=group, method="tstat", n_permutations=19, use_mp=False,
                              rng=0, missing="available_case", min_coverage=0.4)
        rep = r.missing_report
        self.assertEqual(rep["untested_reasons"]["not_estimable"], N - 1)
        self.assertTrue(np.all(np.isnan(np.asarray(r["positive"])[IU][_roi_edges(5)])))

    def test_coverage_rule(self):
        Y, age, _, _ = _data()
        for s in range(7):                       # 23 of 30 < 80 %
            Y = _kill(Y, s, 6)
        r = compute_p_val_glm(Y, interest=age, method="tstat", n_permutations=19, use_mp=False,
                              rng=0, missing="available_case")
        self.assertEqual(r.missing_report["untested_reasons"]["coverage"], N - 1)

    def test_paired_glm_and_multi_contrast(self):
        Y, age, sex, _ = _data()
        Ya, Yb = Y[:15], _kill(Y[15:], 2, 3)
        r = compute_p_val_paired_glm(Ya, Yb, confounds_A=sex[:15], confounds_B=sex[15:],
                                     method="tstat", n_permutations=19, use_mp=False, rng=0,
                                     missing="available_case")
        self.assertIsNotNone(r.missing_report)
        X = np.column_stack([np.ones(n), sex, age])
        out = compute_p_val_glm_multi(_kill(Y, 0, 1), design_matrix=X,
                                      contrasts={"sex": np.array([0, 1.0, 0]), "age": np.array([0, 0, 1.0])},
                                      method="tstat", n_permutations=19, use_mp=False, rng=0,
                                      missing="available_case")
        self.assertEqual(set(out), {"sex", "age"})
        self.assertTrue(all(res.missing_report is not None for res in out.values()))


class TestComBatMissingValues(unittest.TestCase):

    def _features(self, seed=0):
        rng = np.random.default_rng(seed)
        sites = np.repeat(["a", "b", "c"], 12)
        F = np.array([rng.normal({"a": 0, "b": .5, "c": -.3}[s], {"a": 1, "b": 1.6, "c": .7}[s], 80)
                      for s in sites])
        return F, sites, rng.normal(size=(36, 1))

    def test_complete_data_matches_standard_combat(self):
        F, sites, cov = self._features()
        for eb in (False, True):
            ref, new = combat_fit(F, sites, preserve=cov, eb=eb), _combat_fit_available(F, sites, cov, eb)
            for k in ("alpha", "beta", "sigma", "gamma_star", "delta_star"):
                np.testing.assert_allclose(getattr(new, k), getattr(ref, k), atol=1e-12, err_msg=k)

    def test_missing_values_stay_missing_and_present_ones_are_harmonized(self):
        F, sites, cov = self._features()
        G = F.copy()
        G[0, :5] = np.nan
        with self.assertRaisesRegex(ValueError, "ComBat input Y has missing"):
            combat_harmonize(G, sites, preserve=cov)
        out = combat_harmonize(G, sites, preserve=cov, missing="available_case")
        self.assertTrue(np.all(np.isnan(out.Y_adjusted[0, :5])))
        self.assertTrue(np.all(np.isfinite(out.Y_adjusted[1:])))
        full = combat_harmonize(F, sites, preserve=cov).Y_adjusted
        self.assertLess(np.median(np.abs(out.Y_adjusted[1:] - full[1:])), 0.05)

    def test_site_with_fewer_than_two_values_is_not_harmonized(self):
        F, sites, cov = self._features()
        G = F.copy()
        G[24:35, 7] = np.nan                     # site "c" keeps one value of feature 7
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = combat_harmonize(G, sites, preserve=cov, missing="available_case")
        self.assertTrue(np.isnan(out.Y_adjusted[35, 7]))
        self.assertEqual(out.diagnostics["values_not_harmonized"], 1)


class TestAnalyzeGLMWithMissingValues(unittest.TestCase):

    def test_combat_site_dummies_glm(self):
        Y, age, sex, sites = _data()
        out = analyze(_kill(Y, 4, 3), interest=age, confounds=sex[:, None], sites=sites, input="z",
                      harmonize="combat_site_dummies_glm", method="tstat", n_permutations=19,
                      rng=0, use_mp=False, missing="available_case")
        self.assertTrue(any("available-case analysis" in f for f in out.flags))
        self.assertEqual(out.inference.missing_report["regions_without_signal"], {3: 1})


if __name__ == "__main__":
    unittest.main()
