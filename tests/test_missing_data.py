"""
Missing (NaN) and infinite values must never reach the permutation step.

Regression: ``x >= NaN`` is always False, so one NaN in one subject made
every permutation maximum NaN and gave every edge the smallest possible
p-value (all edges "significant"). Loaders also replaced undefined
correlations by 0, passing them off as measured connectivity.
"""

import unittest

import numpy as np
import pandas as pd

from conninfpy import analyze
from conninfpy._engine import _compute_p_values_from_null
from conninfpy._prepare import check_connectivity, nonfinite_report
from conninfpy.acceleration import compute_p_values_accelerated
from conninfpy.glm_stats import compute_p_val_glm
from conninfpy.harmonize import combat_harmonize
from conninfpy.loaders.base import LoadedDataset
from conninfpy.loaders.manifest import drop_subjects_with_missing


def _stack(n=12, N=6, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, N, N))
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    for s in Y:
        np.fill_diagonal(s, 0.0)
    return Y


def _dead_roi(Y, subject, roi):
    Y = Y.copy()
    Y[subject, roi, :] = np.nan
    Y[subject, :, roi] = np.nan
    return Y


class TestCheckConnectivity(unittest.TestCase):

    def test_rejects_nan_and_inf_off_diagonal(self):
        for bad in (np.nan, np.inf, -np.inf):
            Y = _stack()
            Y[3, 0, 1] = Y[3, 1, 0] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"subject\(s\): 3"):
                check_connectivity(Y)

    def test_nan_diagonal_is_accepted_and_zeroed(self):
        Y = _stack()
        Y[:, 2, 2] = np.nan
        out = check_connectivity(Y)
        self.assertTrue(np.all(out[:, 2, 2] == 0))

    def test_message_names_regions_without_signal(self):
        with self.assertRaisesRegex(ValueError, r"Regions without signal \(ROI: number of subjects\): 4: 1"):
            check_connectivity(_dead_roi(_stack(), subject=5, roi=4))


class TestNonfiniteReport(unittest.TestCase):

    def test_separates_dead_regions_from_scattered_edges(self):
        Y = _dead_roi(_stack(), subject=2, roi=3)
        Y[1, 0, 5] = np.inf
        report = nonfinite_report(Y)
        self.assertEqual(report["subjects"], [1, 2])
        self.assertEqual(report["rois"], {3: 1})
        self.assertEqual(report["other_values"], 1)
        self.assertEqual(report["n_values"], 2 * 5 + 1)


class TestEntryPointsRejectMissingValues(unittest.TestCase):

    def test_analyze_two_sample_raises_instead_of_all_significant(self):
        Y = _stack(n=20)
        g2 = Y[10:].copy()
        g2[3, 0, 1] = g2[3, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            analyze(group1=Y[:10], group2=g2, input="z", test_type="two-sample",
                    method="tstat", n_permutations=19, rng=0, use_mp=False)

    def test_glm_rejects_missing_connectivity_and_design(self):
        Y = _stack(n=12)
        x = np.linspace(-1, 1, 12)
        with self.assertRaisesRegex(ValueError, "missing"):
            compute_p_val_glm(_dead_roi(Y, 0, 1), interest=x, method="tstat",
                              n_permutations=9, use_mp=False, rng=0)
        conf = np.ones(12)
        conf[4] = np.nan
        with self.assertRaisesRegex(ValueError, r"confounds has missing .* row\(s\) 4"):
            compute_p_val_glm(Y, interest=x, confounds=conf, method="tstat",
                              n_permutations=9, use_mp=False, rng=0)
        x_bad = x.copy()
        x_bad[2] = np.inf
        with self.assertRaisesRegex(ValueError, "interest has missing"):
            compute_p_val_glm(Y, interest=x_bad, method="tstat", n_permutations=9, use_mp=False, rng=0)

    def test_combat_rejects_missing_values(self):
        Y = _stack(n=12)
        sites = np.repeat(["a", "b"], 6)
        with self.assertRaisesRegex(ValueError, "ComBat input Y has missing"):
            combat_harmonize(_dead_roi(Y, 0, 1), sites)
        preserve = np.linspace(0, 1, 12)
        preserve[0] = np.nan
        with self.assertRaisesRegex(ValueError, "preserve has missing"):
            combat_harmonize(Y, sites, preserve=preserve)


class TestPValueSafetyNet(unittest.TestCase):

    def test_nan_null_maxima_raise(self):
        emp = {"positive": np.ones((4, 4))}
        null = {"positive": np.array([1.0, np.nan, 2.0])}
        with self.assertRaisesRegex(RuntimeError, "NaN statistics"):
            _compute_p_values_from_null(emp, null)
        with self.assertRaisesRegex(RuntimeError, "NaN statistics"):
            compute_p_values_accelerated(emp, null, method="gpd")


class TestDropSubjectsWithMissing(unittest.TestCase):

    def _dataset(self, condition_column=None, subject_ids=None):
        Y = _dead_roi(_stack(n=6), subject=2, roi=1)
        pheno = pd.DataFrame({"x": range(6)})
        return LoadedDataset(data=Y, pheno=pheno, data_kind="fisher_z",
                             subject_ids=subject_ids or [f"s{i}" for i in range(6)],
                             condition_column=condition_column)

    def test_drops_affected_subjects_and_reports_them(self):
        ds = drop_subjects_with_missing(self._dataset())
        self.assertEqual(ds.data.shape[0], 5)
        self.assertEqual(list(ds.pheno["x"]), [0, 1, 3, 4, 5])
        self.assertEqual(ds.subject_ids, ["s0", "s1", "s3", "s4", "s5"])
        self.assertTrue(any("ROI 1 in 1" in w and "s2" in w for w in ds.warnings))

    def test_repeated_measures_drop_every_observation_of_the_subject(self):
        ids = ["a", "b", "c", "a", "b", "c"]      # observation 2 is subject "c"
        ds = drop_subjects_with_missing(self._dataset(condition_column="cond", subject_ids=ids))
        self.assertEqual(ds.subject_ids, ["a", "b", "a", "b"])


class TestCliKeepsUndefinedCorrelations(unittest.TestCase):

    def test_constant_region_gives_nan_not_zero(self):
        from types import SimpleNamespace
        from conninfpy.cli import _connectivity

        ts = np.random.default_rng(0).normal(size=(50, 4))
        ts[:, 2] = 1.0                                   # region without signal
        corr, kind = _connectivity(SimpleNamespace(data_kind="timeseries", data=[ts]))
        self.assertTrue(np.isnan(corr[0, 2, 0]))
        with self.assertRaisesRegex(ValueError, "Regions without signal"):
            check_connectivity(corr)


if __name__ == "__main__":
    unittest.main()
