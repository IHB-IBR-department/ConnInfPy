"""Site-interest association diagnostic for the label-leak-free ComBat arms.

``combat_only`` and ``combat_site_dummies_glm`` never let ComBat see the
interest column (Nygaard 2016 label-leak avoidance). A site effect that
correlates with the tested variable is therefore partly removed as if it were
site noise, which attenuates the interest effect. ``analyze()`` quantifies that
confounding as the multiple correlation ``R = sqrt(R^2)`` between site
indicators and the interest regressor(s) and reports it in ``flags`` and
``combat_diagnostics``.

Checks (each a few seconds):

1. ``_site_interest_association`` returns ``|r|`` for two sites, approaches 1
   for a site-coded regressor with ``K >= 3`` sites, and reports the maximum
   over a dict of predictors.
2. Degenerate inputs (single site, constant / non-numeric column) yield
   ``None`` instead of a misleading number.
3. Through ``analyze()``: a balanced design (``corr_site_interest=0``) gives a
   low ``R`` and a confounded design (``corr_site_interest=0.8``) a high ``R``,
   for both ComBat strategies; the flag text and ``combat_diagnostics`` agree.
4. ``site_dummies_glm`` (no ComBat) does not emit the flag.
"""
from __future__ import annotations

import re
import unittest
import warnings

import numpy as np

from conninfpy import analyze, generate_multisite_glm_dataset
from conninfpy.analysis import _site_interest_association

N_SUBJECTS = 24
N_ROIS = 20
N_SITES = 3
N_PERM = 30
COMBAT_STRATEGIES = ("combat_only", "combat_site_dummies_glm")
FLAG_PREFIX = "site_interest_association: R="
_FLAG_R = re.compile(r"^site_interest_association: R=([0-9.]+) \(R\^2=([0-9.]+)\)")


def _dataset(corr_site_interest: float, seed: int = 0):
    return generate_multisite_glm_dataset(
        n_subjects=N_SUBJECTS, N=N_ROIS, n_sites=N_SITES,
        effect_size=0.0, site_shift_sigma=0.2,
        corr_site_interest=corr_site_interest, seed=seed,
    )


def _analyze(data, harmonize, **kwargs):
    interest = kwargs.pop("interest", data["interest"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return analyze(
            data["Y"], interest=interest, sites=data["sites"],
            confounds=np.random.default_rng(101).standard_normal((N_SUBJECTS, 1)),
            fisher_z=False, method="tstat", n_permutations=N_PERM,
            acceleration=None, use_mp=False, harmonize=harmonize, rng=0,
            **kwargs,
        )


def _flag_r(flags) -> tuple[float, float]:
    matches = [f for f in flags if f.startswith(FLAG_PREFIX)]
    assert len(matches) == 1, f"expected exactly one site_interest_association flag, got {matches!r}"
    m = _FLAG_R.match(matches[0])
    assert m is not None, matches[0]
    return float(m.group(1)), float(m.group(2))


class TestSiteInterestAssociationHelper(unittest.TestCase):
    """Pure helper: ``R = sqrt(R^2)`` of interest on site indicators."""

    def test_two_sites_reduces_to_abs_pearson_r(self):
        rng = np.random.default_rng(0)
        sites = np.repeat([0, 1], 30)
        y = rng.standard_normal(60)
        out = _site_interest_association(sites, y)
        self.assertEqual(out["n_sites"], 2)
        self.assertAlmostEqual(out["R"], abs(np.corrcoef(sites, y)[0, 1]), places=10)
        # Perfect confounding: interest is the site indicator itself.
        self.assertEqual(_site_interest_association(sites, sites.astype(float))["R"], 1.0)

    def test_column_2d_input_matches_1d(self):
        rng = np.random.default_rng(1)
        sites = np.repeat([0, 1, 2], 20)
        y = rng.standard_normal(60)
        a = _site_interest_association(sites, y)["R"]
        b = _site_interest_association(sites, y[:, None])["R"]
        self.assertAlmostEqual(a, b, places=12)

    def test_multiple_sites_site_coded_regressor(self):
        rng = np.random.default_rng(2)
        sites = np.repeat([0, 1, 2], 20)
        noisy_code = sites + 0.5 * rng.standard_normal(60)
        out = _site_interest_association(sites, noisy_code)
        self.assertEqual(out["n_sites"], 3)
        self.assertGreater(out["R"], 0.7)
        self.assertLess(out["R"], 1.0)
        self.assertAlmostEqual(out["R"] ** 2, out["R2"], places=12)

    def test_dict_reports_max_column(self):
        rng = np.random.default_rng(3)
        sites = np.repeat([0, 1, 2], 20)
        out = _site_interest_association(
            sites, {"age": rng.standard_normal(60), "grp": (sites == 1).astype(float)}
        )
        self.assertEqual(out["column"], "grp")
        self.assertEqual(out["R"], 1.0)
        self.assertEqual(out["per_column_R2"]["grp"], 1.0)
        self.assertLess(out["per_column_R2"]["age"], 0.5)

    def test_degenerate_inputs_return_none(self):
        rng = np.random.default_rng(4)
        sites = np.repeat([0, 1], 30)
        self.assertIsNone(_site_interest_association(np.zeros(60, dtype=int), rng.standard_normal(60)))
        self.assertIsNone(_site_interest_association(sites, np.ones(60)))
        self.assertIsNone(_site_interest_association(sites, np.zeros(60)))
        self.assertIsNone(_site_interest_association(sites, ["a"] * 60))
        self.assertIsNone(_site_interest_association(sites, rng.standard_normal(59)))
        self.assertIsNone(_site_interest_association(sites, {}))


class TestCombatSiteInterestFlag(unittest.TestCase):
    """End-to-end through ``analyze()`` for both ComBat strategies."""

    def _check(self, harmonize: str, corr: float, low: bool):
        data = _dataset(corr)
        out = _analyze(data, harmonize)
        r_flag, r2_flag = _flag_r(out.flags)
        diag = out.combat_diagnostics
        self.assertIn("site_interest_corr", diag)
        self.assertIn("site_interest_R2", diag)
        self.assertEqual(diag["strategy"], harmonize)
        # Flag text and numeric diagnostics agree (flag is rounded to 3 decimals).
        self.assertAlmostEqual(r_flag, diag["site_interest_corr"], delta=5e-4)
        self.assertAlmostEqual(r2_flag, diag["site_interest_R2"], delta=5e-4)
        self.assertAlmostEqual(diag["site_interest_corr"] ** 2, diag["site_interest_R2"], places=6)
        self.assertIn("label-leak-free ComBat", " ".join(f for f in out.flags if f.startswith(FLAG_PREFIX)))
        # The same flag is threaded onto the inference result.
        self.assertTrue(any(f.startswith(FLAG_PREFIX) for f in out.inference.flags))
        if low:
            self.assertLess(diag["site_interest_corr"], 0.3)
        else:
            self.assertGreater(diag["site_interest_corr"], 0.6)
        return diag["site_interest_corr"]

    def test_balanced_design_low_association(self):
        for harmonize in COMBAT_STRATEGIES:
            with self.subTest(harmonize=harmonize):
                self._check(harmonize, 0.0, low=True)

    def test_confounded_design_high_association(self):
        for harmonize in COMBAT_STRATEGIES:
            with self.subTest(harmonize=harmonize):
                self._check(harmonize, 0.8, low=False)

    def test_confounded_exceeds_balanced(self):
        for harmonize in COMBAT_STRATEGIES:
            with self.subTest(harmonize=harmonize):
                r_bal = self._check(harmonize, 0.0, low=True)
                r_conf = self._check(harmonize, 0.8, low=False)
                self.assertGreater(r_conf, r_bal + 0.3)

    def test_multi_predictor_reports_max_column(self):
        data = _dataset(0.0)
        interest = {"age": data["interest"], "grp": (data["sites"] == 1).astype(float)}
        out = _analyze(data, "combat_only", interest=interest)
        self.assertIn("age", out)
        r_flag, _ = _flag_r(out["age"].flags)
        self.assertEqual(r_flag, 1.0)
        self.assertEqual(out["age"].combat_diagnostics["site_interest_column"], "grp")
        self.assertEqual(out["age"].combat_diagnostics["site_interest_R2_per_column"]["grp"], 1.0)
        self.assertIn("max over interest columns: 'grp'", " ".join(out["age"].flags))

    def test_no_combat_no_flag(self):
        out = _analyze(_dataset(0.8), None)  # site_dummies_glm: ComBat does not run
        self.assertFalse(any(f.startswith(FLAG_PREFIX) for f in out.flags))
        self.assertNotIn("site_interest_corr", out.combat_diagnostics)


if __name__ == "__main__":
    unittest.main()
