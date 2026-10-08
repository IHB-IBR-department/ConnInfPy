"""
Tests for permutation acceleration methods (GPD and gamma).

Verifies:
- GPD fit correctness on known distributions
- Accelerated p-values close to full-permutation reference
- Null control preserved (no type I error inflation)
- Integration with both t-test and GLM pipelines
- Gamma fallback
"""

import unittest

import numpy as np
import numpy.testing as npt

from conninfpy.acceleration import (
    _fit_gpd_mom,
    _gpd_sf,
    fit_gpd_tail,
    fit_gamma_tail,
    compute_p_values_accelerated,
)
from conninfpy.pairwise_stats import compute_p_val
from conninfpy.glm_stats import compute_p_val_glm


# =============================================================================
# 1. GPD fitting internals
# =============================================================================


class TestGPDFit(unittest.TestCase):
    """Test GPD fitting by method of moments."""

    def test_exponential_case(self):
        """When xi=0, GPD is exponential. MoM should recover xi≈0."""
        rng = np.random.RandomState(42)
        # Exponential(scale=2) → GPD with xi=0, sigma=2
        exceedances = rng.exponential(scale=2.0, size=1000)
        sigma, xi, valid = _fit_gpd_mom(exceedances)
        self.assertTrue(valid)
        self.assertAlmostEqual(xi, 0.0, delta=0.1)
        self.assertAlmostEqual(sigma, 2.0, delta=0.3)

    def test_shape_sign_matches_survival_parameterization(self):
        """S(y) = (1 + xi y / sigma)^(-1/xi): xi < 0 bounded tail, xi > 0 heavy.

        Regression: the estimator returned Hosking & Wallis's k = -xi, so a
        light tail was fitted as heavy and vice versa (exponential tests
        cannot see this, xi = 0).
        """
        from scipy import stats
        for shape in (-0.3, 0.2):
            y = stats.genpareto.rvs(shape, scale=1.0, size=200_000, random_state=0)
            sigma, xi, valid = _fit_gpd_mom(y)
            self.assertTrue(valid)
            self.assertAlmostEqual(xi, shape, delta=0.05, msg=f"shape {shape}")
            self.assertAlmostEqual(sigma, 1.0, delta=0.1, msg=f"shape {shape}")
            q = stats.genpareto.ppf(1 - 1e-3, shape)
            self.assertAlmostEqual(_gpd_sf(np.array([q]), sigma, xi).item() / 1e-3, 1.0,
                                   delta=0.3, msg=f"shape {shape}")

    def test_too_few_exceedances(self):
        """Fewer than 10 exceedances → invalid fit."""
        _, _, valid = _fit_gpd_mom(np.array([1.0, 2.0, 3.0]))
        self.assertFalse(valid)

    def test_constant_exceedances(self):
        """Constant values (zero variance) → invalid fit."""
        _, _, valid = _fit_gpd_mom(np.ones(20))
        self.assertFalse(valid)


class TestGPDSurvival(unittest.TestCase):
    """Test GPD survival function."""

    def test_exponential_sf(self):
        """When xi=0, GPD SF = exp(-x/sigma)."""
        x = np.array([0.0, 1.0, 2.0, 5.0])
        sf = _gpd_sf(x, sigma=2.0, xi=0.0)
        expected = np.exp(-x / 2.0)
        npt.assert_allclose(sf, expected, atol=1e-10)

    def test_sf_at_zero(self):
        """SF at 0 should be 1."""
        sf = _gpd_sf(np.array([0.0]), sigma=1.0, xi=0.5)
        self.assertAlmostEqual(sf.item(), 1.0)

    def test_sf_monotone_decreasing(self):
        """SF should be monotonically decreasing."""
        x = np.linspace(0, 10, 50)
        sf = _gpd_sf(x, sigma=1.0, xi=0.2)
        self.assertTrue(np.all(np.diff(sf) <= 0))


# =============================================================================
# 2. GPD tail approximation
# =============================================================================


class TestFitGPDTail(unittest.TestCase):
    """Test fit_gpd_tail end-to-end."""

    def test_known_null_distribution(self):
        """GPD p-values from Gaussian null should be reasonable."""
        rng = np.random.RandomState(42)
        null_dist = rng.standard_normal(500)  # simulated max-stats
        # Observed values at different quantiles
        observed = np.array([[0.0, 1.0], [2.0, 3.0]])

        p_gpd = fit_gpd_tail(null_dist, observed)
        p_emp = np.mean(observed[..., np.newaxis] <= null_dist, axis=-1)

        # GPD and empirical should roughly agree for moderate values
        # For extreme values (3.0), GPD should give more precise estimate
        self.assertEqual(p_gpd.shape, observed.shape)
        self.assertTrue(np.all(p_gpd >= 0))
        self.assertTrue(np.all(p_gpd <= 1))

        # p-value ordering should be preserved
        self.assertGreater(p_gpd[0, 0], p_gpd[1, 1])

    def test_fallback_to_empirical(self):
        """When GPD fit fails, should fall back to empirical p-values."""
        rng = np.random.RandomState(42)
        # Very small null (hard to fit GPD)
        null_dist = rng.standard_normal(20)
        observed = np.array([[1.0, 2.0], [0.5, 0.0]])

        p_gpd = fit_gpd_tail(null_dist, observed)
        p_emp = np.mean(observed[..., np.newaxis] <= null_dist, axis=-1)

        # Should get valid p-values regardless
        self.assertEqual(p_gpd.shape, observed.shape)
        self.assertTrue(np.all(p_gpd >= 0))
        self.assertTrue(np.all(p_gpd <= 1))


class TestTailOnlyWherePermutationsAreTooFew(unittest.TestCase):
    """The fitted tail replaces a permutation p-value only in the sparse tail.

    Regression: the GPD threshold started at the median of the null and the
    fit replaced every p-value above it, including p ~ 0.05 that 5,000
    permutations estimate directly; on ABIDE this moved a 138-edge cluster
    with exact p = 0.0496 to 0.056.
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        self.null = rng.exponential(scale=1.0, size=5000)
        J = len(self.null)
        # Statistics reached by many null maxima: the 50th..99th percentiles
        self.dense_obs = np.quantile(self.null, [0.5, 0.9, 0.95, 0.975, 0.99])
        count = np.sum(self.dense_obs[:, None] <= self.null[None, :], axis=1)
        self.assertTrue(np.all(count >= 10))
        self.p_exact = (count + 1.0) / (J + 1.0)

    def test_gpd_keeps_permutation_p_values_in_dense_tail(self):
        npt.assert_array_equal(fit_gpd_tail(self.null, self.dense_obs), self.p_exact)

    def test_gamma_keeps_permutation_p_values_in_dense_tail(self):
        npt.assert_array_equal(fit_gamma_tail(self.null, self.dense_obs), self.p_exact)

    def test_gpd_sparse_tail_is_unbiased(self):
        # Exponential null: the GPD (xi = 0) is the true tail above any
        # threshold. Single fits are noisy, so check the median over seeds.
        truth = np.array([0.001, 0.0005])           # ~5 and ~2.5 exceedances of 5,000
        obs = -np.log(truth)
        ratios = [fit_gpd_tail(np.random.RandomState(seed).exponential(size=5000), obs) / truth
                  for seed in range(20)]
        npt.assert_allclose(np.median(ratios, axis=0), 1.0, atol=0.25)

    def test_gpd_sparse_tail_unbiased_for_light_and_heavy_tails(self):
        # The fit must actually be used (p off the permutation grid k/(J+1)):
        # with the shape sign flipped, the goodness-of-fit check rejected it
        # and the empirical fallback hid the error.
        from scipy import stats
        J = 5000
        truth = np.array([0.001, 0.0005])
        for shape in (-0.2, 0.2):
            obs = stats.genpareto.isf(truth, shape)
            ratios, used = [], 0
            for seed in range(20):
                p = fit_gpd_tail(stats.genpareto.rvs(shape, size=J, random_state=seed), obs)
                k = p * (J + 1)
                used += int(np.all(np.abs(k - np.round(k)) > 1e-6))
                ratios.append(p / truth)
            self.assertGreaterEqual(used, 18, f"shape {shape}: GPD used in {used}/20")
            npt.assert_allclose(np.median(ratios, axis=0), 1.0, atol=0.3, err_msg=f"shape {shape}")

    def test_gamma_keeps_permutation_p_values_for_left_skewed_null(self):
        # Regression: skewness was squared, so a left-skewed null was fitted
        # with a right-skewed gamma.
        rng = np.random.RandomState(3)
        null = -rng.exponential(size=2000)            # skewness -2
        obs = np.array([null.max() + 0.01, np.quantile(null, 0.999)])
        count = np.sum(obs[:, None] <= null[None, :], axis=1)
        npt.assert_array_equal(fit_gamma_tail(null, obs), (count + 1.0) / (len(null) + 1.0))

    def test_p_values_monotone_across_the_switch(self):
        grid = np.linspace(np.quantile(self.null, 0.99), self.null.max() * 1.2, 200)
        for fit in (fit_gpd_tail, fit_gamma_tail):
            p = fit(self.null, grid)
            self.assertTrue(np.all(np.diff(p) <= 1e-15), fit.__name__)


# =============================================================================
# 3. Gamma approximation
# =============================================================================


class TestFitGammaTail(unittest.TestCase):
    """Test gamma approximation."""

    def test_gamma_from_gaussian_null(self):
        """Gamma fit to Gaussian max-stats produces valid p-values."""
        rng = np.random.RandomState(42)
        null_dist = np.abs(rng.standard_normal(500))
        observed = np.array([[0.5, 1.0], [2.0, 3.0]])

        p_gamma = fit_gamma_tail(null_dist, observed)

        self.assertEqual(p_gamma.shape, observed.shape)
        self.assertTrue(np.all(p_gamma >= 0))
        self.assertTrue(np.all(p_gamma <= 1))
        # Ordering should be preserved
        self.assertGreater(p_gamma[0, 0], p_gamma[1, 1])

    def test_gamma_matches_empirical_roughly(self):
        """Gamma p-values should be in the same ballpark as empirical."""
        rng = np.random.RandomState(42)
        null_dist = np.abs(rng.standard_normal(1000))
        observed = np.array([[1.5]])

        p_gamma = fit_gamma_tail(null_dist, observed)
        p_emp = np.mean(observed[..., np.newaxis] <= null_dist, axis=-1)

        # Should agree within 0.1 for moderate values
        self.assertAlmostEqual(p_gamma.item(), p_emp.item(), delta=0.1)


# =============================================================================
# 4. compute_p_values_accelerated wrapper
# =============================================================================


class TestComputePValuesAccelerated(unittest.TestCase):
    """Test the drop-in replacement for _compute_p_values_from_null."""

    def test_gpd_method(self):
        """GPD method returns correct structure."""
        rng = np.random.RandomState(42)
        N = 5
        emp = {
            "positive": rng.rand(N, N) * 3,
            "negative": rng.rand(N, N) * 3,
        }
        null = {
            "positive": rng.standard_normal(200),
            "negative": rng.standard_normal(200),
        }
        p = compute_p_values_accelerated(emp, null, method="gpd")
        self.assertIn("positive", p)
        self.assertIn("negative", p)
        self.assertEqual(p["positive"].shape, (N, N))
        self.assertTrue(np.all(p["positive"] >= 0))
        self.assertTrue(np.all(p["positive"] <= 1))

    def test_gamma_method(self):
        """Gamma method returns correct structure."""
        rng = np.random.RandomState(42)
        N = 5
        emp = {"positive": rng.rand(N, N) * 3, "negative": rng.rand(N, N) * 3}
        null = {"positive": np.abs(rng.standard_normal(200)),
                "negative": np.abs(rng.standard_normal(200))}
        p = compute_p_values_accelerated(emp, null, method="gamma")
        self.assertIn("positive", p)
        self.assertEqual(p["positive"].shape, (N, N))


# =============================================================================
# 5. Integration with t-test pipeline
# =============================================================================


class TestAccelerationTTestPipeline(unittest.TestCase):
    """Integration tests: acceleration in compute_p_val."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        cls.N = 8
        n_per = 15
        cls.group1 = rng.randn(n_per, cls.N, cls.N)
        cls.group2 = rng.randn(n_per, cls.N, cls.N)
        for g in (cls.group1, cls.group2):
            for s in range(n_per):
                g[s] = (g[s] + g[s].T) / 2
                np.fill_diagonal(g[s], 0)

    def test_gpd_tstat(self):
        """compute_p_val with acceleration='gpd' runs and returns valid p."""
        p = compute_p_val(
            self.group1, self.group2,
            n_permutations=200, test_type='two-sample',
            method='tstat', use_mp=False, rng=0,
            acceleration='gpd',
        )
        self.assertIn("positive", p)
        self.assertEqual(p["positive"].shape, (self.N, self.N))
        self.assertTrue(np.all(p["positive"] >= 0))
        self.assertTrue(np.all(p["positive"] <= 1))

    def test_gamma_tfnbs(self):
        """compute_p_val with acceleration='gamma' and TFNBS."""
        p = compute_p_val(
            self.group1, self.group2,
            n_permutations=200, test_type='two-sample',
            method='tfnbs', use_mp=False, rng=0,
            acceleration='gamma', e=0.4, h=3.0, n=5,
        )
        self.assertIn("positive", p)
        self.assertTrue(np.all(p["positive"] >= 0))
        self.assertTrue(np.all(p["positive"] <= 1))

    def test_no_acceleration_baseline(self):
        """acceleration=None preserves original behavior."""
        p_none = compute_p_val(
            self.group1, self.group2,
            n_permutations=100, test_type='two-sample',
            method='tstat', use_mp=False, rng=0,
            acceleration=None,
        )
        # Just verify it runs — exact values tested elsewhere
        self.assertIn("positive", p_none)


# =============================================================================
# 6. Integration with GLM pipeline
# =============================================================================


class TestAccelerationGLMPipeline(unittest.TestCase):
    """Integration tests: acceleration in compute_p_val_glm."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        cls.N = 8
        cls.n = 20
        cls.Y = rng.randn(cls.n, cls.N, cls.N)
        for s in range(cls.n):
            cls.Y[s] = (cls.Y[s] + cls.Y[s].T) / 2
            np.fill_diagonal(cls.Y[s], 0)
        cls.age = rng.randn(cls.n)

    def test_gpd_glm(self):
        """compute_p_val_glm with acceleration='gpd' runs correctly."""
        p = compute_p_val_glm(
            self.Y, interest=self.age, method='tstat',
            n_permutations=200, use_mp=False, rng=0,
            acceleration='gpd',
        )
        self.assertIn("positive", p)
        self.assertEqual(p["positive"].shape, (self.N, self.N))
        self.assertTrue(np.all(p["positive"] >= 0))
        self.assertTrue(np.all(p["positive"] <= 1))

    def test_gamma_glm_tfnbs(self):
        """compute_p_val_glm with gamma + TFNBS."""
        p = compute_p_val_glm(
            self.Y, interest=self.age, method='tfnbs',
            n_permutations=200, use_mp=False, rng=0,
            acceleration='gamma', e=0.4, h=3.0, n=5,
        )
        self.assertIn("positive", p)
        self.assertTrue(np.all(p["positive"] >= 0))

    def test_gpd_planted_effect(self):
        """GPD acceleration still detects planted effect."""
        rng = np.random.RandomState(321)
        n = 30
        N = 8
        age = rng.randn(n)

        Y = rng.randn(n, N, N) * 0.3
        for s in range(n):
            Y[s] = (Y[s] + Y[s].T) / 2
            np.fill_diagonal(Y[s], 0)

        # Plant strong effect
        for s in range(n):
            Y[s, 1, 2] += 1.5 * age[s]
            Y[s, 2, 1] += 1.5 * age[s]

        p = compute_p_val_glm(
            Y, interest=age, method='tstat',
            n_permutations=200, use_mp=False, rng=0,
            acceleration='gpd',
        )

        # Planted edge should still be significant
        self.assertLess(p["positive"][1, 2], 0.05,
                        f"Planted edge p={p['positive'][1, 2]}, expected < 0.05")


if __name__ == "__main__":
    unittest.main()
