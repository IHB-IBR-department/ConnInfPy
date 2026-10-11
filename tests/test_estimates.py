"""Coefficient (beta) maps saved with results, predictions and coupling-change classes."""
import unittest
import warnings

import numpy as np

from conninfpy import analyze, compute_p_val
from conninfpy._estimates import coupling_change
from conninfpy.glm_stats import compute_p_val_glm, compute_p_val_glm_multi, compute_p_val_paired_glm

N, n = 6, 30


def _stack(seed=0, mean=0.3, size=n):
    rng = np.random.default_rng(seed)
    y = mean + 0.2 * rng.normal(size=(size, N, N))
    y = (y + y.transpose(0, 2, 1)) / 2
    for s in y:
        np.fill_diagonal(s, 0)
    return y


class TestGLM(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.age = rng.uniform(10, 60, n)
        self.sex = rng.integers(0, 2, n).astype(float)
        self.Y = _stack()

    def test_prediction_at_the_means_is_the_mean_not_the_intercept(self):
        r = compute_p_val_glm(self.Y, interest=self.age, confounds=self.sex, method="tstat",
                              n_permutations=19, use_mp=False, rng=0)
        self.assertEqual(r.beta.shape, (3, N, N))
        self.assertEqual(r.design["columns"], ["intercept", "confound_1", "interest"])
        np.testing.assert_allclose(r.predicted("mean"), self.Y.mean(0), atol=1e-12)
        self.assertGreater(np.abs(r.beta[0] - self.Y.mean(0)).max(), 0.01)   # intercept = FC at age 0
        X = np.column_stack([np.ones(n), self.sex, self.age])
        ref = np.linalg.lstsq(X, self.Y[:, 1, 2], rcond=None)[0]
        np.testing.assert_allclose(r.beta[:, 1, 2], ref, atol=1e-10)
        np.testing.assert_allclose(r.effect_map()[1, 2], ref[2], atol=1e-10)
        lo, hi = r.design["comparison"]
        self.assertIn("mean-1SD", lo)
        self.assertIn("mean+1SD", hi)

    def test_binary_interest_gives_adjusted_group_means(self):
        group = np.repeat([0.0, 1.0], n // 2)
        r = compute_p_val_glm(self.Y, interest=group, method="tstat", n_permutations=19, use_mp=False, rng=0)
        a, b = r.design["comparison"]
        np.testing.assert_allclose(r.predicted(a), self.Y[:15].mean(0), atol=1e-12)
        np.testing.assert_allclose(r.predicted(b), self.Y[15:].mean(0), atol=1e-12)

    def test_available_case_beta_uses_each_edges_subjects(self):
        Y = self.Y.copy()
        Y[0, 2, :] = Y[0, :, 2] = np.nan
        Y[0, 2, 2] = 0.0
        r = compute_p_val_glm(Y, interest=self.age, method="tstat", n_permutations=19, use_mp=False,
                              rng=0, missing="available_case", min_coverage=0.9)
        X = np.column_stack([np.ones(n), self.age])
        ref = np.linalg.lstsq(X[1:], Y[1:, 2, 4], rcond=None)[0]
        np.testing.assert_allclose(r.beta[:, 2, 4], ref, atol=1e-10)
        self.assertTrue(np.isnan(r.beta[:, 2, 2]).all())

    def test_multi_contrast_shares_beta(self):
        X = np.column_stack([np.ones(n), self.sex, self.age])
        out = compute_p_val_glm_multi(self.Y, design_matrix=X,
                                      contrasts={"sex": np.array([0, 1.0, 0]), "age": np.array([0, 0, 1.0])},
                                      method="tstat", n_permutations=19, use_mp=False, rng=0,
                                      design_columns=["intercept", "sex", "age"])
        self.assertIs(out["sex"].beta, out["age"].beta)
        self.assertEqual(out["age"].design["interest"], {"column": "age", "kind": "continuous"})
        np.testing.assert_allclose(out["age"].effect_map(), out["age"].beta[2])

    def test_paired_glm_with_confounds(self):
        A, B = _stack(2), _stack(3, mean=0.4)
        cA, cB = np.random.default_rng(4).normal(size=(2, n))
        r = compute_p_val_paired_glm(A, B, confounds_A=cA, confounds_B=cB, method="tstat",
                                     n_permutations=19, use_mp=False, rng=0)
        np.testing.assert_allclose(r.predicted("A"), A.mean(0), atol=1e-12)
        np.testing.assert_allclose(r.predicted("B (observed)"), B.mean(0), atol=1e-10)
        np.testing.assert_allclose(r.predicted("B (no confound change)") - r.predicted("A"), r.beta[1], atol=1e-12)


class TestPairwise(unittest.TestCase):

    def test_equivalent_designs(self):
        g1, g2 = _stack(5), _stack(6, mean=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            two = compute_p_val(g1, g2, test_type="two-sample", method="tstat", n_permutations=19,
                                use_mp=False, rng=0)
            par = compute_p_val(g1, g2, test_type="paired", method="bonferroni")
            one = compute_p_val(g1, test_type="one-sample", method="tstat", n_permutations=19, use_mp=False, rng=0)
        np.testing.assert_allclose(two.predicted("group2"), g2.mean(0), atol=1e-12)
        np.testing.assert_allclose(two.effect_map(), g2.mean(0) - g1.mean(0), atol=1e-12)
        np.testing.assert_allclose(par.predicted("B"), g2.mean(0), atol=1e-12)
        np.testing.assert_allclose(one.predicted("mean"), g1.mean(0), atol=1e-12)
        self.assertIsNone(one.design["comparison"])

    def test_analyze_sets_the_scale_and_grid_selection_keeps_beta(self):
        g1, g2 = _stack(5), _stack(6, mean=0.5)
        out = analyze(group1=g1, group2=g2, input="z", test_type="two-sample", method="tfnbs",
                      e=[0.4, 0.5], h=[3.0, 3.0], n=10, n_permutations=19, rng=0, use_mp=False)
        r = out.inference
        self.assertEqual(r.design["scale"], "fisher_z")
        sel = r.select(1)
        self.assertIs(sel.beta, r.beta)
        c = sel.coupling()
        np.testing.assert_allclose(c["delta"], np.tanh(g2.mean(0)) - np.tanh(g1.mean(0)), atol=1e-12)


class TestCouplingChange(unittest.TestCase):

    def test_classes(self):
        a = np.array([0.3, 0.5, -0.3, -0.5, 0.2, -0.2, 0.05, 0.4])
        b = np.array([0.5, 0.3, -0.5, -0.3, -0.2, 0.2, 0.4, 0.05])
        cls, d = coupling_change(a, b, scale="native", near_zero=0.1)
        self.assertEqual(list(cls), ["stronger positive coupling", "weaker positive coupling",
                                     "stronger anticorrelation", "weaker anticorrelation",
                                     "sign change: positive to negative", "sign change: negative to positive",
                                     "emerges from ~0", "returns to ~0"])
        np.testing.assert_allclose(d, b - a)
        cls, d = coupling_change(np.array([0.3]), np.array([0.5]), scale="fisher_z")
        np.testing.assert_allclose(d, np.tanh(0.5) - np.tanh(0.3))
        cls, _ = coupling_change(np.array([np.nan]), np.array([0.5]))
        self.assertIsNone(cls[0])


if __name__ == "__main__":
    unittest.main()


class TestAnnotationUsesCoupling(unittest.TestCase):

    def test_summary_and_edges_report_coupling_change(self):
        import tempfile

        from conninfpy.interpret.annotate import annotate_rois, summarize_annotation
        from conninfpy.brainnetome import BrainnetomeReference

        coords = [(-42, 36, 24), (42, 36, 24), (-22, -4, -18), (-38, -22, 56), (-6, -52, 28), (-8, -84, 4)]
        rois = annotate_rois(coords, reference=BrainnetomeReference.load(
            data_dir=tempfile.mkdtemp(), profiles=False, cache_dir=tempfile.mkdtemp()),
            support="sphere", radius_mm=6)
        g1, g2 = _stack(7), _stack(8)
        g2[:, 0, 1] = g2[:, 1, 0] = g2[:, 0, 1] + 0.6          # stronger positive coupling
        out = analyze(group1=g1, group2=g2, input="z", test_type="two-sample", method="tstat",
                      n_permutations=99, rng=0, use_mp=False)
        e = out.inference.annotated_edges(rois)
        self.assertEqual(e.loc[(e.roi_i == 0) & (e.roi_j == 1), "coupling_change"].item(),
                         "stronger positive coupling")
        self.assertIn("delta_r", e.columns)
        rel = summarize_annotation(out.inference, rois)["relations"]
        self.assertIn("coupling_change", set(rel["relation"]))
