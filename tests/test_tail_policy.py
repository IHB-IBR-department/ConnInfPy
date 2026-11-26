"""The analyze() tail-policy contract: two-sided default, prespecified
directions and honest post hoc provenance.

Policy (owner decision): ``analyze()`` tests one two-sided family by default
— both directions are corrected against the joint max null, so a single
nominal alpha covers both. A one-sided hypothesis must be declared before
the run (``tested_tail=``); a direction chosen on a finished result is a
post hoc selection and is marked as such. Explicit ``two_tailed=False``
keeps the legacy separate one-sided families and never promises a
combined-direction FWER of alpha.
"""
from __future__ import annotations

import io
import pickle
import unittest

import numpy as np

from conninfpy import analyze, compute_p_val, compute_p_val_glm
from conninfpy.results import InferenceResult

N_PERM = 120


def _glm_data(n=24, N=6, seed=0):
    """Fisher-z stacks (diagonal 0) with a positive effect of x on one edge."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    Y = 0.3 * rng.standard_normal((n, N, N))
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    Y[:, 0, 1] = Y[:, 1, 0] = 0.2 + 0.15 * x + 0.05 * rng.standard_normal(n)
    Y[:, np.arange(N), np.arange(N)] = 0.0
    return Y, x


def _groups(seed=1, n=12, N=6):
    rng = np.random.default_rng(seed)

    def sym(a):
        a = (a + a.transpose(0, 2, 1)) / 2
        a[:, np.arange(N), np.arange(N)] = 0.0
        return a

    g1, g2 = sym(rng.normal(size=(n, N, N))), sym(rng.normal(size=(n, N, N)))
    for i in range(3):
        for j in range(3):
            if i != j:
                g2[:, i, j] += 1.0
    return sym(g1), sym(g2)


class TestTwoSidedDefault(unittest.TestCase):
    """The default analyze() run is the low-level two_tailed=True run."""

    def test_default_glm_equals_explicit_two_tailed_kernel(self):
        Y, x = _glm_data()
        kw = dict(method="tstat", n_permutations=N_PERM, rng=7, use_mp=False)
        out = analyze(Y, interest=x, input="z", **kw).inference
        low = compute_p_val_glm(Y, interest=x, two_tailed=True, **kw)
        self.assertTrue(out.two_tailed)
        self.assertEqual(out.tail_policy, "two-sided")
        np.testing.assert_allclose(out.positive, low.positive, atol=1e-15)
        np.testing.assert_allclose(out.negative, low.negative, atol=1e-15)
        np.testing.assert_allclose(out.stat_signed, low.stat_signed, atol=1e-15)

    def test_explicit_two_tailed_true_is_the_same_default(self):
        Y, x = _glm_data()
        kw = dict(method="tstat", n_permutations=N_PERM, rng=7, use_mp=False)
        none_default = analyze(Y, interest=x, input="z", **kw).inference
        true_default = analyze(Y, interest=x, input="z", two_tailed=True, **kw).inference
        np.testing.assert_allclose(none_default.positive, true_default.positive, atol=0)

    def test_two_sample_path_default_is_two_sided(self):
        g1, g2 = _groups()
        out = analyze(group1=g1, group2=g2, input="z", method="tstat",
                      n_permutations=N_PERM, rng=5, use_mp=False).inference
        self.assertTrue(out.two_tailed)
        self.assertEqual(out.tail_policy, "two-sided")


class TestPrespecifiedTail(unittest.TestCase):
    """tested_tail= declares the direction before the run; the restriction
    equals the explicit low-level separate-null run restricted to that tail."""

    def test_prespecified_equals_low_level_separate_nulls(self):
        Y, x = _glm_data()
        kw = dict(method="tstat", n_permutations=N_PERM, rng=7, use_mp=False)
        for tail in ("positive", "negative"):
            with self.subTest(tail=tail):
                out = analyze(Y, interest=x, input="z", tested_tail=tail, **kw)
                inf = out.inference
                sep = compute_p_val_glm(Y, interest=x, two_tailed=False, **kw)
                np.testing.assert_allclose(inf[tail], sep[tail], atol=1e-15)
                other = "negative" if tail == "positive" else "positive"
                self.assertTrue(np.all(inf[other] == 1.0))
                self.assertEqual(inf.tested_tail, tail)
                self.assertEqual(inf.tail_policy, f"one-sided-{tail}")
                self.assertEqual(inf.tail_selection, "prespecified")
                self.assertEqual(
                    inf.analysis_provenance["family"]["tail_selection"],
                    "prespecified",
                )
                self.assertEqual(
                    inf.analysis_provenance["family"]["tail_policy"],
                    f"one-sided-{tail}",
                )
                self.assertEqual(out.tail_policy, f"one-sided-{tail}")
                self.assertEqual(out.tested_tail, tail)
                self.assertEqual(out.tail_selection, "prespecified")
                self.assertFalse(out.two_tailed)
                self.assertTrue(any("prespecified one-sided test" in f for f in out.flags))

    def test_prespecified_tail_rejects_bad_values_conflicts_and_fstat(self):
        Y, x = _glm_data()
        kw = dict(method="tstat", n_permutations=20, rng=1, use_mp=False)
        with self.assertRaisesRegex(ValueError, "tested_tail"):
            analyze(Y, interest=x, input="z", tested_tail="up", **kw)
        with self.assertRaisesRegex(ValueError, "two_tailed=True"):
            analyze(Y, interest=x, input="z", tested_tail="positive",
                    two_tailed=True, **kw)
        with self.assertRaisesRegex(ValueError, "F-stat"):
            analyze(Y, interest=x, input="z", tested_tail="positive",
                    stat_type="fstat", design_matrix=np.c_[np.ones(24), x],
                    contrast=np.array([[0.0, 1.0]]), **kw)

    def test_prespecified_tail_both_directions_on_two_sample(self):
        g1, g2 = _groups()
        sep = compute_p_val(g1, g2, test_type="two-sample", method="tstat",
                            two_tailed=False, n_permutations=N_PERM, rng=5, use_mp=False)
        for tail in ("positive", "negative"):
            with self.subTest(tail=tail):
                out = analyze(group1=g1, group2=g2, input="z", method="tstat",
                              tested_tail=tail, n_permutations=N_PERM,
                              rng=5, use_mp=False).inference
                np.testing.assert_allclose(out[tail], sep[tail], atol=1e-15)
                self.assertEqual(out.tail_selection, "prespecified")


class TestLegacySeparateFamilies(unittest.TestCase):
    """Explicit two_tailed=False is the documented legacy path."""

    def test_legacy_equals_low_level_default(self):
        Y, x = _glm_data()
        kw = dict(method="tstat", n_permutations=N_PERM, rng=7, use_mp=False)
        out = analyze(Y, interest=x, input="z", two_tailed=False, **kw)
        low = compute_p_val_glm(Y, interest=x, two_tailed=False, **kw)
        self.assertFalse(out.inference.two_tailed)
        self.assertEqual(out.inference.tail_policy, "separate-one-sided")
        np.testing.assert_allclose(out.positive, low.positive, atol=1e-15)
        np.testing.assert_allclose(out.negative, low.negative, atol=1e-15)
        self.assertTrue(any("separate one-sided families" in f for f in out.flags))

    def test_default_run_carries_no_legacy_flag(self):
        Y, x = _glm_data()
        out = analyze(Y, interest=x, input="z", method="tstat",
                      n_permutations=20, rng=1, use_mp=False)
        self.assertFalse(any("one-sided" in f for f in out.flags))


class TestPostHocSelection(unittest.TestCase):
    """one_sided() on a finished result is marked post-hoc, never prespecified."""

    def test_one_sided_marks_post_hoc(self):
        Y, x = _glm_data()
        sep = compute_p_val_glm(Y, interest=x, method="tstat", two_tailed=False,
                                n_permutations=N_PERM, rng=7, use_mp=False)
        one = sep.one_sided("positive")
        self.assertEqual(one.tail_selection, "post-hoc")
        self.assertEqual(one.analysis_provenance["family"]["tail_selection"], "post-hoc")
        self.assertEqual(one.analysis_provenance["family"]["tail_policy"], "one-sided-positive")
        # The p-values are still the separate-null one-sided p-values.
        np.testing.assert_array_equal(one.positive, sep.positive)

    def test_prespecified_and_post_hoc_share_policies_but_differ_in_provenance(self):
        Y, x = _glm_data()
        kw = dict(method="tstat", n_permutations=N_PERM, rng=7, use_mp=False)
        pre = analyze(Y, interest=x, input="z", tested_tail="positive", **kw).inference
        post = analyze(Y, interest=x, input="z", two_tailed=False, **kw).one_sided("positive")
        np.testing.assert_allclose(pre.positive, post.inference.positive, atol=0)
        self.assertEqual(pre.tail_policy, post.inference.tail_policy)
        self.assertNotEqual(pre.tail_selection, post.inference.tail_selection)
        self.assertEqual(pre.tail_selection, "prespecified")
        self.assertEqual(post.inference.tail_selection, "post-hoc")
        self.assertEqual(post.tail_selection, "post-hoc")

    def test_two_sided_result_still_rejects_one_sided(self):
        Y, x = _glm_data()
        two = analyze(Y, interest=x, input="z", method="tstat",
                      n_permutations=20, rng=1, use_mp=False)
        with self.assertRaisesRegex(ValueError, "two_tailed=False"):
            two.one_sided("positive")


class TestLegacyPickleReload(unittest.TestCase):
    """A result saved before tail_selection existed reloads with its saved
    policy and no selection claim."""

    def _legacy_result(self):
        p = np.full((4, 4), 0.3)
        m = np.full((4, 4), 0.6)
        r = InferenceResult(p, m, method="tstat", two_tailed=False,
                            tested_tail="positive",
                            stat_positive=np.ones((4, 4)),
                            stat_negative=np.zeros((4, 4)))
        r.__dict__.pop("tail_selection", None)   # as in a pre-marker pickle
        return r

    def test_reloaded_result_has_no_selection_claim(self):
        r = self._legacy_result()
        buf = io.BytesIO()
        pickle.dump(r, buf)
        buf.seek(0)
        back = pickle.load(buf)
        self.assertIsNone(back.tail_selection)
        self.assertEqual(back.tail_policy, "one-sided-positive")
        self.assertEqual(back.tested_tail, "positive")

    def test_reloaded_two_sided_result_keeps_two_sided_policy(self):
        p = np.full((4, 4), 0.3)
        m = np.full((4, 4), 0.6)
        r = InferenceResult(p, m, method="tstat", two_tailed=True)
        r.__dict__.pop("tail_selection", None)
        back = pickle.loads(pickle.dumps(r))
        self.assertTrue(back.two_tailed)
        self.assertEqual(back.tail_policy, "two-sided")
        self.assertIsNone(back.tail_selection)

    def test_new_result_round_trips_the_marker(self):
        Y, x = _glm_data()
        out = analyze(Y, interest=x, input="z", method="tstat",
                      tested_tail="negative", n_permutations=20, rng=1, use_mp=False)
        back = pickle.loads(pickle.dumps(out))
        self.assertEqual(back.inference.tested_tail, "negative")
        self.assertEqual(back.inference.tail_selection, "prespecified")
        self.assertEqual(back.tail_policy, "one-sided-negative")


class TestGridAndUnits(unittest.TestCase):
    """Tail metadata flows through (E, H) grids and the NBS/cNBS units."""

    def test_grid_default_and_select(self):
        Y, x = _glm_data()
        kw = dict(input="z", method="tfnbs", n_permutations=60, rng=7,
                  use_mp=False, e=[0.4, 0.5], h=[3.0, 3.1], n=None)
        grid = analyze(Y, interest=x, **kw).inference
        self.assertTrue(grid.is_grid)
        self.assertTrue(grid.two_tailed)
        self.assertEqual(grid.tail_policy, "two-sided")
        cell = grid.select(1)
        self.assertFalse(cell.is_grid)
        self.assertTrue(cell.two_tailed)

    def test_grid_prespecified_tail_and_select(self):
        Y, x = _glm_data()
        kw = dict(input="z", method="tfnbs", n_permutations=60, rng=7,
                  use_mp=False, e=[0.4, 0.5], h=[3.0, 3.1], n=None,
                  tested_tail="positive")
        grid = analyze(Y, interest=x, **kw).inference
        self.assertTrue(grid.tested_tail == "positive")
        cell = grid.select(0)
        self.assertEqual(cell.tested_tail, "positive")
        self.assertEqual(cell.tail_selection, "prespecified")
        self.assertEqual(cell.analysis_provenance["family"]["tail_selection"],
                         "prespecified")

    def test_nbs_component_report_family_and_restriction(self):
        g1, g2 = _groups()
        kw = dict(input="z", method="nbs", threshold=2.0, n_permutations=N_PERM,
                  rng=5, use_mp=False)
        two = analyze(group1=g1, group2=g2, **kw).inference
        self.assertEqual(two.component_report["tail_policy"], "two-sided")
        self.assertIsNone(two.component_report["tested_tail"])
        one = analyze(group1=g1, group2=g2, tested_tail="positive", **kw).inference
        rep = one.component_report
        self.assertEqual(rep["tail_policy"], "one-sided-positive")
        self.assertEqual(rep["tested_tail"], "positive")
        self.assertEqual(rep["tail_selection"], "prespecified")
        self.assertEqual(set(rep["n_components"]), {"positive"})
        table = one.component_table()
        self.assertEqual(table.attrs["tested_tail"], "positive")
        self.assertEqual(table.attrs["tail_selection"], "prespecified")

    def test_cnbs_block_report_family_and_restriction(self):
        g1, g2 = _groups()
        net = np.array([0, 0, 1, 1, 2, 2])
        kw = dict(input="z", method="cnbs", net_labels=net,
                  n_permutations=N_PERM, rng=5, use_mp=False)
        two = analyze(group1=g1, group2=g2, **kw).inference
        self.assertEqual(two.block_report["tail_policy"], "two-sided")
        self.assertIsNone(two.block_report["tested_tail"])

        sep = analyze(group1=g1, group2=g2, two_tailed=False, **kw).inference
        self.assertEqual(sep.block_report["tail_policy"], "separate-one-sided")
        self.assertIsNone(sep.block_report["tested_tail"])
        self.assertNotIn("tail_selection", sep.block_report)

        pre = analyze(group1=g1, group2=g2, tested_tail="positive", **kw).inference
        self.assertEqual(pre.tail_policy, "one-sided-positive")
        self.assertEqual(pre.tail_selection, "prespecified")
        self.assertEqual(pre.block_report["tail_policy"], "one-sided-positive")
        self.assertEqual(pre.block_report["tested_tail"], "positive")
        self.assertEqual(pre.block_report["tail_selection"], "prespecified")
        self.assertEqual(pre.block_report["n_blocks"], sep.block_report["n_blocks"])
        self.assertEqual(pre.block_report["n_blocks_tested"], sep.block_report["n_blocks_tested"])
        self.assertEqual(len(pre.block_report["blocks"]), len(sep.block_report["blocks"]))
        for src_row, pre_row in zip(sep.block_report["blocks"], pre.block_report["blocks"]):
            self.assertEqual(pre_row["p_positive"], src_row["p_positive"])
            self.assertEqual(pre_row["p_positive_uncorrected"], src_row["p_positive_uncorrected"])
            self.assertNotIn("p_negative", pre_row)
            self.assertNotIn("p_negative_uncorrected", pre_row)

        post = sep.one_sided("negative")
        self.assertEqual(post.block_report["tail_policy"], "one-sided-negative")
        self.assertEqual(post.block_report["tested_tail"], "negative")
        self.assertEqual(post.block_report["tail_selection"], "post-hoc")
        for src_row, post_row in zip(sep.block_report["blocks"], post.block_report["blocks"]):
            self.assertEqual(post_row["p_negative"], src_row["p_negative"])
            self.assertEqual(post_row["p_negative_uncorrected"], src_row["p_negative_uncorrected"])
            self.assertNotIn("p_positive", post_row)
            self.assertNotIn("p_positive_uncorrected", post_row)
        # Original separate result and its block_report remain unmutated.
        self.assertEqual(sep.block_report["tail_policy"], "separate-one-sided")
        self.assertIn("p_positive", sep.block_report["blocks"][0])
        self.assertIn("p_negative", sep.block_report["blocks"][0])


class TestMultiAndPairedPaths(unittest.TestCase):
    """The default reaches the multi-contrast and paired-interest paths."""

    def test_multi_contrast_default_and_prespecified(self):
        Y, x = _glm_data()
        z = np.random.default_rng(3).standard_normal(Y.shape[0])
        kw = dict(input="z", method="tstat", n_permutations=80, rng=3, use_mp=False)
        out = analyze(Y, interest={"a": x, "b": z}, **kw)
        for name, res in out.items():
            with self.subTest(name=name):
                self.assertTrue(res.inference.two_tailed)
                self.assertEqual(res.inference.tail_policy, "two-sided")
        out1 = analyze(Y, interest={"a": x, "b": z}, tested_tail="positive", **kw)
        for name, res in out1.items():
            with self.subTest(name=name):
                self.assertEqual(res.inference.tested_tail, "positive")
                self.assertEqual(res.inference.tail_selection, "prespecified")
                self.assertTrue(np.all(res.inference.negative == 1.0))

    def test_paired_interest_default_and_prespecified(self):
        Y, x = _glm_data(n=20)
        rng = np.random.default_rng(4)
        d = 0.2 * rng.standard_normal(Y.shape)
        d = (d + d.transpose(0, 2, 1)) / 2
        kw = dict(input="z", test_type="paired", method="tstat",
                  n_permutations=80, rng=5, use_mp=False)
        out = analyze(group1=Y, group2=Y + 0.4 + d, interest=x, **kw)
        self.assertTrue(out.inference.two_tailed)
        self.assertEqual(out.inference.tail_policy, "two-sided")
        out1 = analyze(group1=Y, group2=Y + 0.4 + d, interest=x,
                       tested_tail="positive", **kw)
        self.assertEqual(out1.inference.tested_tail, "positive")
        self.assertEqual(out1.inference.tail_selection, "prespecified")

    def test_paired_t_test_default(self):
        Y, x = _glm_data(n=16)
        rng = np.random.default_rng(6)
        d = 0.1 * rng.standard_normal(Y.shape)
        d = (d + d.transpose(0, 2, 1)) / 2
        out = analyze(group1=Y, group2=Y + 0.3 + d, input="z", test_type="paired",
                      method="tstat", n_permutations=80, rng=5, use_mp=False)
        self.assertTrue(out.inference.two_tailed)
        self.assertEqual(out.inference.tail_policy, "two-sided")


class TestBundleAndExportMetadata(unittest.TestCase):
    """ResultBundle and its exports carry the tail family and selection."""

    def test_bundle_metadata_prespecified_and_post_hoc(self):
        Y, x = _glm_data()
        kw = dict(input="z", method="tstat", n_permutations=40, rng=1, use_mp=False)
        pre = analyze(Y, interest=x, tested_tail="positive", **kw).to_bundle()
        self.assertEqual(pre.metadata["tail_policy"], "one-sided-positive")
        self.assertEqual(pre.metadata["tested_tail"], "positive")
        self.assertEqual(pre.metadata["tail_selection"], "prespecified")
        self.assertEqual(pre.metadata["family"]["tail_selection"], "prespecified")
        post = analyze(Y, interest=x, two_tailed=False, **kw).one_sided("positive").to_bundle()
        self.assertEqual(post.metadata["tail_selection"], "post-hoc")
        self.assertEqual(post.metadata["family"]["tail_selection"], "post-hoc")

    def test_summarize_attrs_carry_tail_metadata(self):
        Y, x = _glm_data()
        res = analyze(Y, interest=x, input="z", method="tstat",
                      tested_tail="positive", n_permutations=40, rng=1, use_mp=False)
        bundle = res.to_bundle()
        summary = bundle.summarize(grouping="input_position", measure="contrast_beta",
                                   contrast_id="interest", tail="positive")
        self.assertEqual(summary.attrs["tail_policy"], "one-sided-positive")
        self.assertEqual(summary.attrs["tested_tail"], "positive")
        self.assertEqual(summary.attrs["tail_selection"], "prespecified")
        self.assertEqual(summary.attrs["alpha"], 0.05)


if __name__ == "__main__":
    unittest.main()
