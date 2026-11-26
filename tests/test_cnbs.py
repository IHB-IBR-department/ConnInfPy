"""
Constrained NBS (Noble & Scheinost, 2020): block-level inference.

The statistic is the mean signed edge statistic over every edge of each
within- and between-network block; each block is tested against its own
permutation null and the block p-values are corrected across blocks
(Bonferroni or BH). This follows the authors' NBS_benchmarking code
(get_constrained_stats.m, NBSstats_smn.m).

Calibration was checked separately (1,000 complete-null datasets, 30 nodes
in 5 networks = 15 blocks, 30 + 30 subjects, 999 permutations): FWER per
tail 0.059 / 0.044 (two-sample) and 0.046 / 0.062 (GLM with a covariate),
every 95 % interval including 0.05.
"""

import unittest
import warnings

import numpy as np

from conninfpy import compute_p_val
import copy
import pickle
import tempfile
from pathlib import Path

from conninfpy import ResultBundle, analyze
from conninfpy.stats._cnbs import block_p_values
from conninfpy.stats._cnbs import restrict_block_report
from conninfpy.enhance import apply_cnbs
from conninfpy.stats.glm import compute_p_val_glm
from conninfpy.enhance.nbs import cnbs_block_means, cnbs_blocks, get_cnbs_score

LABELS = np.array([0, 0, 0, 1, 1, 2])          # network 2 has a single node


def _sym(edges, n=6):
    t = np.zeros((n, n))
    for i, j, v in edges:
        t[i, j] = t[j, i] = v
    return t


class TestBlockStatistic(unittest.TestCase):

    def test_blocks_are_within_and_between_network_pairs(self):
        b = cnbs_blocks(LABELS, 6)
        # (2, 2) has no edge: a single-node network has no within block
        self.assertEqual(b.pairs, ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)))
        np.testing.assert_array_equal(b.n_edges, [3, 6, 3, 1, 2])

    def test_mean_is_signed_and_over_every_edge_of_the_block(self):
        t = _sym([(0, 1, 2.0), (0, 2, -1.0), (3, 4, 3.0)])     # (1, 2) is 0
        means = cnbs_block_means(t, cnbs_blocks(LABELS, 6))
        np.testing.assert_allclose(means, [1 / 3, 0, 0, 3.0, 0])
        score = get_cnbs_score(t, LABELS)
        self.assertAlmostEqual(score[1, 2], 1 / 3)             # edge with t = 0 carries the block mean
        self.assertEqual(score[0, 0], 0.0)

    def test_tails_are_plus_and_minus_the_mean(self):
        t = _sym([(0, 1, 2.0), (0, 2, -5.0)])
        out = apply_cnbs({"positive": np.maximum(t, 0), "negative": np.maximum(-t, 0)}, LABELS)
        self.assertAlmostEqual(out["positive"][0], -1.0)
        self.assertAlmostEqual(out["negative"][0], 1.0)

    def test_edge_mask_drops_untested_edges(self):
        t = _sym([(0, 1, 2.0), (0, 2, 4.0), (1, 2, 6.0)])
        mask = np.ones((6, 6), dtype=bool)
        mask[1, 2] = mask[2, 1] = False
        means = cnbs_block_means(t, cnbs_blocks(LABELS, 6), mask)
        self.assertAlmostEqual(means[0], 3.0)
        mask[:3, :3] = False
        self.assertTrue(np.isnan(cnbs_block_means(t, cnbs_blocks(LABELS, 6), mask)[0]))


class TestBlockPValues(unittest.TestCase):

    def setUp(self):
        # 3 blocks, 9 permutations
        self.null = np.tile(np.arange(9.0)[:, None], (1, 3))
        self.obs = np.array([8.5, 4.0, np.nan])

    def test_counts_and_corrections(self):
        emp = {"positive": self.obs, "negative": -self.obs}
        null = {"positive": self.null, "negative": -self.null}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, rep = block_p_values(emp, null, LABELS[:5], 5, correction="bonferroni")
        rows = rep["blocks"]
        self.assertEqual(rep["n_blocks_tested"], 2)                      # NaN block left out
        self.assertAlmostEqual(rows[0]["p_positive_uncorrected"], 1 / 10)   # no null >= 8.5
        self.assertAlmostEqual(rows[1]["p_positive_uncorrected"], 6 / 10)   # 4..8 >= 4
        self.assertAlmostEqual(rows[0]["p_positive"], 2 / 10)               # x 2 testable blocks
        self.assertTrue(np.isnan(rows[2]["p_positive"]))
        # every edge carries its block's p; the diagonal is 1
        self.assertAlmostEqual(p["positive"][0, 1], 0.2)
        self.assertAlmostEqual(p["positive"][1, 2], 0.2)
        self.assertEqual(p["positive"][0, 0], 1.0)
        _, rep = block_p_values(emp, null, LABELS[:5], 5, correction="bh")
        self.assertAlmostEqual(rep["blocks"][0]["p_positive"], 0.2)         # BH: 0.1 * 2 / 1
        self.assertAlmostEqual(rep["blocks"][1]["p_positive"], 0.6)

    def test_invalid_correction(self):
        with self.assertRaisesRegex(ValueError, "cnbs_correction"):
            compute_p_val(np.zeros((4, 6, 6)), np.zeros((4, 6, 6)), test_type="two-sample",
                          method="cnbs", net_labels=LABELS, cnbs_correction="holm")


def _planted(seed=0, n=20, shift=0.6, mixed=False):
    """Network 0 (nodes 0-3) carries an effect; with mixed=True half of its
    edges go up and half down, so the block mean is ~0."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2], 4)
    N = labels.size

    def draw():
        g = rng.normal(size=(n, N, N))
        g = (g + g.transpose(0, 2, 1)) / 2
        for s in g:
            np.fill_diagonal(s, 0)
        return g
    g1, g2 = draw(), draw()
    iu = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for k, (i, j) in enumerate(iu):
        sign = (1 if k % 2 == 0 else -1) if mixed else 1
        g2[:, i, j] += sign * shift
        g2[:, j, i] += sign * shift
    return g1, g2, labels


class TestInference(unittest.TestCase):

    def test_planted_block_is_found_and_reported(self):
        g1, g2, labels = _planted()
        r = compute_p_val(g1, g2, test_type="two-sample", method="cnbs", net_labels=labels,
                          n_permutations=199, rng=0, use_mp=False)
        rows = {(b["network_a"], b["network_b"]): b for b in r.block_report["blocks"]}
        self.assertEqual(r.block_report["unit"], "block")
        self.assertLessEqual(rows[(0, 0)]["p_positive"], 0.05)
        self.assertTrue(all(b["p_positive"] > 0.05 for k, b in rows.items() if k != (0, 0)))
        p = np.asarray(r["positive"])
        self.assertEqual(len(np.unique(p[:4, :4][np.triu_indices(4, 1)])), 1)
        self.assertAlmostEqual(p[0, 1], rows[(0, 0)]["p_positive"])

    def test_opposite_edges_cancel_in_the_signed_mean(self):
        g1, g2, labels = _planted(mixed=True, shift=1.0)
        r = compute_p_val(g1, g2, test_type="two-sample", method="cnbs", net_labels=labels,
                          n_permutations=199, rng=0, use_mp=False)
        block = r.block_report["blocks"][0]
        self.assertGreater(block["p_positive"], 0.05)
        self.assertGreater(block["p_negative"], 0.05)

    def test_glm_two_tailed_and_available_case(self):
        g1, g2, labels = _planted()
        Y = np.concatenate([g1, g2])
        group = np.repeat([0.0, 1.0], 20)
        r = compute_p_val_glm(Y, interest=group, method="cnbs", net_labels=labels,
                              n_permutations=199, rng=0, use_mp=False, two_tailed=True)
        self.assertLessEqual(r.block_report["blocks"][0]["p_positive"], 0.05)
        self.assertEqual(r.block_report["blocks"][0]["p_negative"], 1.0)
        Y[0, 5, :] = Y[0, :, 5] = np.nan
        Y[0, 5, 5] = 0.0
        r = compute_p_val_glm(Y, interest=group, method="cnbs", net_labels=labels,
                              n_permutations=199, rng=0, use_mp=False, missing="available_case")
        self.assertIsNotNone(r.block_report)
        self.assertLessEqual(r.block_report["blocks"][0]["p_positive"], 0.05)



class TestBlockReportRestriction(unittest.TestCase):
    """Directional restriction keeps cNBS active/exported block_report honest."""

    def test_restrict_block_report_contract_and_non_mutation(self):
        self.assertIsNone(restrict_block_report(None, "positive"))
        observed = np.array([3.5, -2.5, 0.2])
        null = np.random.default_rng(3).normal(size=(19, 3))
        _, raw = block_p_values(
            {"positive": observed, "negative": -observed},
            {"positive": null, "negative": -null},
            np.array([0, 0, 1, 1]),
            4,
            correction="bh",
            two_tailed=False,
        )
        raw_snapshot = copy.deepcopy(raw)
        with self.assertRaises(ValueError):
            restrict_block_report(raw, "both")

        neg = restrict_block_report(raw, "negative", selection="prespecified")
        self.assertEqual(raw, raw_snapshot)
        self.assertEqual(neg["tail_policy"], "one-sided-negative")
        self.assertEqual(neg["tested_tail"], "negative")
        self.assertEqual(neg["tail_selection"], "prespecified")
        self.assertEqual(neg["correction"], "bh")
        self.assertEqual(neg["n_blocks"], 3)
        self.assertEqual(neg["n_blocks_tested"], 3)
        self.assertEqual(len(neg["blocks"]), 3)
        for r_orig, r_neg in zip(raw["blocks"], neg["blocks"], strict=True):
            self.assertEqual((r_neg["network_a"], r_neg["network_b"]),
                             (r_orig["network_a"], r_orig["network_b"]))
            self.assertEqual(r_neg["n_edges"], r_orig["n_edges"])
            self.assertEqual(r_neg["mean_stat"], r_orig["mean_stat"])
            self.assertEqual(r_neg["p_negative"], r_orig["p_negative"])
            self.assertEqual(r_neg["p_negative_uncorrected"], r_orig["p_negative_uncorrected"])
            self.assertNotIn("p_positive", r_neg)
            self.assertNotIn("p_positive_uncorrected", r_neg)

        # Deep copy independence: mutating restricted report cannot leak back.
        neg["node_labels"][0] = 99
        neg["blocks"][0]["p_negative"] = -1.0
        self.assertEqual(raw, raw_snapshot)

    def test_real_cnbs_untested_significant_block_both_directions_and_roundtrips(self):
        # Small 4-ROI two-group cNBS dataset with a strongly positive block (0-0)
        # so testing only the negative tail encounters an untested-significant source block.
        rng = np.random.default_rng(20)
        n = 18
        g1 = rng.normal(0.0, 0.25, (n, 4, 4))
        g2 = rng.normal(0.0, 0.25, (n, 4, 4))
        g1 = 0.5 * (g1 + g1.transpose(0, 2, 1))
        g2 = 0.5 * (g2 + g2.transpose(0, 2, 1))
        g2[:, 0, 1] += 0.9
        g2[:, 1, 0] += 0.9
        idx = np.arange(4)
        g1[:, idx, idx] = 0.0
        g2[:, idx, idx] = 0.0
        labels = [0, 0, 1, 1]

        sep_out = analyze(
            group1=g1, group2=g2, input="z", method="cnbs", net_labels=labels,
            two_tailed=False, n_permutations=63, rng=20, use_mp=False, acceleration=None,
        )
        sep = sep_out.inference
        sep_snapshot = copy.deepcopy(sep.block_report)
        self.assertLessEqual(sep.block_report["blocks"][0]["p_positive"], 0.05)

        for tail in ("negative", "positive"):
            other = "positive" if tail == "negative" else "negative"
            pre_out = analyze(
                group1=g1, group2=g2, input="z", method="cnbs", net_labels=labels,
                tested_tail=tail, n_permutations=63, rng=20, use_mp=False, acceleration=None,
            )
            post_out = sep_out.one_sided(tail)
            for mode, out in (("prespecified", pre_out), ("post-hoc", post_out)):
                res = out.inference
                self.assertEqual(res.tail_policy, f"one-sided-{tail}")
                self.assertEqual(res.tested_tail, tail)
                self.assertEqual(res.tail_selection, mode)
                self.assertIs(out.block_report, res.block_report)
                np.testing.assert_array_equal(res[tail], sep[tail])
                np.testing.assert_array_equal(res[other], np.ones_like(sep[other]))
                np.testing.assert_array_equal(res.stat_signed, sep.stat_signed)
                np.testing.assert_array_equal(res.effect_map(), sep.effect_map())
                np.testing.assert_array_equal(res.stat_positive, sep.stat_positive)
                np.testing.assert_array_equal(res.stat_negative, sep.stat_negative)
                np.testing.assert_array_equal(res.beta, sep.beta)
                self.assertEqual(res.null_max_dist, sep.null_max_dist)

                rep = res.block_report
                self.assertEqual(rep["tail_policy"], f"one-sided-{tail}")
                self.assertEqual(rep["tested_tail"], tail)
                self.assertEqual(rep["tail_selection"], mode)
                self.assertEqual(rep["n_blocks"], sep_snapshot["n_blocks"])
                self.assertEqual(rep["n_blocks_tested"], sep_snapshot["n_blocks_tested"])
                self.assertEqual(rep["correction"], sep_snapshot["correction"])
                self.assertEqual(rep["node_labels"], labels)
                self.assertEqual(len(rep["blocks"]), 3)
                for b_src, b_res in zip(sep_snapshot["blocks"], rep["blocks"], strict=True):
                    self.assertEqual((b_res["network_a"], b_res["network_b"]),
                                     (b_src["network_a"], b_src["network_b"]))
                    self.assertEqual(b_res["n_edges"], b_src["n_edges"])
                    self.assertEqual(b_res["mean_stat"], b_src["mean_stat"])
                    self.assertEqual(b_res[f"p_{tail}"], b_src[f"p_{tail}"])
                    self.assertEqual(b_res[f"p_{tail}_uncorrected"], b_src[f"p_{tail}_uncorrected"])
                    self.assertIsNone(b_res.get(f"p_{other}"))
                    self.assertNotIn(f"p_{other}", b_res)
                    self.assertNotIn(f"p_{other}_uncorrected", b_res)

                # Bundle + pickle roundtrip keeps native_report consistent.
                with tempfile.TemporaryDirectory() as tmp:
                    bpath = Path(tmp) / "bundle.pkl"
                    res.to_bundle().save(bpath)
                    loaded_bundle = ResultBundle.load(bpath)
                    native = loaded_bundle.metadata["native_report"]
                    self.assertEqual(native["tail_policy"], f"one-sided-{tail}")
                    self.assertEqual(native["tested_tail"], tail)
                    self.assertEqual(native["tail_selection"], mode)
                    self.assertEqual(len(native["blocks"]), 3)
                    self.assertTrue(all(f"p_{other}" not in b for b in native["blocks"]))
                    csv_path = Path(tmp) / "edges.csv"
                    res.to_csv(csv_path, alpha=0.05)
                    if tail == "negative":
                        self.assertEqual(len(res.significant_edges(alpha=0.05)), 0)
                    else:
                        edges = res.significant_edges(alpha=0.05)
                        self.assertTrue((edges["tail"] == "positive").all())

                ns = res.node_summary(sort="roi")
                self.assertEqual(ns.attrs["tails"], [tail])
                with self.assertRaises(ValueError):
                    res.node_summary(tail=other)
                with self.assertRaises(ValueError):
                    res.one_sided(other)

        # Original separate result and its block_report were never mutated.
        self.assertEqual(sep.block_report, sep_snapshot)

    def test_legacy_loaded_cnbs_result_not_auto_reinterpreted_until_explicit_restriction(self):
        g1, g2, labels = _planted()
        sep = compute_p_val(
            g1, g2, test_type="two-sample", method="cnbs", net_labels=labels,
            n_permutations=39, rng=0, use_mp=False, two_tailed=False,
        )
        # Simulate an older saved restricted result whose block_report still carried
        # separate-one-sided metadata and lacked tail_selection.
        legacy = pickle.loads(pickle.dumps(sep))
        legacy.__dict__.pop("tail_selection", None)
        legacy.block_report.pop("tested_tail", None)
        legacy_pre = copy.deepcopy(legacy)
        legacy_pre.tested_tail = "negative"
        legacy_pre.__dict__["negative"] = np.ones_like(legacy["negative"])
        reloaded_pre = pickle.loads(pickle.dumps(legacy_pre))
        self.assertEqual(reloaded_pre.block_report["tail_policy"], "separate-one-sided")
        self.assertIn("p_positive", reloaded_pre.block_report["blocks"][0])

        # Reloaded unrestricted legacy separate result stays untouched, while an
        # explicit derived restriction builds a consistent restricted block_report.
        reloaded_sep = pickle.loads(pickle.dumps(legacy))
        self.assertEqual(reloaded_sep.block_report["tail_policy"], "separate-one-sided")
        derived = reloaded_sep.one_sided("negative")
        self.assertEqual(derived.block_report["tail_policy"], "one-sided-negative")
        self.assertEqual(derived.block_report["tested_tail"], "negative")
        self.assertEqual(derived.block_report["tail_selection"], "post-hoc")
        self.assertTrue(all("p_positive" not in b for b in derived.block_report["blocks"]))
        self.assertEqual(reloaded_sep.block_report["tail_policy"], "separate-one-sided")
        self.assertIn("p_positive", reloaded_sep.block_report["blocks"][0])


if __name__ == "__main__":
    unittest.main()
