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
from conninfpy._cnbs import block_p_values
from conninfpy.enhance import apply_cnbs
from conninfpy.glm_stats import compute_p_val_glm
from conninfpy.nbs_score import cnbs_block_means, cnbs_blocks, get_cnbs_score

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


if __name__ == "__main__":
    unittest.main()
