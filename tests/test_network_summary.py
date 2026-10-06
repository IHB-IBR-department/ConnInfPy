"""Network-pair aggregation: result.network_summary(atlas) / summarize_by_network.

Results are aggregated over pairs of networks from the atlas's per-ROI
``networks`` field (any user-defined partition). The table's numbers back
plot_network_summary and the block-mass agreement in the Open-Close
validation, so these tests pin the counts, the -log10 p mass and the
consistency between the table, the matrix form and the plot.
"""
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

import conninfpy
from conninfpy.analysis import AnalyzeResult
from conninfpy.harmonize import block_mass
from conninfpy.plot import plot_network_summary
from conninfpy.results import InferenceResult, OmnibusInferenceResult, summarize_by_network

NETS = ["A", "A", "A", "B", "B", "C"]          # 3 networks, sizes 3 / 2 / 1


def _sym(p):
    return np.triu(p, 1) + np.triu(p, 1).T + np.eye(len(p))


class TestSummarizeByNetwork(unittest.TestCase):

    def setUp(self):
        n = len(NETS)
        self.pos = _sym(np.full((n, n), 0.5))
        self.neg = _sym(np.full((n, n), 0.5))
        self.pos[0, 1] = self.pos[1, 0] = 0.01      # A—A
        self.pos[0, 3] = self.pos[3, 0] = 0.001     # A—B
        self.neg[3, 4] = self.neg[4, 3] = 0.02      # B—B
        self.table = summarize_by_network({"positive": self.pos, "negative": self.neg}, NETS)

    def row(self, pair):
        return self.table.set_index("network_pair").loc[pair]

    def test_one_row_per_network_pair_with_possible_edges(self):
        # C—C has no possible edge (single ROI) and is omitted
        self.assertEqual(list(self.table["network_pair"]),
                         ["A—A", "A—B", "A—C", "B—B", "B—C"])
        self.assertEqual(self.table["n_edges"].sum(), len(NETS) * (len(NETS) - 1) // 2)
        self.assertEqual(self.row("A—A")["n_edges"], 3)       # within: k(k-1)/2
        self.assertEqual(self.row("A—B")["n_edges"], 6)       # between: ka * kb

    def test_counts_proportions_and_signed_proportion(self):
        self.assertEqual(self.row("A—A")["n_sig_positive"], 1)
        self.assertEqual(self.row("B—B")["n_sig_negative"], 1)
        self.assertAlmostEqual(self.row("A—B")["signed_proportion"], 1 / 6)
        self.assertAlmostEqual(self.row("B—B")["signed_proportion"], -1.0)

    def test_neglog10p_mass(self):
        expected = -np.log10(0.001) + 5 * -np.log10(0.5)
        self.assertAlmostEqual(self.row("A—B")["sum_neglog10p_positive"], expected)
        self.assertAlmostEqual(self.row("A—B")["mean_neglog10p_positive"], expected / 6)

    def test_matrix_form_is_symmetric_in_first_appearance_order(self):
        m = summarize_by_network({"positive": self.pos}, NETS, as_matrix="n_sig_positive")
        self.assertEqual(list(m.index), ["A", "B", "C"])
        np.testing.assert_array_equal(m.to_numpy(), m.to_numpy().T)
        self.assertEqual(m.loc["B", "A"], 1)
        self.assertEqual(m.loc["C", "C"], 0)          # no possible edge → 0

    def test_unknown_matrix_column_lists_choices(self):
        with self.assertRaisesRegex(ValueError, "n_sig_positive"):
            summarize_by_network({"positive": self.pos}, NETS, as_matrix="nope")

    def test_wrong_number_of_labels(self):
        with self.assertRaises(ValueError):
            summarize_by_network({"positive": self.pos}, NETS[:-1])

    def test_counts_agree_with_block_mass(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        bm = block_mass(self.pos, labels, alpha=0.05, return_upper=False)
        m = summarize_by_network({"positive": self.pos}, [str(k) for k in labels],
                                 as_matrix="n_sig_positive")
        np.testing.assert_array_equal(bm, m.to_numpy())


class TestResultMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gen = conninfpy.sim.TopologyDatasetGenerator(n_nodes=30, n_modules=3, seed=1)
        cls.ds = gen.generate("within_module_dense", effect_size=0.35,
                              n_samples=25, time_points=150)
        cls.atlas = conninfpy.AtlasInfo(
            labels=[f"r{i}" for i in range(30)],
            networks=[f"Net{k}" for k in cls.ds.net_labels],
        )
        cls.out = conninfpy.analyze(group1=cls.ds.group1, group2=cls.ds.group2, input="r",
                                    test_type="two-sample", method="tstat",
                                    n_permutations=100, rng=0, use_mp=False)

    def test_analyze_result_and_inference_result_agree(self):
        self.assertIsInstance(self.out, AnalyzeResult)
        a = self.out.network_summary(self.atlas)
        b = self.out.inference.network_summary(self.atlas)
        self.assertTrue(a.equals(b))
        self.assertIn("signed_proportion", a.columns)

    def test_planted_module_dominates(self):
        s = self.out.network_summary(self.atlas).set_index("network_pair")
        planted = self.ds.effect_mask != 0
        nets = np.array([f"Net{k}" for k in self.ds.net_labels])
        module = nets[np.argwhere(planted)[0][0]]
        top = s["n_sig_positive"].idxmax()
        self.assertEqual(top, f"{module}—{module}")

    def test_plot_shows_the_summary_numbers(self):
        ax = plot_network_summary(self.out.inference, atlas=self.atlas)
        shown = np.asarray(ax.images[0].get_array())
        m = self.out.network_summary(self.atlas, as_matrix="signed_proportion")
        np.testing.assert_allclose(shown, m.to_numpy())

    def test_grid_result_needs_param_idx(self):
        grid = conninfpy.analyze(group1=self.ds.group1, group2=self.ds.group2, input="r",
                                 test_type="two-sample", method="tfnbs", e=[0.4, 0.5],
                                 h=[3.0, 2.0], n_permutations=20, rng=0, use_mp=False)
        with self.assertRaisesRegex(ValueError, "param_idx"):
            grid.network_summary(self.atlas)
        self.assertEqual(len(grid.network_summary(self.atlas, param_idx=1)), 6)

    def test_omnibus_result(self):
        n = 30
        p = _sym(np.full((n, n), 0.5))
        p[0, 1] = p[1, 0] = 0.01
        res = OmnibusInferenceResult(p, method="tstat", n_permutations=0, acceleration=None)
        s = res.network_summary(self.atlas)
        self.assertIn("n_sig_omnibus", s.columns)
        self.assertNotIn("signed_proportion", s.columns)
        self.assertEqual(s["n_sig_omnibus"].sum(), 1)

    def test_plain_labels_work_without_an_atlas(self):
        nets = [f"Net{k}" for k in self.ds.net_labels]
        self.assertTrue(self.out.network_summary(nets).equals(
            self.out.network_summary(self.atlas)))


if __name__ == "__main__":
    unittest.main()
