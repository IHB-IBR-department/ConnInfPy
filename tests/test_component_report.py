"""NBS component report: ``component_report``, ``component_table``, ``component_id``.

The inference unit of ``method='nbs'`` is the connected component of the
suprathreshold edges, calibrated against the permutation distribution of the
maximum component statistic. These tests check that the report

* lists the components that were tested (an independent scipy labelling is
  the oracle) with their extent / intensity,
* carries each component's p-value from the p-value map that the result
  holds (and, on the two-sample path, from an independently regenerated null
  distribution of the maximum), and
* reaches ``significant_edges`` / CSV as ``component_id`` and survives
  pickle / copy / deepcopy.

The real-data check against the MATLAB NBS v1.2 output lives in
``tests/test_nbs_schizophrenia_reference.py``.
"""
from __future__ import annotations

import copy
import io
import pickle
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from conninfpy import (
    AtlasInfo,
    InferenceResult,
    OmnibusInferenceResult,
    analyze,
    compute_p_val,
    compute_p_val_glm,
    compute_p_val_glm_multi,
)
from conninfpy._shared.rng import resolve_seed
from conninfpy.enhance import apply_nbs
from conninfpy.enhance.nbs import NBSComponents, get_nbs_score, nbs_components
from conninfpy.stats._components import build_component_report
from conninfpy.stats.pairwise import _compute_enhanced_null_dist
from tests.test_serialization import assert_tree_equal

N = 24
TAU = 3.5
B = 300
SEED = 1


# ---------------------------------------------------------------------------
# Oracle and fixtures
# ---------------------------------------------------------------------------

def oracle_components(t, tau, stat_type):
    """Independent labelling: (nodes, n_edges, statistic) per component, ranked."""
    mask = np.round(t, 10) > tau
    np.fill_diagonal(mask, False)
    n, lab = connected_components(csr_matrix(mask | mask.T), directed=False)
    found = []
    for c in range(n):
        nodes = np.flatnonzero(lab == c)
        if nodes.size < 2:
            continue
        sub = np.triu(mask[np.ix_(nodes, nodes)], 1)
        n_edges = int(sub.sum())
        values = t[np.ix_(nodes, nodes)][sub]
        stat = float(n_edges) if stat_type == "extent" else float(np.sum(values - tau))
        found.append((tuple(int(v) for v in nodes), n_edges, stat))
    found.sort(key=lambda c: (-c[2], -c[1], c[0][0]))
    return found


def _sym(a):
    a = (a + a.transpose(0, 2, 1)) / 2
    idx = np.arange(a.shape[-1])
    a[:, idx, idx] = 0.0
    return a


def planted_groups(seed=7, n=12):
    """Group 2 > group 1 on a 6-node and a 3-node clique; group 1 > group 2 on a 4-node clique."""
    rng = np.random.default_rng(seed)
    g1 = _sym(rng.normal(size=(n, N, N)))
    g2 = _sym(rng.normal(size=(n, N, N)))

    def clique(nodes, shift, g):
        for i in nodes:
            for j in nodes:
                if i != j:
                    g[:, i, j] += shift

    clique(range(0, 6), 1.6, g2)
    clique(range(8, 11), 1.6, g2)
    clique(range(14, 18), 1.6, g1)
    return g1, g2


def two_sample(g1, g2, **kw):
    args = dict(test_type="two-sample", var_equal=True, method="nbs", threshold=TAU,
                nbs_stat="extent", n_permutations=B, rng=SEED, use_mp=False,
                acceleration=None, two_tailed=False)
    args.update(kw)
    return compute_p_val(g1, g2, **args)


def glm_indicator(g1, g2, **kw):
    Y = np.concatenate([g1, g2])
    indicator = np.r_[np.zeros(len(g1)), np.ones(len(g2))]
    args = dict(interest=indicator, method="nbs", threshold=TAU, nbs_stat="extent",
                n_permutations=B, rng=SEED, use_mp=False, acceleration=None,
                two_tailed=False)
    args.update(kw)
    return compute_p_val_glm(Y, **args)


def iu():
    return np.triu_indices(N, k=1)


class ReportChecks(unittest.TestCase):
    """Shared assertions on a result's component report."""

    def check_against_oracle(self, res, stat_type, tails=("positive", "negative")):
        report = res.component_report
        self.assertIsNotNone(report)
        self.assertEqual(report["unit"], "component")
        self.assertEqual(report["stat_type"], stat_type)
        self.assertEqual(res.inference_unit, "component")
        stat_maps = {"positive": res.stat_positive, "negative": res.stat_negative}
        for tail in tails:
            rows = [r for r in report["components"] if r["tail"] == tail]
            expected = oracle_components(stat_maps[tail], report["threshold"], stat_type)
            self.assertEqual(report["n_components"][tail], len(expected))
            self.assertEqual([r["component_id"] for r in rows], list(range(1, len(expected) + 1)))
            labels = report["labels"][tail]
            p_map = np.asarray(res[tail])
            for row, (nodes, n_edges, stat) in zip(rows, expected):
                self.assertEqual(tuple(row["nodes"]), nodes)
                self.assertEqual(row["n_nodes"], len(nodes))
                self.assertEqual(row["n_edges"], n_edges)
                self.assertAlmostEqual(row["statistic"], stat, places=9)
                # the labels mark exactly the component's edges ...
                members = labels == row["component_id"]
                self.assertEqual(int(np.triu(members, 1).sum()), n_edges)
                self.assertTrue(np.array_equal(members, members.T))
                self.assertTrue(set(np.flatnonzero(members.any(axis=0))) == set(nodes))
                # ... and every one of them carries the reported p-value
                np.testing.assert_array_equal(np.unique(p_map[members]), [row["p_fwer"]])
            # edges outside any component are not claimed
            self.assertEqual(int((labels > 0).sum()), 2 * sum(r["n_edges"] for r in rows))

    def check_null_p(self, res, g1, g2, stat_type, two_tailed=False, var_equal=True):
        """p of every component = (1 + #{null max >= statistic}) / (B + 1)."""
        null = _compute_enhanced_null_dist(
            g1, g2, "two-sample", enhance_func=apply_nbs,
            enhance_kwargs={"threshold": TAU, "nbs_stat": stat_type},
            n_permutations=B, random_state=resolve_seed(SEED), use_mp=False,
            var_equal=var_equal,
        )
        if two_tailed:
            joint = np.maximum(null["positive"], null["negative"])
            null = {"positive": joint, "negative": joint}
        for row in res.component_report["components"]:
            count = int(np.sum(null[row["tail"]] >= row["statistic"]))
            self.assertAlmostEqual(row["p_fwer"], (count + 1.0) / (B + 1.0), places=12)


# ---------------------------------------------------------------------------
# The component function
# ---------------------------------------------------------------------------

class TestNbsComponentsFunction(unittest.TestCase):

    @staticmethod
    def _map():
        """Three components: {0,1,2} (2 edges), {5,6,7,8} (3 edges), {10,11} (1 edge)."""
        t = np.zeros((14, 14))
        for (i, j), v in {(0, 1): 4.0, (1, 2): 5.0,
                          (5, 6): 4.5, (6, 7): 3.2, (6, 8): 3.3,
                          (10, 11): 9.0,
                          (3, 4): 3.0}.items():      # at the threshold: not above it
            t[i, j] = t[j, i] = v
        return t

    def test_extent_ranks_components_and_numbers_edges(self):
        found = nbs_components(self._map(), threshold=3.0, stat_type="extent")
        self.assertIsInstance(found, NBSComponents)
        self.assertEqual(found.n_components, 3)
        np.testing.assert_array_equal(found.n_edges, [3, 2, 1])
        np.testing.assert_array_equal(found.statistic, [3.0, 2.0, 1.0])
        np.testing.assert_array_equal(found.n_nodes, [4, 3, 2])
        self.assertEqual([n.tolist() for n in found.nodes], [[5, 6, 7, 8], [0, 1, 2], [10, 11]])
        # edge numbering, both triangles; the edge at the threshold is in no component
        self.assertEqual(found.labels[6, 7], 1)
        self.assertEqual(found.labels[7, 6], 1)
        self.assertEqual(found.labels[0, 1], 2)
        self.assertEqual(found.labels[10, 11], 3)
        self.assertEqual(found.labels[3, 4], 0)
        self.assertEqual(int((found.labels > 0).sum()), 2 * 6)
        self.assertTrue(np.all(np.diag(found.labels) == 0))

    def test_intensity_is_mass_above_the_threshold(self):
        t = self._map()
        found = nbs_components(t, threshold=3.0, stat_type="intensity")
        expected = {(5, 6, 7, 8): (4.5 - 3) + (3.2 - 3) + (3.3 - 3),
                    (0, 1, 2): (4.0 - 3) + (5.0 - 3),
                    (10, 11): 9.0 - 3.0}
        by_nodes = {tuple(n.tolist()): s for n, s in zip(found.nodes, found.statistic)}
        for nodes, stat in expected.items():
            self.assertAlmostEqual(by_nodes[nodes], stat, places=12)
        # ranking by intensity (6.0, 3.0, 2.0) is the reverse of the ranking by extent
        self.assertEqual([n.tolist() for n in found.nodes],
                         [[10, 11], [0, 1, 2], [5, 6, 7, 8]])
        np.testing.assert_allclose(found.statistic, [6.0, 3.0, 2.0], atol=1e-12)

    def test_an_isolated_edge_is_a_component(self):
        found = nbs_components(self._map(), threshold=3.0)
        last = found.n_components - 1
        self.assertEqual(int(found.n_edges[last]), 1)
        self.assertEqual(int(found.n_nodes[last]), 2)

    def test_no_edge_above_the_threshold(self):
        t = self._map()
        found = nbs_components(t, threshold=10.0)
        self.assertEqual(found.n_components, 0)
        self.assertEqual(found.labels.shape, t.shape)
        self.assertFalse(found.labels.any())
        self.assertEqual(found.nodes, ())

    def test_the_components_are_the_ones_get_nbs_score_scores(self):
        t = self._map()
        for stat_type in ("extent", "intensity"):
            found = nbs_components(t, threshold=3.0, stat_type=stat_type)
            score = get_nbs_score(t, threshold=3.0, stat_type=stat_type)
            for k in range(1, found.n_components + 1):
                members = found.labels == k
                np.testing.assert_allclose(score[members], found.statistic[k - 1])
            self.assertTrue(np.all(score[found.labels == 0] == 0))

    def test_ties_are_broken_deterministically(self):
        t = np.zeros((8, 8))
        for i, j in ((4, 5), (0, 1)):          # two single edges with the same statistic
            t[i, j] = t[j, i] = 5.0
        found = nbs_components(t, threshold=3.0, stat_type="extent")
        self.assertEqual([n.tolist() for n in found.nodes], [[0, 1], [4, 5]])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            nbs_components(np.zeros((3, 4)), 3.0)
        with self.assertRaises(ValueError):
            nbs_components(np.zeros((4, 4)), 3.0, stat_type="mass")


# ---------------------------------------------------------------------------
# Pipelines: two-sample and GLM, extent and intensity, both tails
# ---------------------------------------------------------------------------

class TestPairwisePath(ReportChecks):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()
        cls.results = {s: two_sample(cls.g1, cls.g2, nbs_stat=s) for s in ("extent", "intensity")}

    def test_synthetic_has_several_components_in_both_tails(self):
        table = self.results["extent"].component_table()
        self.assertEqual(table["tail"].tolist(), ["positive", "positive", "negative"])
        self.assertEqual(table["component_id"].tolist(), [1, 2, 1])
        self.assertEqual(table["n_edges"].tolist(), [15, 3, 6])
        self.assertEqual(table["n_nodes"].tolist(), [6, 3, 4])
        self.assertEqual(table["nodes"].tolist(), [list(range(6)), [8, 9, 10], [14, 15, 16, 17]])
        # different components, different statistics: p comes from the null of the maximum
        self.assertGreater(len(set(table["p_fwer"])), 1)

    def test_components_match_the_oracle(self):
        for stat_type, res in self.results.items():
            with self.subTest(stat=stat_type):
                self.check_against_oracle(res, stat_type)

    def test_p_values_come_from_the_null_distribution_of_the_maximum(self):
        for stat_type, res in self.results.items():
            with self.subTest(stat=stat_type):
                self.check_null_p(res, self.g1, self.g2, stat_type)

    def test_intensity_statistic_is_the_mass_above_the_threshold(self):
        res = self.results["intensity"]
        t = res.stat_positive
        first = res.component_report["components"][0]
        nodes = np.array(first["nodes"])
        sub = t[np.ix_(nodes, nodes)]
        expected = float(np.sum(sub[sub > TAU] - TAU)) / 2     # both triangles counted
        self.assertAlmostEqual(first["statistic"], expected, places=9)
        self.assertNotEqual(first["statistic"], first["n_edges"])

    def test_two_tailed_uses_the_joint_null(self):
        res = two_sample(self.g1, self.g2, two_tailed=True)
        self.assertEqual(res.component_report["tail_policy"], "two-sided")
        self.check_against_oracle(res, "extent")
        self.check_null_p(res, self.g1, self.g2, "extent", two_tailed=True)

    def test_separate_tails_policy_is_recorded(self):
        report = self.results["extent"].component_report
        self.assertEqual(report["tail_policy"], "separate-one-sided")
        self.assertEqual(report["correction"], "permutation_max")
        self.assertEqual(report["n_permutations"], B)
        self.assertEqual(report["threshold"], TAU)
        self.assertEqual(report["edge_statistic"], "tstat")

    def test_one_sample_and_paired_paths(self):
        rng = np.random.default_rng(3)
        base = _sym(rng.normal(size=(14, N, N)))
        shifted = base + 0.5 * _sym(rng.normal(size=(14, N, N)))    # not a constant difference
        for nodes, shift in ((range(0, 5), 1.3), (range(10, 13), 1.3)):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        shifted[:, i, j] += shift
        paired = compute_p_val(base, shifted, test_type="paired", method="nbs",
                               threshold=TAU, n_permutations=B, rng=SEED, use_mp=False,
                               acceleration=None)
        self.check_against_oracle(paired, "extent")
        self.assertGreaterEqual(paired.component_report["n_components"]["positive"], 2)
        one = compute_p_val(shifted - base, test_type="one-sample", method="nbs",
                            threshold=TAU, n_permutations=B, rng=SEED, use_mp=False,
                            acceleration=None)
        self.check_against_oracle(one, "extent")


class TestGLMPath(ReportChecks):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()
        cls.results = {s: glm_indicator(cls.g1, cls.g2, nbs_stat=s) for s in ("extent", "intensity")}

    def test_components_match_the_oracle(self):
        for stat_type, res in self.results.items():
            with self.subTest(stat=stat_type):
                self.check_against_oracle(res, stat_type)
                self.assertGreaterEqual(len(res.component_report["components"]), 3)

    def test_glm_t_with_a_group_indicator_is_the_pooled_two_sample_t(self):
        """Same statistic map, hence the same components, as the Student two-sample path."""
        glm = self.results["extent"].component_report
        pair = two_sample(self.g1, self.g2).component_report
        for tail in ("positive", "negative"):
            np.testing.assert_array_equal(glm["labels"][tail], pair["labels"][tail])
        self.assertEqual([(r["tail"], r["component_id"], r["n_edges"], r["nodes"])
                          for r in glm["components"]],
                         [(r["tail"], r["component_id"], r["n_edges"], r["nodes"])
                          for r in pair["components"]])

    def test_one_sided_glm_keeps_the_tested_tail(self):
        res = glm_indicator(self.g1, self.g2).one_sided("positive")
        self.assertEqual({r["tail"] for r in res.component_report["components"]}, {"positive"})

    def test_beta_statistic_thresholds_the_coefficient(self):
        with self.assertWarns(UserWarning):         # beta thresholds are on the coefficient scale
            res = glm_indicator(self.g1, self.g2, stat_type="beta", threshold=1.0)
        self.assertEqual(res.component_report["edge_statistic"], "beta")
        self.check_against_oracle(res, "extent")

    def test_multi_contrast_path_reports_each_contrast(self):
        Y = np.concatenate([self.g1, self.g2])
        group = np.r_[np.zeros(len(self.g1)), np.ones(len(self.g2))]
        X = np.column_stack([np.ones(len(Y)), group])
        out = compute_p_val_glm_multi(
            Y, X, {"g2_gt_g1": np.array([0.0, 1.0]), "g1_gt_g2": np.array([0.0, -1.0])},
            method="nbs", threshold=TAU, n_permutations=B, rng=SEED, use_mp=False,
            acceleration=None,
        )
        for name, res in out.items():
            with self.subTest(contrast=name):
                self.check_against_oracle(res, "extent")
        # the opposite contrast swaps the tails
        a, b = out["g2_gt_g1"].component_report, out["g1_gt_g2"].component_report
        self.assertEqual(a["n_components"]["positive"], b["n_components"]["negative"])
        self.assertEqual(a["n_components"]["negative"], b["n_components"]["positive"])


class TestOmnibusPath(ReportChecks):

    def test_f_test_reports_components_of_the_omnibus_map(self):
        g1, g2 = planted_groups()
        Y = np.concatenate([g1, g2])
        group = np.r_[np.zeros(len(g1)), np.ones(len(g2))]
        rng = np.random.default_rng(2)
        X = np.column_stack([np.ones(len(Y)), group, rng.normal(size=len(Y))])
        res = compute_p_val_glm(Y, design_matrix=X, contrast=np.array([[0, 1, 0], [0, 0, 1.0]]),
                                stat_type="fstat", method="nbs", threshold=8.0,
                                n_permutations=B, rng=SEED, use_mp=False, acceleration=None)
        self.assertIsInstance(res, OmnibusInferenceResult)
        report = res.component_report
        self.assertEqual(report["tail_policy"], "omnibus")
        self.assertEqual(report["edge_statistic"], "fstat")
        expected = oracle_components(res.stat_omnibus, 8.0, "extent")
        rows = report["components"]
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual([tuple(r["nodes"]) for r in rows], [c[0] for c in expected])
        p_map = np.asarray(res["omnibus"])
        for row in rows:
            self.assertEqual(row["tail"], "omnibus")
            np.testing.assert_array_equal(
                np.unique(p_map[report["labels"]["omnibus"] == row["component_id"]]), [row["p_fwer"]])
        table = res.component_table()
        self.assertEqual(len(table), len(rows))
        edges = res.significant_edges(alpha=1.0, include_nonsig=True)
        self.assertIn("component_id", edges.columns)
        self.assertEqual(int(edges["component_id"].notna().sum()), sum(r["n_edges"] for r in rows))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(ReportChecks):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()

    def test_no_component_above_the_threshold(self):
        res = two_sample(self.g1, self.g2, threshold=50.0)
        report = res.component_report
        self.assertEqual(report["n_components"], {"positive": 0, "negative": 0})
        self.assertEqual(report["components"], [])
        table = res.component_table()
        self.assertTrue(table.empty)
        self.assertIn("p_fwer", table.columns)
        edges = res.significant_edges(include_nonsig=True)
        self.assertIn("component_id", edges.columns)
        self.assertTrue(edges["component_id"].isna().all())
        self.assertTrue(res.significant_edges().empty)

    def test_components_without_significance_are_listed(self):
        res = two_sample(self.g1, self.g2, threshold=2.4)       # noise clusters appear
        table = res.component_table()
        self.assertGreater(len(table), 3)
        self.assertTrue((~table["significant"]).any())
        self.assertTrue(table["significant"].any())
        flags = (table["p_fwer"] <= 0.05)
        self.assertTrue(np.array_equal(table["significant"].to_numpy(), flags.to_numpy()))
        only = res.component_table(significant_only=True)
        self.assertTrue(only["significant"].all())
        self.assertEqual(len(only), int(table["significant"].sum()))
        strict = res.component_table(alpha=1e-9)
        self.assertFalse(strict["significant"].any())
        self.assertEqual(strict.attrs["alpha"], 1e-9)
        # the edge table: default keeps significant edges; include_nonsig also lists the rest
        sig = res.significant_edges()
        self.assertEqual(set(sig["component_id"].dropna()) <= set(only["component_id"]), True)
        full = res.significant_edges(include_nonsig=True)
        in_components = full[full["component_id"].notna()]
        self.assertEqual(len(in_components), int(table["n_edges"].sum()))
        # a non-significant component's edges: p above alpha, id present
        weak = table[~table["significant"]].iloc[0]
        rows = in_components[(in_components["component_tail"] == weak["tail"])
                             & (in_components["component_id"] == weak["component_id"])]
        self.assertEqual(len(rows), weak["n_edges"])
        self.assertTrue((rows["p_min"] > 0.05).all())

    def test_isolated_edges_are_components(self):
        res = two_sample(self.g1, self.g2, threshold=2.4)
        table = res.component_table()
        singles = table[table["n_edges"] == 1]
        self.assertGreater(len(singles), 0)
        self.assertTrue((singles["n_nodes"] == 2).all())
        self.check_against_oracle(res, "extent")

    def test_one_sided_lists_only_the_tested_tail_with_unchanged_ids(self):
        both = two_sample(self.g1, self.g2)
        neg = both.one_sided("negative")
        pos = both.one_sided("positive")
        self.assertEqual(neg.component_report["tail_policy"], "one-sided-negative")
        self.assertEqual(neg.component_report["tested_tail"], "negative")
        self.assertEqual(neg.component_table()["tail"].unique().tolist(), ["negative"])
        self.assertEqual(pos.component_table()["tail"].unique().tolist(), ["positive"])
        self.assertEqual(neg.component_table()["component_id"].tolist(), [1])
        # an untested tail has no table rows and no component ids in the edge table
        self.assertTrue(neg.component_table(tail="positive").empty)
        edges = neg.significant_edges(alpha=1.0, include_nonsig=True)
        in_comp = edges[edges["component_id"].notna()]
        self.assertEqual(set(in_comp["component_tail"]), {"negative"})
        self.assertEqual(len(in_comp), 6)
        # the one-sided p of the component is the p of its edges
        row = neg.component_report["components"][0]
        labels = neg.component_report["labels"]["negative"]
        np.testing.assert_array_equal(np.unique(neg["negative"][labels == 1]), [row["p_fwer"]])
        # the result it came from is untouched
        self.assertEqual(len(both.component_report["components"]), 3)

    def test_available_case_with_missing_values(self):
        g1 = self.g1.copy()
        for s in range(5):                      # ROI 20 missing in 5 of 12 subjects (< 80% coverage)
            g1[s, 20, :] = np.nan
            g1[s, :, 20] = np.nan
        res = compute_p_val(g1, self.g2, test_type="two-sample", var_equal=True, method="nbs",
                            threshold=TAU, n_permutations=B, rng=SEED, use_mp=False,
                            acceleration=None, missing="available_case", two_tailed=False)
        self.assertTrue(np.isnan(res["positive"][20, 0]))
        self.assertGreater(len(res.missing_report["untested_edges"]), 0)
        report = res.component_report
        self.assertGreater(len(report["components"]), 0)
        # untested edges are in no component, and no component has a NaN p-value
        self.assertFalse(report["labels"]["positive"][20].any())
        self.assertFalse(report["labels"]["negative"][:, 20].any())
        self.assertTrue(all(np.isfinite(r["p_fwer"]) for r in report["components"]))
        self.check_against_oracle(res, "extent")
        # the edge table shows the ids and leaves untested edges out
        edges = res.significant_edges(alpha=1.0, include_nonsig=True)
        self.assertEqual(int(edges["component_id"].notna().sum()),
                         sum(r["n_edges"] for r in report["components"]))

    def test_available_case_glm_path(self):
        Y = np.concatenate([self.g1, self.g2])
        for s in range(6):
            Y[s, 20, :] = np.nan
            Y[s, :, 20] = np.nan
        indicator = np.r_[np.zeros(12), np.ones(12)]
        res = compute_p_val_glm(Y, interest=indicator, method="nbs", threshold=TAU,
                                n_permutations=B, rng=SEED, use_mp=False, acceleration=None,
                                missing="available_case", two_tailed=False)
        self.assertTrue(np.isnan(res["positive"][20, 0]))
        self.assertGreater(len(res.component_report["components"]), 0)
        self.assertFalse(res.component_report["labels"]["positive"][20].any())
        self.check_against_oracle(res, "extent")

    def test_available_case_f_test(self):
        Y = np.concatenate([self.g1, self.g2])
        for s in range(6):
            Y[s, 20, :] = np.nan
            Y[s, :, 20] = np.nan
        rng = np.random.default_rng(2)
        X = np.column_stack([np.ones(24), np.r_[np.zeros(12), np.ones(12)], rng.normal(size=24)])
        res = compute_p_val_glm(Y, design_matrix=X, contrast=np.array([[0, 1, 0], [0, 0, 1.0]]),
                                stat_type="fstat", method="nbs", threshold=8.0,
                                n_permutations=B, rng=SEED, use_mp=False, acceleration=None,
                                missing="available_case")
        self.assertIsInstance(res, OmnibusInferenceResult)
        self.assertTrue(np.isnan(res["omnibus"][20, 0]))
        report = res.component_report
        self.assertGreater(len(report["components"]), 0)
        self.assertFalse(report["labels"]["omnibus"][20].any())
        self.assertTrue(all(np.isfinite(r["p_fwer"]) for r in report["components"]))
        expected = oracle_components(res.stat_omnibus, 8.0, "extent")
        self.assertEqual([tuple(r["nodes"]) for r in report["components"]], [c[0] for c in expected])

    def test_edge_mask_family_limits_components_to_the_family(self):
        mask = np.zeros((N, N), dtype=bool)
        keep = list(range(0, 10)) + list(range(14, 18))      # drops the 3-node clique's partner edges
        mask[np.ix_(keep, keep)] = True
        np.fill_diagonal(mask, False)
        res = two_sample(self.g1, self.g2, edge_mask=mask)
        report = res.component_report
        for tail in ("positive", "negative"):
            self.assertFalse(report["labels"][tail][~mask].any())
        self.assertTrue(np.isnan(res["positive"][~mask & ~np.eye(N, dtype=bool)]).all())
        # components are those of the masked statistic map
        for tail, stat in (("positive", res.stat_positive), ("negative", res.stat_negative)):
            expected = oracle_components(np.where(mask, stat, 0.0), TAU, "extent")
            rows = [r for r in report["components"] if r["tail"] == tail]
            self.assertEqual([tuple(r["nodes"]) for r in rows], [c[0] for c in expected])
            p_map = np.asarray(res[tail])
            for r in rows:
                members = report["labels"][tail] == r["component_id"]
                np.testing.assert_array_equal(np.unique(p_map[members]), [r["p_fwer"]])

    def test_acceleration_does_not_change_the_agreement_with_the_map(self):
        res = two_sample(self.g1, self.g2, acceleration="gpd", n_permutations=400)
        self.check_against_oracle(res, "extent")

    def test_other_methods_have_no_component_report(self):
        for method, extra in (("tstat", {}), ("tfnbs", {"n": 8})):
            with self.subTest(method=method):
                res = two_sample(self.g1, self.g2, method=method, n_permutations=30,
                                 threshold=TAU, nbs_stat="extent", **extra) \
                    if method == "tfnbs" else compute_p_val(
                        self.g1, self.g2, test_type="two-sample", method=method,
                        n_permutations=30, rng=SEED, use_mp=False, acceleration=None)
                self.assertIsNone(res.component_report)
                self.assertNotIn("component_id", res.significant_edges(include_nonsig=True).columns)
                with self.assertRaises(ValueError):
                    res.component_table()

    def test_table_arguments_are_validated(self):
        res = two_sample(self.g1, self.g2)
        with self.assertRaises(ValueError):
            res.component_table(tail="up")
        with self.assertRaises(ValueError):
            res.component_table(atlas=_atlas(N + 1))


# ---------------------------------------------------------------------------
# The invariant behind p_fwer: one p-value per component
# ---------------------------------------------------------------------------

class TestComponentPInvariant(unittest.TestCase):

    @staticmethod
    def _maps():
        """Two positive components ({0,1,2}: 2 edges; {5,6}: 1 edge) and one negative ({8,9})."""
        n = 12
        pos, neg = np.zeros((n, n)), np.zeros((n, n))
        for (i, j), v in {(0, 1): 4.0, (1, 2): 5.0, (5, 6): 6.0}.items():
            pos[i, j] = pos[j, i] = v
        neg[8, 9] = neg[9, 8] = 4.0
        p_pos, p_neg = np.ones((n, n)), np.ones((n, n))
        for i, j in ((0, 1), (1, 2)):
            p_pos[i, j] = p_pos[j, i] = 0.03
        p_pos[5, 6] = p_pos[6, 5] = 0.2
        p_neg[8, 9] = p_neg[9, 8] = 0.04
        return {"positive": pos, "negative": neg}, {"positive": p_pos, "negative": p_neg}

    def _report(self, stat, p):
        return build_component_report(stat, p, method="nbs", threshold=3.0, nbs_stat="extent",
                                      n_permutations=99)

    def test_consistent_map_gives_the_p_values_without_warning(self):
        stat, p = self._maps()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = self._report(stat, p)
        got = {(r["tail"], r["component_id"]): r["p_fwer"] for r in report["components"]}
        self.assertEqual(got, {("positive", 1): 0.03, ("positive", 2): 0.2, ("negative", 1): 0.04})

    def test_float_rounding_between_edges_is_tolerated(self):
        stat, p = self._maps()
        p["positive"][1, 2] = p["positive"][2, 1] = 0.03 * (1 + 1e-14)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = self._report(stat, p)
        self.assertAlmostEqual(report["components"][0]["p_fwer"], 0.03, places=12)

    def test_edges_with_different_p_warn_and_give_nan(self):
        stat, p = self._maps()
        p["positive"][1, 2] = p["positive"][2, 1] = 0.5       # corrupts component 1 only
        with self.assertWarnsRegex(RuntimeWarning, r"component 1 of the positive tail"):
            report = self._report(stat, p)
        got = {(r["tail"], r["component_id"]): r["p_fwer"] for r in report["components"]}
        self.assertTrue(np.isnan(got[("positive", 1)]))
        self.assertEqual(got[("positive", 2)], 0.2)           # the others are untouched
        self.assertEqual(got[("negative", 1)], 0.04)

    def test_some_nan_and_some_not_warns_but_all_nan_is_untested(self):
        stat, p = self._maps()
        p["negative"][8, 9] = np.nan                           # a one-edge component: all NaN
        p["positive"][0, 1] = p["positive"][1, 0] = np.nan     # two edges, one NaN: ambiguous
        with self.assertWarnsRegex(RuntimeWarning, r"component 1 of the positive tail") as caught:
            report = self._report(stat, p)
        self.assertEqual(len(caught.warnings), 1)              # the all-NaN component is silent
        got = {(r["tail"], r["component_id"]): r["p_fwer"] for r in report["components"]}
        self.assertTrue(np.isnan(got[("positive", 1)]))
        self.assertTrue(np.isnan(got[("negative", 1)]))
        self.assertEqual(got[("positive", 2)], 0.2)

    def test_nan_p_is_not_significant_in_the_table(self):
        stat, p = self._maps()
        p["positive"][1, 2] = p["positive"][2, 1] = 0.5
        with self.assertWarns(RuntimeWarning):
            report = self._report(stat, p)
        from conninfpy.stats._components import component_table
        table = component_table(report, alpha=0.05)
        row = table[(table["tail"] == "positive") & (table["component_id"] == 1)].iloc[0]
        self.assertTrue(np.isnan(row["p_fwer"]))
        self.assertFalse(row["significant"])
        self.assertEqual(len(component_table(report, alpha=0.05, significant_only=True)), 1)

    def test_no_warning_on_the_real_paths(self):
        g1, g2 = planted_groups()
        Y = np.concatenate([g1, g2])
        indicator = np.r_[np.zeros(12), np.ones(12)]
        Yna = Y.copy()
        for s in range(6):
            Yna[s, 20, :] = np.nan
            Yna[s, :, 20] = np.nan
        mask = np.ones((N, N), dtype=bool)
        mask[20:, :] = mask[:, 20:] = False
        X = np.column_stack([np.ones(24), indicator, np.random.default_rng(2).normal(size=24)])
        runs = {
            "two-sample": lambda: two_sample(g1, g2),
            "two-sample two-tailed": lambda: two_sample(g1, g2, two_tailed=True),
            "intensity": lambda: two_sample(g1, g2, nbs_stat="intensity"),
            "edge_mask": lambda: two_sample(g1, g2, edge_mask=mask),
            "gpd": lambda: two_sample(g1, g2, acceleration="gpd", n_permutations=400),
            "glm": lambda: glm_indicator(g1, g2),
            "glm available_case": lambda: compute_p_val_glm(
                Yna, interest=indicator, method="nbs", threshold=TAU, n_permutations=B, rng=SEED,
                use_mp=False, acceleration=None, missing="available_case"),
            "pairwise available_case": lambda: compute_p_val(
                Yna[:12], g2, test_type="two-sample", var_equal=True, method="nbs", threshold=TAU,
                n_permutations=B, rng=SEED, use_mp=False, acceleration=None,
                missing="available_case"),
            "F": lambda: compute_p_val_glm(
                Y, design_matrix=X, contrast=np.array([[0, 1, 0], [0, 0, 1.0]]), stat_type="fstat",
                method="nbs", threshold=8.0, n_permutations=B, rng=SEED, use_mp=False,
                acceleration=None),
        }
        for name, run in runs.items():
            with self.subTest(path=name):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    res = run()
                self.assertGreater(len(res.component_report["components"]), 0)
                self.assertFalse(any(np.isnan(r["p_fwer"]) for r in res.component_report["components"]))


# ---------------------------------------------------------------------------
# Atlas annotation
# ---------------------------------------------------------------------------

def _atlas(n, networks=True):
    labels = [f"roi{i}" for i in range(n)]
    nets = [("Vis" if i < 3 else "DMN" if i < 8 else "SMN") for i in range(n)] if networks else [""] * n
    return AtlasInfo(labels=labels, networks=nets)


class TestAtlasAnnotation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()
        cls.res = two_sample(cls.g1, cls.g2)

    def test_no_atlas_means_no_roi_names_or_networks(self):
        cols = set(self.res.component_table().columns)
        self.assertEqual(cols, {"tail", "component_id", "n_edges", "n_nodes", "statistic",
                                "statistic_type", "p_fwer", "significant", "nodes"})

    def test_roi_names_and_network_counts(self):
        table = self.res.component_table(_atlas(N))
        first = table.iloc[0]                       # nodes 0..5: Vis x3, DMN x3
        self.assertEqual(first["roi_names"], [f"roi{i}" for i in range(6)])
        self.assertEqual(first["networks"], ["DMN (3)", "Vis (3)"])
        # 15 edges: Vis-Vis 3, DMN-DMN 3, Vis-DMN 9
        self.assertEqual(first["network_pairs"], ["DMN—Vis (9)", "DMN—DMN (3)", "Vis—Vis (3)"])
        counts = sum(int(s.rsplit("(", 1)[1].rstrip(")")) for s in first["network_pairs"])
        self.assertEqual(counts, first["n_edges"])
        second = table.iloc[1]                      # nodes 8, 9, 10: all SMN
        self.assertEqual(second["networks"], ["SMN (3)"])
        self.assertEqual(second["network_pairs"], ["SMN—SMN (3)"])

    def test_atlas_without_networks_adds_names_only(self):
        table = self.res.component_table(_atlas(N, networks=False))
        self.assertIn("roi_names", table.columns)
        self.assertNotIn("networks", table.columns)
        self.assertNotIn("network_pairs", table.columns)

    def test_unknown_only_networks_are_not_reported_as_networks(self):
        atlas = AtlasInfo(labels=[f"r{i}" for i in range(N)], networks=["Unknown"] * N)
        self.assertNotIn("networks", self.res.component_table(atlas).columns)

    def test_edge_table_atlas_columns_still_work_with_component_id(self):
        edges = self.res.significant_edges(atlas=_atlas(N))
        self.assertIn("component_id", edges.columns)
        self.assertIn("network_pair", edges.columns)
        cols = list(edges.columns)
        self.assertEqual(cols[cols.index("tail") + 1:cols.index("tail") + 3],
                         ["component_tail", "component_id"])


# ---------------------------------------------------------------------------
# component_id in significant_edges and CSV
# ---------------------------------------------------------------------------

class TestEdgeTableAndCsv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()
        cls.res = two_sample(cls.g1, cls.g2)

    def test_component_ids_in_significant_edges(self):
        edges = self.res.significant_edges()
        table = self.res.component_table()
        self.assertEqual(len(edges), int(table["n_edges"].sum()))
        counts = edges.groupby(["component_tail", "component_id"]).size()
        for _, row in table.iterrows():
            self.assertEqual(int(counts[(row["tail"], row["component_id"])]), row["n_edges"])
        # the edges of a component are its nodes' pairs, with its p-value
        first = edges[(edges["component_tail"] == "positive") & (edges["component_id"] == 1)]
        self.assertEqual(set(first["roi_i"]) | set(first["roi_j"]), set(range(6)))
        self.assertEqual(first["p_min"].nunique(), 1)
        self.assertAlmostEqual(first["p_min"].iloc[0], table.iloc[0]["p_fwer"], places=12)
        # the key of the report: (tail, component_id) agrees with the `tail` column of a hit
        self.assertTrue((edges["tail"] == edges["component_tail"]).all())

    def test_component_id_is_a_nullable_integer_column(self):
        full = self.res.significant_edges(include_nonsig=True)
        self.assertEqual(str(full["component_id"].dtype), "Int64")
        self.assertEqual(int(full["component_id"].isna().sum()),
                         len(full) - int(self.res.component_table()["n_edges"].sum()))
        self.assertTrue((full.loc[full["component_id"].isna(), "component_tail"] == "").all())

    def test_csv_export_carries_component_id(self):
        buf = io.StringIO()
        self.res.to_csv(buf)
        back = pd.read_csv(io.StringIO(buf.getvalue()))
        self.assertIn("component_id", back.columns)
        self.assertIn("component_tail", back.columns)
        edges = self.res.significant_edges()
        np.testing.assert_array_equal(back["component_id"].to_numpy(), edges["component_id"].to_numpy(dtype=int))
        self.assertEqual(back["component_tail"].tolist(), edges["component_tail"].tolist())
        # include_nonsig: edges in no component leave the cell empty
        buf = io.StringIO()
        self.res.to_csv(buf, include_nonsig=True)
        back = pd.read_csv(io.StringIO(buf.getvalue()))
        self.assertTrue(back["component_id"].isna().any())
        self.assertEqual(int(back["component_id"].notna().sum()),
                         int(self.res.component_table()["n_edges"].sum()))

    def test_sorting_and_top_k_keep_the_ids(self):
        top = self.res.significant_edges(sort="statistic", top_k=5)
        everything = self.res.significant_edges(include_nonsig=True).set_index("edge_id")
        for _, row in top.iterrows():
            self.assertEqual(row["component_id"], everything.loc[row["edge_id"], "component_id"])


# ---------------------------------------------------------------------------
# analyze(), serialization
# ---------------------------------------------------------------------------

class TestAnalyzeDelegation(unittest.TestCase):

    def test_analyze_exposes_the_report(self):
        g1, g2 = planted_groups()
        out = analyze(Y=np.concatenate([g1, g2]), interest=np.r_[np.zeros(12), np.ones(12)],
                      method="nbs", threshold=TAU, n_permutations=B, rng=SEED, use_mp=False,
                      input="z", acceleration=None)
        self.assertIs(out.component_report, out.inference.component_report)
        self.assertIsNotNone(out.component_report)
        pd.testing.assert_frame_equal(out.component_table(), out.inference.component_table())
        one = out.one_sided("positive")
        self.assertEqual(one.component_table()["tail"].unique().tolist(), ["positive"])

    def test_other_methods_report_none(self):
        g1, g2 = planted_groups()
        out = analyze(Y=np.concatenate([g1, g2]), interest=np.r_[np.zeros(12), np.ones(12)],
                      method="tstat", n_permutations=20, rng=SEED, use_mp=False, input="z",
                      acceleration=None)
        self.assertIsNone(out.component_report)


class TestSerialization(unittest.TestCase):
    """NETSTAT-8 mechanism: the report is part of __dict__ and round-trips."""

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()
        cls.res = two_sample(cls.g1, cls.g2)

    def _check(self, restored, original):
        self.assertIsInstance(restored, InferenceResult)
        assert_tree_equal(self, original.component_report, restored.component_report, ".component_report")
        pd.testing.assert_frame_equal(original.component_table(_atlas(N)), restored.component_table(_atlas(N)))
        pd.testing.assert_frame_equal(original.significant_edges(), restored.significant_edges())

    def test_pickle_roundtrip_both_protocols(self):
        for protocol in (pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL):
            with self.subTest(protocol=protocol):
                self._check(pickle.loads(pickle.dumps(self.res, protocol=protocol)), self.res)

    def test_copy_and_deepcopy(self):
        self._check(copy.copy(self.res), self.res)
        deep = copy.deepcopy(self.res)
        self._check(deep, self.res)
        # independent arrays
        deep.component_report["labels"]["positive"][:] = 0
        deep.component_report["components"][0]["p_fwer"] = -1.0
        self.assertTrue(self.res.component_report["labels"]["positive"].any())
        self.assertGreater(self.res.component_report["components"][0]["p_fwer"], 0.0)

    def test_one_sided_after_restoring(self):
        restored = pickle.loads(pickle.dumps(self.res))
        pd.testing.assert_frame_equal(restored.one_sided("negative").component_table(),
                                      self.res.one_sided("negative").component_table())
        restored_one = pickle.loads(pickle.dumps(self.res.one_sided("positive")))
        self.assertEqual(restored_one.component_report["tested_tail"], "positive")

    def test_omnibus_roundtrip(self):
        g1, g2 = planted_groups()
        Y = np.concatenate([g1, g2])
        rng = np.random.default_rng(2)
        X = np.column_stack([np.ones(len(Y)), np.r_[np.zeros(12), np.ones(12)], rng.normal(size=len(Y))])
        res = compute_p_val_glm(Y, design_matrix=X, contrast=np.array([[0, 1, 0], [0, 0, 1.0]]),
                                stat_type="fstat", method="nbs", threshold=8.0,
                                n_permutations=50, rng=SEED, use_mp=False, acceleration=None)
        back = pickle.loads(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
        assert_tree_equal(self, res.component_report, back.component_report)
        pd.testing.assert_frame_equal(res.component_table(), back.component_table())

    def test_results_pickled_without_the_attribute_still_load(self):
        """A result that predates the report has the class default, not an AttributeError."""
        old = pickle.loads(pickle.dumps(self.res))
        del old.__dict__["component_report"]
        self.assertIsNone(old.component_report)
        self.assertNotIn("component_id", old.significant_edges(include_nonsig=True).columns)


if __name__ == "__main__":
    unittest.main()
