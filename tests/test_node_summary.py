"""``node_summary``: which ROIs gather the detected connections (descriptive).

Edge results are checked against hand counts and independent loops, component
(NBS) and block (cNBS) results against the reports' own membership, and every
test that builds a result checks that the summary leaves it unchanged.
"""
from __future__ import annotations

import copy
import pickle
import unittest

import numpy as np
import pandas as pd

from conninfpy import (
    AtlasInfo,
    InferenceResult,
    OmnibusInferenceResult,
    analyze,
    compute_p_val,
    compute_p_val_glm,
)
from tests.test_component_report import (
    B,
    N,
    TAU,
    glm_indicator,
    oracle_components,
    planted_groups,
    two_sample,
)
from tests.test_serialization import assert_results_equal

ALPHA = 0.05
LABELS = np.arange(N) // 6          # four networks of six ROIs: 0-5, 6-11, 12-17, 18-23
NET_NAMES = ["Vis", "DMN", "SMN", "FPN"]


def atlas_for(n=N, networks=True, hemisphere=True):
    nets = [NET_NAMES[min(i // 6, 3)] for i in range(n)] if networks else [""] * n
    return AtlasInfo(labels=[f"roi{i}" for i in range(n)], networks=nets,
                     hemisphere=[("L" if i % 2 == 0 else "R") for i in range(n)] if hemisphere else None)


def sym(n, values):
    """Symmetric (n, n) matrix with ``values[(i, j)]`` on both triangles."""
    m = np.zeros((n, n))
    for (i, j), v in values.items():
        m[i, j] = m[j, i] = v
    return m


# ---------------------------------------------------------------------------
# A result built by hand: every count below is worked out on paper
# ---------------------------------------------------------------------------

def hand_result():
    """7 ROIs. Significant (p = .01): positive (0,1) (0,2) (0,3) (4,5), negative (1,2).
    Untested (NaN p in both tails): (2,5) and (3,5). Effects c'beta: 01:1 02:2 03:3 45:4 12:-5."""
    n = 7
    p_pos, p_neg = np.full((n, n), 0.5), np.full((n, n), 0.5)
    for i, j in ((0, 1), (0, 2), (0, 3), (4, 5)):
        p_pos[i, j] = p_pos[j, i] = 0.01
    p_neg[1, 2] = p_neg[2, 1] = 0.01
    for i, j in ((2, 5), (3, 5)):
        p_pos[i, j] = p_pos[j, i] = np.nan
        p_neg[i, j] = p_neg[j, i] = np.nan
    effect = sym(n, {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (4, 5): 4.0, (1, 2): -5.0})
    stat = np.abs(effect)
    return InferenceResult(p_pos, p_neg, method="tstat", stat_positive=np.where(effect > 0, stat, 0.0),
                           stat_negative=np.where(effect < 0, stat, 0.0), beta=effect[None],
                           design={"contrast": [1.0]}, two_tailed=False)


HAND_ATLAS = AtlasInfo(labels=[f"r{i}" for i in range(7)], networks=["A", "A", "B", "B", "C", "C", "C"],
                       hemisphere=["L", "R", "L", "R", "L", "R", "L"])


class TestEdgeResultByHand(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res = hand_result()
        cls.before = copy.deepcopy(cls.res)
        cls.table = cls.res.node_summary(HAND_ATLAS, sort="roi")

    def row(self, roi):
        return self.table.set_index("roi").loc[roi]

    def test_one_row_per_roi_and_columns(self):
        self.assertEqual(self.table["roi"].tolist(), list(range(7)))
        self.assertEqual(self.table["roi_name"].tolist(), [f"r{i}" for i in range(7)])
        self.assertEqual(self.table["hemisphere"].tolist(), ["L", "R", "L", "R", "L", "R", "L"])
        self.assertTrue((self.table["inference_unit"] == "edge").all())
        self.assertFalse([c for c in self.table.columns if "log10" in c or "neglog" in c])

    def test_counts_and_denominator_are_the_tested_incident_edges(self):
        # tested incident edges: 6 per ROI minus the untested (2,5) and (3,5)
        self.assertEqual(self.table["n_tested"].tolist(), [6, 6, 5, 5, 6, 4, 6])
        self.assertEqual(self.table["n_sig_positive"].tolist(), [3, 1, 1, 1, 1, 1, 0])
        self.assertEqual(self.table["n_sig_negative"].tolist(), [0, 1, 1, 0, 0, 0, 0])
        self.assertEqual(self.table["n_sig_total"].tolist(), [3, 2, 2, 1, 1, 1, 0])
        np.testing.assert_allclose(self.table["prop_sig"], [3 / 6, 2 / 6, 2 / 5, 1 / 5, 1 / 6, 1 / 4, 0.0])
        np.testing.assert_allclose(self.table["prop_sig_positive"], [3 / 6, 1 / 6, 1 / 5, 1 / 5, 1 / 6, 1 / 4, 0.0])

    def test_mean_effect_is_the_effect_map_over_significant_edges_of_the_tail(self):
        pos = self.table["mean_effect_positive"].to_numpy()
        neg = self.table["mean_effect_negative"].to_numpy()
        np.testing.assert_allclose(pos, [2.0, 1.0, 2.0, 3.0, 4.0, 4.0, np.nan])       # (1+2+3)/3 for ROI 0
        np.testing.assert_allclose(neg, [np.nan, -5.0, -5.0, np.nan, np.nan, np.nan, np.nan])

    def test_partner_networks_and_participation(self):
        # ROI 0: partners 1 (A), 2 (B), 3 (B) -> P = 1 - (1/9 + 4/9) = 4/9
        self.assertAlmostEqual(self.row(0)["participation"], 4 / 9)
        self.assertEqual(self.row(0)["partner_networks"], ["B (2)", "A (1)"])
        self.assertEqual(self.row(0)["n_networks_reached"], 2)
        # ROI 1: partners 0 (A), 2 (B) -> P = 1/2
        self.assertAlmostEqual(self.row(1)["participation"], 0.5)
        # ROI 2: partners 0 (A), 1 (A): one network -> 0; ROI 3 and 4: one partner -> 0
        for roi in (2, 3, 4, 5):
            self.assertEqual(self.row(roi)["participation"], 0.0)
        self.assertEqual(self.row(4)["partner_networks"], ["C (1)"])

    def test_a_roi_without_detections_is_nan_not_zero(self):
        r = self.row(6)
        self.assertTrue(np.isnan(r["participation"]))
        self.assertEqual(r["partner_networks"], [])
        self.assertEqual(r["n_networks_reached"], 0)
        self.assertEqual(r["n_sig_total"], 0)
        self.assertTrue(np.isnan(r["mean_effect_positive"]))

    def test_default_sort_is_by_detections_then_roi(self):
        by_detections = self.res.node_summary(HAND_ATLAS)
        self.assertEqual(by_detections["roi"].tolist(), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(self.res.node_summary(HAND_ATLAS, top_k=2)["roi"].tolist(), [0, 1])
        self.assertEqual(self.res.node_summary(HAND_ATLAS, alpha=0.001)["n_sig_total"].sum(), 0)
        wider = self.res.node_summary(HAND_ATLAS, alpha=0.6, sort="detections")
        self.assertEqual(wider["n_sig_total"].max(), wider.iloc[0]["n_sig_total"])

    def test_tail_selection(self):
        pos = self.res.node_summary(HAND_ATLAS, tail="positive", sort="roi")
        self.assertNotIn("n_sig_negative", pos.columns)
        self.assertEqual(pos["n_sig_total"].tolist(), self.table["n_sig_positive"].tolist())
        # participation follows the selected tails: ROI 1 has only its positive edge to A left
        self.assertEqual(pos.set_index("roi").loc[1, "participation"], 0.0)
        with self.assertRaises(ValueError):
            self.res.node_summary(tail="up")

    def test_the_result_is_not_changed(self):
        assert_results_equal(self, self.before, self.res)
        self.assertEqual(HAND_ATLAS.networks, ["A", "A", "B", "B", "C", "C", "C"])

    def test_attrs_record_the_policy_and_the_descriptive_status(self):
        a = self.table.attrs
        self.assertEqual((a["alpha"], a["inference_unit"], a["tails"]), (0.05, "edge", ["positive", "negative"]))
        self.assertEqual(a["tail_policy"], "separate-one-sided")
        self.assertIn("descriptive", a["descriptive"])
        self.assertIn("NaN p-value", a["denominator"])
        self.assertIn("c'beta", a["effect_source"])


class TestAtlasHandling(unittest.TestCase):

    def test_no_atlas_gives_roi_index_and_no_network_columns(self):
        t = hand_result().node_summary()
        self.assertEqual(t.columns[0], "roi")
        for c in ("roi_name", "network", "participation", "partner_networks", "n_networks_reached"):
            self.assertNotIn(c, t.columns)
        self.assertEqual(t["n_sig_total"].sum(), 2 * 5)      # five significant edges, both endpoints

    def test_atlas_without_network_labels_adds_names_only(self):
        blank = AtlasInfo(labels=[f"r{i}" for i in range(7)], networks=[""] * 7)
        t = hand_result().node_summary(blank)
        self.assertIn("roi_name", t.columns)
        for c in ("network", "participation", "partner_networks", "n_networks_reached", "hemisphere"):
            self.assertNotIn(c, t.columns)
        unknown = AtlasInfo(labels=[f"r{i}" for i in range(7)], networks=["Unknown"] * 7)
        self.assertNotIn("participation", hand_result().node_summary(unknown).columns)

    def test_wrong_atlas_length_is_rejected(self):
        with self.assertRaises(ValueError):
            hand_result().node_summary(atlas_for(8))

    def test_network_column_is_the_atlas_network(self):
        t = hand_result().node_summary(HAND_ATLAS, sort="roi")
        self.assertEqual(t["network"].tolist(), ["A", "A", "B", "B", "C", "C", "C"])


# ---------------------------------------------------------------------------
# Real pipelines: denominators with edge_mask and available_case
# ---------------------------------------------------------------------------

def independent_counts(res, alpha=ALPHA):
    """Per ROI: tested incident edges and significant incident edges (either tail), by plain loops."""
    n = res["positive"].shape[0]
    tested, sig = np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_pos, p_neg = res["positive"][i, j], res["negative"][i, j]
            if np.isnan(p_pos):
                continue
            tested[i] += 1
            sig[i] += int(p_pos <= alpha or p_neg <= alpha)
    return tested, sig


class TestDenominators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()

    def test_plain_result_denominator_is_n_minus_one(self):
        res = two_sample(self.g1, self.g2, method="tstat", n_permutations=B)
        t = res.node_summary(sort="roi")
        self.assertTrue((t["n_tested"] == N - 1).all())
        tested, sig = independent_counts(res)
        np.testing.assert_array_equal(t["n_tested"], tested)
        np.testing.assert_array_equal(t["n_sig_total"], sig)
        np.testing.assert_allclose(t["prop_sig"], sig / tested)

    def test_edge_mask_family_limits_the_denominator(self):
        mask = np.zeros((N, N), dtype=bool)
        keep = list(range(0, 10)) + list(range(14, 18))
        mask[np.ix_(keep, keep)] = True
        np.fill_diagonal(mask, False)
        res = two_sample(self.g1, self.g2, method="tstat", edge_mask=mask)
        t = res.node_summary(sort="roi")
        np.testing.assert_array_equal(t["n_tested"], mask.sum(axis=1))
        self.assertEqual(int(t.loc[t["roi"] == 20, "n_tested"].iloc[0]), 0)
        self.assertTrue(np.isnan(t.loc[t["roi"] == 20, "prop_sig"].iloc[0]))      # 0 tested: NaN, not 0
        tested, sig = independent_counts(res)
        np.testing.assert_array_equal(t["n_sig_total"], sig)
        self.assertTrue((t["n_tested"] < N - 1).any())

    def test_available_case_untested_edges_are_in_neither_numerator_nor_denominator(self):
        g1 = self.g1.copy()
        for s in range(5):                           # ROI 20 observed in 7 of 12 subjects (< 80 %)
            g1[s, 20, :] = np.nan
            g1[s, :, 20] = np.nan
        res = compute_p_val(g1, self.g2, test_type="two-sample", var_equal=True, method="tfnbs", e=0.4, h=3.0,
                            n=10, n_permutations=B, rng=1, use_mp=False, acceleration=None,
                            missing="available_case", two_tailed=False)
        untested = np.zeros((N, N), dtype=int)
        for i, j in res.missing_report["untested_edges"]:        # the report, not the p map
            untested[i, j] = untested[j, i] = 1
        self.assertGreater(untested.sum(), 0)
        t = res.node_summary(sort="roi")
        np.testing.assert_array_equal(t["n_tested"], N - 1 - untested.sum(axis=1))
        self.assertEqual(int(t.loc[t["roi"] == 20, "n_tested"].iloc[0]), 0)
        tested, sig = independent_counts(res)
        np.testing.assert_array_equal(t["n_sig_total"], sig)

    def test_glm_t_path_uses_the_contrast_effect(self):
        res = glm_indicator(self.g1, self.g2, method="tfnbs", e=0.4, h=3.0, n=10)
        t = res.node_summary(sort="roi")
        eff = res.effect_map()
        self.assertIsNotNone(eff)
        for roi in (0, 3, 14):
            js = [j for j in range(N) if j != roi and res["positive"][roi, j] <= ALPHA]
            expected = np.mean([eff[roi, j] for j in js]) if js else np.nan
            np.testing.assert_allclose(t.loc[t["roi"] == roi, "mean_effect_positive"].iloc[0], expected)
        # the effect is c'beta, not the t statistic
        self.assertFalse(np.allclose(eff, res.stat_signed))

    def test_without_an_effect_map_the_column_is_nan(self):
        res = hand_result()
        res.beta, res.design = None, None
        t = res.node_summary(HAND_ATLAS)
        self.assertTrue(t["mean_effect_positive"].isna().all())
        self.assertTrue(t["mean_effect_negative"].isna().all())
        self.assertGreater(t["n_sig_total"].sum(), 0)
        self.assertIn("NaN when the result carries none", t.attrs["effect_source"])

    def test_two_tailed_results_report_both_tails_and_the_policy(self):
        res = two_sample(self.g1, self.g2, method="tstat", two_tailed=True)
        t = res.node_summary()
        self.assertEqual(t.attrs["tail_policy"], "two-sided")
        self.assertEqual(t.attrs["tails"], ["positive", "negative"])
        self.assertIn("n_sig_negative", t.columns)

    def test_analyze_forwards_node_summary(self):
        out = analyze(Y=np.concatenate([self.g1, self.g2]), interest=np.r_[np.zeros(12), np.ones(12)],
                      method="tstat", n_permutations=50, rng=1, use_mp=False, input="z", acceleration=None)
        pd.testing.assert_frame_equal(out.node_summary(atlas_for()), out.inference.node_summary(atlas_for()))


class TestOneSidedAndGrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()

    def test_one_sided_reports_only_the_tested_tail(self):
        both = two_sample(self.g1, self.g2, method="tstat")
        pos = both.one_sided("positive")
        t = pos.node_summary(sort="roi")
        self.assertEqual(t.attrs["tails"], ["positive"])
        self.assertEqual(t.attrs["tested_tail"], "positive")
        self.assertEqual(t.attrs["tail_policy"], "one-sided-positive")
        self.assertNotIn("n_sig_negative", t.columns)
        full = both.node_summary(sort="roi")
        np.testing.assert_array_equal(t["n_sig_positive"], full["n_sig_positive"])
        np.testing.assert_array_equal(t["n_sig_total"], full["n_sig_positive"])      # the untested tail adds nothing
        with self.assertRaisesRegex(ValueError, "negative tail was not tested"):
            pos.node_summary(tail="negative")
        self.assertEqual(pos.node_summary(tail="positive", sort="roi")["n_sig_total"].tolist(),
                         t["n_sig_total"].tolist())

    def test_parameter_grid_needs_a_cell_and_matches_select(self):
        res = compute_p_val(self.g1, self.g2, test_type="two-sample", var_equal=True, method="tfnbs",
                            e=[0.3, 0.5], h=[3.0, 2.0], n=10, n_permutations=B, rng=1, use_mp=False,
                            acceleration=None, two_tailed=False)
        self.assertTrue(res.is_grid)
        with self.assertRaises(ValueError):
            res.node_summary()
        t = res.node_summary(param_idx=1, sort="roi")
        sel = res.select(1).node_summary(sort="roi")
        for c in ("n_tested", "n_sig_positive", "n_sig_negative", "n_sig_total"):
            np.testing.assert_array_equal(t[c], sel[c])
        cell = res["positive"][:, :, 1]
        manual = [(np.nansum(cell[i] <= ALPHA)) for i in range(N)]       # diagonal p is 1, never <= alpha
        np.testing.assert_array_equal(t["n_sig_positive"], manual)
        self.assertEqual((t.attrs["param_idx"], t.attrs["e"], t.attrs["h"]), (1, 0.5, 2.0))
        other = res.node_summary(param_idx=0, sort="roi")
        self.assertEqual(other.attrs["e"], 0.3)


# ---------------------------------------------------------------------------
# Component (NBS) and block (cNBS) results: membership, no p mass
# ---------------------------------------------------------------------------

FORBIDDEN = ("n_sig", "prop_sig", "mean_effect", "log10")


class TestComponentResults(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()
        cls.res = two_sample(cls.g1, cls.g2)
        cls.before = copy.deepcopy(cls.res)
        cls.table = cls.res.node_summary(atlas_for(), sort="roi")

    def test_unit_comes_from_the_result_and_no_significance_counts_are_formed(self):
        self.assertEqual(self.res.inference_unit, "component")
        self.assertTrue((self.table["inference_unit"] == "component").all())
        for c in self.table.columns:
            self.assertFalse(any(c.startswith(f) or f in c for f in FORBIDDEN), c)
        self.assertIn("component", self.table.attrs["descriptive"])
        self.assertIn("no per-ROI significance", self.table.attrs["descriptive"])
        self.assertIsNone(self.table.attrs["effect_source"])

    def test_membership_matches_the_oracle_components(self):
        report = self.res.component_report
        stat = {"positive": self.res.stat_positive, "negative": self.res.stat_negative}
        for tail in ("positive", "negative"):
            expected = oracle_components(stat[tail], TAU, "extent")
            rows = [r for r in report["components"] if r["tail"] == tail]
            sig_ids = {r["component_id"]: r for r in rows if r["p_fwer"] <= ALPHA}
            self.assertTrue(sig_ids)
            for roi in range(N):
                ids = [cid for cid, r in sig_ids.items() if roi in r["nodes"]]
                got = self.table.loc[self.table["roi"] == roi]
                self.assertEqual(got[f"component_ids_{tail}"].iloc[0], ids)
                self.assertEqual(int(got[f"n_components_{tail}"].iloc[0]), len(ids))
                # edges of the ROI inside those components, from the independent oracle
                n_edges = 0
                for cid in ids:
                    nodes, _, _ = expected[cid - 1]
                    mask = np.round(stat[tail], 10) > TAU
                    n_edges += sum(mask[roi, j] for j in nodes if j != roi)
                self.assertEqual(int(got[f"n_edges_in_components_{tail}"].iloc[0]), n_edges)

    def test_only_significant_components_count(self):
        weak = two_sample(self.g1, self.g2, threshold=2.4)
        table = weak.component_table()
        self.assertTrue((~table["significant"]).any())
        t = weak.node_summary(sort="roi")
        sig = table[table["significant"]]
        for roi in (0, 5, 14, 20):
            for tail in ("positive", "negative"):
                ids = sorted(sig[(sig["tail"] == tail) & sig["nodes"].map(lambda nodes, roi=roi: roi in nodes)]["component_id"])
                self.assertEqual(sorted(t.loc[t["roi"] == roi, f"component_ids_{tail}"].iloc[0]), ids)

    def test_participation_follows_the_edges_of_the_significant_components(self):
        report = self.res.component_report
        t = self.table.set_index("roi")
        nets = np.array(NET_NAMES)[np.minimum(np.arange(N) // 6, 3)]
        for roi in (0, 3, 14, 20):
            partners = []
            for r in report["components"]:
                if r["p_fwer"] <= ALPHA:
                    labels = report["labels"][r["tail"]] == r["component_id"]
                    partners += [nets[j] for j in range(N) if j != roi and labels[roi, j]]
            if partners:
                k = pd.Series(partners).value_counts()
                self.assertAlmostEqual(t.loc[roi, "participation"], 1 - ((k / k.sum()) ** 2).sum())
                self.assertEqual(t.loc[roi, "n_networks_reached"], len(k))
            else:
                self.assertTrue(np.isnan(t.loc[roi, "participation"]))      # NaN, not 0
        self.assertTrue(np.isnan(t.loc[20, "participation"]))

    def test_result_is_not_changed_and_pickles_afterwards(self):
        assert_results_equal(self, self.before, self.res)
        again = pickle.loads(pickle.dumps(self.res)).node_summary(atlas_for(), sort="roi")
        pd.testing.assert_frame_equal(again, self.table)

    def test_one_sided_keeps_the_tested_tail(self):
        t = self.res.one_sided("negative").node_summary(sort="roi")
        self.assertEqual(t.attrs["tails"], ["negative"])
        self.assertIn("n_components_negative", t.columns)
        self.assertNotIn("n_components_positive", t.columns)
        full = self.res.node_summary(sort="roi")
        np.testing.assert_array_equal(t["n_edges_in_components_negative"], full["n_edges_in_components_negative"])
        np.testing.assert_array_equal(t["n_edges_in_components"], full["n_edges_in_components_negative"])

    def test_missing_report_is_explained(self):
        old = pickle.loads(pickle.dumps(self.res))
        del old.__dict__["component_report"]
        with self.assertRaisesRegex(ValueError, "component report"):
            old.node_summary()

    def test_f_omnibus_nbs_uses_the_single_tail(self):
        Y = np.concatenate([self.g1, self.g2])
        rng = np.random.default_rng(2)
        X = np.column_stack([np.ones(24), np.r_[np.zeros(12), np.ones(12)], rng.normal(size=24)])
        res = compute_p_val_glm(Y, design_matrix=X, contrast=np.array([[0, 1, 0], [0, 0, 1.0]]), stat_type="fstat",
                                method="nbs", threshold=8.0, n_permutations=B, rng=1, use_mp=False, acceleration=None)
        self.assertIsInstance(res, OmnibusInferenceResult)
        t = res.node_summary(atlas_for(), sort="roi")
        self.assertEqual(t.attrs["tails"], ["omnibus"])
        for c in ("n_components", "component_ids", "n_edges_in_components"):
            self.assertIn(c, t.columns)
        self.assertFalse([c for c in t.columns if "positive" in c or "negative" in c or "effect" in c])
        sig = [r for r in res.component_report["components"] if r["p_fwer"] <= ALPHA]
        self.assertTrue(sig)
        for roi in range(N):
            self.assertEqual(t.loc[t["roi"] == roi, "component_ids"].iloc[0],
                             [r["component_id"] for r in sig if roi in r["nodes"]])


def block_expectations(res, alpha=ALPHA):
    """Per ROI and tail: block ids and number of tested edges inside the significant blocks (plain loops)."""
    labels = np.asarray(res.block_report["node_labels"])
    out = {}
    for tail in ("positive", "negative"):
        ids = {i: [] for i in range(N)}
        edges = {i: 0 for i in range(N)}
        for b in res.block_report["blocks"]:
            if not (b[f"p_{tail}"] <= alpha):
                continue
            a, c = b["network_a"], b["network_b"]
            for i in range(N):
                if labels[i] not in (a, c):
                    continue
                ids[i].append("—".join(sorted((str(a), str(c)))))
                for j in range(N):
                    if j != i and {labels[i], labels[j]} == {a, c} and np.isfinite(res[tail][i, j]):
                        edges[i] += 1
        out[tail] = (ids, edges)
    return out


class TestBlockResults(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()
        cls.res = compute_p_val(cls.g1, cls.g2, test_type="two-sample", var_equal=True, method="cnbs",
                                net_labels=LABELS, n_permutations=399, rng=1, use_mp=False, acceleration=None,
                                two_tailed=False)
        cls.before = copy.deepcopy(cls.res)
        cls.table = cls.res.node_summary(atlas_for(), sort="roi")

    def test_block_report_records_the_partition(self):
        self.assertEqual(self.res.block_report["node_labels"], LABELS.tolist())
        self.assertEqual(self.res.inference_unit, "block")

    def test_membership_matches_the_block_report(self):
        expected = block_expectations(self.res)
        self.assertTrue(any(expected["positive"][0][i] for i in range(N)))
        for tail, (ids, edges) in expected.items():
            for roi in range(N):
                got = self.table.loc[self.table["roi"] == roi].iloc[0]
                self.assertEqual(got[f"block_ids_{tail}"], ids[roi])
                self.assertEqual(got[f"n_blocks_{tail}"], len(ids[roi]))
                self.assertEqual(got[f"n_edges_in_blocks_{tail}"], edges[roi])
        # a ROI of the planted 6-node block is in the within-block 0—0, whose six-ROI block has 5 edges each
        self.assertIn("0—0", self.table.loc[self.table["roi"] == 0, "block_ids_positive"].iloc[0])

    def test_no_p_mass_and_descriptive_note(self):
        for c in self.table.columns:
            self.assertFalse(any(f in c for f in FORBIDDEN), c)
        self.assertIn("block", self.table.attrs["descriptive"])
        self.assertTrue((self.table["inference_unit"] == "block").all())

    def test_participation_from_the_edges_in_significant_blocks(self):
        # ROI 0 sits in the significant within-block 0—0 (and maybe more): all partners are Vis ROIs
        row = self.table.loc[self.table["roi"] == 0].iloc[0]
        expected_ids = block_expectations(self.res)["positive"][0][0]
        partners = []
        labels = np.asarray(self.res.block_report["node_labels"])
        for tail in ("positive", "negative"):
            for b in self.res.block_report["blocks"]:
                if b[f"p_{tail}"] <= ALPHA and labels[0] in (b["network_a"], b["network_b"]):
                    for j in range(1, N):
                        if {labels[0], labels[j]} == {b["network_a"], b["network_b"]}:
                            partners.append(NET_NAMES[labels[j]])
        k = pd.Series(partners).value_counts()
        self.assertAlmostEqual(row["participation"], 1 - ((k / k.sum()) ** 2).sum())
        self.assertTrue(expected_ids)

    def test_two_tailed_block_result(self):
        res = compute_p_val(self.g1, self.g2, test_type="two-sample", var_equal=True, method="cnbs",
                            net_labels=LABELS, n_permutations=399, rng=1, use_mp=False, acceleration=None,
                            two_tailed=True)
        t = res.node_summary()
        self.assertEqual(t.attrs["tail_policy"], "two-sided")
        self.assertEqual(t.attrs["tails"], ["positive", "negative"])
        expected = block_expectations(res)
        for roi in (0, 14):
            self.assertEqual(t.loc[t["roi"] == roi, "block_ids_positive"].iloc[0], expected["positive"][0][roi])

    def test_result_unchanged_and_old_block_report_is_explained(self):
        assert_results_equal(self, self.before, self.res)
        old = pickle.loads(pickle.dumps(self.res))
        del old.block_report["node_labels"]
        with self.assertRaisesRegex(ValueError, "node labels"):
            old.node_summary()

    def test_f_omnibus_blocks(self):
        Y = np.concatenate([self.g1, self.g2])
        rng = np.random.default_rng(2)
        X = np.column_stack([np.ones(24), np.r_[np.zeros(12), np.ones(12)], rng.normal(size=24)])
        res = compute_p_val_glm(Y, design_matrix=X, contrast=np.array([[0, 1, 0], [0, 0, 1.0]]), stat_type="fstat",
                                method="cnbs", net_labels=LABELS, n_permutations=399, rng=1, use_mp=False,
                                acceleration=None)
        t = res.node_summary(sort="roi")
        sig = [b for b in res.block_report["blocks"] if b["p_omnibus"] <= ALPHA]
        self.assertTrue(sig)
        for roi in (0, 7, 14, 21):
            ids = [("—".join(sorted((str(b["network_a"]), str(b["network_b"])))))
                   for b in sig if LABELS[roi] in (b["network_a"], b["network_b"])]
            self.assertEqual(t.loc[t["roi"] == roi, "block_ids"].iloc[0], ids)


# ---------------------------------------------------------------------------
# The unit of every method comes from the result, and FBC-TFNBS is edge-wise
# ---------------------------------------------------------------------------

class TestUnitSelection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1, cls.g2 = planted_groups()

    def test_unit_per_method(self):
        base = dict(test_type="two-sample", var_equal=True, n_permutations=30, rng=1, use_mp=False, acceleration=None)
        expected = {"tstat": "edge", "bh_fdr": "edge", "bh_fdr_perm": "edge", "tfnbs": "edge",
                    "ni_tfnbs": "edge", "fbc_tfnbs": "edge", "nbs": "component", "cnbs": "block"}
        for method, unit in expected.items():
            kw = dict(base)
            if method in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
                kw["net_labels"] = LABELS
            if method == "nbs":
                kw["threshold"] = TAU
            if method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
                kw["n"] = 10
            res = compute_p_val(self.g1, self.g2, method=method, **kw)
            with self.subTest(method=method):
                self.assertEqual(res.inference_unit, unit)
                t = res.node_summary(sort="roi")
                self.assertTrue((t["inference_unit"] == unit).all())
                self.assertEqual("n_sig_total" in t.columns, unit == "edge")

    def test_fbc_tfnbs_scores_edges_not_blocks(self):
        """Why FBC-TFNBS is an edge result: the score (hence the p-value) is a property of the edge."""
        from conninfpy.enhance import apply_fbc_tfnbs
        from conninfpy.stats.pairwise import compute_t_stat

        t = compute_t_stat(self.g1, self.g2, test_type="two-sample")
        score = apply_fbc_tfnbs(t, net_labels=LABELS, e=0.4, h=3.0, n=10, min_cluster_size=3)["positive"]
        within = score[:6, :6][np.triu_indices(6, 1)]          # the 15 edges of one block
        self.assertGreater(np.unique(np.round(within, 8)).size, 1)


class TestOmnibusEdgeResult(unittest.TestCase):

    def test_f_omnibus_edge_summary(self):
        g1, g2 = planted_groups()
        Y = np.concatenate([g1, g2])
        rng = np.random.default_rng(2)
        X = np.column_stack([np.ones(24), np.r_[np.zeros(12), np.ones(12)], rng.normal(size=24)])
        res = compute_p_val_glm(Y, design_matrix=X, contrast=np.array([[0, 1, 0], [0, 0, 1.0]]), stat_type="fstat",
                                method="tfnbs", e=0.4, h=3.0, n=10, n_permutations=B, rng=1, use_mp=False,
                                acceleration=None)
        before = copy.deepcopy(res)
        t = res.node_summary(atlas_for(), sort="roi")
        assert_results_equal(self, before, res)
        self.assertEqual(t.attrs["tails"], ["omnibus"])
        self.assertTrue((t["n_tested"] == N - 1).all())
        manual = [(res["omnibus"][i][np.arange(N) != i] <= ALPHA).sum() for i in range(N)]
        np.testing.assert_array_equal(t["n_sig_total"], manual)
        np.testing.assert_allclose(t["prop_sig"], np.array(manual) / (N - 1))
        self.assertFalse([c for c in t.columns if "effect" in c or "positive" in c or "negative" in c])
        self.assertIn("participation", t.columns)
        self.assertIsNone(t.attrs["effect_source"])


if __name__ == "__main__":
    unittest.main()
