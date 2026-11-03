"""Connectogram atlas coverage: unassigned ROIs are not placed, detections are still counted.

Same fixture pattern as ``tests/test_annotate.py``: the bundled BNA-246 label
image only (no network access); the local probabilistic / tractography maps are
not needed, so no extra skip conditions apply.
"""
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.interpret.annotate import (_unit_weights, aggregate_connections, annotate_bna_regions,
                                          annotate_rois)
from conninfpy.results import InferenceResult

POINTS = {
    "L dlPFC": (-42, 36, 24), "R dlPFC": (42, 36, 24), "L amygdala": (-22, -4, -18),
    "L M1 hand": (-38, -22, 56), "L putamen": (-26, 2, 2),
    "L cerebellum": (-20, -60, -30),      # no labelled support: outside the atlas
    "L ventricle margin": (0, -22, 26),   # labelled share below min_coverage (default 0.5)
}
NETS = ["Frontoparietal", "Frontoparietal", "Limbic", "Somatomotor", "Limbic", "none", "Limbic"]


def _reference():
    """Bundled atlas only (an empty folder disables the local-file lookup)."""
    return BrainnetomeReference.load(data_dir=tempfile.mkdtemp(), profiles=False,
                                     cache_dir=tempfile.mkdtemp())


def _result(n, detected, tail="positive"):
    """InferenceResult over ``n`` ROIs with the given index pairs detected in ``tail``."""
    p = np.ones((n, n))
    stat = np.zeros((n, n))
    for a, b in detected:
        p[a, b] = p[b, a] = 0.01
        stat[a, b] = stat[b, a] = 4.0
    return InferenceResult(
        p if tail == "positive" else np.ones((n, n)),
        np.ones((n, n)) if tail == "positive" else p,
        method="tstat",
        stat_positive=stat if tail == "positive" else np.zeros((n, n)),
        stat_negative=np.zeros((n, n)) if tail == "positive" else -stat,
    )


class TestCoverageOfPlacedROIs(unittest.TestCase):
    """Unassigned ROIs are not placed; their detected connections stay counted."""

    @classmethod
    def setUpClass(cls):
        cls.rois = annotate_rois(list(POINTS.values()), names=list(POINTS), reference=_reference(),
                                 support="sphere", radius_mm=6)
        # detected: dlPFC-dlPFC (both placed), dlPFC-cerebellum (no support),
        # amygdala-ventricle margin (low coverage)
        cls.res = _result(len(POINTS), [(0, 1), (0, 5), (2, 6)])

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_outside_and_low_coverage_rois_are_unassigned(self):
        t = self.rois.table.set_index("roi")
        self.assertEqual(t.loc["L cerebellum", "level"], "unassigned")
        self.assertEqual(self.rois.weights[list(POINTS).index("L cerebellum")].sum(), 0.0)
        self.assertEqual(t.loc["L ventricle margin", "level"], "unassigned")
        cov = self.rois.weights[list(POINTS).index("L ventricle margin")].sum()
        self.assertTrue(0.0 < cov < 0.5)

    def test_coverage_counts_detected_total_and_placed(self):
        agg = aggregate_connections(self.res, self.rois, assign="dominant", networks=NETS,
                                    nonzero_only=False)
        cov = agg.attrs["coverage"]
        self.assertEqual(cov["n_rois"], len(POINTS))
        self.assertEqual(cov["n_placed"], 5)
        self.assertEqual(cov["unplaced_no_support"], ["L cerebellum"])
        self.assertEqual(cov["unplaced_low_coverage"], ["L ventricle margin"])
        self.assertEqual((cov["detected_total"], cov["detected_placed"]), (3, 1))
        self.assertEqual((cov["tested_total"], cov["tested_placed"]), (21, 10))
        self.assertEqual((cov["level"], cov["assign"], cov["tail"]), ("gyrus", "dominant", "positive"))

    def test_connectogram_footer_reports_unplaced_rois(self):
        from conninfpy.plot import plot_connectogram

        ax = plot_connectogram(self.res, self.rois, assign="dominant", networks=NETS)
        footer = ax.texts[-1].get_text()
        self.assertIn("5 of 7 ROIs placed (1 outside the atlas, 1 below min_coverage)", footer)
        self.assertIn("carry 1 of 3 detected connections (33%)", footer)
        self.assertIn("2 detected connections touch an unplaced ROI", footer)

    def test_weighted_assignment_conserves_connections_among_placed(self):
        agg = aggregate_connections(self.res, self.rois, assign="weighted", networks=NETS,
                                    nonzero_only=False)
        cov = agg.attrs["coverage"]
        self.assertAlmostEqual(agg["detected"].sum(), cov["detected_placed"])
        self.assertAlmostEqual(agg["tested"].sum(), cov["tested_placed"])


class TestBNARegionsAllPlaced(unittest.TestCase):
    """BNA-246 input: every region is placed, coverage is complete, the connectogram draws."""

    @classmethod
    def setUpClass(cls):
        cls.rois = annotate_bna_regions([1, 2, 210, 242], reference=_reference())
        cls.res = _result(4, [(0, 1)])          # A8m_L <-> A8m_R

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_coverage_is_complete(self):
        agg = aggregate_connections(self.res, self.rois, assign="dominant")
        cov = agg.attrs["coverage"]
        self.assertEqual((cov["n_rois"], cov["n_placed"]), (4, 4))
        self.assertEqual((cov["unplaced_no_support"], cov["unplaced_low_coverage"]), ([], []))
        self.assertEqual((cov["detected_total"], cov["detected_placed"]), (1, 1))
        self.assertEqual((cov["tested_total"], cov["tested_placed"]), (6, 6))

    def test_connectogram_draws(self):
        from matplotlib.collections import LineCollection

        from conninfpy.plot import plot_connectogram

        ax = plot_connectogram(self.res, self.rois, assign="dominant")
        chords = [c for c in ax.collections if isinstance(c, LineCollection)]
        self.assertEqual(len(chords), 1)
        footer = ax.texts[-1].get_text()
        self.assertIn("4 of 4 ROIs placed (0 outside the atlas, 0 below min_coverage)", footer)
        self.assertIn("carry 1 of 1 detected connections (100%)", footer)
        self.assertNotIn("touch an unplaced ROI", footer)


class TestCoordinateROIs(unittest.TestCase):
    """Coordinate ROIs: sphere and point supports place each ROI in its dominant unit."""

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_sphere_support_dominant_unit(self):
        rois = annotate_rois([(-42, 36, 24)], names=["L dlPFC"], reference=_reference(),
                             support="sphere", radius_mm=6, space="MNI152NLin6Asym")
        self.assertEqual(rois.table.loc[0, "level"], "subregion")
        w, units = _unit_weights(rois, "gyrus", "dominant", placed_only=True)
        self.assertEqual(units[int(w[0].argmax())], "MFG_L")
        self.assertAlmostEqual(w.sum(), 1.0)

    def test_point_support_dominant_unit(self):
        rois = annotate_rois([(-42, 36, 24), (-28, -10, 30)], names=["L dlPFC", "L WM"],
                             reference=_reference(), support="point")
        self.assertEqual(rois.table.loc[0, "level"], "subregion")
        w, units = _unit_weights(rois, "subregion", "dominant", placed_only=True)
        self.assertEqual(units[int(w[0].argmax())], "A9/46v_L")
        self.assertAlmostEqual(w[0].sum(), 1.0)
        self.assertEqual(w[1].sum(), 0.0)      # white-matter point: unassigned, not placed


class TestUnitWeightsDefaultUnchanged(unittest.TestCase):
    """placed_only=False (default) keeps the weighted composition of every labelled ROI."""

    @classmethod
    def setUpClass(cls):
        cls.rois = annotate_rois(list(POINTS.values()), names=list(POINTS), reference=_reference(),
                                 support="sphere", radius_mm=6)

    def test_default_ignores_the_unassigned_marker(self):
        default, units = _unit_weights(self.rois, "gyrus", "weighted")
        placed, _ = _unit_weights(self.rois, "gyrus", "weighted", placed_only=True)
        labelled = self.rois.weights.sum(axis=1) > 0
        # every ROI with labelled support keeps a normalised row (adapter.py:170 regression)
        np.testing.assert_allclose(default[labelled].sum(axis=1), 1.0)
        # placed rows are identical; only unassigned rows differ
        keep = np.asarray(self.rois.table["level"] != "unassigned")
        np.testing.assert_allclose(default[keep], placed[keep])
        i_low = list(POINTS).index("L ventricle margin")
        self.assertGreater(default[i_low].sum(), 0.0)
        self.assertEqual(placed[i_low].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
