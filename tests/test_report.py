"""One-call Brainnetome interpretation of a result (conninfpy.interpret.report)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from conninfpy.brainnetome import BrainnetomeReference
from conninfpy.interpret import annotate_rois, interpret_result
from conninfpy.interpret.report import reproduction_note, roi_markdown
from conninfpy.results import InferenceResult

POINTS = [(-42, 36, 24), (42, 36, 24), (-22, -4, -18), (-38, -22, 56), (-26, 2, 2), (-6, -52, 28)]
NETS = ["Frontoparietal", "Frontoparietal", "Limbic", "Somatomotor", "Limbic", "Default"]


def _result():
    n = len(POINTS)
    pos, neg = np.ones((n, n)), np.ones((n, n))
    for a, b in ((0, 1), (0, 3)):
        pos[a, b] = pos[b, a] = 0.01
    neg[2, 4] = neg[4, 2] = 0.02
    stat = np.where(pos < 1, 4.0, 0.0)
    return InferenceResult(pos, neg, method="tstat", stat_positive=stat, stat_negative=np.where(neg < 1, 3.0, 0.0))


class TestInterpretResult(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ref = BrainnetomeReference.load(data_dir=tempfile.mkdtemp(), profiles=False, cache_dir=tempfile.mkdtemp())
        cls.rois = annotate_rois(POINTS, reference=ref, support="sphere", radius_mm=6, networks=NETS)
        cls.res = _result()

    def test_tables_follow_the_source_networks(self):
        rep = interpret_result(self.res, self.rois, name="toy", source_atlas="six spheres")
        self.assertEqual(len(rep.edges), 3)
        net = rep.relations_by_tail[rep.relations_by_tail.relation == "network_relation"].set_index("category")
        self.assertEqual(net.at["within Frontoparietal", "n_positive"], 1)
        self.assertEqual(net.at["within Limbic", "n_negative"], 1)
        self.assertEqual(set(rep.groups), {"lobe", "network"})
        self.assertIn("Frontoparietal", set(rep.groups["network"]["group"]))
        md = rep.markdown(reproduction="max |Δp| = 0")
        self.assertIn("## toy", md)
        self.assertIn("Network relation", md)
        self.assertIn("ROI annotation", roi_markdown(self.rois, "spheres"))

    def test_save_writes_tables_evidence_and_connectograms(self):
        rep = interpret_result(self.res, self.rois, name="toy")
        with tempfile.TemporaryDirectory() as tmp:
            d = rep.save(Path(tmp) / "toy", levels=("gyrus",))
            names = {p.name for p in d.iterdir()}
            for f in ("connections_annotated.csv", "relations.csv", "relations_by_tail.csv", "groups_lobe.csv",
                      "groups_network.csv", "brainmap_domains.csv", "distance_qc.csv", "coupling_by_tail.csv",
                      "evidence.json", "connectogram_gyrus.png", "connectogram_gyrus.pdf"):
                self.assertIn(f, names)

    def test_display_labels_do_not_change_the_result(self):
        from conninfpy import analyze

        rng = np.random.default_rng(0)
        n, k = len(POINTS), 20
        a = 0.3 + 0.1 * rng.normal(size=(k, n, n))
        a = (a + a.transpose(0, 2, 1)) / 2
        b = a + 0.1 * rng.normal(size=(k, n, n))
        b = (b + b.transpose(0, 2, 1)) / 2
        b[:, 0, 1] = b[:, 1, 0] = a[:, 0, 1] + 0.4 + 0.05 * rng.normal(size=k)
        for x in (a, b):
            for m in x:
                np.fill_diagonal(m, 0.0)
        res = analyze(group1=a, group2=b, input="z", test_type="paired", method="tstat",
                      n_permutations=99, rng=0, use_mp=False).inference
        before = list(res.design["comparison"])
        rep = interpret_result(res, self.rois, labels={"A": "eyes open", "B": "eyes closed"})
        self.assertEqual(rep.comparison, ("eyes open", "eyes closed"))
        self.assertEqual(list(res.design["comparison"]), before)
        self.assertIn("fc[eyes open]", rep.edges.columns)

    def test_reproduction_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stored.npz"
            np.savez(path, positive=self.res.positive, negative=self.res.negative)
            self.assertIn("max |Δp| = 0", reproduction_note(self.res, path))
            self.assertIn("same detections at α = 0.05: True", reproduction_note(self.res, path))
        self.assertEqual(reproduction_note(self.res, None), "no stored map")


class TestBundledProfiles(unittest.TestCase):

    def test_profiles_are_read_without_network(self):
        from unittest import mock

        from conninfpy.atlas import fetch_bna_behavioral_decoding

        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("no network expected")):
            bd, pc = fetch_bna_behavioral_decoding()
        self.assertEqual((len(bd), len(pc)), (246, 246))


if __name__ == "__main__":
    unittest.main()
