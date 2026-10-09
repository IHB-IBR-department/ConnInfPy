"""Brainnetome annotation of ROIs and connections (conninfpy.annotate).

Uses the bundled BNA-246 label image; no network access (BrainMap profiles are
injected where needed). Checks that need the local probabilistic or
tractography maps are skipped without them.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from conninfpy.annotate import _lens_share, annotate_edges, annotate_rois, summarize_annotation
from conninfpy.atlas import get_bna_246_nifti_path
from conninfpy.brainnetome import BrainnetomeReference
from conninfpy.results import InferenceResult

DATA = Path(__file__).resolve().parents[1] / "datasets" / "atlases"
HAS_PM = (DATA / "BNA_PM_4D.nii.gz").exists()
HAS_SC = (DATA / "BNA_SC_3D_246.zip").exists()

POINTS = {
    "L dlPFC": (-42, 36, 24), "R dlPFC": (42, 36, 24), "L amygdala": (-22, -4, -18),
    "L M1 hand": (-38, -22, 56), "L putamen": (-26, 2, 2), "L cerebellum": (-20, -60, -30),
    "L dlPFC near": (-40, 32, 26),
}


def _reference(data_dir=None):
    """Bundled atlas only (an empty folder disables the local-file lookup), or with local files.

    Local-file references use the default cache, so the probabilistic-map cache
    is built once per machine (a few minutes) and reused.
    """
    if data_dir is None:
        return BrainnetomeReference.load(data_dir=tempfile.mkdtemp(), profiles=False,
                                         cache_dir=tempfile.mkdtemp())
    return BrainnetomeReference.load(data_dir=data_dir, profiles=False)


class TestRoiAnnotation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref = _reference()
        cls.rois = annotate_rois(list(POINTS.values()), names=list(POINTS), reference=cls.ref,
                                 support="sphere", radius_mm=6)
        cls.t = cls.rois.table.set_index("roi")

    def test_pure_region_is_assigned_at_subregion_level(self):
        r = self.t.loc["L dlPFC"]
        self.assertEqual((r["level"], r["assigned"], r["hemisphere"]), ("subregion", "A9/46v_L", "L"))
        self.assertAlmostEqual(r["region_purity"], 1.0)
        self.assertEqual(r["description"], "A9/46v, ventral area 9/46")

    def test_mixed_sphere_is_assigned_only_at_the_pure_level(self):
        amyg = self.t.loc["L amygdala"]          # mAmyg 44%, rHipp 30%, lAmyg 19%
        self.assertEqual((amyg["level"], amyg["assigned"], amyg["structure"]), ("gyrus", "Amyg_L", "amygdala"))
        put = self.t.loc["L putamen"]            # centre in GP, sphere mostly putamen
        self.assertEqual(self.ref.names(self.ref.labels_at([POINTS["L putamen"]]))[0], "GP_L")
        self.assertEqual((put["level"], put["assigned"]), ("gyrus", "BG_L"))
        self.assertIn("dlPu_L 0.43", put["regions"])

    def test_outside_the_atlas_is_unassigned(self):
        r = self.t.loc["L cerebellum"]
        self.assertEqual(r["level"], "unassigned")
        self.assertIn("cerebellum", r["flags"])
        self.assertEqual(self.rois.weights[list(POINTS).index("L cerebellum")].sum(), 0.0)

    def test_weights_and_grouping(self):
        np.testing.assert_allclose(self.rois.weights.sum(1), self.t["coverage"].to_numpy())
        groups = dict(zip(POINTS, self.rois.grouping("lobe")))
        self.assertEqual(groups["L dlPFC"], "Frontal_L")
        self.assertEqual(groups["L cerebellum"], "unassigned")
        self.assertEqual(dict(zip(POINTS, self.rois.grouping("subregion")))["L amygdala"], "mixed")
        atlas = self.rois.to_atlas("lobe")
        self.assertEqual(atlas.networks, self.rois.grouping("lobe"))

    def test_point_support_and_bounded_nearest_voxel(self):
        wm = [(-28, -10, 30)]
        r = annotate_rois(wm, reference=self.ref, support="point").table.iloc[0]
        self.assertEqual(r["level"], "unassigned")
        self.assertIn("point estimate", r["flags"])
        r = annotate_rois(wm, reference=self.ref, support="point", nearest_mm=10).table.iloc[0]
        self.assertNotEqual(r["level"], "unassigned")
        self.assertIn("nearest labelled voxel", r["flags"])

    def test_parcel_support(self):
        img = get_bna_246_nifti_path(2)            # BNA itself as the 'parcellation'
        t = annotate_rois(reference=self.ref, support="parcel", parcel_image=img,
                          parcel_labels=[1, 2, 211]).table
        self.assertEqual(list(t["region"]), ["A8m_L", "A8m_R", "mAmyg_L"])
        self.assertTrue((t["region_purity"] > 0.9).all())

    def test_space_is_declared_not_guessed(self):
        with self.assertRaises(NotImplementedError):
            annotate_rois([(0, 0, 0)], reference=self.ref, support="point", space="Talairach")
        with self.assertRaises(ValueError):
            annotate_rois([(0, 0, 0)], reference=self.ref, support="point", space="mni-ish")
        r = annotate_rois([POINTS["L dlPFC"]], reference=self.ref, support="sphere", radius_mm=6,
                          space="MNI152NLin2009cAsym").table.iloc[0]
        self.assertIn("approximate space", r["flags"])

    def test_profiles_are_listed_not_combined(self):
        ref = _reference()
        ref.profiles = {"BD": {k: {} for k in range(1, 247)}, "PC": {k: {} for k in range(1, 247)}}
        ref.profiles["BD"][21] = {"Cognition.Memory.Working": 2.5, "Action.Execution": 1.4}   # A9/46v_L
        rois = annotate_rois([POINTS["L dlPFC"], POINTS["L amygdala"]], reference=ref,
                             support="sphere", radius_mm=6)
        t = rois.table
        self.assertEqual(t.loc[0, "domains"], "Action; Cognition")
        self.assertEqual(t.loc[0, "profile_status"], "terms")
        self.assertTrue(t.loc[1, "profile_status"].startswith("no published terms"))
        long = rois.profiles("BD")
        self.assertEqual(sorted(long["term"]), ["Action.Execution", "Cognition.Memory.Working"])
        self.assertEqual(set(long["likelihood_ratio"]), {2.5, 1.4})


class TestEdges(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rois = annotate_rois(list(POINTS.values()), names=list(POINTS), reference=_reference(),
                                 support="sphere", radius_mm=6)

    def test_relations(self):
        names = list(POINTS)
        pairs = pd.DataFrame({"roi_i": [names.index("L dlPFC"), names.index("L dlPFC"), names.index("L M1 hand")],
                              "roi_j": [names.index("R dlPFC"), names.index("L dlPFC near"), names.index("L putamen")]})
        e = annotate_edges(pairs, self.rois, smoothing_fwhm_mm=6)
        self.assertEqual(list(e["hemispheric_relation"]),
                         ["homotopic", "left intrahemispheric", "left intrahemispheric"])
        self.assertEqual(list(e["anatomical_relation"]), ["same subregion", "same subregion", "between lobes"])
        self.assertEqual(e.loc[2, "circuit"], "cortico-basal ganglia")
        self.assertEqual(e.loc[0, "network_relation"], "within Frontoparietal")
        self.assertEqual(e.loc[0, "shared_support"], 0.0)
        self.assertGreater(e.loc[1, "shared_support"], 0.3)          # 4.9 mm apart, r = 6
        self.assertTrue(e.loc[1, "within_smoothing"])

    def test_lens_share(self):
        r, d = 6.0, 4.0
        expected = np.pi * (4 * r + d) * (2 * r - d) ** 2 / 12 / (4 / 3 * np.pi * r ** 3)
        self.assertAlmostEqual(_lens_share(d, r, r), expected)
        self.assertEqual(_lens_share(0.0, r, r), 1.0)
        self.assertEqual(_lens_share(12.0, r, r), 0.0)
        self.assertEqual(_lens_share(1.0, 0.0, r), 0.0)

    @unittest.skipUnless(HAS_SC, "local BNA_SC_3D_246.zip not available")
    def test_structural_support_tracks_known_pathways(self):
        rois = annotate_rois([(-38, -22, 56), (-26, 2, 2), (-8, -84, 4)], reference=_reference(data_dir=DATA),
                             support="sphere", radius_mm=6)
        e = annotate_edges(pd.DataFrame({"roi_i": [0, 0], "roi_j": [1, 2]}), rois)
        self.assertGreater(e.loc[0, "sc"], 5 * e.loc[1, "sc"])     # M1-putamen vs M1-V1


@unittest.skipUnless(HAS_PM, "local BNA_PM_4D.nii.gz not available")
class TestMembership(unittest.TestCase):

    def test_probabilistic_membership_of_amygdala_sphere(self):
        rois = annotate_rois([POINTS["L amygdala"]], reference=_reference(data_dir=DATA),
                             support="sphere", radius_mm=6)
        m = rois.membership[0]
        self.assertGreater(m[210] + m[212], 0.8)                     # mAmyg_L + lAmyg_L


class TestSummary(unittest.TestCase):

    def test_detections_are_described_against_tested_opportunities(self):
        rois = annotate_rois(list(POINTS.values()), names=list(POINTS), reference=_reference(),
                             support="sphere", radius_mm=6)
        n = len(POINTS)
        p = np.ones((n, n))
        names = list(POINTS)
        a, b = names.index("L dlPFC"), names.index("R dlPFC")
        p[a, b] = p[b, a] = 0.01
        stat = np.zeros((n, n))
        stat[a, b] = stat[b, a] = 4.0
        res = InferenceResult(p, np.ones((n, n)), method="tstat", stat_positive=stat,
                              stat_negative=np.zeros((n, n)))
        s = summarize_annotation(res, rois, alpha=0.05)
        rel = s["relations"]
        for _, g in rel.groupby("relation"):
            self.assertEqual(g["n_tested"].sum(), n * (n - 1) // 2)
            self.assertEqual(g["n_detected"].sum(), 1)
            self.assertAlmostEqual(g["share_of_tested"].sum(), 1.0)
        hom = rel[(rel.relation == "hemispheric_relation") & (rel.category == "homotopic")].iloc[0]
        self.assertEqual(hom["n_detected"], 1)
        self.assertEqual(s["meta"]["n_detected"], 1)
        self.assertEqual(s["distance_qc"].loc[0, "share_detections_short"], 0.0)   # 84 mm apart


if __name__ == "__main__":
    unittest.main()


class TestResultMethods(unittest.TestCase):

    def test_annotated_edges_and_removed_decoder(self):
        rois = annotate_rois([(-42, 36, 24), (42, 36, 24), (-22, -4, -18)], reference=_reference(),
                             support="sphere", radius_mm=6)
        p = np.ones((3, 3))
        p[0, 1] = p[1, 0] = 0.01
        res = InferenceResult(p, np.ones((3, 3)), method="tstat", stat_positive=np.where(p < 1, 4.0, 0.0),
                              stat_negative=np.zeros((3, 3)))
        e = res.annotated_edges(rois)
        self.assertEqual(len(e), 1)
        self.assertEqual(e.loc[0, "hemispheric_relation"], "homotopic")
        self.assertEqual(e.loc[0, "p_positive"], 0.01)              # inference columns unchanged
        with self.assertRaisesRegex(NotImplementedError, "annotate_rois"):
            res.decoded_edges()
        with self.assertRaisesRegex(ValueError, "ROIs"):
            res.annotated_edges(annotate_rois([(-42, 36, 24)], reference=_reference(),
                                              support="sphere", radius_mm=6))
