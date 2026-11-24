"""NETSTAT-37: mixed atlas fixtures and the honest limits of the spatial annotation.

The fixture is a synthetic 16x16x16 NIfTI (8 mm voxels, MNI-mm affine) with
six non-sequential label values and a user LUT, plus one label that the LUT
does not describe:

* three cortical ROIs (network ``Frontoparietal`` / ``Somatomotor`` and one
  ``unknown``: the LUT has no network there and none is invented),
* one cortical patch spanning three lobes (``mixed`` anatomically, network
  label ``mixed`` from the LUT),
* one cerebellum and one brainstem ROI (``not_applicable`` networks, no
  BNA-246 support: ``unassigned``),
* one label without any LUT entry (``region_599``: named by its id, network
  ``unassigned``, labelled share below ``min_coverage``).

What is pinned here:

* :meth:`conninfpy.AtlasInfo.from_nifti` keeps the original, non-sequential
  label values and every network label verbatim (no renumbering, no
  ``unknown`` -> invented network, no LUT entry -> invented name);
* ``annotate_rois`` -> ``annotate_edges`` -> ``summarize_annotation`` ->
  ``aggregate_connections`` keep the excluded and unassigned ROIs visible for
  the edge, component and block inference units, with consistent
  detected/tested/placed counts;
* the figures (connectogram footer, network/block heatmaps) name those ROIs
  and networks instead of dropping them;
* an unknown MNI variant is flagged ``approximate space`` without a numeric
  offset, and the FSL grid is not flagged.

No real data, no external atlas and no download is involved.
"""
from __future__ import annotations

import functools
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pandas as pd

from conninfpy.atlas import AtlasInfo
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.interpret.annotate import (aggregate_connections, annotate_edges, annotate_rois,
                                          summarize_annotation)
from conninfpy.interpret.evidence import (build_annotation_evidence, describe_coverage,
                                          validate_evidence)
from conninfpy.results import InferenceResult
from conninfpy.stats._components import build_component_report

# label value (deliberately non-sequential) -> name, network, LUT structure, voxels (MNI mm)
ROIS = [
    (7, "L dlPFC", "Frontoparietal", "cortex", [(-40, 32, 24)]),
    (23, "R dlPFC", "unknown", "cortex", [(40, 32, 24)]),
    (101, "L M1", "Somatomotor", "cortex", [(-40, -16, 56)]),
    (205, "mixed patch", "mixed", "cortex", [(16, 40, 16), (-16, -40, 48), (-56, -16, -24)]),
    (310, "L cerebellum", "not_applicable", "cerebellum", [(-16, -56, -40)]),
    (404, "brainstem", "not_applicable", "brainstem", [(0, -32, -32)]),
    (599, "region_599", "unassigned", "", [(-24, 0, -24), (0, 0, 24), (8, 8, 24)]),
]
NO_LUT_ID = 599                      # present in the image, absent from the LUT
VOXEL_MM = 8.0
ORIGIN_MM = -64.0
ALPHA = 0.05
N = len(ROIS)
DETECTED = [(0, 1), (1, 2), (0, 2), (0, 4)]      # (0, 4) touches an unplaced ROI
LABELS = [r[1] for r in ROIS]
NETWORKS = [r[2] for r in ROIS]
UNPLACED = ["L cerebellum", "brainstem", "region_599"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _image_and_lut(drop_lut_id=None):
    """Synthetic label image and LUT with the original, non-sequential ids."""
    affine = np.diag([VOXEL_MM, VOXEL_MM, VOXEL_MM, 1.0])
    affine[:3, 3] = ORIGIN_MM
    data = np.zeros((16, 16, 16), dtype=np.int16)
    for lid, _name, _net, _structure, voxels in ROIS:
        for x, y, z in voxels:
            ijk = np.rint((np.array([x, y, z], float) - affine[:3, 3]) / VOXEL_MM).astype(int)
            assert np.allclose(affine[:3, :3] @ ijk + affine[:3, 3], [x, y, z])
            data[tuple(ijk)] = lid
    lut = pd.DataFrame([
        {"id": lid, "name": name, "network": net, "structure": structure}
        for lid, name, net, structure, _voxels in ROIS if lid != drop_lut_id
    ])
    return nib.Nifti1Image(data, affine), lut


@functools.lru_cache(maxsize=None)
def _atlas():
    img, lut = _image_and_lut(drop_lut_id=NO_LUT_ID)     # region_599 has no LUT record
    return AtlasInfo.from_nifti(img, lut=lut, source="synthetic mixed atlas (NETSTAT-37 fixture)")


@functools.lru_cache(maxsize=None)
def _reference():
    """Bundled BNA-246 only: an empty data folder disables the local-file lookup."""
    return BrainnetomeReference.load(data_dir=tempfile.mkdtemp(), profiles=False,
                                     cache_dir=tempfile.mkdtemp())


@functools.lru_cache(maxsize=None)
def _annotations():
    atlas, ref = _atlas(), _reference()
    parcel = annotate_rois(support="parcel", parcel_image=atlas.image,
                           parcel_labels=list(atlas.label_ids), names=list(atlas.labels),
                           networks=list(atlas.networks), reference=ref)
    sphere = annotate_rois(atlas.coords, support="sphere", radius_mm=6, names=list(atlas.labels),
                           networks=list(atlas.networks), reference=ref)
    return parcel, sphere


def _edge_p_maps():
    p = np.ones((N, N))
    for a, b in DETECTED:
        p[a, b] = p[b, a] = 0.01
    stat = np.zeros((N, N))
    for a, b in DETECTED:
        stat[a, b] = stat[b, a] = 4.0
    return p, stat


@functools.lru_cache(maxsize=None)
def _results():
    """The same detections read at the three inference units (edge, component, block)."""
    p, stat = _edge_p_maps()
    zeros = np.zeros((N, N))
    edge = InferenceResult(p, np.ones((N, N)), method="tstat", stat_positive=stat, stat_negative=zeros)
    report = build_component_report({"positive": stat}, {"positive": p}, method="nbs",
                                    threshold=3.0, nbs_stat="extent")
    component = InferenceResult(p, np.ones((N, N)), method="nbs", stat_positive=stat,
                                stat_negative=zeros, component_report=report)
    block = InferenceResult(p, np.ones((N, N)), method="cnbs", stat_positive=stat,
                            stat_negative=zeros, block_report={
                                "unit": "block", "correction": "bonferroni",
                                "node_labels": _atlas().network_index().tolist(),
                                "tested_tail": None,
                                "blocks": [{"network_a": 0, "network_b": 0, "n_edges": 3,
                                            "p_positive": 0.01}]})
    return {"edge": edge, "component": component, "block": block}


class _MatplotlibCase(unittest.TestCase):

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")


# ---------------------------------------------------------------------------
# Generic NIfTI + user LUT path
# ---------------------------------------------------------------------------
class TestGenericNiftiLutPath(unittest.TestCase):
    """Original ids, LUT names and network labels survive ``AtlasInfo.from_nifti``."""

    def test_original_label_ids_are_kept_without_renumbering(self):
        atlas = _atlas()
        self.assertEqual(list(atlas.label_ids), [r[0] for r in ROIS])
        self.assertEqual(list(atlas.labels), LABELS)
        self.assertTrue((np.diff(atlas.label_ids) > 0).all())       # ascending, as nilearn extracts
        self.assertEqual(len(atlas), N)
        self.assertIsNotNone(atlas.image)

    def test_lut_names_and_network_labels_are_verbatim(self):
        atlas = _atlas()
        self.assertEqual(list(atlas.networks), NETWORKS)
        self.assertIn("unknown", atlas.networks)                    # not replaced by a BNA network
        self.assertEqual(atlas.networks.count("not_applicable"), 2)
        self.assertIn("mixed", atlas.networks)

    def test_label_without_lut_record_is_named_by_its_id_and_not_invented(self):
        atlas = _atlas()
        row = list(atlas.labels).index("region_599")
        self.assertEqual(int(atlas.label_ids[row]), NO_LUT_ID)
        self.assertEqual(atlas.networks[row], "unassigned")
        self.assertNotIn(NO_LUT_ID, _image_and_lut(drop_lut_id=NO_LUT_ID)[1]["id"].tolist())

    def test_structure_column_is_not_carried_by_atlas_info(self):
        # A documented limitation, not a silent substitution: AtlasInfo has no
        # structure field, so the LUT's cortex/subcortex/... column is dropped
        # while names, ids and networks are kept.
        atlas = _atlas()
        self.assertFalse(hasattr(atlas, "structure"))
        self.assertEqual(len(atlas.networks), len(atlas.labels))

    def test_lut_without_network_column_makes_each_region_its_own_group(self):
        img, lut = _image_and_lut(drop_lut_id=NO_LUT_ID)
        names_only = dict(zip(lut["id"], lut["name"]))
        atlas = AtlasInfo.from_nifti(img, lut=names_only)
        self.assertEqual(list(atlas.networks), list(atlas.labels))

    def test_exclude_drops_the_region_and_its_voxels(self):
        img, lut = _image_and_lut(drop_lut_id=NO_LUT_ID)
        atlas = AtlasInfo.from_nifti(img, lut=lut, exclude=["Brainstem"])
        self.assertNotIn("brainstem", atlas.labels)
        self.assertNotIn(404, list(atlas.label_ids))
        self.assertFalse((np.asarray(atlas.image.dataobj) == 404).any())
        with self.assertRaisesRegex(ValueError, "no region named"):
            AtlasInfo.from_nifti(img, lut=lut, exclude=["not in this atlas"])


# ---------------------------------------------------------------------------
# Spatial limit: flagged, never quantified
# ---------------------------------------------------------------------------
class TestSpatialLimitIsHonest(unittest.TestCase):
    """An unknown MNI variant is approximate; no numeric offset is promised."""

    def _flags(self, space):
        rois = annotate_rois(_atlas().coords, support="sphere", radius_mm=6,
                             networks=NETWORKS, reference=_reference(), space=space)
        return rois.table["flags"].fillna("").tolist()

    def test_unknown_variant_is_flagged_approximate_without_a_number(self):
        for flags in self._flags("MNI152"):
            self.assertIn("approximate space", flags)
            self.assertIn("unquantified", flags)
            self.assertNotIn("1-3", flags)
            self.assertNotIn("1–3", flags)
            self.assertNotIn(" mm)", flags)                         # no millimetre offset at all

    def test_fsl_grid_is_not_flagged(self):
        for flags in self._flags("MNI152NLin6Asym"):
            self.assertNotIn("approximate space", flags)

    def test_sphere_radius_stays_checkable(self):
        _parcel, sphere = _annotations()
        t = sphere.table.set_index("roi")
        self.assertTrue((t["radius_mm"] == 6.0).all())
        edges = annotate_edges(pd.DataFrame({"roi_i": [0], "roi_j": [1]}), sphere)
        # the recorded radius is what the overlap warning is based on
        self.assertTrue(0.0 <= float(edges.loc[0, "shared_support"]) <= 1.0)

    def test_app_help_promises_no_numeric_offset(self):
        try:
            from apps.components.bna_annotation import _SPACE_HELP
        except ImportError as exc:                              # pragma: no cover - streamlit absent
            self.skipTest(f"app module not importable: {exc}")
        self.assertIn("unquantified", _SPACE_HELP)
        self.assertIn("approximate", _SPACE_HELP)
        for bad in ("1-3 mm", "1–3 mm", "about 1"):
            self.assertNotIn(bad, _SPACE_HELP)

    def test_usage_docs_state_no_numeric_offset(self):
        docs = Path(__file__).resolve().parents[1] / "docs" / "source" / "usage.rst"
        text = docs.read_text(encoding="utf-8")
        self.assertIn("unquantified amount", text)
        self.assertNotIn("differ by about 1–3 mm", text)
        self.assertNotIn("differ by about 1-3 mm", text)


# ---------------------------------------------------------------------------
# Mixed annotation of ROIs and edges (parcel and sphere supports)
# ---------------------------------------------------------------------------
class TestMixedAnnotation(unittest.TestCase):
    """Excluded ROIs stay visible and unknown labels stay unknown."""

    @classmethod
    def setUpClass(cls):
        cls.parcel, cls.sphere = _annotations()
        cls.t = cls.parcel.table.set_index("roi")

    def test_parcel_path_keeps_the_user_network_labels(self):
        self.assertEqual(list(self.parcel.table["network"]), NETWORKS)
        self.assertEqual(self.t.loc["R dlPFC", "network"], "unknown")
        self.assertEqual(self.t.loc["L cerebellum", "network"], "not_applicable")

    def test_unknown_roi_gets_no_invented_network_or_function(self):
        r = self.t.loc["R dlPFC"]
        self.assertEqual(r["network"], "unknown")
        self.assertEqual(r["level"], "subregion")                 # anatomy is from the BNA overlap
        self.assertEqual(r["yeo7"], "Frontoparietal")             # the BNA crosswalk, kept as reference
        self.assertEqual(r["profile_status"], "profiles unavailable")
        self.assertEqual(r["domains"], "")

    def test_excluded_and_mixed_rois_are_their_own_rows(self):
        self.assertEqual(self.t.loc["L cerebellum", "level"], "unassigned")
        self.assertEqual(self.t.loc["brainstem", "level"], "unassigned")
        self.assertEqual(float(self.t.loc["L cerebellum", "coverage"]), 0.0)
        mixed = self.t.loc["mixed patch"]
        self.assertEqual(mixed["level"], "mixed")
        self.assertLess(mixed["region_purity"], 0.5)              # no lobe holds the purity threshold
        self.assertEqual(self.t.loc["region_599", "level"], "unassigned")
        low = float(self.t.loc["region_599", "coverage"])
        self.assertTrue(0.0 < low < 0.5)

    def test_sphere_path_matches_the_same_labels(self):
        self.assertEqual(list(self.sphere.table["network"]), NETWORKS)
        self.assertTrue((self.sphere.table["radius_mm"] == 6.0).all())
        self.assertEqual(list(self.sphere.table["roi"]), LABELS)

    def test_edges_do_not_name_a_network_for_unknown_or_not_applicable(self):
        pairs = pd.DataFrame({"roi_i": [0, 0, 0, 4, 3], "roi_j": [2, 1, 4, 5, 3]})
        e = annotate_edges(pairs, self.parcel)
        # (0, 2) is a real between-network pair; the rest have no network at one endpoint
        self.assertEqual(list(e["network_relation"]),
                         ["between", "n/a", "n/a", "n/a", "n/a"])
        self.assertEqual(list(e["i_roi"]), [LABELS[0], LABELS[0], LABELS[0], LABELS[4], LABELS[3]])
        self.assertEqual(e.loc[3, "anatomical_relation"], "unassigned")
        self.assertEqual(e.loc[4, "anatomical_relation"], "mixed")


# ---------------------------------------------------------------------------
# End to end at the three inference units
# ---------------------------------------------------------------------------
class TestUnitsEndToEnd(_MatplotlibCase):
    """ROI restriction, counts and figures agree for edge, component and block results."""

    @classmethod
    def setUpClass(cls):
        cls.parcel, _ = _annotations()
        cls.results = _results()

    def test_inference_unit_is_preserved(self):
        for unit, result in self.results.items():
            self.assertEqual(result.inference_unit, unit)
            summary = summarize_annotation(result, self.parcel, alpha=ALPHA, tail="both")
            self.assertEqual(summary["meta"]["inference_unit"], unit)
            self.assertEqual(summary["meta"]["roi_restriction"]["applies_to"], unit)

    def test_roi_restriction_names_the_excluded_rois(self):
        for unit, result in self.results.items():
            meta = summarize_annotation(result, self.parcel, alpha=ALPHA, tail="both")["meta"]
            restriction = meta["roi_restriction"]
            self.assertEqual(restriction["unassigned_rois"], ["L cerebellum", "brainstem", "region_599"])
            self.assertEqual(restriction["mixed_rois"], ["mixed patch"])
            self.assertEqual(restriction["counts"]["n_rois"], N)
            self.assertIn(unit, restriction["statement"])

    def test_summary_groups_keep_unassigned_and_mixed(self):
        for result in self.results.values():
            regions = summarize_annotation(result, self.parcel, alpha=ALPHA, tail="both")["regions"]
            groups = set(regions["group"])
            self.assertIn("unassigned", groups)
            self.assertIn("mixed", groups)
            self.assertEqual(int(regions.set_index("group").loc["unassigned", "n_rois"]), 3)
            self.assertEqual(int(regions.set_index("group").loc["mixed", "n_rois"]), 1)

    def test_coverage_counts_agree_with_the_summary(self):
        for unit, result in self.results.items():
            summary = summarize_annotation(result, self.parcel, alpha=ALPHA, tail="both")
            agg = aggregate_connections(result, self.parcel, tail="both", alpha=ALPHA,
                                        assign="dominant")
            cov = agg.attrs["coverage"]
            self.assertEqual(cov["inference_unit"], unit)
            self.assertEqual((cov["n_rois"], cov["n_placed"]), (N, N - len(UNPLACED)))
            self.assertEqual(cov["unplaced_no_support"], ["L cerebellum", "brainstem"])
            self.assertEqual(cov["unplaced_low_coverage"], ["region_599"])
            self.assertEqual(cov["unplaced_rois"], UNPLACED)
            self.assertEqual(cov["detected_total"], summary["meta"]["n_detected"])
            self.assertEqual(cov["tested_total"], summary["meta"]["n_tested"])
            self.assertEqual(cov["detected_total"], len(DETECTED))
            self.assertEqual(cov["detected_placed"], 3)          # (0, 4) touches the cerebellum
            self.assertEqual(cov["tested_total"], N * (N - 1) // 2)
            self.assertEqual(cov["tested_placed"], (N - 3) * (N - 4) // 2)

    def test_aggregation_keeps_every_network_label(self):
        for result in self.results.values():
            agg = aggregate_connections(result, self.parcel, tail="both", alpha=ALPHA,
                                        assign="dominant")
            used = set(agg["network_a"]) | set(agg["network_b"])
            self.assertTrue(used <= set(NETWORKS))
            self.assertIn("unknown", used)          # not folded into a named network
            self.assertNotIn("Subcortical", used)

    def test_evidence_packet_records_the_unit_and_the_excluded_rois(self):
        for unit, result in self.results.items():
            summary = summarize_annotation(result, self.parcel, alpha=ALPHA, tail="both")
            summary["edges"]["detected"] = (
                (summary["edges"]["p_positive"] <= ALPHA) | (summary["edges"]["p_negative"] <= ALPHA))
            detected = summary["edges"][summary["edges"]["detected"]]
            packet = build_annotation_evidence(detected, self.parcel, contrast_name="fixture",
                                               inference_unit=unit, summary=summary)
            validate_evidence(packet)
            restriction = packet["query"]["roi_restriction"]
            self.assertEqual(restriction["applies_to"], unit)
            self.assertEqual(restriction["unassigned_rois"], UNPLACED)
            self.assertEqual(restriction["mixed_rois"], ["mixed patch"])
            self.assertEqual(restriction["counts"]["n_endpoint_rois"], len(packet["rois"]))
            cov = describe_coverage(packet)
            self.assertIn(f"inference unit: {unit}", cov["report_sentence"])
            self.assertEqual(cov["inference_unit"], unit)
            self.assertEqual(cov["roi_restriction"]["applies_to"], unit)
            self.assertIn("unassigned and mixed ROIs keep their own group", cov["report_sentence"])

    def test_detected_component_members_keep_their_unplaced_endpoint(self):
        result = self.results["component"]
        summary = summarize_annotation(result, self.parcel, alpha=ALPHA, tail="both")
        edges = summary["edges"]
        self.assertIn(LABELS[4], set(edges["i_roi"]) | set(edges["j_roi"]))
        component_edges = edges[edges["t_signed"] > 0]
        self.assertEqual(len(component_edges), len(DETECTED))

    def test_connectogram_footer_reports_placement(self):
        from conninfpy.plot import plot_connectogram

        for result in self.results.values():
            ax = plot_connectogram(result, self.parcel, tail="positive", level="gyrus",
                                   assign="dominant")
            footer = ax.texts[-1].get_text()
            self.assertIn(f"{N - len(UNPLACED)} of {N} ROIs placed", footer)
            self.assertIn("2 outside the atlas, 1 below min_coverage", footer)
            self.assertIn("1 detected connection touches an unplaced ROI", footer)

    def test_network_and_block_figures_keep_placeholder_labels(self):
        from conninfpy.plot import plot_block_mass, plot_network_summary

        result = self.results["edge"]
        p, _stat = _edge_p_maps()
        net_atlas = AtlasInfo(labels=LABELS, networks=NETWORKS)
        distinct = list(dict.fromkeys(NETWORKS))        # network_index coding order
        ax = plot_block_mass(p, net_atlas.network_index(), network_names=distinct)
        self.assertEqual({t.get_text() for t in ax.get_xticklabels()}, set(distinct))

        ax = plot_network_summary(result, atlas=net_atlas)
        self.assertEqual({t.get_text() for t in ax.get_xticklabels()}, set(distinct))
        self.assertIn("unknown", {t.get_text() for t in ax.get_yticklabels()})

    def test_atlas_from_annotation_keeps_the_placeholder_groups(self):
        atlas = self.parcel.to_atlas(level="network")
        self.assertEqual(list(atlas.networks), NETWORKS)
        self.assertEqual(len(set(atlas.network_index())), len(set(NETWORKS)))


if __name__ == "__main__":                                   # pragma: no cover
    unittest.main()
