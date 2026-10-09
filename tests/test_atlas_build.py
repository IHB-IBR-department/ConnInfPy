"""
Atlases built from label images and coordinate-defined ROIs.

A synthetic 2 mm label image with an MNI-like affine (negative x step) holds
four box-shaped regions; coordinates inside, near and far from them check
the placement rules, and the grouping feeds ``summarize_by_network``.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

nib = __import__("importlib").import_module("nibabel")

from conninfpy.atlas import AtlasInfo, assign_coordinates
from conninfpy.results import summarize_by_network


AFFINE = np.array([[-2.0, 0, 0, 40.0],
                   [0, 2.0, 0, -40.0],
                   [0, 0, 2.0, -20.0],
                   [0, 0, 0, 1.0]])


def _label_image(dtype=np.int16):
    """40x40x20 grid; regions 1/2 left (x > 20 vox -> x < 0 mm), 3/4 right."""
    data = np.zeros((40, 40, 20), dtype=dtype)
    data[24:30, 5:11, 5:11] = 1     # left
    data[24:30, 20:26, 5:11] = 2    # left
    data[10:16, 5:11, 5:11] = 3     # right
    data[10:16, 20:26, 5:11] = 7    # right; non-consecutive label value
    return nib.Nifti1Image(data, AFFINE)


def _world(i, j, k):
    return (AFFINE @ np.array([i, j, k, 1.0]))[:3]


LUT = pd.DataFrame({"index": [0, 1, 2, 3, 7],
                    "name": ["Background", "L_A", "L_B", "R_A", "R_B"],
                    "network": ["", "Net1", "Net2", "Net1", "Net2"]})


class TestFromNifti(unittest.TestCase):

    def test_regions_in_label_order_with_lut_names_and_networks(self):
        atlas = AtlasInfo.from_nifti(_label_image(), lut=LUT)
        self.assertEqual(atlas.labels, ["L_A", "L_B", "R_A", "R_B"])
        self.assertEqual(atlas.networks, ["Net1", "Net2", "Net1", "Net2"])
        np.testing.assert_array_equal(atlas.label_ids, [1, 2, 3, 7])

    def test_centroids_are_in_world_coordinates(self):
        atlas = AtlasInfo.from_nifti(_label_image())
        np.testing.assert_allclose(atlas.coords[0], _world(26.5, 7.5, 7.5))
        self.assertEqual(atlas.hemisphere, ["L", "L", "R", "R"])

    def test_without_names_or_networks_each_region_is_its_group(self):
        atlas = AtlasInfo.from_nifti(_label_image())
        self.assertEqual(atlas.labels, ["region_1", "region_2", "region_3", "region_7"])
        self.assertEqual(atlas.networks, atlas.labels)

    def test_float_label_image_and_network_mapping(self):
        atlas = AtlasInfo.from_nifti(_label_image(np.float32), lut={1: "a", 2: "b", 3: "c", 7: "d"},
                                     networks=lambda name: "left" if name in "ab" else "right")
        self.assertEqual(atlas.networks, ["left", "left", "right", "right"])

    def test_exclude_drops_region_from_atlas_and_image(self):
        atlas = AtlasInfo.from_nifti(_label_image(), lut=LUT, exclude=["L_B"])
        self.assertEqual(atlas.labels, ["L_A", "R_A", "R_B"])
        self.assertNotIn(2, np.unique(np.asarray(atlas.image.dataobj)))
        with self.assertRaisesRegex(ValueError, "no region named"):
            AtlasInfo.from_nifti(_label_image(), lut=LUT, exclude=["nope"])

    def test_rejects_non_integer_and_4d_images(self):
        data = np.asarray(_label_image(np.float32).dataobj) + 0.3
        with self.assertRaisesRegex(ValueError, "integer labels"):
            AtlasInfo.from_nifti(nib.Nifti1Image(data, AFFINE))
        with self.assertRaisesRegex(ValueError, "3-D label image"):
            AtlasInfo.from_nifti(nib.Nifti1Image(np.zeros((4, 4, 4, 3)), AFFINE))

    def test_freesurfer_style_lut_file(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("# id name r g b a\n1 alpha 0 0 0 0\n2 beta 0 0 0 0\n3 gamma 0 0 0 0\n7 delta 0 0 0 0\n")
        try:
            atlas = AtlasInfo.from_nifti(_label_image(), lut=f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(atlas.labels, ["alpha", "beta", "gamma", "delta"])


class TestFromNilearnBunch(unittest.TestCase):

    def test_schaefer_names_give_yeo_networks(self):
        lut = pd.DataFrame({"index": [0, 1, 2, 3, 7],
                            "name": ["Background", "7Networks_LH_Vis_1", "7Networks_LH_Default_PFC_1",
                                     "7Networks_RH_Vis_1", "7Networks_RH_SomMot_2"]})
        atlas = AtlasInfo.from_nilearn({"maps": _label_image(), "lut": lut, "atlas_type": "deterministic"})
        self.assertEqual(atlas.networks, ["Visual", "Default", "Visual", "SomMot"])

    def test_probabilistic_and_lutless_atlases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "probabilistic"):
            AtlasInfo.from_nilearn({"maps": _label_image(), "lut": LUT, "atlas_type": "probabilistic"})
        with self.assertRaisesRegex(ValueError, "nilearn >= 0.13"):
            AtlasInfo.from_nilearn({"maps": _label_image(), "labels": ["a"]})


class TestFromCoordinates(unittest.TestCase):

    def setUp(self):
        self.atlas = AtlasInfo.from_nifti(_label_image(), lut=LUT)
        self.inside = _world(27, 8, 8)            # region L_A
        self.near = _world(23, 22, 8)             # 1 voxel (2 mm) outside L_B
        self.far = _world(2, 38, 18)              # far from everything

    def test_inside_near_and_far(self):
        t = assign_coordinates([self.inside, self.near, self.far], self.atlas)
        self.assertEqual(list(t["region"]), ["L_A", "L_B", "unassigned"])
        self.assertEqual(list(t["method"]), ["voxel", "nearest voxel", "unassigned"])
        self.assertEqual(list(t["inside"]), [True, False, False])
        self.assertAlmostEqual(t["distance_mm"].iloc[1], 2.0)

    def test_max_distance_controls_nearest_assignment(self):
        t = assign_coordinates([self.near], self.atlas, max_distance=1.0)
        self.assertEqual(t["region"].iloc[0], "unassigned")

    def test_groups_by_network_when_the_reference_has_networks(self):
        rois = AtlasInfo.from_coordinates([self.inside, self.near, self.far], self.atlas)
        self.assertEqual(rois.networks, ["Net1", "Net2", "unassigned"])
        self.assertEqual(rois.labels, ["ROI_1", "ROI_2", "ROI_3"])
        self.assertEqual(list(rois.assignment["roi"]), rois.labels)
        rois = AtlasInfo.from_coordinates([self.inside, self.near], self.atlas, group_by="region")
        self.assertEqual(rois.networks, ["L_A", "L_B"])

    def test_groups_by_region_when_the_reference_has_none(self):
        rois = AtlasInfo.from_coordinates([self.inside, self.near], AtlasInfo.from_nifti(_label_image()))
        self.assertEqual(rois.networks, ["region_1", "region_2"])

    def test_given_networks_override_the_grouping(self):
        rois = AtlasInfo.from_coordinates([self.inside, self.near], self.atlas, networks=["DMN", "DMN"])
        self.assertEqual(rois.networks, ["DMN", "DMN"])
        self.assertEqual(list(rois.assignment["region"]), ["L_A", "L_B"])

    def test_list_of_references_falls_through_in_order(self):
        left_only = AtlasInfo.from_nifti(_label_image(), lut=LUT, exclude=["R_A", "R_B"])
        right_only = AtlasInfo.from_nifti(_label_image(), lut=LUT, exclude=["L_A", "L_B"])
        right_point = _world(12, 8, 8)            # inside R_A
        t = assign_coordinates([self.inside, right_point], [left_only, right_only])
        self.assertEqual(list(t["region"]), ["L_A", "R_A"])
        self.assertEqual(list(t["atlas"]), [0, 1])
        self.assertEqual(list(t["method"]), ["voxel", "voxel"])

    def test_atlas_without_image_uses_centroids_with_a_warning(self):
        no_image = AtlasInfo(labels=self.atlas.labels, networks=self.atlas.networks, coords=self.atlas.coords)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t = assign_coordinates([self.inside], no_image)
        self.assertTrue(any("nearest region centroid" in str(x.message) for x in w))
        self.assertEqual(t["method"].iloc[0], "centroid")
        self.assertEqual(t["region"].iloc[0], "L_A")

    def test_network_summary_aggregates_coordinate_rois(self):
        coords = [self.inside, _world(26, 9, 8), self.near, _world(12, 8, 8)]  # Net1, Net1, Net2, Net1
        rois = AtlasInfo.from_coordinates(coords, self.atlas)
        p = np.ones((4, 4))
        p[0, 1] = p[1, 0] = 0.01      # within Net1
        p[0, 2] = p[2, 0] = 0.01      # Net1-Net2
        summary = summarize_by_network({"positive": p}, rois)
        within = summary[(summary.network_a == "Net1") & (summary.network_b == "Net1")]
        self.assertEqual(int(within["n_edges"].iloc[0]), 3)
        self.assertEqual(int(within["n_sig_positive"].iloc[0]), 1)

    def test_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            AtlasInfo.from_coordinates([[1, 2]], self.atlas)
        with self.assertRaisesRegex(ValueError, "reference atlas or the ROI networks"):
            AtlasInfo.from_coordinates([self.inside])


class TestBundledBrainnetomeImage(unittest.TestCase):

    def test_with_image_matches_the_bundled_table(self):
        with_image = AtlasInfo.bna_246(with_image=True)
        table = AtlasInfo.bna_246()
        self.assertEqual(with_image.labels, table.labels)
        self.assertEqual(with_image.networks, table.networks)
        np.testing.assert_allclose(with_image.coords, table.coords, atol=0.01)


if __name__ == "__main__":
    unittest.main()
