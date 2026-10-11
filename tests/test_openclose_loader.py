"""Unit tests for examples.open_close._shared.data."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.open_close._shared.data import (
    BNA_DROPOUT_IDS,
    OpenCloseDataset,
    MISSED_ROI_IDS,
    bad_retest_subjects,
)


class TestOpenCloseLoader(unittest.TestCase):
    """Structural checks on the Schaefer-200 / 182 loader."""

    def test_missed_rois_are_18_distinct(self) -> None:
        self.assertEqual(len(MISSED_ROI_IDS), 18)
        self.assertEqual(len(set(MISSED_ROI_IDS)), 18)
        self.assertTrue(all(1 <= x <= 200 for x in MISSED_ROI_IDS))

    def test_bad_retest_subject_is_documented(self) -> None:
        bad = bad_retest_subjects()
        self.assertIn("sub-3258811", bad)

    def test_ihb_shapes(self) -> None:
        ds = OpenCloseDataset.load("ihb")
        self.assertEqual(ds.open_ts.shape, (84, 120, 182))
        self.assertEqual(ds.close_ts.shape, (84, 120, 182))
        self.assertIsNone(ds.close_ts_run1)
        self.assertEqual(ds.n_rois, 182)
        self.assertEqual(ds.n_subjects, 84)
        self.assertEqual(ds.n_edges, 182 * 181 // 2)

    def test_china_shapes(self) -> None:
        ds = OpenCloseDataset.load("china")
        self.assertEqual(ds.open_ts.shape, (48, 240, 182))
        self.assertEqual(ds.close_ts.shape, (48, 240, 182))
        self.assertEqual(ds.close_ts_run1.shape, (48, 240, 182))

    def test_no_drop_yields_200(self) -> None:
        ds = OpenCloseDataset.load("ihb", drop_missing_rois=False)
        self.assertEqual(ds.open_ts.shape[-1], 200)

    def test_connectivity_z_structure(self) -> None:
        ds = OpenCloseDataset.load("ihb")
        o_z, c_z = ds.connectivity_z(run=0)
        self.assertEqual(o_z.shape, (84, 182, 182))
        self.assertEqual(c_z.shape, (84, 182, 182))
        # Symmetric and zero-diagonal
        self.assertTrue(np.allclose(o_z[0], o_z[0].T))
        self.assertTrue(np.allclose(np.diag(o_z[0]), 0.0))
        # Fisher z is finite — edges clipped to avoid atanh infinities
        self.assertTrue(np.all(np.isfinite(o_z)))

    def test_china_run1_access(self) -> None:
        ds = OpenCloseDataset.load("china")
        o_z, c1_z = ds.connectivity_z(run=1)
        self.assertEqual(c1_z.shape, (48, 182, 182))

    def test_ihb_run1_raises(self) -> None:
        ds = OpenCloseDataset.load("ihb")
        with self.assertRaises(ValueError):
            ds.connectivity_z(run=1)

    def test_net_labels_are_contiguous_yeo7(self) -> None:
        ds = OpenCloseDataset.load("ihb")
        uniq = sorted(set(ds.net_labels))
        self.assertEqual(uniq, list(range(7)))
        self.assertEqual(len(ds.net_labels), 182)

    def test_label_names_are_the_expected_seven(self) -> None:
        ds = OpenCloseDataset.load("ihb")
        names = set(ds.get_label_names())
        self.assertEqual(
            names,
            {
                "Default Mode", "Dorsal Attention", "Frontoparietal Control",
                "Limbic", "Somatomotor", "Ventral Attention", "Visual",
            },
        )

    def test_retest_mask_excludes_bad_subject(self) -> None:
        ds = OpenCloseDataset.load("china")
        mask = ds.retest_subject_mask()
        self.assertEqual(len(mask), 48)
        # Exactly one subject excluded: sub-3258811
        self.assertEqual(int((~mask).sum()), 1)
        bad_idx = np.where(~mask)[0][0]
        self.assertEqual(ds.subject_ids[bad_idx], "sub-3258811")

    def test_ihb_retest_mask_is_all_true(self) -> None:
        ds = OpenCloseDataset.load("ihb")
        self.assertTrue(np.all(ds.retest_subject_mask()))


_BNA_FILES = [Path(__file__).resolve().parents[1] / "datasets" / "open_close" / f"{e}_{c}_Brainnetome_strategy-4_{g}.npy"
              for e in ("ihb", "china") for c in ("open", "close") for g in ("GSR", "noGSR")]


@unittest.skipUnless(all(f.exists() for f in _BNA_FILES), "Brainnetome Open-Close series not present")
class TestBrainnetomeSeries(unittest.TestCase):
    """BNA-246 products: same participants, 4 dropout regions removed by default."""

    def test_shapes_labels_and_dropouts(self) -> None:
        for exp, n_sub, T in (("ihb", 84, 120), ("china", 48, 240)):
            for gsr in (True, False):
                ds = OpenCloseDataset.load(exp, atlas="brainnetome", gsr=gsr)
                self.assertEqual(ds.open_ts.shape, (n_sub, T, 242))
                self.assertEqual(len(ds.atlas_df), 242)
                self.assertFalse(set(BNA_DROPOUT_IDS) & set(ds.atlas_df["id"]))
                self.assertEqual((ds.atlas_df["network"] == "Subcortical").sum(), 36)
                self.assertEqual((ds.atlas, ds.gsr), ("brainnetome", gsr))
        full = OpenCloseDataset.load("china", atlas="brainnetome", drop_missing_rois=False)
        self.assertEqual(full.close_ts_run1.shape, (48, 240, 246))

    def test_same_subject_order_as_schaefer(self) -> None:
        s = OpenCloseDataset.load("ihb")
        b = OpenCloseDataset.load("ihb", atlas="brainnetome")
        self.assertEqual(s.subject_ids, b.subject_ids)

    def test_schaefer_without_gsr_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "brainnetome"):
            OpenCloseDataset.load("ihb", gsr=False)


if __name__ == "__main__":
    unittest.main()
