"""NIfTI signal extraction through nilearn (NiftiDirectoryLoader, FmriprepDerivativesLoader).

These paths had no coverage, so a nilearn release that rejected the
loaders' ``standardize='zscore'`` (nilearn 0.14 accepts only
'zscore_sample') broke extraction on every fresh install unnoticed. CI
installs the latest nilearn, so these tests catch the next such change.
Skipped when nilearn (the ``[app]`` extra) is not installed.
"""
import glob
import inspect
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    import nilearn  # noqa: F401
    HAS_NILEARN = True
except ImportError:  # core-only install
    HAS_NILEARN = False


def _labels_and_subjects(tmp: Path, n_subjects=3, T=60, seed=0):
    """Four labelled blocks; each subject's block voxels carry a known signal."""
    rng = np.random.default_rng(seed)
    shape, affine = (12, 12, 12), np.diag([2.0, 2.0, 2.0, 1.0])
    labels = np.zeros(shape, dtype=np.int16)
    labels[1:5, 1:5, 1:5], labels[7:11, 1:5, 1:5] = 1, 2
    labels[1:5, 7:11, 7:11], labels[7:11, 7:11, 7:11] = 3, 4
    nib.save(nib.Nifti1Image(labels, affine), tmp / "atlas.nii.gz")
    (tmp / "func").mkdir()
    planted = []
    for s in range(n_subjects):
        sig = rng.standard_normal((T, 4))
        img = rng.standard_normal(shape + (T,)) * 0.3 + 100
        for k in range(4):
            img[labels == k + 1] += sig[:, k]
        nib.save(nib.Nifti1Image(img.astype(np.float32), affine), tmp / "func" / f"sub-0{s}_bold.nii.gz")
        planted.append(sig)
    return np.array(planted)


@unittest.skipUnless(HAS_NILEARN, "nilearn not installed ([app] extra)")
class TestNiftiDirectoryLoader(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.planted = _labels_and_subjects(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, **kwargs):
        from conninfpy.loaders import NiftiDirectoryLoader
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return NiftiDirectoryLoader(dir_path=str(self.tmp / "func"),
                                        atlas_image_path=str(self.tmp / "atlas.nii.gz"),
                                        **kwargs).load()

    def test_extracts_one_series_per_region(self):
        ds = self._load()
        self.assertEqual(ds.data_kind, "timeseries")
        self.assertEqual(np.asarray(ds.data).shape, (3, 60, 4))

    def test_extracted_series_follow_the_planted_signals(self):
        ts = np.asarray(self._load().data)
        for s in range(3):
            for k in range(4):
                r = np.corrcoef(ts[s, :, k], self.planted[s, :, k])[0, 1]
                self.assertGreater(r, 0.95)

    def test_legacy_zscore_is_accepted_and_keeps_correlations(self):
        """'zscore' (rejected by nilearn 0.14) maps to 'zscore_sample'."""
        a = np.asarray(self._load(standardize="zscore").data)
        b = np.asarray(self._load().data)
        ca = np.array([np.corrcoef(x.T) for x in a])
        cb = np.array([np.corrcoef(x.T) for x in b])
        np.testing.assert_allclose(ca, cb, atol=1e-10)


@unittest.skipUnless(HAS_NILEARN, "nilearn not installed ([app] extra)")
class TestFmriprepDerivativesLoader(unittest.TestCase):

    def test_all_denoising_strategies(self):
        try:
            from nilearn._utils.data_gen import create_fake_bids_dataset
        except ImportError:  # private nilearn helper; skip if it moves
            self.skipTest("nilearn fake-BIDS generator not available")
        from conninfpy.loaders import FmriprepDerivativesLoader

        with tempfile.TemporaryDirectory() as tmp:
            kw = dict(base_dir=tmp, n_sub=2, n_ses=0, tasks=["rest"], n_runs=[1],
                      with_derivatives=True, with_confounds=True)
            seed_arg = ("rand_gen" if "rand_gen" in
                        inspect.signature(create_fake_bids_dataset).parameters else "random_state")
            kw[seed_arg] = 0
            deriv = Path(create_fake_bids_dataset(**kw)) / "derivatives"
            bold = sorted(glob.glob(str(deriv / "sub-*/func/*space-MNI*desc-preproc_bold.nii.gz")))[0]
            img = nib.load(bold)
            labels = np.zeros(img.shape[:3], dtype=np.int16)
            labels[:2], labels[2:] = 1, 2
            atlas = Path(tmp) / "atlas.nii.gz"
            nib.save(nib.Nifti1Image(labels, img.affine), atlas)
            for strategy in range(1, 7):
                for gsr in (False, True):
                    with self.subTest(strategy=strategy, gsr=gsr), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ds = FmriprepDerivativesLoader(
                            derivatives_dir=str(deriv), atlas_image_path=str(atlas),
                            bold_pattern="sub-*/func/*space-MNI*desc-preproc_bold.nii.gz",
                            denoising={"strategy": strategy, "use_GSR": gsr},
                        ).load()
                        self.assertEqual(np.asarray(ds.data).shape[0], 2)
                        self.assertEqual(np.asarray(ds.data).shape[-1], 2)


if __name__ == "__main__":
    unittest.main()
