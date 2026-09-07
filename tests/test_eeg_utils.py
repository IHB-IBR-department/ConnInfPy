from unittest import TestCase, skipUnless
import numpy as np
import conninfpy.eeg_utils as eeg_utils
from conninfpy.eeg_utils import Electrodes, PairsElectrodes1020, Bands
from pathlib import Path
import pickle

# Get path to datasets directory relative to this file
DATASETS_DIR = Path(__file__).parent.parent / 'datasets'

# The EEG dataframe is local-only data (gitignored); skip the tests that
# need it on machines without the file (e.g. fresh clones / CI runners).
EEG_DF = DATASETS_DIR / 'eeg_dataframe_nansfilled.csv'
HAS_EEG_DF = EEG_DF.exists()

@skipUnless(HAS_EEG_DF, f"local-only dataset missing: {EEG_DF}")
class TesteegUtils(TestCase):

    def setUp(self) -> None:
        data = np.random.randn(20, 171, 7)
        subj_list = [f'sub_{i}' for i in range(20)]
        pairs = PairsElectrodes1020(Electrodes)
        self.path_to_df = DATASETS_DIR / 'eeg_dataframe_nansfilled.csv'

    def test_read_from_eeg_dataframe(self):
        stable_fo = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fo')
        stable_fz = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fz')
        self.assertTrue(stable_fo.data.shape == (177, 171, 7))
        self.assertTrue(stable_fz.data.shape == (177, 171, 7))
        self.assertEqual(len(stable_fo.subj_list), len(stable_fz.subj_list))

    def test_reshape_eeg_data(self):
        stable_fo = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fo')
        stable_fz = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fz')
        reshaped_data = eeg_utils.reshape_eeg_data(stable_fo.data - stable_fz.data, reshape_bands=False)
        self.assertTrue(reshaped_data.shape, (177, 19, 19, 7))
        reshaped_data = eeg_utils.reshape_eeg_data(stable_fo.data - stable_fz.data, reshape_bands=True)
        self.assertTrue(reshaped_data.shape, (177, 19 * 7, 19 * 7))

    
    def test_inverse_reshape_eeg_data(self):
        stable_fo = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fo')
        stable_fz = eeg_utils.read_from_eeg_dataframe(self.path_to_df, cond_prefix='fz')

        orginal_data = stable_fo.data - stable_fz.data
        reshaped_data = eeg_utils.reshape_eeg_data(orginal_data, reshape_bands=True)
        inversed_data = eeg_utils.inverse_reshape_eeg_data(reshaped_data, reshape_bands=True)
        np.testing.assert_almost_equal(orginal_data, inversed_data)

