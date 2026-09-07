"""EEG-specific data structures and reshape helpers."""
import numpy as np
from enum import Enum
from typing import List, Tuple, Type, Union, Optional
from itertools import combinations
import pandas as pd


__all__ = [
    "Electrodes",
    "Bands",
    "PairsElectrodes1020",
    "EEGData",
    "read_from_eeg_dataframe",
    "reshape_eeg_data",
    "inverse_reshape_eeg_data",
]


class Electrodes(Enum):
    """
    Class to return EEG electrode names and respective IDs based on 10-20 EEG system.

    >>> Electrodes.Fp2.value
    2
    >>> Electrodes.P3.name
    'P3'
    """
    Fp1 = 1
    Fp2 = 2
    F7 = 3
    F3 = 4
    Fz = 5
    F4 = 6
    F8 = 7
    T3 = 8
    T4 = 9
    T5 = 10
    T6 = 11
    O1 = 12
    O2 = 13
    C3 = 14
    Cz = 15
    C4 = 16
    P3 = 17
    Pz = 18
    P4 = 19

class Bands(Enum):
    """
    Class to return EEG frequency bands with respective IDs.

    >>> Bands.alpha1.value
    3
    >>> Bands.get_name_by_id(6)
    'beta2'
    """

    delta = 1
    theta = 2
    alpha1 = 3
    alpha2 = 4
    beta1 = 5
    beta2 = 6
    gamma = 7

    @staticmethod
    def get_name_by_id(id):
        """Electrode name for a 1-based electrode id."""
        return Bands(id).name

    @staticmethod
    def get_values():
        """List of member names for this enum."""
        return [el.value for el in Bands]
    

class PairsElectrodes1020:
    """
    Class to resolve & return electrode pairs based on 10-20 EEG system.

    Methods: 
        electrode_pairs: Function to resolve indexed electrode names.
        create_pairs_dict: Function to create mapped dictionary of electodes with pairs.

    Attributes: 
        electrodes (List[Electrodes]): List of electrodes to consider 
        nearest (List[Tuple[str, str]]): List of electrode pairs 

    >>> electrodes = [Electrodes.Fp1, Electrodes.Fp2, Electrodes.Fz]
    >>> pairs_obj = PairsElectrodes1020(electrodes)
    >>> pairs_obj.electrode_pairs
    [('Fp1', 'Fp2'), ('Fp1', 'Fz'), ('Fp2', 'Fz')]
    >>> pairs_dict = pairs_obj.create_pairs_dict(pairs_obj.nearest, filter_by=['Fp1'])
    >>> ('Fp1', 'Fp2') in pairs_dict
    True
    """
    def __init__(self, electrodes: Type[Electrodes]):
        self.electrodes = electrodes
        self.nearest = [('Fp1', 'Fp2'),
                        ('Fp1', 'Fz'),
                        ('Fp2', 'Fz'),
                        ('Fp1', 'F3'),
                        ('Fp1', 'F7'),
                        ('Fp2', 'F4'),
                        ('Fp2', 'F8'),
                        ('F7', 'T3'),
                        ('F7', 'C3'),
                        ('F7', 'F3'),
                        ('F3', 'C3'),
                        ('F3', 'Cz'),
                        ('F3', 'Fz'),
                        ('Fz', 'C3'),
                        ('Fz', 'C4'),
                        ('Fz', 'Cz'),
                        ('Fz', 'F4'),
                        ('F4', 'Cz'),
                        ('F4', 'C4'),
                        ('F4', 'T4'),
                        ('F4', 'F8'),
                        ('F8', 'T4'),
                        ('F8', 'C4'),
                        ('T3', 'T5'),
                        ('T3', 'C3'),
                        ('T3', 'P3'),
                        ('C3', 'P3'),
                        ('C3', 'Cz'),
                        ('C3', 'Pz'),
                        ('C3', 'T5'),
                        ]

    @property
    def electrode_pairs(self):
        """
        Returns electrode names as an indexed list as per combinations

        >>> electrodes = [Electrodes.Fp1, Electrodes.Fp2, Electrodes.Fz]
        >>> pairs_obj = PairsElectrodes1020(electrodes)
        >>> pairs_obj.electrode_pairs
        [('Fp1', 'Fp2'), ('Fp1', 'Fz'), ('Fp2', 'Fz')]
        """
        els = list(map(lambda x: x.name, self.electrodes))
        return list(combinations(els, 2))

    def create_pairs_dict(self, pairs_list, filter_by=None):
        """Map electrode pairs to the matching entries of ``pairs_list``.

        Parameters
        ----------
        pairs_list : list of tuple
            Electrode-pair column names to be mapped.
        filter_by : list of str, optional
            Keep only pair names containing at least one of these
            substrings.

        Returns
        -------
        dict
            Keys are electrode-pair tuples (from ``self.electrodes``),
            values are the matching entries of ``pairs_list``.

        Examples
        --------
        >>> electrodes = [Electrodes.Fp1, Electrodes.Fp2, Electrodes.Fz]
        >>> pairs_obj = PairsElectrodes1020(electrodes)
        >>> pairs_obj.electrode_pairs
        [('Fp1', 'Fp2'), ('Fp1', 'Fz'), ('Fp2', 'Fz')]
        """
        pairs_dict = dict()
        p_list = pairs_list.copy()
        els = list(map(lambda x: x.name, self.electrodes))
        if filter_by:
            for opt in filter_by:
                p_list = [pair for pair in p_list if opt in pair]
        for i, el1 in enumerate(els):
            el1_p_list = [pair for pair in p_list if el1 in pair]
            for el2 in els[i + 1:]:
                pairs_dict[(el1, el2)] = [pair for pair in el1_p_list if el2 in pair]
        return pairs_dict


class EEGData:
    """Container for band-resolved EEG connectivity data.

    Attributes
    ----------
    data : np.ndarray of shape (n_subjects, n_chan_pairs, n_freqs)
        Connectivity values per subject, electrode pair, and band.
    subj_list : list of str
        Subject identifiers for the first axis of ``data``.
    electrodes : type[Electrodes]
        Electrode enum used to build the pairs.
    el_pairs_list : iterable of tuple
        Electrode pairs indexing the second axis of ``data``.
    bands : list
        Frequency-band labels for the third axis of ``data``.
    """

    def __init__(self, data, subj_list, electrodes, el_pairs_list, bands):
        self.data = data
        self.subj_list = subj_list
        self.electrodes = electrodes
        self.el_pairs_list = el_pairs_list
        self.bands = bands


def read_from_eeg_dataframe(path_to_df,
                            cond_prefix='fo',
                            band_list=None):
    
    """Read EEG connectivity data from a wide CSV into an :class:`EEGData`.

    The CSV has one row per subject and one column per
    ``<cond_prefix>_<band>_<pair>`` entry; the 10-20 electrode pairs are
    reconstructed from the column names.

    Parameters
    ----------
    path_to_df : str
        Path to the CSV file.
    cond_prefix : str, optional
        Condition prefix identifying the relevant columns (default
        ``'fo'``).
    band_list : list of int, optional
        Frequency bands to keep; defaults to all seven ``Bands``.

    Returns
    -------
    EEGData
        With ``data`` of shape (n_subjects, n_chan_pairs, n_bands).

    Examples
    --------
    >>> eeg_data = read_from_eeg_dataframe('datasets/eeg_dataframe_nansfilled.csv', cond_prefix='fo')
    >>> eeg_data.data.shape
    (177, 171, 7)
    """
    if band_list is None:
        band_list = [1, 2, 3, 4, 5, 6, 7]        
        bands = Bands
    df = pd.read_csv(path_to_df, index_col=0)
    subj_list = list(df.index)
    pairs = PairsElectrodes1020(Electrodes)
    pairs_list = list(df.columns)
    data = []
    for b in band_list:
        pairs_dict = pairs.create_pairs_dict(pairs_list, filter_by=[cond_prefix, f'_{b}_'])
        columns = [col[0] for col in list(pairs_dict.values())]
        data.append(df[columns].values)
    data_arr = np.array(data).swapaxes(0, 1).swapaxes(1, 2)
    return EEGData(data_arr, subj_list, Electrodes, (pairs_dict.keys()), bands)


def reshape_eeg_data(data: np.ndarray,
                     reshape_bands: bool = True
                    ) -> np.ndarray:
    """Unfold pair-indexed EEG data into per-band (n_chans, n_chans) matrices.

    Parameters
    ----------
    data : np.ndarray of shape (n_subjects, n_chan_pairs, n_bands)
        Pair-indexed data (a single subject may be passed as 2D).
    reshape_bands : bool, optional
        If True, return block-diagonal matrices of shape
        (n_subjects, n_chans * n_bands, n_chans * n_bands) where each
        diagonal block is one band. If False, return
        (n_subjects, n_chans, n_chans, n_bands). Default True.

    Returns
    -------
    np.ndarray
        Shape (n_chans, n_chans, n_bands) or
        (n_chans * n_bands, n_chans * n_bands) for single-subject input;
        batched shapes otherwise.

    Notes
    -----
    Assumes the 19-electrode 10-20 system (``Electrodes``).

    Examples
    --------
    >>> n_pairs = np.random.rand(2, len(PairsElectrodes1020(Electrodes).electrode_pairs), 3)
    >>> reshape_eeg_data(n_pairs, reshape_bands=False).shape
    (2, 19, 19, 3)
    >>> reshape_eeg_data(n_pairs, reshape_bands=True).shape
    (2, 57, 57)
    """
    num_els = len(Electrodes)  # 19 electrodes
    el_pairs_list = PairsElectrodes1020(Electrodes).electrode_pairs

    single_subject = data.ndim == 2
    if single_subject:
        data = data[np.newaxis, ...]
        
    n_subjects, _, n_frequencies = data.shape
    reshaped_data = np.zeros((n_subjects, num_els, num_els, n_frequencies))

    # Fill in the 19x19 matrices for each frequency
    for pair_idx, (el1, el2) in enumerate(el_pairs_list):
        i, j = Electrodes[el1].value - 1, Electrodes[el2].value - 1
        reshaped_data[:, i, j, :] = data[:, pair_idx, :]
        reshaped_data[:, j, i, :] = data[:, pair_idx, :]  

    if reshape_bands:
        # block-diagonal form: one n_chans × n_chans block per band
        to_reshape = reshaped_data.copy()
        reshaped_data = np.zeros((n_subjects, num_els * n_frequencies, num_els * n_frequencies))
        for k in range(n_frequencies):
            reshaped_data[:, k * num_els:(k + 1) * num_els, k * num_els:(k + 1) * num_els] = to_reshape[..., k]

    return reshaped_data[0] if single_subject else reshaped_data


def inverse_reshape_eeg_data(reshaped_data: np.ndarray,
                             reshape_bands: bool = True) -> np.ndarray:
    """Inverse of :func:`reshape_eeg_data`.

    Parameters
    ----------
    reshaped_data : np.ndarray
        Shape (n_subjects, n_chans, n_chans, n_bands), or
        (n_subjects, n_chans * n_bands, n_chans * n_bands) when
        ``reshape_bands=True`` (a single subject may be passed as 2D).
    reshape_bands : bool, optional
        Whether the input is in band-flattened (block-diagonal) form.

    Returns
    -------
    np.ndarray
        Pair-indexed data of shape (n_subjects, n_chan_pairs, n_bands),
        or (n_chan_pairs, n_bands) for single-subject input.

    Examples
    --------
    >>> n_pairs = np.random.rand(2, len(PairsElectrodes1020(Electrodes).electrode_pairs), 3)
    >>> reshaped = reshape_eeg_data(n_pairs, reshape_bands=True)
    >>> reshaped.shape
    (2, 57, 57)
    >>> inverse_reshape_eeg_data(reshaped, reshape_bands=True).shape
    (2, 171, 3)
    """
    num_els = len(Electrodes)  # 19 electrodes
    el_pairs_list = PairsElectrodes1020(Electrodes).electrode_pairs

    single_subject = reshaped_data.ndim == 2
    if single_subject:
        reshaped_data = reshaped_data[np.newaxis, ...]

    n_subjects = reshaped_data.shape[0]

    if reshape_bands:
        n_frequencies = reshaped_data.shape[1] // num_els
        extracted = np.zeros((n_subjects, num_els, num_els, n_frequencies))
        for k in range(n_frequencies):
            extracted[..., k] = reshaped_data[:, k * num_els:(k + 1) * num_els,
                                               k * num_els:(k + 1) * num_els]
        reshaped_data = extracted

    n_frequencies = reshaped_data.shape[-1]
    original_data = np.zeros((n_subjects, len(el_pairs_list), n_frequencies))
    for pair_idx, (el1, el2) in enumerate(el_pairs_list):
        i, j = Electrodes[el1].value - 1, Electrodes[el2].value - 1
        original_data[:, pair_idx, :] = reshaped_data[:, i, j, :]

    return original_data[0] if single_subject else original_data
