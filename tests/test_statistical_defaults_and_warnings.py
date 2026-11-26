"""Regression checks for statistical defaults and interpretation warnings."""
import inspect
import warnings

import numpy as np
import pytest

from conninfpy import defaults
from conninfpy.analysis import analyze
from conninfpy.stats.glm import compute_p_val_glm, compute_p_val_glm_multi
from conninfpy.stats.pairwise import compute_p_val, _compute_parametric_p_values


def data():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(8, 3, 3))
    x = (x + x.transpose(0, 2, 1)) / 2
    x[:, np.arange(3), np.arange(3)] = 0
    return x


def test_defaults_and_citations():
    for fn in (compute_p_val, compute_p_val_glm, compute_p_val_glm_multi,
               _compute_parametric_p_values):
        assert inspect.signature(fn).parameters['two_tailed'].default is True
    assert inspect.signature(analyze).parameters['two_tailed'].default is None
    assert defaults.DEFAULT_NBS_THRESHOLD == 3.0
    citation = ' '.join(defaults.__doc__.split())
    assert 'Vinokur, L., Smith, R. E., Dhollander, T., Vaughan, D., Jackson, G. D., & Connelly, A. (2023).' in citation
    assert 'Hao, Z., Wang, P., Xia, X., Pan, Y., & Dou, W. (2024).' in citation
    assert 'Parameter Sensitivity of Network-Based Statistical Inference. Research Square (preprint), doi:10.21203/rs.3.rs-3081615/v1.' in citation
    assert 'Threshold-Free Network-Oriented Statistics in Neuroscience (TFNOS). bioRxiv (preprint), doi:10.1101/2024.01.03.574016.' in citation
    assert 'H=3.0`` yielded empirical FDR < 10%' in citation
    assert 'H=7.0`` empirical FDR < 5%' in citation


def test_nbs_warning_pairwise_and_glm():
    x = data()
    with pytest.warns(UserWarning, match='component level'):
        compute_p_val(x, test_type='one-sample', method='nbs', threshold=2.0,
                      n_permutations=2, use_mp=False, rng=12)
    design = np.ones((len(x), 1))
    with pytest.warns(UserWarning, match='component level'):
        compute_p_val_glm(x, design_matrix=design, contrast=np.array([1.]),
                          method='nbs', threshold=2.0, n_permutations=2,
                          use_mp=False, rng=12)
    with pytest.warns(UserWarning, match='component level'):
        compute_p_val_glm_multi(x, design, {'a': np.array([1.])},
                                method='nbs', threshold=2.0, n_permutations=2,
                                use_mp=False, rng=12)


def test_bh_permutation_resolution_warning_and_boundary():
    x = data()  # three edges; B=59 makes the minimum rank exactly one
    with pytest.warns(UserWarning, match=r'm=3 edges, B=9.*k_min=6.*q=0.05'):
        compute_p_val(x, test_type='one-sample', method='bh_fdr_perm',
                      n_permutations=9, use_mp=False, rng=12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_p_val(x, test_type='one-sample', method='bh_fdr_perm',
                      n_permutations=59, use_mp=False, rng=12)
    assert not any('bh_fdr_perm resolution' in str(w.message) for w in caught)


def test_bh_resolution_uses_only_tested_edges_with_missing_data():
    x = data()
    x[:4, 0, 1] = np.nan
    x[:4, 1, 0] = np.nan
    with pytest.warns(UserWarning, match=r'm=2 edges, B=9.*k_min=4'):
        compute_p_val(x, test_type='one-sample', method='bh_fdr_perm',
                      missing='available_case', min_coverage=0.8,
                      n_permutations=9, use_mp=False, rng=12)


def test_analyze_defaults_and_nbs_warning():
    x = data()
    interest = np.arange(len(x), dtype=float)
    result = analyze(x, interest=interest, input='z', method='tstat',
                     n_permutations=2, use_mp=False, rng=12)
    assert result.two_tailed is True
    with pytest.warns(UserWarning, match='component level'):
        analyze(x, interest=interest, input='z', method='nbs', threshold=2.0,
                n_permutations=2, use_mp=False, rng=12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        compute_p_val(x, test_type='one-sample', method='nbs',
                      n_permutations=2, use_mp=False, rng=12)
    assert not any('NBS threshold' in str(w.message) for w in caught)
