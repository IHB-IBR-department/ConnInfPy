"""Regression oracles for paired differences, interactions and CLI alignment."""
import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy import analyze, compute_p_val_glm, ResultBundle
from conninfpy.cli import build_analysis, ConfigError, run
from conninfpy.loaders.base import LoadedDataset
from conninfpy._shared.alignment import paired_id_permutation
import conninfpy.stats.glm as glm

FAST = dict(method='tstat', n_permutations=39, acceleration=None, use_mp=False, rng=11)


def matrices(n, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .3, (n, 4, 4))
    x = (x + x.transpose(0, 2, 1)) / 2
    x[:, np.arange(4), np.arange(4)] = 0
    return x


def compare(a, b):
    for field in ('positive', 'negative', 'stat_positive', 'stat_negative', 'beta'):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), equal_nan=True, atol=1e-12)


@pytest.mark.parametrize('scale', ['z', 'r'])
def test_paired_interest_matches_manual_difference(scale):
    rng = np.random.default_rng(9)
    a, b = matrices(24), matrices(24, 2)
    x, c, c1, c2 = (rng.normal(size=24) for _ in range(4))
    given_a, given_b = (np.tanh(a), np.tanh(b)) if scale == 'r' else (a, b)
    result = analyze(group1=given_a, group2=given_b, test_type='paired', interest=x,
                     confounds=c, confounds_group1=c1, confounds_group2=c2,
                     input=scale, **FAST).inference
    manual = analyze(b-a, interest=x, confounds=np.column_stack([c, c2-c1]), input='z', **FAST).inference
    compare(result, manual)
    assert result.design['scale'] == 'difference_fisher_z'
    assert 'B minus A' in result.design['response_transform']
    assert result.design['paired_difference'] == {'input': scale, 'orientation': 'B - A', 'subject_confound_columns': 1, 'condition_confound_columns': 1, 'condition_confound_transform': 'B - A'}
    assert result.analysis_provenance['units'] == 'Fisher z difference (B - A)'
    bundle = result.to_bundle()
    assert bundle.metadata['design']['scale'] == 'difference_fisher_z'
    assert bundle.effect_spec['definition'] == 'cᵀβ'


def test_group_change_coefficient_is_manual_group_delta():
    a, b = matrices(24), matrices(24, 2)
    group = np.tile([0., 1.], 12)
    result = analyze(group1=a, group2=b, test_type='paired', interest=group, input='z', **FAST).inference
    delta = b - a
    np.testing.assert_allclose(result.effect_map(), delta[group == 1].mean(axis=0) - delta[group == 0].mean(axis=0), atol=1e-12)


def test_interaction_matches_manual_design_and_null(monkeypatch):
    rng = np.random.default_rng(4)
    y = matrices(26)
    a, b, c = (rng.normal(size=26) for _ in range(3))
    ac, bc = a-a.mean(), b-b.mean()
    X = np.column_stack([np.ones(26), c, ac, bc, ac*bc])
    captured = []
    original = glm._collect_results_to_arrays
    def collect(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append({k: v.copy() for k, v in result.items()})
        return result
    monkeypatch.setattr(glm, '_collect_results_to_arrays', collect)
    results = analyze(y, interest={'a': a, 'b': b}, confounds=c,
                      interactions=[('a', 'b')], input='z', **FAST)
    multi_null = captured[-1]
    # analyze() defaults to the two-sided family; the manual low-level
    # oracle must declare the same family explicitly.
    manual = compute_p_val_glm(y, design_matrix=X,
                               contrast=np.array([0, 0, 0, 0, 1]),
                               two_tailed=True, **FAST)
    compare(results['a:b'].inference, manual)
    for tail in ('positive', 'negative'):
        np.testing.assert_allclose(multi_null[tail], captured[-1][tail], atol=1e-12)
    assert results['a:b'].inference.design['columns'] == ['intercept', 'confound_1', 'a', 'b', 'a:b']


def dataset():
    n = 16
    rng = np.random.default_rng(3)
    ph = pd.DataFrame({'subject_id': list(range(n))*2, 'condition': ['A']*n+['B']*n,
                       'group': list(np.tile(['c', 't'], n//2))*2,
                       'age': list(rng.normal(size=n))*2,
                       'motion': rng.normal(size=2*n)})
    order = rng.permutation(2*n)
    return LoadedDataset(data=matrices(2*n)[order], pheno=ph.iloc[order].reset_index(drop=True), data_kind='fisher_z')


def test_cli_alignment_and_validation():
    ds = dataset()
    d = {'type':'paired', 'condition':'condition', 'levels':['A','B'], 'interest':['group','age'],
         'interactions':[['group','age']], 'confounds':['motion']}
    kw, _ = build_analysis(ds, {'design':d})
    a = ds.pheno[ds.pheno.condition == 'A']
    b_rows = ds.pheno[ds.pheno.condition == 'B']
    b = b_rows.set_index('subject_id').loc[a.subject_id]   # B rows in A order
    # Condition-B inputs reach analyze() in the caller's row order; analyze()
    # applies the single B -> A permutation itself, from the IDs.
    np.testing.assert_allclose(kw['confounds_group2'].ravel(), b_rows.motion)
    out = analyze(**kw, **FAST)
    perm = out['group:age'].inference.design['paired_alignment']['permutation_b_to_a']
    np.testing.assert_allclose(kw['confounds_group2'].ravel()[perm], b.motion)
    assert 'group:age' in out
    missing = dataset(); missing.pheno = missing.pheno.iloc[:-1]; missing.data = missing.data[:-1]
    with pytest.raises(ConfigError, match='missing pair'): build_analysis(missing, {'design': d})
    dup = dataset(); dup.pheno.loc[0, ['subject_id', 'condition']] = dup.pheno.loc[1, ['subject_id', 'condition']].values
    with pytest.raises(ConfigError, match='repeats'): build_analysis(dup, {'design': d})
    bad = dataset(); bad.pheno.loc[0, 'group'] = 'changed'
    with pytest.raises(ConfigError, match='constant'): build_analysis(bad, {'design': d})
    with pytest.raises(ConfigError, match='levels'): build_analysis(ds, {'design': {**d, 'levels':['A','C']}})
    with pytest.raises(ConfigError, match='both condition-specific'):
        build_analysis(ds, {'design': {**d, 'subject_confounds':['motion']}})
    categorical = dataset()
    categorical.pheno['motion_cat'] = categorical.pheno['group']
    kc, _ = build_analysis(categorical, {'design': {**d, 'confounds':['motion_cat']}})
    np.testing.assert_array_equal(
        kc['confounds_group1'],
        kc['confounds_group2'][paired_id_permutation(kc['ids_group1'], kc['ids_group2'])])
    categorical.pheno.loc[categorical.pheno.condition == 'B', 'motion_cat'] = 'other'
    with pytest.raises(ConfigError, match='incompatible levels'):
        build_analysis(categorical, {'design': {**d, 'confounds':['motion_cat']}})


def test_paired_shape_and_confounds_validation():
    a = matrices(16)
    with pytest.raises(ValueError, match='identical'):
        analyze(group1=a, group2=a[:-1], test_type='paired', interest=np.arange(16), input='z', **FAST)
    with pytest.raises(ValueError, match='both condition'):
        analyze(group1=a, group2=a, test_type='paired', interest=np.arange(16), confounds_group1=np.ones(16), input='z', **FAST)
    with pytest.raises(ValueError, match='requires Fisher'):
        analyze(group1=np.tanh(a), group2=np.tanh(a), test_type='paired', interest=np.arange(16), input='r', r_to_z=False, **FAST)


def test_yaml_paired_interaction_bundle_independent(tmp_path):
    ds = dataset()
    np.save(tmp_path / 'data.npy', ds.data)
    ds.pheno.to_csv(tmp_path / 'pheno.csv', index=False)
    (tmp_path / 'study.yaml').write_text(yaml.safe_dump({'schema_version': 1, 'loader': 'NumpyLoader', 'paths': {'data_path': 'data.npy', 'pheno_path': 'pheno.csv'}}))
    config = {'data': 'study.yaml', 'input': 'z', 'design': {'type': 'paired', 'condition': 'condition', 'levels': ['A', 'B'], 'interest': ['group', 'age'], 'interactions': [['group', 'age']], 'confounds': ['motion']}, 'inference': FAST, 'output': 'out'}
    path = tmp_path / 'analysis.yaml'
    path.write_text(yaml.safe_dump(config))
    outputs = run(path)
    bundle = ResultBundle.load(outputs['bundle_group:age'])
    (tmp_path / 'data.npy').unlink()
    assert bundle.effect_spec['scale'] == 'difference_fisher_z'
    assert bundle.metadata['contrast_id'] == 'group:age'
    assert 'group' in bundle.metadata['design']['predictor_centering_means']
    assert bundle.metadata['design']['categorical_coding']['group'] == {'levels': ['c', 't'], 'reference': 'c'}
    assert bundle.metadata['design']['input_design']['confounds'] == ['motion']
    assert bundle.metadata['family']['n_eligible'] == 6
    np.testing.assert_allclose(bundle.effect, np.load(outputs['pvals_group:age'])['effect'])
