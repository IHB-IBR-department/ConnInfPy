import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.covariance import LedoitWolf, OAS

from conninfpy import compute_connectivity
from conninfpy.connectivity import connectivity_from_dataset
from conninfpy.loaders import LoadedDataset, load, validate_loaded_dataset


def test_pearson_shapes_nan_frames_and_constants():
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(35, 7)), rng.normal(size=(43, 7))
    a[3, 1] = np.nan
    b[5, 4] = np.inf
    out, d = compute_connectivity([a, b], return_diagnostics=True)
    for matrix, x in zip(out, (a, b)):
        np.testing.assert_allclose(matrix, np.corrcoef(x[np.isfinite(x).all(axis=1)].T), atol=1e-12, rtol=0)
    assert d['n_valid_tr'].tolist() == [34, 42]
    assert d['delta_s'].tolist() == [0, 0]
    np.testing.assert_allclose(compute_connectivity(a), out[0], atol=1e-12, rtol=0)
    c = rng.normal(size=(20, 4)); c[:, 1] = 4
    with pytest.warns(UserWarning, match='constant ROIs'):
        r = compute_connectivity(c)
    assert np.isnan(r[1]).all() and np.isfinite(r[0, 2])
    with pytest.raises(ValueError, match='constant ROIs'):
        compute_connectivity(c, kind='partial_correlation')
    with pytest.raises(ValueError, match='complete finite frames'):
        compute_connectivity(np.full((4, 3), np.nan))


@pytest.mark.parametrize('name, estimator', [('ledoit_wolf', LedoitWolf), ('oas', OAS)])
def test_shrinkage_matches_sklearn_and_partial_matches_nilearn(name, estimator):
    nilearn = pytest.importorskip('nilearn.connectome')
    rng = np.random.default_rng(8)
    xs = rng.normal(size=(3, 50, 10)) * np.arange(1, 11)
    mats, diag = compute_connectivity(xs, shrinkage=name, return_diagnostics=True)
    standardized = (xs - xs.mean(axis=1, keepdims=True)) / xs.std(axis=1, keepdims=True)
    for i, x in enumerate(standardized):
        est = estimator(assume_centered=True).fit(x)
        cov = est.covariance_
        expected = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        np.testing.assert_allclose(mats[i], expected, atol=1e-10, rtol=0)
        assert abs(diag['delta_s'][i] - est.shrinkage_) < 1e-10
    expected_partial = nilearn.ConnectivityMeasure(kind='partial correlation', cov_estimator=estimator(assume_centered=True), standardize=False).fit_transform(list(standardized))
    partial = compute_connectivity(xs, kind='partial_correlation', shrinkage=name)
    np.testing.assert_allclose(partial, expected_partial, atol=1e-8, rtol=0)


def test_fixed_shrinkage_default_partial_and_singular():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(150, 400))
    r, d = compute_connectivity(x, kind='partial_correlation', return_diagnostics=True)
    assert np.isfinite(r).all() and 0 <= d['delta_s'][0] <= 1
    with pytest.raises(ValueError, match='singular'):
        compute_connectivity(x, kind='partial_correlation', shrinkage=None)
    with pytest.raises(ValueError, match='singular'):
        compute_connectivity(x, kind='partial_correlation', shrinkage=0.)
    y = rng.normal(size=(2, 50, 9))
    matrices, d = compute_connectivity(y, shrinkage=.2, return_diagnostics=True)
    z = (y[0] - y[0].mean(0)) / y[0].std(0)
    cov = z.T @ z / len(z); cov = .8 * cov + .2 * np.trace(cov) / 9 * np.eye(9)
    expected = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    np.testing.assert_allclose(matrices[0], expected, atol=1e-12, rtol=0)
    assert d['delta_s'].tolist() == [.2, .2]


def test_group_diagnostics_and_short_frames():
    rng = np.random.default_rng(12)
    ds = LoadedDataset([rng.normal(size=(8, 3)), rng.normal(size=(13, 3))], pd.DataFrame({'arm': ['a', 'b']}), 'timeseries')
    with pytest.warns(UserWarning) as records:
        connectivity_from_dataset(ds, shrinkage='oas', group_column='arm')
    assert ds.pheno.n_valid_tr.tolist() == [8, 13]
    assert ds.pheno.delta_s.between(0, 1).all()
    assert any('valid frames' in str(w.message) for w in records)
    assert any('empirical distributions of n_valid_tr' in str(w.message) for w in records)
    assert ds.provenance['connectivity']['shrinkage'] == 'oas'


def test_manifest_variable_length_cli_and_rois(tmp_path):
    from conninfpy.cli import build_analysis
    rng = np.random.default_rng(4)
    for i, n in enumerate((20, 25, 31, 35)):
        x = rng.normal(size=(n, 5)); x[0, 1] = np.nan
        np.save(tmp_path / f'sub-{i}.npy', x)
    manifest = {'loader': 'TimeseriesDirectoryLoader', 'paths': {'dir_path': str(tmp_path)},
                'params': {'pattern': '*.npy', 'connectivity_kind': 'partial_correlation', 'shrinkage': 'oas', 'keep_rois': [1, 2, 4]},
                'checks': {'expected_rois': 3, 'min_timepoints': 19, 'min_subjects': 4}}
    f = tmp_path / 'data.yaml'; f.write_text(yaml.safe_dump(manifest))
    ds = load(f)
    assert ds.data.shape == (4, 3, 3)
    assert sorted(ds.pheno.n_valid_tr) == [19, 24, 30, 34]
    assert ds.provenance['connectivity']['kind'] == 'partial_correlation'
    kwargs, summary = build_analysis(ds, {'design': {'type': 'one-sample'}})
    assert kwargs['group1'].shape == (4, 3, 3)
    assert len(summary['connectivity_diagnostics']['delta_s']) == 4
    manifest['params'].pop('connectivity_kind'); manifest['params'].pop('shrinkage')
    f.write_text(yaml.safe_dump(manifest))
    raw = load(f)
    assert isinstance(raw.data, list) and validate_loaded_dataset(raw).ok
    kwargs, summary = build_analysis(raw, {'design': {'type': 'one-sample'}, 'connectivity': {'kind': 'partial_correlation', 'shrinkage': 0.25}})
    assert raw.pheno.n_valid_tr.tolist() == [19, 24, 30, 34]
    assert np.allclose(raw.pheno.delta_s, .25)
    assert kwargs['group1'].shape == (4, 3, 3)


def test_open_close_loader_uses_shared_estimator(tmp_path):
    from apps.utils.lab_loaders import OpenCloseLoader
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 30, 4)); b = rng.normal(size=(3, 35, 4)); b[1, 1, 0] = np.nan
    np.save(tmp_path / 'a.npy', a); np.save(tmp_path / 'b.npy', b)
    ds = OpenCloseLoader(str(tmp_path / 'a.npy'), str(tmp_path / 'b.npy'), connectivity_kind='partial_correlation', shrinkage=.3).load()
    assert ds.data.shape == (6, 4, 4)
    assert ds.pheno.n_valid_tr.tolist() == [30, 30, 30, 35, 34, 35]
    assert np.allclose(ds.pheno.delta_s, .3)
    np.testing.assert_allclose(ds.data[0], compute_connectivity(a[0], kind='partial_correlation', shrinkage=.3), atol=1e-12)


def test_fmriprep_loader_preserves_unequal_scrubbed_lengths(tmp_path, monkeypatch):
    from conninfpy.loaders import FmriprepDerivativesLoader
    from conninfpy.connectivity import connectivity_from_dataset
    from nilearn import maskers
    files = [tmp_path / f'sub-{i}_bold.nii.gz' for i in range(2)]
    for f in files:
        f.write_bytes(b'fake')
    rng = np.random.default_rng(777)
    series = [rng.normal(size=(16, 4)), rng.normal(size=(23, 4))]
    class FakeMasker:
        def __init__(self, **kwargs): pass
        def fit_transform(self, path, **kwargs): return series[files.index(type(files[0])(path))]
    monkeypatch.setattr(maskers, 'NiftiLabelsMasker', FakeMasker)
    loader = FmriprepDerivativesLoader(derivatives_dir=str(tmp_path), atlas_image_path='atlas.nii.gz', cache_dir=str(tmp_path / 'cache'))
    monkeypatch.setattr(loader, '_get_files', lambda _: [(str(f), None) for f in files])
    ds = loader.load()
    assert isinstance(ds.data, list)
    assert [len(x) for x in ds.data] == [16, 23]
    assert ds.provenance['truncated_to'] is None
    connectivity_from_dataset(ds)
    assert ds.pheno.n_valid_tr.tolist() == [16, 23]
    np.testing.assert_allclose(ds.data[1], np.corrcoef(series[1].T), atol=1e-12, rtol=0)


def test_internal_builtin_loaders_options_and_diagnostics(tmp_path):
    from conninfpy.loaders import AbideSchaeferLoader, ConditionTimeseriesArrayLoader
    rng = np.random.default_rng(27)
    a, b = rng.normal(size=(22, 5)), rng.normal(size=(27, 5))
    np.save(tmp_path / 'dict.npy', {'s1': {'time_series': a, 'label': 'a'}, 's2': {'time_series': b, 'label': 'b'}})
    with pytest.warns(UserWarning, match='n_valid_tr'):
        ds = AbideSchaeferLoader(str(tmp_path / 'dict.npy'), connectivity_kind='partial_correlation', shrinkage=.3).load()
    assert ds.pheno.n_valid_tr.tolist() == [22, 27]
    assert np.allclose(ds.pheno.delta_s, .3)
    np.save(tmp_path / 'site_open.npy', np.stack([a, a + 1]))
    np.save(tmp_path / 'site_close.npy', np.stack([b, b + 1]))
    (tmp_path / 'subjects.txt').write_text('one\ntwo\n')
    ds = ConditionTimeseriesArrayLoader(str(tmp_path), str(tmp_path / 'subjects.txt'), site='site', file_template='{site}_{condition}.npy', connectivity_kind='partial_correlation', shrinkage=.4).load()
    assert ds.pheno.n_valid_tr.tolist() == [22, 22, 27, 27]
    assert np.allclose(ds.pheno.delta_s, .4)
    assert ds.provenance['connectivity']['kind'] == 'partial_correlation'


def test_nifti_groups_keep_each_observation_length(tmp_path, monkeypatch):
    from conninfpy.loaders import NiftiDirectoryLoader
    from nilearn import maskers
    rng = np.random.default_rng(81)
    a, b = rng.normal(size=(15, 4)), rng.normal(size=(22, 4))
    for label in ('a', 'b'):
        (tmp_path / label).mkdir()
        (tmp_path / label / f'{label}.nii.gz').write_bytes(b'fake')
    class FakeMasker:
        def __init__(self, **kwargs): pass
        def fit_transform(self, path, **kwargs): return a if str(path).endswith('a.nii.gz') else b
    monkeypatch.setattr(maskers, 'NiftiLabelsMasker', FakeMasker)
    ds = NiftiDirectoryLoader(subjects={'a': str(tmp_path / 'a'), 'b': str(tmp_path / 'b')}, atlas_image_path='atlas.nii.gz').load()
    assert [len(x) for x in ds.data] == [15, 22]
    assert ds.pheno.group.tolist() == ['a', 'b']
    with pytest.warns(UserWarning, match='n_valid_tr'):
        connectivity_from_dataset(ds)
    assert ds.pheno.n_valid_tr.tolist() == [15, 22]
