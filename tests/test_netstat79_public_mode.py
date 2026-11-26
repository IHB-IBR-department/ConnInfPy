import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from conninfpy.app.utils import helpers
from conninfpy.app.components.data_ingestion import _open_manifest, _UploadedNumpyLoader
from conninfpy.app.utils.manifest_wizard import inspect_and_prepare_manifest

APP = Path(__file__).resolve().parents[1] / 'conninfpy/app/streamlit_app.py'

@pytest.mark.parametrize('raw,expected', [('', False), ('0', False), ('false', False), ('1', True), ('TrUe', True), (' YES ', True), ('On', True)])
def test_public_mode_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', raw)
    assert helpers.is_public_mode() is expected


def test_path_helpers_reject_public_paths(monkeypatch, tmp_path):
    from conninfpy.app.utils.atlas_wizard import load_custom_atlas_nifti
    from conninfpy.app.components.data_ingestion import _demo_path
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', 'yes')
    for call in (lambda: helpers.resolve_project_path(tmp_path), lambda: _open_manifest(tmp_path / 'm.yaml'),
                 lambda: inspect_and_prepare_manifest(tmp_path), lambda: helpers.load_custom_atlas_csv(tmp_path / 'a.csv'),
                 lambda: load_custom_atlas_nifti(tmp_path / 'atlas.nii')):
        with pytest.raises(ValueError, match='disabled in public mode'):
            call()
    with pytest.raises(ValueError, match='Only built-in'):
        _demo_path('path', value=str(tmp_path / 'secret'), key='test')


def test_upload_loader_uses_only_bytes():
    buffer = io.BytesIO()
    np.save(buffer, np.broadcast_to(np.eye(3), (4, 3, 3)))
    loader = _UploadedNumpyLoader(buffer.getvalue(), 'data.npy', 'correlation', b'subject_id\na\nb\nc\nd\n')
    assert loader.preview().n_observations == 4
    assert loader.load().subject_ids == list('abcd')
    buffer = io.BytesIO(); np.save(buffer, np.array([{}], dtype=object))
    with pytest.raises(ValueError, match='Object arrays'):
        _UploadedNumpyLoader(buffer.getvalue(), 'bad.npy', 'correlation').load()


def test_public_ui_hides_paths_and_manifest(monkeypatch):
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    mode = next(x for x in app.radio if x.label == 'Select Ingestion Method')
    assert 'Dataset Manifest (data.yaml)' not in mode.options
    assert 'Upload data' in mode.options
    mode.set_value('Built-in data').run()
    assert not app.exception
    assert not any('path' in x.label.lower() for x in app.text_input)
    next(x for x in app.selectbox if x.label == 'Select Template Structure').set_value('OpenClose').run()
    assert not app.exception
    assert next(x for x in app.selectbox if x.label == 'Cohort').options == ['IHB']
    assert not any('path' in x.label.lower() or 'subject order' in x.label.lower() for x in app.text_input)
    next(x for x in app.radio if x.label == 'Atlas Metadata').set_value('Custom atlas').run()
    assert not app.exception
    assert not any(x.label == 'Custom atlas source' for x in app.radio)
    assert not any('path' in x.label.lower() for x in app.text_input)
    next(x for x in app.radio if x.label == 'Select Ingestion Method').set_value('Upload data').run()
    assert not app.exception
    assert len(app.get('file_uploader')) >= 1


def test_public_workspace_verification_does_not_read_paths(monkeypatch, tmp_path):
    from conninfpy.app.utils.workspace_io import verify_source_references, capture_source_hashes
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    monkeypatch.setattr(Path, 'exists', lambda self: pytest.fail('checked server path'))
    assert verify_source_references({'x': {'path': str(tmp_path / 'secret')}})['overall'] == 'unknown'
    assert capture_source_hashes(extra_paths={'x': str(tmp_path / 'secret')}) == {}


def test_public_abide_demo_loads(monkeypatch):
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    next(x for x in app.radio if x.label == 'Select Ingestion Method').set_value('Built-in data').run()
    app.button(key='load_dataset_btn').click().run()
    assert not app.exception
    assert app.session_state.connectivity_data is not None
    assert str(helpers.PROJECT_ROOT) not in str(app.session_state.dataset_info)
    app.button(key='nav_step_2').click().run()
    next(x for x in app.selectbox if x.label == 'Select method profile:').set_value('Conservative network-aware (NI-TFNBS)').run()
    assert not app.exception
    assert not any('SC Prior Matrix' in x.label for x in app.text_input)


def test_public_upload_loads(monkeypatch):
    from conninfpy.app.components import data_ingestion
    from types import SimpleNamespace
    raw = io.BytesIO()
    matrices = np.tile(np.eye(4), (6, 1, 1))
    matrices[:, 0, 1] = matrices[:, 1, 0] = 0.2
    np.save(raw, matrices)
    fake = SimpleNamespace(name='uploaded.npy', getvalue=lambda: raw.getvalue())
    original = data_ingestion.persistent_file_uploader
    monkeypatch.setattr(data_ingestion, 'persistent_file_uploader',
                        lambda label, *, key, **kw: fake if key == 'data:upload_numpy' else (None if key == 'data:upload_pheno' else original(label, key=key, **kw)))
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    next(x for x in app.radio if x.label == 'Select Ingestion Method').set_value('Upload data').run()
    assert not app.exception
    app.button(key='load_dataset_btn').click().run()
    assert not app.exception
    assert app.session_state.connectivity_data.shape == (6, 4, 4)


def test_public_openclose_ihb_loads(monkeypatch):
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    next(x for x in app.radio if x.label == 'Select Ingestion Method').set_value('Built-in data').run()
    next(x for x in app.selectbox if x.label == 'Select Template Structure').set_value('OpenClose').run()
    app.button(key='load_dataset_btn').click().run()
    assert not app.exception
    assert app.session_state.connectivity_data is not None
    assert str(helpers.PROJECT_ROOT) not in str(app.session_state.dataset_info)
