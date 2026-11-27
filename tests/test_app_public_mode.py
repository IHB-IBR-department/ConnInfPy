import io
from pathlib import Path
import numpy as np
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


def test_paired_bijection_with_site_col_stable_and_mismatched():
    import pandas as pd
    from conninfpy.app.components.design_inference import _build_design_arrays

    Y = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    stable_df = pd.DataFrame({
        'subject': ['sub1', 'sub2', 'sub2', 'sub1'],
        'cond': ['open', 'open', 'close', 'close'],
        'site': ['S1', 'S2', 'S2', 'S1'],
    })
    out = _build_design_arrays(
        Y, stable_df,
        test_type='paired',
        question_choice='Within-Subject Paired Contrast',
        site_col='site',
        condition_col='cond',
        subject_col='subject',
        baseline_val='open',
        target_val='close',
        require_paired_bijection=True,
    )
    assert out['sites'] == ['S1', 'S2']
    assert out['ids_group1'] == ['sub1', 'sub2']

    mismatched_df = stable_df.copy()
    mismatched_df.loc[2, 'site'] = 'S3'
    with pytest.raises(ValueError, match='sub2'):
        _build_design_arrays(
            Y, mismatched_df,
            test_type='paired',
            question_choice='Within-Subject Paired Contrast',
            site_col='site',
            condition_col='cond',
            subject_col='subject',
            baseline_val='open',
            target_val='close',
            require_paired_bijection=True,
        )


def test_wheel_mode_without_demo_files_hides_builtin_data(monkeypatch, tmp_path):
    from conninfpy.app.components import data_ingestion

    monkeypatch.setattr(data_ingestion, 'PROJECT_ROOT', tmp_path)
    for public_val in ('0', '1'):
        monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', public_val)
        app = AppTest.from_file(str(APP), default_timeout=120).run()
        assert not app.exception
        mode = next(x for x in app.radio if x.label == 'Select Ingestion Method')
        assert 'Built-in data' not in mode.options


def test_public_mode_suppresses_st_exception_on_load_error(monkeypatch):
    from conninfpy.app.components import data_ingestion
    from types import SimpleNamespace

    raw = io.BytesIO()
    matrices = np.tile(np.eye(4), (6, 1, 1))
    matrices[:, 0, 1] = matrices[:, 1, 0] = 0.2
    np.save(raw, matrices)
    fake = SimpleNamespace(name='uploaded.npy', getvalue=lambda: raw.getvalue())
    original = data_ingestion.persistent_file_uploader
    monkeypatch.setattr(
        data_ingestion,
        'persistent_file_uploader',
        lambda label, *, key, **kw: fake if key == 'data:upload_numpy' else (None if key == 'data:upload_pheno' else original(label, key=key, **kw)),
    )
    import inspect

    orig_load = data_ingestion._UploadedNumpyLoader.load

    def _flaky_load(self):
        frame = inspect.currentframe()
        caller = frame.f_back.f_code.co_name if frame and frame.f_back else ''
        if caller != 'preview':
            raise RuntimeError(f'boom at {helpers.PROJECT_ROOT}/secret.npy')
        return orig_load(self)

    monkeypatch.setattr(data_ingestion._UploadedNumpyLoader, 'load', _flaky_load)
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    next(x for x in app.radio if x.label == 'Select Ingestion Method').set_value('Upload data').run()
    app.button(key='load_dataset_btn').click().run()
    assert len(app.error) >= 1
    assert str(helpers.PROJECT_ROOT) not in app.error[0].value
    assert not app.exception


def test_bundled_atlas_sidebar_with_empty_cache(monkeypatch, tmp_path):
    """NETSTAT-84: an empty atlas cache must not offer (or crash on) BNA-246."""
    monkeypatch.setenv('CONNINFPY_ATLAS_DIR', str(tmp_path / 'empty_atlases'))
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    sb = next(x for x in app.selectbox if x.label == 'Reference Parcellation')
    assert 'BNA-246' not in sb.options
    assert sb.options == ['Schaefer-100 Yeo-7', 'Schaefer-200 Yeo-7', 'Schaefer-400 Yeo-7']
    sb.set_value('Schaefer-200 Yeo-7').run()
    assert not app.exception
    # A stale 'BNA-246' selection retained from an older session is sanitized
    # before the selectbox is declared (no crash, no invalid widget value).
    stale = AppTest.from_file(str(APP), default_timeout=120)
    stale.session_state['atlas_choice'] = 'BNA-246'
    stale.run()
    assert not stale.exception
    sb = next(x for x in stale.selectbox if x.label == 'Reference Parcellation')
    assert sb.value in sb.options


def test_bna_selection_without_cache_shows_download_hint(monkeypatch, tmp_path):
    """NETSTAT-84: even if BNA-246 is offered, a missing file is a friendly error."""
    from conninfpy.atlas import resources as atlas_resources
    monkeypatch.setenv('CONNINFPY_ATLAS_DIR', str(tmp_path / 'empty_atlases'))
    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    monkeypatch.setattr(atlas_resources, 'atlas_resource_available', lambda *a, **kw: True)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    sb = next(x for x in app.selectbox if x.label == 'Reference Parcellation')
    assert 'BNA-246' in sb.options
    sb.set_value('BNA-246').run()
    assert not app.exception
    assert any('conninfpy download-atlases' in e.value for e in app.sidebar.error)


def test_public_mode_failure_never_renders_traceback(monkeypatch):
    """NETSTAT-84: CONNINFPY_PUBLIC_MODE=1 hides task.error tracebacks/paths."""

    def script():
        # AppTest.from_function re-executes the source in a fresh namespace,
        # so everything the script needs must be defined inside it.
        from types import SimpleNamespace
        from conninfpy.app.components import design_inference
        task = SimpleNamespace(
            error='boom\nTraceback (most recent call last):\n  File "/server/secret/path.py", line 1, in run\nValueError: boom',
        )
        design_inference._render_failed_inference(task)

    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '1')
    app = AppTest.from_function(script, default_timeout=60).run()
    assert not app.exception
    assert not any(x.label == 'Error Traceback' for x in app.text_area)
    rendered = ' '.join(str(getattr(el, 'value', '')) for el in
                        [*app.error, *app.warning, *app.info, *app.text_area, *app.markdown])
    assert '/server/secret' not in rendered
    assert 'Traceback' not in rendered
    assert len(app.error) >= 1  # the failure itself is still announced

    monkeypatch.setenv('CONNINFPY_PUBLIC_MODE', '0')
    app = AppTest.from_function(script, default_timeout=60).run()
    assert not app.exception
    tb = next(x for x in app.text_area if x.label == 'Error Traceback')
    assert 'Traceback' in tb.value
