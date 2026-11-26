import pickle
import io
import json
import zipfile
import numpy as np
import pytest

from conninfpy.loaders.builtins import AbideSchaeferLoader, NumpyLoader
from conninfpy._cli_tools import init_manifest
from conninfpy.post_analysis import ResultBundle


def test_object_npy_requires_opt_in(tmp_path):
    path = tmp_path / 'object.npy'
    np.save(path, {'subject': {'time_series': np.ones((4, 2)), 'label': 'x', 'site': 'y'}})
    with pytest.raises(ValueError, match='trust_pickle'):
        AbideSchaeferLoader(str(path)).load()
    assert len(AbideSchaeferLoader(str(path), trust_pickle=True).load().subject_ids) == 1
    with pytest.raises(ValueError, match='trust_pickle'):
        NumpyLoader(str(path)).load()
    with pytest.raises(ValueError, match='trust-pickle'):
        init_manifest(data_file=path)


def test_legacy_bundle_is_not_unpickled(tmp_path):
    marker = tmp_path / 'executed'
    class Payload:
        def __reduce__(self):
            return (eval, (f"open({str(marker)!r}, 'w').write('executed')",))
    path = tmp_path / 'legacy.pkl'
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(ValueError, match='legacy pickle'):
        ResultBundle.load(path)
    assert not marker.exists()


def test_bundle_rejects_pickled_array(tmp_path):
    raw = io.BytesIO()
    np.save(raw, np.array([{"unsafe": True}], dtype=object))
    path = tmp_path / 'bundle.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({
            'schema_version': 1, 'bundle': {'type': 'array', 'key': '0'}
        }))
        archive.writestr('arrays/0.npy', raw.getvalue())
    with pytest.raises(ValueError, match='Object arrays'):
        ResultBundle.load(path)
