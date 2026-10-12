"""Atlas-original ownership, integrity and optional/offline behavior."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conninfpy.atlas import BrainnetomeReference
from conninfpy.atlas import resources


@pytest.fixture
def original(tmp_path, monkeypatch):
    payload = b"reference-original"
    filename = "BNA_SC_3D_246.zip"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "source_url": "https://atlas.brainnetome.org/download.html",
        "resources": {filename: {"sha256": hashlib.sha256(payload).hexdigest()}},
    }))
    real_bundled_path = resources.bundled_path
    monkeypatch.setattr(resources, "bundled_path", lambda name:
                        manifest if name == "brainnetome_external.json" else real_bundled_path(name))
    folder = tmp_path / "originals"
    folder.mkdir()
    path = folder / filename
    path.write_bytes(payload)
    return folder, path


def test_data_directory_precedence_and_no_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNINFPY_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("CONNINFPY_BNA_DIR", raising=False)
    assert resources.brainnetome_data_dir() == tmp_path / "data/brainnetome"
    monkeypatch.setenv("CONNINFPY_BNA_DIR", str(tmp_path / "legacy"))
    assert resources.brainnetome_data_dir() == tmp_path / "legacy"
    assert resources.brainnetome_data_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert not (tmp_path / "data").exists()


def test_home_default_and_expansion(tmp_path, monkeypatch):
    monkeypatch.delenv("CONNINFPY_DATA", raising=False)
    monkeypatch.delenv("CONNINFPY_BNA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert resources.brainnetome_data_dir() == tmp_path / "conninfpy_data/brainnetome"


def test_cwd_atlas_directory_is_not_an_implicit_source(tmp_path, monkeypatch):
    legacy = tmp_path / "datasets/atlases"
    legacy.mkdir(parents=True)
    (legacy / "BNA_SC_3D_246.zip").write_bytes(b"legacy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONNINFPY_BNA_DIR", raising=False)
    monkeypatch.setenv("CONNINFPY_DATA", str(tmp_path / "configured"))
    assert resources._find_data_dir(None) is None
    assert resources.brainnetome_resource_path("BNA_SC_3D_246.zip") is None


def test_explicit_missing_directory_does_not_fall_back(original, tmp_path, monkeypatch):
    folder, _ = original
    monkeypatch.setenv("CONNINFPY_BNA_DIR", str(folder))
    missing = tmp_path / "absent"
    ref = BrainnetomeReference.load(data_dir=missing, profiles=False)
    assert ref.data_dir == missing
    assert ref.structural_connectivity() is None
    assert resources._find_data_dir(missing) is None


def test_valid_original_is_returned(original):
    folder, path = original
    assert resources.brainnetome_resource_path(path.name, data_dir=folder) == path


def test_changed_original_is_rejected_after_cached_validation(original):
    folder, path = original
    resources.brainnetome_resource_path(path.name, data_dir=folder)
    path.write_bytes(b"damaged-original!!")
    with pytest.raises(ValueError, match="SHA256"):
        resources.brainnetome_resource_path(path.name, data_dir=folder)


def test_missing_required_original_has_acquisition_and_configuration_help(original):
    folder, path = original
    path.unlink()
    assert resources.brainnetome_resource_path(path.name, data_dir=folder) is None
    with pytest.raises(FileNotFoundError, match="CONNINFPY_DATA") as exc:
        resources.brainnetome_resource_path(path.name, data_dir=folder, required=True)
    assert "atlas.brainnetome.org/download.html" in str(exc.value)


def test_reference_validates_original_before_using_derived_cache(original, tmp_path):
    folder, path = original
    path.write_bytes(b"unrecognized-source")
    cache = tmp_path / "cache"
    cache.mkdir()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    np.save(cache / f"sc_derived_{digest[:16]}.npy", np.zeros((246, 246)))
    ref = BrainnetomeReference.load(data_dir=folder, cache_dir=cache, profiles=False)
    with pytest.raises(ValueError, match="SHA256"):
        ref.structural_connectivity()


@pytest.mark.parametrize("filename", ["../BNA_SC_3D_246.zip", "unknown.nii.gz"])
def test_only_manifest_resources_can_be_located(filename, tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        resources.brainnetome_resource_path(filename, data_dir=tmp_path)


def test_explicit_source_verification_supports_arbitrary_folder_name(original):
    _, path = original
    assert resources.verify_brainnetome_resource(path, path.name) == path


def test_offline_reference_without_optional_originals(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNINFPY_DATA", str(tmp_path / "missing"))
    monkeypatch.delenv("CONNINFPY_BNA_DIR", raising=False)
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *args, **kwargs:
                        pytest.fail("Bundled atlas loading must not use the network"))
    ref = BrainnetomeReference.load()
    assert len(ref.regions) == 246 and set(ref.profiles) == {"BD", "PC"}
    assert ref.membership([(0, 0, 0)]) is None
    assert ref.structural_connectivity() is None
    assert not (tmp_path / "missing").exists()


def test_data_directory_change_does_not_redirect_derived_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNINFPY_DATA", str(tmp_path / "originals"))
    monkeypatch.setenv("CONNINFPY_CACHE", str(tmp_path / "derived"))
    monkeypatch.delenv("CONNINFPY_BNA_DIR", raising=False)
    ref = BrainnetomeReference.load(profiles=False)
    assert ref.data_dir == tmp_path / "originals/brainnetome"
    assert ref.cache_dir == tmp_path / "derived/bna246"


@pytest.mark.parametrize("nodes", [100, 200, 400])
def test_relocated_fixture_matches_manifest_and_label_order(nodes):
    folder = Path(__file__).resolve().parents[1] / "examples/simulations/sc_prior/data"
    locations = json.loads((folder / "locations.json").read_text())["files"]
    base = f"sc_schaefer{nodes}_hcp"
    matrix = np.load(folder / f"{base}.npy")
    labels = (folder / f"{base}.labels.txt").read_text().splitlines()
    assert matrix.shape == (nodes, nodes) and len(labels) == nodes
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0)
    for suffix in [".npy", ".labels.txt", ".json"]:
        filename = base + suffix
        path = folder / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == locations[filename]["sha256"]
    # Preserve the extraction record, including its historical file locations.
    provenance = json.loads((folder / f"{base}.json").read_text())
    assert provenance["files"]["matrix"] == locations[f"{base}.npy"]["original_path"]
