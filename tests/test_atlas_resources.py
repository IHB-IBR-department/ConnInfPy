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


@pytest.fixture
def synthetic_bna_cache(tmp_path, monkeypatch):
    import nibabel as nib
    import pandas as pd

    cache_dir = tmp_path / "synthetic_atlas_cache"
    cache_dir.mkdir()
    rows = [
        {
            "roi_id": i,
            "name": f"R{i}_{'L' if i % 2 else 'R'}",
            "hemisphere": "L" if i % 2 else "R",
            "homolog_id": i + 1 if i % 2 else i - 1,
            "code": f"C_{i}",
            "description": f"Desc {i}",
            "gyrus": "G1",
            "gyrus_name": "Gyrus 1",
            "lobe": "Frontal",
            "structure": "cortex",
            "yeo7_id": 1,
            "yeo7": "Visual",
            "yeo17_id": 1,
            "yeo17": "VisCent",
            "x": float(i),
            "y": 0.0,
            "z": 0.0,
        }
        for i in range(1, 247)
    ]
    pd.DataFrame(rows).to_csv(cache_dir / "bna246_hierarchy.csv", index=False)
    arr = np.zeros((247, 2, 2), dtype=np.int16)
    for idx in range(1, 247):
        arr[idx, 0, 0] = idx
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(cache_dir / "BN_Atlas_246_1mm.nii.gz"))
    (cache_dir / "BDf_FDR05.json").write_text(
        json.dumps({str(i): {"Action": 1.0} for i in range(1, 247)}),
        encoding="utf-8",
    )
    (cache_dir / "PCf_FDR05.json").write_text(
        json.dumps({str(i): {"Go": 1.0} for i in range(1, 247)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONNINFPY_ATLAS_DIR", str(cache_dir))
    entries = {
        fname: {
            "sha256": hashlib.sha256((cache_dir / fname).read_bytes()).hexdigest(),
            "suite": "Brainnetome BNA-246",
            "source_url": "https://atlas.brainnetome.org/download.html",
            "license": "Brainnetome non-commercial research use, attribution, and share-alike",
        }
        for fname in ("bna246_hierarchy.csv", "BN_Atlas_246_1mm.nii.gz", "BDf_FDR05.json", "PCf_FDR05.json")
    }
    monkeypatch.setattr(resources, "_atlas_manifest_entries", lambda: entries)
    return cache_dir


def test_atlas_cache_dir_precedence_and_no_creation(tmp_path, monkeypatch):
    env_dir = tmp_path / "env_atlases"
    explicit_dir = tmp_path / "explicit_atlases"
    monkeypatch.setenv("CONNINFPY_ATLAS_DIR", str(env_dir))
    assert resources.atlas_cache_dir() == env_dir
    assert resources.atlas_cache_dir(explicit_dir) == explicit_dir
    monkeypatch.delenv("CONNINFPY_ATLAS_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert resources.atlas_cache_dir() == tmp_path / ".cache" / "conninfpy" / "atlases"
    assert not env_dir.exists()
    assert not explicit_dir.exists()


def test_missing_atlas_resource_raises_with_download_instructions_and_no_network(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty_atlases"
    monkeypatch.setenv("CONNINFPY_ATLAS_DIR", str(empty_dir))
    monkeypatch.setattr(
        resources.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("Missing cache lookup must not access the network"),
    )
    assert not resources.atlas_resource_available("BN_Atlas_246_1mm.nii.gz")
    with pytest.raises(FileNotFoundError) as exc:
        resources.atlas_resource_path("BN_Atlas_246_1mm.nii.gz")
    msg = str(exc.value)
    assert "conninfpy download-atlases" in msg
    assert "CONNINFPY_ATLAS_DIR" in msg
    assert "pan.cstcloud.cn" in msg


def test_corrupted_atlas_resource_fails_sha256_verification(synthetic_bna_cache):
    target = synthetic_bna_cache / "bna246_hierarchy.csv"
    target.write_bytes(b"corrupted-contents")
    assert not resources.atlas_resource_available("bna246_hierarchy.csv")
    with pytest.raises(ValueError, match="SHA256"):
        resources.atlas_resource_path("bna246_hierarchy.csv")


def test_third_party_data_unbundled_from_repo_and_package():
    repo_root = Path(__file__).resolve().parent.parent
    atlas_data_dir = resources.bundled_path("schaefer100_yeo7.csv").parent
    unbundled_atlases = (
        "BN_Atlas_246_1mm.nii.gz",
        "BN_Atlas_246_2mm.nii.gz",
        "bna246.csv",
        "bna246_hierarchy.csv",
        "BDf_FDR05.json",
        "PCf_FDR05.json",
        "atl-Anatom.tsv",
        "atl-Anatom_space-MNI_dseg.nii.gz",
        "atl-Buckner7.tsv",
        "atl-Buckner7_space-MNI_dseg.nii.gz",
        "atl-Buckner17.tsv",
        "atl-Buckner17_space-MNI_dseg.nii.gz",
        "atl-NettekovenSym32.lut",
        "atl-NettekovenSym32_space-MNI_dseg.nii.gz",
    )
    for fname in unbundled_atlases:
        assert not (atlas_data_dir / fname).exists()

    open_close_dir = repo_root / "datasets" / "demo" / "open_close"
    for fname in (
        "china_open_Schaefer200_strategy-4_GSR.npy",
        "china_close_Schaefer200_strategy-4_GSR.npy",
        "subject_order_china.txt",
    ):
        assert not (open_close_dir / fname).exists()
    assert (open_close_dir / "ihb_open_Schaefer200_strategy-4_GSR.npy").is_file()

    abide_dir = repo_root / "datasets" / "demo" / "ABIDE_mini"
    assert not (abide_dir / "abide_mini.npy").exists()
    assert (abide_dir / "abide_mini.npz").is_file()


def test_abide_mini_npz_loads_without_pickle():
    from conninfpy.loaders.builtins import AbideSchaeferLoader

    repo_root = Path(__file__).resolve().parent.parent
    abide_dir = repo_root / "datasets" / "demo" / "ABIDE_mini"
    npz_path = abide_dir / "abide_mini.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        for key in archive.files:
            assert archive[key].dtype != object

    dataset = AbideSchaeferLoader(
        data_path=str(npz_path),
        pheno_csv_path=str(abide_dir / "ABIDE1_Phenotypic_mini.csv"),
    ).load()
    assert len(dataset.subject_ids) > 0
    assert dataset.data.shape[1:] == (100, 100)


def test_acquire_scripts_verify_only(tmp_path):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    empty_atlases = tmp_path / "atlases"
    empty_atlases.mkdir()
    r1 = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "download_atlases.py"), "--verify-only", "--dest", str(empty_atlases)],
        capture_output=True,
        text=True,
    )
    assert r1.returncode != 0
    assert "Missing" in r1.stdout or "missing" in r1.stdout.lower()

    empty_oc = tmp_path / "open_close"
    empty_oc.mkdir()
    r2 = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "prepare_china_open_close.py"), "--verify-only", "--dest", str(empty_oc)],
        capture_output=True,
        text=True,
    )
    assert r2.returncode != 0
    assert "Missing" in r2.stdout or "missing" in r2.stdout.lower()


def test_explicit_missing_directory_does_not_fall_back(original, synthetic_bna_cache, tmp_path, monkeypatch):
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
    path.write_bytes(b"damaged-original-content!!")
    with pytest.raises(ValueError, match="SHA256"):
        resources.brainnetome_resource_path(path.name, data_dir=folder)


def test_missing_required_original_has_acquisition_and_configuration_help(original):
    folder, path = original
    path.unlink()
    assert resources.brainnetome_resource_path(path.name, data_dir=folder) is None
    with pytest.raises(FileNotFoundError, match="CONNINFPY_DATA") as exc:
        resources.brainnetome_resource_path(path.name, data_dir=folder, required=True)
    assert "atlas.brainnetome.org/download.html" in str(exc.value)


def test_reference_validates_original_before_using_derived_cache(original, synthetic_bna_cache, tmp_path):
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


def test_offline_reference_without_optional_originals(synthetic_bna_cache, tmp_path, monkeypatch):
    monkeypatch.setenv("CONNINFPY_DATA", str(tmp_path / "missing"))
    monkeypatch.delenv("CONNINFPY_BNA_DIR", raising=False)
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *args, **kwargs:
                        pytest.fail("Bundled atlas loading must not use the network"))
    ref = BrainnetomeReference.load()
    assert len(ref.regions) == 246 and set(ref.profiles) == {"BD", "PC"}
    assert ref.membership([(0, 0, 0)]) is None
    assert ref.structural_connectivity() is None
    assert not (tmp_path / "missing").exists()


def test_data_directory_change_does_not_redirect_derived_cache(synthetic_bna_cache, tmp_path, monkeypatch):
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
